import asyncio
from datetime import datetime, timezone
from google import genai
from google.genai import types
from google.genai import errors as gx
from .settings import GEMINI_API_KEY, SALIENCE_CUTOFF, TOPICS
from .schemas import Plan, SalienceVerdict, Answer
from .state import ResearchState
from .graphiti_setup import get_graphiti
from .web import search_and_rank, fetch_markdown_polite

_g_client = genai.Client(api_key=GEMINI_API_KEY)

# google-genai 1.x exposes ClientError (4xx) and ServerError (5xx) under
# google.genai.errors instead of google.api_core.exceptions. Transient codes
# we want to retry on: 429 (ResourceExhausted), 500 (Internal),
# 503 (Unavailable), 504 (DeadlineExceeded). We catch the broad classes and
# filter by .code so non-transient 4xx/5xx propagate immediately.
_RETRY_CODES = {429, 500, 503, 504}
_RETRY_EXC = (gx.ClientError, gx.ServerError, asyncio.TimeoutError)

async def _gen(model: str, contents, **cfg) -> object:
    """generate_content with backoff on transient errors."""
    for i in range(4):
        try:
            return await _g_client.aio.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**cfg),
            )
        except (gx.ClientError, gx.ServerError) as e:
            if getattr(e, "code", None) not in _RETRY_CODES or i == 3:
                raise
            await asyncio.sleep(2 ** i + 1)
        except asyncio.TimeoutError:
            if i == 3: raise
            await asyncio.sleep(2 ** i + 1)
    raise RuntimeError("unreachable: _gen exhausted retries without raising")

def _parsed_or_raise(resp, schema_name: str):
    """response.parsed is None on parse failure even with response_schema."""
    if resp.parsed is None:
        raise ValueError(f"Gemini failed to produce valid {schema_name}; raw: {resp.text[:500]}")
    return resp.parsed

# ---------- nodes ----------

async def plan_node(state: ResearchState) -> dict:
    resp = await _gen(
        "gemini-2.5-pro",
        f"Decompose into 3-5 search-engine queries:\n{state['question']}",
        response_mime_type="application/json",
        response_schema=Plan,
        thinking_config=types.ThinkingConfig(thinking_budget=512),
    )
    return {
        "plan": _parsed_or_raise(resp, "Plan").sub_questions,
        "iteration": state.get("iteration", 0) + 1,
    }

async def search_node(state: ResearchState) -> dict:
    return {"candidate_urls": await search_and_rank(state["plan"], k=8)}

async def ingest_one(state: dict) -> dict:
    """Per-URL: fetch -> salience -> (maybe) write to Graphiti. Reducer merges fan-in.

    Any per-URL failure (Firecrawl 403, Gemini 503/exhausted retries, Graphiti
    extraction error) must NOT abort the whole fan-out: swallow + record a skip
    so the rest of the URLs still flow through.
    """
    url = state["url"]
    try:
        md = await fetch_markdown_polite(url)
        if not md:
            return {"documents": [{"url": url, "skipped": True, "error": "empty markdown"}]}
        topics_str = ", ".join(TOPICS) if TOPICS else "(no topics configured)"
        verdict_resp = await _gen(
            "gemini-2.5-pro",
            f"Research focus: {state['question']}\nLong-term topics: {topics_str}\n\nDocument:\n{md[:20000]}",
            response_mime_type="application/json",
            response_schema=SalienceVerdict,
            thinking_config=types.ThinkingConfig(thinking_budget=256),
            system_instruction=(
                "Score relevance 0-1 to the research focus AND the long-term topics. "
                "Extract novel atomic claims as short factual sentences."
            ),
        )
        verdict = _parsed_or_raise(verdict_resp, "SalienceVerdict")
        if verdict.score < SALIENCE_CUTOFF:
            return {"documents": [{"url": url, "skipped": True,
                                    "score": verdict.score, "reason": verdict.reason}]}
        g = await get_graphiti()
        ep_result = await g.add_episode(
            name=url,
            episode_body=md,
            source_description=f"web:{url}",
            reference_time=datetime.now(timezone.utc),
        )
        ep_uuid = ep_result.episode.uuid
        return {
            "documents": [{"url": url, "episode_uuid": ep_uuid, "score": verdict.score}],
            "salient_episode_ids": [ep_uuid],
        }
    except Exception as e:
        return {"documents": [{"url": url, "skipped": True,
                                "error": f"{type(e).__name__}: {str(e)[:200]}"}]}

async def retrieve_node(state: ResearchState) -> dict:
    g = await get_graphiti()
    results = await g.search(query=state["question"], num_results=20)
    chunks = [f"[{r.uuid}] {r.fact}" for r in results]
    return {"retrieval_chunks": chunks, "context": "\n\n".join(chunks)}

async def answer_node(state: ResearchState) -> dict:
    resp = await _gen(
        "gemini-2.5-pro",
        f"Question: {state['question']}\n\nContext (cite by [uuid]):\n{state['context']}",
        response_mime_type="application/json",
        response_schema=Answer,
        thinking_config=types.ThinkingConfig(thinking_budget=4096),
    )
    a = _parsed_or_raise(resp, "Answer")
    return {"answer": a.answer, "citations": a.citations}
