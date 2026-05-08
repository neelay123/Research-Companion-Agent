# Research Companion Agent — Design

**Date:** 2026-05-04
**Status:** Draft, awaiting user review
**Predecessor:** "Building a Research Companion Agent with Episodic + Semantic Memory — Final Draft (Revision 2)" (pasted into chat)

This doc is the post-pressure-test consolidation of Revision 2. It encodes locked decisions, fixes code bugs in the original §8, and tags every external claim that has not been verified against live docs/SDKs as `UNVERIFIED:` so the implementer can sanity-check before `pip install`.

---

## 1. Locked decisions

| ID | Decision | Value |
|----|----------|-------|
| D | Frontend | Streamlit only, single page |
| I | Secrets | `.env` + `python-dotenv` |
| E | Reflection trigger | Manual: `python -m research_agent.reflect` |
| B | Reflection structure | Separate top-level `StateGraph`, shared SQLite checkpointer file, distinct `thread_id` namespace (`reflect-<date>`) |
| C | Eval fixture | Build-once: real Firecrawl + real Gemini + real Graphiti write into scratch Kuzu, clean shutdown, filesystem-copy `.kuzu` to `tests/fixtures/graph.kuzu` (committed). Tests open read-only |
| — | Mocks | **None.** Eval suite uses live Gemini judge on every run. Eval fixture rebuild uses live Firecrawl + live Gemini |
| — | Salience threshold | **Two-tier**: `0.5` cutoff. Below → drop. At/above → full extraction via `add_episode`. Env-overridable (`SALIENCE_CUTOFF`). Three-tier dropped because Graphiti has no documented "raw store, skip extraction" param; adding a side-table for raw text is v1 work |
| — | Topics / research interests | `config.toml`, list of strings under `[research].topics`. Loaded once at process start |
| — | Tavily relevance threshold | `0.5` default, `TAVILY_MIN_SCORE` env override |
| — | Embedding dimension | **1536, locked.** Re-embedding the corpus is prohibitive. MTEB delta vs 3072 ≈ 0 per `gemini-embedding-001` card (UNVERIFIED) |
| — | Concurrency cap | LangGraph `max_concurrency=4` is the single source of truth. Graphiti `SEMAPHORE_LIMIT` set high (`32`) so LangGraph dominates |
| — | Deferred to v1 | LangSmith tracing, cost-cap kill switch, FalkorDB migration, Postgres checkpointer |

## 2. Stack

- **Memory engine:** `graphiti-core[google-genai,kuzu]` (UNVERIFIED: `graphiti-core` 0.28.x exists with both extras)
- **Graph DB:** Kuzu embedded, single file. v1 graduates to FalkorDB (one-line driver swap)
- **Orchestration:** LangGraph 0.6+ `StateGraph` with `AsyncSqliteSaver` checkpointer (UNVERIFIED: package `langgraph-checkpoint-sqlite`, import `langgraph.checkpoint.sqlite.aio.AsyncSqliteSaver`)
- **LLM:** Gemini 2.5 via `google-genai`. Originally specced as Pro for all reasoning calls; switched to Flash post-cap-blow (commit 15182f6) — Flash has higher RPM headroom and is less prone to 503 "high demand" than Pro. All reasoning sites (plan, salience, answer, reflect synth, eval judge) use `gemini-2.5-flash` (UNVERIFIED: pricing for 2.5-flash)
- **Embedder:** `gemini-embedding-001` @ 1536 dims via Matryoshka (UNVERIFIED: SDK param name — likely `output_dimensionality` on Google SDK; Graphiti's `GeminiEmbedderConfig` field name unverified, may be `embedding_dim`)
- **Reranker:** `GeminiRerankerClient` pinned to `gemini-2.5-flash-lite` (verified working in T5 smoke test)
- **Web research:** Firecrawl primary, Tavily fallback. No mocks.
- **Eval:** DeepEval with `GeminiModel` judge (UNVERIFIED: native non-LiteLLM `GeminiModel` in `deepeval.models`)
- **Frontend:** Streamlit, one page

## 3. Repository layout

```
research-companion/
├── pyproject.toml
├── .env.example
├── config.toml                       # topics, thresholds
├── docs/superpowers/specs/           # this file
├── src/research_agent/
│   ├── __init__.py
│   ├── settings.py                   # dotenv + config.toml loader
│   ├── graphiti_setup.py             # Graphiti client (singleton)
│   ├── schemas.py                    # Pydantic: Plan, SalienceVerdict, Answer, entity types
│   ├── state.py                      # ResearchState, ReflectState
│   ├── web.py                        # firecrawl + tavily adapter
│   ├── nodes.py                      # all LangGraph node functions
│   ├── graph.py                      # builder.compile() factories for research + reflect
│   ├── reflect.py                    # CLI entrypoint: python -m research_agent.reflect
│   └── ui.py                         # Streamlit page: streamlit run -m research_agent.ui
├── tests/
│   ├── fixtures/
│   │   └── graph.kuzu                # committed snapshot
│   ├── build_fixture.py              # real services → scratch Kuzu → copy
│   ├── test_eval.py                  # DeepEval pytest cases
│   └── eval_dataset.py               # 6–10 LLMTestCase instances
└── checkpoints.sqlite                # gitignored, runtime
```

## 4. Architecture (single page)

```
┌─────────────────────────────────────────────────────────────┐
│ Streamlit UI (ui.py)                                        │
│   asks question → app.astream(stream_mode="updates")        │
│   renders each event as a progress line                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                  ┌────────────▼────────────┐
                  │  Research StateGraph     │
                  │  thread_id = "session-N" │
                  │                          │
                  │  plan → search → fan-out │
                  │  → ingest_one (×N) → ?   │
                  │  → retrieve → answer     │
                  └────────────┬─────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        Firecrawl       Gemini 2.5 Pro    Graphiti
        Tavily          (LLM + judge)     (Kuzu file)
                                                │
                                                │ separate process,
                                                │ shared checkpoint
                                                │ file, different
                                                │ thread_id
                                                ▼
                                      ┌──────────────────┐
                                      │ Reflect StateGraph│
                                      │ thread_id =       │
                                      │ "reflect-<date>"  │
                                      └──────────────────┘
```

Two top-level `StateGraph`s. Both share `checkpoints.sqlite` via the same `AsyncSqliteSaver` connection string. Different `thread_id` namespaces keep their checkpoint histories independent.

## 5. State schemas

```python
# src/research_agent/state.py
from typing import TypedDict, Annotated
from operator import add

class ResearchState(TypedDict):
    question: str
    plan: list[str]
    candidate_urls: list[str]
    documents: Annotated[list[dict], add]            # reducer for fan-in
    salient_episode_ids: Annotated[list[str], add]
    retrieval_chunks: list[str]                       # one per result, for DeepEval RETRIEVAL_CONTEXT
    context: str                                      # joined version, for the answer prompt
    answer: str
    citations: list[str]
    iteration: int

class ReflectState(TypedDict):
    since_iso: str
    episode_uuids: list[str]                          # gathered, then re-fetched in synthesize
    patterns: list[str]                               # synthesized text, written back as new episodes
    new_synthesis_uuids: Annotated[list[str], add]
```

State holds **UUIDs and primitives only**. Never live Graphiti `EpisodicNode` / `EntityNode` objects — they don't survive `JsonPlusSerializer` and would balloon the SQLite checkpoints.

## 6. Pydantic schemas

```python
# src/research_agent/schemas.py
from pydantic import BaseModel, Field

# --- structured-output for Gemini ---
class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=5)

class SalienceVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    novel_claims: list[str]

class Answer(BaseModel):
    answer: str
    citations: list[str]

# --- Graphiti entity types (Phase 2) ---
class Paper(BaseModel):
    title: str
    arxiv_id: str | None = None
    year: int | None = None

class Author(BaseModel):
    name: str
    affiliation: str | None = None

class Claim(BaseModel):
    statement: str
    # Note: dropped `confidence` field. Gemini structured output is unreliable on
    # Optional/union-with-None primitives — either schema rejection or always-null. Graphiti's
    # extractor surfaces confidence-equivalent signals via edge metadata anyway.
```

Keep schemas flat. UNVERIFIED: Gemini `response_schema` rejects deeply nested or heavily-constrained schemas with `InvalidArgument: 400`.

## 7. Settings + Graphiti singleton

```python
# src/research_agent/settings.py
import os, tomllib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # .env in repo root

GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]
FIRECRAWL_API_KEY = os.environ["FIRECRAWL_API_KEY"]
TAVILY_API_KEY    = os.environ.get("TAVILY_API_KEY")  # optional
SALIENCE_CUTOFF   = float(os.environ.get("SALIENCE_CUTOFF", "0.5"))
TAVILY_MIN_SCORE  = float(os.environ.get("TAVILY_MIN_SCORE", "0.5"))

with open(Path(__file__).parent.parent.parent / "config.toml", "rb") as f:
    _cfg = tomllib.load(f)
TOPICS: list[str] = _cfg.get("research", {}).get("topics", [])
```

```python
# src/research_agent/graphiti_setup.py
"""Lazy Graphiti singleton wired to Gemini for LLM, embedding, and reranking.

The constructor's `cross_encoder` is mandatory: omitting it falls back to
OpenAIRerankerClient, which silently demands OPENAI_API_KEY at first use.

`get_graphiti()` is async because graphiti-core 0.29.0's
Graphiti.build_indices_and_constraints() is a no-op for the Kuzu driver — the
real implementation lives at driver.graph_ops, and must be invoked once before
the first hybrid search. Doing it here means no consumer needs to know the
workaround exists.
"""
import asyncio
import os
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.driver.kuzu_driver import KuzuDriver
from .settings import GEMINI_API_KEY

# graphiti-core's SEMAPHORE_LIMIT default is 20 (graphiti_core/helpers.py).
# Set high so LangGraph's max_concurrency is the binding constraint.
os.environ.setdefault("SEMAPHORE_LIMIT", "32")

_instance: Graphiti | None = None
_indices_built: bool = False
_init_lock: asyncio.Lock | None = None

def _build(db_path: str) -> Graphiti:
    return Graphiti(
        graph_driver=KuzuDriver(db=db_path),
        # Flash for extraction: higher RPM headroom, less prone to 503 "high demand"
        # than Pro. Extraction quality acceptable for hobby project; agent reasoning
        # in nodes.py still uses Pro.
        llm_client=GeminiClient(config=LLMConfig(
            api_key=GEMINI_API_KEY, model="gemini-2.5-flash",
        )),
        embedder=GeminiEmbedder(config=GeminiEmbedderConfig(
            api_key=GEMINI_API_KEY,
            embedding_model="gemini-embedding-001",
            embedding_dim=1536,
        )),
        cross_encoder=GeminiRerankerClient(config=LLMConfig(
            api_key=GEMINI_API_KEY,
            model="gemini-2.5-flash-lite",
        )),
    )

async def get_graphiti() -> Graphiti:
    """Lazy singleton. Reads RESEARCH_KUZU_PATH on first call so tests can override.

    Async because the first call must build the FTS indices via the Kuzu
    driver's graph_ops shim (Graphiti.build_indices_and_constraints itself is a
    no-op for KuzuDriver in 0.29.0).

    Guarded by an asyncio.Lock because LangGraph fan-out may dispatch many
    concurrent first-callers; without the lock multiple coroutines race past
    the `_indices_built` check and CREATE_FTS_INDEX raises "already exists".
    """
    global _instance, _indices_built, _init_lock
    if _init_lock is None:
        _init_lock = asyncio.Lock()
    async with _init_lock:
        if _instance is None:
            path = os.environ.get("RESEARCH_KUZU_PATH", "./graph.kuzu")
            _instance = _build(path)
        if not _indices_built:
            try:
                await _instance.driver.graph_ops.build_indices_and_constraints(_instance.driver)
            except RuntimeError as e:
                # Pre-built fixture: indices already exist on disk, this process just
                # didn't know yet. Treat as success.
                if "already exists" not in str(e):
                    raise
            _indices_built = True
        return _instance

def reset_graphiti() -> None:
    """Tests only: drop singleton + index-built flag so next get_graphiti() re-reads env."""
    global _instance, _indices_built, _init_lock
    _instance = None
    _indices_built = False
    _init_lock = None
```

## 8. Web adapter

```python
# src/research_agent/web.py
import asyncio, logging
from collections import defaultdict
from urllib.parse import urlparse
# firecrawl-py 4.x: package imports as `firecrawl`; primary class is `Firecrawl`
# (the legacy `FirecrawlApp` alias exists but exposes no `scrape_url`/`scrape`
# methods on this version — we use `Firecrawl` directly).
from firecrawl import Firecrawl
from tavily import TavilyClient
from .settings import FIRECRAWL_API_KEY, TAVILY_API_KEY, TAVILY_MIN_SCORE

log = logging.getLogger(__name__)
_firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)
_tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


async def search_and_rank(queries: list[str], k: int = 8) -> list[str]:
    """Tavily fans out cheap, returns URLs above threshold.

    Tavily v0.7+ `search(query, max_results=k)` returns a dict shaped
    `{"results": [{"url": str, "score": float, "title": ..., "content": ...}, ...]}`.
    """
    if not _tavily:
        raise RuntimeError("TAVILY_API_KEY required for search fan-out")
    urls: dict[str, float] = {}
    for q in queries:
        resp = await asyncio.to_thread(_tavily.search, q, max_results=k)
        for r in resp.get("results", []):
            if r.get("score", 0) >= TAVILY_MIN_SCORE:
                urls[r["url"]] = max(urls.get(r["url"], 0), r["score"])
    return [u for u, _ in sorted(urls.items(), key=lambda kv: -kv[1])[:k]]


async def fetch_markdown(url: str, attempts: int = 3) -> str:
    """Firecrawl with exponential backoff. Raises after final attempt — node lets LangGraph checkpoint+resume.

    Firecrawl v4.x: method is `scrape(url, *, formats=[...])` (not `scrape_url(url, params=...)`),
    and the return value is a Pydantic `firecrawl.v2.types.Document` whose markdown is on the
    `.markdown` attribute (not `.get("markdown")`).
    """
    for i in range(attempts):
        try:
            doc = await asyncio.to_thread(
                _firecrawl.scrape, url, formats=["markdown"],
            )
            return getattr(doc, "markdown", "") or ""
        except Exception as e:
            log.warning("firecrawl %s attempt %d: %s", url, i + 1, e)
            if i == attempts - 1:
                raise
            await asyncio.sleep(2 ** i)
    return ""  # unreachable


# Per-domain stagger so we don't hammer one host.
# Module-global is intentional: Streamlit single-process app shares one event loop.
_domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def fetch_markdown_polite(url: str) -> str:
    host = urlparse(url).netloc.lower()  # case-fold so Arxiv.org and arxiv.org share one lock
    async with _domain_locks[host]:
        return await fetch_markdown(url)
```

## 9. Nodes

```python
# src/research_agent/nodes.py
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
        "gemini-2.5-flash",
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
            "gemini-2.5-flash",
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
        "gemini-2.5-flash",
        f"Question: {state['question']}\n\nContext (cite by [uuid]):\n{state['context']}",
        response_mime_type="application/json",
        response_schema=Answer,
        thinking_config=types.ThinkingConfig(thinking_budget=4096),
    )
    a = _parsed_or_raise(resp, "Answer")
    return {"answer": a.answer, "citations": a.citations}
```

## 10. Graph factories

```python
# src/research_agent/graph.py
from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .state import ResearchState, ReflectState
from .nodes import plan_node, search_node, ingest_one, retrieve_node, answer_node

CHECKPOINT_DSN = "checkpoints.sqlite"

def fan_out_ingest(state: ResearchState) -> list[Send]:
    return [Send("ingest_one", {"url": u, "question": state["question"]})
            for u in state["candidate_urls"]]

def should_search_more(state: ResearchState) -> str:
    # Hard cap at 2 plan-cycles to prevent runaway recursion on adversarial topics.
    # plan_node increments `iteration`; once we're past 2, give up and answer with what we have.
    if state.get("iteration", 0) >= 2: return "retrieve"
    if len(state.get("salient_episode_ids", [])) < 3: return "plan"
    return "retrieve"

def _build_research() -> StateGraph:
    b = StateGraph(ResearchState)
    b.add_node("plan", plan_node)
    b.add_node("search", search_node)
    b.add_node("ingest_one", ingest_one)
    b.add_node("retrieve", retrieve_node)
    b.add_node("answer", answer_node)
    b.add_edge(START, "plan")
    b.add_edge("plan", "search")
    b.add_conditional_edges("search", fan_out_ingest, ["ingest_one"])
    b.add_conditional_edges("ingest_one", should_search_more,
                            {"plan": "plan", "retrieve": "retrieve"})
    b.add_edge("retrieve", "answer")
    b.add_edge("answer", END)
    return b

@asynccontextmanager
async def research_app():
    """Lifecycle-managed app. Use:  async with research_app() as app: ..."""
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DSN) as ckpt:
        yield _build_research().compile(checkpointer=ckpt)
```

Lifecycle fix from Revision 2: `__aenter__` is paired with `__aexit__` via `async with`, no leaks. Streamlit/CLI/reflect all enter the same context manager.

## 11. Reflection (separate top-level)

```python
# src/research_agent/reflect.py
"""Manual reflection pass. Run:  python -m research_agent.reflect [--since 2026-04-01]"""
import argparse, asyncio
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from google.genai import types
from pydantic import BaseModel
from graphiti_core.nodes import EpisodicNode
from .state import ReflectState
from .graphiti_setup import get_graphiti
from .nodes import _gen, _parsed_or_raise
from .graph import CHECKPOINT_DSN

class Synthesis(BaseModel):
    patterns: list[str]

async def gather_recent(s: ReflectState) -> dict:
    g = await get_graphiti()
    # graphiti-core 0.29.0 has no `get_episodes(after=...)`. Use retrieve_episodes
    # with reference_time=now to grab the most-recent window, then filter by
    # valid_at >= since_iso Python-side. Kuzu returns naive datetimes for valid_at,
    # so compare naive-to-naive.
    since_dt = datetime.fromisoformat(s["since_iso"])
    if since_dt.tzinfo is not None:
        since_dt = since_dt.astimezone(timezone.utc).replace(tzinfo=None)
    eps = await g.retrieve_episodes(
        reference_time=datetime.now(timezone.utc),
        last_n=1000,
    )
    def _as_naive(dt):
        return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt
    return {"episode_uuids": [e.uuid for e in eps if _as_naive(e.valid_at) >= since_dt]}

async def synthesize(s: ReflectState) -> dict:
    g = await get_graphiti()
    if not s["episode_uuids"]:
        return {"patterns": []}
    # graphiti-core 0.29.0 has no `get_episode_by_uuid`. Use the EpisodicNode
    # classmethod `get_by_uuids(driver, uuids)` for batched fetch.
    eps = await EpisodicNode.get_by_uuids(g.driver, s["episode_uuids"])
    body = "\n---\n".join(getattr(e, "content", "") for e in eps)
    if not body.strip():
        return {"patterns": []}
    resp = await _gen(
        "gemini-2.5-flash",
        f"Find non-obvious patterns connecting these episodes:\n{body[:80000]}",
        response_mime_type="application/json",
        response_schema=Synthesis,
        thinking_config=types.ThinkingConfig(thinking_budget=4096),
    )
    return {"patterns": _parsed_or_raise(resp, "Synthesis").patterns}

async def write_back(s: ReflectState) -> dict:
    g = await get_graphiti()
    uuids = []
    for p in s["patterns"]:
        try:
            ep_result = await g.add_episode(
                name=f"reflection:{date.today().isoformat()}",
                episode_body=p,
                source_description="reflection",
                reference_time=datetime.now(timezone.utc),
            )
            uuids.append(ep_result.episode.uuid)
        except Exception as e:
            print(f"  skip pattern (add_episode failed): {type(e).__name__}: {str(e)[:120]}")
    return {"new_synthesis_uuids": uuids}

def _build_reflect() -> StateGraph:
    b = StateGraph(ReflectState)
    b.add_node("gather", gather_recent)
    b.add_node("think", synthesize)
    b.add_node("write", write_back)
    b.add_edge(START, "gather")
    b.add_edge("gather", "think")
    b.add_edge("think", "write")
    b.add_edge("write", END)
    return b

@asynccontextmanager
async def reflect_app():
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DSN) as ckpt:
        yield _build_reflect().compile(checkpointer=ckpt)

async def _main(since: str) -> None:
    async with reflect_app() as app:
        cfg = {"configurable": {"thread_id": f"reflect-{date.today().isoformat()}"}}
        async for ev in app.astream({"since_iso": since}, cfg, stream_mode="updates"):
            print(ev)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="1970-01-01")
    args = p.parse_args()
    asyncio.run(_main(args.since))
```

State holds UUIDs only (`episode_uuids`) and synthesized text (`patterns`); episode bodies are re-fetched per-UUID inside `synthesize` to keep checkpoint size bounded. Same rule as §5: never put live Graphiti node objects or large fetched payloads into LangGraph state.

## 12. Streamlit UI

```python
# src/research_agent/ui.py
"""streamlit run -m research_agent.ui"""
import asyncio
import streamlit as st
from .graph import research_app

st.set_page_config(page_title="Research Companion", layout="wide")
st.title("Research Companion")

q = st.text_input("Question:")
go = st.button("Ask")

async def run(question: str):
    progress = st.empty()
    answer_box = st.empty()
    citations_box = st.empty()
    events: list[str] = []
    async with research_app() as app:
        cfg = {
            "configurable": {"thread_id": f"session-{hash(question)}",
                             "max_concurrency": 4},
            "recursion_limit": 50,   # plan loop bumps iteration, hard-capped at 2; 50 leaves headroom for fan-out
        }
        async for ev in app.astream({"question": question}, cfg, stream_mode="updates"):
            events.append(str(ev)[:300])
            progress.code("\n".join(events[-20:]))
        # Pull final values from the checkpoint
        state = await app.aget_state(cfg)
        answer_box.markdown(state.values.get("answer", "(no answer)"))
        citations_box.write(state.values.get("citations", []))

if go and q:
    asyncio.run(run(q))
```

UNVERIFIED: `app.aget_state(cfg)` is the LangGraph 0.6+ way to read the final checkpointed state.

## 13. Eval harness — no mocks

```python
# tests/build_fixture.py
"""Run once. Builds tests/fixtures/graph.kuzu from real Firecrawl + Gemini.
   IMPORTANT: do not run during normal pytest — costs real money."""
# CRITICAL ordering: set RESEARCH_KUZU_PATH BEFORE any research_agent import
# so the lazy graphiti singleton picks it up on first call.
import os, asyncio, shutil, sys
from pathlib import Path

PINNED_QUESTIONS = [
    "What are state-space models like Mamba?",
    "Who are the authors of the original Mamba paper?",
    # ... 4-6 more pinned questions whose source URLs are stable (arxiv abs, archive.org)
]
SCRATCH = Path("tests/fixtures/_scratch.kuzu")
TARGET  = Path("tests/fixtures/graph.kuzu")

# Set env BEFORE imports so the singleton, when first built inside research_agent.nodes
# import chain, reads the scratch path — not the production default.
os.environ["RESEARCH_KUZU_PATH"] = str(SCRATCH)

from research_agent.graph import research_app          # noqa: E402
from research_agent.graphiti_setup import reset_graphiti  # noqa: E402

async def main():
    if SCRATCH.exists(): shutil.rmtree(SCRATCH, ignore_errors=True)
    reset_graphiti()  # in case anything imported earlier already cached the default
    async with research_app() as app:
        for q in PINNED_QUESTIONS:
            cfg = {"configurable": {"thread_id": f"fixture-{hash(q)}"},
                   "recursion_limit": 50}
            async for _ in app.astream({"question": q}, cfg, stream_mode="updates"):
                pass
    # Clean shutdown happens in the async with __aexit__
    # Now copy. Kuzu single-process write lock is released; read-only attach is safe.
    if TARGET.exists(): shutil.rmtree(TARGET, ignore_errors=True)
    shutil.copytree(SCRATCH, TARGET) if SCRATCH.is_dir() else shutil.copy(SCRATCH, TARGET)
    print(f"fixture written: {TARGET}")

if __name__ == "__main__":
    asyncio.run(main())
```

UNVERIFIED: Kuzu DB is a directory or a single file depending on version — handle both via `is_dir()`. Confirm against installed Kuzu.

```python
# tests/eval_dataset.py
from deepeval.test_case import LLMTestCase

DATASET: list[LLMTestCase] = [
    LLMTestCase(
        input="When was the Mamba paper first posted to arXiv?",
        actual_output="",        # filled by test runner
        retrieval_context=[],    # filled by test runner
        expected_output="December 2023",
    ),
    # ... 5–9 more covering: multi-hop, temporal, contradiction, citation-correctness
]
```

```python
# tests/test_eval.py
import os, pytest, asyncio
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCaseParams
from research_agent.graph import research_app
from .eval_dataset import DATASET

# Live Gemini judge — no mocks. Runs cost real Gemini tokens per test.
JUDGE = GeminiModel(model="gemini-2.5-flash", api_key=os.environ["GEMINI_API_KEY"])

CITATION_CORRECTNESS = GEval(
    name="CitationCorrectness", model=JUDGE, threshold=0.8,
    criteria=("Every factual claim in `actual_output` must be supported by at least one "
              "episode UUID present in `retrieval_context`. Fail if any claim lacks a citation."),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.RETRIEVAL_CONTEXT],
)

TEMPORAL_CORRECTNESS = GEval(
    name="TemporalCorrectness", model=JUDGE, threshold=0.7,
    criteria=("If the question asks about a point in time, the answer must reflect what was "
              "true at that time per the retrieval context, not the latest known fact."),
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                       LLMTestCaseParams.RETRIEVAL_CONTEXT],
)

@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory):
    """Open the committed Kuzu fixture read-only."""
    import shutil
    from research_agent.graphiti_setup import reset_graphiti
    src = "tests/fixtures/graph.kuzu"
    dst = tmp_path_factory.mktemp("kuzu") / "graph.kuzu"
    shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy(src, dst)
    os.environ["RESEARCH_KUZU_PATH"] = str(dst)
    # If any prior import bound the singleton against the default path, drop it.
    reset_graphiti()
    yield dst

@pytest.mark.parametrize("tc", DATASET, ids=[t.input[:40] for t in DATASET])
def test_eval(fixture_db, tc):
    async def run():
        async with research_app() as app:
            cfg = {"configurable": {"thread_id": f"test-{hash(tc.input)}"},
                   "recursion_limit": 50}
            async for _ in app.astream({"question": tc.input}, cfg, stream_mode="updates"):
                pass
            state = await app.aget_state(cfg)
            tc.actual_output = state.values["answer"]
            tc.retrieval_context = state.values["retrieval_chunks"]
    asyncio.run(run())
    assert_test(tc, [CITATION_CORRECTNESS, TEMPORAL_CORRECTNESS])
```

`graphiti_setup.get_graphiti()` is `async` and reads `RESEARCH_KUZU_PATH` lazily on first call, so the fixture path set in the pytest fixture is honored. The first call also builds Kuzu's FTS indices once (graphiti-core 0.29.0's `Graphiti.build_indices_and_constraints()` is a no-op for KuzuDriver — see §14 Group F). Tests copy the committed fixture into a tempdir per session so the original is never mutated.

## 14. UNVERIFIED claims (reading list before `pip install`)

Group A — versions / packages:
- `graphiti-core` 0.28.x exists with `[google-genai,kuzu]` extras
- `langgraph-checkpoint-sqlite` package, `AsyncSqliteSaver` import path
- DeepEval `GeminiModel` native (no LiteLLM)

Group B — Graphiti API surface:
- `KuzuDriver(db=...)` constructor signature
- `GeminiClient`, `GeminiEmbedder`, `GeminiRerankerClient` import paths and config field names (`embedding_dim` vs `output_dimensionality`)
- Default `cross_encoder` falls back to `OpenAIRerankerClient` when omitted (CONFIRMED in T5)
- `add_episode` signature — CONFIRMED in T5: `reference_time` is mandatory (None rejected), pass `datetime.now(timezone.utc)`
- `search` uses configured reranker by default, or needs explicit flag — CONFIRMED in T5: search() invokes reranker without an explicit flag
- `get_episodes(after=...)` exists for the reflection pass
- Param to skip triple extraction for raw-store-only episodes

Group C — Gemini SDK:
- `google-genai` (unified) is current; `client.aio.models.generate_content` async
- Pricing $1.25/$10 per M tokens at ≤200K context
- Pro rejects `thinking_budget=0`; Flash/Flash-Lite accept
- `gemini-embedding-001` Matryoshka dims, max input 2048 tokens
- Current GA flash-lite name — CONFIRMED in T5: `gemini-2.5-flash-lite` works for the reranker

Group D — web SDKs:
- Firecrawl Python SDK method `scrape_url(url, params={"formats": [...]})`
- Tavily Python SDK `client.search(q, max_results=)` returns `{"results": [{"url", "score"}]}`

Group E — infra:
- LangGraph issue #5790 (`langgraph dev` ignores user checkpointer) — only relevant if using `langgraph dev`
- Kuzu fixture is directory vs single file
- FalkorDB built-in HNSW (only relevant at v1 graduation)

Group F — undocumented behavior to confirm empirically:
- `add_episode` idempotency on duplicate URLs — re-ingesting the same arxiv abs across sessions: does Graphiti dedupe by `name`/`source_description`, or do duplicate episodic nodes accumulate?
- Per-domain `_domain_locks` in `web.py` are module-global — fine for the single-process Streamlit app, silently ineffective if a worker process is added later

Graphiti 0.29.0 API quirks discovered during T5 (now codified in `graphiti_setup.py` / nodes / reflect):
1. `Graphiti.build_indices_and_constraints()` is a **no-op for KuzuDriver** in 0.29.0; the real implementation lives at `driver.graph_ops.build_indices_and_constraints(driver)`. Handled inside `get_graphiti()` so consumers don't need to know the workaround exists.
2. `add_episode` returns `AddEpisodeResults`, **not** an `EpisodicNode`; the episode UUID is at `result.episode.uuid`. Consumers that previously used `ep.uuid` must switch to `result.episode.uuid`.
3. `search()` returns `list[EntityEdge]` only — episodes whose body contains a single named entity yield zero edges and an empty result set. Smoke-test bodies must be deliberately relational.
4. `reference_time=None` is rejected; pass a `datetime` (UTC, e.g. `datetime.now(timezone.utc)`).

Each tag in the code is exactly `UNVERIFIED:` so a single grep surfaces them all.

## 15. Phase order (unchanged from Revision 2)

1. Eval harness skeleton + 6 test cases (failing).
2. Graphiti + Gemini wiring; entity types; first end-to-end ingest.
3. Salience gate; conditional looping `should_search_more`.
4. Reflection top-level graph; manual `python -m research_agent.reflect`.
5. Contradictions surfacing in `answer_node` (Phase 4 work — defer until 1–4 stable).

## 16. Bugs — fix log

Revision 2 (R2) bugs in rows 1–10. Round-2 review of *this* spec adds rows 11–18.



| # | Issue | Fix in this spec |
|---|-------|------------------|
| 1 | `AsyncSqliteSaver` opened with bare `__aenter__`, never exited (leak) | `@asynccontextmanager` factories, `async with research_app() as app:` everywhere |
| 2 | `response.parsed` could be `None`, no fallback | `_parsed_or_raise` helper |
| 3 | No 429 retry; per-URL fan-out blasts free tier | `_gen` wrapper with exponential backoff |
| 4 | LangGraph + Graphiti concurrency stacked | `max_concurrency=4` in LangGraph, `SEMAPHORE_LIMIT=32` in Graphiti — single bound |
| 5 | `{topics}` referenced but never stored | `config.toml` → `settings.TOPICS`, injected into salience prompt |
| 6 | Reranker configured but retrieval path didn't confirm use | Tagged UNVERIFIED with explicit fallback note in `retrieve_node` |
| 7 | Reflection labeled "subgraph" but compiled separately | Spec calls it what it is: separate top-level graph |
| 8 | Eval fixture "copy Kuzu file" while DB live | Build-once: clean shutdown via `async with` exit, then `shutil.copy*` |
| 9 | Streamlit/CLI dual mention | Streamlit only; CLI exists only for `reflect` |
| 10 | Line budget mismatch (300 vs 340) | Recounted: state ~30 + nodes ~150 + graph factories ~50 + web ~40 + reflect ~50 + UI ~30 + eval ~80 + schemas ~30 = **~460 lines.** R2's 300 was measuring "domain code only" (nodes + Graphiti wiring + adapter + DeepEval + helpers). 460 here measures the full project including Streamlit UI, reflection top-level graph, fixture builder, retry/lifecycle plumbing — none of which were wrong omissions in R2; the scope just expanded. Both numbers are honest for what they measure |
| 11 | Reflection state would leak `_episodes`/`_patterns` into checkpoint (transient keys still serialized) | `ReflectState` now holds `episode_uuids` + `patterns` as proper fields; `synthesize` re-fetches bodies per-UUID |
| 12 | Three-tier salience depended on an UNVERIFIED Graphiti param to skip extraction | Collapsed to two-tier (`SALIENCE_CUTOFF` 0.5). Side-table for raw text deferred to v1 |
| 13 | `RESEARCH_KUZU_PATH` could be set after `research_agent` import → singleton bound to wrong path | Fixture builder sets env var before any `research_agent` import; pytest fixture calls `reset_graphiti()` after setting env |
| 14 | `_gen` retried only 429s, not transient 5xx | Retry set extended to `ResourceExhausted`, `ServiceUnavailable`, `InternalServerError`, `DeadlineExceeded` |
| 15 | DeepEval `retrieval_context` got the joined string instead of separate chunks | New `retrieval_chunks: list[str]` field on `ResearchState`; `retrieve_node` populates both `chunks` and joined `context`; eval reads `chunks` |
| 16 | `Claim.confidence: float \| None` would trip Gemini structured output | Field dropped; comment explains why |
| 17 | `iteration` field set to `0` in `plan_node`, never incremented → early-exit dead code | `plan_node` increments via `state.get("iteration", 0) + 1`; `recursion_limit=50` set explicitly in run config |
| 18 | Per-domain Firecrawl locks would silently fail across worker processes | Acceptable in single-process design; logged in §14 Group F so future-you doesn't get bitten |
| 19 | google-genai 1.x has no `google.api_core.exceptions` hierarchy | Switched to `from google.genai import errors as gx`; catch `ClientError`/`ServerError` and filter by `.code in {429,500,503,504}` (commit f633a68) |
| 20 | Defensive `raise` after `_gen` retry loop fall-through | Added unreachable RuntimeError so future refactors fail loudly (commit 9734ea2) |
| 21 | firecrawl-py 4.x renamed `FirecrawlApp` → `Firecrawl`, `scrape_url(url, params={"formats":[...]})` → `scrape(url, formats=[...])`, return is Pydantic Document not dict | Updated web.py imports and call (commit 2c3eba0) |
| 22 | Per-domain locks bypassed by mixed-case netloc (Arxiv.org vs arxiv.org) | `urlparse(url).netloc.lower()` (commit 048c64e) |
| 23 | Graphiti 0.29.0 `add_episode` returns `AddEpisodeResults`; episode UUID at `result.episode.uuid` not `.uuid` | All call sites updated (commit 76db5cf, 9b56202) |
| 24 | `Graphiti.build_indices_and_constraints()` is no-op for KuzuDriver in 0.29.0; real impl at `driver.graph_ops` | Centralized inside `get_graphiti()` async lazy init (commit 44b517f) |
| 25 | First-call race on Graphiti init when LangGraph fan-out dispatches concurrent ingest_one | Asyncio.Lock guard inside `get_graphiti()` (commit 5d24aa2) |
| 26 | Pre-built fixture has indices already; `build_indices_and_constraints` raises "already exists" | Try/except swallows specifically that RuntimeError (commit b48962a) |
| 27 | `add_episode` rejects `reference_time=None`; must pass datetime | All sites use `datetime.now(timezone.utc)` (commit 9b56202) |
| 28 | Per-URL ingest failures (Firecrawl 403, Gemini 503-exhausted, Graphiti extraction errors) abort whole fan-out | Bulletproof outer `try/except` in `ingest_one` returns skip dict; pipeline continues (commits 5d24aa2, 14cdb2c) |
| 29 | Reflect API names wrong: `g.get_episodes(after=...)` and `g.get_episode_by_uuid` don't exist in 0.29.0 | Use `g.retrieve_episodes(reference_time, last_n=N)` + `EpisodicNode.get_by_uuids(g.driver, uuids)` (commit 5b502fa) |
| 30 | `EpisodicNode.valid_at` is naive datetime; `since_iso` parsed UTC-aware → TypeError comparing | Added `_as_naive()` helper; both sides naive (commit 5b502fa) |
| 31 | Switched all reasoning calls (plan, salience, answer, reflect synth, eval judge) from `gemini-2.5-pro` to `gemini-2.5-flash` to extend cap headroom | Flash on all sites; only Graphiti reranker uses flash-lite (commit 15182f6) |
| 32 | DeepEval module-level `JUDGE = GeminiModel(...)` is benign at collection (no live call) but instantiation must not call API | Verified safe in T16 (commit 88b7c92) |
| 33 | `test_required_env_missing` failed once `.env` had real key — `load_dotenv` repopulated after `delenv` | Patch `dotenv.load_dotenv` at source so reload's `from dotenv import load_dotenv` re-binds the stub (commit e147b32) |

## 17. Open questions for user

None blocking. Implementer should:
- Resolve all UNVERIFIED tags before `pip install` (web access required)
- Decide eval dataset content (6–10 specific Q+expected_output pairs) — drafted skeletons only
- Pick stable URLs for fixture build (arxiv abs / archive.org)
