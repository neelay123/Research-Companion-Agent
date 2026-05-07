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
