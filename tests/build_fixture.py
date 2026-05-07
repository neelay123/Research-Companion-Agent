"""Run once. Builds tests/fixtures/graph.kuzu from real Firecrawl + Gemini.
   IMPORTANT: do not run during normal pytest — costs real money."""
import os, asyncio, shutil
from pathlib import Path

PINNED_QUESTIONS = [
    "What are state-space models like Mamba?",
    "Who are the authors of the original Mamba paper?",
    "When was Mamba first posted to arXiv?",
    "What is the time complexity of attention vs SSMs?",
    "What is RWKV?",
    "How do Mamba and RWKV compare on long-sequence tasks?",
]
SCRATCH = Path("tests/fixtures/_scratch.kuzu")
TARGET  = Path("tests/fixtures/graph.kuzu")

# Set env BEFORE imports so the singleton picks up scratch path on first build.
os.environ["RESEARCH_KUZU_PATH"] = str(SCRATCH)

from research_agent.graph import research_app          # noqa: E402
from research_agent.graphiti_setup import reset_graphiti  # noqa: E402

async def main():
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True) if SCRATCH.is_dir() else SCRATCH.unlink()
    reset_graphiti()
    async with research_app() as app:
        for q in PINNED_QUESTIONS:
            cfg = {"configurable": {"thread_id": f"fixture-{hash(q)}"},
                   "recursion_limit": 50}
            print(f"--- {q} ---")
            try:
                async for ev in app.astream({"question": q}, cfg, stream_mode="updates"):
                    for node_name in ev:
                        print(f"  ran: {node_name}")
            except Exception as e:
                # Skip a question on persistent upstream failure (Gemini 503 storms,
                # Firecrawl outages); the previous questions' episodes are already
                # committed to Kuzu so partial fixture is still useful.
                print(f"  ABORTED question on: {type(e).__name__}: {str(e)[:200]}")
    if TARGET.exists():
        shutil.rmtree(TARGET, ignore_errors=True) if TARGET.is_dir() else TARGET.unlink()
    if SCRATCH.is_dir():
        shutil.copytree(SCRATCH, TARGET)
    else:
        shutil.copy(SCRATCH, TARGET)
    print(f"fixture written: {TARGET}")

if __name__ == "__main__":
    asyncio.run(main())
