import asyncio
import pytest
from research_agent.graphiti_setup import reset_graphiti
from research_agent.graph import research_app

pytestmark = pytest.mark.smoke

def test_full_research_run(tmp_path, monkeypatch):
    """One question end-to-end. Costs ~$0.20."""
    monkeypatch.setenv("RESEARCH_KUZU_PATH", str(tmp_path / "scratch.kuzu"))
    reset_graphiti()
    async def go():
        async with research_app() as app:
            cfg = {"configurable": {"thread_id": "smoke-1", "max_concurrency": 4},
                   "recursion_limit": 50}
            async for ev in app.astream(
                {"question": "What are state space models like Mamba?"},
                cfg, stream_mode="updates",
            ):
                pass
            state = await app.aget_state(cfg)
        assert state.values.get("answer")
        assert isinstance(state.values.get("citations", []), list)
        assert state.values.get("retrieval_chunks")
    asyncio.run(go())
