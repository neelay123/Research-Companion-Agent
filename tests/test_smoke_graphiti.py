"""Live round-trip smoke test for Graphiti + Kuzu + Gemini wiring.

Costs roughly $0.01 in Gemini API calls per run. Skipped unless
``-m smoke`` is selected.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from research_agent.graphiti_setup import get_graphiti, reset_graphiti

pytestmark = pytest.mark.smoke


@pytest.fixture
def scratch_db(tmp_path, monkeypatch):
    p = tmp_path / "scratch.kuzu"
    monkeypatch.setenv("RESEARCH_KUZU_PATH", str(p))
    reset_graphiti()
    yield p
    reset_graphiti()


def test_graphiti_round_trip(scratch_db):
    """One add_episode + one search via get_graphiti(). Costs ~1¢ in Gemini calls."""
    from datetime import datetime, timezone
    async def go():
        g = await get_graphiti()
        # build_indices_and_constraints is now handled inside get_graphiti() on first call.
        result = await g.add_episode(
            name="smoke",
            episode_body=(
                "Mamba is a state-space model introduced in 2023 by Albert Gu and "
                "Tri Dao at Carnegie Mellon."
            ),
            source_description="smoke-test",
            reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        # add_episode returns AddEpisodeResults; episode UUID is at .episode.uuid
        assert result.episode.uuid
        # Single-entity sentences yield zero EntityEdges; assert relational edges exist
        assert len(result.edges) >= 1, "Gemini extracted zero edges (try a more relational sentence)"
        results = await g.search(query="state space models", num_results=5)
        assert len(results) >= 1
    asyncio.run(go())