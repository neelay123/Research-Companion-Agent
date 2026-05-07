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
    """One add_episode + one search. Costs ~1 cent in Gemini calls."""

    async def go():
        g = get_graphiti()
        # NOTE (graphiti-core 0.29.0 quirk): Graphiti.build_indices_and_constraints
        # delegates to KuzuDriver.build_indices_and_constraints, which is a no-op.
        # The actual FTS index creation lives on driver.graph_ops, so call it
        # explicitly. Without this, search() raises:
        #   "Table RelatesToNode_ doesn't have an index with name edge_name_and_fact"
        await g.driver.graph_ops.build_indices_and_constraints(g.driver)
        # Episode body intentionally relational (two named entities + a verb)
        # because Graphiti.search() returns EntityEdges only — a single-entity
        # sentence yields zero edges and an empty result set.
        result = await g.add_episode(
            name="smoke",
            episode_body=(
                "Mamba is a state-space model architecture developed by "
                "Albert Gu and Tri Dao at Carnegie Mellon University in 2023."
            ),
            source_description="smoke-test",
            reference_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        # add_episode returns AddEpisodeResults; the EpisodicNode is on .episode
        assert result.episode.uuid
        assert len(result.edges) >= 1, "Gemini extracted no edges from the episode"
        results = await g.search(query="Mamba", num_results=5)
        assert len(results) >= 1

    asyncio.run(go())