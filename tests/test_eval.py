import os, asyncio, pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCaseParams
from research_agent.graph import research_app
from research_agent.graphiti_setup import reset_graphiti
from .eval_dataset import DATASET

pytestmark = pytest.mark.eval

# Verified: GeminiModel(model=..., api_key=..., ...) — kwargs match deepeval 1.x signature.
JUDGE = GeminiModel(model="gemini-2.5-flash", api_key=os.environ["GEMINI_API_KEY"])

# Verified: GEval(name, evaluation_params, criteria, model, threshold, ...).
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
    import shutil
    src = "tests/fixtures/graph.kuzu"
    dst = tmp_path_factory.mktemp("kuzu") / "graph.kuzu"
    shutil.copytree(src, dst) if os.path.isdir(src) else shutil.copy(src, dst)
    os.environ["RESEARCH_KUZU_PATH"] = str(dst)
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
