"""Lazy Graphiti singleton wired to Gemini for LLM, embedding, and reranking.

The constructor's `cross_encoder` is mandatory: omitting it falls back to
OpenAIRerankerClient, which silently demands OPENAI_API_KEY at first use.

`get_graphiti()` is async because graphiti-core 0.29.0's
Graphiti.build_indices_and_constraints() is a no-op for the Kuzu driver — the
real implementation lives at driver.graph_ops, and must be invoked once before
the first hybrid search. Doing it here means no consumer needs to know the
workaround exists.
"""
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

def _build(db_path: str) -> Graphiti:
    return Graphiti(
        graph_driver=KuzuDriver(db=db_path),
        llm_client=GeminiClient(config=LLMConfig(
            api_key=GEMINI_API_KEY, model="gemini-2.5-pro",
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
    """
    global _instance, _indices_built
    if _instance is None:
        path = os.environ.get("RESEARCH_KUZU_PATH", "./graph.kuzu")
        _instance = _build(path)
    if not _indices_built:
        await _instance.driver.graph_ops.build_indices_and_constraints(_instance.driver)
        _indices_built = True
    return _instance

def reset_graphiti() -> None:
    """Tests only: drop singleton + index-built flag so next get_graphiti() re-reads env."""
    global _instance, _indices_built
    _instance = None
    _indices_built = False
