"""Lazy Graphiti singleton wired to Kuzu + Gemini (LLM, embedder, reranker).

The singleton is created on first call to ``get_graphiti()`` so tests can
override ``RESEARCH_KUZU_PATH`` via monkeypatch before construction.

Critical: ``cross_encoder`` MUST be passed explicitly. The Graphiti default
is ``OpenAIRerankerClient`` which requires ``OPENAI_API_KEY`` (we don't have one).
"""

import os

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig

from .settings import GEMINI_API_KEY

# SEMAPHORE_LIMIT default in graphiti-core is 20 (graphiti_core/helpers.py).
# Set higher so LangGraph max_concurrency is the effective bound on parallelism.
os.environ.setdefault("SEMAPHORE_LIMIT", "32")

_instance: Graphiti | None = None


def _build(db_path: str) -> Graphiti:
    return Graphiti(
        graph_driver=KuzuDriver(db=db_path),
        llm_client=GeminiClient(
            config=LLMConfig(api_key=GEMINI_API_KEY, model="gemini-2.5-pro"),
        ),
        embedder=GeminiEmbedder(
            config=GeminiEmbedderConfig(
                api_key=GEMINI_API_KEY,
                embedding_model="gemini-embedding-001",
                embedding_dim=1536,
            ),
        ),
        cross_encoder=GeminiRerankerClient(
            config=LLMConfig(
                api_key=GEMINI_API_KEY,
                model="gemini-2.5-flash-lite",
            ),
        ),
    )


def get_graphiti() -> Graphiti:
    """Lazy singleton. Reads RESEARCH_KUZU_PATH at first call so tests can override."""
    global _instance
    if _instance is None:
        path = os.environ.get("RESEARCH_KUZU_PATH", "./graph.kuzu")
        _instance = _build(path)
    return _instance


def reset_graphiti() -> None:
    """Tests only: drop singleton so next get_graphiti() re-reads env."""
    global _instance
    _instance = None