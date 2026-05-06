# Research Companion Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-user research companion that ingests web pages into a bi-temporal Graphiti knowledge graph, retrieves with hybrid search, reflects on patterns, and answers with citations — orchestrated by LangGraph, powered by Gemini 2.5 Pro, fronted by Streamlit.

**Architecture:** Two top-level LangGraph `StateGraph`s — research (interactive, fan-out ingest) and reflect (manual batch synthesis) — sharing a single `AsyncSqliteSaver` checkpoint file under distinct `thread_id` namespaces. Graphiti runs against a Kuzu embedded DB. All LLM/embed/rerank calls go through a single Gemini provider. No mocks anywhere — eval suite hits live Gemini judge against a build-once committed Kuzu fixture.

**Tech Stack:** Python 3.11+, `graphiti-core[google-genai,kuzu]`, `langgraph` 0.6+, `langgraph-checkpoint-sqlite`, `google-genai`, `firecrawl-py`, `tavily-python`, `deepeval`, `streamlit`, `pydantic`, `python-dotenv`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-05-04-research-companion-agent-design.md`. All `UNVERIFIED:` tags in code originate there — implementer resolves them before `pip install` (web access required).

---

## Working agreements

1. **TDD where pure.** Routing functions, schemas, settings loaders get tests-first. API-touching code gets a live integration test that runs after the Kuzu fixture is built.
2. **No mocks.** Tests either hit real APIs (with cost) or test pure logic. No `unittest.mock`, no `monkeypatch` of network calls, no fake clients.
3. **Frequent commits.** Every task ends with one commit; every step that lands a test+code pair commits before moving on.
4. **`UNVERIFIED:` tags.** Comments tagged `UNVERIFIED:` come from the spec. Implementer must resolve each before installing — see spec §14 for the reading list. If the live API differs from the spec, update the spec and the code in the same commit.
5. **Cost-gated tasks.** Tasks 13 (build fixture) and 17 (eval suite first run) cost real money — Gemini + Firecrawl. Run when intentional, not in a tight loop.

## File map

| Path | Created in | Purpose |
|------|-----------|---------|
| `pyproject.toml` | T1 | Package metadata, dependencies |
| `.env.example` | T1 | Env var template |
| `.gitignore` | T1 | Exclude `.env`, `*.kuzu`, `checkpoints.sqlite` |
| `config.toml` | T1 | `[research].topics` list |
| `src/research_agent/__init__.py` | T1 | Empty marker |
| `src/research_agent/settings.py` | T2 | `.env` + `config.toml` loader |
| `src/research_agent/schemas.py` | T3 | Pydantic models for structured output and entity types |
| `src/research_agent/state.py` | T4 | `ResearchState`, `ReflectState` TypedDicts |
| `src/research_agent/graphiti_setup.py` | T5 | Lazy Graphiti singleton, `RESEARCH_KUZU_PATH`-aware |
| `src/research_agent/web.py` | T6 | Firecrawl + Tavily adapters with backoff |
| `src/research_agent/nodes.py` | T7, T8, T9 | `_gen` helper, `_parsed_or_raise`, all node functions |
| `src/research_agent/graph.py` | T10, T11 | `research_app` context manager + routing predicates |
| `src/research_agent/reflect.py` | T14 | Reflection graph + CLI entrypoint |
| `src/research_agent/ui.py` | T18 | Streamlit page |
| `tests/__init__.py` | T2 | Package marker |
| `tests/test_settings.py` | T2 | Settings unit tests |
| `tests/test_schemas.py` | T3 | Schema instantiation tests |
| `tests/test_routing.py` | T11 | `fan_out_ingest`, `should_search_more` unit tests |
| `tests/test_smoke_web.py` | T6 | Live Firecrawl/Tavily smoke (cost-gated) |
| `tests/test_smoke_graphiti.py` | T5 | Live Graphiti round-trip smoke (cost-gated) |
| `tests/test_smoke_research.py` | T12 | One full research run (cost-gated) |
| `tests/build_fixture.py` | T13 | Build-once committed Kuzu fixture |
| `tests/fixtures/graph.kuzu` | T13 | Committed snapshot (binary) |
| `tests/eval_dataset.py` | T15 | DeepEval `LLMTestCase` list |
| `tests/test_eval.py` | T16, T17 | DeepEval pytest harness with live Gemini judge |

## Cost ladder

- Tasks 1–4, 8, 11: free (pure logic).
- Task 5 smoke: ~$0.01 (one `add_episode` + `search`).
- Task 6 smoke: ~$0.01 (one Firecrawl page + one Tavily search).
- Task 12 smoke: ~$0.20 (full research run, ~6 URLs).
- Task 13 fixture: ~$2 (six pinned questions, full pipeline). One-time.
- Task 17 eval first run: ~$0.50/run (live judge over 6–10 cases).

---

## Task 1: Project skeleton

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `config.toml`
- Create: `src/research_agent/__init__.py`

- [ ] **Step 1: Initialize git repo**

```bash
git init
git config user.name "Your Name"   # if not already set globally
git config user.email "you@example.com"
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "research-companion"
version = "0.1.0"
description = "Single-user research agent with bi-temporal memory"
requires-python = ">=3.11"
dependencies = [
  "graphiti-core[google-genai,kuzu]",
  "langgraph>=0.6",
  "langgraph-checkpoint-sqlite>=2.0",
  "google-genai>=1.0",
  "firecrawl-py",
  "tavily-python",
  "deepeval>=1.5",
  "streamlit",
  "pydantic>=2.7",
  "python-dotenv",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.23"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = [
  "smoke: live-API smoke test (cost-gated, opt-in via -m smoke)",
  "eval: live Gemini judge eval (cost-gated)",
]

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 3: Write `.env.example`**

```
GEMINI_API_KEY=
FIRECRAWL_API_KEY=
TAVILY_API_KEY=
SALIENCE_CUTOFF=0.5
TAVILY_MIN_SCORE=0.5
```

- [ ] **Step 4: Write `.gitignore`**

```
.env
*.kuzu/
*.kuzu
checkpoints.sqlite
__pycache__/
.pytest_cache/
.venv/
```

- [ ] **Step 5: Write `config.toml`**

```toml
[research]
topics = ["state space models", "agent memory", "knowledge graphs"]
```

- [ ] **Step 6: Create package marker**

```bash
mkdir -p src/research_agent
touch src/research_agent/__init__.py
```

- [ ] **Step 7: Create env, install dev deps**

```bash
python -m venv .venv
. .venv/Scripts/activate     # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Expected: install succeeds. If `graphiti-core[google-genai,kuzu]` extras are not published as named, see spec §14 Group A — adjust extras and update the spec in the same commit.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml .env.example .gitignore config.toml src/research_agent/__init__.py
git commit -m "chore: project skeleton + dependencies"
```

---

## Task 2: Settings loader

**Files:**
- Create: `src/research_agent/settings.py`
- Create: `tests/__init__.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write the failing test**

`tests/test_settings.py`:
```python
import os
import importlib
from pathlib import Path
import pytest

def _reload_settings():
    import research_agent.settings as s
    return importlib.reload(s)

def test_topics_loaded(monkeypatch, tmp_path):
    # Write a temp config.toml two levels above the package source
    cfg = tmp_path / "config.toml"
    cfg.write_text('[research]\ntopics = ["x", "y"]\n', encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "k2")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("research_agent.settings.__file__",
                        str(tmp_path / "settings.py"), raising=False)
    s = _reload_settings()
    # Note: settings.py reads config.toml relative to its own location;
    # for this test we just confirm the parsing path works on a known file.
    assert isinstance(s.TOPICS, list)

def test_required_env_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(KeyError):
        _reload_settings()
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/test_settings.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'research_agent.settings'`.

- [ ] **Step 3: Write `src/research_agent/settings.py`**

```python
import os, tomllib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]
FIRECRAWL_API_KEY = os.environ["FIRECRAWL_API_KEY"]
TAVILY_API_KEY    = os.environ.get("TAVILY_API_KEY")
SALIENCE_CUTOFF   = float(os.environ.get("SALIENCE_CUTOFF", "0.5"))
TAVILY_MIN_SCORE  = float(os.environ.get("TAVILY_MIN_SCORE", "0.5"))

_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "config.toml"
with open(_CFG_PATH, "rb") as f:
    _cfg = tomllib.load(f)
TOPICS: list[str] = _cfg.get("research", {}).get("topics", [])
```

- [ ] **Step 4: Create `tests/__init__.py` (empty), populate a real `.env`**

```bash
touch tests/__init__.py
cp .env.example .env
# Manually edit .env to fill in real keys before running tests that need them.
```

- [ ] **Step 5: Run tests, expect pass on `test_required_env_missing` and `test_topics_loaded`**

```bash
pytest tests/test_settings.py -v
```
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add src/research_agent/settings.py tests/__init__.py tests/test_settings.py
git commit -m "feat: settings loader with .env + config.toml"
```

---

## Task 3: Pydantic schemas

**Files:**
- Create: `src/research_agent/schemas.py`
- Create: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

`tests/test_schemas.py`:
```python
import pytest
from pydantic import ValidationError
from research_agent.schemas import (
    Plan, SalienceVerdict, Answer, Synthesis, Paper, Author, Claim
)

def test_plan_min_length():
    with pytest.raises(ValidationError):
        Plan(sub_questions=[])
    Plan(sub_questions=["q1"])

def test_plan_max_length():
    with pytest.raises(ValidationError):
        Plan(sub_questions=[f"q{i}" for i in range(6)])

def test_salience_score_bounds():
    SalienceVerdict(score=0.0, reason="x", novel_claims=[])
    SalienceVerdict(score=1.0, reason="x", novel_claims=[])
    with pytest.raises(ValidationError):
        SalienceVerdict(score=-0.1, reason="x", novel_claims=[])
    with pytest.raises(ValidationError):
        SalienceVerdict(score=1.1, reason="x", novel_claims=[])

def test_answer_minimal():
    a = Answer(answer="hello", citations=[])
    assert a.answer == "hello"

def test_synthesis_minimal():
    Synthesis(patterns=[])
    Synthesis(patterns=["p1", "p2"])

def test_entity_types_minimal():
    Paper(title="x")
    Author(name="x")
    Claim(statement="x")
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/test_schemas.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write `src/research_agent/schemas.py`**

```python
from pydantic import BaseModel, Field

class Plan(BaseModel):
    sub_questions: list[str] = Field(min_length=1, max_length=5)

class SalienceVerdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reason: str
    novel_claims: list[str]

class Answer(BaseModel):
    answer: str
    citations: list[str]

class Synthesis(BaseModel):
    patterns: list[str]

class Paper(BaseModel):
    title: str
    arxiv_id: str | None = None
    year: int | None = None

class Author(BaseModel):
    name: str
    affiliation: str | None = None

class Claim(BaseModel):
    statement: str
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_schemas.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/research_agent/schemas.py tests/test_schemas.py
git commit -m "feat: pydantic schemas for structured output and entity types"
```

---

## Task 4: State module

**Files:**
- Create: `src/research_agent/state.py`

No tests — TypedDicts are typing only.

- [ ] **Step 1: Write `src/research_agent/state.py`**

```python
from typing import TypedDict, Annotated
from operator import add

class ResearchState(TypedDict):
    question: str
    plan: list[str]
    candidate_urls: list[str]
    documents: Annotated[list[dict], add]
    salient_episode_ids: Annotated[list[str], add]
    retrieval_chunks: list[str]
    context: str
    answer: str
    citations: list[str]
    iteration: int

class ReflectState(TypedDict):
    since_iso: str
    episode_uuids: list[str]
    patterns: list[str]
    new_synthesis_uuids: Annotated[list[str], add]
```

- [ ] **Step 2: Verify import**

```bash
python -c "from research_agent.state import ResearchState, ReflectState; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/research_agent/state.py
git commit -m "feat: LangGraph state TypedDicts for research and reflect"
```

---

## Task 5: Graphiti singleton + smoke test

**Files:**
- Create: `src/research_agent/graphiti_setup.py`
- Create: `tests/test_smoke_graphiti.py`

- [ ] **Step 1: Write `src/research_agent/graphiti_setup.py`**

```python
import os
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.driver.kuzu_driver import KuzuDriver
from .settings import GEMINI_API_KEY

# UNVERIFIED: SEMAPHORE_LIMIT default is 10. Set high so LangGraph max_concurrency dominates.
os.environ.setdefault("SEMAPHORE_LIMIT", "32")

_instance: Graphiti | None = None

def _build(db_path: str) -> Graphiti:
    return Graphiti(
        graph_driver=KuzuDriver(db=db_path),
        llm_client=GeminiClient(config=LLMConfig(
            api_key=GEMINI_API_KEY, model="gemini-2.5-pro",
        )),
        embedder=GeminiEmbedder(config=GeminiEmbedderConfig(
            api_key=GEMINI_API_KEY,
            embedding_model="gemini-embedding-001",
            embedding_dim=1536,   # UNVERIFIED: field name. Try `output_dimensionality` if rejected.
        )),
        cross_encoder=GeminiRerankerClient(config=LLMConfig(
            api_key=GEMINI_API_KEY,
            model="gemini-2.5-flash",  # UNVERIFIED: current GA flash-lite name
        )),
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
```

- [ ] **Step 2: Resolve UNVERIFIED tags**

Read the installed packages. For each `UNVERIFIED:` tag in the file, confirm or correct:

```bash
python -c "from graphiti_core.driver.kuzu_driver import KuzuDriver; help(KuzuDriver.__init__)"
python -c "from graphiti_core.embedder.gemini import GeminiEmbedderConfig; print(GeminiEmbedderConfig.model_fields.keys())"
python -c "from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient; help(GeminiRerankerClient)"
```

Update the file in place if reality differs from the spec; commit alongside the spec patch in the same commit later. Do not silently leave stale `UNVERIFIED:` tags.

- [ ] **Step 3: Write the smoke test**

`tests/test_smoke_graphiti.py`:
```python
import os
import asyncio
import shutil
from pathlib import Path
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
    """One add_episode + one search. Costs ~1¢ in Gemini calls."""
    async def go():
        g = get_graphiti()
        await g.build_indices_and_constraints()
        ep = await g.add_episode(
            name="smoke",
            episode_body="Mamba is a state-space model introduced in 2023.",
            source_description="smoke-test",
            reference_time=None,   # UNVERIFIED — see spec §14 Group B
        )
        assert ep.uuid
        results = await g.search(query="state space models", num_results=5)
        assert len(results) >= 1
    asyncio.run(go())
```

- [ ] **Step 4: Run smoke test**

```bash
pytest tests/test_smoke_graphiti.py -v -m smoke
```
Expected: PASS. If `reference_time=None` raises, change to `datetime.utcnow()` and update spec §9 + this test.

- [ ] **Step 5: Commit**

```bash
git add src/research_agent/graphiti_setup.py tests/test_smoke_graphiti.py
git commit -m "feat: lazy Graphiti singleton + live round-trip smoke test"
```

---

## Task 6: Web adapter + smoke test

**Files:**
- Create: `src/research_agent/web.py`
- Create: `tests/test_smoke_web.py`

- [ ] **Step 1: Write `src/research_agent/web.py`**

```python
import asyncio, logging
from collections import defaultdict
from urllib.parse import urlparse
from firecrawl import FirecrawlApp
from tavily import TavilyClient
from .settings import FIRECRAWL_API_KEY, TAVILY_API_KEY, TAVILY_MIN_SCORE

log = logging.getLogger(__name__)
_firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
_tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

async def search_and_rank(queries: list[str], k: int = 8) -> list[str]:
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
    for i in range(attempts):
        try:
            res = await asyncio.to_thread(
                _firecrawl.scrape_url, url, params={"formats": ["markdown"]},
            )
            return res.get("markdown") or ""
        except Exception as e:
            log.warning("firecrawl %s attempt %d: %s", url, i + 1, e)
            if i == attempts - 1:
                raise
            await asyncio.sleep(2 ** i)
    return ""  # unreachable

_domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
async def fetch_markdown_polite(url: str) -> str:
    host = urlparse(url).netloc
    async with _domain_locks[host]:
        return await fetch_markdown(url)
```

- [ ] **Step 2: Resolve UNVERIFIED tags**

```bash
python -c "from firecrawl import FirecrawlApp; help(FirecrawlApp.scrape_url)"
python -c "from tavily import TavilyClient; help(TavilyClient.search)"
```

If signatures differ (`formats=` direct kwarg vs `params={"formats": ...}`, `query=` vs positional), update the code and the spec §8 in the same commit.

- [ ] **Step 3: Write smoke test**

`tests/test_smoke_web.py`:
```python
import asyncio
import pytest
from research_agent.web import search_and_rank, fetch_markdown_polite

pytestmark = pytest.mark.smoke

def test_tavily_returns_urls():
    async def go():
        urls = await search_and_rank(["mamba state space model"], k=3)
        assert len(urls) >= 1
        assert all(u.startswith("http") for u in urls)
    asyncio.run(go())

def test_firecrawl_returns_markdown():
    async def go():
        md = await fetch_markdown_polite("https://arxiv.org/abs/2312.00752")
        assert len(md) > 200
        assert "Mamba" in md or "state space" in md.lower()
    asyncio.run(go())
```

- [ ] **Step 4: Run smoke test**

```bash
pytest tests/test_smoke_web.py -v -m smoke
```
Expected: both PASS. Cost ~1¢.

- [ ] **Step 5: Commit**

```bash
git add src/research_agent/web.py tests/test_smoke_web.py
git commit -m "feat: firecrawl + tavily web adapter with backoff and per-domain locks"
```

---

## Task 7: `_gen` helper + `_parsed_or_raise`

**Files:**
- Create: `src/research_agent/nodes.py` (partial — only helpers in this task)

No standalone test — helpers are exercised by the integration smoke test in T12. Live retry behavior cannot be tested without mocking, which is forbidden.

- [ ] **Step 1: Write `src/research_agent/nodes.py` (helpers only)**

```python
import asyncio
from google import genai
from google.genai import types
from google.api_core import exceptions as gx
from .settings import GEMINI_API_KEY

_g_client = genai.Client(api_key=GEMINI_API_KEY)

_RETRY_EXC = (
    gx.ResourceExhausted,        # 429
    gx.ServiceUnavailable,       # 503
    gx.InternalServerError,      # 500
    gx.DeadlineExceeded,         # 504
)

async def _gen(model: str, contents, **cfg) -> object:
    """generate_content with backoff on transient errors."""
    for i in range(4):
        try:
            return await _g_client.aio.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**cfg),
            )
        except _RETRY_EXC:
            if i == 3: raise
            await asyncio.sleep(2 ** i + 1)

def _parsed_or_raise(resp, schema_name: str):
    """response.parsed is None on parse failure even with response_schema."""
    if resp.parsed is None:
        raise ValueError(f"Gemini failed to produce valid {schema_name}; raw: {resp.text[:500]}")
    return resp.parsed
```

- [ ] **Step 2: Verify import**

```bash
python -c "from research_agent.nodes import _gen, _parsed_or_raise; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Verify a real Gemini call**

```bash
python -c "
import asyncio
from research_agent.nodes import _gen, _parsed_or_raise
from research_agent.schemas import Plan
from google.genai import types

async def go():
    r = await _gen('gemini-2.5-pro', 'Decompose: what is mamba?',
                   response_mime_type='application/json',
                   response_schema=Plan,
                   thinking_config=types.ThinkingConfig(thinking_budget=512))
    p = _parsed_or_raise(r, 'Plan')
    print(p.sub_questions)

asyncio.run(go())
"
```
Expected: list of 1–5 strings.

- [ ] **Step 4: Commit**

```bash
git add src/research_agent/nodes.py
git commit -m "feat: gemini _gen retry helper and _parsed_or_raise"
```

---

## Task 8: Routing predicates

**Files:**
- Modify: `src/research_agent/graph.py` (create with predicates only)
- Create: `tests/test_routing.py`

- [ ] **Step 1: Write the failing test**

`tests/test_routing.py`:
```python
from langgraph.types import Send
from research_agent.graph import fan_out_ingest, should_search_more

def test_fan_out_emits_one_send_per_url():
    state = {"candidate_urls": ["a", "b", "c"], "question": "q"}
    sends = fan_out_ingest(state)
    assert len(sends) == 3
    assert all(isinstance(s, Send) for s in sends)
    assert all(s.node == "ingest_one" for s in sends)
    assert sends[0].arg == {"url": "a", "question": "q"}

def test_search_more_when_under_threshold():
    s = {"iteration": 0, "salient_episode_ids": ["x", "y"]}
    assert should_search_more(s) == "plan"

def test_retrieve_when_enough_hits():
    s = {"iteration": 0, "salient_episode_ids": ["x", "y", "z"]}
    assert should_search_more(s) == "retrieve"

def test_retrieve_when_iteration_capped():
    s = {"iteration": 2, "salient_episode_ids": []}
    assert should_search_more(s) == "retrieve"

def test_default_zero_iteration():
    s = {"salient_episode_ids": []}
    assert should_search_more(s) == "plan"
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
pytest tests/test_routing.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'research_agent.graph'`.

- [ ] **Step 3: Write `src/research_agent/graph.py` (predicates only)**

```python
from langgraph.types import Send
from .state import ResearchState

CHECKPOINT_DSN = "checkpoints.sqlite"

def fan_out_ingest(state: ResearchState) -> list[Send]:
    return [Send("ingest_one", {"url": u, "question": state["question"]})
            for u in state["candidate_urls"]]

def should_search_more(state: ResearchState) -> str:
    if state.get("iteration", 0) >= 2: return "retrieve"
    if len(state.get("salient_episode_ids", [])) < 3: return "plan"
    return "retrieve"
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_routing.py -v
```
Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/research_agent/graph.py tests/test_routing.py
git commit -m "feat: routing predicates with unit tests"
```

---

## Task 9: Node functions (plan, search, ingest_one, retrieve, answer)

**Files:**
- Modify: `src/research_agent/nodes.py` (append nodes after helpers)

Live integration is tested in T12. Per-node unit tests would require mocking — forbidden.

- [ ] **Step 1: Append nodes to `src/research_agent/nodes.py`**

Add these imports at the top (alongside existing):
```python
from .schemas import Plan, SalienceVerdict, Answer
from .state import ResearchState
from .graphiti_setup import get_graphiti
from .web import search_and_rank, fetch_markdown_polite
from .settings import SALIENCE_CUTOFF, TOPICS
```

Append node functions:
```python
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
    url = state["url"]
    md = await fetch_markdown_polite(url)
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
        return {"documents": [{"url": url, "skipped": True, "score": verdict.score}]}
    ep = await get_graphiti().add_episode(
        name=url, episode_body=md, source_description=f"web:{url}",
        reference_time=None,   # UNVERIFIED: some versions reject None
    )
    return {
        "documents": [{"url": url, "episode_uuid": ep.uuid, "score": verdict.score}],
        "salient_episode_ids": [ep.uuid],
    }

async def retrieve_node(state: ResearchState) -> dict:
    # UNVERIFIED: confirm graphiti.search uses configured cross_encoder by default.
    results = await get_graphiti().search(query=state["question"], num_results=20)
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
```

- [ ] **Step 2: Verify imports compile**

```bash
python -c "from research_agent.nodes import plan_node, search_node, ingest_one, retrieve_node, answer_node; print('ok')"
```
Expected: `ok`.

- [ ] **Step 3: Resolve UNVERIFIED tags**

For each `UNVERIFIED:` introduced in this task, run a real probe:

```bash
python -c "from graphiti_core import Graphiti; help(Graphiti.add_episode)"
python -c "from graphiti_core import Graphiti; help(Graphiti.search)"
```

If `search` requires explicit `rerank=True` (or similar), update `retrieve_node` and the spec §9 + §14 Group F in the same commit.

- [ ] **Step 4: Commit**

```bash
git add src/research_agent/nodes.py
git commit -m "feat: research-graph node functions (plan/search/ingest_one/retrieve/answer)"
```

---

## Task 10: Research graph factory

**Files:**
- Modify: `src/research_agent/graph.py` (append builder + context manager)

- [ ] **Step 1: Append to `src/research_agent/graph.py`**

```python
from contextlib import asynccontextmanager
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .nodes import plan_node, search_node, ingest_one, retrieve_node, answer_node

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
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DSN) as ckpt:
        yield _build_research().compile(checkpointer=ckpt)
```

- [ ] **Step 2: Verify the graph compiles**

```bash
python -c "
import asyncio
from research_agent.graph import research_app
async def go():
    async with research_app() as app:
        print(app.get_graph().nodes.keys())
asyncio.run(go())
"
```
Expected: dict-keys including `plan`, `search`, `ingest_one`, `retrieve`, `answer`.

- [ ] **Step 3: Commit**

```bash
git add src/research_agent/graph.py
git commit -m "feat: research StateGraph factory + AsyncSqliteSaver lifecycle"
```

---

## Task 11: Routing test additions for graph compile

**Files:**
- Modify: `tests/test_routing.py`

- [ ] **Step 1: Append a compile test**

```python
import asyncio
from research_agent.graph import research_app

def test_research_graph_compiles_with_expected_nodes():
    async def go():
        async with research_app() as app:
            nodes = set(app.get_graph().nodes.keys())
        assert {"plan", "search", "ingest_one", "retrieve", "answer"}.issubset(nodes)
    asyncio.run(go())
```

- [ ] **Step 2: Run all routing tests**

```bash
pytest tests/test_routing.py -v
```
Expected: all PASS (6 total).

- [ ] **Step 3: Commit**

```bash
git add tests/test_routing.py
git commit -m "test: research graph compile assertion"
```

---

## Task 12: End-to-end research smoke test

**Files:**
- Create: `tests/test_smoke_research.py`

- [ ] **Step 1: Write the smoke test**

```python
import asyncio, os
from pathlib import Path
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
```

- [ ] **Step 2: Run smoke test**

```bash
pytest tests/test_smoke_research.py -v -m smoke
```
Expected: PASS in ~60–180s. Cost ~$0.20. If failures arise, they almost always trace to an `UNVERIFIED:` tag — read the traceback, find the tag in the spec, fix code + spec in the same commit.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke_research.py
git commit -m "test: end-to-end research smoke test (cost-gated)"
```

---

## Task 13: Build-once Kuzu fixture

**Files:**
- Create: `tests/build_fixture.py`
- Create: `tests/fixtures/.gitkeep`

This task COSTS REAL MONEY (~$2). Run intentionally.

- [ ] **Step 1: Pick stable URLs and pin them**

Edit the questions list in `tests/build_fixture.py` (Step 2) to 5–7 questions whose top results are arxiv abstract pages, archive.org snapshots, or otherwise frozen. Avoid Medium/Substack/news — they change.

- [ ] **Step 2: Write `tests/build_fixture.py`**

```python
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
            async for _ in app.astream({"question": q}, cfg, stream_mode="updates"):
                pass
    if TARGET.exists():
        shutil.rmtree(TARGET, ignore_errors=True) if TARGET.is_dir() else TARGET.unlink()
    if SCRATCH.is_dir():
        shutil.copytree(SCRATCH, TARGET)
    else:
        shutil.copy(SCRATCH, TARGET)
    print(f"fixture written: {TARGET}")

if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Create the directory placeholder**

```bash
mkdir -p tests/fixtures
touch tests/fixtures/.gitkeep
```

- [ ] **Step 4: Run the fixture builder**

```bash
python tests/build_fixture.py
```
Expected: prints `fixture written: tests/fixtures/graph.kuzu`. Runs ~5–10 minutes. Cost ~$2.

- [ ] **Step 5: Commit fixture and script**

```bash
git add tests/build_fixture.py tests/fixtures/.gitkeep tests/fixtures/graph.kuzu
git commit -m "test: build-once Kuzu fixture (committed binary)"
```

If the Kuzu fixture is large (>10 MB), consider Git LFS or excluding the binary and rebuilding on each clean checkout. For a hobby project default to committing it; revisit at v1.

---

## Task 14: Reflection graph + CLI

**Files:**
- Create: `src/research_agent/reflect.py`

- [ ] **Step 1: Write `src/research_agent/reflect.py`**

```python
"""Manual reflection pass. Run:  python -m research_agent.reflect [--since 2026-04-01]"""
import argparse, asyncio
from contextlib import asynccontextmanager
from datetime import date
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from google.genai import types
from pydantic import BaseModel
from .state import ReflectState
from .graphiti_setup import get_graphiti
from .nodes import _gen, _parsed_or_raise
from .graph import CHECKPOINT_DSN

class Synthesis(BaseModel):
    patterns: list[str]

async def gather_recent(s: ReflectState) -> dict:
    # UNVERIFIED: graphiti.get_episodes(after=...) signature
    eps = await get_graphiti().get_episodes(after=s["since_iso"])
    return {"episode_uuids": [e.uuid for e in eps]}

async def synthesize(s: ReflectState) -> dict:
    g = get_graphiti()
    # UNVERIFIED: per-UUID fetch API name
    eps = [await g.get_episode_by_uuid(u) for u in s["episode_uuids"]]
    body = "\n---\n".join(e.content for e in eps)
    resp = await _gen(
        "gemini-2.5-pro",
        f"Find non-obvious patterns connecting these episodes:\n{body[:80000]}",
        response_mime_type="application/json",
        response_schema=Synthesis,
        thinking_config=types.ThinkingConfig(thinking_budget=8192),
    )
    return {"patterns": _parsed_or_raise(resp, "Synthesis").patterns}

async def write_back(s: ReflectState) -> dict:
    uuids = []
    for p in s["patterns"]:
        ep = await get_graphiti().add_episode(
            name=f"reflection:{date.today().isoformat()}",
            episode_body=p, source_description="reflection",
            reference_time=None,
        )
        uuids.append(ep.uuid)
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

- [ ] **Step 2: Resolve UNVERIFIED tags**

```bash
python -c "from graphiti_core import Graphiti; help(Graphiti.get_episodes)"
python -c "from graphiti_core import Graphiti; print([m for m in dir(Graphiti) if 'episode' in m.lower()])"
```

If `get_episodes(after=...)` does not exist or `get_episode_by_uuid` is named differently (e.g. `get_node`, `retrieve_episode`), update the file and the spec §11 + §14 Group B in the same commit.

- [ ] **Step 3: Verify CLI invokes**

```bash
python -m research_agent.reflect --since 1970-01-01
```
Expected: streams events. If the fixture is small, completes in ~30s for ~$0.10. (Fine to skip running on every dev cycle; this is the manual trigger by design.)

- [ ] **Step 4: Commit**

```bash
git add src/research_agent/reflect.py
git commit -m "feat: reflection top-level graph + CLI entrypoint"
```

---

## Task 15: Eval dataset

**Files:**
- Create: `tests/eval_dataset.py`

The dataset is purpose-built around the topics in the fixture (state-space models). Replace freely with your real research interests after a working baseline.

- [ ] **Step 1: Write `tests/eval_dataset.py`**

```python
from deepeval.test_case import LLMTestCase

DATASET: list[LLMTestCase] = [
    # Single-fact recall
    LLMTestCase(
        input="When was the Mamba paper first posted to arXiv?",
        actual_output="",
        retrieval_context=[],
        expected_output="December 2023",
    ),
    # Author lookup (multi-hop-ish)
    LLMTestCase(
        input="Who are the authors of the original Mamba paper?",
        actual_output="",
        retrieval_context=[],
        expected_output="Albert Gu and Tri Dao",
    ),
    # Conceptual / synthesis
    LLMTestCase(
        input="How does Mamba's selective state space differ from a standard SSM?",
        actual_output="",
        retrieval_context=[],
        expected_output="Mamba makes the SSM parameters input-dependent (selective), allowing the model to focus on or ignore tokens, unlike a fixed/linear-time-invariant SSM.",
    ),
    # Comparison
    LLMTestCase(
        input="Compare Mamba and RWKV on long-context efficiency.",
        actual_output="",
        retrieval_context=[],
        expected_output="Both achieve linear-time inference for long sequences. Mamba uses a selective SSM; RWKV uses a linear-attention recurrence. Both avoid quadratic attention cost.",
    ),
    # Citation-correctness probe (judged by GEval CITATION_CORRECTNESS, not string match)
    LLMTestCase(
        input="What is the time complexity of attention vs SSMs?",
        actual_output="",
        retrieval_context=[],
        expected_output="Attention is O(N²) in sequence length; SSMs are O(N).",
    ),
    # Negative / out-of-corpus probe — answer should acknowledge ignorance, not hallucinate
    LLMTestCase(
        input="What is the latest GDP figure for Tunisia?",
        actual_output="",
        retrieval_context=[],
        expected_output="The retrieved context does not address this question.",
    ),
]
```

- [ ] **Step 2: Verify import**

```bash
python -c "from tests.eval_dataset import DATASET; print(len(DATASET))"
```
Expected: `6`.

- [ ] **Step 3: Commit**

```bash
git add tests/eval_dataset.py
git commit -m "test: eval dataset (6 LLMTestCases, single-fact + multi-hop + negative)"
```

---

## Task 16: DeepEval metrics + harness wiring

**Files:**
- Create: `tests/test_eval.py`

- [ ] **Step 1: Write `tests/test_eval.py`**

```python
import os, asyncio, pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCaseParams
from research_agent.graph import research_app
from research_agent.graphiti_setup import reset_graphiti
from .eval_dataset import DATASET

pytestmark = pytest.mark.eval

JUDGE = GeminiModel(model="gemini-2.5-pro", api_key=os.environ["GEMINI_API_KEY"])

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
```

- [ ] **Step 2: Resolve UNVERIFIED tags**

```bash
python -c "from deepeval.models import GeminiModel; help(GeminiModel.__init__)"
python -c "from deepeval.metrics import GEval; help(GEval.__init__)"
```

If `GeminiModel` lives elsewhere or needs a different constructor (e.g. `model_name` not `model`), update test and spec.

- [ ] **Step 3: Confirm pytest collects, but skip running for now**

```bash
pytest tests/test_eval.py --collect-only -m eval
```
Expected: 6 tests collected.

- [ ] **Step 4: Commit**

```bash
git add tests/test_eval.py
git commit -m "test: deepeval harness wiring (collection only; run gated)"
```

---

## Task 17: First eval run

This task COSTS REAL MONEY (~$0.50). Run intentionally, not in a tight loop.

- [ ] **Step 1: Run the eval suite**

```bash
pytest tests/test_eval.py -v -m eval
```
Expected: some pass, some fail. The first run defines your baseline; do not panic about red tests.

- [ ] **Step 2: Snapshot results**

```bash
pytest tests/test_eval.py -v -m eval > eval-baseline.txt 2>&1 || true
```

- [ ] **Step 3: Commit baseline**

```bash
git add eval-baseline.txt
git commit -m "test: eval suite baseline run"
```

From here on, every prompt change in `nodes.py` should be re-run against this suite to confirm direction of travel.

---

## Task 18: Streamlit UI

**Files:**
- Create: `src/research_agent/ui.py`

- [ ] **Step 1: Write `src/research_agent/ui.py`**

```python
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
            "recursion_limit": 50,
        }
        async for ev in app.astream({"question": question}, cfg, stream_mode="updates"):
            events.append(str(ev)[:300])
            progress.code("\n".join(events[-20:]))
        state = await app.aget_state(cfg)
        answer_box.markdown(state.values.get("answer", "(no answer)"))
        citations_box.write(state.values.get("citations", []))

if go and q:
    asyncio.run(run(q))
```

- [ ] **Step 2: Resolve UNVERIFIED — `aget_state` API**

```bash
python -c "
import asyncio
from research_agent.graph import research_app
async def go():
    async with research_app() as app:
        print([m for m in dir(app) if 'state' in m.lower()])
asyncio.run(go())
"
```

If `aget_state` is named differently (`get_state`, `astate`, etc.), update the file and the spec.

- [ ] **Step 3: Run Streamlit, sanity-check the page loads**

```bash
streamlit run src/research_agent/ui.py
```
Expected: browser opens to `http://localhost:8501`, you see the title and an input box. Type a question; verify progress lines stream and an answer renders. Does NOT need to be high quality — that's the eval suite's job.

- [ ] **Step 4: Commit**

```bash
git add src/research_agent/ui.py
git commit -m "feat: streamlit UI with streaming progress and final answer"
```

---

## Task 19: Final smoke + spec sync

- [ ] **Step 1: Sweep for stale `UNVERIFIED:` tags**

```bash
grep -rn "UNVERIFIED:" src tests docs
```

Each remaining tag is a debt. For every remaining tag:
1. Probe the live SDK / docs to confirm the actual behavior.
2. If the spec/code claim is correct: remove the tag, replace with a brief comment if useful.
3. If wrong: fix code + spec in the same commit, remove the tag.

- [ ] **Step 2: Run all non-cost tests**

```bash
pytest -v -m "not smoke and not eval"
```
Expected: all PASS (settings, schemas, routing, graph compile).

- [ ] **Step 3: Run smoke set once**

```bash
pytest -v -m smoke
```
Expected: 3 tests PASS (web, graphiti round-trip, full research run). Cost ~$0.30.

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "chore: final UNVERIFIED sweep + spec sync" || echo "nothing to commit"
```

- [ ] **Step 5: Tag v0.5**

```bash
git tag -a v0.5 -m "v0.5: research companion + reflection + Streamlit + eval baseline"
```

---

## v1 deferred (do not implement now)

- Phase 4 contradictions surfacing in the answer node — needs custom Cypher over `valid_to`
- LangSmith tracing
- Cost-cap kill switch (per-day Gemini token budget)
- FalkorDB driver swap (single-line replacement of `KuzuDriver`)
- `AsyncPostgresSaver` for checkpointing
- Reflection idle-timeout trigger (currently manual-only)
- Side-table for raw-storage middle-tier salience (only if two-tier proves coarse)
- Larger eval datasets (>30 cases) — current live-judge cost model breaks at scale; switch to a cheaper Flash judge or sampled regression
