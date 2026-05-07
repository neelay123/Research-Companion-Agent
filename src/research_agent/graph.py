from contextlib import asynccontextmanager
from langgraph.types import Send
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from .state import ResearchState
from .nodes import plan_node, search_node, ingest_one, retrieve_node, answer_node

CHECKPOINT_DSN = "checkpoints.sqlite"

def fan_out_ingest(state: ResearchState) -> list[Send]:
    return [Send("ingest_one", {"url": u, "question": state["question"]})
            for u in state["candidate_urls"]]

def should_search_more(state: ResearchState) -> str:
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
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DSN) as ckpt:
        yield _build_research().compile(checkpointer=ckpt)
