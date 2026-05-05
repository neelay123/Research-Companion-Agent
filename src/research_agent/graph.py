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
