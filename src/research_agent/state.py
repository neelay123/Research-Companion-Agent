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
