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
