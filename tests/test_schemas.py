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
