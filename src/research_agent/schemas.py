"""Pydantic models for the research agent.

Two purposes — keep schemas flat (Gemini response_schema rejects deeply nested or
heavily-constrained schemas with InvalidArgument: 400):

1. Structured output for Gemini calls — Plan, SalienceVerdict, Answer, Synthesis.
2. Graphiti entity types (consumed by Graphiti's extractor, NOT by Gemini's
   response_schema) — Paper, Author, Claim. Optional/None fields here are safe;
   Graphiti owns the prompt translation.
"""
from pydantic import BaseModel, Field

# --- structured-output for Gemini ---

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

# --- Graphiti entity types (Phase 2) ---

class Paper(BaseModel):
    title: str
    arxiv_id: str | None = None
    year: int | None = None

class Author(BaseModel):
    name: str
    affiliation: str | None = None

class Claim(BaseModel):
    statement: str
    # Note: dropped `confidence` field. Gemini structured output is unreliable on
    # Optional/union-with-None primitives — either schema rejection or always-null.
    # Graphiti's extractor surfaces confidence-equivalent signals via edge metadata.
