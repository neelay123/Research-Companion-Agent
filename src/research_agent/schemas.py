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
