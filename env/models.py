from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class Observation(BaseModel):
    task_id: str
    paper_id: str
    title: str
    abstract: str
    methodology: str
    results: str
    claims: List[str]
    step: int
    max_steps: int
    metadata: dict = Field(default_factory=dict)


class Action(BaseModel):
    decision: Literal["accept", "reject", "revise"]
    identified_flaws: List[str] = Field(default_factory=list)
    justification: str
    confidence: float = Field(ge=0.0, le=1.0)
    requested_changes: Optional[List[str]] = None


class Reward(BaseModel):
    total: float
    decision_score: float
    flaw_detection_score: float
    justification_score: float
    efficiency_bonus: float
    breakdown: dict = Field(default_factory=dict)
