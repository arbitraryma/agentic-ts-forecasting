"""
Structured outputs for the agent workflow and the API.

These models describe results. They do not compute them.
"""

from typing import Literal

from pydantic import BaseModel, Field


class PlanResult(BaseModel):
    task: str
    horizon: int = Field(gt=0)
    steps: list[str]
    reason: str


class SelectedExpert(BaseModel):
    model: str
    window_type: str
    validation_mae: float
    utility: float | None = None
    runtime: float | None = None


class ForecastPayload(BaseModel):
    diagnostics: dict
    candidates: list[str]
    validation_summary: list[dict]
    selected_expert: SelectedExpert
    forecast: list[float]
    evidence: str


class CriticResult(BaseModel):
    decision: Literal["ACCEPT", "RETRY"]
    issues: list[str] = Field(default_factory=list)
    reason: str = ""
    recommended_action: str | None = None


class ForecastResponse(BaseModel):
    selected_model: str
    window_type: str
    validation_mae: float
    forecast: list[float]
    critic_decision: str
    explanation: str
