from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Finding(BaseModel):
    disease: str
    probability: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    detected: bool


class PredictionResponse(BaseModel):
    findings: List[Finding] = Field(description="All classes, sorted by probability")
    detected: List[str] = Field(description="Classes whose probability >= their threshold")
    review_priority: str = Field(description="High / Medium / Low")
    explained_class: Optional[str] = None
    heatmap_base64: Optional[str] = None
    model_file: str
    follow_up: Optional[Dict[str, Any]] = Field(
        None, description="Predefined specialty suggestion (only with llm_explain=true)")
    explanation: Optional[Dict[str, Any]] = Field(
        None, description="Groq explanation result (only with llm_explain=true)")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_file: Optional[str] = None
