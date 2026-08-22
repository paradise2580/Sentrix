"""
api/schemas.py

Role
----
Pydantic models defining every request and response shape the API
accepts and returns. FastAPI uses these to validate incoming requests
automatically — malformed input is rejected with a clear 422 error
before it ever reaches model or database code.
"""

from pydantic import BaseModel, Field
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    model_name: str | None = None
    model_stage: str | None = None


class SellerRiskResponse(BaseModel):
    seller_id: str          # real Olist seller IDs are hash strings
    seller_city: str | None = None
    seller_state: str | None = None
    risk_score: float
    risk_band: str
    model_name: str
    predicted_at: datetime | None = None


class FeatureContribution(BaseModel):
    feature: str
    contribution: float


class ExplanationResponse(BaseModel):
    seller_id: str
    risk_score: float
    risk_band: str
    top_features: list[FeatureContribution]


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                            description="A natural-language question about seller delivery risk.")
    top_k: int | None = Field(None, ge=1, le=10)


class ChatSource(BaseModel):
    text: str
    metadata: dict
    distance: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]
    grounded: bool
    llm_used: bool = False


class ModelMetric(BaseModel):
    model: str
    roc_auc: float
    pr_auc: float
    f1: float
    precision: float
    recall: float
    ks_statistic: float


class ModelMetricsResponse(BaseModel):
    best_model: str
    comparison: list[ModelMetric]


class StateRisk(BaseModel):
    seller_state: str
    seller_count: int
    avg_risk: float
    critical_count: int


class SummaryResponse(BaseModel):
    """Portfolio-level KPIs — lets the dashboard render its landing view
    with one small call instead of pulling every seller row."""
    total_sellers: int
    band_counts: dict[str, int]
    avg_risk: float
    best_model: str | None = None
    best_pr_auc: float | None = None
    risk_histogram: list[int]          # 20 bins, 0.0-1.0
    by_state: list[StateRisk]
