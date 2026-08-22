"""
api/app.py

Role
----
Thin routing layer exposing SENTRIX's intelligence over HTTP. Every
endpoint CALLS a module built in earlier phases — it never implements
model, database, or RAG logic itself. This separation is what lets the
Streamlit dashboard (Phase 9), or any other client, consume the same
predictions without duplicating logic.

Endpoints
---------
GET  /health              — liveness check, reports which model is live
GET  /sellers              — all sellers with their current risk score
GET  /explain/{seller_id}  — SHAP explanation for one seller
POST /chat                 — RAG-grounded natural-language Q&A
GET  /metrics               — the Phase 5 model comparison table

Run with:
    uvicorn api.app:app --reload --port 8000
"""

import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config_loader import load_config, get_project_root
from src.ingestion.loader import DataLoader
from src.rag.chain import answer_question
from src.logger import get_logger
from src.exception import SentrixException

from api.schemas import (
    HealthResponse, SellerRiskResponse, ExplanationResponse, FeatureContribution,
    ChatRequest, ChatResponse, ModelMetricsResponse, ModelMetric,
    SummaryResponse, StateRisk,
)

logger = get_logger(__name__)
cfg = load_config()

app = FastAPI(
    title=cfg["api"]["title"],
    description="Supply Chain Disruption Intelligence Platform — prediction, explanation, and RAG chat API.",
    version=cfg["project"]["version"],
)

# Permissive CORS for local development — the Streamlit dashboard and any
# other client run on a different port and need to call this API directly.
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness check. Also reports which model MLflow currently has in Production."""
    try:
        import mlflow
        tracking_dir = get_project_root() / cfg["paths"]["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")
        client = mlflow.MlflowClient()
        versions = client.search_model_versions("name='sentrix-risk-model'")
        prod = next((v for v in versions if v.current_stage == "Production"), None)

        return HealthResponse(
            status="ok",
            model_name=prod.run_id if prod else None,
            model_stage="Production" if prod else "none registered",
        )
    except Exception:
        return HealthResponse(status="ok", model_name=None, model_stage="mlflow unavailable")


@app.get("/sellers", response_model=list[SellerRiskResponse])
def get_sellers(risk_band: str | None = None, limit: int = 500):
    """
    All sellers with their current risk score, as stored by Phase 5's
    generate_predictions.py. Optionally filter by risk_band
    (low / medium / high / critical).
    """
    try:
        loader = DataLoader()
        predictions = loader.read_table("predictions")
        sellers = loader.read_table("sellers")
        merged = predictions.merge(sellers, on="seller_id", how="left")

        if risk_band:
            merged = merged[merged["risk_band"] == risk_band]

        merged = merged.sort_values("risk_score", ascending=False).head(limit)

        return [
            SellerRiskResponse(
                seller_id=str(row["seller_id"]),
                seller_city=row.get("seller_city"),
                seller_state=row.get("seller_state"),
                risk_score=float(row["risk_score"]),
                risk_band=row["risk_band"],
                model_name=row["model_name"],
                predicted_at=row.get("predicted_at"),
            )
            for _, row in merged.iterrows()
        ]
    except SentrixException as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.get("/explain/{seller_id}", response_model=ExplanationResponse)
def explain_seller(seller_id: str):
    """SHAP explanation for one seller's current risk score."""
    try:
        loader = DataLoader()
        predictions = loader.read_table("predictions")
        row = predictions[predictions["seller_id"] == seller_id]

        if row.empty:
            raise HTTPException(status_code=404, detail=f"No prediction found for seller {seller_id}")

        row = row.iloc[0]
        top_features = json.loads(row["top_features"]) if row["top_features"] else []

        return ExplanationResponse(
            seller_id=seller_id,
            risk_score=float(row["risk_score"]),
            risk_band=row["risk_band"],
            top_features=[FeatureContribution(**f) for f in top_features],
        )
    except SentrixException as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """RAG-grounded natural-language question answering over the seller knowledge base."""
    try:
        result = answer_question(request.question, top_k=request.top_k)
        return ChatResponse(**result)
    except SentrixException as e:
        raise HTTPException(status_code=503, detail=f"RAG pipeline unavailable: {e}")


@app.get("/metrics", response_model=ModelMetricsResponse)
def get_metrics():
    """The Phase 5 model comparison table — every model's evaluation metrics."""
    try:
        eval_dir = get_project_root() / "artifacts" / "evaluation"
        comparison_path = eval_dir / "model_comparison.csv"
        summary_path = eval_dir / "best_model_summary.joblib"

        if not comparison_path.exists():
            raise HTTPException(status_code=404, detail="No evaluation results found. Run Phase 5 first.")

        comparison = pd.read_csv(comparison_path)
        import joblib
        best_summary = joblib.load(summary_path)

        return ModelMetricsResponse(
            best_model=best_summary["best_model"],
            comparison=[ModelMetric(**row) for row in comparison.to_dict(orient="records")],
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Evaluation artifacts not found.")




@app.get("/summary", response_model=SummaryResponse)
def get_summary():
    """
    Portfolio-level KPIs for the dashboard landing view: band counts, average
    risk, a 20-bin risk histogram, and per-state aggregates. One small call
    instead of the client pulling every seller row and aggregating locally.
    """
    try:
        import numpy as np
        loader = DataLoader()
        preds = loader.read_table("predictions")
        sellers = loader.read_table("sellers")
        df = preds.merge(sellers, on="seller_id", how="left")

        counts, _ = np.histogram(df["risk_score"], bins=20, range=(0.0, 1.0))

        by_state = (
            df.groupby("seller_state")
              .agg(seller_count=("seller_id", "count"),
                   avg_risk=("risk_score", "mean"),
                   critical_count=("risk_band", lambda s: int((s == "critical").sum())))
              .reset_index()
              .sort_values("avg_risk", ascending=False)
        )

        best_model, best_pr_auc = None, None
        try:
            import joblib
            summary = joblib.load(get_project_root() / "artifacts" / "evaluation" / "best_model_summary.joblib")
            best_model = summary["best_model"]
            best_pr_auc = float(summary["metrics"]["pr_auc"])
        except Exception:
            pass

        return SummaryResponse(
            total_sellers=len(df),
            band_counts={k: int(v) for k, v in df["risk_band"].value_counts().items()},
            avg_risk=float(df["risk_score"].mean()),
            best_model=best_model,
            best_pr_auc=best_pr_auc,
            risk_histogram=[int(c) for c in counts],
            by_state=[StateRisk(**r) for r in by_state.to_dict("records")],
        )
    except SentrixException as e:
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.get("/")
def root():
    return {
        "service": cfg["api"]["title"],
        "docs": "/docs",
        "endpoints": ["/health", "/summary", "/sellers", "/explain/{seller_id}", "/chat", "/metrics"],
    }
