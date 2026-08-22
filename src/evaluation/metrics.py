"""
src/evaluation/metrics.py

Role
----
Score every model on the same held-out test set with the metrics that
actually matter for an imbalanced risk-prediction problem. Phase 3's EDA
showed disruptions are ~22% of supplier-days — real but moderate
imbalance — which is why PR-AUC (not accuracy) is the primary metric
used to rank models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, precision_recall_curve,
)

from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic — the maximum separation between the
    cumulative distribution of predicted probabilities for the positive
    vs negative class. Standard in credit/risk modeling (the metric a
    credit risk team would ask for by name).
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    df["cum_pos"] = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    df["cum_neg"] = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float(np.max(np.abs(df["cum_pos"] - df["cum_neg"])))


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute the full metric suite for one model's predictions on the test set.
    y_prob is the predicted PROBABILITY of the positive class (disruption).
    """
    try:
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),   # the primary metric
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "ks_statistic": ks_statistic(y_true, y_prob),
            "threshold_used": threshold,
        }

        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()
        metrics["true_negatives"], metrics["false_positives"], \
            metrics["false_negatives"], metrics["true_positives"] = cm.ravel()

        return metrics
    except Exception as e:
        raise SentrixException(e, sys)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Sweep thresholds and pick the one maximizing F1 — a defensible, simple
    default that balances precision and recall. In production this is a
    business decision (how many false alarms is acceptable vs missed
    disruptions), but F1-optimal is a sound, explainable starting point.
    """
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])  # last point has no corresponding threshold

        return {
            "optimal_threshold": float(thresholds[best_idx]),
            "precision_at_optimal": float(precisions[best_idx]),
            "recall_at_optimal": float(recalls[best_idx]),
            "f1_at_optimal": float(f1_scores[best_idx]),
        }
    except Exception as e:
        raise SentrixException(e, sys)


def compare_models(results: dict) -> pd.DataFrame:
    """
    results: {model_name: {"y_true": ..., "y_prob": ...}}
    Returns a comparison table sorted by PR-AUC descending — the ranking
    used to select the model promoted to MLflow's Production stage.
    """
    try:
        rows = []
        for name, r in results.items():
            m = evaluate_model(r["y_true"], r["y_prob"])
            rows.append({
                "model": name,
                "roc_auc": round(m["roc_auc"], 4),
                "pr_auc": round(m["pr_auc"], 4),
                "f1": round(m["f1"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "ks_statistic": round(m["ks_statistic"], 4),
            })

        comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
        logger.info(f"Model comparison:\n{comparison.to_string()}")
        return comparison
    except Exception as e:
        raise SentrixException(e, sys)


def assign_risk_band(prob: float, bands: dict) -> str:
    """Convert a raw probability into the Low/Medium/High/Critical band the dashboard shows."""
    if prob >= bands["high"]:
        return "critical"
    elif prob >= bands["medium"]:
        return "high"
    elif prob >= bands["low"]:
        return "medium"
    return "low"
