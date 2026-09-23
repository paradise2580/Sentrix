"""
Evaluation metrics for an imbalanced risk-ranking problem.

Models are ranked by PR-AUC, since accuracy is meaningless when only ~5% of
orders are late. capture@k and lift@k report what a team reviewing the top
k% of the list would actually catch.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, brier_score_loss,
    precision_score, recall_score, confusion_matrix, precision_recall_curve,
)

from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

# Review capacities reported for every model (share of the list reviewed).
DEFAULT_K_VALUES = (0.05, 0.10, 0.20)


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Max gap between the score CDFs of positives and negatives."""
    df = pd.DataFrame({"y": y_true, "p": y_prob}).sort_values("p")
    df["cum_pos"] = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    df["cum_neg"] = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return float(np.max(np.abs(df["cum_pos"] - df["cum_neg"])))


def capture_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float) -> dict:
    """
    Take the top k fraction by predicted risk and report:
      capture   share of all positives inside the top k
      precision share of the top k that are positive
      lift      capture / k (1.0 = no better than random)
    """
    n = len(y_true)
    n_k = max(int(round(n * k)), 1)
    order = np.argsort(-y_prob, kind="stable")
    top = y_true[order][:n_k]

    total_pos = max(int(y_true.sum()), 1)
    capture = float(top.sum()) / total_pos
    return {
        "k": k,
        "n_reviewed": n_k,
        "capture": capture,
        "precision": float(top.mean()),
        "lift": capture / k if k > 0 else float("nan"),
    }


def calibration_curve_points(y_true: np.ndarray, y_prob: np.ndarray,
                             n_bins: int = 10) -> pd.DataFrame:
    """
    Bucket predictions into bins and compare mean predicted vs observed rate
    per bin. Ranking metrics can't see calibration, so it is checked here.
    """
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    out = (
        df.groupby("bin")
          .agg(mean_predicted=("p", "mean"),
               observed_rate=("y", "mean"),
               n=("y", "size"))
          .reset_index(drop=True)
    )
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray,
                               n_bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rates. 0 is perfect."""
    pts = calibration_curve_points(y_true, y_prob, n_bins)
    weights = pts["n"] / pts["n"].sum()
    return float((weights * (pts["mean_predicted"] - pts["observed_rate"]).abs()).sum())


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
                   k_values: tuple = DEFAULT_K_VALUES) -> dict:
    """Full metric suite for one model. y_prob is the score for the positive class."""
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        # Brier/ECE only make sense for probabilities. scripts/ablation.py also
        # passes raw feature values here, so those get NaN instead of an error.
        is_probability = bool(y_prob.min() >= 0.0 and y_prob.max() <= 1.0)

        metrics = {
            "roc_auc": roc_auc_score(y_true, y_prob),
            "pr_auc": average_precision_score(y_true, y_prob),   # primary ranking metric
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "ks_statistic": ks_statistic(y_true, y_prob),
            "brier": float(brier_score_loss(y_true, y_prob)) if is_probability else float("nan"),
            "ece": expected_calibration_error(y_true, y_prob) if is_probability else float("nan"),
            "is_probability": is_probability,
            "base_rate": float(y_true.mean()),
            "threshold_used": threshold,
            "n_eval": int(len(y_true)),
        }

        # PR-AUC / base rate = "how many times better than random".
        metrics["pr_auc_lift"] = metrics["pr_auc"] / max(metrics["base_rate"], 1e-9)

        for k in k_values:
            at_k = capture_at_k(y_true, y_prob, k)
            pct = int(round(k * 100))
            metrics[f"capture_at_{pct}pct"] = at_k["capture"]
            metrics[f"precision_at_{pct}pct"] = at_k["precision"]
            metrics[f"lift_at_{pct}pct"] = at_k["lift"]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics["confusion_matrix"] = cm.tolist()
        metrics["true_negatives"], metrics["false_positives"], \
            metrics["false_negatives"], metrics["true_positives"] = cm.ravel()

        return metrics
    except Exception as e:
        raise SentrixException(e, sys)


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Threshold that maximises F1. Reported for reference only; F1 treats both
    errors as equally bad, so the product uses find_cost_optimal_threshold.
    """
    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_idx = int(np.argmax(f1_scores[:-1]))  # last point has no threshold

        return {
            "optimal_threshold": float(thresholds[best_idx]),
            "precision_at_optimal": float(precisions[best_idx]),
            "recall_at_optimal": float(recalls[best_idx]),
            "f1_at_optimal": float(f1_scores[best_idx]),
        }
    except Exception as e:
        raise SentrixException(e, sys)


def find_cost_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                                cost_fn: float = 10.0, cost_fp: float = 1.0) -> dict:
    """
    Threshold that minimises total cost, where cost_fn is the cost of a missed
    late order and cost_fp the cost of a false alarm (default 10:1).
    """
    try:
        y_true = np.asarray(y_true)
        candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))

        rows = []
        for t in candidates:
            y_pred = (y_prob >= t).astype(int)
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            rows.append({"threshold": float(t), "fn": fn, "fp": fp,
                         "total_cost": fn * cost_fn + fp * cost_fp})

        curve = pd.DataFrame(rows)
        best = curve.loc[curve["total_cost"].idxmin()]

        # Cost with no model: alert on nobody, so every positive is a miss.
        baseline_cost = float((y_true == 1).sum() * cost_fn)

        return {
            "cost_optimal_threshold": float(best["threshold"]),
            "total_cost": float(best["total_cost"]),
            "baseline_cost_no_model": baseline_cost,
            "cost_reduction": 1.0 - float(best["total_cost"]) / max(baseline_cost, 1e-9),
            "cost_fn": cost_fn,
            "cost_fp": cost_fp,
            "curve": curve,
        }
    except Exception as e:
        raise SentrixException(e, sys)


def compare_models(results: dict) -> pd.DataFrame:
    """
    results: {model_name: {"y_true": ..., "y_prob": ...}}
    Returns a table sorted by PR-AUC. All models must be scored on the same
    rows, because PR-AUC depends on the base rate.
    """
    try:
        lengths = {name: len(r["y_true"]) for name, r in results.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(
                "Models were scored on different-sized evaluation sets and "
                f"cannot be ranked together: {lengths}"
            )

        rows = []
        for name, r in results.items():
            m = evaluate_model(r["y_true"], r["y_prob"])
            rows.append({
                "model": name,
                "pr_auc": round(m["pr_auc"], 4),
                "pr_auc_lift": round(m["pr_auc_lift"], 2),
                "roc_auc": round(m["roc_auc"], 4),
                "ks_statistic": round(m["ks_statistic"], 4),
                "capture_at_10pct": round(m["capture_at_10pct"], 4),
                "lift_at_10pct": round(m["lift_at_10pct"], 2),
                "capture_at_20pct": round(m["capture_at_20pct"], 4),
                "brier": round(m["brier"], 4),
                "ece": round(m["ece"], 4),
                "f1": round(m["f1"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
            })

        comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
        logger.info(f"Model comparison:\n{comparison.to_string()}")
        return comparison
    except Exception as e:
        raise SentrixException(e, sys)


def assign_risk_band(prob: float, bands: dict) -> str:
    """Band a probability with fixed cutoffs. The pipeline uses the quantile version."""
    if prob >= bands["high"]:
        return "critical"
    elif prob >= bands["medium"]:
        return "high"
    elif prob >= bands["low"]:
        return "medium"
    return "low"


def assign_risk_bands_by_quantile(probs: np.ndarray, quantiles: dict) -> np.ndarray:
    """
    Band by rank instead of fixed probability cutoffs. With a ~5% base rate a
    calibrated model rarely outputs high probabilities, so fixed cutoffs would
    leave the top bands empty.

    quantiles: {"critical": 0.95, "high": 0.80, "medium": 0.50}
    """
    probs = np.asarray(probs, dtype=float)
    cuts = {name: float(np.quantile(probs, q)) for name, q in quantiles.items()}

    bands = np.full(len(probs), "low", dtype=object)
    bands[probs >= cuts["medium"]] = "medium"
    bands[probs >= cuts["high"]] = "high"
    bands[probs >= cuts["critical"]] = "critical"
    return bands
