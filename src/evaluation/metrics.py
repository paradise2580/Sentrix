"""
src/evaluation/metrics.py

Role
----
Score every model on the same held-out test set with the metrics that
actually matter for an imbalanced risk-ranking problem.

Why PR-AUC and not accuracy
---------------------------
The positive class (a seller who will be late at least once in the next 30
days) is the minority. A model that predicts "never late" for everyone
scores high accuracy and is worthless. PR-AUC measures performance on the
class we care about, and is the metric used to rank models.

Why the operating metrics matter more than PR-AUC
-------------------------------------------------
PR-AUC is a single number that summarises the whole curve. Nobody runs a
whole curve. An operations team can investigate a fixed number of sellers
per week, so what they need to know is: "if we work the top K% of the risk
list, what fraction of the sellers who actually go late do we catch, and
how much better is that than picking K% at random?" That is capture@k and
lift@k, computed below and reported alongside PR-AUC.
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

# Alert-capacity levels reported for every model. These are the shares of the
# seller list an operations team could realistically review in a cycle.
DEFAULT_K_VALUES = (0.05, 0.10, 0.20)


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


def capture_at_k(y_true: np.ndarray, y_prob: np.ndarray, k: float) -> dict:
    """
    Rank every seller-day by predicted risk, take the top k fraction, and
    report what that slice actually contains.

    Returns
    -------
    dict with:
      capture   share of ALL true positives that fall in the top-k slice
                ("we reviewed 10% of the list and caught 34% of the
                sellers who went late")
      precision share of the top-k slice that are true positives
                (the hit rate an analyst experiences)
      lift      capture / k — how many times better than reviewing a
                random k fraction. lift = 1.0 means the model is useless.
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
    Reliability data: bucket predictions by decile of predicted probability
    and compare the mean prediction in each bucket to the observed rate.

    A well-calibrated risk score means "0.30" actually happens ~30% of the
    time. Ranking metrics (PR-AUC, KS) are completely blind to this — a
    model can rank perfectly and still be systematically overconfident,
    which matters the moment anyone attaches a rupee value or an SLA to
    the score.
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
    """
    Weighted mean gap between predicted and observed rates across bins.
    0 is perfect; report alongside Brier score, which mixes calibration
    and discrimination into one number and so can't isolate either.
    """
    pts = calibration_curve_points(y_true, y_prob, n_bins)
    weights = pts["n"] / pts["n"].sum()
    return float((weights * (pts["mean_predicted"] - pts["observed_rate"]).abs()).sum())


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5,
                   k_values: tuple = DEFAULT_K_VALUES) -> dict:
    """
    Compute the full metric suite for one model's predictions on the test set.
    y_prob is the predicted PROBABILITY of the positive class (disruption).
    """
    try:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        # Ranking metrics work on ANY score — they only read the ordering.
        # Calibration metrics do not: Brier and ECE compare a score to an
        # observed rate, which is meaningless unless the score IS a
        # probability. This function is also used to rank raw features
        # directly (see scripts/ablation.py), so it reports NaN for the
        # calibration metrics rather than refusing to score at all.
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

        # PR-AUC is not comparable across datasets with different base rates;
        # the normalised version (PR-AUC / base rate) is the honest "how many
        # times better than random" statement.
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
    Sweep thresholds and pick the one maximizing F1 — a defensible, simple
    default that balances precision and recall.

    Note this is deliberately NOT how the product actually operates. F1
    weights a false alarm and a missed late delivery equally, which no
    business does. See find_cost_optimal_threshold for the version that
    takes the real asymmetry as an input.
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
    Choose the threshold that minimises total expected cost rather than
    maximising a symmetric score.

    cost_fn : cost of MISSING a seller who then delivers late (lost
              customer trust, refund, support contact)
    cost_fp : cost of a FALSE ALARM (an analyst spends time on a seller
              who was fine)

    The 10:1 default encodes the assumption that a missed late delivery is
    an order of magnitude more expensive than a wasted review. The ratio is
    the only number a business stakeholder actually has to supply, and every
    downstream operating point follows from it — which is the point of
    exposing it as an argument instead of hardcoding 0.5.
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

        # What the same decisions would have cost with no model at all:
        # alerting on nobody (every positive is a miss).
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

    Returns a comparison table sorted by PR-AUC descending.

    IMPORTANT: every entry must have been scored on the IDENTICAL set of
    rows. PR-AUC depends on the base rate, so two models evaluated on
    different subsets cannot be ranked against each other. run_evaluation
    enforces this by scoring every model on one shared evaluation index;
    this function asserts it rather than trusting the caller.
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
    """
    Convert a probability into a band using ABSOLUTE cutoffs.
    Retained for callers that supply their own thresholds; the pipeline
    itself uses assign_risk_bands_by_quantile.
    """
    if prob >= bands["high"]:
        return "critical"
    elif prob >= bands["medium"]:
        return "high"
    elif prob >= bands["low"]:
        return "medium"
    return "low"


def assign_risk_bands_by_quantile(probs: np.ndarray, quantiles: dict) -> np.ndarray:
    """
    Band the scored population by rank rather than by absolute probability.

    With a base rate near 17%, a *correctly calibrated* model rarely emits a
    probability above 0.75 — so fixed cutoffs of 0.25/0.50/0.75 leave the top
    bands permanently empty and make a well-calibrated model look worse than
    an overconfident one. Ranking sidesteps that entirely: "critical" means
    the worst 5% of sellers scored right now, which is both what an ops team
    asks for and what capture@5% measures.

    quantiles: {"critical": 0.95, "high": 0.80, "medium": 0.50}
    """
    probs = np.asarray(probs, dtype=float)
    cuts = {name: float(np.quantile(probs, q)) for name, q in quantiles.items()}

    bands = np.full(len(probs), "low", dtype=object)
    bands[probs >= cuts["medium"]] = "medium"
    bands[probs >= cuts["high"]] = "high"
    bands[probs >= cuts["critical"]] = "critical"
    return bands
