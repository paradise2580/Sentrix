"""
src/evaluation/calibration.py

Role
----
Turn a model's raw score into a probability you can actually put a number
behind.

Why this exists
---------------
Every ranking metric in this project — PR-AUC, ROC-AUC, KS, capture@k — is
invariant to any monotonic transformation of the score. A model can rank
sellers perfectly and still claim 0.70 for a group that goes late 25% of
the time. That is fine if the score is only ever used to sort a worklist,
and it is a real problem the moment anyone multiplies the score by a cost,
sets an SLA against it, or shows it to a seller as "your risk is 70%".

Three of the six models here are actively miscalibrated by construction:
LightGBM is trained on SMOTE-resampled data (which inflates the apparent
positive rate), and Logistic Regression / Random Forest use
class_weight="balanced" (which does the same). Their rankings are
meaningful; their probabilities are not.

Approach
--------
Isotonic regression fitted on the CALIBRATION block — a slice of time that
sits after training and before test, separated from both by an embargo.
The model never trained on it and the test set never touches it, so the
test-set calibration numbers reported after this are honest.

Isotonic rather than Platt/sigmoid because it is non-parametric: it can
correct an arbitrary monotone distortion, and there are enough
calibration rows here to support it without overfitting (Platt is the
better choice below roughly a thousand samples).
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import (
    expected_calibration_error, calibration_curve_points,
)
from sklearn.metrics import brier_score_loss

from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def fit_calibrator(y_calib: np.ndarray, p_calib: np.ndarray, method: str = "isotonic"):
    """
    Fit a post-hoc calibration map from raw scores to calibrated
    probabilities, using held-out calibration data.

    Operates on SCORES, not on the model — so the same function calibrates
    the LSTM and the tree models identically, which sklearn's
    CalibratedClassifierCV cannot do (it requires an sklearn estimator).
    """
    try:
        y_calib = np.asarray(y_calib)
        p_calib = np.asarray(p_calib)

        if len(y_calib) < 200 or len(np.unique(y_calib)) < 2:
            raise ValueError(
                f"Calibration block too small or single-class "
                f"(n={len(y_calib)}, classes={np.unique(y_calib)})"
            )

        if method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            cal.fit(p_calib, y_calib)
        elif method == "sigmoid":
            cal = LogisticRegression()
            cal.fit(p_calib.reshape(-1, 1), y_calib)
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        logger.info(f"Calibrator fitted ({method}) on {len(y_calib):,} held-out rows")
        return {"method": method, "model": cal}
    except Exception as e:
        raise SentrixException(e, sys)


def apply_calibrator(calibrator: dict, p: np.ndarray) -> np.ndarray:
    """Map raw scores through a fitted calibrator."""
    try:
        p = np.asarray(p)
        if calibrator["method"] == "isotonic":
            return np.clip(calibrator["model"].predict(p), 0.0, 1.0)
        return calibrator["model"].predict_proba(p.reshape(-1, 1))[:, 1]
    except Exception as e:
        raise SentrixException(e, sys)


def calibration_report(y_true: np.ndarray, p_raw: np.ndarray,
                       p_cal: np.ndarray) -> dict:
    """
    Before/after summary for the README and the dashboard.

    Brier score mixes calibration and discrimination; ECE isolates
    calibration. Both are reported because a drop in ECE with an unchanged
    ranking metric is the clean evidence that calibration — and only
    calibration — is what changed.
    """
    try:
        return {
            "brier_raw": float(brier_score_loss(y_true, p_raw)),
            "brier_calibrated": float(brier_score_loss(y_true, p_cal)),
            "ece_raw": expected_calibration_error(y_true, p_raw),
            "ece_calibrated": expected_calibration_error(y_true, p_cal),
            "observed_base_rate": float(np.mean(y_true)),
            "mean_prediction_raw": float(np.mean(p_raw)),
            "mean_prediction_calibrated": float(np.mean(p_cal)),
            "reliability_raw": calibration_curve_points(y_true, p_raw).to_dict("records"),
            "reliability_calibrated": calibration_curve_points(y_true, p_cal).to_dict("records"),
        }
    except Exception as e:
        raise SentrixException(e, sys)
