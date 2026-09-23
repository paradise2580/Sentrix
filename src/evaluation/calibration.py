"""
Maps raw model scores to calibrated probabilities with isotonic regression.

Ranking metrics ignore calibration, and SMOTE / class_weight="balanced"
inflate raw scores. The calibrator is fitted on the CALIB block, which the
model never trained on and the test set never touches.
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
    Fit an isotonic map from raw scores to probabilities on held-out data.
    Works on scores rather than a model, so any model can be calibrated.
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
    """Before/after Brier score and ECE for the README and dashboard."""
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
