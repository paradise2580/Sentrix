"""
src/models/baseline.py

Role
----
Establish the performance floor before reaching for anything advanced.
You cannot claim XGBoost is genuinely better without a simpler model to
beat it against — that comparison is the whole point of a baseline.

Models
------
Logistic Regression  — the regulatory-friendly baseline. Simple, fast,
                        coefficients are directly interpretable (credit
                        risk teams often require a model like this to
                        exist even if it isn't the one deployed).
Random Forest         — a stronger tree baseline that captures
                        non-linear interactions the linear model misses.

Class imbalance (confirmed in Phase 3 EDA: ~22% positive) is handled via
class_weight="balanced" in both models — the config-driven default rather
than a hardcoded assumption.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config_loader import load_config
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def build_logistic_regression() -> LogisticRegression:
    """Construct a Logistic Regression classifier using config.yaml hyperparameters."""
    try:
        cfg = load_config()["model"]["logistic_regression"]
        model = LogisticRegression(
            max_iter=cfg["max_iter"],
            class_weight=cfg["class_weight"],
        )
        logger.info(f"Logistic Regression built with params: {cfg}")
        return model
    except Exception as e:
        raise SentrixException(e, sys)


def build_random_forest() -> RandomForestClassifier:
    """Construct a Random Forest classifier using config.yaml hyperparameters."""
    try:
        cfg = load_config()["model"]["random_forest"]
        model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            class_weight=cfg["class_weight"],
            random_state=load_config()["project"]["random_state"],
            n_jobs=-1,
        )
        logger.info(f"Random Forest built with params: {cfg}")
        return model
    except Exception as e:
        raise SentrixException(e, sys)
