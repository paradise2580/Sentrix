"""
Baseline models: Logistic Regression and Random Forest, both with
class_weight="balanced". They set the bar the boosted models must beat.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.config_loader import load_config
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def build_logistic_regression() -> LogisticRegression:
    """Build a Logistic Regression from config.yaml."""
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
    """Build a Random Forest from config.yaml."""
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
