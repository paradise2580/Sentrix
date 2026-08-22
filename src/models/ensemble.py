"""
src/models/ensemble.py

Role
----
A stacking meta-learner that combines the base models' predictions.
Often the single best performer because it blends the strengths of
linear, tree, and boosted models rather than betting on just one.

The LSTM is intentionally excluded from the sklearn StackingClassifier
(it isn't sklearn-compatible) — its predictions are blended in separately
by the trainer, since it operates on sequences rather than flat rows.
"""

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def build_ensemble(base_models: list[tuple[str, object]]) -> StackingClassifier:
    """
    base_models: list of (name, fitted_or_unfitted_estimator) tuples, e.g.
        [("logreg", LogisticRegression()), ("xgb", XGBClassifier()), ...]
    The meta-learner (final_estimator) is a simple Logistic Regression —
    it just learns how to weight the base models' outputs, so it doesn't
    need to be complex itself.
    """
    try:
        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=3,   # 3 folds keeps this tractable on constrained compute; 5 is preferable with more
            n_jobs=-1,
        )
        logger.info(f"Ensemble built from base models: {[name for name, _ in base_models]}")
        return ensemble
    except Exception as e:
        raise SentrixException(e, sys)
