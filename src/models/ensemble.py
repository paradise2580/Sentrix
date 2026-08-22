"""
src/models/ensemble.py

Role
----
A stacking meta-learner that combines the base models' predictions.

Why the CV scheme is the whole story here
-----------------------------------------
A stacker trains its meta-learner on OUT-OF-FOLD predictions from the base
models. With the default k-fold scheme, fold 1's out-of-fold predictions
come from base models fitted on folds 2 and 3 — that is, on data from the
FUTURE relative to the rows being scored. On a seller-day panel with a
30-day forward label, that reintroduces exactly the leakage the outer
purged split was built to remove, one level down, and the meta-learner
learns weights that only make sense with hindsight.

The first version of this file used `cv=3` and the resulting ensemble
scored ROC-AUC 0.538 on the held-out test set — barely better than a coin
flip, and worse than every one of its own base models. That is the
signature of a meta-learner fitted on leaked out-of-fold predictions: it
looks strong in-fold and collapses out of sample.

TimeSeriesSplit fixes it: every fold trains on the past and scores the
future, matching how the model is used.

The LSTM is intentionally excluded from the sklearn StackingClassifier
(it isn't sklearn-compatible and it operates on sequences, not flat rows).
"""

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from src.config_loader import load_config
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def build_ensemble(base_models: list[tuple[str, object]],
                   n_splits: int | None = None) -> StackingClassifier:
    """
    base_models: list of (name, estimator) tuples.

    The meta-learner is a plain Logistic Regression — it only has to learn
    how to weight a handful of base-model outputs, so complexity there buys
    nothing and costs interpretability.

    IMPORTANT: the rows passed to .fit() must already be in chronological
    order, because TimeSeriesSplit slices positionally. prepare_data
    guarantees this (its splits are produced from a date-sorted frame).
    """
    try:
        cfg = load_config()["model"]
        n_splits = n_splits or cfg.get("ensemble", {}).get("n_splits", 3)

        ensemble = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
            cv=TimeSeriesSplit(n_splits=n_splits),
            passthrough=False,
            n_jobs=-1,
        )
        logger.info(
            f"Ensemble built from {[name for name, _ in base_models]} "
            f"with TimeSeriesSplit(n_splits={n_splits})"
        )
        return ensemble
    except Exception as e:
        raise SentrixException(e, sys)
