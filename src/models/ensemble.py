"""
Stacking ensemble whose meta-learner is trained on forward-in-time
out-of-fold predictions.

Written by hand because sklearn's StackingClassifier uses k-fold, which lets
the meta-learner see the future (that version scored ROC-AUC 0.538), and
it rejects TimeSeriesSplit because that split is not a partition.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from src.config_loader import load_config
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


class TemporalStackingClassifier(ClassifierMixin, BaseEstimator):
    """
    Stacking with forward-chaining (TimeSeriesSplit) out-of-fold predictions.

    Rows passed to fit() must already be in chronological order.

    estimators      list of (name, estimator) base models
    final_estimator meta-learner, default balanced LogisticRegression
    n_splits        number of forward-chaining folds
    """

    def __init__(self, estimators, final_estimator=None, n_splits: int = 3):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)

        splitter = TimeSeriesSplit(n_splits=self.n_splits)
        meta_X, meta_y = [], []

        # Each fold trains on the past and predicts the next block.
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            fold_preds = []
            for name, est in self.estimators:
                model = clone(est)
                model.fit(X[train_idx], y[train_idx])
                fold_preds.append(model.predict_proba(X[test_idx])[:, 1])
            meta_X.append(np.column_stack(fold_preds))
            meta_y.append(y[test_idx])
            logger.info(f"Stack fold {fold}/{self.n_splits}: fitted on "
                        f"{len(train_idx):,} rows, scored {len(test_idx):,}")

        meta_X = np.vstack(meta_X)
        meta_y = np.concatenate(meta_y)

        # The earliest block is only ever training data, never scored.
        logger.info(f"Meta-learner training set: {len(meta_y):,} of {len(y):,} "
                    f"training rows ({len(meta_y) / len(y):.0%}) — the earliest "
                    f"block is training-only by construction")

        self.final_estimator_ = clone(
            self.final_estimator or LogisticRegression(max_iter=1000, class_weight="balanced")
        )
        self.final_estimator_.fit(meta_X, meta_y)

        # Refit base models on the full training set for inference.
        self.estimators_ = []
        for name, est in self.estimators:
            model = clone(est)
            model.fit(X, y)
            self.estimators_.append((name, model))

        coefs = getattr(self.final_estimator_, "coef_", None)
        if coefs is not None:
            weights = dict(zip([n for n, _ in self.estimators], coefs[0].round(3)))
            logger.info(f"Meta-learner weights: {weights}")

        return self

    def _meta_features(self, X) -> np.ndarray:
        return np.column_stack(
            [m.predict_proba(np.asarray(X))[:, 1] for _, m in self.estimators_]
        )

    def predict_proba(self, X) -> np.ndarray:
        return self.final_estimator_.predict_proba(self._meta_features(X))

    def predict(self, X) -> np.ndarray:
        return self.final_estimator_.predict(self._meta_features(X))


def build_ensemble(base_models: list[tuple[str, object]],
                   n_splits: int | None = None) -> TemporalStackingClassifier:
    """base_models: list of (name, estimator). The LSTM is not included."""
    try:
        cfg = load_config()["model"]
        n_splits = n_splits or cfg.get("ensemble", {}).get("n_splits", 3)

        ensemble = TemporalStackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
            n_splits=n_splits,
        )
        logger.info(
            f"Temporal stacking ensemble built from {[n for n, _ in base_models]} "
            f"with forward-chaining CV (n_splits={n_splits})"
        )
        return ensemble
    except Exception as e:
        raise SentrixException(e, sys)
