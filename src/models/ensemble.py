"""
src/models/ensemble.py

Role
----
A stacking meta-learner that blends the base models' predictions, built so
that the blend itself cannot leak.

Why this is hand-rolled instead of sklearn's StackingClassifier
---------------------------------------------------------------
A stacker trains its meta-learner on OUT-OF-FOLD base predictions, so the
CV scheme inside the stacker matters as much as the outer train/test
split. With a k-fold scheme, fold 1's out-of-fold predictions come from
base models fitted on folds 2 and 3 — data from the FUTURE of the rows
being scored. On a seller-day panel with a 30-day forward label that
reintroduces exactly the leakage the outer purged split removes, one level
down, and the meta-learner learns weights that only make sense with
hindsight.

The first version of this file used `StackingClassifier(cv=3)`. The
resulting ensemble scored ROC-AUC 0.538 on the held-out test set — barely
better than a coin flip, and worse than every one of its own base models.
That is the signature of a meta-learner fitted on leaked predictions: it
looks strong in-fold and collapses out of sample.

The obvious fix — `StackingClassifier(cv=TimeSeriesSplit(3))` — does not
work. sklearn builds the out-of-fold matrix with `cross_val_predict`,
which requires the CV to be a PARTITION: every sample must appear in
exactly one test fold. TimeSeriesSplit is deliberately not a partition
(the first training block is never in any test fold), so sklearn raises
`ValueError: cross_val_predict only works for partitions`.

So the forward-chaining stack is written out explicitly below. Each fold
fits the base models on the past and predicts the future; the meta-learner
trains only on those honest predictions; the base models are then refit on
the full training set for inference. It is about forty lines, and it is the
only way to get a leak-free temporal stack out of this stack of libraries.
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
    Stacking with forward-chaining out-of-fold predictions.

    Rows passed to fit() MUST already be in chronological order —
    TimeSeriesSplit slices positionally, so an unsorted matrix silently
    degrades into a random split. trainer.prepare_data guarantees the
    ordering.

    Parameters
    ----------
    estimators : list of (name, estimator)
        Base models. Cloned before every fit, so the caller's instances
        are never mutated.
    final_estimator : estimator, optional
        The meta-learner. Defaults to balanced Logistic Regression — it
        only has to weight a handful of base outputs, so complexity there
        buys nothing and costs interpretability.
    n_splits : int
        Forward-chaining folds used to build the meta-training set.
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

        # --- forward-chaining out-of-fold predictions -----------------------
        # Fold k fits on rows [0 : t_k) and predicts [t_k : t_k+1). No fold
        # is ever scored by a model that has seen its future.
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

        # The earliest block never appears above — it is only ever training
        # data for a fold, never a scored one. That is the price of not
        # leaking, and it is why this is not a partition.
        logger.info(f"Meta-learner training set: {len(meta_y):,} of {len(y):,} "
                    f"training rows ({len(meta_y) / len(y):.0%}) — the earliest "
                    f"block is training-only by construction")

        self.final_estimator_ = clone(
            self.final_estimator or LogisticRegression(max_iter=1000, class_weight="balanced")
        )
        self.final_estimator_.fit(meta_X, meta_y)

        # --- refit base models on the FULL training set for inference -------
        # The fold models were only ever a device for generating honest
        # meta-features; throwing away 1/(n+1) of the data at serving time
        # would be a waste.
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
    """
    base_models: list of (name, estimator) tuples.

    The LSTM is intentionally excluded — it isn't sklearn-compatible and it
    consumes sequences rather than flat rows, so it is evaluated as its own
    model rather than blended in here.
    """
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
