"""
src/models/boosting.py

Role
----
The workhorses. Gradient-boosted trees usually win on tabular data like
this one, and Optuna tunes them systematically against PR-AUC rather than
by hand-guessing values.

Why the tuning CV is a TimeSeriesSplit
--------------------------------------
A shuffled StratifiedKFold over a seller-day panel scores each fold with a
model fitted on rows from that fold's own future — and, because rolling
features and a 30-day forward label make neighbouring rows near-duplicates,
also on rows that are almost copies of the ones being scored. Both inflate
the tuning score, so Optuna optimises for the wrong thing and picks
hyperparameters (usually: deeper, more trees) that overfit.

The outer split in trainer.py is purged and embargoed for exactly this
reason. Leaving the inner tuning loop shuffled would undo that work one
level down, which is a very easy mistake to ship because every number it
produces looks better.
"""

import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import optuna

from src.config_loader import load_config
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

# Optuna's own logs are noisy at INFO level; keep it to warnings only
optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_xgboost(params: dict | None = None) -> XGBClassifier:
    """Construct an XGBClassifier — either config.yaml defaults or tuned params."""
    try:
        cfg = load_config()["model"]["xgboost"]
        params = params or {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
        }
        model = XGBClassifier(
            **params,
            eval_metric=cfg["eval_metric"],
            random_state=load_config()["project"]["random_state"],
            n_jobs=-1,
        )
        logger.info(f"XGBoost built with params: {params}")
        return model
    except Exception as e:
        raise SentrixException(e, sys)


def build_lightgbm(params: dict | None = None) -> LGBMClassifier:
    """Construct an LGBMClassifier — either config.yaml defaults or tuned params."""
    try:
        cfg = load_config()["model"]["lightgbm"]
        params = params or {
            "n_estimators": cfg["n_estimators"],
            "max_depth": cfg["max_depth"],
            "learning_rate": cfg["learning_rate"],
        }
        model = LGBMClassifier(
            **params,
            random_state=load_config()["project"]["random_state"],
            n_jobs=-1,
            verbosity=-1,
        )
        logger.info(f"LightGBM built with params: {params}")
        return model
    except Exception as e:
        raise SentrixException(e, sys)


def tune_xgboost(X, y) -> dict:
    """
    Hyperparameter search for XGBoost via Optuna, optimising PR-AUC under a
    forward-chaining TimeSeriesSplit.

    X and y must arrive in chronological order — TimeSeriesSplit slices
    positionally, so an unsorted matrix silently degrades into a random
    split. trainer.prepare_data guarantees the ordering.

    Tuning runs on the most recent `tune_sample_rows` rows rather than the
    full panel. That is a deliberate trade, not a shortcut: hyperparameter
    RANKING is stable under subsampling while fit time is not, and the
    recent tail is the regime the model will actually be scored on. The
    chosen params are then refit on the complete training set.
    """
    try:
        cfg = load_config()
        opt_cfg = cfg["model"]["optuna"]
        seed = cfg["project"]["random_state"]

        X, y = np.asarray(X), np.asarray(y)
        sample_rows = opt_cfg.get("tune_sample_rows")
        if sample_rows and len(y) > sample_rows:
            X_t, y_t = X[-sample_rows:], y[-sample_rows:]
            logger.info(f"Tuning on the most recent {sample_rows:,} of {len(y):,} training rows")
        else:
            X_t, y_t = X, y

        cv = TimeSeriesSplit(n_splits=opt_cfg["cv_folds"])

        # Boosted trees handle imbalance through the loss, not by resampling.
        pos = float((y_t == 1).sum())
        scale_pos_weight = float((y_t == 0).sum()) / max(pos, 1.0)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }
            model = XGBClassifier(**params, scale_pos_weight=scale_pos_weight,
                                  eval_metric="logloss", random_state=seed, n_jobs=-1)
            scores = cross_val_score(model, X_t, y_t, scoring=opt_cfg["scoring"], cv=cv)
            return float(scores.mean())

        study = optuna.create_study(
            direction=opt_cfg["direction"],
            sampler=optuna.samplers.TPESampler(seed=seed),   # reproducible search
        )
        study.optimize(objective, n_trials=opt_cfg["n_trials"], show_progress_bar=False)

        best = dict(study.best_params)
        best["scale_pos_weight"] = scale_pos_weight
        logger.info(f"Optuna best PR-AUC: {study.best_value:.4f} over "
                    f"{opt_cfg['n_trials']} trials — params: {best}")
        return best
    except Exception as e:
        raise SentrixException(e, sys)
