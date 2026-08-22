"""
src/models/boosting.py

Role
----
The workhorses. Gradient-boosted trees usually win on tabular data like
ours, which is why AMEX and PayU name XGBoost directly in their job
descriptions. Optuna tunes hyperparameters systematically — via cross-
validated search on PR-AUC — rather than by hand-guessing values.
"""

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
    Systematic hyperparameter search for XGBoost via Optuna, optimizing
    the metric config.yaml specifies (PR-AUC / average_precision) under
    stratified cross-validation.
    """
    try:
        cfg = load_config()
        opt_cfg = cfg["model"]["optuna"]
        seed = cfg["project"]["random_state"]
        cv = StratifiedKFold(n_splits=opt_cfg["cv_folds"], shuffle=True, random_state=seed)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
            model = XGBClassifier(**params, eval_metric="logloss", random_state=seed, n_jobs=-1)
            scores = cross_val_score(model, X, y, scoring=opt_cfg["scoring"], cv=cv)
            return scores.mean()

        study = optuna.create_study(direction=opt_cfg["direction"])
        study.optimize(objective, n_trials=opt_cfg["n_trials"], show_progress_bar=False)

        logger.info(f"Optuna best PR-AUC: {study.best_value:.4f}, params: {study.best_params}")
        return study.best_params
    except Exception as e:
        raise SentrixException(e, sys)
