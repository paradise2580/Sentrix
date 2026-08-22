"""
src/models/trainer.py

Role
----
The single place that takes the feature table all the way through to
six saved model artifacts. Every model in Phase 4 routes through here so
the train/test split, preprocessing, and imbalance handling are applied
identically and only once.

Train/test split strategy
--------------------------
A chronological split (not a random shuffle) — the last N% of dates by
as_of_date become the test set. This mirrors how the model will actually
be used in production (train on the past, evaluate on data the model has
never seen from the future) and avoids the subtle leakage a random split
would introduce given the rolling-window features.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE

from src.config_loader import load_config, get_project_root
from src.preprocessing.pipeline import (
    fit_and_save_preprocessor, load_preprocessor, transform, get_feature_columns,
)
from src.models.baseline import build_logistic_regression, build_random_forest
from src.models.boosting import build_xgboost, build_lightgbm, tune_xgboost
from src.models.ensemble import build_ensemble
from src.models.deep import train_lstm, predict_lstm
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def chronological_split(df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by as_of_date so the test set is strictly later in time than training."""
    try:
        df = df.sort_values("as_of_date")
        cutoff_idx = int(len(df) * (1 - test_size))
        cutoff_date = df.iloc[cutoff_idx]["as_of_date"]

        train_df = df[df["as_of_date"] < cutoff_date].copy()
        test_df = df[df["as_of_date"] >= cutoff_date].copy()

        logger.info(
            f"Chronological split at {cutoff_date}: "
            f"train={len(train_df)} rows, test={len(test_df)} rows"
        )
        return train_df, test_df
    except Exception as e:
        raise SentrixException(e, sys)


def apply_smote(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic oversampling of the minority class on the TRAINING set only."""
    try:
        smote = SMOTE(random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(f"SMOTE applied: {len(y)} -> {len(y_res)} rows "
                     f"(positive class {y.mean():.1%} -> {y_res.mean():.1%})")
        return X_res, y_res
    except Exception as e:
        raise SentrixException(e, sys)


def save_model(model, name: str) -> Path:
    """Persist a fitted model to artifacts/models/<name>.joblib."""
    cfg = load_config()
    models_dir = get_project_root() / cfg["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info(f"Saved model '{name}' to {path}")
    return path


def prepare_data(feature_table_path: str | None = None):
    """
    Shared setup used by every stage: load features, chronological split,
    fit (or load) the preprocessor, transform both splits, and build the
    SMOTE-resampled training set. Cheap (~1-2s) — safe to re-run per stage.
    """
    cfg = load_config()
    seed = cfg["project"]["random_state"]
    target = cfg["model"]["target_column"]

    path = feature_table_path or (get_project_root() / cfg["paths"]["data_processed"] / "features.csv")
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    logger.info(f"Loaded feature table: {df.shape}")

    train_df, test_df = chronological_split(df, cfg["preprocessing"]["test_size"])

    fit_and_save_preprocessor(train_df)
    bundle = load_preprocessor()

    X_train = transform(train_df, bundle)
    y_train = train_df[target].values
    X_test = transform(test_df, bundle)
    y_test = test_df[target].values
    X_train_smote, y_train_smote = apply_smote(X_train, y_train, seed)

    return {
        "df": df, "train_df": train_df, "test_df": test_df,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "X_train_smote": X_train_smote, "y_train_smote": y_train_smote,
    }


def train_baseline_stage(data: dict) -> dict:
    """Stage 1: Logistic Regression + Random Forest. Fast (~30-40s)."""
    results = {}
    logger.info("=== Training Logistic Regression ===")
    logreg = build_logistic_regression()
    logreg.fit(data["X_train"], data["y_train"])
    save_model(logreg, "logistic_regression")
    results["logistic_regression"] = {"model": logreg, "X_test": data["X_test"], "y_test": data["y_test"]}

    logger.info("=== Training Random Forest ===")
    rf = build_random_forest()
    rf.fit(data["X_train"], data["y_train"])
    save_model(rf, "random_forest")
    results["random_forest"] = {"model": rf, "X_test": data["X_test"], "y_test": data["y_test"]}
    return results


def train_boosting_stage(data: dict) -> dict:
    """Stage 2: XGBoost (Optuna-tuned) + LightGBM (trained on SMOTE data). ~60-90s."""
    results = {}
    logger.info("=== Tuning + Training XGBoost (Optuna) ===")
    best_params = tune_xgboost(data["X_train"], data["y_train"])
    xgb = build_xgboost(best_params)
    xgb.fit(data["X_train"], data["y_train"])
    save_model(xgb, "xgboost")
    joblib.dump(best_params, get_project_root() / load_config()["paths"]["models"] / "xgboost_best_params.joblib")
    results["xgboost"] = {"model": xgb, "X_test": data["X_test"], "y_test": data["y_test"]}

    logger.info("=== Training LightGBM ===")
    lgbm = build_lightgbm()
    lgbm.fit(data["X_train_smote"], data["y_train_smote"])
    save_model(lgbm, "lightgbm")
    results["lightgbm"] = {"model": lgbm, "X_test": data["X_test"], "y_test": data["y_test"]}
    return results


def train_ensemble_stage(data: dict) -> dict:
    """Stage 3: Stacking ensemble. Slowest single stage (~2-3 min, refits base models under CV)."""
    cfg = load_config()
    models_dir = get_project_root() / cfg["paths"]["models"]
    best_params_path = models_dir / "xgboost_best_params.joblib"
    best_params = joblib.load(best_params_path) if best_params_path.exists() else None

    logger.info("=== Training Stacking Ensemble ===")
    # The stacker refits every base model once per CV fold, so a 300-tree RF
    # inside it costs ~n_folds x the standalone cost. A lighter RF is used
    # HERE ONLY (the standalone random_forest artifact keeps its full 300
    # trees) to keep ensemble training tractable without weakening the
    # models that are evaluated individually.
    light_rf = build_random_forest()
    light_rf.set_params(n_estimators=60, max_depth=8)

    ensemble = build_ensemble([
        ("logreg", build_logistic_regression()),
        ("rf", light_rf),
        ("xgb", build_xgboost(best_params)),
    ])
    ensemble.fit(data["X_train"], data["y_train"])
    save_model(ensemble, "ensemble")
    return {"ensemble": {"model": ensemble, "X_test": data["X_test"], "y_test": data["y_test"]}}


def train_lstm_stage(data: dict) -> dict:
    """Stage 4: PyTorch LSTM over supplier sequences. ~1-2 min for 5 epochs."""
    cfg = load_config()
    numeric_cols, _ = get_feature_columns(data["df"])

    logger.info("=== Training LSTM ===")
    lstm_model, lstm_meta = train_lstm(data["train_df"], numeric_cols)

    models_dir = get_project_root() / cfg["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lstm_model.state_dict(), models_dir / "lstm.pt")
    joblib.dump(lstm_meta, models_dir / "lstm_meta.joblib")
    logger.info(f"Saved LSTM model + metadata to {models_dir}")

    lstm_probs = predict_lstm(lstm_model, data["test_df"], lstm_meta)
    target = cfg["model"]["target_column"]
    lstm_test_labels = []
    for _, group in data["test_df"].sort_values(["seller_id", "as_of_date"]).groupby("seller_id"):
        lstm_test_labels.extend(group[target].values[lstm_meta["seq_len"]:])

    return {"lstm": {"model": lstm_model, "probs": lstm_probs, "y_test": np.array(lstm_test_labels)}}


# Stage registry — used by both the CLI and any orchestrator (e.g. Airflow) that
# wants to run stages independently instead of one long blocking call.
_STAGES = {
    "baseline": train_baseline_stage,
    "boosting": train_boosting_stage,
    "ensemble": train_ensemble_stage,
    "lstm": train_lstm_stage,
}


def train_all_models(feature_table_path: str | None = None, stages: list[str] | None = None) -> dict:
    """
    Run one or more training stages end to end. Defaults to all four stages
    (six models total: 2 baseline + 2 boosting + 1 ensemble + 1 LSTM).

    Each stage is independently callable and saves its own artifacts, so a
    long training job can be resumed stage-by-stage rather than re-run from
    scratch — the same pattern a real training pipeline uses when a single
    run would otherwise exceed a compute session's time budget.
    """
    try:
        stages = stages or list(_STAGES.keys())
        data = prepare_data(feature_table_path)

        results = {}
        for stage_name in stages:
            results.update(_STAGES[stage_name](data))

        logger.info(f"Stage(s) complete: {stages}")
        return results
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train SENTRIX models, stage by stage.")
    parser.add_argument(
        "--stages", nargs="+", choices=list(_STAGES.keys()), default=list(_STAGES.keys()),
        help="Which training stage(s) to run. Default: all.",
    )
    args = parser.parse_args()
    train_all_models(stages=args.stages)
