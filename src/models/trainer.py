"""
Trains every model from the feature table, so the split, preprocessing and
imbalance handling are identical for all of them.

Split (by purchase date):
    [ TRAIN ] <embargo> [ CALIB ] <embargo> [ TEST ]

Labels resolve at delivery, weeks after purchase, so a 30-day embargo is
dropped before each block to keep training labels out of later blocks.
CALIB is only used to fit probability calibration.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE

from src.config_loader import load_config, get_project_root
from src.preprocessing.pipeline import (
    fit_and_save_preprocessor, load_preprocessor, transform, transform_to_frame,
)
from src.models.baseline import build_logistic_regression, build_random_forest
from src.models.boosting import build_xgboost, build_lightgbm, tune_xgboost
from src.models.ensemble import build_ensemble
from src.models.deep import train_lstm
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def purged_temporal_split(
    df: pd.DataFrame,
    test_size: float,
    calib_size: float,
    embargo_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by as_of_date into (train_df, calib_df, test_df), dropping
    `embargo_days` before each later block.
    """
    try:
        df = df.sort_values("as_of_date").reset_index(drop=True)
        dates = df["as_of_date"]

        test_start = dates.quantile(1 - test_size, interpolation="nearest")
        calib_start = dates.quantile(1 - test_size - calib_size, interpolation="nearest")

        embargo = pd.Timedelta(days=int(embargo_days))
        train_end = calib_start - embargo
        calib_end = test_start - embargo

        train_df = df[df["as_of_date"] < train_end].copy()
        calib_df = df[(df["as_of_date"] >= calib_start) & (df["as_of_date"] < calib_end)].copy()
        test_df = df[df["as_of_date"] >= test_start].copy()

        dropped = len(df) - (len(train_df) + len(calib_df) + len(test_df))
        logger.info(
            f"Purged temporal split (embargo={embargo_days}d): "
            f"train < {train_end.date()} ({len(train_df):,} rows) | "
            f"calib {calib_start.date()}–{calib_end.date()} ({len(calib_df):,} rows) | "
            f"test >= {test_start.date()} ({len(test_df):,} rows) | "
            f"{dropped:,} rows discarded into embargo gaps"
        )
        if train_df.empty or test_df.empty:
            raise ValueError(
                "Embargo consumed an entire block — the feature table's date "
                "range is too short for this test_size/embargo combination."
            )
        return train_df, calib_df, test_df
    except Exception as e:
        raise SentrixException(e, sys)


def chronological_split(df: pd.DataFrame, test_size: float,
                        embargo_days: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-way version of purged_temporal_split (no calibration block)."""
    train_df, _, test_df = purged_temporal_split(df, test_size, 0.0, embargo_days)
    return train_df, test_df


def apply_smote(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Oversample the minority class. Training data only, never calib/test."""
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


def prepare_data(feature_table_path: str | None = None) -> dict:
    """
    Load features, split, fit the preprocessor on train only, transform every
    block, and build the SMOTE training set. Deterministic, so evaluation can
    rebuild the same split.
    """
    cfg = load_config()
    seed = cfg["project"]["random_state"]
    target = cfg["model"]["target_column"]
    pre = cfg["preprocessing"]

    path = feature_table_path or (
        get_project_root() / cfg["paths"]["data_processed"] / "features.csv"
    )
    df = pd.read_csv(path, parse_dates=["as_of_date"])
    logger.info(f"Loaded feature table: {df.shape}, positive rate {df[target].mean():.2%}")

    train_df, calib_df, test_df = purged_temporal_split(
        df,
        test_size=pre["test_size"],
        calib_size=pre.get("calibration_size", 0.10),
        embargo_days=pre.get("embargo_days", pre.get("label_horizon_days", 30)),
    )

    # Fitted on train only.
    fit_and_save_preprocessor(train_df)
    bundle = load_preprocessor()

    X_train, y_train = transform(train_df, bundle), train_df[target].values
    X_calib, y_calib = transform(calib_df, bundle), calib_df[target].values
    X_test, y_test = transform(test_df, bundle), test_df[target].values

    X_train_smote, y_train_smote = apply_smote(X_train, y_train, seed)

    # Frames for the LSTM, sorted so sequences are deterministic.
    seq_train = transform_to_frame(train_df, bundle).sort_values(["seller_id", "as_of_date"])
    seq_test = transform_to_frame(test_df, bundle).sort_values(["seller_id", "as_of_date"])

    return {
        "df": df, "train_df": train_df, "calib_df": calib_df, "test_df": test_df,
        "X_train": X_train, "y_train": y_train,
        "X_calib": X_calib, "y_calib": y_calib,
        "X_test": X_test, "y_test": y_test,
        "X_train_smote": X_train_smote, "y_train_smote": y_train_smote,
        "seq_train": seq_train, "seq_test": seq_test,
        "feature_names": bundle["feature_names_out"],
        "bundle": bundle,
    }


def train_baseline_stage(data: dict) -> dict:
    """Stage 1: Logistic Regression + Random Forest."""
    results = {}
    logger.info("=== Training Logistic Regression ===")
    logreg = build_logistic_regression()
    logreg.fit(data["X_train"], data["y_train"])
    save_model(logreg, "logistic_regression")
    results["logistic_regression"] = {"model": logreg}

    logger.info("=== Training Random Forest ===")
    rf = build_random_forest()
    rf.fit(data["X_train"], data["y_train"])
    save_model(rf, "random_forest")
    results["random_forest"] = {"model": rf}
    return results


def train_boosting_stage(data: dict) -> dict:
    """Stage 2: XGBoost (Optuna-tuned) + LightGBM (trained on SMOTE data)."""
    results = {}
    logger.info("=== Tuning + Training XGBoost (Optuna) ===")
    best_params = tune_xgboost(data["X_train"], data["y_train"])
    xgb = build_xgboost(best_params)
    xgb.fit(data["X_train"], data["y_train"])
    save_model(xgb, "xgboost")
    joblib.dump(best_params,
                get_project_root() / load_config()["paths"]["models"] / "xgboost_best_params.joblib")
    results["xgboost"] = {"model": xgb}

    logger.info("=== Training LightGBM ===")
    lgbm = build_lightgbm()
    lgbm.fit(data["X_train_smote"], data["y_train_smote"])
    save_model(lgbm, "lightgbm")
    results["lightgbm"] = {"model": lgbm}
    return results


def train_ensemble_stage(data: dict) -> dict:
    """Stage 3: temporal stacking ensemble (see models/ensemble.py)."""
    cfg = load_config()
    models_dir = get_project_root() / cfg["paths"]["models"]
    best_params_path = models_dir / "xgboost_best_params.joblib"
    best_params = joblib.load(best_params_path) if best_params_path.exists() else None

    logger.info("=== Training Stacking Ensemble ===")
    # Lighter RF inside the stack only, since it is refit once per fold.
    light_rf = build_random_forest()
    light_rf.set_params(n_estimators=60, max_depth=8)

    ensemble = build_ensemble([
        ("logreg", build_logistic_regression()),
        ("rf", light_rf),
        ("xgb", build_xgboost(best_params)),
    ])
    ensemble.fit(data["X_train"], data["y_train"])
    save_model(ensemble, "ensemble")
    return {"ensemble": {"model": ensemble}}


def train_lstm_stage(data: dict) -> dict:
    """Stage 4: PyTorch LSTM over per-seller sequences (optional)."""
    cfg = load_config()
    feature_names = data["feature_names"]

    logger.info("=== Training LSTM ===")
    lstm_model, lstm_meta = train_lstm(data["seq_train"], feature_names)

    models_dir = get_project_root() / cfg["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(lstm_model.state_dict(), models_dir / "lstm.pt")
    joblib.dump(lstm_meta, models_dir / "lstm_meta.joblib")
    logger.info(f"Saved LSTM model + metadata to {models_dir}")

    return {"lstm": {"model": lstm_model, "meta": lstm_meta}}


# Stages can be run independently from the CLI.
_STAGES = {
    "baseline": train_baseline_stage,
    "boosting": train_boosting_stage,
    "ensemble": train_ensemble_stage,
    "lstm": train_lstm_stage,
}

# LSTM is off by default: orders are independent shipments, not a sequence.
# Run it with --stages lstm.
_DEFAULT_STAGES = ["baseline", "boosting", "ensemble"]


def train_all_models(feature_table_path: str | None = None,
                     stages: list[str] | None = None) -> dict:
    """
    Run the given training stages (default: baseline, boosting, ensemble).
    Each stage saves its own artifacts.
    """
    try:
        stages = stages or list(_DEFAULT_STAGES)
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
        "--stages", nargs="+", choices=list(_STAGES.keys()), default=list(_DEFAULT_STAGES),
        help="Which training stage(s) to run. Default: all.",
    )
    args = parser.parse_args()
    train_all_models(stages=args.stages)
