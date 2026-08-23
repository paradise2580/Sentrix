"""
src/evaluation/generate_predictions.py

Role
----
Closes the loop: for every seller's most recent snapshot, score it with the
winning model, calibrate the score, band it, explain it with SHAP, and
write {risk_score, risk_band, top_features} into the MySQL predictions
table that the API and dashboard read from.

Scores are written CALIBRATED
-----------------------------
The number stored here is what the dashboard shows a human and what any
downstream cost calculation multiplies. A raw score from a model trained
with class_weight="balanced" or on SMOTE-resampled data is systematically
inflated — fine for ranking, wrong to display as "38% chance of being
late". The isotonic calibrator fitted during evaluation on held-out data
is applied before anything is stored.

Bands are assigned by RANK, not by absolute probability — see
metrics.assign_risk_bands_by_quantile for why.

Run with:
    python -m src.evaluation.generate_predictions
"""

import json
import joblib
import numpy as np
import pandas as pd

from src.config_loader import load_config, get_project_root
from src.ingestion.loader import DataLoader
from src.preprocessing.pipeline import (
    load_preprocessor, transform, transform_to_frame, get_feature_columns,
)
from src.models.explainer import (
    build_explainer, get_readable_feature_names, explain_single_prediction,
)
from src.models.deep import predict_lstm
from src.evaluation.run_evaluation import load_lstm_model
from src.evaluation.calibration import apply_calibrator
from src.evaluation.metrics import assign_risk_bands_by_quantile
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

# Model whose SHAP values explain the stored predictions when the winning
# model cannot be explained directly (a StackingClassifier has no single
# feature space; an LSTM's inputs are sequences, not rows).
EXPLAINER_PROXY = "xgboost"


def _score_latest_tabular(model, latest: pd.DataFrame, bundle: dict) -> np.ndarray:
    return model.predict_proba(transform(latest, bundle))[:, 1]


def _score_latest_lstm(df: pd.DataFrame, latest: pd.DataFrame, bundle: dict) -> np.ndarray:
    """
    Score each seller's most recent day with the sequence model. Needs the
    preceding seq_len days, so it slices the tail of the full panel rather
    than the one-row-per-seller snapshot.
    """
    model, meta = load_lstm_model()
    seq_len = meta["seq_len"]

    tail = (df.sort_values(["seller_id", "as_of_date"])
              .groupby("seller_id").tail(seq_len + 1))
    seq_frame = transform_to_frame(tail, bundle).sort_values(["seller_id", "as_of_date"])
    probs, row_index = predict_lstm(model, seq_frame, meta)

    scored = pd.Series(probs, index=row_index)
    # Sellers with fewer than seq_len+1 days cannot be scored by the LSTM;
    # they fall back to the population median rather than silently vanishing.
    aligned = scored.reindex(latest.index)
    n_missing = int(aligned.isna().sum())
    if n_missing:
        logger.warning(f"{n_missing} sellers lack {seq_len} days of history for the "
                       f"LSTM; assigned the median score")
        aligned = aligned.fillna(float(np.nanmedian(probs)))
    return aligned.values


def generate_and_store_predictions() -> pd.DataFrame:
    try:
        cfg = load_config()
        eval_dir = get_project_root() / "artifacts" / "evaluation"

        best_summary = joblib.load(eval_dir / "best_model_summary.joblib")
        best_model_name = best_summary["best_model"]
        logger.info(f"Scoring with the evaluation winner: {best_model_name}")

        features_path = get_project_root() / cfg["paths"]["data_processed"] / "features.csv"
        df = pd.read_csv(features_path, parse_dates=["as_of_date"]).sort_values(
            ["seller_id", "as_of_date"]).reset_index(drop=True)

        latest = df.groupby("seller_id").tail(1)
        logger.info(f"Scoring latest snapshot for {len(latest):,} sellers")

        bundle = load_preprocessor()

        if best_model_name == "lstm":
            raw = _score_latest_lstm(df, latest, bundle)
        else:
            model = joblib.load(
                get_project_root() / cfg["paths"]["models"] / f"{best_model_name}.joblib")
            raw = _score_latest_tabular(model, latest, bundle)

        # --- calibrate ------------------------------------------------------
        calibrator_path = eval_dir / "calibrator.joblib"
        if calibrator_path.exists():
            probs = apply_calibrator(joblib.load(calibrator_path), raw)
            logger.info(f"Applied isotonic calibration: mean score "
                        f"{raw.mean():.4f} -> {probs.mean():.4f}")
        else:
            probs = raw
            logger.warning("No calibrator found — storing RAW scores. Run "
                           "src.evaluation.run_evaluation first.")

        # Band on the RAW score, display the CALIBRATED one.
        #
        # Isotonic regression is a step function: every raw score inside a bin
        # maps to a single output value, so the calibrated scores collapse onto
        # a few dozen distinct numbers. Quantile cuts then land on large ties
        # and push everything at the boundary into the higher band — the top
        # band came out 5.5% instead of 5%, and "medium" swallowed 41% of the
        # population instead of 30%.
        #
        # Calibration is monotone, so banding on the raw score changes no
        # ordering. It only restores the resolution calibration flattened, and
        # the bands then match the percentages the UI claims.
        bands = assign_risk_bands_by_quantile(raw, cfg["model"]["risk_band_quantiles"])

        # --- explain --------------------------------------------------------
        # SHAP needs a model with a flat feature space. The ensemble and the
        # LSTM don't have one, so the standalone XGBoost stands in and the
        # substitution is recorded in the row rather than hidden.
        explain_name = best_model_name
        if best_model_name in {"ensemble", "lstm"}:
            explain_name = EXPLAINER_PROXY
            logger.info(f"{best_model_name} is not directly explainable; "
                        f"using {EXPLAINER_PROXY} for SHAP attributions")

        explain_model = joblib.load(
            get_project_root() / cfg["paths"]["models"] / f"{explain_name}.joblib")

        X_latest = transform(latest, bundle)
        explainer = build_explainer(explain_model, background=X_latest)
        numeric_cols, categorical_cols = get_feature_columns(df)
        feature_names = get_readable_feature_names(
            numeric_cols, categorical_cols, bundle["preprocessor"])

        records = []
        for i, (_, row) in enumerate(latest.iterrows()):
            top_features = explain_single_prediction(
                explainer, X_latest[i], feature_names, top_n=5)
            records.append({
                "seller_id": str(row["seller_id"]),   # real Olist IDs are hash strings
                "risk_score": round(float(probs[i]), 4),
                "risk_band": bands[i],
                # When SHAP had to fall back to a proxy, that substitution
                # travels with the row instead of being hidden in a log line.
                "model_name": (best_model_name if explain_name == best_model_name
                               else f"{best_model_name} (SHAP via {explain_name})"),
                "top_features": json.dumps(top_features),
            })

        predictions_df = pd.DataFrame(records)

        # Same reasoning for ordering: rank by raw score so the worklist has a
        # strict order, then hand back the calibrated probability for display.
        predictions_df = (predictions_df.assign(_raw=raw)
                                        .sort_values("_raw", ascending=False)
                                        .drop(columns="_raw")
                                        .reset_index(drop=True))

        loader = DataLoader()
        loader.truncate_table(cfg["mysql"]["tables"]["predictions"])
        loader.write_df(predictions_df, cfg["mysql"]["tables"]["predictions"],
                        if_exists="append")

        logger.info(f"Wrote {len(predictions_df):,} calibrated predictions to MySQL")
        counts = predictions_df["risk_band"].value_counts()
        print(pd.DataFrame({"sellers": counts,
                            "share": (counts / len(predictions_df)).map("{:.1%}".format)}))
        print(predictions_df.sort_values("risk_score", ascending=False).head(10)[
            ["seller_id", "risk_score", "risk_band"]].to_string(index=False))

        return predictions_df
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    generate_and_store_predictions()
