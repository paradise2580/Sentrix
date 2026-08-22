"""
src/evaluation/generate_predictions.py

Role
----
Closes the loop from Phase 1 through Phase 5: for every supplier's most
recent as_of_date snapshot, score it with the best model, explain the
score with SHAP, and write {risk_score, risk_band, top_features} into the
MySQL predictions table — the exact table Phase 1 designed for this.

This is what Phase 8's API and Phase 9's dashboard read from, so a
prediction only needs to be computed once here rather than re-run live
on every dashboard page load.

Run with:
    python -m src.evaluation.generate_predictions
"""

import json
import joblib
import pandas as pd

from src.config_loader import load_config, get_project_root
from src.ingestion.loader import DataLoader
from src.preprocessing.pipeline import load_preprocessor, transform, get_feature_columns
from src.models.explainer import build_explainer, get_readable_feature_names, explain_single_prediction
from src.evaluation.metrics import assign_risk_band
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def generate_and_store_predictions() -> pd.DataFrame:
    try:
        cfg = load_config()

        # Use the model Phase 5's evaluation crowned the winner
        eval_dir = get_project_root() / "artifacts" / "evaluation"
        best_summary = joblib.load(eval_dir / "best_model_summary.joblib")
        best_model_name = best_summary["best_model"]
        logger.info(f"Using best model from evaluation: {best_model_name}")

        model = joblib.load(get_project_root() / cfg["paths"]["models"] / f"{best_model_name}.joblib")

        features_path = get_project_root() / cfg["paths"]["data_processed"] / "features.csv"
        df = pd.read_csv(features_path, parse_dates=["as_of_date"])

        # Only need each supplier's MOST RECENT snapshot — that's "current risk"
        latest = df.sort_values("as_of_date").groupby("seller_id").tail(1).reset_index(drop=True)
        logger.info(f"Scoring latest snapshot for {len(latest)} sellers")

        bundle = load_preprocessor()
        X_latest = transform(latest, bundle)

        probs = model.predict_proba(X_latest)[:, 1]

        # SHAP cannot explain a StackingClassifier directly, so the standalone
        # XGBoost stands in for the ensemble. Every other model type is
        # explained natively by build_explainer (tree / linear / kernel).
        explain_model = model
        if best_model_name == "ensemble":
            explain_model = joblib.load(get_project_root() / cfg["paths"]["models"] / "xgboost.joblib")

        explainer = build_explainer(explain_model, background=X_latest)
        numeric_cols, categorical_cols = get_feature_columns(df)
        feature_names = get_readable_feature_names(numeric_cols, categorical_cols, bundle["preprocessor"])

        bands = cfg["model"]["risk_bands"]
        records = []
        for i, (_, row) in enumerate(latest.iterrows()):
            prob = float(probs[i])
            top_features = explain_single_prediction(explainer, X_latest[i], feature_names, top_n=5)

            records.append({
                "seller_id": str(row["seller_id"]),  # real Olist IDs are hash strings, not ints
                "risk_score": round(prob, 4),
                "risk_band": assign_risk_band(prob, bands),
                "model_name": best_model_name,
                "top_features": json.dumps(top_features),
            })

        predictions_df = pd.DataFrame(records)

        loader = DataLoader()
        loader.truncate_table(cfg["mysql"]["tables"]["predictions"])
        loader.write_df(predictions_df, cfg["mysql"]["tables"]["predictions"], if_exists="append")

        logger.info(f"Wrote {len(predictions_df)} predictions to MySQL")
        print(predictions_df["risk_band"].value_counts())
        print(predictions_df.sort_values("risk_score", ascending=False).head(10)[
            ["seller_id", "risk_score", "risk_band"]
        ].to_string(index=False))

        return predictions_df
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    generate_and_store_predictions()
