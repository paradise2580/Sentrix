"""
Scores recent orders with the winning model, calibrates the scores, rolls
them up to one risk score per seller, explains each seller with averaged
SHAP values, and writes the result to the predictions table.

A seller's risk is the MEAN risk of their recent orders, not the sum, so
shipping more orders does not make a seller look riskier. Sellers with too
few recent orders are not ranked.

Run with:
    python -m src.evaluation.generate_predictions
"""

import json
import sys

import joblib
import pandas as pd

from src.config_loader import load_config, get_project_root
from src.ingestion.loader import DataLoader
from src.preprocessing.pipeline import (
    load_preprocessor, transform, get_feature_columns,
)
from src.models.explainer import (
    build_explainer, get_readable_feature_names, explain_predictions,
)
from src.evaluation.calibration import apply_calibrator
from src.evaluation.metrics import assign_risk_bands_by_quantile
from src.logger import get_logger
from src.exception import SentrixException

logger = get_logger(__name__)

# SHAP stand-in for models that can't be explained directly.
EXPLAINER_PROXY = "xgboost"
UNEXPLAINABLE = {"ensemble", "lstm"}


def _recent_orders(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    Orders from the last `window_days`, counted back from the newest order
    in the data (the dataset ends in 2018, so not from today).
    """
    cutoff = df["as_of_date"].max() - pd.Timedelta(days=int(window_days))
    recent = df[df["as_of_date"] >= cutoff]
    logger.info(f"Scoring {len(recent):,} orders from the trailing "
                f"{window_days}d ({cutoff.date()} to {df['as_of_date'].max().date()})")
    return recent


def _load_winner(cfg: dict, eval_dir):
    summary = joblib.load(eval_dir / "best_model_summary.joblib")
    name = summary["best_model"]
    if name == "lstm":
        # The LSTM can't score single orders, so fall back to the proxy.
        logger.warning("Evaluation winner was the sequence model; scoring with "
                       f"{EXPLAINER_PROXY} instead, which scores rows directly")
        name = EXPLAINER_PROXY
    model = joblib.load(get_project_root() / cfg["paths"]["models"] / f"{name}.joblib")
    logger.info(f"Scoring with the evaluation winner: {name}")
    return model, name


def generate_and_store_predictions() -> pd.DataFrame:
    try:
        cfg = load_config()
        oc = cfg.get("order_model", {})
        eval_dir = get_project_root() / "artifacts" / "evaluation"

        features_path = get_project_root() / cfg["paths"]["data_processed"] / "features.csv"
        df = pd.read_csv(features_path, parse_dates=["as_of_date"])

        recent = _recent_orders(df, oc.get("seller_rollup_window", 90)).reset_index(drop=True)
        if recent.empty:
            raise ValueError("No orders in the rollup window — nothing to score.")

        model, best_model_name = _load_winner(cfg, eval_dir)
        bundle = load_preprocessor()

        X = transform(recent, bundle)
        raw_order = model.predict_proba(X)[:, 1]

        # Calibrate per order, before averaging: the calibrator was fitted
        # on orders.
        calibrator_path = eval_dir / "calibrator.joblib"
        if calibrator_path.exists():
            cal_order = apply_calibrator(joblib.load(calibrator_path), raw_order)
            logger.info(f"Applied isotonic calibration: mean order score "
                        f"{raw_order.mean():.4f} -> {cal_order.mean():.4f}")
        else:
            cal_order = raw_order
            logger.warning("No calibrator found — storing RAW scores. Run "
                           "src.evaluation.run_evaluation first.")

        # SHAP per order, averaged per seller below.
        explain_name = EXPLAINER_PROXY if best_model_name in UNEXPLAINABLE else best_model_name
        if explain_name != best_model_name:
            logger.info(f"{best_model_name} is not directly explainable; "
                        f"using {EXPLAINER_PROXY} for SHAP attributions")
        explain_model = joblib.load(
            get_project_root() / cfg["paths"]["models"] / f"{explain_name}.joblib")

        explainer = build_explainer(explain_model, background=X)
        shap_matrix = explain_predictions(explainer, X)

        numeric_cols, categorical_cols = get_feature_columns(df)
        feature_names = get_readable_feature_names(
            numeric_cols, categorical_cols, bundle["preprocessor"])
        if shap_matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"SHAP returned {shap_matrix.shape[1]} columns for "
                f"{len(feature_names)} feature names — the preprocessor and the "
                f"explained model disagree about the feature space."
            )

        # Roll up to sellers.
        scored = pd.DataFrame({
            "seller_id": recent["seller_id"].astype(str).values,
            "raw": raw_order,
            "calibrated": cal_order,
        })
        grouped = scored.groupby("seller_id")
        seller = grouped.agg(raw=("raw", "mean"),
                             risk_score=("calibrated", "mean"),
                             n_orders=("raw", "size"))

        min_orders = oc.get("seller_rollup_min_orders", 5)
        before = len(seller)
        seller = seller[seller["n_orders"] >= min_orders]
        logger.info(f"Ranked {len(seller):,} sellers with >= {min_orders} recent orders "
                    f"({before - len(seller):,} dropped as too thin to rank)")
        if seller.empty:
            raise ValueError(
                f"No seller has {min_orders} orders in the rollup window. Widen "
                f"order_model.seller_rollup_window or lower seller_rollup_min_orders."
            )

        # Mean SHAP per seller: what drives this seller's typical order.
        shap_df = pd.DataFrame(shap_matrix, columns=feature_names)
        shap_df["seller_id"] = scored["seller_id"].values
        seller_shap = shap_df.groupby("seller_id").mean().loc[seller.index]

        # Band on the raw score, display the calibrated one. Isotonic output
        # has many ties, which skews quantile cuts; calibration keeps the
        # order, so banding on raw scores gives the same ranking.
        bands = assign_risk_bands_by_quantile(
            seller["raw"].values, cfg["model"]["risk_band_quantiles"])

        model_label = (best_model_name if explain_name == best_model_name
                       else f"{best_model_name} (SHAP via {explain_name})")

        records = []
        for i, (seller_id, row) in enumerate(seller.iterrows()):
            contributions = seller_shap.loc[seller_id]
            top = contributions.reindex(
                contributions.abs().sort_values(ascending=False).index).head(5)
            records.append({
                "seller_id": seller_id,          # real Olist IDs are hash strings
                "risk_score": round(float(row["risk_score"]), 4),
                "risk_band": bands[i],
                "model_name": model_label,
                "top_features": json.dumps(
                    [{"feature": f, "contribution": round(float(v), 6)}
                     for f, v in top.items()]),
            })

        predictions_df = (pd.DataFrame(records)
                          .assign(_raw=seller["raw"].values)
                          .sort_values("_raw", ascending=False)
                          .drop(columns="_raw")
                          .reset_index(drop=True))

        loader = DataLoader()
        loader.truncate_table(cfg["mysql"]["tables"]["predictions"])
        loader.write_df(predictions_df, cfg["mysql"]["tables"]["predictions"],
                        if_exists="append")
        logger.info(f"Wrote {len(predictions_df):,} calibrated seller predictions to MySQL")

        counts = predictions_df["risk_band"].value_counts()
        print("\nRISK BANDS")
        print(pd.DataFrame({"sellers": counts,
                            "share": (counts / len(predictions_df)).map("{:.1%}".format)}))
        print(f"\nOrders scored: {len(recent):,} | sellers ranked: {len(predictions_df):,} "
              f"| mean seller risk: {predictions_df['risk_score'].mean():.3f}")
        print("\nTOP 10 SELLERS BY MEAN ORDER RISK")
        print(predictions_df.head(10)[
            ["seller_id", "risk_score", "risk_band"]].to_string(index=False))

        return predictions_df
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    generate_and_store_predictions()
