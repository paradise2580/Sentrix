"""
src/evaluation/generate_predictions.py

Role
----
Closes the loop. Scores every recent ORDER with the winning model,
calibrates those scores, rolls them up to a per-seller risk, explains each
seller with aggregated SHAP, and writes {risk_score, risk_band,
top_features} into the predictions table the API and dashboard read.

Why the rollup is a MEAN and not a count
----------------------------------------
This is the design decision the whole project turns on, and it is the fix
for the failure that killed two earlier versions.

SENTRIX originally predicted a seller-level target directly, and the
ablation showed the model was a volume detector: a forest given only the
three order_count_* columns scored ROC-AUC 0.7317 against the full
47-feature model's 0.6494. The label — "any late order in the next 30
days" — was close to a deterministic function of how many orders a seller
shipped.

Predicting orders and summing the risk would reintroduce exactly that. A
seller shipping 200 parcels would top the queue permanently, not because
their parcels are risky but because they have more of them.

So a seller's risk is the MEAN predicted risk of their recent orders. That
is a rate. It cannot be inflated by shipping more, and it answers the
question an ops team actually asks: "is this seller's typical parcel more
likely than usual to miss its date?" The volume confound is closed by the
shape of the statistic rather than argued away in a README.

Sellers with fewer than `seller_rollup_min_orders` recent orders are not
ranked at all — a mean over two parcels is noise, and putting it in a
worklist wastes a reviewer's time.

Scores are written CALIBRATED
-----------------------------
The stored number is what a human sees and what any cost calculation
multiplies. A raw score from a model trained with class_weight="balanced"
is systematically inflated — fine for ranking, wrong to display as "38%
chance of being late". The isotonic calibrator fitted on the held-out
calibration block is applied before anything is stored.

Bands are assigned by RANK on the raw score — see
metrics.assign_risk_bands_by_quantile, and the note below on why isotonic
output cannot be banded directly.

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

# Model whose SHAP values explain the stored predictions when the winning
# model cannot be explained directly (a StackingClassifier has no single
# feature space).
EXPLAINER_PROXY = "xgboost"
UNEXPLAINABLE = {"ensemble", "lstm"}


def _recent_orders(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    The trailing window of orders, measured from the newest order in the
    table rather than from today.

    The Olist snapshot ends in 2018. Anchoring on the wall clock would
    return an empty frame and an empty dashboard; anchoring on the data
    keeps the demo honest about what period it is showing.
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
        # The sequence model is not part of the order-level default and has
        # no row-wise feature space to score a single parcel with.
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

        # --- calibrate at ORDER level --------------------------------------
        # The calibrator was fitted on held-out orders, so it belongs here,
        # before aggregation. Calibrating a mean of raw scores instead would
        # apply an order-level mapping to a quantity it was never fitted on.
        calibrator_path = eval_dir / "calibrator.joblib"
        if calibrator_path.exists():
            cal_order = apply_calibrator(joblib.load(calibrator_path), raw_order)
            logger.info(f"Applied isotonic calibration: mean order score "
                        f"{raw_order.mean():.4f} -> {cal_order.mean():.4f}")
        else:
            cal_order = raw_order
            logger.warning("No calibrator found — storing RAW scores. Run "
                           "src.evaluation.run_evaluation first.")

        # --- SHAP at order level, averaged per seller ----------------------
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

        # --- roll up to sellers --------------------------------------------
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

        # Mean signed SHAP per seller: what drives THIS seller's typical parcel.
        shap_df = pd.DataFrame(shap_matrix, columns=feature_names)
        shap_df["seller_id"] = scored["seller_id"].values
        seller_shap = shap_df.groupby("seller_id").mean().loc[seller.index]

        # --- band on the RAW score, display the CALIBRATED one --------------
        #
        # Isotonic regression is a step function: every raw score inside a bin
        # maps to one output value, so calibrated scores collapse onto a few
        # dozen distinct numbers. Quantile cuts then land on large ties and
        # push everything at the boundary into the higher band — the top band
        # came out 5.5% instead of 5%, and "medium" swallowed 41% of the
        # population instead of 30%.
        #
        # Calibration is monotone, so banding on the raw score changes no
        # ordering. It only restores the resolution calibration flattened.
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
