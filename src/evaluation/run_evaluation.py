"""
Scores every saved model on one shared test set, ranks them by PR-AUC,
calibrates the winner on the CALIB block, and picks a cost-based threshold.

All models must be scored on the same rows, because PR-AUC depends on the
base rate.

Run with:
    python -m src.evaluation.run_evaluation
"""

import joblib
import numpy as np
import pandas as pd
import torch

from src.config_loader import load_config, get_project_root
from src.models.trainer import prepare_data
from src.models.deep import DisruptionLSTM, predict_lstm
from src.preprocessing.pipeline import transform_to_frame
from src.evaluation.metrics import (
    compare_models, evaluate_model, find_optimal_threshold,
    find_cost_optimal_threshold, DEFAULT_K_VALUES,
)
from src.evaluation.calibration import (
    fit_calibrator, apply_calibrator, calibration_report,
)
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

SKLEARN_MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "ensemble"]


def load_sklearn_model(name: str):
    cfg = load_config()
    return joblib.load(get_project_root() / cfg["paths"]["models"] / f"{name}.joblib")


def load_lstm_model():
    """Rebuild the LSTM from its saved metadata, so architecture can't drift."""
    models_dir = get_project_root() / load_config()["paths"]["models"]
    meta = joblib.load(models_dir / "lstm_meta.joblib")

    # Refuse old checkpoints without saved architecture metadata.
    missing = [k for k in ("hidden_size", "num_layers", "feature_cols") if k not in meta]
    if missing:
        raise ValueError(
            f"lstm_meta.joblib is missing {missing} — it predates the current "
            f"training code. Re-run: python -m src.models.trainer --stages lstm"
        )

    model = DisruptionLSTM(
        n_features=len(meta["feature_cols"]),
        hidden_size=meta["hidden_size"],
        num_layers=meta["num_layers"],
    )
    model.load_state_dict(torch.load(models_dir / "lstm.pt", map_location="cpu"))
    return model, meta


def _score_block(block_df: pd.DataFrame, X_block: np.ndarray, y_block: np.ndarray,
                 bundle: dict, lstm_model, lstm_meta) -> dict:
    """Score every model on one block. Returns {"y": ..., "probs": {name: array}}."""
    if lstm_model is None:
        # No LSTM: every model scores the whole block.
        probs = {}
        pos = np.arange(len(block_df))
    else:
        seq_frame = transform_to_frame(block_df, bundle).sort_values(["seller_id", "as_of_date"])
        p_lstm, row_index = predict_lstm(lstm_model, seq_frame, lstm_meta)

        if len(row_index) == 0:
            raise ValueError("The LSTM scored zero rows in this block — the block is empty.")

        pos = block_df.index.get_indexer(row_index)
        if (pos < 0).any():
            raise ValueError("LSTM row index does not align with the block frame.")

        probs = {"lstm": p_lstm}
    for name in SKLEARN_MODELS:
        model = load_sklearn_model(name)
        probs[name] = model.predict_proba(X_block[pos])[:, 1]

    return {"y": y_block[pos], "probs": probs, "n_full_block": len(block_df)}


def run_full_evaluation() -> pd.DataFrame:
    try:
        cfg = load_config()
        data = prepare_data()
        bundle = data["bundle"]

        # The LSTM is optional; it is evaluated only if its artifacts exist.
        try:
            lstm_model, lstm_meta = load_lstm_model()
            # Refuse a checkpoint trained on a different feature space.
            trained_on = list(lstm_meta.get("feature_cols", []))
            if trained_on != list(bundle["feature_names_out"]):
                logger.warning(
                    f"Stale sequence-model artifacts: trained on {len(trained_on)} "
                    f"features, current preprocessor emits "
                    f"{len(bundle['feature_names_out'])}. Skipping the LSTM."
                )
                lstm_model, lstm_meta = None, None
        except Exception as exc:                                    # noqa: BLE001
            logger.info(f"No sequence model evaluated ({exc.__class__.__name__}); "
                        f"scoring tabular models on the full block")
            lstm_model, lstm_meta = None, None

        # ---------------------------------------------------------- test block
        test = _score_block(data["test_df"], data["X_test"], data["y_test"],
                            bundle, lstm_model, lstm_meta)
        y_eval = test["y"]

        coverage = len(y_eval) / test["n_full_block"]
        logger.info(
            f"Shared evaluation set: {len(y_eval):,} of {test['n_full_block']:,} test rows "
            f"({coverage:.1%}), base rate {y_eval.mean():.2%}"
        )
        if coverage < 0.999:
            logger.warning(
                f"{test['n_full_block'] - len(y_eval):,} test rows went unscored — "
                f"expected full coverage with left-padded sequences."
            )

        results = {name: {"y_true": y_eval, "y_prob": p} for name, p in test["probs"].items()}
        comparison = compare_models(results)

        print("\n" + "=" * 96)
        print(f"MODEL COMPARISON — {len(y_eval):,} shared test rows, "
              f"base rate {y_eval.mean():.2%}, ranked by PR-AUC")
        print("=" * 96)
        print(comparison.to_string(index=False))

        best_model_name = comparison.iloc[0]["model"]
        p_raw = test["probs"][best_model_name]

        # --------------------------------------------------- calibrate the winner
        calib = _score_block(data["calib_df"], data["X_calib"], data["y_calib"],
                             bundle, lstm_model, lstm_meta)
        calibrator = fit_calibrator(calib["y"], calib["probs"][best_model_name],
                                    method=cfg["model"].get("calibration_method", "isotonic"))
        p_cal = apply_calibrator(calibrator, p_raw)
        cal_report = calibration_report(y_eval, p_raw, p_cal)

        print(f"\nCALIBRATION — {best_model_name}, isotonic, fitted on "
              f"{len(calib['y']):,} held-out rows")
        print(f"  Brier  {cal_report['brier_raw']:.4f} -> {cal_report['brier_calibrated']:.4f}")
        print(f"  ECE    {cal_report['ece_raw']:.4f} -> {cal_report['ece_calibrated']:.4f}")
        print(f"  Mean prediction {cal_report['mean_prediction_raw']:.4f} -> "
              f"{cal_report['mean_prediction_calibrated']:.4f} "
              f"(observed {cal_report['observed_base_rate']:.4f})")

        # ------------------------------------------------------- operating point
        f1_thresh = find_optimal_threshold(y_eval, p_cal)
        cost = find_cost_optimal_threshold(
            y_eval, p_cal,
            cost_fn=cfg["model"].get("cost_false_negative", 10.0),
            cost_fp=cfg["model"].get("cost_false_positive", 1.0),
        )

        print(f"\nOPERATING POINTS ({best_model_name}, calibrated)")
        print(f"  F1-optimal threshold   {f1_thresh['optimal_threshold']:.4f}  "
              f"P={f1_thresh['precision_at_optimal']:.3f} R={f1_thresh['recall_at_optimal']:.3f}")
        print(f"  Cost-optimal threshold {cost['cost_optimal_threshold']:.4f}  "
              f"(FN:FP = {cost['cost_fn']:.0f}:{cost['cost_fp']:.0f}) — "
              f"{cost['cost_reduction']:.1%} lower expected cost than not modelling at all")

        best_metrics = evaluate_model(y_eval, p_cal, cost["cost_optimal_threshold"])

        print("\nALERT CAPACITY — what an ops team actually gets")
        for k in DEFAULT_K_VALUES:
            pct = int(round(k * 100))
            print(f"  Review top {pct:>2}% of sellers -> catch "
                  f"{best_metrics[f'capture_at_{pct}pct']:.1%} of the sellers who go late "
                  f"({best_metrics[f'lift_at_{pct}pct']:.2f}x random)")

        print("\nConfusion matrix at the cost-optimal threshold:")
        print(f"  TN={best_metrics['true_negatives']}  FP={best_metrics['false_positives']}")
        print(f"  FN={best_metrics['false_negatives']}  TP={best_metrics['true_positives']}")

        # ------------------------------------------------------------ persist
        eval_dir = get_project_root() / "artifacts" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        comparison.to_csv(eval_dir / "model_comparison.csv", index=False)
        cost["curve"].to_csv(eval_dir / "cost_curve.csv", index=False)
        pd.DataFrame(cal_report["reliability_calibrated"]).to_csv(
            eval_dir / "reliability_calibrated.csv", index=False)
        pd.DataFrame(cal_report["reliability_raw"]).to_csv(
            eval_dir / "reliability_raw.csv", index=False)
        joblib.dump(calibrator, eval_dir / "calibrator.joblib")
        joblib.dump(
            {
                "best_model": best_model_name,
                "threshold": cost["cost_optimal_threshold"],
                "f1_optimal_threshold": f1_thresh["optimal_threshold"],
                "metrics": best_metrics,
                "calibration": {k: v for k, v in cal_report.items()
                                if not k.startswith("reliability")},
                "cost": {k: v for k, v in cost.items() if k != "curve"},
                "eval_rows": int(len(y_eval)),
                "eval_base_rate": float(y_eval.mean()),
            },
            eval_dir / "best_model_summary.joblib",
        )
        logger.info(f"Evaluation artifacts saved to {eval_dir}")

        return comparison
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    run_full_evaluation()
