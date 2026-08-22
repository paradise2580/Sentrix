"""
src/evaluation/run_evaluation.py

Role
----
Loads every model artifact saved in Phase 4, scores each on the identical
chronological test set, and ranks them by PR-AUC. The winner is what
Phase 6 registers as "Production" in MLflow.

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
from src.preprocessing.pipeline import get_feature_columns
from src.evaluation.metrics import compare_models, find_optimal_threshold, evaluate_model
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def load_sklearn_model(name: str):
    cfg = load_config()
    path = get_project_root() / cfg["paths"]["models"] / f"{name}.joblib"
    return joblib.load(path)


def load_lstm_model(n_features: int):
    cfg = load_config()["model"]["lstm"]
    models_dir = get_project_root() / load_config()["paths"]["models"]

    model = DisruptionLSTM(n_features, cfg["hidden_size"], cfg["num_layers"])
    model.load_state_dict(torch.load(models_dir / "lstm.pt"))
    meta = joblib.load(models_dir / "lstm_meta.joblib")
    return model, meta


def run_full_evaluation() -> pd.DataFrame:
    try:
        data = prepare_data()
        y_test = data["y_test"]

        results = {}

        # --- sklearn-compatible models: predict_proba on the shared X_test ---
        for name in ["logistic_regression", "random_forest", "xgboost", "lightgbm", "ensemble"]:
            model = load_sklearn_model(name)
            y_prob = model.predict_proba(data["X_test"])[:, 1]
            results[name] = {"y_true": y_test, "y_prob": y_prob}
            logger.info(f"Loaded and scored: {name}")

        # --- LSTM: separate sequence-based prediction path ---
        numeric_cols, _ = get_feature_columns(data["df"])
        lstm_model, lstm_meta = load_lstm_model(len(numeric_cols))
        lstm_probs = predict_lstm(lstm_model, data["test_df"], lstm_meta)

        lstm_labels = []
        target = load_config()["model"]["target_column"]
        for _, group in data["test_df"].sort_values(["seller_id", "as_of_date"]).groupby("seller_id"):
            lstm_labels.extend(group[target].values[lstm_meta["seq_len"]:])
        results["lstm"] = {"y_true": np.array(lstm_labels), "y_prob": lstm_probs}
        logger.info("Loaded and scored: lstm")

        comparison = compare_models(results)
        print("\n" + "=" * 70)
        print("MODEL COMPARISON — ranked by PR-AUC")
        print("=" * 70)
        print(comparison.to_string(index=False))

        best_model_name = comparison.iloc[0]["model"]
        best_result = results[best_model_name]
        threshold_info = find_optimal_threshold(best_result["y_true"], best_result["y_prob"])

        print(f"\nBest model: {best_model_name}")
        print(f"Optimal decision threshold: {threshold_info['optimal_threshold']:.4f}")
        print(f"  -> Precision: {threshold_info['precision_at_optimal']:.4f}")
        print(f"  -> Recall:    {threshold_info['recall_at_optimal']:.4f}")
        print(f"  -> F1:        {threshold_info['f1_at_optimal']:.4f}")

        full_metrics = evaluate_model(
            best_result["y_true"], best_result["y_prob"], threshold_info["optimal_threshold"]
        )
        print(f"\nConfusion matrix at optimal threshold:")
        print(f"  TN={full_metrics['true_negatives']}  FP={full_metrics['false_positives']}")
        print(f"  FN={full_metrics['false_negatives']}  TP={full_metrics['true_positives']}")

        # Persist for Phase 6 (MLflow registration) and Phase 11 (CI/CD model gate)
        eval_dir = get_project_root() / "artifacts" / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(eval_dir / "model_comparison.csv", index=False)
        joblib.dump(
            {"best_model": best_model_name, "threshold": threshold_info["optimal_threshold"],
             "metrics": full_metrics},
            eval_dir / "best_model_summary.joblib",
        )
        logger.info(f"Evaluation artifacts saved to {eval_dir}")

        return comparison
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    run_full_evaluation()
