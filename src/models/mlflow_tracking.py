"""
src/models/mlflow_tracking.py

Role
----
Every training run's parameters, metrics, and model artifact get logged
to MLflow — so instead of remembering "which XGBoost run scored best,"
there's a browsable, permanent record. The single best model (by PR-AUC,
per Phase 5) is then promoted to the "Production" stage in MLflow's Model
Registry, which is what Phase 8's API loads at startup — one source of
truth for what's actually live.
"""

import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch

from src.config_loader import load_config, get_project_root
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

_REGISTRY_NAME = "sentrix-risk-model"


def _set_tracking_uri():
    cfg = load_config()
    tracking_dir = get_project_root() / cfg["paths"]["mlflow_tracking_uri"]
    tracking_dir.mkdir(parents=True, exist_ok=True)
    db_path = tracking_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("sentrix-disruption-prediction")


def log_model_run(model_name: str, model_obj, params: dict, metrics: dict,
                   is_pytorch: bool = False, input_example=None) -> str:
    """Log one model's params, metrics, and artifact as an MLflow run. Returns the run_id."""
    try:
        _set_tracking_uri()
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "confusion_matrix"
            })
            if is_pytorch:
                # MLflow's default 'pt2' traced-graph format requires a concrete
                # example input to trace the forward pass through.
                mlflow.pytorch.log_model(
                    model_obj, artifact_path="model",
                    input_example=input_example, serialization_format="pickle",
                )
            else:
                # Standard pickle serialization — skops (MLflow's newer default) flags
                # XGBoost/LightGBM booster internals as "untrusted", which is a sensible
                # default for THIRD-PARTY models but unnecessary friction for models we
                # trained ourselves in this same pipeline.
                mlflow.sklearn.log_model(
                    model_obj, artifact_path="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                )

            logger.info(f"Logged run for '{model_name}' — run_id={run.info.run_id}")
            return run.info.run_id
    except Exception as e:
        raise SentrixException(e, sys)


def log_all_trained_models() -> dict:
    """
    Reads Phase 4's saved model artifacts + Phase 5's comparison table and
    logs every model as its own MLflow run, so the full picture (not just
    the winner) is browsable in the MLflow UI.
    """
    try:
        cfg = load_config()
        models_dir = get_project_root() / cfg["paths"]["models"]
        eval_dir = get_project_root() / "artifacts" / "evaluation"

        import pandas as pd
        comparison = pd.read_csv(eval_dir / "model_comparison.csv")

        param_lookup = {
            "logistic_regression": cfg["model"]["logistic_regression"],
            "random_forest": cfg["model"]["random_forest"],
            "xgboost": joblib.load(models_dir / "xgboost_best_params.joblib")
                if (models_dir / "xgboost_best_params.joblib").exists()
                else cfg["model"]["xgboost"],
            "lightgbm": cfg["model"]["lightgbm"],
            "ensemble": {"base_models": "logreg+rf+xgb", "cv": 3},
            "lstm": cfg["model"]["lstm"],
        }

        run_ids = {}
        for _, row in comparison.iterrows():
            name = row["model"]
            metrics = row.drop("model").to_dict()

            if name == "lstm":
                import torch
                from src.models.deep import DisruptionLSTM
                meta = joblib.load(models_dir / "lstm_meta.joblib")
                lstm_cfg = cfg["model"]["lstm"]
                model_obj = DisruptionLSTM(len(meta["feature_cols"]), lstm_cfg["hidden_size"], lstm_cfg["num_layers"])
                model_obj.load_state_dict(torch.load(models_dir / "lstm.pt"))
                example_input = torch.randn(1, meta["seq_len"], len(meta["feature_cols"]))
                run_id = log_model_run(name, model_obj, param_lookup[name], metrics,
                                        is_pytorch=True, input_example=example_input)
            else:
                model_obj = joblib.load(models_dir / f"{name}.joblib")
                run_id = log_model_run(name, model_obj, param_lookup[name], metrics, is_pytorch=False)

            run_ids[name] = run_id

        logger.info(f"Logged {len(run_ids)} model runs to MLflow")
        return run_ids
    except Exception as e:
        raise SentrixException(e, sys)


def register_best_model(run_ids: dict) -> None:
    """
    Registers the best model (per Phase 5's PR-AUC ranking) under
    _REGISTRY_NAME and transitions it to the 'Production' stage. This is
    the model Phase 8's API loads — one call, one source of truth.
    """
    try:
        _set_tracking_uri()
        cfg = load_config()
        eval_dir = get_project_root() / "artifacts" / "evaluation"
        best_summary = joblib.load(eval_dir / "best_model_summary.joblib")
        best_name = best_summary["best_model"]
        best_run_id = run_ids[best_name]

        model_uri = f"runs:/{best_run_id}/model"
        client = mlflow.MlflowClient()

        try:
            client.create_registered_model(_REGISTRY_NAME)
        except mlflow.exceptions.MlflowException:
            pass  # already exists

        mv = client.create_model_version(_REGISTRY_NAME, model_uri, best_run_id)
        client.transition_model_version_stage(
            name=_REGISTRY_NAME, version=mv.version, stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"Registered '{best_name}' (run {best_run_id}) as {_REGISTRY_NAME} "
                     f"v{mv.version} -> Production")
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    run_ids = log_all_trained_models()
    register_best_model(run_ids)
