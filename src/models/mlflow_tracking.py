"""
Logs every model's params, metrics and artifact to MLflow, and registers the
best model as "Production" in the Model Registry.
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


_EXPERIMENT = "sentrix-disruption-prediction"


def _set_tracking_uri():
    """
    Point both the tracking DB and the artifact store at
    paths.mlflow_tracking_uri, so nothing is written to ./mlruns.
    """
    cfg = load_config()
    tracking_dir = (get_project_root() / cfg["paths"]["mlflow_tracking_uri"]).resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")

    if mlflow.get_experiment_by_name(_EXPERIMENT) is None:
        mlflow.create_experiment(
            _EXPERIMENT, artifact_location=tracking_dir.as_uri(),
        )
    mlflow.set_experiment(_EXPERIMENT)


def log_model_run(model_name: str, model_obj, params: dict, metrics: dict,
                  is_pytorch: bool = False, input_example=None) -> str:
    """Log one model as an MLflow run. Returns the run_id."""
    try:
        _set_tracking_uri()
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and k != "confusion_matrix"
            })
            if is_pytorch:
                # The PyTorch flavour needs an example input to trace the model.
                mlflow.pytorch.log_model(
                    model_obj, artifact_path="model",
                    input_example=input_example, serialization_format="pickle",
                )
            else:
                # Plain pickle: skops rejects XGBoost/LightGBM internals.
                mlflow.sklearn.log_model(
                    model_obj, artifact_path="model",
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
                )

            logger.info(f"Logged run for '{model_name}' — run_id={run.info.run_id}")
            return run.info.run_id
    except Exception as e:
        raise SentrixException(e, sys)


def log_all_trained_models() -> dict:
    """Log every saved model as its own MLflow run, using the evaluation table."""
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
            "ensemble": {"base_models": "logreg+rf+xgb",
                         "cv": f"TimeSeriesSplit({cfg['model']['ensemble']['n_splits']})"},
            "lstm": cfg["model"]["lstm"],
        }

        run_ids = {}
        for _, row in comparison.iterrows():
            name = row["model"]
            metrics = row.drop("model").to_dict()

            if name == "lstm":
                import torch
                from src.models.deep import DisruptionLSTM
                # Architecture comes from saved metadata, not config.yaml.
                meta = joblib.load(models_dir / "lstm_meta.joblib")
                model_obj = DisruptionLSTM(len(meta["feature_cols"]),
                                           meta["hidden_size"], meta["num_layers"])
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
    """Register the best model by PR-AUC and move it to the 'Production' stage."""
    try:
        _set_tracking_uri()
        load_config()
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
