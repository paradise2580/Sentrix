"""
Airflow DAG: refresh external signals -> rebuild features -> check drift and
regenerate predictions -> re-index RAG.

Airflow needs its own environment (it pins SQLAlchemy < 2.0, this project
uses 2.x), so every task runs a SENTRIX script via BashOperator instead of
importing project code.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# Path to the SENTRIX project root as seen from wherever the Airflow
# worker runs (adjust for your deployment — e.g. a mounted volume path
# in a container, or an absolute path on an EC2 host).
PROJECT_ROOT = "/opt/sentrix"
PYTHON_BIN = f"{PROJECT_ROOT}/venv/bin/python"

default_args = {
    "owner": "sentrix",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sentrix_refresh",
    description="Refresh SENTRIX signals, features, and drift checks on a schedule.",
    default_args=default_args,
    schedule=timedelta(hours=6),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["sentrix", "ml-pipeline"],
) as dag:

    refresh_signals = BashOperator(
        task_id="refresh_external_signals",
        bash_command=(
            f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.ingestion.synthetic_signals"
        ),
    )

    rebuild_features = BashOperator(
        task_id="rebuild_features",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} scripts/build_features.py",
    )

    check_drift = BashOperator(
        task_id="check_drift",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.monitoring.drift",
    )

    regenerate_predictions = BashOperator(
        task_id="regenerate_predictions",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.evaluation.generate_predictions",
    )

    reindex_rag = BashOperator(
        task_id="reindex_rag_knowledge_base",
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_BIN} -m src.rag.indexer",
    )

    # refresh signals -> rebuild features -> drift check + rescore sellers,
    # then re-index the RAG knowledge base off the fresh predictions.
    refresh_signals >> rebuild_features >> [check_drift, regenerate_predictions]
    regenerate_predictions >> reindex_rag
