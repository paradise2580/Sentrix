"""
pipelines/sentrix_dag.py

Role
----
Schedules SENTRIX's data pipeline to refresh automatically — the
production analogue of running scripts/ manually. On a real deployment
this DAG runs every 6 hours: fetch fresh signals, rebuild features,
retrain if needed, check for drift.

IMPORTANT — environment isolation
-----------------------------------
Airflow hard-pins sqlalchemy<2.0, which conflicts directly with this
project's SQLAlchemy 2.0 usage (src/ingestion/db.py) and with modern
versions of FastAPI/MLflow's own dependencies. This is a well-known,
real constraint of Airflow's dependency footprint — the correct
production pattern (and the one this DAG is written for) is:

    Airflow runs in its OWN isolated environment (a dedicated virtualenv,
    a separate Docker container, or a managed service like MWAA / Cloud
    Composer / Astronomer) and triggers SENTRIX's actual pipeline code via
    subprocess calls, a DockerOperator/KubernetesPodOperator, or an HTTP
    call to the FastAPI service — NEVER via direct Python imports sharing
    one environment with the serving stack.

This DAG is written that way: every task shells out to a SENTRIX script
via BashOperator rather than importing SENTRIX modules directly into
Airflow's own Python process. That keeps the two dependency trees fully
separate, which is the actual fix for the conflict above.
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
