"""
src/monitoring/drift.py

Role
----
Models silently decay when the real world shifts (a new disruption
pattern, a supplier base that's grown, a feature distribution that's
moved). This module compares a "reference" slice of the feature table
(what the model was trained on) against a "current" slice (recent data)
and flags data drift — the same check a production ML system runs on a
schedule (via the Airflow DAG) to know when retraining is needed.
"""

from pathlib import Path
import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset

from src.config_loader import load_config, get_project_root
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def split_reference_and_current(df: pd.DataFrame, date_col: str = "as_of_date",
                                 split_fraction: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the feature table chronologically: the earlier slice is
    'reference' (what a model would have been trained on), the later
    slice is 'current' (freshly arrived data to check for drift against).
    """
    df = df.sort_values(date_col)
    cutoff_idx = int(len(df) * split_fraction)
    cutoff_date = df.iloc[cutoff_idx][date_col]

    reference = df[df[date_col] < cutoff_date].copy()
    current = df[df[date_col] >= cutoff_date].copy()
    return reference, current


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                      feature_cols: list[str], output_path: str | None = None) -> dict:
    """
    Runs Evidently's DataDriftPreset comparing reference vs current on the
    given numeric feature columns, saves an HTML report, and returns a
    summary dict (which columns drifted, overall drift share).
    """
    try:
        cfg = load_config()
        output_path = output_path or (get_project_root() / cfg["monitoring"]["drift_report_path"])
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        definition = DataDefinition(numerical_columns=feature_cols)
        ref_dataset = Dataset.from_pandas(reference[feature_cols], data_definition=definition)
        cur_dataset = Dataset.from_pandas(current[feature_cols], data_definition=definition)

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(current_data=cur_dataset, reference_data=ref_dataset)
        snapshot.save_html(str(output_path))

        result_dict = snapshot.dict()
        logger.info(f"Drift report saved to {output_path}")

        return {
            "output_path": str(output_path),
            "reference_rows": len(reference),
            "current_rows": len(current),
            "raw_result": result_dict,
        }
    except Exception as e:
        raise SentrixException(e, sys)


def check_drift_from_feature_table(feature_table_path: str | None = None) -> dict:
    """
    End-to-end: load the saved feature table, split it chronologically
    into reference/current, run the drift report. This is the function
    the Airflow DAG calls on its schedule.
    """
    try:
        cfg = load_config()
        path = feature_table_path or (get_project_root() / cfg["paths"]["data_processed"] / "features.csv")
        df = pd.read_csv(path, parse_dates=["as_of_date"])

        numeric_cols, _ = _get_numeric_feature_cols(df)
        reference, current = split_reference_and_current(df)

        logger.info(f"Drift check: reference={len(reference)} rows, current={len(current)} rows, "
                     f"{len(numeric_cols)} features compared")

        return run_drift_report(reference, current, numeric_cols)
    except Exception as e:
        raise SentrixException(e, sys)


def _get_numeric_feature_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    from src.preprocessing.pipeline import get_feature_columns
    return get_feature_columns(df)


def summarize_drift(raw_result: dict) -> dict:
    """
    Extracts a clean, human-readable summary from Evidently's raw snapshot
    dict: how many features drifted, and which ones, ranked by severity.
    This is what an Airflow task would check to decide whether
    to trigger a retraining alert.
    """
    metrics = raw_result.get("metrics", [])

    overall = next(
        (m for m in metrics if m["metric_name"].startswith("DriftedColumnsCount")), None
    )
    per_column = [
        {
            "column": m["config"]["column"],
            "method": m["config"]["method"],
            "drift_value": m["value"],
            "threshold": m["config"]["threshold"],
            "drifted": m["value"] > m["config"]["threshold"],
        }
        for m in metrics if m["metric_name"].startswith("ValueDrift")
    ]
    per_column.sort(key=lambda x: x["drift_value"], reverse=True)

    return {
        "drifted_column_count": int(overall["value"]["count"]) if overall else None,
        "drifted_column_share": overall["value"]["share"] if overall else None,
        "columns_ranked_by_drift": per_column,
    }


if __name__ == "__main__":
    result = check_drift_from_feature_table()
    print(f"Drift report saved to: {result['output_path']}")
    print(f"Reference rows: {result['reference_rows']}, Current rows: {result['current_rows']}")

    summary = summarize_drift(result["raw_result"])
    print(f"\nDrifted columns: {summary['drifted_column_count']} "
          f"({summary['drifted_column_share']:.1%} of all features)")
    print("\nTop 5 by drift magnitude:")
    for col in summary["columns_ranked_by_drift"][:5]:
        flag = "DRIFTED" if col["drifted"] else "ok"
        print(f"  [{flag:8s}] {col['column']:35s} {col['method']:30s} = {col['drift_value']:.4f}")
