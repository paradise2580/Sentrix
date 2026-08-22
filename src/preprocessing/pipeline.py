"""
src/preprocessing/pipeline.py

Role
----
Wraps final cleaning + scaling + encoding into a single scikit-learn
ColumnTransformer. Critical reason this exists: the EXACT SAME
transformation must run at training time and at prediction time. Fitting
this once and saving it to artifacts/ guarantees that — a new supplier
scored by the API in Phase 8 is transformed identically to how the
training data was transformed. No "works in notebook, breaks in
production" drift.
"""

from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config_loader import load_config, get_project_root
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

# Columns that are identifiers or the label — never fed into the transformer
NON_FEATURE_COLUMNS = ["seller_id", "as_of_date", "disruption_next_30d"]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric vs categorical."""
    candidate_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    numeric_cols = df[candidate_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in candidate_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Construct the ColumnTransformer — impute+scale numerics, impute+one-hot categoricals."""
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def fit_and_save_preprocessor(df: pd.DataFrame, save_path: str | None = None) -> ColumnTransformer:
    """Fit the preprocessor on the training feature table and persist it to disk."""
    try:
        cfg = load_config()
        save_path = save_path or (get_project_root() / cfg["paths"]["preprocessor"])

        numeric_cols, categorical_cols = get_feature_columns(df)
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        preprocessor.fit(df[numeric_cols + categorical_cols])

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"preprocessor": preprocessor, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols},
            save_path,
        )
        logger.info(f"Preprocessor fitted on {len(numeric_cols)} numeric + "
                     f"{len(categorical_cols)} categorical columns, saved to {save_path}")
        return preprocessor
    except Exception as e:
        raise SentrixException(e, sys)


def load_preprocessor(path: str | None = None) -> dict:
    """Load the fitted preprocessor bundle (preprocessor + the column lists it expects)."""
    try:
        cfg = load_config()
        path = path or (get_project_root() / cfg["paths"]["preprocessor"])
        bundle = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return bundle
    except Exception as e:
        raise SentrixException(e, sys)


def transform(df: pd.DataFrame, bundle: dict):
    """Apply an already-fitted preprocessor bundle to new data (training or inference)."""
    try:
        cols = bundle["numeric_cols"] + bundle["categorical_cols"]
        return bundle["preprocessor"].transform(df[cols])
    except Exception as e:
        raise SentrixException(e, sys)
