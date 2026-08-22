"""
src/preprocessing/pipeline.py

Role
----
Wraps final cleaning + scaling + encoding into a single scikit-learn
ColumnTransformer. Critical reason this exists: the EXACT SAME
transformation must run at training time and at prediction time. Fitting
this once and saving it to artifacts/ guarantees that — a seller scored by
the API is transformed identically to how the training data was
transformed. No "works in notebook, breaks in production" drift.

The transformer is fitted on the TRAINING SLICE ONLY. Fitting the scaler
or the imputer on the full table would leak test-period statistics (the
median, the mean, the variance) into training — a subtle but real form of
leakage that survives even a correct row-level split.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

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
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])


def fit_and_save_preprocessor(df: pd.DataFrame, save_path: str | None = None) -> ColumnTransformer:
    """
    Fit the preprocessor on the TRAINING feature table and persist it to disk.
    Callers must pass the training slice, never the full table.
    """
    try:
        cfg = load_config()
        save_path = save_path or (get_project_root() / cfg["paths"]["preprocessor"])

        numeric_cols, categorical_cols = get_feature_columns(df)
        preprocessor = build_preprocessor(numeric_cols, categorical_cols)
        preprocessor.fit(df[numeric_cols + categorical_cols])

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"preprocessor": preprocessor,
             "numeric_cols": numeric_cols,
             "categorical_cols": categorical_cols,
             "feature_names_out": _feature_names_out(preprocessor, numeric_cols, categorical_cols)},
            save_path,
        )
        logger.info(f"Preprocessor fitted on {len(numeric_cols)} numeric + "
                    f"{len(categorical_cols)} categorical columns, saved to {save_path}")
        return preprocessor
    except Exception as e:
        raise SentrixException(e, sys)


def _feature_names_out(preprocessor, numeric_cols: list[str],
                       categorical_cols: list[str]) -> list[str]:
    """
    Column names of the transformed matrix, in matrix order. One-hot
    encoding changes both the count and the order of columns, so SHAP
    output would otherwise map to 'feature_17' instead of a real name.
    """
    try:
        return [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    except Exception:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encode"]
        return numeric_cols + list(cat_encoder.get_feature_names_out(categorical_cols))


def load_preprocessor(path: str | None = None) -> dict:
    """Load the fitted preprocessor bundle (preprocessor + the column lists it expects)."""
    try:
        cfg = load_config()
        path = path or (get_project_root() / cfg["paths"]["preprocessor"])
        bundle = joblib.load(path)
        if "feature_names_out" not in bundle:   # bundles saved by an older run
            bundle["feature_names_out"] = _feature_names_out(
                bundle["preprocessor"], bundle["numeric_cols"], bundle["categorical_cols"]
            )
        logger.info(f"Preprocessor loaded from {path}")
        return bundle
    except Exception as e:
        raise SentrixException(e, sys)


def transform(df: pd.DataFrame, bundle: dict) -> np.ndarray:
    """Apply an already-fitted preprocessor bundle to new data (training or inference)."""
    try:
        cols = bundle["numeric_cols"] + bundle["categorical_cols"]
        out = bundle["preprocessor"].transform(df[cols])
        return out.toarray() if sp.issparse(out) else np.asarray(out)
    except Exception as e:
        raise SentrixException(e, sys)


def transform_to_frame(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Same transformation, returned as a DataFrame that keeps seller_id,
    as_of_date and the label alongside the scaled feature columns.

    The LSTM needs this: it groups rows by seller and slices them into
    sequences, so it cannot consume a bare numpy matrix that has lost the
    seller and date columns. Before this existed the LSTM was fed RAW
    unscaled features while every other model got the fitted scaler —
    which made the model comparison meaningless (an unscaled
    days_since_last_late of 9999 saturates an LSTM's gates immediately).
    """
    try:
        X = transform(df, bundle)
        out = pd.DataFrame(X, index=df.index, columns=bundle["feature_names_out"])
        for col in NON_FEATURE_COLUMNS:
            if col in df.columns:
                out[col] = df[col].values
        return out
    except Exception as e:
        raise SentrixException(e, sys)
