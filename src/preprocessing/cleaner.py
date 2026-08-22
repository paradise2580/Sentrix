"""
src/preprocessing/cleaner.py

Role
----
Make raw data trustworthy before any feature is computed from it. Each
cleaning concern is its own function so a failure is easy to isolate —
e.g. "did outlier removal break this?" is answerable without re-reading
the whole pipeline.
"""

import pandas as pd
import numpy as np

from src.config_loader import load_config
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def handle_missing(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Impute missing numeric values; drop rows missing critical identifiers
    (seller_id, date columns) since those can't be safely imputed.
    """
    try:
        df = df.copy()
        id_like_cols = [c for c in df.columns if "id" in c.lower() or "date" in c.lower()]
        before = len(df)
        df = df.dropna(subset=[c for c in id_like_cols if c in df.columns])
        dropped = before - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} rows missing identifier/date columns")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                fill_value = df[col].median() if strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(fill_value)

        return df
    except Exception as e:
        raise SentrixException(e, sys)


def remove_outliers(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Clip numeric columns to the IQR-based bound defined in config.yaml."""
    try:
        cfg = load_config()
        multiplier = cfg["preprocessing"]["outlier_iqr_multiplier"]
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            df[col] = df[col].clip(lower, upper)

        return df
    except Exception as e:
        raise SentrixException(e, sys)


def fix_types(df: pd.DataFrame, date_cols: list[str] = None,
              category_cols: list[str] = None) -> pd.DataFrame:
    """Coerce date columns to datetime and categorical columns to category dtype."""
    try:
        df = df.copy()
        for col in (date_cols or []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        for col in (category_cols or []):
            if col in df.columns:
                df[col] = df[col].astype("category")
        return df
    except Exception as e:
        raise SentrixException(e, sys)


def drop_duplicates(df: pd.DataFrame, subset: list[str] = None) -> pd.DataFrame:
    """Remove exact or key-based duplicate rows."""
    try:
        before = len(df)
        df = df.drop_duplicates(subset=subset)
        removed = before - len(df)
        if removed:
            logger.info(f"Dropped {removed} duplicate rows")
        return df
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    sample = pd.DataFrame({
        "seller_id": [1, 1, 2, None],
        "value": [10, np.nan, 500, 12],
    })
    print("Before:\n", sample)
    cleaned = handle_missing(sample)
    print("After handle_missing:\n", cleaned)
