"""tests/test_cleaner.py — unit tests for src/preprocessing/cleaner.py"""

import numpy as np
import pandas as pd

from src.preprocessing.cleaner import handle_missing, remove_outliers, fix_types, drop_duplicates


def test_handle_missing_fills_numeric_nulls():
    df = pd.DataFrame({"supplier_id": [1, 2, 3], "value": [10.0, np.nan, 30.0]})
    result = handle_missing(df)
    assert result["value"].isna().sum() == 0
    assert result["value"].iloc[1] == df["value"].median()


def test_handle_missing_drops_rows_missing_identifiers():
    df = pd.DataFrame({"supplier_id": [1, None, 3], "value": [10.0, 20.0, 30.0]})
    result = handle_missing(df)
    assert len(result) == 2
    assert result["supplier_id"].isna().sum() == 0


def test_remove_outliers_clips_extreme_values():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5, 1000]})
    result = remove_outliers(df, columns=["value"])
    assert result["value"].max() < 1000


def test_fix_types_converts_date_column():
    df = pd.DataFrame({"event_date": ["2026-01-01", "2026-01-02"]})
    result = fix_types(df, date_cols=["event_date"])
    assert pd.api.types.is_datetime64_any_dtype(result["event_date"])


def test_fix_types_converts_category_column():
    df = pd.DataFrame({"region": ["Shanghai", "Chennai"]})
    result = fix_types(df, category_cols=["region"])
    assert result["region"].dtype.name == "category"


def test_drop_duplicates_removes_exact_dupes():
    df = pd.DataFrame({"supplier_id": [1, 1, 2], "value": [10, 10, 20]})
    result = drop_duplicates(df)
    assert len(result) == 2


def test_drop_duplicates_by_subset():
    df = pd.DataFrame({"supplier_id": [1, 1, 2], "value": [10, 99, 20]})
    result = drop_duplicates(df, subset=["supplier_id"])
    assert len(result) == 2
