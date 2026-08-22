"""
tests/test_fred_signals.py

Tests the REAL FRED commodity-signal path without hitting the live API
(no network dependency in CI). The FRED response shape is stubbed exactly
as the API returns it, including "." for missing observations.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.ingestion.fred_signals import (
    fred_key_available, to_daily_volatility, build_commodity_signal,
)


def _fake_fred_response(start="2017-01-01", end="2017-06-30"):
    """Business-day price series with periodic '.' gaps, like real FRED."""
    dates = pd.date_range(start, end, freq="B")
    prices = 50 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, len(dates)))
    return {"observations": [
        {"date": d.strftime("%Y-%m-%d"), "value": ("." if i % 30 == 0 else f"{p:.2f}")}
        for i, (d, p) in enumerate(zip(dates, prices))
    ]}


class _StubResponse:
    status_code = 200
    def __init__(self, payload): self._payload = payload
    def raise_for_status(self): pass
    def json(self): return self._payload


def test_key_detection_reflects_env():
    with patch.dict("os.environ", {"FRED_API_KEY": ""}, clear=False):
        assert fred_key_available() is False
    with patch.dict("os.environ", {"FRED_API_KEY": "abc123"}, clear=False):
        assert fred_key_available() is True


def test_volatility_is_scaled_and_gapless():
    """Volatility must be 0-1 and defined on every calendar day."""
    dates = pd.date_range("2017-01-01", "2017-03-31", freq="B")
    prices = pd.DataFrame({
        "date": dates,
        "value": 50 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, len(dates))),
    })
    all_days = pd.date_range("2017-01-01", "2017-03-31", freq="D")

    vol = to_daily_volatility(prices, all_days)

    assert len(vol) == len(all_days)          # weekends filled, no gaps
    assert vol["value"].notna().all()
    assert vol["value"].between(0, 1).all()


def test_real_rows_are_flagged_not_synthetic():
    """FRED-sourced rows must carry is_synthetic=0 — the provenance guarantee."""
    with patch.dict("os.environ", {"FRED_API_KEY": "test_key"}), \
         patch("src.ingestion.fred_signals.requests.get",
               return_value=_StubResponse(_fake_fred_response())):
        df = build_commodity_signal(["SP", "RJ"], "2017-01-01", "2017-06-30")

    assert (df["is_synthetic"] == 0).all()
    assert set(df["seller_state"]) == {"SP", "RJ"}
    assert (df["source"] == "commodity").all()


def test_missing_key_raises_clearly():
    with patch.dict("os.environ", {"FRED_API_KEY": ""}, clear=False):
        with pytest.raises(Exception):
            build_commodity_signal(["SP"], "2017-01-01", "2017-06-30")
