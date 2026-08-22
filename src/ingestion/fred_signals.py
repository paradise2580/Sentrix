"""
src/ingestion/fred_signals.py

Role
----
Fetches REAL historical commodity/economic series from FRED (Federal Reserve
Economic Data, St. Louis Fed) and converts them into the commodity-volatility
signal used by SENTRIX's feature layer.

Why FRED specifically — and why only this signal is real
---------------------------------------------------------
SENTRIX's orders span Sept 2016 - Oct 2018. To be genuinely useful, an
external-signal API must return data FOR THOSE DATES. Free tiers were
evaluated for each signal type:

  news      NewsAPI free tier serves roughly the last 30 days only. It
            cannot backfill 2017, so live news would silently mismatch the
            order dates. -> kept synthetic.
  weather   OpenWeatherMap's free tier is current/forecast; multi-year
            historical requires a paid plan. -> kept synthetic.
  commodity FRED is a public government archive with decades of daily
            history, free, and queryable by date range. -> REAL.

So this module upgrades exactly the one signal that free tooling can
honestly support, rather than pretending all three are real.

Series used
-----------
  DCOILWTICO  Crude Oil WTI spot price (daily) - fuel cost, a real driver
              of freight/logistics cost and delivery pressure.

Volatility, not price
---------------------
The raw price level is not comparable to the other 0-1 signals, so we
convert it to rolling volatility (std-dev of daily returns) and min-max
scale it to 0-1 - the same scale the synthetic weather/port signals use.

No key / offline behaviour
--------------------------
If FRED_API_KEY is absent or the API is unreachable, this raises a clear
error and the caller falls back to the synthetic commodity series, so the
pipeline never breaks. Provenance is recorded per row in MySQL
(is_synthetic = 0 for FRED rows, 1 for generated rows).
"""

import os
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src.logger import get_logger
from src.exception import SentrixException
import sys

load_dotenv()
logger = get_logger(__name__)

_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_SERIES_ID = "DCOILWTICO"          # Crude Oil WTI spot price, daily
_VOL_WINDOW = 14                    # rolling window for volatility


def fred_key_available() -> bool:
    return bool(os.getenv("FRED_API_KEY", "").strip())


def fetch_series(start: str, end: str, series_id: str = _SERIES_ID) -> pd.DataFrame:
    """
    Fetch one FRED series over a date range.
    Returns DataFrame[date, value] with missing observations dropped.
    """
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in .env")

    try:
        resp = requests.get(_FRED_URL, timeout=30, params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
        })
        resp.raise_for_status()
        obs = resp.json().get("observations", [])

        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        # FRED marks missing observations with "."
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).reset_index(drop=True)

        logger.info(f"FRED {series_id}: fetched {len(df)} REAL observations "
                    f"({start} to {end})")
        return df
    except Exception as e:
        raise SentrixException(e, sys)


def to_daily_volatility(prices: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Convert a daily price series into a 0-1 scaled rolling-volatility signal
    aligned to every calendar day (FRED skips weekends/holidays, so values
    are forward-filled onto the full daily grid the panel expects).
    """
    df = prices.set_index("date").reindex(all_dates).ffill().bfill()
    df.index.name = "signal_date"

    returns = df["value"].pct_change()
    vol = returns.rolling(_VOL_WINDOW, min_periods=2).std()

    lo, hi = vol.min(), vol.max()
    scaled = (vol - lo) / (hi - lo) if hi > lo else vol * 0
    scaled = scaled.fillna(scaled.median()).clip(0, 1)

    return pd.DataFrame({
        "signal_date": df.index,
        "value": scaled.round(4).values,
    })


def build_commodity_signal(states: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Build REAL commodity-volatility rows for every state over the date range.

    Oil price is a national/global signal, so the same series applies to all
    states - unlike weather, it is not location-specific. Rows are emitted
    per state so the table shape matches the other signals and joins cleanly.
    """
    try:
        all_dates = pd.date_range(start, end, freq="D")
        prices = fetch_series(start, end)
        vol = to_daily_volatility(prices, all_dates)

        frames = []
        for state in states:
            s = vol.copy()
            s["seller_state"] = state
            s["source"] = "commodity"
            s["is_synthetic"] = 0          # REAL data from FRED
            frames.append(s)

        out = pd.concat(frames, ignore_index=True)
        logger.info(f"Built {len(out):,} REAL commodity signal rows "
                    f"({len(states)} states x {len(all_dates)} days)")
        return out[["seller_state", "signal_date", "source", "value", "is_synthetic"]]
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    if not fred_key_available():
        print("FRED_API_KEY not set in .env - get a free key at:")
        print("  https://fred.stlouisfed.org/docs/api/api_key.html")
        raise SystemExit(1)

    df = build_commodity_signal(["SP", "RJ"], "2016-09-01", "2018-11-01")
    print(df.head(10).to_string(index=False))
    print(f"\n{len(df):,} REAL rows, is_synthetic=0")
