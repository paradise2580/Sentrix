"""
src/ingestion/synthetic_signals.py

Role
----
Builds the external risk-signal layer: daily weather severity, port /
logistics congestion, and commodity (fuel) volatility per Brazilian state,
covering the real date range of the Olist orders.

Mixed provenance - this layer is NOT uniformly synthetic
---------------------------------------------------------
  commodity  REAL, from FRED (Federal Reserve Economic Data) when
             FRED_API_KEY is set - see src/ingestion/fred_signals.py.
             FRED is a free public archive with genuine daily history
             covering 2016-2018, so it can be matched to real order dates.
             Falls back to generated values if the key is missing.
  weather    Generated. OpenWeatherMap's free tier cannot backfill
             multi-year history, so live data could not be date-matched
             to 2016-2018 orders.
  port       Generated. No free public source for historical port
             congestion.

Every row records its own provenance via is_synthetic (0 = real, 1 =
generated), so the boundary is provable in the database rather than
merely asserted here.

Why this layer is synthetic — and why that is defensible
---------------------------------------------------------
Everything else in SENTRIX is real Olist data. This layer is generated
because no public dataset pairs real marketplace orders with real daily
weather / port-congestion / fuel-price data for each seller's location.
That pairing is exactly the kind of proprietary, joined-up data companies
build internally — which is why predicting supplier risk is a real problem
worth solving, not a sign the approach is flawed.

Honesty guarantees built into this generator
---------------------------------------------
1. Signals are generated from the REAL state list and the REAL Olist date
   range — they attach to real entities, not invented ones.
2. Signals are generated INDEPENDENTLY of the real is_late label. We do NOT
   peek at outcomes and rig signals to "predict" them. Doing so would
   manufacture a fake correlation and produce dishonestly good metrics.
   If a model finds these features weakly useful, that is incidental; if it
   finds them useless, that is an honest and reportable result.
3. Every row is written with is_synthetic=1 in MySQL, so synthetic data can
   never be silently mistaken for real data downstream.
"""

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.ingestion.loader import DataLoader
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

_SYNTHETIC_SOURCES = ["weather", "port"]   # commodity comes from FRED when available


def _generate_state_series(state: str, dates: pd.DatetimeIndex,
                            rng: np.random.Generator,
                            sources: list[str]) -> list[dict]:
    """
    Produce a plausible daily series per source for one state.

    Uses a smooth random walk plus occasional multi-day "shock" windows, so
    values look like real environmental/logistics data (autocorrelated, with
    episodic spikes) rather than white noise. No dependence on any label.
    """
    rows = []
    for source in sources:
        level = rng.uniform(0.2, 0.4)          # baseline for this state/source
        shock_days_left = 0
        shock_size = 0.0

        for date in dates:
            if shock_days_left == 0 and rng.random() < 0.01:
                shock_days_left = int(rng.integers(3, 12))
                shock_size = rng.uniform(0.2, 0.5)

            shock = shock_size if shock_days_left > 0 else 0.0
            if shock_days_left > 0:
                shock_days_left -= 1

            # mean-reverting random walk keeps the series smooth and bounded
            level += rng.normal(0, 0.03) + 0.05 * (0.3 - level)
            value = float(np.clip(level + shock, 0.0, 1.0))

            rows.append({
                "seller_state": state,
                "signal_date": date.date(),
                "source": source,
                "value": round(value, 4),
                "is_synthetic": 1,
            })
    return rows


def generate_signals(seed: int | None = None,
                      sources: list[str] | None = None) -> pd.DataFrame:
    """
    Build the synthetic signal table spanning the REAL states and REAL date
    range present in the loaded Olist data.
    """
    try:
        cfg = load_config()
        rng = np.random.default_rng(seed or cfg["project"]["random_state"])
        loader = DataLoader()

        states = loader.read_query(
            "SELECT DISTINCT seller_state FROM sellers WHERE seller_state IS NOT NULL"
        )["seller_state"].tolist()

        date_bounds = loader.read_query(
            "SELECT MIN(DATE(order_purchase_timestamp)) AS min_d, "
            "MAX(DATE(order_purchase_timestamp)) AS max_d FROM orders"
        ).iloc[0]

        dates = pd.date_range(date_bounds["min_d"], date_bounds["max_d"], freq="D")
        logger.info(
            f"Generating synthetic signals for {len(states)} real states "
            f"across {len(dates)} real days ({date_bounds['min_d']} to {date_bounds['max_d']})"
        )

        sources = sources or _SYNTHETIC_SOURCES
        rows = []
        for state in states:
            rows.extend(_generate_state_series(state, dates, rng, sources))

        df = pd.DataFrame(rows)
        logger.info(f"Generated {len(df):,} synthetic rows for {sources} (is_synthetic=1)")
        return df
    except Exception as e:
        raise SentrixException(e, sys)


def generate_and_load(reset: bool = True) -> dict:
    """
    Build the full external-signal layer and load it into MySQL.

    Commodity volatility is pulled REAL from FRED when FRED_API_KEY is set;
    if the key is absent or FRED is unreachable, it is generated instead and
    flagged accordingly. Weather and port are always generated.

    Returns a provenance breakdown so the caller can see exactly how many
    rows are real vs generated.
    """
    try:
        loader = DataLoader()
        if reset:
            loader.truncate_table("external_signals")

        # --- generated: weather + port ---
        generated = generate_signals(sources=_SYNTHETIC_SOURCES)

        # --- commodity: REAL from FRED, else generated fallback ---
        from src.ingestion.fred_signals import fred_key_available, build_commodity_signal

        states = generated["seller_state"].unique().tolist()
        start = str(generated["signal_date"].min())
        end = str(generated["signal_date"].max())

        commodity, commodity_is_real = None, False
        if fred_key_available():
            try:
                commodity = build_commodity_signal(states, start, end)
                commodity_is_real = True
                logger.info("Commodity signal: REAL data from FRED (is_synthetic=0)")
            except Exception as e:
                logger.warning(
                    f"FRED fetch failed ({e.__class__.__name__}) - falling back to "
                    "generated commodity signal. Set a valid FRED_API_KEY and "
                    "re-run to use real data."
                )
        else:
            logger.info(
                "FRED_API_KEY not set - commodity signal will be generated. "
                "Free key: https://fred.stlouisfed.org/docs/api/api_key.html"
            )

        if commodity is None:
            commodity = generate_signals(sources=["commodity"])

        df = pd.concat([generated, commodity], ignore_index=True)
        loader.write_df(df, "external_signals", if_exists="append", chunksize=10000)

        breakdown = {
            "total_rows": len(df),
            "real_rows": int((df["is_synthetic"] == 0).sum()),
            "generated_rows": int((df["is_synthetic"] == 1).sum()),
            "commodity_source": "FRED (real)" if commodity_is_real else "generated",
        }
        logger.info(f"Signal layer loaded: {breakdown}")
        return breakdown
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    b = generate_and_load(reset=True)
    print(f"\nExternal signal layer loaded: {b['total_rows']:,} rows")
    print(f"  REAL (is_synthetic=0):      {b['real_rows']:,}")
    print(f"  Generated (is_synthetic=1): {b['generated_rows']:,}")
    print(f"  Commodity source:           {b['commodity_source']}")
    print("\nAll rows use real seller states and the real Olist date range.")
    print("Generated values are produced independently of the label.")
