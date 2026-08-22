"""
tests/conftest.py

Shared fixtures. Unit tests use small in-memory frames shaped exactly like
the real Olist-derived tables — fast, deterministic, no infrastructure.
Integration-flavoured tests skip cleanly when MySQL/ChromaDB aren't up.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest


def mysql_is_reachable() -> bool:
    try:
        from src.ingestion.db import get_engine
        with get_engine(force_new=True).connect():
            return True
    except Exception:
        return False


@pytest.fixture
def skip_if_no_mysql():
    if not mysql_is_reachable():
        pytest.skip("MySQL not reachable — skipping integration test")


@pytest.fixture
def sample_sellers_df() -> pd.DataFrame:
    """Shaped like the real sellers table (hash IDs, Brazilian states)."""
    return pd.DataFrame({
        "seller_id": ["a1b2c3", "d4e5f6", "g7h8i9"],
        "seller_zip_code_prefix": [1310, 20040, 80010],
        "seller_city": ["sao paulo", "rio de janeiro", "curitiba"],
        "seller_state": ["SP", "RJ", "PR"],
    })


@pytest.fixture
def sample_seller_day_orders() -> pd.DataFrame:
    """Real-shaped seller-day order aggregates (from load_seller_day_orders)."""
    rng = np.random.default_rng(42)
    rows = []
    for seller_id in ["a1b2c3", "d4e5f6", "g7h8i9"]:
        for d in pd.date_range("2017-01-01", periods=60, freq="D"):
            n = int(rng.integers(0, 5))
            if n:
                rows.append({
                    "seller_id": seller_id, "order_date": d,
                    "orders_n": n, "late_n": int(rng.binomial(n, 0.08)),
                    "avg_delay_days": float(rng.normal(-3, 5)),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_feature_table() -> pd.DataFrame:
    """A small table matching the real feature-table schema, for model tests."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame({
        "seller_id": rng.choice(["a1b2c3", "d4e5f6", "g7h8i9"], n),
        "as_of_date": pd.date_range("2017-01-01", periods=n, freq="D"),
        "late_count_7d": rng.integers(0, 3, n).astype(float),
        "order_count_7d": rng.integers(1, 20, n).astype(float),
        "late_rate_7d": rng.uniform(0, 0.5, n),
        "late_count_30d": rng.integers(0, 10, n).astype(float),
        "order_count_30d": rng.integers(5, 80, n).astype(float),
        "late_rate_30d": rng.uniform(0, 0.4, n),
        "late_rate_trend": rng.normal(0, 0.1, n),
        "days_since_last_late": rng.uniform(0, 200, n),
        "review_score_avg_7d": rng.uniform(1, 5, n),
        "bad_reviews_7d": rng.integers(0, 4, n).astype(float),
        "review_score_avg_30d": rng.uniform(1, 5, n),
        "weather_severity_7d": rng.uniform(0, 1, n),
        "port_congestion_7d": rng.uniform(0, 1, n),
        "commodity_volatility_7d": rng.uniform(0, 1, n),
        "lifetime_late_rate": rng.uniform(0, 0.3, n),
        "state_peer_late_rate": rng.uniform(0, 0.3, n),
        "seller_tenure_days": rng.integers(0, 700, n),
        "seller_state": rng.choice(["SP", "RJ", "PR"], n),
        "disruption_next_30d": rng.integers(0, 2, n),
    })
