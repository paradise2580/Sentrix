"""tests/test_loader.py — integration test for MySQL round-trips."""

import pandas as pd
from sqlalchemy import text

from src.ingestion.loader import DataLoader


def test_write_and_read_round_trip(skip_if_no_mysql):
    loader = DataLoader()
    table = "test_roundtrip_scratch"
    original = pd.DataFrame({"id": [1, 2, 3], "label": ["a", "b", "c"]})

    loader.write_df(original, table, if_exists="replace")
    result = loader.read_table(table)

    assert len(result) == len(original)
    assert sorted(result["label"]) == sorted(original["label"])

    with loader.engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


def test_real_olist_tables_are_populated(skip_if_no_mysql):
    """The real data should be loaded, with a plausible late rate."""
    loader = DataLoader()
    counts = loader.read_query("""
        SELECT (SELECT COUNT(*) FROM sellers)  AS sellers,
               (SELECT COUNT(*) FROM orders)   AS orders,
               (SELECT COUNT(*) FROM reviews)  AS reviews
    """).iloc[0]
    assert counts["sellers"] > 0 and counts["orders"] > 0 and counts["reviews"] > 0

    rate = loader.read_query(
        "SELECT AVG(is_late) r FROM orders WHERE is_late IS NOT NULL"
    ).iloc[0]["r"]
    assert 0.0 < float(rate) < 0.5, "real Olist late rate should be a small minority"


def test_signal_provenance_is_explicit(skip_if_no_mysql):
    """Every external signal row is flagged 0 (real, FRED) or 1 (generated)."""
    loader = DataLoader()
    bad = loader.read_query(
        "SELECT COUNT(*) n FROM external_signals "
        "WHERE is_synthetic IS NULL OR is_synthetic NOT IN (0, 1)"
    ).iloc[0]["n"]
    assert int(bad) == 0

    # weather and port can never be real — no free historical source exists
    leaked = loader.read_query(
        "SELECT COUNT(*) n FROM external_signals "
        "WHERE source IN ('weather','port') AND is_synthetic = 0"
    ).iloc[0]["n"]
    assert int(leaked) == 0
