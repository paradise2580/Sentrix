"""
src/preprocessing/feature_eng.py

Role
----
Builds the model-ready feature table from the real Olist data (plus the one
clearly-flagged synthetic signal layer).

Grain
-----
One row per (seller_id, as_of_date) — a "seller-day" panel. Sellers are the
suppliers whose delivery risk SENTRIX predicts.

Label (REAL)
------------
    late_rate over the next 30 days >= threshold  ->  1

Derived from Olist's real order_delivered_customer_date vs
order_estimated_delivery_date. Nothing about the outcome is invented.

This is a RATE, not "any late order". See attach_label for the ablation that
forced the change: the previous any-late label was so volume-confounded that
ranking sellers by order count alone beat the full model.

Leakage control
---------------
Every feature looks strictly BACKWARD from as_of_date; the label looks
strictly FORWARD. Rolling windows are computed with .shift(1) so the
current day's own outcome can never leak into its own features.

Feature families
----------------
1. Delivery history   (REAL)      rolling late counts/rates, volume, recency
2. Review sentiment   (REAL)      rolling mean review score, bad-review counts
3. External signals   (SYNTHETIC) weather / port / commodity by seller state
4. Peer & profile     (REAL)      lifetime late rate, state-peer late rate
"""

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.ingestion.loader import DataLoader
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

# The RETIRED seller-day label. This module is no longer the project's
# feature builder — SENTRIX models orders (see preprocessing/order_features)
# because scripts/label_screen2.py showed the seller-level target does not
# persist across the window boundary. The builder is kept because the
# screening scripts that produced that evidence import it, and a reader who
# wants to reproduce the finding needs the code that generated it.
#
# It deliberately does NOT read config.yaml's target_column any more: that
# name now belongs to the order-level label.
TARGET = "high_late_rate_next_30d"


# ---------------------------------------------------------------- extraction
def load_seller_day_orders() -> pd.DataFrame:
    """
    Real, delivered orders joined to their seller, collapsed to one row per
    (seller_id, order_date) with that day's order count and late count.
    """
    loader = DataLoader()
    sql = """
        SELECT oi.seller_id,
               DATE(o.order_purchase_timestamp) AS order_date,
               COUNT(*)                          AS orders_n,
               SUM(o.is_late)                    AS late_n,
               AVG(o.delay_days)                 AS avg_delay_days
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        WHERE o.is_late IS NOT NULL
        GROUP BY oi.seller_id, DATE(o.order_purchase_timestamp)
    """
    df = loader.read_query(sql)
    df["order_date"] = pd.to_datetime(df["order_date"])
    logger.info(f"Loaded {len(df):,} real seller-day order aggregates")
    return df


def load_seller_day_reviews() -> pd.DataFrame:
    """Real customer reviews aggregated to (seller_id, review_date)."""
    loader = DataLoader()
    sql = """
        SELECT oi.seller_id,
               DATE(r.review_creation_date) AS review_date,
               AVG(r.review_score)          AS avg_review_score,
               SUM(r.review_score <= 2)     AS bad_reviews_n,
               COUNT(*)                     AS reviews_n
        FROM reviews r
        JOIN order_items oi ON oi.order_id = r.order_id
        WHERE r.review_creation_date IS NOT NULL
        GROUP BY oi.seller_id, DATE(r.review_creation_date)
    """
    df = loader.read_query(sql)
    df["review_date"] = pd.to_datetime(df["review_date"])
    logger.info(f"Loaded {len(df):,} real seller-day review aggregates")
    return df


def load_signals_wide() -> pd.DataFrame:
    """SYNTHETIC external signals, pivoted to one row per (state, date)."""
    loader = DataLoader()
    df = loader.read_query(
        "SELECT seller_state, signal_date, source, value FROM external_signals"
    )
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    wide = df.pivot_table(index=["seller_state", "signal_date"],
                          columns="source", values="value").reset_index()
    wide.columns.name = None
    return wide.rename(columns={
        "weather": "weather_severity",
        "port": "port_congestion",
        "commodity": "commodity_volatility",
    })


# ---------------------------------------------------------------- panel
def build_seller_day_panel(orders: pd.DataFrame, active_min_orders: int) -> pd.DataFrame:
    """
    Continuous daily grid per seller, from their first to last active day.
    Sellers below active_min_orders total orders are dropped — too little
    history to model, and they'd dominate the panel with empty rows.
    """
    totals = orders.groupby("seller_id")["orders_n"].sum()
    keep = totals[totals >= active_min_orders].index
    orders = orders[orders["seller_id"].isin(keep)]
    logger.info(f"Kept {len(keep):,} sellers with >= {active_min_orders} orders")

    if orders.empty:
        # Every seller filtered out by active_min_orders — return an empty
        # panel with the right columns rather than letting pd.concat([]) raise.
        logger.warning("No sellers met the activity threshold; empty panel returned")
        return pd.DataFrame(columns=["seller_id", "as_of_date", "orders_n",
                                      "late_n", "avg_delay_days"])

    spans = orders.groupby("seller_id")["order_date"].agg(["min", "max"])
    frames = []
    for seller_id, row in spans.iterrows():
        dates = pd.date_range(row["min"], row["max"], freq="D")
        frames.append(pd.DataFrame({"seller_id": seller_id, "as_of_date": dates}))
    panel = pd.concat(frames, ignore_index=True)

    panel = panel.merge(
        orders.rename(columns={"order_date": "as_of_date"}),
        on=["seller_id", "as_of_date"], how="left",
    )
    for c in ["orders_n", "late_n"]:
        panel[c] = panel[c].fillna(0)
    logger.info(f"Seller-day panel: {len(panel):,} rows")
    return panel.sort_values(["seller_id", "as_of_date"]).reset_index(drop=True)


# ---------------------------------------------------------------- families
def add_delivery_history_features(panel: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Family 1 (REAL). shift(1) keeps today's own outcome out of today's features."""
    g = panel.groupby("seller_id")
    prev_late = g["late_n"].shift(1).fillna(0)
    prev_orders = g["orders_n"].shift(1).fillna(0)
    panel["_prev_late"], panel["_prev_orders"] = prev_late, prev_orders

    for w in windows:
        gl = panel.groupby("seller_id")["_prev_late"]
        go = panel.groupby("seller_id")["_prev_orders"]
        late_sum = gl.rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        ord_sum = go.rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        panel[f"late_count_{w}d"] = late_sum
        panel[f"order_count_{w}d"] = ord_sum
        panel[f"late_rate_{w}d"] = late_sum / ord_sum.replace(0, np.nan)

    panel["late_rate_trend"] = panel["late_rate_7d"] - panel["late_rate_30d"]

    # days since the seller's last late delivery
    def days_since_late(sub: pd.DataFrame) -> pd.Series:
        out, last = [], None
        for d, n in zip(sub["as_of_date"], sub["_prev_late"]):
            out.append(9999 if last is None else (d - last).days)
            if n > 0:
                last = d
        return pd.Series(out, index=sub.index)

    panel["days_since_last_late"] = (
        panel.groupby("seller_id", group_keys=False)[["as_of_date", "_prev_late"]]
        .apply(days_since_late)
    )
    return panel.drop(columns=["_prev_late", "_prev_orders"])


def add_review_features(panel: pd.DataFrame, reviews: pd.DataFrame,
                         windows: list[int]) -> pd.DataFrame:
    """Family 2 (REAL) — rolling sentiment from genuine customer review scores."""
    panel = panel.merge(
        reviews.rename(columns={"review_date": "as_of_date"}),
        on=["seller_id", "as_of_date"], how="left",
    )
    panel = panel.sort_values(["seller_id", "as_of_date"])
    prev_score = panel.groupby("seller_id")["avg_review_score"].shift(1)
    prev_bad = panel.groupby("seller_id")["bad_reviews_n"].shift(1).fillna(0)
    panel["_ps"], panel["_pb"] = prev_score, prev_bad

    for w in windows:
        panel[f"review_score_avg_{w}d"] = (
            panel.groupby("seller_id")["_ps"].rolling(w, min_periods=1)
            .mean().reset_index(level=0, drop=True)
        )
        panel[f"bad_reviews_{w}d"] = (
            panel.groupby("seller_id")["_pb"].rolling(w, min_periods=1)
            .sum().reset_index(level=0, drop=True)
        )
    return panel.drop(columns=["_ps", "_pb", "avg_review_score", "bad_reviews_n", "reviews_n"])


def add_external_signal_features(panel: pd.DataFrame, sellers: pd.DataFrame,
                                  signals: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Family 3 (SYNTHETIC) — joined on real seller state and real date."""
    panel = panel.merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    panel = panel.merge(
        signals.rename(columns={"signal_date": "as_of_date"}),
        on=["seller_state", "as_of_date"], how="left",
    ).sort_values(["seller_id", "as_of_date"])

    for col in ["weather_severity", "port_congestion", "commodity_volatility"]:
        for w in windows:
            panel[f"{col}_{w}d"] = (
                panel.groupby("seller_id")[col].rolling(w, min_periods=1)
                .mean().reset_index(level=0, drop=True)
            )
    return panel.drop(columns=["weather_severity", "port_congestion", "commodity_volatility"])


def add_profile_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Family 4 (REAL) — expanding lifetime late rate + state-peer comparison."""
    panel = panel.sort_values(["seller_id", "as_of_date"])
    cum_late = panel.groupby("seller_id")["late_count_30d"].cumsum()
    cum_ord = panel.groupby("seller_id")["order_count_30d"].cumsum()
    panel["lifetime_late_rate"] = cum_late / cum_ord.replace(0, np.nan)
    panel["state_peer_late_rate"] = panel.groupby(
        ["seller_state", "as_of_date"])["late_rate_30d"].transform("mean")
    panel["seller_tenure_days"] = (
        panel["as_of_date"] - panel.groupby("seller_id")["as_of_date"].transform("min")
    ).dt.days
    return panel


# Columns computed from the FUTURE. They exist only to build the label and
# must never reach a model — hence one place to name them and one place to
# drop them.
FORWARD_ONLY_COLUMNS = ["late_rate_next_30d", "forward_orders"]


def attach_label(panel: pd.DataFrame, horizon_days: int,
                 late_rate_threshold: float, min_forward_orders: int,
                 target_col: str = TARGET) -> pd.DataFrame:
    """
    REAL forward label: over orders placed in (as_of_date, as_of_date +
    horizon], does this seller's LATE RATE exceed late_rate_threshold?

    Why a rate and not "any late order"
    -----------------------------------
    The original label was `any late delivery in the next 30 days`. With a
    per-order late rate around 8%, that is close to a deterministic function
    of order count: ship 100 orders and P(at least one late) is ~99.97%;
    ship 3 and it is 22%. So the label mostly encoded how BUSY a seller was,
    not how RISKY.

    That was not a suspicion, it was measured. scripts/ablation.py ranked
    sellers by `order_count_30d` alone, with no model at all, and scored
    PR-AUC 0.4455 against the full 47-feature model's 0.3911. A single column
    beat the champion by 14%, and adding the other 44 features to the three
    volume ones made things worse.

    A rate divides that confound out: a seller shipping 100 orders with 8
    late looks exactly like a seller shipping 10 with 0.8 late.

    Eligibility
    -----------
    A rate needs a denominator. Seller-days with fewer than
    min_forward_orders in the forward window are left UNLABELLED rather than
    labelled, because 0/1 and 1/1 are noise. This does condition the modelled
    population on future activity, which is worth stating plainly: the model
    is trained and evaluated on sellers who go on to trade. The alternative —
    scoring a seller who ships nothing as "0% late" — would conflate "did not
    ship" with "shipped perfectly", which is a worse distortion than the one
    it avoids.
    """
    panel = panel.sort_values(["seller_id", "as_of_date"])

    def forward_sum(column: str) -> pd.Series:
        """Sum of `column` over (as_of_date, as_of_date + horizon]."""
        rolled = (panel.groupby("seller_id")[column]
                  .rolling(horizon_days, min_periods=1).sum()
                  .reset_index(level=0, drop=True))
        return rolled.groupby(panel["seller_id"]).shift(-horizon_days)

    forward_late = forward_sum("late_n")
    forward_orders = forward_sum("orders_n")

    late_rate = forward_late / forward_orders.replace(0, np.nan)
    panel["late_rate_next_30d"] = late_rate
    panel["forward_orders"] = forward_orders

    label = (late_rate >= late_rate_threshold).astype("float")
    # Unmeasurable, not negative.
    label[forward_orders < min_forward_orders] = np.nan
    label[forward_orders.isna()] = np.nan
    panel[target_col] = label

    measurable = int(label.notna().sum())
    logger.info(
        f"Label '{target_col}': late_rate >= {late_rate_threshold:.0%} over "
        f"{horizon_days}d, among seller-days with >= {min_forward_orders} "
        f"forward orders. {measurable:,} of {len(panel):,} rows labelled "
        f"({measurable / len(panel):.1%}); positive rate "
        f"{label.mean():.2%} of those."
    )
    return panel


# ---------------------------------------------------------------- orchestrator
def build_feature_table() -> pd.DataFrame:
    try:
        cfg = load_config()
        windows = cfg["preprocessing"]["rolling_windows"]
        horizon = cfg["preprocessing"].get("label_horizon_days", 30)
        active_min = cfg["preprocessing"].get("active_min_orders", 10)

        loader = DataLoader()
        sellers = loader.read_table("sellers")
        orders = load_seller_day_orders()
        reviews = load_seller_day_reviews()
        signals = load_signals_wide()

        logger.info("Building seller-day panel...")
        panel = build_seller_day_panel(orders, active_min)

        logger.info("Family 1: delivery history (REAL)...")
        panel = add_delivery_history_features(panel, windows)

        logger.info("Family 2: review sentiment (REAL)...")
        panel = add_review_features(panel, reviews, windows)

        logger.info("Family 3: external signals (SYNTHETIC)...")
        panel = add_external_signal_features(panel, sellers, signals, windows)

        logger.info("Family 4: profile & peer (REAL)...")
        panel = add_profile_features(panel)

        logger.info(f"Attaching REAL forward label (horizon={horizon}d)...")
        panel = attach_label(
            panel, horizon,
            late_rate_threshold=cfg["preprocessing"]["late_rate_threshold"],
            min_forward_orders=cfg["preprocessing"]["min_forward_orders"],
        )

        # Drop the raw counts AND every column derived from the future. The
        # forward columns exist only to build the label; leaving even one of
        # them in the feature table would hand the model the answer.
        panel = panel.drop(
            columns=["orders_n", "late_n", "avg_delay_days"] + FORWARD_ONLY_COLUMNS,
            errors="ignore",
        )
        panel = panel.dropna(subset=[TARGET])
        panel[TARGET] = panel[TARGET].astype(int)

        leaked = [c for c in panel.columns if "next" in c and c != TARGET]
        if leaked:
            raise ValueError(f"Forward-looking columns survived into the feature "
                             f"table: {leaked}")

        logger.info(f"Feature table: {panel.shape}, "
                    f"positive rate {panel[TARGET].mean():.2%}")
        return panel.reset_index(drop=True)
    except Exception as e:
        raise SentrixException(e, sys)
