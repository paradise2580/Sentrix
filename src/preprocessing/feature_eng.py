"""
LEGACY: the seller-day feature table from the earlier seller-level model.

The live model uses preprocessing/order_features.py. This module is kept
because scripts/label_screen.py and scripts/ablation.py import it to
reproduce the results that led to the switch to order-level prediction.

Grain: one row per (seller_id, as_of_date). Label: seller's late rate over
the next 30 days >= late_rate_threshold. Features look only backward
(rolling windows use shift(1)); the label looks only forward.
"""

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.ingestion.loader import DataLoader
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

# Label of the retired seller-level model (not config.yaml's target_column,
# which now belongs to the order-level model).
TARGET = "high_late_rate_next_30d"


# ---------------------------------------------------------------- extraction
def load_seller_day_orders() -> pd.DataFrame:
    """Delivered orders per (seller_id, order_date): order count and late count."""
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
    Daily grid per seller from first to last active day. Sellers with fewer
    than active_min_orders orders are dropped.
    """
    totals = orders.groupby("seller_id")["orders_n"].sum()
    keep = totals[totals >= active_min_orders].index
    orders = orders[orders["seller_id"].isin(keep)]
    logger.info(f"Kept {len(keep):,} sellers with >= {active_min_orders} orders")

    if orders.empty:
        # Nothing left after filtering: return an empty panel with the right columns.
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
    """Family 2 (REAL): rolling review-score features."""
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
    """Family 3 (SYNTHETIC): joined on seller state and date."""
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
    """Family 4 (REAL): lifetime late rate and state-peer comparison."""
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


# Forward-looking columns used only to build the label; always dropped.
FORWARD_ONLY_COLUMNS = ["late_rate_next_30d", "forward_orders"]


def attach_label(panel: pd.DataFrame, horizon_days: int,
                 late_rate_threshold: float, min_forward_orders: int,
                 target_col: str = TARGET) -> pd.DataFrame:
    """
    Forward label: does the seller's late rate over (as_of_date,
    as_of_date + horizon] reach late_rate_threshold?

    A rate rather than "any late order", because "any late order" mostly
    measured how many orders a seller shipped. Seller-days with fewer than
    min_forward_orders forward orders are left unlabelled.
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

        # Drop raw counts and every forward-looking column.
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
