"""
Builds the order-level feature table: one row per (order_id, seller_id),
labelled with whether the order arrived after its promised date.

Every feature must be known at purchase time. Excluded for leakage:
order_approved_at, order_delivered_carrier_date,
order_delivered_customer_date and delay_days. Seller and state history is
lagged by `history_lag_days`, because recent orders have no outcome yet.

Orders, not sellers: a seller's late rate doesn't carry over from one
month to the next (scripts/label_screen2.py), but an order's risk is set by
things known at purchase. Seller risk is the mean of their orders' risk.
"""

import sys

import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.ingestion.loader import DataLoader
from src.logger import get_logger
from src.exception import SentrixException

logger = get_logger(__name__)

TARGET = load_config()["model"]["target_column"]

# Only columns known at purchase time are selected.
ORDER_SQL = """
    SELECT  oi.order_id,
            oi.seller_id,
            o.order_purchase_timestamp,
            o.order_estimated_delivery_date,
            o.is_late,
            MIN(oi.shipping_limit_date)      AS shipping_limit_date,
            COUNT(*)                          AS n_items,
            SUM(oi.price)                     AS total_price,
            SUM(oi.freight_value)             AS total_freight,
            AVG(p.product_weight_g)           AS avg_weight_g,
            MAX(p.product_length_cm * p.product_height_cm * p.product_width_cm)
                                              AS max_volume_cm3,
            MIN(p.product_category_english)   AS category,
            MIN(c.customer_state)             AS customer_state,
            MIN(c.customer_zip_code_prefix)   AS customer_zip,
            MIN(s.seller_state)               AS seller_state,
            MIN(s.seller_zip_code_prefix)     AS seller_zip
    FROM order_items oi
    JOIN orders    o ON o.order_id   = oi.order_id
    JOIN products  p ON p.product_id = oi.product_id
    JOIN customers c ON c.customer_id = o.customer_id
    JOIN sellers   s ON s.seller_id  = oi.seller_id
    WHERE o.is_late IS NOT NULL
    GROUP BY oi.order_id, oi.seller_id, o.order_purchase_timestamp,
             o.order_estimated_delivery_date, o.is_late
"""


def load_orders() -> pd.DataFrame:
    df = DataLoader().read_query(ORDER_SQL)

    for col in ["order_purchase_timestamp", "order_estimated_delivery_date",
                "shipping_limit_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # MySQL returns DECIMAL aggregates as Python Decimal (dtype=object), which
    # would otherwise be treated as categorical and one-hot encoded.
    for col in ["n_items", "total_price", "total_freight", "avg_weight_g",
                "max_volume_cm3", "customer_zip", "seller_zip", "is_late"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Loaded {len(df):,} order-seller rows")
    return df


# --------------------------------------------------------------- features
def add_promise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Days promised for delivery and for handover to the carrier.
    `promised_days` is the strongest single feature in the project.
    """
    purchase = df["order_purchase_timestamp"]
    df["promised_days"] = (df["order_estimated_delivery_date"] - purchase).dt.total_seconds() / 86400
    df["shipping_limit_days"] = (df["shipping_limit_date"] - purchase).dt.total_seconds() / 86400
    # Share of the promised window the seller may use before handover.
    df["handover_share"] = df["shipping_limit_days"] / df["promised_days"].replace(0, np.nan)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, weekday, month and weekend flag of the purchase."""
    purchase = df["order_purchase_timestamp"]
    df["purchase_hour"] = purchase.dt.hour
    df["purchase_dow"] = purchase.dt.dayofweek
    df["purchase_month"] = purchase.dt.month
    df["purchase_is_weekend"] = (purchase.dt.dayofweek >= 5).astype(int)
    return df


def add_shipment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Freight ratio, price per item, weight, volume and density."""
    df["freight_ratio"] = df["total_freight"] / df["total_price"].replace(0, np.nan)
    df["price_per_item"] = df["total_price"] / df["n_items"].replace(0, np.nan)
    df["density"] = df["avg_weight_g"] / df["max_volume_cm3"].replace(0, np.nan)
    df["log_weight"] = np.log1p(df["avg_weight_g"].clip(lower=0))
    df["log_volume"] = np.log1p(df["max_volume_cm3"].clip(lower=0))
    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distance proxy from Brazilian postcode (CEP) prefixes, which are assigned
    geographically. A rough stand-in for distance, not kilometres.
    """
    df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
    df["zip_gap"] = (df["customer_zip"].astype(float) - df["seller_zip"].astype(float)).abs()
    # float, not Int64: sklearn imputers need NaN, not pd.NA.
    df["seller_zip_region"] = df["seller_zip"].astype(float) // 1000
    df["customer_zip_region"] = df["customer_zip"].astype(float) // 1000
    return df


def add_lagged_history(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """
    Seller and customer-state late rate as of `lag_days` before each order.

    The lag matters: recent orders have no delivery outcome yet, so using
    them would leak information that production would not have.
    Cumulative daily counts are joined back with merge_asof.
    """
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)
    df["purchase_date"] = df["order_purchase_timestamp"].dt.normalize()
    lag = pd.Timedelta(days=lag_days)

    for key, prefix in [("seller_id", "seller"), ("customer_state", "state")]:
        daily = (df.groupby([key, "purchase_date"])
                 .agg(n=("is_late", "size"), late=("is_late", "sum"))
                 .reset_index()
                 .sort_values("purchase_date"))
        daily["cum_n"] = daily.groupby(key)["n"].cumsum()
        daily["cum_late"] = daily.groupby(key)["late"].cumsum()

        left = df[[key, "purchase_date"]].copy()
        left["as_of"] = left["purchase_date"] - lag
        left = left.sort_values("as_of")

        joined = pd.merge_asof(
            left, daily[[key, "purchase_date", "cum_n", "cum_late"]].rename(
                columns={"purchase_date": "as_of"}),
            on="as_of", by=key, direction="backward",
        ).sort_index()

        n = joined["cum_n"]
        df[f"{prefix}_prior_orders"] = n.values
        df[f"{prefix}_prior_late_rate"] = (joined["cum_late"] / n.replace(0, np.nan)).values

    return df.drop(columns=["purchase_date"])


def collapse_categories(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Keep the `top_n` most common categories; the rest become 'other'."""
    top = df["category"].value_counts().head(top_n).index
    df["category"] = df["category"].where(df["category"].isin(top), "other").fillna("other")
    return df


# ----------------------------------------------------------- orchestrator
def build_order_feature_table() -> pd.DataFrame:
    """One row per (order_id, seller_id), features knowable at purchase time."""
    try:
        cfg = load_config()
        oc = cfg.get("order_model", {})
        lag_days = oc.get("history_lag_days", 30)
        top_n = oc.get("top_categories", 20)

        df = load_orders()
        df = add_promise_features(df)
        df = add_calendar_features(df)
        df = add_shipment_features(df)
        df = add_route_features(df)
        df = add_lagged_history(df, lag_days)
        df = collapse_categories(df, top_n)

        df[TARGET] = df["is_late"].astype(int)
        # Downstream modules split on `as_of_date`.
        df["as_of_date"] = df["order_purchase_timestamp"]

        drop = ["order_purchase_timestamp", "order_estimated_delivery_date",
                "shipping_limit_date", "is_late",
                "customer_zip", "seller_zip", "avg_weight_g", "max_volume_cm3"]
        df = df.drop(columns=[c for c in drop if c in df.columns])

        # Drop orders with a missing or non-positive promise window.
        df = df[df["promised_days"].notna() & (df["promised_days"] > 0)]

        # Leakage guard: fail if any outcome column survived. The target is
        # excluded explicitly.
        outcome_markers = ("order_delivered", "delay_days", "order_approved")
        leaked = [c for c in df.columns
                  if c != TARGET and c.startswith(outcome_markers)]
        if leaked:
            raise ValueError(f"Outcome columns survived into the feature table: {leaked}")

        df = df.sort_values("as_of_date").reset_index(drop=True)
        logger.info(f"Order feature table: {df.shape}, "
                    f"late rate {df[TARGET].mean():.2%}, "
                    f"{df['as_of_date'].min().date()} to {df['as_of_date'].max().date()}")
        return df
    except Exception as e:
        raise SentrixException(e, sys)
