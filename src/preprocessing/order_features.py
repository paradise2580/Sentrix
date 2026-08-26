"""
src/preprocessing/order_features.py

Builds the ORDER-level feature table: one row per (order_id, seller_id),
labelled with whether that order was delivered after its promised date.

Why the project models orders and not seller-days
-------------------------------------------------
The first two versions of SENTRIX predicted a seller-level target — "will
this seller have trouble in the next 30 days" — and both failed, in ways
that were measured rather than guessed:

  1. scripts/ablation.py showed the v1 "any late order" label was a volume
     detector. A forest given ONLY the three order_count_* columns scored
     ROC-AUC 0.7317; the full 47-feature model scored 0.6494. One column
     beat the model.

  2. scripts/label_screen.py built a label that neutralises volume by
     construction (worst 20% of forward late rate within month x
     volume-decile cells). It worked as designed — volume_roc 0.4946, base
     rate drift 1.59 points — and the model then scored 0.5056. Removing
     the confound removed the performance with it.

  3. scripts/label_screen2.py found out why. Split-half reliability — a
     seller's late rate in days 1-15 of the window against their own rate
     in days 16-30 — climbs from 0.21 to 0.39 as the denominator grows, so
     a real seller effect exists. But persistence AUC, the trailing 30-day
     late rate used directly as a score, sits at 0.46-0.52 at EVERY
     denominator. The seller effect is contemporaneous and does not carry
     across the window boundary.

     In plain terms: a seller who is running late this month is genuinely
     running late all month, and that tells you almost nothing about next
     month. Lateness here is a shock — a bad batch, a carrier problem, a
     demand spike — not a stable seller trait. No feature set predicts a
     target that does not persist.

The order is a different question, and a well-posed one. "Will THIS parcel
miss THIS promised date" is answered by things known the moment it is
placed: how much slack the promise leaves, how far it has to travel, how
heavy it is, what it cost to ship. Those are properties of the shipment,
not forecasts of a seller's future mood.

Seller-level risk does not disappear — it is recovered by AGGREGATING
predicted order risk up to the seller (see evaluation/generate_predictions).
Modelling the unit where the signal lives and deciding on the unit where
the decision is made is the correct split of the problem, and it also
dissolves the volume confound for free: a seller's risk becomes the MEAN
predicted risk of their open orders, which is a rate and cannot be inflated
by shipping more.

Leakage control
---------------
Every feature is knowable at PURCHASE time. Explicitly excluded, and
listed here because each one is individually tempting:

    order_approved_at              known hours later, not at purchase
    order_delivered_carrier_date   the handover — happens after
    order_delivered_customer_date  this IS the outcome
    delay_days                     this IS the outcome, in days

The seller's own history is included but LAGGED by
`history_lag_days`. A seller's late rate cannot be computed from orders
whose delivery outcome is not yet known, and an order purchased today has
no outcome for roughly two weeks. Using an unlagged expanding mean would
hand the model outcomes that, in production, would not exist yet.
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

# Only features knowable at purchase time reach the model; see module docstring.
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

    # MySQL returns SUM()/AVG() over DECIMAL columns as Python Decimal, which
    # pandas stores as dtype=object. Left alone, get_feature_columns would
    # classify total_price as CATEGORICAL and one-hot encode several thousand
    # distinct prices. Cast explicitly rather than trusting the driver.
    for col in ["n_items", "total_price", "total_freight", "avg_weight_g",
                "max_volume_cm3", "customer_zip", "seller_zip", "is_late"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(f"Loaded {len(df):,} order-seller rows")
    return df


# --------------------------------------------------------------- features
def add_promise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slack in the promise, and the handover deadline.

    `promised_days` is the single most informative thing known at purchase:
    the marketplace sets the estimate per order from distance and carrier,
    so a tight promise on a long route is the shape of a late delivery
    before anything has shipped.
    """
    purchase = df["order_purchase_timestamp"]
    df["promised_days"] = (df["order_estimated_delivery_date"] - purchase).dt.total_seconds() / 86400
    df["shipping_limit_days"] = (df["shipping_limit_date"] - purchase).dt.total_seconds() / 86400
    # How much of the promised window the seller is allowed to spend before
    # even handing the parcel over.
    df["handover_share"] = df["shipping_limit_days"] / df["promised_days"].replace(0, np.nan)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Congestion is seasonal and weekly; both are known at purchase."""
    purchase = df["order_purchase_timestamp"]
    df["purchase_hour"] = purchase.dt.hour
    df["purchase_dow"] = purchase.dt.dayofweek
    df["purchase_month"] = purchase.dt.month
    df["purchase_is_weekend"] = (purchase.dt.dayofweek >= 5).astype(int)
    return df


def add_shipment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physical and commercial properties of the parcel itself."""
    df["freight_ratio"] = df["total_freight"] / df["total_price"].replace(0, np.nan)
    df["price_per_item"] = df["total_price"] / df["n_items"].replace(0, np.nan)
    df["density"] = df["avg_weight_g"] / df["max_volume_cm3"].replace(0, np.nan)
    df["log_weight"] = np.log1p(df["avg_weight_g"].clip(lower=0))
    df["log_volume"] = np.log1p(df["max_volume_cm3"].clip(lower=0))
    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distance proxy from Brazilian postcode prefixes.

    CEP prefixes are allocated geographically — the first digits move
    broadly north-west to south-east — so the gap between a seller's and a
    customer's prefix is a usable stand-in for distance without needing a
    geocoding table. It is a proxy, not kilometres, and is labelled as one.
    """
    df["same_state"] = (df["seller_state"] == df["customer_state"]).astype(int)
    df["zip_gap"] = (df["customer_zip"].astype(float) - df["seller_zip"].astype(float)).abs()
    # Plain float, not the nullable Int64 dtype — SimpleImputer and
    # StandardScaler choke on pd.NA, and a missing prefix must survive as NaN
    # for the median imputer to handle it.
    df["seller_zip_region"] = df["seller_zip"].astype(float) // 1000
    df["customer_zip_region"] = df["customer_zip"].astype(float) // 1000
    return df


def add_lagged_history(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    """
    The seller's and the route's observed late rate, as of `lag_days` before
    this order was placed.

    The lag is the whole point. An order placed today has no delivery
    outcome for roughly two weeks, so a seller's "current" late rate is not
    knowable at purchase time. Computing an expanding mean without the lag
    would feed the model outcomes that do not exist yet in production —
    the same class of error as the volume confound, just harder to see.

    Implementation: aggregate outcomes per (key, day), take a cumulative
    sum over days, then as-of join each order to the cumulative state at
    (purchase_date - lag_days). merge_asof does the backward lookup in one
    pass instead of a per-row scan.
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
    """
    Keep the `top_n` most common product categories; everything else becomes
    'other'. 74 one-hot columns of which most fire a handful of times is
    variance, not signal.
    """
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
        # `as_of_date` is the name every downstream module splits on.
        df["as_of_date"] = df["order_purchase_timestamp"]

        drop = ["order_purchase_timestamp", "order_estimated_delivery_date",
                "shipping_limit_date", "is_late",
                "customer_zip", "seller_zip", "avg_weight_g", "max_volume_cm3"]
        df = df.drop(columns=[c for c in drop if c in df.columns])

        # A promise that has already expired, or a missing timestamp, is not a
        # scorable order.
        df = df[df["promised_days"].notna() & (df["promised_days"] > 0)]

        # The guard must not flag the target itself — `is_late_delivery`
        # contains "deliver", and an earlier version of this check failed the
        # build on its own label.
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
