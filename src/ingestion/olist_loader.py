"""
Loads the Olist Brazilian e-commerce CSVs into MySQL and computes the label:

    is_late = order_delivered_customer_date > order_estimated_delivery_date

Run with:
    python -m src.ingestion.olist_loader
"""

from pathlib import Path
import pandas as pd

from src.config_loader import load_config, get_project_root
from src.ingestion.loader import DataLoader
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

# Required files. The customers file is optional and loaded when present.
_FILES = {
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "translation": "product_category_name_translation.csv",
}

_OPTIONAL_FILES = {"customers": "olist_customers_dataset.csv"}


def _olist_dir() -> Path:
    cfg = load_config()
    return get_project_root() / cfg["paths"]["data_raw"] / "olist"


def verify_files_present() -> None:
    """Fail fast with a clear message if any expected CSV is missing."""
    olist_dir = _olist_dir()
    missing = [f for f in _FILES.values() if not (olist_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing Olist file(s) in {olist_dir}:\n  " + "\n  ".join(missing) +
            "\n\nDownload from: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
        )
    logger.info(f"All {len(_FILES)} Olist files found in {olist_dir}")


def load_sellers() -> pd.DataFrame:
    df = pd.read_csv(_olist_dir() / _FILES["sellers"])
    return df[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]]


def load_customers() -> pd.DataFrame | None:
    """Returns None when the customers file isn't present."""
    path = _olist_dir() / _OPTIONAL_FILES["customers"]
    if not path.exists():
        logger.info("customers file not present — skipping (not required for seller-risk modelling)")
        return None
    df = pd.read_csv(path)
    return df[["customer_id", "customer_unique_id", "customer_zip_code_prefix",
               "customer_city", "customer_state"]]


def load_products() -> pd.DataFrame:
    """Products with English category names."""
    products = pd.read_csv(_olist_dir() / _FILES["products"])
    translation = pd.read_csv(_olist_dir() / _FILES["translation"], encoding="utf-8-sig")

    df = products.merge(translation, on="product_category_name", how="left")
    df = df.rename(columns={"product_category_name_english": "product_category_english"})
    df["product_category_english"] = df["product_category_english"].fillna(
        df["product_category_name"]
    )
    return df[["product_id", "product_category_name", "product_category_english",
               "product_weight_g", "product_length_cm", "product_height_cm",
               "product_width_cm"]]


def load_orders() -> pd.DataFrame:
    """
    Load orders and compute is_late. Undelivered orders keep is_late NULL,
    since their outcome is unknown.
    """
    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    df = pd.read_csv(_olist_dir() / _FILES["orders"], parse_dates=date_cols)

    delivered = df["order_delivered_customer_date"].notna() & df["order_estimated_delivery_date"].notna()

    df["delay_days"] = pd.NA
    df.loc[delivered, "delay_days"] = (
        df.loc[delivered, "order_delivered_customer_date"]
        - df.loc[delivered, "order_estimated_delivery_date"]
    ).dt.total_seconds() / 86400.0

    df["is_late"] = pd.NA
    df.loc[delivered, "is_late"] = (df.loc[delivered, "delay_days"] > 0).astype(int)

    late_rate = df.loc[delivered, "is_late"].mean()
    logger.info(
        f"Computed REAL label on {delivered.sum():,} delivered orders "
        f"({len(df) - delivered.sum():,} not yet delivered, left NULL). "
        f"Late rate: {late_rate:.2%}"
    )
    return df[["order_id", "customer_id", "order_status"] + date_cols + ["is_late", "delay_days"]]


def load_order_items() -> pd.DataFrame:
    df = pd.read_csv(_olist_dir() / _FILES["order_items"], parse_dates=["shipping_limit_date"])
    return df[["order_id", "order_item_id", "product_id", "seller_id",
               "shipping_limit_date", "price", "freight_value"]]


def load_reviews() -> pd.DataFrame:
    """Customer reviews."""
    df = pd.read_csv(_olist_dir() / _FILES["reviews"], parse_dates=["review_creation_date"])
    df = df.drop_duplicates(subset=["review_id", "order_id"])
    return df[["review_id", "order_id", "review_score",
               "review_comment_message", "review_creation_date"]]


def load_all_to_mysql(reset: bool = True) -> dict:
    """Load every Olist table into MySQL. Returns row counts per table."""
    try:
        verify_files_present()
        loader = DataLoader()

        if reset:
            for table in ["predictions", "external_signals", "reviews",
                          "order_items", "orders", "products", "customers", "sellers"]:
                loader.truncate_table(table)

        steps = [
            ("sellers", load_sellers),
            ("customers", load_customers),
            ("products", load_products),
            ("orders", load_orders),
            ("order_items", load_order_items),
            ("reviews", load_reviews),
        ]

        counts = {}
        for table, fn in steps:
            logger.info(f"Loading {table}...")
            df = fn()
            if df is None:
                continue
            loader.write_df(df, table, if_exists="append", chunksize=5000)
            counts[table] = len(df)

        logger.info(f"Olist load complete: {counts}")
        return counts
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    counts = load_all_to_mysql(reset=True)
    print("\nLoaded real Olist data into MySQL:")
    for table, n in counts.items():
        print(f"  {table:15s} {n:>8,} rows")
