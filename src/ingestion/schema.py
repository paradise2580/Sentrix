"""
src/ingestion/schema.py

Role
----
Defines and creates the MySQL tables SENTRIX runs on.

Data provenance (IMPORTANT — this distinction is deliberate and load-bearing)
-----------------------------------------------------------------------------
REAL tables      — loaded verbatim from the Olist Brazilian E-Commerce public
                   dataset (real marketplace: ~100k orders, ~3k sellers,
                   2016-2018). Sellers act as the "suppliers" whose delivery
                   risk this project predicts. The late-delivery label is
                   COMPUTED FROM REAL DATES, never invented.
SYNTHETIC table  — external_signals only. No public dataset pairs real orders
                   with real daily weather / port-congestion / commodity data
                   for each seller's location, so that layer is generated.
                   Every row carries is_synthetic=1 so it can never be
                   silently mistaken for real data.
"""

from sqlalchemy import text
from src.ingestion.db import get_engine
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)

_DDL_STATEMENTS = {
    # ---------------- REAL (Olist) ----------------
    "sellers": """
        CREATE TABLE IF NOT EXISTS sellers (
            seller_id               VARCHAR(50) PRIMARY KEY,
            seller_zip_code_prefix  INT,
            seller_city             VARCHAR(100),
            seller_state            VARCHAR(10),
            INDEX idx_seller_state (seller_state)
        ) ENGINE=InnoDB;
    """,
    "customers": """
        CREATE TABLE IF NOT EXISTS customers (
            customer_id                VARCHAR(50) PRIMARY KEY,
            customer_unique_id         VARCHAR(50),
            customer_zip_code_prefix   INT,
            customer_city              VARCHAR(100),
            customer_state             VARCHAR(10)
        ) ENGINE=InnoDB;
    """,
    "products": """
        CREATE TABLE IF NOT EXISTS products (
            product_id                  VARCHAR(50) PRIMARY KEY,
            product_category_name       VARCHAR(100),
            product_category_english    VARCHAR(100),
            product_weight_g            FLOAT,
            product_length_cm           FLOAT,
            product_height_cm           FLOAT,
            product_width_cm            FLOAT,
            INDEX idx_category (product_category_english)
        ) ENGINE=InnoDB;
    """,
    "orders": """
        CREATE TABLE IF NOT EXISTS orders (
            order_id                        VARCHAR(50) PRIMARY KEY,
            customer_id                     VARCHAR(50),
            order_status                    VARCHAR(30),
            order_purchase_timestamp        DATETIME,
            order_approved_at               DATETIME NULL,
            order_delivered_carrier_date    DATETIME NULL,
            order_delivered_customer_date   DATETIME NULL,
            order_estimated_delivery_date   DATETIME NULL,
            -- REAL label, computed from the two real date columns above:
            -- 1 when actual delivery ran past the estimate given to the customer.
            is_late                         TINYINT NULL,
            delay_days                      FLOAT NULL,
            INDEX idx_purchase_date (order_purchase_timestamp),
            INDEX idx_is_late (is_late)
        ) ENGINE=InnoDB;
    """,
    "order_items": """
        CREATE TABLE IF NOT EXISTS order_items (
            order_id             VARCHAR(50),
            order_item_id        INT,
            product_id           VARCHAR(50),
            seller_id            VARCHAR(50),
            shipping_limit_date  DATETIME NULL,
            price                FLOAT,
            freight_value        FLOAT,
            PRIMARY KEY (order_id, order_item_id),
            INDEX idx_seller (seller_id),
            INDEX idx_product (product_id)
        ) ENGINE=InnoDB;
    """,
    "reviews": """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id                VARCHAR(50),
            order_id                 VARCHAR(50),
            review_score             INT,
            review_comment_message   TEXT,
            review_creation_date     DATETIME NULL,
            PRIMARY KEY (review_id, order_id),
            INDEX idx_order (order_id),
            INDEX idx_score (review_score)
        ) ENGINE=InnoDB;
    """,
    # ---------------- SYNTHETIC (clearly flagged) ----------------
    "external_signals": """
        CREATE TABLE IF NOT EXISTS external_signals (
            signal_id     BIGINT PRIMARY KEY AUTO_INCREMENT,
            seller_state  VARCHAR(10) NOT NULL,
            signal_date   DATE NOT NULL,
            source        ENUM('weather','port','commodity') NOT NULL,
            value         FLOAT,
            -- Always 1. External risk signals are generated, never real:
            -- no public dataset pairs real orders with real daily
            -- weather/port/commodity data per location.
            is_synthetic  TINYINT NOT NULL DEFAULT 1,
            INDEX idx_state_date_source (seller_state, signal_date, source)
        ) ENGINE=InnoDB;
    """,
    # ---------------- MODEL OUTPUT ----------------
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id  BIGINT PRIMARY KEY AUTO_INCREMENT,
            seller_id      VARCHAR(50) NOT NULL,
            risk_score     FLOAT NOT NULL,
            risk_band      ENUM('low','medium','high','critical') NOT NULL,
            model_name     VARCHAR(100),
            top_features   JSON,
            predicted_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_seller (seller_id)
        ) ENGINE=InnoDB;
    """,
}

_CREATION_ORDER = [
    "sellers", "customers", "products", "orders",
    "order_items", "reviews", "external_signals", "predictions",
]


def create_all_tables() -> None:
    """Create every SENTRIX table if it doesn't already exist."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            for table_name in _CREATION_ORDER:
                conn.execute(text(_DDL_STATEMENTS[table_name]))
                logger.info(f"Table ensured: {table_name}")
    except Exception as e:
        raise SentrixException(e, sys)


def drop_all_tables() -> None:
    """Drop all tables — used for a clean rebuild during development."""
    try:
        engine = get_engine()
        with engine.begin() as conn:
            conn.execute(text("SET FOREIGN_KEY_CHECKS=0"))
            for table_name in reversed(_CREATION_ORDER):
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                logger.info(f"Table dropped: {table_name}")
            # legacy tables from the pre-Olist synthetic schema
            for legacy in ["suppliers", "disruption_events"]:
                conn.execute(text(f"DROP TABLE IF EXISTS {legacy}"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS=1"))
    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    create_all_tables()
    print("All SENTRIX tables created successfully.")
