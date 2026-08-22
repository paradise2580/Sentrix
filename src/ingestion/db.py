"""
src/ingestion/db.py

Role
----
One function that builds the SQLAlchemy engine used to talk to MySQL.
Every module that needs a database connection calls get_engine() instead
of constructing its own connection string — so credentials and connection
logic live in exactly one place.

Credentials resolve from .env first (for local/deployed secrets), falling
back to config.yaml (for non-secret defaults).
"""

import os
from sqlalchemy import Engine, create_engine
from sqlalchemy.engine.url import URL
from dotenv import load_dotenv

from src.config_loader import load_config
from src.logger import get_logger
from src.exception import SentrixException
import sys

load_dotenv()
logger = get_logger(__name__)

_engine_cache: Engine | None = None


def _resolve(key_env: str, key_cfg: str, cfg_section: dict, default=None):
    """Prefer an environment variable; fall back to config.yaml; then default."""
    return os.getenv(key_env) or cfg_section.get(key_cfg, default)


def get_engine(force_new: bool = False) -> Engine:
    """
    Build (or return the cached) SQLAlchemy engine for the SENTRIX MySQL database.
    """
    global _engine_cache
    if _engine_cache is not None and not force_new:
        return _engine_cache

    try:
        cfg = load_config()["mysql"]

        url = URL.create(
            drivername="mysql+pymysql",
            username=_resolve("MYSQL_USER", "user", cfg),
            password=_resolve("MYSQL_PASSWORD", "password", cfg),
            host=_resolve("MYSQL_HOST", "host", cfg),
            port=int(_resolve("MYSQL_PORT", "port", cfg)),
            database=_resolve("MYSQL_DATABASE", "database", cfg),
        )

        _engine_cache = create_engine(url, pool_pre_ping=True)
        logger.info(f"MySQL engine created for host={cfg.get('host')} db={cfg.get('database')}")
        return _engine_cache

    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    engine = get_engine()
    with engine.connect() as conn:
        print("Database connection OK:", engine.url.database)
