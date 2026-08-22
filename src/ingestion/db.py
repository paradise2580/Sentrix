"""
src/ingestion/db.py

Role
----
One function that builds the SQLAlchemy engine used to talk to MySQL.
Every module that needs a database connection calls get_engine() instead
of constructing its own connection string — so credentials and connection
logic live in exactly one place.

Credential policy
-----------------
Non-secret connection details (host, port, user, database) have defaults
in config.yaml and can be overridden by MYSQL_* environment variables.

The PASSWORD has no config.yaml default on purpose. config.yaml is
committed; .env is not. Keeping a working password in a committed file is
the single most common way a project leaks a credential, and "it's only a
local dev password" stops being true the moment the same file is copied to
a deployment. If MYSQL_PASSWORD is unset, this fails immediately with an
instruction rather than silently trying to connect with None.
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


def _resolve_password() -> str:
    password = os.getenv("MYSQL_PASSWORD")
    if not password:
        raise ValueError(
            "MYSQL_PASSWORD is not set. Copy .env.example to .env and fill it in "
            "(the password is intentionally not stored in config.yaml, which is "
            "committed to version control)."
        )
    return password


def get_engine(force_new: bool = False) -> Engine:
    """Build (or return the cached) SQLAlchemy engine for the SENTRIX MySQL database."""
    global _engine_cache
    if _engine_cache is not None and not force_new:
        return _engine_cache

    try:
        cfg = load_config()["mysql"]

        url = URL.create(
            drivername="mysql+pymysql",
            username=_resolve("MYSQL_USER", "user", cfg),
            password=_resolve_password(),
            host=_resolve("MYSQL_HOST", "host", cfg),
            port=int(_resolve("MYSQL_PORT", "port", cfg)),
            database=_resolve("MYSQL_DATABASE", "database", cfg),
        )

        _engine_cache = create_engine(url, pool_pre_ping=True)
        logger.info(f"MySQL engine created for host={url.host} db={url.database}")
        return _engine_cache

    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    engine = get_engine()
    with engine.connect() as conn:
        print("Database connection OK:", engine.url.database)
