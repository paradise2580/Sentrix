"""
Builds the SQLAlchemy engine every module uses.

- MySQL for the full pipeline (ingestion, features, training).
- SQLite for serving, when SENTRIX_DB_URL is set. The deployed API only
  reads precomputed rows, so a committed SQLite file is enough.

MYSQL_PASSWORD must come from the environment (.env); it is never stored
in config.yaml.
"""

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine.url import URL
from dotenv import load_dotenv

from src.config_loader import load_config, get_project_root
from src.logger import get_logger
from src.exception import SentrixException
import sys

load_dotenv()
logger = get_logger(__name__)

_engine_cache: Engine | None = None

SERVING_URL_ENV = "SENTRIX_DB_URL"


def _resolve(key_env: str, key_cfg: str, cfg_section: dict, default=None):
    """Environment variable first, then config.yaml, then the default."""
    return os.getenv(key_env) or cfg_section.get(key_cfg, default)


def _resolve_password() -> str:
    password = os.getenv("MYSQL_PASSWORD")
    if not password:
        raise ValueError(
            "MYSQL_PASSWORD is not set. Copy .env.example to .env and fill it in "
            "(the password is intentionally not stored in config.yaml, which is "
            "committed to version control). To run against the serving SQLite "
            f"snapshot instead, set {SERVING_URL_ENV}."
        )
    return password


def _serving_engine(url: str) -> Engine:
    """Engine from an explicit URL (serving). Relative sqlite paths resolve from the project root."""
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url[len("sqlite:///"):]
        if not Path(rel).is_absolute():
            url = f"sqlite:///{(get_project_root() / rel).resolve()}"
    logger.info(f"Serving engine ({SERVING_URL_ENV}): {url}")
    return create_engine(url)


def get_engine(force_new: bool = False) -> Engine:
    """Build (or return the cached) engine for the configured backend."""
    global _engine_cache
    if _engine_cache is not None and not force_new:
        return _engine_cache

    try:
        serving_url = os.getenv(SERVING_URL_ENV)
        if serving_url:
            _engine_cache = _serving_engine(serving_url)
            return _engine_cache

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
        print("Database connection OK:", engine.url.database or engine.url)
