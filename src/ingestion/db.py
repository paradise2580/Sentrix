"""
src/ingestion/db.py

Role
----
One function that builds the SQLAlchemy engine every module talks to the
database through, so credentials and connection logic live in exactly one
place.

Two backends, one interface
---------------------------
- **MySQL** for the full pipeline: ingestion, feature building, training.
  This is where the 100k real Olist orders live.
- **SQLite** for serving, selected by setting SENTRIX_DB_URL.

The serving path reads a few thousand precomputed prediction rows and
never writes. Standing up a managed MySQL for that would be renting a
database to serve a file — and on a free tier the managed Postgres option
*expires after 30 days*, which would silently break a demo link exactly
when someone clicks it. A SQLite file committed alongside the code has no
service to expire, no credentials to leak, and no cold-start penalty.

Credential policy
-----------------
Host, port, user and database have non-secret defaults in config.yaml and
can be overridden by MYSQL_* environment variables. The PASSWORD has no
config.yaml default on purpose: config.yaml is committed, .env is not, and
a working password in a committed file is the most common way a project
leaks a credential. If MYSQL_PASSWORD is unset, this fails immediately
with an instruction rather than trying to connect with None.
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
    """Prefer an environment variable; fall back to config.yaml; then default."""
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
    """
    Build an engine from an explicit SQLAlchemy URL — the serving path.

    A relative sqlite path is resolved against the project root, not the
    working directory, so `uvicorn api.app:app` behaves the same whether it
    is launched from the repo root or from inside a container's WORKDIR.
    """
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url[len("sqlite:///"):]
        if not Path(rel).is_absolute():
            url = f"sqlite:///{(get_project_root() / rel).resolve()}"
    logger.info(f"Serving engine ({SERVING_URL_ENV}): {url}")
    return create_engine(url)


def get_engine(force_new: bool = False) -> Engine:
    """Build (or return the cached) engine for whichever backend is configured."""
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
