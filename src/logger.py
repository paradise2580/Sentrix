"""
src/logger.py

Role
----
Give every script in SENTRIX a consistent, timestamped way to record what
it's doing, instead of scattered print() statements. When something breaks
in production (or mid-demo in an interview), the logs tell you exactly
where and when.

Usage in any module
--------------------
    from src.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded 12,400 rows from MySQL")
"""

import logging
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

# One log file per process run, so a single training run or API session
# has its own traceable file rather than one giant ever-growing log.
_LOG_FILE = _LOG_DIR / f"sentrix_{datetime.now():%Y%m%d_%H%M%S}.log"

_configured = False


def _configure_root_logger() -> None:
    """Set up the root logger once per process. Idempotent."""
    global _configured
    if _configured:
        return

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(_LOG_FILE)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger (conventionally __name__ of the calling module),
    writing to both the current run's log file and the console.
    """
    _configure_root_logger()
    return logging.getLogger(name)


if __name__ == "__main__":
    logger = get_logger(__name__)
    logger.info("Logger self-test: this line should appear in console AND in logs/*.log")
    logger.warning("Logger self-test: warning level works")
    print(f"\nLog file for this run: {_LOG_FILE}")
