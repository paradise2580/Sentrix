"""
src/ingestion/export_snapshots.py

Role
----
DVC versions files, not live database state — so before data can be
DVC-tracked, it needs to exist as a file. This exports the current MySQL
table contents to data/raw/*.csv, giving DVC something concrete to track
a version of. Re-running this after new data lands produces a new file
version DVC can diff and roll back to.

Run with:
    python -m src.ingestion.export_snapshots
"""

from src.ingestion.loader import DataLoader
from src.config_loader import load_config, get_project_root
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


def export_all_snapshots() -> None:
    try:
        cfg = load_config()
        loader = DataLoader()
        raw_dir = get_project_root() / cfg["paths"]["data_raw"]
        raw_dir.mkdir(parents=True, exist_ok=True)

        for table in cfg["mysql"]["tables"].values():
            df = loader.read_table(table)
            out_path = raw_dir / f"{table}.csv"
            df.to_csv(out_path, index=False)
            logger.info(f"Exported {len(df)} rows from '{table}' -> {out_path}")

    except Exception as e:
        raise SentrixException(e, sys)


if __name__ == "__main__":
    export_all_snapshots()
