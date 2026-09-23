"""
Prepares the files the deployed app serves:

    data/serving/sentrix.db    predictions + scored sellers (SQLite)
    artifacts/chroma/          the vector index, rebuilt clean

The API does no inference at request time, so a read-only SQLite file is
enough and nothing expires on a free host.

Run with:
    python scripts/export_serving_data.py
"""

import shutil
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy import create_engine

from src.config_loader import load_config, get_project_root
from src.ingestion.loader import DataLoader
from src.logger import get_logger

logger = get_logger(__name__)

TABLES = ["predictions", "sellers"]
OUT_REL = Path("data") / "serving" / "sentrix.db"


def export_sqlite() -> Path:
    """Copy the tables the API reads from MySQL into a fresh SQLite file."""
    out = get_project_root() / OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()          # always start from a fresh file

    loader = DataLoader()
    engine = create_engine(f"sqlite:///{out}")

    total = 0
    for table in TABLES:
        df = loader.read_table(table)

        # Only export sellers that were scored.
        if table == "sellers" and "predictions" in TABLES:
            scored = set(loader.read_table("predictions")["seller_id"])
            df = df[df["seller_id"].isin(scored)]

        df.to_sql(table, engine, index=False, if_exists="replace")
        logger.info(f"  {table}: {len(df):,} rows")
        total += len(df)

    size_kb = out.stat().st_size / 1024
    logger.info(f"SQLite snapshot: {total:,} rows, {size_kb:,.0f} KB -> {out}")
    return out


def compact_chroma() -> None:
    """
    Delete and rebuild the vector index. delete_collection leaves old files
    on disk, so deleting the directory is the only way to reclaim space.
    """
    persist = get_project_root() / load_config()["paths"]["chroma_db"]
    if persist.exists():
        before = sum(f.stat().st_size for f in persist.rglob("*") if f.is_file())
        shutil.rmtree(persist)
        logger.info(f"Cleared {before / 1e6:.1f} MB of vector index (orphaned segments included)")

    from src.rag.indexer import build_index
    build_index(reset=True)

    after = sum(f.stat().st_size for f in persist.rglob("*") if f.is_file())
    logger.info(f"Vector index rebuilt: {after / 1e6:.1f} MB")


def main() -> None:
    logger.info("Exporting serving snapshot...")
    db = export_sqlite()
    compact_chroma()

    print("\nServing snapshot ready:")
    print(f"  {db.relative_to(get_project_root())}")
    print(f"  {Path(load_config()['paths']['chroma_db'])}")
    print("\nThe API reads these when SENTRIX_DB_URL is set, e.g.")
    print(f"  SENTRIX_DB_URL=sqlite:///{OUT_REL.as_posix()}")


if __name__ == "__main__":
    main()
