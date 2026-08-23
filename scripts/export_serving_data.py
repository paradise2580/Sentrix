"""
scripts/export_serving_data.py

Snapshots everything the deployed app needs into a single SQLite file, and
compacts the vector index so it can be committed.

Why a snapshot rather than a hosted database
--------------------------------------------
The API does no model inference at request time. `/sellers`, `/explain`,
`/summary` and `/metrics` all read rows that generate_predictions.py
computed offline. That means the serving tier needs a few thousand
read-only rows, not a database server.

Free managed Postgres expires after 30 days. A demo link that dies a month
after you put it on your CV is worse than no link. A SQLite file in the
repo has nothing to expire and nothing to leak.

What gets written
-----------------
    data/serving/sentrix.db    predictions + sellers, read-only
    artifacts/chroma/          the vector index, rebuilt clean

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
        out.unlink()          # a fresh file, never an append onto a stale one

    loader = DataLoader()
    engine = create_engine(f"sqlite:///{out}")

    total = 0
    for table in TABLES:
        df = loader.read_table(table)

        # Only the sellers actually scored need to travel. Carrying all 3,095
        # when 1,325 are scored just pads the file with rows no endpoint reads.
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
    Rebuild the vector index from scratch so only one collection is on disk.

    chromadb's delete_collection removes the collection record but leaves
    its segment directory behind. Re-indexing therefore grows the directory
    by ~6 MB every single time — five rebuilds had taken artifacts/chroma to
    46 MB, most of it orphaned. Deleting the directory before rebuilding is
    the only way to actually reclaim it.
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
