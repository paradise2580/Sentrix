"""
Builds the order-level feature table and writes it to
data/processed/features.csv, which training and evaluation read.

Run with:
    python scripts/build_order_features.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config_loader import load_config, get_project_root
from src.preprocessing.order_features import build_order_feature_table
from src.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    df = build_order_feature_table()

    out = get_project_root() / cfg["paths"]["data_processed"] / "features.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    target = cfg["model"]["target_column"]
    logger.info(f"Order feature table {df.shape} -> {out}")
    print(f"\nRows:        {len(df):,}")
    print(f"Columns:     {df.shape[1]}")
    print(f"Late rate:   {df[target].mean():.2%}")
    print(f"Date range:  {df['as_of_date'].min()} to {df['as_of_date'].max()}")
    print(f"Saved to:    {out}")


if __name__ == "__main__":
    main()
