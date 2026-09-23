"""
LEGACY: builds the old seller-day feature table (preprocessing/feature_eng.py)
and saves it to data/processed/features.csv. The current model uses
scripts/build_order_features.py.

Run with:
    python scripts/build_features.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing.feature_eng import build_feature_table
from src.config_loader import load_config, get_project_root
from src.logger import get_logger

logger = get_logger(__name__)


def main():
    cfg = load_config()
    features = build_feature_table()

    out_path = get_project_root() / cfg["paths"]["data_processed"] / "features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    logger.info(f"Feature table {features.shape} -> {out_path}")
    print(f"Feature table shape: {features.shape}")
    print(f"Positive rate: {features['high_late_rate_next_30d'].mean():.2%}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
