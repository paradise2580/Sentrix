"""
Experiment (seller-level model, now retired): scores several candidate
seller-level labels on the same purged split.

For each label it reports:
    volume_roc   ROC-AUC using only order-count columns (want ~0.50)
    novol_roc    ROC-AUC using everything except volume (the real signal)
    drift        |train base rate - test base rate| (want small)

Background and results: docs/DESIGN.md.

Run with:
    python scripts/label_screen.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config_loader import load_config
from src.ingestion.loader import DataLoader
from src.preprocessing.feature_eng import (
    load_seller_day_orders, load_seller_day_reviews, load_signals_wide,
    build_seller_day_panel, add_delivery_history_features, add_review_features,
    add_external_signal_features, add_profile_features,
)
from src.models.trainer import purged_temporal_split
from src.logger import get_logger

logger = get_logger(__name__)

HORIZON = 30
NON_FEATURE = {"seller_id", "as_of_date", "seller_state",
               "orders_n", "late_n", "avg_delay_days",
               "forward_late", "forward_orders", "forward_rate", "y"}


# ------------------------------------------------------------------ panel
def build_panel() -> pd.DataFrame:
    """Everything build_feature_table does, minus the label."""
    cfg = load_config()
    windows = cfg["preprocessing"]["rolling_windows"]
    active_min = cfg["preprocessing"].get("active_min_orders", 10)

    loader = DataLoader()
    sellers = loader.read_table("sellers")
    panel = build_seller_day_panel(load_seller_day_orders(), active_min)
    panel = add_delivery_history_features(panel, windows)
    panel = add_review_features(panel, load_seller_day_reviews(), windows)
    panel = add_external_signal_features(panel, sellers, load_signals_wide(), windows)
    panel = add_profile_features(panel)
    return panel.sort_values(["seller_id", "as_of_date"])


def attach_forward(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward-window late count and order count."""
    def forward_sum(column: str) -> pd.Series:
        rolled = (panel.groupby("seller_id")[column]
                  .rolling(HORIZON, min_periods=1).sum()
                  .reset_index(level=0, drop=True))
        return rolled.groupby(panel["seller_id"]).shift(-HORIZON)

    panel["forward_late"] = forward_sum("late_n")
    panel["forward_orders"] = forward_sum("orders_n")
    panel["forward_rate"] = panel["forward_late"] / panel["forward_orders"].replace(0, np.nan)
    return panel


# ------------------------------------------------------------- candidates
def cand_any_late(p: pd.DataFrame) -> pd.Series:
    """v1, for reference: any late order in the window."""
    y = (p["forward_late"] >= 1).astype(float)
    y[p["forward_orders"].isna() | (p["forward_orders"] < 1)] = np.nan
    return y


def cand_rate(p: pd.DataFrame, thresh: float, min_n: int) -> pd.Series:
    """v2 family: a fixed threshold on the forward rate."""
    y = (p["forward_rate"] >= thresh).astype(float)
    y[p["forward_orders"] < min_n] = np.nan
    y[p["forward_rate"].isna()] = np.nan
    return y


def cand_shrunk(p: pd.DataFrame, thresh: float, strength: float, min_n: int) -> pd.Series:
    """
    Empirical-Bayes shrunk rate (late + a) / (orders + a + b), which pulls
    sellers with few orders toward the marketplace rate.
    """
    p0 = float(p["late_n"].sum() / p["orders_n"].sum())
    a, b = p0 * strength, (1 - p0) * strength
    shrunk = (p["forward_late"] + a) / (p["forward_orders"] + a + b)
    y = (shrunk >= thresh).astype(float)
    y[p["forward_orders"] < min_n] = np.nan
    y[p["forward_orders"].isna()] = np.nan
    return y


def cand_zscore(p: pd.DataFrame, z: float, min_n: int) -> pd.Series:
    """
    Excess lateness over the marketplace rate in standard errors:
        (late - n*p0) / sqrt(n*p0*(1-p0))
    """
    p0 = float(p["late_n"].sum() / p["orders_n"].sum())
    n = p["forward_orders"]
    excess = (p["forward_late"] - n * p0) / np.sqrt(n * p0 * (1 - p0))
    y = (excess >= z).astype(float)
    y[n < min_n] = np.nan
    y[n.isna()] = np.nan
    return y


def cand_stratified(p: pd.DataFrame, worst_frac: float, min_n: int) -> pd.Series:
    """
    Worst `worst_frac` of forward late rate within each (month x forward-volume
    decile) cell, so volume carries no information about the label.
    """
    d = p[["forward_rate", "forward_orders", "as_of_date"]].copy()
    d["month"] = d["as_of_date"].dt.to_period("M")
    eligible = d["forward_orders"] >= min_n

    d["vol_bin"] = np.nan
    d.loc[eligible, "vol_bin"] = (
        d.loc[eligible].groupby("month")["forward_orders"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10,
                                     labels=False, duplicates="drop"))
    )

    y = pd.Series(np.nan, index=p.index)
    grouped = d.loc[eligible].groupby(["month", "vol_bin"])["forward_rate"]
    # rank within cell: 1.0 = worst rate in the cell
    pct = grouped.rank(pct=True, method="average")
    y.loc[pct.index] = (pct >= 1 - worst_frac).astype(float)
    y[p["forward_rate"].isna()] = np.nan
    return y


CANDIDATES = {
    "v1 any_late (shipped, volume-confounded)": lambda p: cand_any_late(p),
    "v2 rate>=15%, min 5 fwd (just failed)": lambda p: cand_rate(p, 0.15, 5),
    "v2b rate>=15%, min 20 fwd": lambda p: cand_rate(p, 0.15, 20),
    "eb shrunk>=15%, k=20, min 3 fwd": lambda p: cand_shrunk(p, 0.15, 20, 3),
    "z-excess >=1.0 sd, min 5 fwd": lambda p: cand_zscore(p, 1.0, 5),
    "stratified worst 20% (month x vol decile)": lambda p: cand_stratified(p, 0.20, 5),
}


# -------------------------------------------------------------- screening
def score_candidate(panel: pd.DataFrame, y: pd.Series, cfg: dict) -> dict | None:
    """Purged split, three quick forests, and the three screening numbers."""
    df = panel.copy()
    df["y"] = y
    df = df.dropna(subset=["y"])
    if df["y"].nunique() < 2 or len(df) < 5_000:
        return None

    pre = cfg["preprocessing"]
    train_df, _, test_df = purged_temporal_split(
        df, test_size=pre["test_size"], calib_size=0.0,
        embargo_days=pre.get("embargo_days", HORIZON),
    )
    if len(test_df) < 1_000 or train_df["y"].nunique() < 2 or test_df["y"].nunique() < 2:
        return None

    feats = [c for c in df.columns
             if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    vol = [c for c in feats if c.startswith("order_count_")]
    novol = [c for c in feats if c not in vol]

    med = train_df[feats].median()
    Xtr, Xte = train_df[feats].fillna(med), test_df[feats].fillna(med)
    ytr, yte = train_df["y"].values, test_df["y"].values

    def rf_auc(cols):
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    class_weight="balanced", n_jobs=-1,
                                    random_state=cfg["project"]["random_state"])
        rf.fit(Xtr[cols], ytr)
        p = rf.predict_proba(Xte[cols])[:, 1]
        return roc_auc_score(yte, p), average_precision_score(yte, p)

    vol_roc, _ = rf_auc(vol)
    novol_roc, novol_pr = rf_auc(novol)
    full_roc, full_pr = rf_auc(feats)

    base = yte.mean()
    return {
        "rows": len(df),
        "labelled_%": 100 * len(df) / len(panel),
        "train_base": ytr.mean(),
        "test_base": base,
        "drift": abs(ytr.mean() - base),
        "volume_roc": vol_roc,
        "novol_roc": novol_roc,
        "full_roc": full_roc,
        "full_pr_lift": full_pr / base,
    }


def main() -> None:
    cfg = load_config()
    logger.info("Building the seller-day panel once...")
    panel = attach_forward(build_panel())
    logger.info(f"Panel: {panel.shape}")

    rows = {}
    for name, fn in CANDIDATES.items():
        logger.info(f"Screening: {name}")
        try:
            res = score_candidate(panel, fn(panel), cfg)
        except Exception as exc:                      # noqa: BLE001
            logger.warning(f"  {name}: failed — {exc}")
            continue
        if res is None:
            logger.warning(f"  {name}: not enough labelled rows to screen")
            continue
        rows[name] = res

    table = pd.DataFrame(rows).T
    for col in ["train_base", "test_base", "drift"]:
        table[col] = (100 * table[col]).round(2)
    for col in ["volume_roc", "novol_roc", "full_roc", "full_pr_lift"]:
        table[col] = table[col].round(4)
    table["rows"] = table["rows"].astype(int)
    table["labelled_%"] = table["labelled_%"].round(1)

    print("\n" + "=" * 108)
    print("LABEL SCREEN — base rates in %, ROC from a 200-tree forest on the same purged split")
    print("=" * 108)
    print(table.to_string())
    print("\nWhat to look for:")
    print("  volume_roc  ~0.50 means order count alone cannot answer the label. This is the gate.")
    print("  novol_roc   how much the risk features are worth once volume cannot help.")
    print("  drift       train-vs-test base rate gap; large values mean the blocks are")
    print("              different regimes and a model fitted on one will invert on the other.")

    out = Path(__file__).resolve().parent.parent / "artifacts" / "evaluation" / "label_screen.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    logger.info(f"Label screen saved to {out}")


if __name__ == "__main__":
    main()
