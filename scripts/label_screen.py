"""
scripts/label_screen.py

Picks the target definition from evidence instead of from intuition.

Why this script exists
----------------------
Two label definitions have now been shipped and measured, and both failed
in the same way — by encoding order VOLUME rather than delivery RISK:

  v1  "any late order in the next 30 days"
      Ranking sellers by order_count_30d alone, with no model, scored
      PR-AUC 0.4455 against the full model's 0.3911. The label was a
      busy-seller detector: P(>=1 late | n orders) = 1 - (1-p)^n.

  v2  "forward late RATE >= 15%, among sellers with >= 5 forward orders"
      Worse. Every tabular model landed BELOW 0.5 ROC-AUC on the test
      block (logistic regression: 0.4054). The mechanism is the mirror
      image of v1: at n=5 a 15% threshold still means "at least one late
      order", while at n=100 a seller at the 8% marketplace rate has
      P(rate >= 15%) ~ 0.4%. So the positive rate now FALLS with volume.
      Olist volume grows through 2018, so the training block (17.3%
      positive) and the test block (12.2%) are different regimes, the
      learned relationship inverts, and AUC lands under a coin flip.

The lesson is not "try a third threshold". It is that a label whose base
rate is a function of the denominator will always leak volume, in one
direction or the other. So this script builds the panel ONCE and scores
several candidate labels against the same purged split, reporting for each
the only three numbers that decide the question:

    volume_roc   ROC-AUC of a model given ONLY the order_count_* columns.
                 A sound label sits at ~0.50 here. Anything else means
                 order count alone carries the answer.
    novol_roc    ROC-AUC using every feature EXCEPT volume. This is the
                 real signal — whether delivery history, reviews and peer
                 context predict future reliability.
    drift        |train base rate - test base rate|. A label whose class
                 balance moves between blocks cannot be learned in one and
                 applied in the other.

A candidate is only worth a full retrain if volume_roc is near 0.50,
novol_roc is meaningfully above it, and drift is small.

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
    """Forward-window late count and order count — the raw material of every candidate."""
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
    Empirical-Bayes shrunk rate: (late + a) / (orders + a + b), with the
    prior centred on the marketplace-wide late rate.

    A seller with 1 late out of 3 is not evidence of a 33% late rate; a
    seller with 40 late out of 120 is. Shrinkage encodes exactly that,
    pulling thin denominators toward the marketplace mean instead of
    letting them swing the label.
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
    Standardised excess over the marketplace rate:
        (late - n*p0) / sqrt(n*p0*(1-p0))

    Asks "is this seller worse than the marketplace by more than sampling
    noise explains", which is the question an ops lead actually means. The
    denominator makes the bar scale with n instead of against it.
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
    Relative label: the worst `worst_frac` of forward late rate WITHIN a
    (calendar month x forward-volume decile) cell.

    Volume is held fixed inside every cell, so order count carries no
    information about the label by construction — the ablation should come
    back at 0.50 rather than being argued down. Conditioning on month
    additionally pins the base rate to worst_frac in every period, which is
    what kills the train/test regime shift that sank v2.

    The target it defines is "unusually unreliable FOR A SELLER OF THIS
    SIZE", which is also the useful triage question: a large seller is
    always worth watching on absolute counts, and that is precisely why
    absolute counts make a poor alert queue.
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
    """Purged split, three quick forests, the three numbers that decide it."""
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
