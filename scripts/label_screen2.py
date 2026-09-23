"""
Experiment (seller-level model, now retired): is a seller's future late
rate predictable at all?

  1. Split-half reliability: does a seller's late rate in days 1-15 of a
     window match days 16-30? This is the ceiling for any model.
  2. Persistence AUC: does the trailing 30-day late rate rank the forward
     label?
  3. Both, plus a forest, for min_forward_orders in {5, 10, 20, 30, 50}.

The answer (the effect exists within a month but doesn't carry into the
next) is why the project switched to order-level prediction. See
docs/DESIGN.md.

Run with:
    python scripts/label_screen2.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))       # so label_screen imports

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from src.config_loader import load_config
from src.ingestion.loader import DataLoader
from src.logger import get_logger
from label_screen import build_panel, attach_forward, NON_FEATURE
from src.models.trainer import purged_temporal_split

logger = get_logger(__name__)

HORIZON = 30
HALF = HORIZON // 2
MIN_N_SWEEP = [5, 10, 20, 30, 50]


# --------------------------------------------------------------- ceiling
def split_half_reliability(panel: pd.DataFrame, min_half: int) -> dict:
    """Correlation of a seller's late rate in (t, t+15] vs (t+15, t+30]."""
    def window_sum(col: str, start: int, end: int) -> pd.Series:
        width = end - start
        rolled = (panel.groupby("seller_id")[col]
                  .rolling(width, min_periods=1).sum()
                  .reset_index(level=0, drop=True))
        return rolled.groupby(panel["seller_id"]).shift(-end)

    first_late, first_ord = window_sum("late_n", 0, HALF), window_sum("orders_n", 0, HALF)
    second_late, second_ord = window_sum("late_n", HALF, HORIZON), window_sum("orders_n", HALF, HORIZON)

    ok = (first_ord >= min_half) & (second_ord >= min_half)
    r1 = (first_late / first_ord)[ok]
    r2 = (second_late / second_ord)[ok]
    both = pd.concat([r1, r2], axis=1).dropna()
    if len(both) < 500:
        return {"n": len(both), "pearson": np.nan, "spearman": np.nan}

    return {
        "n": len(both),
        "pearson": float(stats.pearsonr(both.iloc[:, 0], both.iloc[:, 1])[0]),
        "spearman": float(stats.spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]),
    }


# ------------------------------------------------------------ label + fit
def stratified_label(panel: pd.DataFrame, worst_frac: float, min_n: int) -> pd.Series:
    """Worst `worst_frac` of forward rate within (month x forward-volume decile)."""
    d = panel[["forward_rate", "forward_orders", "as_of_date"]].copy()
    d["month"] = d["as_of_date"].dt.to_period("M")
    eligible = d["forward_orders"] >= min_n

    d["vol_bin"] = np.nan
    d.loc[eligible, "vol_bin"] = (
        d.loc[eligible].groupby("month")["forward_orders"]
        .transform(lambda s: pd.qcut(s.rank(method="first"), 10,
                                     labels=False, duplicates="drop"))
    )

    y = pd.Series(np.nan, index=panel.index)
    pct = d.loc[eligible].groupby(["month", "vol_bin"])["forward_rate"].rank(
        pct=True, method="average")
    y.loc[pct.index] = (pct >= 1 - worst_frac).astype(float)
    y[panel["forward_rate"].isna()] = np.nan
    return y


def evaluate_at(panel: pd.DataFrame, min_n: int, cfg: dict) -> dict | None:
    df = panel.copy()
    df["y"] = stratified_label(panel, 0.20, min_n)
    df = df.dropna(subset=["y"])
    if len(df) < 3_000:
        return None

    pre = cfg["preprocessing"]
    train_df, _, test_df = purged_temporal_split(
        df, test_size=pre["test_size"], calib_size=0.0,
        embargo_days=pre.get("embargo_days", HORIZON))
    if len(test_df) < 500 or test_df["y"].nunique() < 2:
        return None

    feats = [c for c in df.columns
             if c not in NON_FEATURE and pd.api.types.is_numeric_dtype(df[c])]
    # Only rate features: with volume fixed by the label, counts add noise.
    rate_feats = [c for c in feats if not c.startswith(("order_count_", "late_count_",
                                                        "bad_reviews_"))]

    med = train_df[feats].median()
    Xtr, Xte = train_df[feats].fillna(med), test_df[feats].fillna(med)
    ytr, yte = train_df["y"].values, test_df["y"].values

    def rf_auc(cols):
        rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                    class_weight="balanced", n_jobs=-1,
                                    random_state=cfg["project"]["random_state"])
        rf.fit(Xtr[cols], ytr)
        return roc_auc_score(yte, rf.predict_proba(Xte[cols])[:, 1])

    # Single-column persistence: does yesterday's rate rank tomorrow's?
    persist = np.nan
    if "late_rate_30d" in test_df.columns:
        col = test_df["late_rate_30d"].fillna(train_df["late_rate_30d"].median())
        persist = roc_auc_score(yte, col)

    rel = split_half_reliability(panel, max(3, min_n // 2))

    return {
        "rows": len(df),
        "test_rows": len(test_df),
        "test_base_%": round(100 * yte.mean(), 2),
        "splithalf_r": round(rel["pearson"], 4),
        "splithalf_rho": round(rel["spearman"], 4),
        "persistence_auc": round(persist, 4),
        "rate_feats_auc": round(rf_auc(rate_feats), 4),
        "all_feats_auc": round(rf_auc(feats), 4),
    }


def dump_schemas(loader: DataLoader) -> None:
    """Print table schemas (input for the order-level rebuild)."""
    print("\n" + "=" * 78)
    print("RAW TABLE SCHEMAS")
    print("=" * 78)
    for table in ["orders", "order_items", "products", "customers", "sellers"]:
        try:
            desc = loader.read_query(f"DESCRIBE {table}")
            cols = ", ".join(f"{r.Field}" for r in desc.itertuples())
            print(f"\n{table}:\n  {cols}")
        except Exception as exc:                                   # noqa: BLE001
            print(f"\n{table}: unavailable — {exc}")


def main() -> None:
    cfg = load_config()
    panel = attach_forward(build_panel())
    logger.info(f"Panel: {panel.shape}")

    rows = {}
    for min_n in MIN_N_SWEEP:
        logger.info(f"Evaluating stratified label at min_forward_orders={min_n}")
        res = evaluate_at(panel, min_n, cfg)
        if res is None:
            logger.warning(f"  min_n={min_n}: too few labelled rows")
            continue
        rows[f"min_fwd={min_n}"] = res

    table = pd.DataFrame(rows).T
    print("\n" + "=" * 100)
    print("IS THE TARGET PREDICTABLE? stratified worst-20% label, swept by denominator")
    print("=" * 100)
    print(table.to_string())
    print("""
How to read this:
  splithalf_r      the CEILING. Same seller, same month, first half of the window
                   vs second half. Near 0 => the forward rate has no stable
                   seller-level component and no model can predict it.
  persistence_auc  trailing 30d late rate used directly as the score, no model.
  rate_feats_auc   forest on rate/ratio features only (counts excluded).
  all_feats_auc    forest on everything.

If splithalf_r climbs with the denominator and the AUCs follow it, the signal was
real and the old label was too noisy to show it. If splithalf_r stays near zero at
min_fwd=50, seller-level forward lateness is not a predictable quantity in this
dataset and the project belongs at order level.""")

    out = Path(__file__).resolve().parent.parent / "artifacts" / "evaluation" / "predictability.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out)
    logger.info(f"Saved to {out}")

    dump_schemas(DataLoader())


if __name__ == "__main__":
    main()
