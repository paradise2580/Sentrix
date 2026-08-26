"""
scripts/label_screen2.py

Asks the question that has to be answered before any more modelling:
is the target PREDICTABLE, or is it noise?

Where this comes from
---------------------
label_screen.py established two things. First, the shipped v1 label was a
volume detector — a forest given only the three order_count_* columns
scored ROC-AUC 0.7317 against the full 47-feature model's 0.6494. Second,
a label that neutralises volume by construction (worst 20% of forward late
rate within month x volume-decile cells) does neutralise it — volume_roc
0.4946, drift 1.59 points — and the model then scores 0.5056. A coin flip.

Two explanations fit that equally well and they demand opposite responses:

  A. There is no seller-level signal. Lateness in this marketplace is
     driven by carrier, distance and calendar, not by stable seller
     quality. The project should then be rebuilt at ORDER level.

  B. There is signal, but the label cannot see it. A forward late rate
     measured over 5 orders has a standard error of
     sqrt(0.08*0.92/5) ~ 0.121, which is likely larger than the true
     spread between sellers. Ranking on that mostly ranks luck. The fix is
     a bigger denominator, not a different feature.

This script separates them with three measurements:

1. SPLIT-HALF RELIABILITY — the ceiling.
   Take the forward window, cut it in half, and ask how well a seller's
   late rate in the first half predicts their rate in the second half.
   Same seller, same period, no modelling, no feature engineering. If
   THAT is near zero, the quantity has no stable seller-level component
   and no model can ever predict it. This is the number that decides
   between A and B, and nothing else in this project can substitute for it.

2. PERSISTENCE AUC — one feature, no model.
   How well does the seller's trailing 30-day late rate rank the forward
   label? A model that cannot beat this single column is not earning its
   complexity.

3. THE min_n SWEEP.
   Both of the above, plus a full forest, at min_forward_orders in
   {5, 10, 20, 30, 50}. If explanation B is right, every number rises
   monotonically with the denominator and the story is "the signal was
   always there, it needed enough orders to become measurable". If they
   stay flat at 0.5, explanation A is right and the seller-day framing is
   finished.

It also dumps the raw table schemas, because the order-level rebuild is
where this goes if the answer is A.

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
    """
    Correlate a seller's late rate in (t, t+15] against (t+15, t+30].

    This is the honest upper bound on any model of the forward rate. The
    two halves are the same seller in the same month under the same
    carriers — if they do not agree with each other, nothing measured
    BEFORE t is going to agree with either.
    """
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
    # Count columns are volume in disguise; with volume held fixed by the
    # label they can only add noise. "rate" features are the real candidates.
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
    """The order-level rebuild needs to know exactly what columns exist."""
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
