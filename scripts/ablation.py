"""
scripts/ablation.py

Answers one question, on the same held-out rows every other number in this
project is measured on:

    How much of the model's performance comes from one obvious shortcut?

Why this exists
---------------
The first version of this project labelled "does this seller have ANY late
delivery in the next 30 days". That is volume-confounded by construction: a
seller shipping 100 orders a month is nearly certain to have one arrive
late; a seller shipping three often will not. This script was written to
test that suspicion and it confirmed it — ranking sellers by
`order_count_30d` alone, with no model, scored PR-AUC 0.4455 against the
full 47-feature model's 0.3911. A single column beat the champion by 14%.

That result is why SENTRIX now models individual ORDERS. At order grain the
volume confound is structurally impossible — each row is one parcel, and
order count is not a feature at all.

So this script adapts to whichever grain the feature table is in:

  SELLER GRAIN (the retired configuration, reproducible from here)
    order_count_30d alone / volume-only / everything-except-volume / full

  ORDER GRAIN (current)
    promised_days alone / promise-only / everything-except-promise / full

    The order-level shortcut to worry about is different: `promised_days` is
    the marketplace's OWN delivery estimate, already built from distance and
    carrier. If the model scores no better than that column alone, it has
    learned to read an existing estimate back rather than to improve on it.

In both cases the reading is the same. If the single column is close to the
full model, the model is a shortcut wearing a hat. If the "everything except"
variant holds up on its own, the remaining features carry independent signal.

Run with:
    python scripts/ablation.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config_loader import load_config, get_project_root
from src.models.boosting import build_xgboost
from src.models.trainer import prepare_data
from src.evaluation.metrics import evaluate_model
from src.logger import get_logger

logger = get_logger(__name__)

VOLUME_PREFIXES = ("order_count_",)


def main() -> None:
    cfg = load_config()
    data = prepare_data()

    names = data["feature_names"]
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    vol_idx = [i for i, n in enumerate(names) if n.startswith(VOLUME_PREFIXES)]
    non_vol_idx = [i for i in range(len(names)) if i not in set(vol_idx)]

    print(f"\nEvaluation set: {len(y_te):,} rows, base rate {y_te.mean():.2%}")

    rows = []

    def fit_model(idx, label):
        """
        Fit the CHAMPION model family on a feature subset.

        This used to be a fixed RandomForest, which quietly made the whole
        table meaningless: on the order-level data the forest scores ROC-AUC
        0.5249 — essentially random — while the tuned XGBoost reaches 0.7551.
        Every "full model" row was therefore benchmarking the ablation against
        the worst model in the project, so any subset could look competitive.
        Ablating a model means ablating the model you actually ship.
        """
        params_path = get_project_root() / cfg["paths"]["models"] / "xgboost_best_params.joblib"
        params = joblib.load(params_path) if params_path.exists() else None
        model = build_xgboost(params)
        model.fit(X_tr[:, idx], y_tr)
        p = model.predict_proba(X_te[:, idx])[:, 1]
        rows.append((label, evaluate_model(y_te, p)))

    def single_column(col_name, label):
        """
        Rank by one standardised column, no model at all — in whichever
        DIRECTION actually predicts.

        Sign matters and it is easy to miss. `promised_days` scored ROC-AUC
        0.2011 raw, which reads as "useless" only if you forget that 0.20 is
        as far from chance as 0.80 is: a longer promised window means LESS
        lateness, so the column predicts strongly once negated. Reporting the
        raw number would have understated the most informative single feature
        in the dataset and flattered the model it is compared against.
        """
        if col_name not in names:
            return False
        col = X_te[:, names.index(col_name)]
        if roc_auc_score(y_te, col) < 0.5:
            col, label = -col, f"{label} [negated]"
        rows.append((label, evaluate_model(y_te, col)))
        return True

    if vol_idx:
        # SELLER GRAIN — the original question: is the model a busy-seller
        # detector? This is the configuration that failed, kept so the result
        # can be reproduced.
        print(f"Volume features ({len(vol_idx)}): {[names[i] for i in vol_idx]}")
        single_column("order_count_30d", "order_count_30d alone (no model)")
        fit_model(vol_idx, f"volume features only ({len(vol_idx)})")
        fit_model(non_vol_idx, f"everything EXCEPT volume ({len(non_vol_idx)})")
        headline = "ABLATION — how much of the signal is order volume?"
    else:
        # ORDER GRAIN — volume is not a feature at all, so the confound is
        # structurally absent. The equivalent question is whether the model is
        # just reading back the marketplace's own delivery estimate.
        print("No order_count_* columns — order grain. Volume cannot be a "
              "confound here because each row is a single parcel.")
        single_column("promised_days", "promised_days alone (no model)")
        promise_idx = [i for i, n in enumerate(names)
                       if n.startswith(("promised_days", "shipping_limit_days",
                                        "handover_share"))]
        if promise_idx:
            fit_model(promise_idx, f"promise features only ({len(promise_idx)})")
            rest = [i for i in range(len(names)) if i not in set(promise_idx)]
            fit_model(rest, f"everything EXCEPT promise ({len(rest)})")
        headline = "ABLATION — is the model just reading back the promised window?"

    fit_model(list(range(len(names))), f"full model ({len(names)})")

    table = pd.DataFrame([{
        "variant": label,
        "pr_auc": round(m["pr_auc"], 4),
        "vs_random": round(m["pr_auc_lift"], 2),
        "roc_auc": round(m["roc_auc"], 4),
        "capture@10%": round(m["capture_at_10pct"], 4),
    } for label, m in rows])

    print("\n" + "=" * 78)
    print(headline)
    print("=" * 78)
    print(table.to_string(index=False))

    full = table.loc[table["variant"].str.startswith("full"), "pr_auc"].iloc[0]

    # Match on the exact variant labels, not a prefix. "promise" also matches
    # "promised_days alone (no model)", so the earlier version silently
    # reported the single column where it claimed to report the subset model.
    for needle, phrasing in [("volume features only", "Volume features alone"),
                             ("promise features only", "The promise features alone")]:
        hit = table.loc[table["variant"].str.startswith(needle), "pr_auc"]
        if len(hit):
            print(f"\n{phrasing} reach {hit.iloc[0] / full:.0%} of the "
                  f"full model's PR-AUC.")

    solo = table.loc[table["variant"].str.contains("alone"), :]
    if len(solo):
        r = solo.iloc[0]
        print(f"One column with no model at all ({r['variant']}) reaches "
              f"{r['pr_auc'] / full:.0%} of the full model's PR-AUC "
              f"at ROC-AUC {r['roc_auc']:.4f}.")
    hit = table.loc[table["variant"].str.startswith("everything"), "pr_auc"]
    if len(hit):
        print(f"Without it the model still reaches {hit.iloc[0] / full:.0%}.")

    print("\nRead it this way: the first number is what a single obvious signal")
    print("would get you for free; the second is what the rest of the features")
    print("are worth on their own. A model is only interesting where both are")
    print("true — the shortcut falls short, and the remaining features stand up.")

    out = Path(__file__).resolve().parent.parent / "artifacts" / "evaluation" / "ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logger.info(f"Ablation table saved to {out}")


if __name__ == "__main__":
    main()
