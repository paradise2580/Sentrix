"""
scripts/ablation.py

Answers one question, on the same held-out rows every other number in this
project is measured on:

    How much of the model's performance is just "this seller is busy"?

Why this exists
---------------
The label is "does this seller have ANY late delivery in the next 30 days".
That is volume-confounded by construction: a seller shipping 100 orders a
month is nearly certain to have one arrive late; a seller shipping three
often will not. And the SHAP attribution agrees — order_count_30d,
order_count_14d and order_count_7d are the three largest drivers.

So the honest question is not "is the model good" but "is the model better
than counting orders". This script measures exactly that:

  1. ORDER COUNT ALONE      rank sellers by order_count_30d, nothing else
  2. VOLUME-ONLY MODEL      logistic regression on the order_count_* columns
  3. NO-VOLUME MODEL        the champion's features MINUS every count column
  4. FULL MODEL             the champion as shipped

If (1) is close to (4), the model is a volume detector wearing a hat.
If (3) holds up on its own, the risk features carry independent signal.

Run with:
    python scripts/ablation.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config_loader import load_config
from src.models.trainer import prepare_data
from src.evaluation.metrics import evaluate_model
from src.logger import get_logger

logger = get_logger(__name__)

VOLUME_PREFIXES = ("order_count_",)


def _subset(feature_names: list[str], keep) -> list[int]:
    return [i for i, n in enumerate(feature_names) if keep(n)]


def main() -> None:
    cfg = load_config()
    seed = cfg["project"]["random_state"]
    data = prepare_data()

    names = data["feature_names"]
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    is_volume = [n.startswith(VOLUME_PREFIXES) for n in names]
    vol_idx = [i for i, v in enumerate(is_volume) if v]
    non_vol_idx = [i for i, v in enumerate(is_volume) if not v]

    print(f"\nEvaluation set: {len(y_te):,} rows, base rate {y_te.mean():.2%}")
    print(f"Volume features ({len(vol_idx)}): {[names[i] for i in vol_idx]}")

    rows = []

    # 1. Rank by raw order count. No model at all.
    #
    # The column is standardised, so it is a SCORE, not a probability —
    # evaluate_model reports NaN for Brier/ECE here rather than pretending
    # otherwise. Only the ranking metrics are meaningful for this row, which
    # is all the comparison needs.
    if "order_count_30d" in names:
        col = names.index("order_count_30d")
        rows.append(("order_count_30d alone (no model)", evaluate_model(y_te, X_te[:, col])))

    def fit_rf(idx, label):
        rf = RandomForestClassifier(n_estimators=300, max_depth=10,
                                    class_weight="balanced",
                                    random_state=seed, n_jobs=-1)
        rf.fit(X_tr[:, idx], y_tr)
        p = rf.predict_proba(X_te[:, idx])[:, 1]
        rows.append((label, evaluate_model(y_te, p)))

    fit_rf(vol_idx, f"volume features only ({len(vol_idx)})")
    fit_rf(non_vol_idx, f"everything EXCEPT volume ({len(non_vol_idx)})")
    fit_rf(list(range(len(names))), f"full model ({len(names)})")

    table = pd.DataFrame([{
        "variant": label,
        "pr_auc": round(m["pr_auc"], 4),
        "vs_random": round(m["pr_auc_lift"], 2),
        "roc_auc": round(m["roc_auc"], 4),
        "capture@10%": round(m["capture_at_10pct"], 4),
    } for label, m in rows])

    print("\n" + "=" * 78)
    print("ABLATION — how much of the signal is order volume?")
    print("=" * 78)
    print(table.to_string(index=False))

    full = table.loc[table["variant"].str.startswith("full"), "pr_auc"].iloc[0]
    vol = table.loc[table["variant"].str.startswith("volume"), "pr_auc"].iloc[0]
    nonvol = table.loc[table["variant"].str.startswith("everything"), "pr_auc"].iloc[0]

    print(f"\nVolume alone reaches {vol / full:.0%} of the full model's PR-AUC.")
    print(f"Without any volume feature the model still reaches {nonvol / full:.0%}.")
    print("\nRead it this way: the first number is how much a busy-seller detector")
    print("would get you for free; the second is how much the risk features are")
    print("worth on their own. A model is only interesting where both are true —")
    print("volume alone falls short, and the non-volume features stand up.")

    out = Path(__file__).resolve().parent.parent / "artifacts" / "evaluation" / "ablation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    logger.info(f"Ablation table saved to {out}")


if __name__ == "__main__":
    main()
