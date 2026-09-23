"""
Regenerates the README results section (between the RESULTS markers) from
artifacts/evaluation/, so the README can't drift from the real numbers.

Run with:
    python scripts/render_results.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from src.config_loader import load_config, get_project_root
from src.logger import get_logger

logger = get_logger(__name__)

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

PRETTY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost (Optuna-tuned)",
    "lightgbm": "LightGBM (SMOTE)",
    "ensemble": "Stacking Ensemble",
    "lstm": "LSTM (PyTorch)",
}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_results_markdown() -> str:
    eval_dir = get_project_root() / "artifacts" / "evaluation"
    comparison = pd.read_csv(eval_dir / "model_comparison.csv")
    summary = joblib.load(eval_dir / "best_model_summary.joblib")

    cfg = load_config()
    horizon = cfg["preprocessing"]["label_horizon_days"]
    embargo = cfg["preprocessing"]["embargo_days"]

    best = summary["best_model"]
    m = summary["metrics"]
    cal = summary["calibration"]
    cost = summary["cost"]
    n_eval = summary["eval_rows"]
    base = summary["eval_base_rate"]

    lines = [START, ""]

    lines += [
        f"Evaluated on **{n_eval:,} held-out orders** from the final "
        f"chronological block, base rate **{_fmt_pct(base)}**. Train and test are "
        f"separated by a {embargo}-day embargo, so no training label resolves "
        f"inside the evaluation window.",
        "",
        "### Model comparison",
        "",
        "Every model is scored on the *same* rows. PR-AUC depends on the base rate, "
        "so models evaluated on different subsets cannot be ranked against each "
        "other — `compare_models` raises rather than print a table that mixes them.",
        "",
        "| Model | PR-AUC | vs. random | ROC-AUC | KS | Capture @10% | Lift @10% | Brier |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for _, r in comparison.iterrows():
        name = PRETTY.get(r["model"], r["model"])
        mark = "**" if r["model"] == best else ""
        lines.append(
            f"| {mark}{name}{mark} | {mark}{r['pr_auc']:.4f}{mark} | "
            f"{r['pr_auc_lift']:.2f}x | {r['roc_auc']:.4f} | {r['ks_statistic']:.4f} | "
            f"{_fmt_pct(r['capture_at_10pct'])} | {r['lift_at_10pct']:.2f}x | "
            f"{r['brier']:.4f} |"
        )

    lines += [
        "",
        "### What an operations team actually gets",
        "",
        "PR-AUC summarises a curve nobody runs. A team can review a fixed number of "
        "orders per cycle, so the number that matters is how much of the risk they "
        "capture at that capacity.",
        "",
        "| Review the top… | Orders flagged | Of all late orders, caught | vs. random |",
        "|---|---|---|---|",
    ]
    for pct in (5, 10, 20):
        cap = m.get(f"capture_at_{pct}pct")
        lift = m.get(f"lift_at_{pct}pct")
        if cap is None:
            continue
        lines.append(
            f"| {pct}% | {int(round(n_eval * pct / 100)):,} | "
            f"**{_fmt_pct(cap)}** | {lift:.2f}x |"
        )

    lines += [
        "",
        "### Calibration",
        "",
        f"`{PRETTY.get(best, best)}` is calibrated with isotonic regression fitted on the "
        f"held-out calibration block — data the model never trained on and the test set "
        f"never touches.",
        "",
        "| | Raw score | Calibrated |",
        "|---|---|---|",
        f"| Brier score | {cal['brier_raw']:.4f} | **{cal['brier_calibrated']:.4f}** |",
        f"| Expected calibration error | {cal['ece_raw']:.4f} | **{cal['ece_calibrated']:.4f}** |",
        f"| Mean predicted probability | {cal['mean_prediction_raw']:.4f} | "
        f"{cal['mean_prediction_calibrated']:.4f} |",
        "",
        f"Observed rate on the evaluation set: **{cal['observed_base_rate']:.4f}**. "
        "Ranking metrics are invariant to any monotone rescaling of the score, so they "
        "are blind to this entirely — which is why a risk product needs both.",
        "",
        "### Operating point",
        "",
        f"The decision threshold is chosen by minimising expected cost at a "
        f"**{cost['cost_fn']:.0f}:{cost['cost_fp']:.0f}** ratio (a missed late parcel vs. an "
        f"analyst's wasted review), not by defaulting to 0.5 — which silently assumes the "
        f"two errors are equally bad.",
        "",
        f"- Cost-optimal threshold: **{summary['threshold']:.4f}** "
        f"(F1-optimal would be {summary['f1_optimal_threshold']:.4f})",
        f"- Precision {m['precision']:.3f} · Recall {m['recall']:.3f} · F1 {m['f1']:.3f}",
        f"- Expected cost vs. not modelling at all: **{_fmt_pct(cost['cost_reduction'])} lower**",
        f"- Confusion matrix: TP={m['true_positives']:,} FP={m['false_positives']:,} "
        f"FN={m['false_negatives']:,} TN={m['true_negatives']:,}",
        f"- Alert volume at that threshold: "
        f"**{_fmt_pct((m['true_positives'] + m['false_positives']) / n_eval)} of all "
        f"orders**",
        "",
        "That last line is why the capacity view above is the one to run the product "
        "on. A 10:1 cost ratio says false alarms are cheap, so the cost-minimising "
        "threshold alerts on a large share of the population — arithmetically right, "
        "operationally useless. Supply a real cost ratio and re-derive it; supply a "
        "weekly review capacity and read the capture table instead.",
        "",
        f"*Label horizon {horizon} days. Generated by `scripts/render_results.py` from "
        f"`artifacts/evaluation/` — do not edit this section by hand.*",
        "",
        END,
    ]
    return "\n".join(lines)


def main() -> None:
    readme = get_project_root() / "README.md"
    text = readme.read_text(encoding="utf-8")

    if START not in text or END not in text:
        raise SystemExit(f"README.md is missing the {START} / {END} markers.")

    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)

    readme.write_text(head + build_results_markdown() + tail, encoding="utf-8")
    logger.info("README results section regenerated from artifacts/evaluation/")
    print("README.md results section updated.")


if __name__ == "__main__":
    main()
