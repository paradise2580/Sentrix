"""
Model gate for CI: fails (exit 1) if the best model's PR-AUC is below
monitoring.pr_auc_gate in config.yaml. Unit tests check that code runs;
this checks that the model is still good enough.

Run with:
    python scripts/validate_model.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import joblib
from src.config_loader import load_config, get_project_root
from src.logger import get_logger

logger = get_logger(__name__)


def validate_model() -> bool:
    cfg = load_config()
    gate = cfg["monitoring"]["pr_auc_gate"]

    eval_dir = get_project_root() / "artifacts" / "evaluation"
    summary_path = eval_dir / "best_model_summary.joblib"

    if not summary_path.exists():
        print(f"FAIL: no evaluation summary at {summary_path}. "
              f"Run src.evaluation.run_evaluation first.")
        return False

    summary = joblib.load(summary_path)
    best_model = summary["best_model"]
    pr_auc = summary["metrics"]["pr_auc"]

    print(f"Model gate check: '{best_model}' scored PR-AUC={pr_auc:.4f} (threshold={gate})")

    if pr_auc < gate:
        print(f"FAIL: PR-AUC {pr_auc:.4f} is below the required gate of {gate}. Build blocked.")
        return False

    print(f"PASS: PR-AUC {pr_auc:.4f} meets the required gate of {gate}.")
    return True


if __name__ == "__main__":
    passed = validate_model()
    sys.exit(0 if passed else 1)
