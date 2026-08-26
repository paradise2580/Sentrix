"""
tests/test_split_integrity.py

The tests that protect the project's most important correctness claim:
that no information from the evaluation period reaches the model.

Leakage is not the kind of bug that raises an exception. It makes every
number go UP, which is exactly why it survives code review — it looks like
progress. So it gets tested explicitly, and the tests fail the build.
"""

import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_config

from src.models.trainer import purged_temporal_split
from src.evaluation.metrics import (
    capture_at_k, compare_models, expected_calibration_error,
    assign_risk_bands_by_quantile,
)
from src.evaluation.calibration import fit_calibrator, apply_calibrator

HORIZON = 30


@pytest.fixture
def panel() -> pd.DataFrame:
    """Two years of seller-days across five sellers."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-01-01", periods=730, freq="D")
    rows = []
    for seller in [f"s{i}" for i in range(5)]:
        for d in dates:
            rows.append({"seller_id": seller, "as_of_date": d,
                         "x": rng.normal(), load_config()["model"]["target_column"]: rng.integers(0, 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- split shape
def test_blocks_are_strictly_ordered_in_time(panel):
    train, calib, test = purged_temporal_split(panel, 0.2, 0.1, HORIZON)
    assert train["as_of_date"].max() < calib["as_of_date"].min()
    assert calib["as_of_date"].max() < test["as_of_date"].min()


def test_embargo_gap_is_at_least_the_label_horizon(panel):
    """
    The core guarantee. A label on the last training day resolves
    HORIZON days later; if that date is inside the next block, the two
    blocks share information and the held-out score is inflated.
    """
    train, calib, test = purged_temporal_split(panel, 0.2, 0.1, HORIZON)

    train_to_calib = (calib["as_of_date"].min() - train["as_of_date"].max()).days
    calib_to_test = (test["as_of_date"].min() - calib["as_of_date"].max()).days

    assert train_to_calib >= HORIZON, (
        f"only {train_to_calib}d between train and calib — a training label "
        f"resolves inside the calibration block")
    assert calib_to_test >= HORIZON, (
        f"only {calib_to_test}d between calib and test — a calibration label "
        f"resolves inside the test block")


def test_no_row_appears_in_two_blocks(panel):
    train, calib, test = purged_temporal_split(panel, 0.2, 0.1, HORIZON)
    key = lambda d: set(zip(d["seller_id"], d["as_of_date"]))
    assert not key(train) & key(calib)
    assert not key(calib) & key(test)
    assert not key(train) & key(test)


def test_embargo_actually_discards_rows(panel):
    """A split that keeps every row has not embargoed anything."""
    train, calib, test = purged_temporal_split(panel, 0.2, 0.1, HORIZON)
    assert len(train) + len(calib) + len(test) < len(panel)


def test_zero_embargo_keeps_everything(panel):
    """Control case: with no embargo the three blocks partition the panel."""
    train, calib, test = purged_temporal_split(panel, 0.2, 0.1, 0)
    assert len(train) + len(calib) + len(test) == len(panel)


def test_impossible_embargo_is_rejected(panel):
    """Failing loudly beats silently training on four rows."""
    with pytest.raises(Exception):
        purged_temporal_split(panel, 0.2, 0.1, embargo_days=10_000)


# ------------------------------------------------------- comparison integrity
def test_models_scored_on_different_row_counts_cannot_be_ranked():
    """
    PR-AUC depends on the base rate, so a table mixing models scored on
    different subsets is meaningless. compare_models must refuse rather
    than print it.
    """
    rng = np.random.default_rng(1)
    y_a, p_a = rng.integers(0, 2, 500), rng.random(500)
    y_b, p_b = rng.integers(0, 2, 400), rng.random(400)

    with pytest.raises(Exception):
        compare_models({"a": {"y_true": y_a, "y_prob": p_a},
                        "b": {"y_true": y_b, "y_prob": p_b}})


# ------------------------------------------------------------ ops metrics
def test_capture_at_k_of_a_perfect_ranker_is_maximal():
    y = np.array([1] * 10 + [0] * 90)
    p = np.concatenate([np.ones(10), np.zeros(90)])       # perfect ordering
    out = capture_at_k(y, p, 0.10)
    assert out["capture"] == pytest.approx(1.0)
    assert out["lift"] == pytest.approx(10.0)


def test_capture_at_k_of_a_random_ranker_is_about_k():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 20_000)
    out = capture_at_k(y, rng.random(20_000), 0.20)
    assert out["lift"] == pytest.approx(1.0, abs=0.1)     # no better than chance


# ------------------------------------------------------------- calibration
def test_isotonic_calibration_reduces_calibration_error():
    """
    Simulate the exact failure mode class_weight='balanced' and SMOTE
    produce: correct ranking, systematically inflated probabilities.
    """
    rng = np.random.default_rng(3)
    true_p = rng.beta(2, 8, 8000)                  # true rate ~20%
    y = rng.binomial(1, true_p)
    inflated = np.clip(true_p * 2.5, 0, 1)         # ranking intact, scale wrong

    split = 4000
    calibrator = fit_calibrator(y[:split], inflated[:split])
    corrected = apply_calibrator(calibrator, inflated[split:])

    before = expected_calibration_error(y[split:], inflated[split:])
    after = expected_calibration_error(y[split:], corrected)
    assert after < before, f"calibration made it worse: {before:.4f} -> {after:.4f}"


# ------------------------------------------------------------- risk banding
def test_quantile_bands_fill_every_tier():
    """
    The failure absolute cutoffs caused: a calibrated model whose scores
    never exceed 0.75 left 'critical' permanently empty.
    """
    rng = np.random.default_rng(5)
    probs = rng.beta(2, 8, 5000)                   # nothing above ~0.6
    bands = assign_risk_bands_by_quantile(
        probs, {"critical": 0.95, "high": 0.80, "medium": 0.50})

    counts = pd.Series(bands).value_counts()
    assert set(counts.index) == {"low", "medium", "high", "critical"}
    assert counts["critical"] == pytest.approx(len(probs) * 0.05, rel=0.15)
