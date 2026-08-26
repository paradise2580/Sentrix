"""
tests/test_models.py — unit tests for src/models/baseline.py, boosting.py,
ensemble.py. Uses a small synthetic feature table (sample_feature_table
fixture) for fast smoke-testing of the fit/predict wiring — NOT a full
training run on the real feature table (that's the trainer's job, run
explicitly via the trainer, not on every test invocation).
"""

import numpy as np

from src.config_loader import load_config
from src.models.baseline import build_logistic_regression, build_random_forest
from src.models.boosting import build_xgboost, build_lightgbm
from src.models.ensemble import build_ensemble


def _xy_from_sample(sample_feature_table):
    feature_cols = [
        "late_count_7d", "late_rate_7d", "order_count_30d", "days_since_last_late",
        "review_score_avg_7d", "port_congestion_7d", "weather_severity_7d",
        "lifetime_late_rate",
    ]
    X = sample_feature_table[feature_cols].values
    y = sample_feature_table[load_config()["model"]["target_column"]].values
    return X, y


def test_logistic_regression_builds_with_correct_params():
    model = build_logistic_regression()
    assert model.class_weight == "balanced"
    assert model.max_iter == 1000


def test_random_forest_builds_with_correct_params():
    model = build_random_forest()
    assert model.class_weight == "balanced"
    assert model.n_estimators == 300


def test_logistic_regression_fits_and_predicts_valid_probabilities(sample_feature_table):
    X, y = _xy_from_sample(sample_feature_table)
    model = build_logistic_regression()
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    assert probs.shape[0] == len(y)
    assert np.all((probs >= 0) & (probs <= 1))


def test_random_forest_fits_and_predicts_valid_probabilities(sample_feature_table):
    X, y = _xy_from_sample(sample_feature_table)
    model = build_random_forest()
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))


def test_xgboost_fits_and_predicts_valid_probabilities(sample_feature_table):
    X, y = _xy_from_sample(sample_feature_table)
    model = build_xgboost()
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))


def test_lightgbm_fits_and_predicts_valid_probabilities(sample_feature_table):
    X, y = _xy_from_sample(sample_feature_table)
    model = build_lightgbm()
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))


def test_ensemble_combines_base_models_and_predicts(sample_feature_table):
    X, y = _xy_from_sample(sample_feature_table)
    ensemble = build_ensemble([
        ("logreg", build_logistic_regression()),
        ("rf", build_random_forest()),
    ])
    ensemble.fit(X, y)

    probs = ensemble.predict_proba(X)[:, 1]
    assert np.all((probs >= 0) & (probs <= 1))
