"""tests/test_feature_eng.py — unit tests for the seller-day feature builder."""


from src.preprocessing.feature_eng import (
    build_seller_day_panel, add_delivery_history_features, attach_label,
)


def test_panel_is_continuous_daily_per_seller(sample_seller_day_orders):
    panel = build_seller_day_panel(sample_seller_day_orders, active_min_orders=1)
    for seller_id, sub in panel.groupby("seller_id"):
        gaps = sub["as_of_date"].diff().dropna().dt.days.unique()
        assert set(gaps) <= {1}, f"{seller_id} has date gaps"


def test_panel_drops_low_volume_sellers(sample_seller_day_orders):
    panel = build_seller_day_panel(sample_seller_day_orders, active_min_orders=10_000)
    assert len(panel) == 0


def test_rolling_features_do_not_leak_same_day(sample_seller_day_orders):
    """
    A seller's very first day must have zero prior late count — proving the
    shift(1) guard keeps that day's own outcome out of its own features.
    """
    panel = build_seller_day_panel(sample_seller_day_orders, active_min_orders=1)
    panel = add_delivery_history_features(panel, [7, 14, 30])
    firsts = panel.groupby("seller_id").head(1)
    assert (firsts["late_count_7d"] == 0).all()


def test_label_is_forward_looking(sample_seller_day_orders):
    panel = build_seller_day_panel(sample_seller_day_orders, active_min_orders=1)
    # min_forward_orders=1 because the fixture is tiny; the production value
    # (config.yaml) is higher so a rate is never computed off one order.
    panel = attach_label(panel, horizon_days=30, late_rate_threshold=0.15,
                         min_forward_orders=1)
    labelled = panel.dropna(subset=["high_late_rate_next_30d"])
    assert labelled["high_late_rate_next_30d"].isin([0.0, 1.0]).all()
    # the final horizon of each seller's history has no future to observe
    assert panel["high_late_rate_next_30d"].isna().sum() > 0


def test_label_absent_from_feature_columns(sample_feature_table):
    """The CONFIGURED target and the identifiers must never reach the model."""
    from src.config_loader import load_config
    from src.preprocessing.pipeline import get_feature_columns
    num, cat = get_feature_columns(sample_feature_table)
    assert load_config()["model"]["target_column"] not in num + cat
    assert "seller_id" not in num + cat
    assert "order_id" not in num + cat
