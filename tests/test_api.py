"""tests/test_api.py — API tests via FastAPI TestClient (in-process)."""

import pytest
from fastapi.testclient import TestClient

from api.app import app
from tests.conftest import mysql_is_reachable

client = TestClient(app)


def test_health_always_responds():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_root_lists_seller_routes():
    body = client.get("/").json()
    assert "/sellers" in body["endpoints"]
    assert "/chat" in body["endpoints"]


@pytest.mark.skipif(not mysql_is_reachable(), reason="MySQL not reachable")
def test_sellers_endpoint_schema():
    r = client.get("/sellers")
    assert r.status_code == 200
    body = r.json()
    if body:
        assert {"seller_id", "risk_score", "risk_band"} <= body[0].keys()
        assert isinstance(body[0]["seller_id"], str)   # real IDs are hashes


@pytest.mark.skipif(not mysql_is_reachable(), reason="MySQL not reachable")
def test_sellers_filter_by_band():
    body = client.get("/sellers", params={"risk_band": "low"}).json()
    assert all(s["risk_band"] == "low" for s in body)


@pytest.mark.skipif(not mysql_is_reachable(), reason="MySQL not reachable")
def test_explain_unknown_seller_returns_404():
    assert client.get("/explain/does_not_exist_hash").status_code == 404


def test_chat_rejects_short_question():
    assert client.post("/chat", json={"question": "hi"}).status_code == 422


def test_metrics_returns_every_trained_model():
    """The comparison table is non-empty and consistent (no fixed model count)."""
    r = client.get("/metrics")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        comparison = r.json()["comparison"]
        assert len(comparison) >= 4
        for row in comparison:
            assert {"model", "pr_auc", "roc_auc"} <= set(row)
            assert 0.0 <= row["roc_auc"] <= 1.0


@pytest.mark.skipif(not mysql_is_reachable(), reason="MySQL not reachable")
def test_summary_endpoint_returns_portfolio_kpis():
    """/summary returns the dashboard's KPIs in one call."""
    r = client.get("/summary")
    assert r.status_code == 200
    body = r.json()

    assert body["total_sellers"] > 0
    assert set(body["band_counts"]) <= {"low", "medium", "high", "critical"}
    assert sum(body["band_counts"].values()) == body["total_sellers"]
    assert len(body["risk_histogram"]) == 20
    assert 0.0 <= body["avg_risk"] <= 1.0
    assert len(body["by_state"]) > 0


@pytest.mark.skipif(not mysql_is_reachable(), reason="MySQL not reachable")
def test_sellers_endpoint_respects_limit():
    """The limit guard keeps the dashboard from fetching every row at once."""
    body = client.get("/sellers", params={"limit": 10}).json()
    assert len(body) <= 10
