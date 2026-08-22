"""
tests/test_dashboard.py

Regression tests for the dashboard's failure modes.

A Streamlit page that raises during render shows the user a blank screen —
so the dashboard must degrade gracefully when the backend is down or has no
data yet, rather than crashing. This previously failed with:

    TypeError: unsupported format string passed to NoneType.__format__

because a None avg_risk was formatted with :.2f.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

DASHBOARD = Path(__file__).resolve().parent.parent / "frontend" / "dashboard.py"


def _stub_streamlit():
    st = MagicMock()
    st.columns = lambda spec: [MagicMock() for _ in (spec if isinstance(spec, list) else range(spec))]
    st.tabs = lambda names: [MagicMock() for _ in names]
    st.selectbox = lambda label, options, **kw: (list(options)[0] if options else None)
    st.text_input = lambda *a, **k: ""
    st.button = lambda *a, **k: False
    st.session_state = {}
    st.cache_data = lambda **kw: (lambda fn: fn)
    sys.modules["streamlit"] = st
    return st


def _run_dashboard():
    src = DASHBOARD.read_text(encoding="utf-8")
    ns = {"__name__": "__main__", "__file__": str(DASHBOARD)}
    try:
        exec(compile(src, "dashboard.py", "exec"), ns)
    except SystemExit:
        pass          # st.stop() is a legitimate early exit


def test_renders_when_api_is_completely_down(monkeypatch):
    """A dead backend must produce a warning, never an unhandled exception."""
    _stub_streamlit()

    class Dead:
        def raise_for_status(self):
            raise requests.exceptions.ConnectionError("connection refused")

    monkeypatch.setattr(requests, "get", lambda *a, **k: Dead())
    monkeypatch.setattr(requests, "post", lambda *a, **k: Dead())

    _run_dashboard()   # must not raise


def test_renders_when_summary_fields_are_none(monkeypatch):
    """
    The original bug: /summary returned avg_risk=None and the f-string
    ':.2f' blew up, blanking the whole page.
    """
    _stub_streamlit()

    empty_summary = {
        "total_sellers": 0, "band_counts": {}, "avg_risk": None,
        "best_model": None, "best_pr_auc": None,
        "risk_histogram": [], "by_state": [],
    }

    class Resp:
        def __init__(self, payload): self._p = payload
        def raise_for_status(self): pass
        def json(self): return self._p

    def fake_get(url, params=None, timeout=None):
        if "/summary" in url:
            return Resp(empty_summary)
        if "/health" in url:
            return Resp({"status": "ok", "model_stage": "none"})
        return Resp([])

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp({}))

    _run_dashboard()   # must not raise
