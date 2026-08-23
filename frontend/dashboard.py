"""
frontend/dashboard.py

Role
----
The window users actually see. Holds NO business logic — every number comes
from an HTTP call to the FastAPI backend, so the dashboard could be swapped
for React tomorrow without touching model, database, or RAG code.

Performance note
----------------
Landing KPIs come from a single /summary call rather than pulling all 1,325
seller rows into the browser and aggregating client-side. API responses are
cached with st.cache_data so switching tabs doesn't re-hit the backend.

Run with:
    streamlit run frontend/dashboard.py
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from src.config_loader import load_config

cfg = load_config()

# The deployed dashboard and the deployed API are separate services with
# separate URLs, so the backend address cannot be baked into config.yaml.
# Environment first, config.yaml as the local default.
API_BASE = os.getenv("SENTRIX_API_BASE") or cfg["frontend"]["api_base_url"]

st.set_page_config(page_title="SENTRIX", page_icon="📦", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1300px; }
    #MainMenu, footer { visibility: hidden; }

    .hero { border-bottom: 1px solid rgba(255,255,255,.12); padding-bottom: 14px; margin-bottom: 22px; }
    .hero h1 { margin: 0; font-size: 34px; letter-spacing: -.5px; }
    .hero p  { margin: 4px 0 0; opacity: .65; font-size: 14px; }

    .kpi {
        border: 1px solid rgba(255,255,255,.12); border-radius: 12px;
        padding: 16px 18px; height: 100%;
    }
    .kpi .v { font-size: 30px; font-weight: 700; line-height: 1.1; }
    .kpi .l { font-size: 11px; letter-spacing: .8px; text-transform: uppercase;
              opacity: .6; margin-top: 6px; }
    .kpi .s { font-size: 12px; opacity: .5; margin-top: 2px; }

    .k-low  .v { color: #22c55e; }
    .k-med  .v { color: #eab308; }
    .k-high .v { color: #f97316; }
    .k-crit .v { color: #ef4444; }
    .k-neut .v { color: #e2e8f0; }

    .prov {
        border-left: 3px solid #3b82f6; background: rgba(59,130,246,.08);
        padding: 10px 14px; border-radius: 0 8px 8px 0; font-size: 12.5px;
        opacity: .9; margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] { font-size: 15px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers (cached so tab switches don't re-hit the backend)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def api_get(path: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API unreachable at {API_BASE}{path} — is uvicorn running?  ({e})")
        return None


def api_post(path: str, body: dict):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API unreachable at {API_BASE}{path}  ({e})")
        return None


def fmt(value, spec=".2f", fallback="—"):
    """
    Format a number that might be None/NaN. The dashboard must degrade to a
    dash rather than crash the whole page when the backend has no data yet
    (e.g. predictions table not populated).
    """
    try:
        if value is None:
            return fallback
        return format(float(value), spec)
    except (TypeError, ValueError):
        return fallback


def kpi(col, value, label, sub="", cls="k-neut"):
    col.markdown(
        f'<div class="kpi {cls}"><div class="v">{value}</div>'
        f'<div class="l">{label}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True)


BAND_COLORS = {"low": "#22c55e", "medium": "#eab308", "high": "#f97316", "critical": "#ef4444"}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
health = api_get("/health") or {}
summary = api_get("/summary")

st.markdown(
    '<div class="hero"><h1>📦 SENTRIX</h1>'
    '<p>Seller Delivery-Risk Intelligence — predicting which marketplace sellers '
    'will miss a delivery in the next 30 days</p></div>',
    unsafe_allow_html=True)

st.markdown(
    '<div class="prov"><b>Data provenance</b> — Sellers, orders, reviews and the '
    'late-delivery label are <b>real</b> (Olist Brazilian marketplace, 99,441 orders, '
    '8.11% genuine late rate). Commodity volatility is <b>real</b> (FRED oil prices). '
    'Weather &amp; port-congestion signals are <b>generated</b> and flagged '
    '<code>is_synthetic=1</code> in the database — free APIs cannot backfill 2016–2018.</div>',
    unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------
if summary:
    bands = summary["band_counts"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi(c1, f"{summary['total_sellers']:,}", "Sellers monitored")
    kpi(c2, bands.get("critical", 0), "Critical risk", "act now", "k-crit")
    kpi(c3, bands.get("high", 0), "High risk", "watch closely", "k-high")
    kpi(c4, bands.get("medium", 0), "Medium risk", "", "k-med")
    kpi(c5, bands.get("low", 0), "Low risk", "healthy", "k-low")
    kpi(c6, fmt(summary.get("avg_risk")), "Avg risk score",
        f"model: {summary.get('best_model') or '—'}")

    st.caption(
        f"Live model stage: **{health.get('model_stage', 'unknown')}**  ·  "
        f"Best model **{summary.get('best_model') or '—'}** "
        f"(PR-AUC {fmt(summary.get('best_pr_auc'), '.3f')})  ·  API: {API_BASE}")
elif summary is not None and summary.get("total_sellers", 0) == 0:
    st.warning(
        "The API is up, but no predictions are stored yet. Populate them with:\n\n"
        "```\npython -m src.evaluation.generate_predictions\n```")
else:
    st.warning(
        "Backend not reachable. Start it in another terminal:\n\n"
        "```\nuvicorn api.app:app --reload --port 8000\n```")

st.write("")

tab_overview, tab_sellers, tab_explain, tab_models, tab_chat = st.tabs(
    ["📈 Overview", "📋 Seller Risk", "🔍 Explain", "🧪 Model Performance", "💬 Ask SENTRIX"]
)

# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------
with tab_overview:
    if summary and summary.get("total_sellers", 0) > 0:
        left, right = st.columns([1, 1])

        with left:
            st.markdown("##### Risk score distribution")
            hist = summary.get("risk_histogram") or [0] * 20
            centers = [(i + 0.5) / 20 for i in range(20)]
            colors = ["#22c55e" if c < .25 else "#eab308" if c < .5
                      else "#f97316" if c < .75 else "#ef4444" for c in centers]
            fig = go.Figure(go.Bar(x=centers, y=hist, marker_color=colors,
                                   hovertemplate="risk %{x:.2f}<br>%{y} sellers<extra></extra>"))
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="predicted risk score", yaxis_title="sellers",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width='stretch')

        with right:
            st.markdown("##### Average risk by state")
            by_state = summary.get("by_state") or []
            if not by_state:
                # NOT st.stop() — that halts the entire script, blanking every
                # other tab because one panel had no data.
                st.info("No per-state data available yet.")
            else:
                bs = pd.DataFrame(by_state).head(12).sort_values("avg_risk")
                fig = go.Figure(go.Bar(
                    x=bs["avg_risk"], y=bs["seller_state"], orientation="h",
                    marker_color="#ef4444", text=bs["seller_count"].map(lambda n: f"{n} sellers"),
                    textposition="outside", hovertemplate="%{y}: avg risk %{x:.2f}<extra></extra>"))
                fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title="average risk score",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width='stretch')

        st.markdown("##### How to read this")
        st.markdown(
            "- **Risk score** = *calibrated* probability that a seller has a late "
            "delivery in the next 30 days, learned from real historical outcomes. "
            "Calibrated means 0.30 corresponds to roughly a 30% observed rate — "
            "the raw model score does not.\n"
            "- **Bands are capacity-based, not fixed cutoffs.** Critical = the worst "
            "5% of sellers scored right now, high = the next 15%, medium = the next "
            "30%. With a ~17% base rate a well-calibrated model rarely emits a "
            "probability above 0.75, so absolute cutoffs would leave the top bands "
            "permanently empty — and make a correctly calibrated model look safer "
            "than an overconfident one.\n"
            "- Every score is explainable: open **Explain** for the exact feature "
            "contributions (SHAP) behind any seller's number.")

# ---------------------------------------------------------------------------
# Tab 2 — Seller risk table
# ---------------------------------------------------------------------------
with tab_sellers:
    c1, c2 = st.columns([1, 3])
    band = c1.selectbox("Risk band", ["All", "critical", "high", "medium", "low"])
    params = {"limit": 500} if band == "All" else {"risk_band": band, "limit": 500}

    sellers = api_get("/sellers", params=params)
    if sellers:
        df = pd.DataFrame(sellers)
        c2.caption(f"Showing top {len(df)} sellers by risk score "
                   f"(of {summary['total_sellers']:,} monitored)" if summary else "")

        view = df[["seller_id", "seller_city", "seller_state",
                   "risk_score", "risk_band", "model_name"]].copy()
        view.columns = ["Seller ID", "City", "State", "Risk", "Band", "Model"]

        st.dataframe(
            view, width='stretch', hide_index=True, height=520,
            column_config={
                "Risk": st.column_config.ProgressColumn(
                    "Risk", min_value=0.0, max_value=1.0, format="%.2f"),
                "Seller ID": st.column_config.TextColumn("Seller ID", width="medium"),
            })
    elif sellers == []:
        st.info("No sellers in this band.")

# ---------------------------------------------------------------------------
# Tab 3 — SHAP explanation
# ---------------------------------------------------------------------------
with tab_explain:
    st.markdown("##### Why is a seller flagged at its current risk level?")
    st.caption("Feature contributions come from SHAP — the model's own reasoning, not a guess.")

    riskiest = api_get("/sellers", params={"limit": 200})
    if riskiest:
        labels = {f"{s['seller_id'][:12]}…  ·  {s['seller_city'] or '—'} "
                  f"({s['seller_state'] or '—'})  ·  {s['risk_band']} {s['risk_score']:.2f}": s["seller_id"]
                  for s in riskiest}

        picked = st.selectbox("Seller (top 200 by risk)", list(labels.keys()))
        seller_id = labels[picked]

        exp = api_get(f"/explain/{seller_id}")
        if exp:
            m1, m2, m3 = st.columns([1, 1, 3])
            m1.metric("Risk score", fmt(exp.get("risk_score")))
            m2.metric("Band", exp["risk_band"].upper())
            m3.code(seller_id, language=None)

            feats = pd.DataFrame(exp["top_features"]).sort_values("contribution")
            fig = go.Figure(go.Bar(
                x=feats["contribution"], y=feats["feature"], orientation="h",
                marker_color=["#ef4444" if v > 0 else "#3b82f6" for v in feats["contribution"]],
                hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
            fig.update_layout(
                height=360, margin=dict(l=0, r=0, t=30, b=0),
                title="Red raises risk · blue lowers it",
                xaxis_title="SHAP contribution",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width='stretch')

            top = feats.reindex(feats["contribution"].abs().sort_values(ascending=False).index).iloc[0]
            direction = "raising" if top["contribution"] > 0 else "lowering"
            st.info(f"Biggest driver: **{top['feature']}** — {direction} this seller's risk "
                    f"by {abs(top['contribution']):.3f}.")

# ---------------------------------------------------------------------------
# Tab 4 — Model performance
# ---------------------------------------------------------------------------
with tab_models:
    metrics = api_get("/metrics")
    if metrics:
        st.markdown(f"##### Six models compared — **{metrics['best_model']}** promoted to Production")
        st.caption("Ranked by PR-AUC on a purged chronological holdout — train on the past, "
                   "test on the future, with a 30-day embargo between them so no training "
                   "label resolves inside the test window. Every model is scored on the "
                   "identical set of rows. PR-AUC is the primary metric because late "
                   "deliveries are the minority class.")

        mdf = pd.DataFrame(metrics["comparison"])

        fig = go.Figure()
        fig.add_bar(x=mdf["model"], y=mdf["pr_auc"], name="PR-AUC", marker_color="#ef4444")
        fig.add_bar(x=mdf["model"], y=mdf["roc_auc"], name="ROC-AUC", marker_color="#3b82f6")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), barmode="group",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, width='stretch')

        if "capture_at_10pct" in mdf.columns and mdf["capture_at_10pct"].notna().any():
            st.markdown("##### What an ops team actually gets")
            st.caption("PR-AUC summarises a curve nobody runs. This is the number a team "
                       "with a fixed weekly review capacity cares about: work the top 10% "
                       "of the ranked list, and this is the share of sellers who go late "
                       "that you catch.")
            cap = mdf.sort_values("capture_at_10pct")
            fig2 = go.Figure(go.Bar(
                x=cap["capture_at_10pct"], y=cap["model"], orientation="h",
                marker_color="#22c55e",
                text=cap["lift_at_10pct"].map(lambda v: f"{v:.2f}x random"),
                textposition="outside",
                hovertemplate="%{y}: catches %{x:.1%} of late sellers<extra></extra>"))
            fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="share of late sellers caught in the top 10%",
                               xaxis_tickformat=".0%",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, width='stretch')

        st.dataframe(mdf, width='stretch', hide_index=True,
                     column_config={c: st.column_config.NumberColumn(c, format="%.4f")
                                    for c in mdf.select_dtypes("number").columns})
    else:
        st.info("Run `python -m src.evaluation.run_evaluation` to populate model metrics.")

# ---------------------------------------------------------------------------
# Tab 5 — RAG chat
# ---------------------------------------------------------------------------
with tab_chat:
    st.markdown("##### Ask SENTRIX")
    st.caption("Answers are retrieved from the model's own predictions and real customer "
               "reviews (ChromaDB), then synthesised by an LLM — grounded, not guessed.")

    examples = [
        "Which sellers in São Paulo have critical delivery risk?",
        "What are customers complaining about most?",
        "Summarise the riskiest sellers and why they're flagged.",
    ]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, width='stretch'):
            st.session_state["q"] = ex

    q = st.text_input("Your question", value=st.session_state.get("q", ""),
                      placeholder="e.g. Which sellers should I worry about this month?")

    if st.button("Ask", type="primary") and q:
        with st.spinner("Retrieving context and reasoning…"):
            res = api_post("/chat", {"question": q})
        if res:
            st.markdown(res["answer"])
            if not res.get("llm_used"):
                st.warning("No GROQ_API_KEY set — showing raw retrieved context instead of "
                           "an LLM-written answer.")
            with st.expander(f"Sources — {len(res['sources'])} retrieved chunks"):
                for s in res["sources"]:
                    st.markdown(f"- {s['text']}")
