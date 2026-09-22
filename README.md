# SENTRIX — Delivery-Risk Intelligence

**Predicts which sellers on a marketplace are about to deliver late, ranks them so an operations team knows who to call first, and explains why for every single one.**

[![CI](https://github.com/paradise2580/Sentrix/actions/workflows/ci.yml/badge.svg)](https://github.com/paradise2580/Sentrix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

**▶ [Live dashboard](https://sentrix-dashboard.onrender.com)** · **[API docs](https://sentrix-api-t1dw.onrender.com/docs)**

> Hosted on a free tier. A scheduled keep-alive ping (`.github/workflows/keepalive.yml`)
> pings both services every 5 minutes so they never hit Render's 15-minute idle sleep —
> no cold-start wait for a visitor opening the link.

<!-- Add a dashboard screenshot here — save it as docs/dashboard.png and uncomment:
![SENTRIX dashboard](docs/dashboard.png)
-->

---

## The problem

A marketplace operations team can chase maybe a hundred sellers a week. There are three thousand. Late deliveries drive refunds, bad reviews and churn — but only after it is too late to prevent them.

SENTRIX scores every order at purchase time for the risk it misses its promised date, rolls those scores up per seller, and produces a ranked queue. Work the top of the list and you catch roughly three times as many late sellers as picking at random.

---

## Tech stack

| Layer | What it uses |
|---|---|
| **Frontend** | Streamlit, Plotly |
| **Backend / API** | FastAPI, Uvicorn, Pydantic |
| **Database** | MySQL (SQLite snapshot for the deployed demo) |
| **ML models** | scikit-learn (Random Forest, Logistic Regression), XGBoost, LightGBM, custom stacking ensemble |
| **Explainability** | SHAP |
| **Tuning** | Optuna with `TimeSeriesSplit` |
| **GenAI / RAG** | LangChain, ChromaDB, Groq |
| **Experiment tracking** | MLflow |
| **Data versioning** | DVC |
| **Orchestration** | Apache Airflow |
| **Monitoring** | Evidently AI (drift detection) |
| **CI / CD** | GitHub Actions, flake8, pytest |
| **Containers** | Docker, Docker Compose |
| **Deployment** | Render (API + dashboard, both Dockerised) |

---

## What it does

**1. Predicts.** Five models trained and compared on 97,817 order-level rows with 93 encoded features. The stacking ensemble won on PR-AUC and is promoted to production.

**2. Ranks.** Order-level scores roll up to a per-seller risk score, then into four capacity-based bands — critical is the worst 5% of sellers right now, high is the next 15%, medium the next 30%.

**3. Explains.** Every seller's score comes with its SHAP feature contributions, so "why is this seller flagged" has an actual answer rather than a shrug.

**4. Answers questions.** A RAG layer over the model's own predictions and real customer reviews lets a non-technical user ask things like *"which sellers in São Paulo have critical risk?"* in plain English.

---

## Results

<!-- RESULTS:START -->

Evaluated on **19,564 held-out orders** from the final chronological block, base rate **5.2%**. Train and test are separated by a 30-day embargo, so no training label resolves inside the evaluation window.

### Model comparison

Every model is scored on the *same* rows. PR-AUC depends on the base rate, so models evaluated on different subsets cannot be ranked against each other — `compare_models` raises rather than print a table that mixes them.

| Model | PR-AUC | vs. random | ROC-AUC | KS | Capture @10% | Lift @10% | Brier |
|---|---|---|---|---|---|---|---|
| **Stacking Ensemble** | **0.1528** | 2.93x | 0.7613 | 0.3958 | 32.1% | 3.21x | 0.3606 |
| XGBoost (Optuna-tuned) | 0.1494 | 2.86x | 0.7551 | 0.3804 | 32.1% | 3.21x | 0.1571 |
| Logistic Regression | 0.1167 | 2.24x | 0.7158 | 0.3244 | 25.1% | 2.51x | 0.4029 |
| LightGBM (SMOTE) | 0.0884 | 1.69x | 0.6360 | 0.1914 | 20.8% | 2.08x | 0.0544 |
| Random Forest | 0.0527 | 1.01x | 0.5249 | 0.0869 | 8.8% | 0.88x | 0.1869 |

### What an operations team actually gets

PR-AUC summarises a curve nobody runs. A team can review a fixed number of orders per cycle, so the number that matters is how much of the risk they capture at that capacity.

| Review the top… | Orders flagged | Of all late orders, caught | vs. random |
|---|---|---|---|
| 5% | 978 | **18.5%** | 3.70x |
| 10% | 1,956 | **29.9%** | 2.99x |
| 20% | 3,913 | **51.3%** | 2.57x |

### Calibration

`Stacking Ensemble` is calibrated with isotonic regression fitted on the held-out calibration block — data the model never trained on and the test set never touches.

| | Raw score | Calibrated |
|---|---|---|
| Brier score | 0.3606 | **0.0492** |
| Expected calibration error | 0.5422 | **0.0328** |
| Mean predicted probability | 0.5944 | 0.0210 |

Observed rate on the evaluation set: **0.0522**. Ranking metrics are invariant to any monotone rescaling of the score, so they are blind to this entirely — which is why a risk product needs both.

### Operating point

The decision threshold is chosen by minimising expected cost at a **10:1** ratio (a missed late parcel vs. an analyst's wasted review), not by defaulting to 0.5 — which silently assumes the two errors are equally bad.

- Cost-optimal threshold: **0.0284** (F1-optimal would be 0.0426)
- Precision 0.135 · Recall 0.510 · F1 0.213
- Expected cost vs. not modelling at all: **18.3% lower**
- Confusion matrix: TP=521 FP=3,343 FN=500 TN=15,200
- Alert volume at that threshold: **19.8% of all orders**

That last line is why the capacity view above is the one to run the product on. A 10:1 cost ratio says false alarms are cheap, so the cost-minimising threshold alerts on a large share of the population — arithmetically right, operationally useless. Supply a real cost ratio and re-derive it; supply a weekly review capacity and read the capture table instead.

*Label horizon 30 days. Generated by `scripts/render_results.py` from `artifacts/evaluation/` — do not edit this section by hand.*

<!-- RESULTS:END -->

---

## The data is real

Built on the **Olist Brazilian e-commerce dataset**: 99,441 genuine orders from 3,095 genuine sellers (8.02% genuine late-delivery rate), with the late-delivery outcome computed from actual delivery dates — not a simulated label.

Commodity volatility is real too (FRED oil prices). Weather and port-congestion signals are **generated** and flagged `is_synthetic=1` in the database, because no free API can backfill 2016–2018. That distinction is surfaced in the dashboard rather than buried.

---

## How it works

```
Olist CSVs  →  MySQL  →  feature engineering  →  purged temporal split
                                                        ↓
                          five models trained, compared, best one promoted
                                                        ↓
                          isotonic calibration  →  SHAP explanations
                                                        ↓
                      FastAPI serving layer  →  Streamlit dashboard
                                                        ↓
                          ChromaDB + LangChain + Groq  →  RAG chat
```

Airflow orchestrates the pipeline, MLflow tracks every run, DVC versions the data, and Evidently watches for drift in production.

---

## Run it locally

**Requirements:** Python 3.12, Docker (optional), MySQL (optional — the demo path uses a SQLite snapshot).

```bash
git clone https://github.com/paradise2580/Sentrix.git
cd Sentrix

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# Terminal 1 — the API
uvicorn api.app:app --reload --port 8000

# Terminal 2 — the dashboard
streamlit run frontend/dashboard.py
```

Or with Docker:

```bash
docker compose -f docker-compose.serve.yml up
```

**Run the tests:**

```bash
pytest -q          # 53 tests
flake8 .           # lint, blocking in CI
```

---

## Project structure

```
api/          FastAPI serving layer
src/          feature engineering, models, evaluation, calibration
frontend/     Streamlit dashboard
pipelines/    Airflow DAGs
scripts/      one-off utilities and report generators
tests/        pytest suite
docker/       Dockerfiles for each service
config/       YAML configuration
notebooks/    exploratory analysis
```

---

## Known limitations

Stated rather than hidden, because a model whose weaknesses are unknown is a model nobody should trust.

- **The full model underperforms a one-feature baseline on this ablation.** `artifacts/evaluation/ablation.csv` ranks sellers on `promised_days` alone — no model at all — and gets PR-AUC 0.2531 / ROC-AUC 0.7989, both higher than the full 93-feature model's 0.1494 / 0.7551. Dropping every feature except the promise date still beats the full model (PR-AUC 0.2075); dropping the promise date instead collapses PR-AUC to 0.0464. Read plainly, this ablation says the other 90 features are net noise for this model class and this metric — worth a real investigation (feature leakage in the reverse direction, a regularisation or class-imbalance issue, or the ensemble just needs retuning) before calling the ranking model production-ready. Stated here rather than left for someone else to find in the CSV.
- **Calibration still under-predicts.** After isotonic regression, the calibrated mean prediction is 2.1% against an observed rate of 5.2% on the same evaluation block — the model is better calibrated than the raw score (ECE drops from 0.542 to 0.033) but still understates aggregate risk by roughly half.
- **The split is temporal, not grouped.** The same sellers appear in train and test. That is deliberate — production scores sellers it has already seen — but it measures temporal generalisation, not cold-start.
- **Random Forest badly underperforms here** (PR-AUC 0.0527, worse than random at the top-10% cutoff) despite being a strong baseline in the earlier seller-grain version of this model. It is kept in the comparison as a documented negative result rather than quietly dropped.
- **Stacking adds almost nothing** over its best base model (ensemble PR-AUC 0.1528 vs. XGBoost 0.1494), because the base models are too correlated to disagree usefully.
- **The full pipeline job is manual-only in CI.** It needs the Kaggle-licensed Olist CSVs, which cannot be redistributed.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Anshivya Nagpal](https://github.com/paradise2580).
