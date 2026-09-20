# SENTRIX — Delivery-Risk Intelligence

**Predicts which sellers on a marketplace are about to deliver late, ranks them so an operations team knows who to call first, and explains why for every single one.**

[![CI](https://github.com/paradise2580/Sentrix/actions/workflows/ci.yml/badge.svg)](https://github.com/paradise2580/Sentrix/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

**▶ [Live dashboard](https://sentrix-dashboard.onrender.com)** · **[API docs](https://sentrix-api.onrender.com/docs)**

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
| **ML models** | scikit-learn (Random Forest, Logistic Regression), XGBoost, LightGBM, PyTorch (LSTM), custom temporal stacking ensemble |
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
| **BI layer** | Tableau Public |

---

## What it does

**1. Predicts.** Six models trained and compared on 380,775 order-level rows with 33 features. Random Forest won on PR-AUC and is promoted to production.

**2. Ranks.** Order-level scores roll up to a per-seller risk score, then into four capacity-based bands — critical is the worst 5% of sellers right now, high is the next 15%, medium the next 30%.

**3. Explains.** Every seller's score comes with its SHAP feature contributions, so "why is this seller flagged" has an actual answer rather than a shrug.

**4. Answers questions.** A RAG layer over the model's own predictions and real customer reviews lets a non-technical user ask things like *"which sellers in São Paulo have critical risk?"* in plain English.

---

## Results

Measured on a held-out future block of **76,815 orders** (base rate 16.78%), with a purged temporal split — train on the past, test on the future, 30-day embargo between them so no training label resolves inside the test window.

| Model | PR-AUC | ROC-AUC | Capture @ top 10% | Lift |
|---|---|---|---|---|
| **Random Forest** ✅ | **0.3911** | 0.7102 | 29.3% | 2.33× |
| Temporal stacking ensemble | 0.3901 | 0.6802 | 29.1% | 2.33× |
| Logistic Regression | 0.3785 | 0.6355 | 29.5% | 2.26× |
| LightGBM | 0.3430 | 0.6330 | 27.2% | 2.04× |
| XGBoost | 0.3281 | 0.6126 | 25.3% | 1.96× |
| LSTM | 0.2485 | 0.5711 | 19.8% | 1.48× |

**What an operations team actually gets:**

| Review capacity | Late sellers caught | vs. random |
|---|---|---|
| Top 5% | 17.1% | 3.41× |
| Top 10% | 29.5% | 2.95× |
| Top 20% | 45.6% | 2.28× |

**Calibration.** Raw model scores are not probabilities. Isotonic regression on a held-out block cut Brier score from 0.1833 to 0.1357 and expected calibration error from 0.2322 to 0.0882 — so a score of 0.05 now means roughly a 5% observed rate.

---

## The data is real

Built on the **Olist Brazilian e-commerce dataset**: 99,441 genuine orders from 3,095 genuine sellers, with a real late-delivery outcome computed from actual delivery dates — not a simulated label.

Commodity volatility is real too (FRED oil prices). Weather and port-congestion signals are **generated** and flagged `is_synthetic=1` in the database, because no free API can backfill 2016–2018. That distinction is surfaced in the dashboard rather than buried.

---

## How it works

```
Olist CSVs  →  MySQL  →  feature engineering  →  purged temporal split
                                                        ↓
                          six models trained, compared, best one promoted
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

- **Calibration only partly closes.** Mean prediction 0.250 against an observed 0.168. The calibration block is a 12-day window whose base rate differs from the test period; widening it costs training rows.
- **The split is temporal, not grouped.** The same sellers appear in train and test. That is deliberate — production scores sellers it has already seen — but it measures temporal generalisation, not cold-start.
- **The LSTM is not competitive** (PR-AUC 0.2485) and is kept as a documented negative result rather than quietly dropped.
- **Stacking adds almost nothing** over its best base model, because the base models are too correlated to disagree usefully.
- **The full pipeline job is manual-only in CI.** It needs the Kaggle-licensed Olist CSVs, which cannot be redistributed.

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Anshivya Nagpal](https://github.com/paradise2580).
