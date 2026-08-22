# SENTRIX — Seller Delivery-Risk Intelligence

Predicts which marketplace sellers will miss a delivery in the next 30 days,
explains **why** for each one, and lets a non-technical user interrogate the
results in plain English.

Built on **99,441 real orders** from 3,095 real Brazilian e-commerce sellers,
with a real outcome label — not a simulated dataset.

<!-- Replace the two URLs below once CI has run and the demo is deployed. -->
![CI](https://github.com/paradise2580/Sentrix/actions/workflows/ci.yml/badge.svg)

---

## The problem

A marketplace's operations team can chase maybe a hundred sellers a week. There
are three thousand. Late deliveries cost refunds, support contacts and repeat
custom — but only after they have already happened, and the team finds out from
the complaint, not before.

SENTRIX ranks every seller by the probability they will be late in the next 30
days, so the hundred that get chased are the right hundred.

**Where it lands:** reviewing the top 10% of the seller list catches **29.5%**
of every seller who goes on to deliver late — 2.95× better than reviewing 10% at
random. At 20% capacity it catches 45.6%.

---

## Architecture

```mermaid
flowchart LR
    subgraph ingest["Ingestion"]
        A["Olist CSVs<br/>9 files, ~200MB"] --> B[("MySQL<br/>8 tables")]
        F["FRED API<br/>real WTI prices"] --> B
        G["Generated signals<br/>weather · port"] --> B
    end

    subgraph features["Feature engineering"]
        B --> C["Seller-day panel<br/>380,775 rows × 30 features"]
        C --> D{"Purged split<br/>+ 30d embargo"}
    end

    subgraph models["Modelling"]
        D --> E1["LogReg · RandomForest"]
        D --> E2["XGBoost (Optuna)<br/>LightGBM (SMOTE)"]
        D --> E3["Stacking ensemble<br/>TimeSeriesSplit CV"]
        D --> E4["LSTM<br/>30-day sequences"]
        E1 & E2 & E3 & E4 --> H["Shared evaluation<br/>+ isotonic calibration"]
    end

    subgraph serve["Serving"]
        H --> I["MLflow registry<br/>Production stage"]
        H --> J["SHAP explanations"]
        I & J --> K[("predictions<br/>table")]
        K --> L["FastAPI"]
        L --> M["Streamlit dashboard"]
        L --> N["RAG chat<br/>ChromaDB + Groq"]
    end

    style D fill:#7c3aed,color:#fff
    style H fill:#7c3aed,color:#fff
```

DVC versions the data, Airflow schedules the refresh, Evidently watches for
drift, and GitHub Actions gates every merge on both the tests and the model's
PR-AUC.

---

## The part worth reading: what stops this from being wrong

Most of the engineering effort here went into *not* reporting an inflated
number. Three specific failure modes were found and fixed, and each is now
covered by a test that fails the build.

### 1. The label overlaps the split

The label asks "is this seller late in the next 30 days?" A plain chronological
cut leaves training rows whose labels are decided by events inside the test
window. Train and test then share information and the held-out score is
optimistic — and nothing raises an exception, because leakage does not crash,
it just makes every metric go up.

The split is therefore **purged and embargoed**, the standard treatment for
overlapping-label time series:

```
2016-09 ─────────── TRAIN ─────────── 2018-02-10  ╎ 30d ╎  CALIB  ╎ 30d ╎  ──── TEST ──── 2018-10
                 239,210 rows                     embargo  10,793   embargo   76,815 rows
                                                            rows
                                        53,957 rows deliberately discarded
```

The middle block exists so probability calibration can be fitted on data the
model never trained on and the test set never touches. Fitting the calibrator
on the test set would quietly turn the test score into a training score.

### 2. Six models, two different test sets

The LSTM could not score a seller's first 30 days — no window to look back over
— so it was evaluated on a *subset* of the test rows while the five tabular
models were evaluated on all of them. All six were then printed in one table
ranked by PR-AUC, which is not comparable across base rates. The ranking was
meaningless.

Worse, the dropped rows were not a random sample. Early seller-days carry a
higher late rate, so excluding them pulled the test block's base rate from
**16.8% down to 15.0%** across 39% fewer rows — the subset was easier as well as
smaller, which flatters every model on it.

Fixed by left-padding the sequences: a seller with three days of history gets a
30-step window whose first 27 rows are zeros, which in standardised space is the
training mean — "no information", which is what an unobserved day is. Every
model is now scored on all 76,815 test rows at the true base rate. It is also
the correct production behaviour: a new seller needs a score on day 3, and that
is exactly when a delivery-risk score is most useful.

`compare_models` still raises if handed results of differing lengths.

The LSTM was also being fed **raw unscaled features** while every other model
got the fitted `StandardScaler` — an unscaled `days_since_last_late` of 9999
saturates the gates on the first forward pass. It now consumes the same
transformed matrix as everything else.

### 3. Leakage reintroduced one level down

The outer split was fixed, but two inner loops still used shuffled k-fold CV on
the same panel:

- **The stacking ensemble.** A stacker trains its meta-learner on out-of-fold
  base predictions; with k-fold, fold 1 is scored by models fitted on its own
  future. The result was ROC-AUC **0.538** on held-out data — worse than every
  one of its own base models, which is the signature of a meta-learner fitted on
  leaked predictions. Now `TimeSeriesSplit`.
- **Optuna's tuning loop.** Same problem, same fix. Optuna was optimising
  against an inflated score and therefore selecting hyperparameters that
  overfit.

### Tested, not asserted

`tests/test_split_integrity.py` verifies the embargo is at least the label
horizon, that no row appears in two blocks, that rows are genuinely discarded,
that mismatched evaluation sets are rejected, that calibration reduces
calibration error, and that an impossible embargo fails loudly instead of
training on four rows.

---

## Results

<!-- RESULTS:START -->

Evaluated on **76,815 held-out seller-days** from the final chronological block, base rate **16.8%**. Train and test are separated by a 30-day embargo, so no training label resolves inside the evaluation window.

### Model comparison

Every model is scored on the *same* rows. PR-AUC depends on the base rate, so models evaluated on different subsets cannot be ranked against each other — `compare_models` raises rather than print a table that mixes them.

| Model | PR-AUC | vs. random | ROC-AUC | KS | Capture @10% | Lift @10% | Brier |
|---|---|---|---|---|---|---|---|
| **Random Forest** | **0.3911** | 2.33x | 0.7102 | 0.3278 | 29.3% | 2.93x | 0.1833 |
| Stacking Ensemble | 0.3901 | 2.33x | 0.6802 | 0.3071 | 29.1% | 2.91x | 0.1826 |
| Logistic Regression | 0.3785 | 2.26x | 0.6355 | 0.2847 | 29.5% | 2.95x | 0.2009 |
| LightGBM (SMOTE) | 0.3430 | 2.04x | 0.6330 | 0.2448 | 27.2% | 2.71x | 0.1735 |
| XGBoost (Optuna-tuned) | 0.3281 | 1.96x | 0.6126 | 0.2117 | 25.3% | 2.53x | 0.1637 |
| LSTM (PyTorch) | 0.2485 | 1.48x | 0.5711 | 0.1190 | 19.8% | 1.98x | 0.2635 |

### What an operations team actually gets

PR-AUC summarises a curve nobody runs. A team can review a fixed number of sellers per cycle, so the number that matters is how much of the risk they capture at that capacity.

| Review the top… | Sellers flagged | Of all who go late, caught | vs. random |
|---|---|---|---|
| 5% | 3,841 | **17.1%** | 3.41x |
| 10% | 7,682 | **29.5%** | 2.95x |
| 20% | 15,363 | **45.6%** | 2.28x |

### Calibration

`Random Forest` is calibrated with isotonic regression fitted on the held-out calibration block — data the model never trained on and the test set never touches.

| | Raw score | Calibrated |
|---|---|---|
| Brier score | 0.1833 | **0.1357** |
| Expected calibration error | 0.2322 | **0.0882** |
| Mean predicted probability | 0.3999 | 0.2501 |

Observed rate on the evaluation set: **0.1678**. Ranking metrics are invariant to any monotone rescaling of the score, so they are blind to this entirely — which is why a risk product needs both.

### Operating point

The decision threshold is chosen by minimising expected cost at a **10:1** ratio (a missed late seller vs. an analyst's wasted review), not by defaulting to 0.5 — which silently assumes the two errors are equally bad.

- Cost-optimal threshold: **0.1429** (F1-optimal would be 0.3890)
- Precision 0.239 · Recall 0.782 · F1 0.366
- Expected cost vs. not modelling at all: **53.2% lower**
- Confusion matrix: TP=10,076 FP=32,168 FN=2,812 TN=31,759
- Alert volume at that threshold: **55.0% of all seller-days**

That last line is why the capacity view above is the one to run the product on. A 10:1 cost ratio says false alarms are cheap, so the cost-minimising threshold alerts on a large share of the population — arithmetically right, operationally useless. Supply a real cost ratio and re-derive it; supply a weekly review capacity and read the capture table instead.

*Label horizon 30 days. Generated by `scripts/render_results.py` from `artifacts/evaluation/` — do not edit this section by hand.*

<!-- RESULTS:END -->

### Five findings stated plainly rather than buried

- **The boosted trees lost.** Random Forest wins, and plain Logistic Regression
  beats both XGBoost and LightGBM. With rolling rates and a lifetime rate as
  features, the signal is close to monotone in the target and there is little
  interaction structure left for boosting to find — so it pays for its
  flexibility in variance and gets nothing back. This is what baselines are for.
- **Stacking added essentially nothing.** The ensemble lands within 0.001 PR-AUC
  of its best base model. Once the leakage was removed there was no free lunch
  left: the base models are highly correlated, so a meta-learner has little to
  arbitrage. Worth knowing, and worth reporting rather than quietly dropping.
- **The LSTM is the worst model here, by a distance.** A sequence model needs the
  *shape* of a trajectory to matter beyond its current level. Rolling 7/14/30-day
  windows already encode most of that shape as plain columns, so the LSTM is
  paying for 47 × 30 inputs to rediscover what four features already say.
- **A ranking score is not a probability.** LightGBM trains on SMOTE-resampled
  data; Logistic Regression and Random Forest use `class_weight="balanced"`. All
  rank fine and all emit inflated probabilities. Isotonic calibration cuts ECE by
  62% without touching the ranking.
- **Calibration did not fully close, and here is why.** Mean prediction lands at
  0.250 against an observed 0.168. The calibration block is a 12-day window whose
  own base rate differs from the test period, so isotonic learned a mapping that
  only partly transfers. Widening that block is the fix; it would move the
  training boundary and require a full retrain, so it is listed as a known
  limitation rather than silently smoothed over.

### On the cost-optimal threshold

At a 10:1 miss-to-false-alarm ratio the cost-minimising threshold flags **55% of
all seller-days**. That is arithmetically correct and operationally useless — no
team reviews half its sellers. It is reported because it is what the stated cost
ratio actually implies, and it is a useful reductio: if false alarms really were
a tenth as expensive as misses, you would alert on almost everything.

The number to run the product on is the capacity view — top 5%, 10%, 20% — which
is why risk bands here are percentile-based rather than absolute probability
cuts. A stakeholder who supplies a genuine cost ratio can re-derive the threshold
from `find_cost_optimal_threshold`; a stakeholder who supplies a weekly review
capacity reads it straight off the capture table.

---

## Data provenance — read this before judging the results

| Layer | Source | Status |
|---|---|---|
| Sellers, orders, order items, products | Olist Brazilian E-Commerce (Kaggle, CC BY-NC-SA) | **REAL** |
| Customer reviews (sentiment features) | Olist | **REAL** |
| Late-delivery label | `order_delivered_customer_date > order_estimated_delivery_date` | **REAL** |
| Commodity (fuel) volatility | FRED — real daily WTI oil prices | **REAL** |
| Weather / port congestion | Generated per real seller-state + real date | **SYNTHETIC** |

**Scale:** 3,095 sellers · 99,441 orders · 112,650 order items · 99,224 reviews.

**Two different rates, often confused:** 8.11% of *orders* arrive late. 22.47%
of *seller-days* are positive — a seller-day is positive if **any** order in the
following 30 days is late, so one late order marks up to 30 rows. The models are
trained at seller-day grain, so 22.47% is the base rate every metric below is
measured against.

### Why one signal layer is synthetic, and why *that* layer

Orders span **Sept 2016 – Oct 2018**, so an external-signal API has to return
data *for those dates*. Each free tier was checked against that requirement:

| Signal | Free-tier reality | Decision |
|---|---|---|
| **Commodity** | FRED serves decades of free daily history by date range | **Use real data** |
| Weather | OpenWeatherMap free tier is current/forecast; history is paid | Generate |
| Port congestion | No free public source for historical port congestion | Generate |

Calling a live weather API here would have silently matched 2026 weather against
2017 orders — worse than generating it honestly. So the one signal free tooling
can legitimately support was made real, and the rest are generated
**independently of the label**, so they cannot manufacture a correlation.

**Provenance is enforced in the database, not just claimed here.** Every
`external_signals` row stores `is_synthetic` (0 = real FRED, 1 = generated), and
`test_signal_provenance_is_explicit` fails the build if any row is ambiguous or
if weather/port is ever mislabelled as real.

---

## Features — 30 columns at seller-day grain

| Family | Provenance | Examples |
|---|---|---|
| Delivery history | REAL | rolling 7/14/30-day late counts and rates, trend, days since last late |
| Review sentiment | REAL | rolling mean review score, bad-review counts |
| External signals | MIXED | commodity volatility (real FRED), weather / port (generated) by state |
| Profile & peer | REAL | lifetime late rate, state-peer late rate, tenure |

Every feature looks strictly **backward** via `.shift(1)`; the label looks
strictly **forward**. The preprocessor is fitted on the training slice only —
fitting the scaler or imputer on the full table leaks test-period statistics
even when the row-level split is correct.

---

## Stack

MySQL · pandas · scikit-learn · XGBoost · LightGBM · PyTorch · Optuna · SMOTE ·
SHAP · MLflow · DVC · ChromaDB · LangChain · Groq · FastAPI · Streamlit · pytest
· Evidently · Docker · GitHub Actions · Airflow

---

## Run it

```bash
python -m venv venv && venv\Scripts\activate     # Windows
pip install -r requirements.txt
copy .env.example .env                           # MYSQL_PASSWORD is required

python -m src.ingestion.schema                   # create tables
python -m src.ingestion.olist_loader             # load real Olist data
python -m src.ingestion.synthetic_signals        # real FRED + flagged synthetic
python scripts/build_features.py                 # 380,775-row feature table

python -m src.models.trainer                     # all four stages
python -m src.evaluation.run_evaluation          # rank, calibrate, pick threshold
python -m src.evaluation.generate_predictions    # score + explain every seller
python -m src.models.mlflow_tracking             # register the winner
python -m src.rag.indexer                        # build the chat index

python scripts/render_results.py                 # regenerate this README's results
pytest tests/ -v

uvicorn api.app:app --reload --port 8000         # terminal 1
streamlit run frontend/dashboard.py              # terminal 2
```

Or `docker compose up`.

Training stages are independently resumable —
`python -m src.models.trainer --stages boosting` re-runs one stage without
retraining the rest.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + which model MLflow has in Production |
| `GET /summary` | Portfolio KPIs, risk histogram, per-state aggregates |
| `GET /sellers?risk_band=` | Every seller ranked by calibrated risk |
| `GET /explain/{seller_id}` | SHAP attribution for one seller's score |
| `POST /chat` | RAG-grounded Q&A over predictions and real review text |
| `GET /metrics` | Full model comparison table |

---

## Limitations

Stated because a portfolio project that claims none is not being read carefully.

- **The weather and port signals are generated.** Their measured contribution is
  therefore not evidence about real weather or real ports. The real-data
  families carry the model; the synthetic layer is a clearly-labelled
  demonstration of how an external-signal join would work.
- **One marketplace, one country, one two-year window.** Nothing here is
  evidence the model transfers to a different logistics network.
- **The split is temporal, not grouped — the same sellers appear in train and
  test.** That is deliberate: in production you score sellers you already have
  history for, so a temporal split is the honest simulation of that. But it
  means these numbers measure *temporal* generalisation and say nothing about
  cold start. Measuring "how well does this score a seller it has never seen?"
  needs a `GroupKFold` on `seller_id`, and would score lower — the delivery
  history and lifetime-rate families, which carry most of the signal, are
  precisely what a new seller does not have.
- **Risk bands are capacity-based, not absolute.** "Critical" means the worst 5%
  of sellers scored right now, not a fixed probability. With a 22% base rate a
  correctly calibrated model rarely exceeds 0.75, so absolute cutoffs would leave
  the top bands permanently empty — which would make a well-calibrated model look
  worse than an overconfident one.
- **Airflow needs its own environment.** It pins `sqlalchemy<2.0`, which
  conflicts with FastAPI and MLflow. See `requirements-airflow.txt`; the DAG in
  `pipelines/` is written for that separation and triggers the pipeline by
  subprocess, never by shared imports.
- **The calibration block is narrow.** Twelve days, 10,793 rows, after the
  30-day embargo takes its bite. Isotonic regression has enough data to fit, but
  the block's base rate differs from the test period, which is why calibration
  closes most of the gap and not all of it. Widening `calibration_size` in
  `config.yaml` fixes it at the cost of training rows.
- **The LSTM is not competitive and is kept as a negative result.** Deleting it
  would make the comparison table look better and say less.
- **The RAG embedder falls back to TF-IDF** when `huggingface.co` is
  unreachable. Retrieval mechanics are identical and it switches back
  automatically.

---

## Repository map

```
src/ingestion/      Olist loaders, MySQL schema, FRED + generated signals
src/preprocessing/  seller-day feature builder, fitted transform pipeline
src/models/         baselines, boosting, stacking, LSTM, SHAP, MLflow
src/evaluation/     shared-set evaluation, calibration, ops metrics, scoring
src/monitoring/     Evidently drift detection
src/rag/            ChromaDB index, retriever, LangChain + Groq chain
api/                FastAPI service
frontend/           Streamlit dashboard
tests/              unit + integration + split-integrity suites
pipelines/          Airflow DAG
```
