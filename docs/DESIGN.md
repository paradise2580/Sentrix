# SENTRIX — Design decisions

Why the project is built the way it is. The code comments say *what* each
piece does; this document explains *why*, and records the experiments behind
each decision.

---

## 1. The pipeline at a glance

```
Olist CSVs ─► MySQL ─► order feature table ─► purged temporal split
                                                    │
                  train 5 models ◄──────────────────┘
                        │
     compare on one shared test set (PR-AUC) ─► winner
                        │
   isotonic calibration (CALIB block) ─► cost-based threshold
                        │
  score last 90 days of orders ─► mean per seller ─► percentile bands + SHAP
                        │
      SQLite snapshot ─► FastAPI ─► Streamlit       ChromaDB ─► RAG chat
```

| Step | Code |
|---|---|
| Load raw data, compute `is_late` | `src/ingestion/olist_loader.py` |
| External signals (FRED + synthetic) | `src/ingestion/synthetic_signals.py`, `fred_signals.py` |
| Order features | `src/preprocessing/order_features.py` |
| Split, preprocessing, training | `src/models/trainer.py` |
| Evaluation, calibration, threshold | `src/evaluation/run_evaluation.py` |
| Seller scores, bands, SHAP | `src/evaluation/generate_predictions.py` |
| Serving snapshot | `scripts/export_serving_data.py` |
| API / dashboard | `api/app.py`, `frontend/dashboard.py` |
| RAG | `src/rag/` |

---

## 2. Why the model predicts orders, not sellers

The goal is a ranked list of *sellers*. The first two versions predicted a
seller-level target directly, and both failed on measured evidence.

### v1: "will this seller have any late order in the next 30 days?"

With a per-order late rate around 8%, this label mostly measures volume:
a seller with 100 orders is ~99.97% likely to have at least one late
delivery, and a seller with 3 orders about 22%.

`scripts/ablation.py` confirmed it. Ranking sellers by `order_count_30d`
alone, with no model, scored **PR-AUC 0.4455** against the full 47-feature
model's **0.3911**. A forest given only the order-count columns got ROC-AUC
0.7317 vs 0.6494 for the full model. **The model was a busy-seller detector.**

### v2: "will this seller's late rate exceed 15%?"

A rate removes volume from the definition, but the threshold brought it
back the other way. With 5 orders, 15% still means "at least one late",
while a seller with 100 orders at the normal 8% rate almost never reaches
15%. So the positive rate *fell* with volume, and since Olist volume grows
through 2018, train (17.3% positive) and test (12.2%) behaved differently.
Every model scored below 0.5 ROC-AUC (logistic regression: 0.4054).

`scripts/label_screen.py` then tried a label that removes volume by
construction (worst 20% of late rate within month × volume-decile cells).
It worked as designed (volume-only ROC-AUC 0.4946, base-rate drift 1.59
points), and the model then scored **0.5056, a coin flip**.

### Why the seller target can't be predicted

`scripts/label_screen2.py` asked whether the target was predictable at all:

- **Split-half reliability:** a seller's late rate in days 1–15 vs days
  16–30 of the same window. It rises from 0.21 to 0.39 as the minimum order
  count grows, so a real seller effect exists *within* a month.
- **Persistence AUC:** the trailing 30-day late rate used directly as a
  score for the next 30 days. It stays at 0.46–0.52 at every minimum order
  count, so the effect does *not* carry into the next month.

In plain terms: lateness here is a short-lived shock (a bad batch, a
carrier problem, a demand spike), not a stable seller trait. Nothing
measured before a month predicts the next month's seller late rate.

### v3 (current): predict each order, then aggregate

"Will this parcel miss its promised date?" can be answered from what is
known at purchase: how tight the promise is, distance, weight, freight
cost, the calendar. Seller risk is then the **mean** predicted risk of the
seller's recent orders:

- A mean is a rate, so shipping more orders doesn't make a seller look
  riskier. This removes the v1 volume problem by construction.
- Model the unit where the signal is (the order); decide on the unit
  where action happens (the seller).

---

## 3. Leakage controls

Leakage never raises an error; it just makes scores look better. Each
control below is in the code, and the split is covered by
`tests/test_split_integrity.py`.

| Risk | Control |
|---|---|
| Features that only exist after purchase | Only purchase-time columns are selected. `order_approved_at`, carrier/customer delivery dates and `delay_days` are excluded, and the build fails if an outcome column survives. |
| Seller history using outcomes not yet known | Seller and state late rates are computed as of **30 days before** each order (`history_lag_days`), using `merge_asof`. Orders take ~2 weeks to deliver, so a current late rate isn't knowable at purchase. |
| Labels crossing the train/test boundary | Purged split with a **30-day embargo** (below). |
| Test statistics leaking into preprocessing | The imputer, scaler and encoder are fitted on the training block only. |
| Stacking meta-learner seeing the future | Hand-written temporal stacking (below). |
| Tuning on the future | Optuna uses `TimeSeriesSplit`, not shuffled k-fold. Shuffled folds inflate the tuning score and favour deeper, overfit models. |
| SMOTE's invented rows in evaluation | SMOTE is applied to training data only. |

### The purged, embargoed split

```
[ TRAIN ] <30 days dropped> [ CALIB ] <30 days dropped> [ TEST ]
```

Rows are dated by purchase time, but a label is only known at delivery,
one to three weeks later. Without a gap, training orders placed just before
the cut have outcomes landing inside the test period, and test orders'
history features are built from outcomes the training rows revealed. The
embargo (≥ the label horizon) removes that overlap.

CALIB is a separate block because the calibrator must be fitted on data
the model didn't train on, and judged on data neither has seen. Fitting it
on the test set would turn the test score into a training score.

### Temporal stacking (`src/models/ensemble.py`)

A stacker trains its meta-learner on out-of-fold predictions of the base
models. The first version used `StackingClassifier(cv=3)`: with k-fold,
some folds are predicted by models trained on *later* data. That ensemble
scored **ROC-AUC 0.538**, worse than each of its own base models, which is
the signature of a meta-learner trained on leaked predictions.

`StackingClassifier(cv=TimeSeriesSplit(3))` doesn't work either: sklearn's
`cross_val_predict` needs every row to be in exactly one test fold, and
TimeSeriesSplit never tests the first block. So the stack is written by
hand in about forty lines: each fold trains on the past and predicts the
next block, the meta-learner learns from those predictions only, and the
base models are refit on all training data for inference.

---

## 4. Class imbalance

About 5% of test orders are late.

- Logistic regression and random forest: `class_weight="balanced"`.
- LightGBM: trained on SMOTE-resampled data.
- XGBoost: handles imbalance through its loss.
- LSTM: `pos_weight` in the loss. SMOTE can't be used for sequences,
  because interpolating two sequences creates a history no seller had.

All of these inflate raw scores, which is why calibration (section 6)
exists.

---

## 5. Evaluation

- **PR-AUC is the ranking metric.** Accuracy is meaningless at a 5% base
  rate: predicting "never late" is 95% accurate.
- **All models are scored on the same rows.** PR-AUC depends on the base
  rate, so models scored on different subsets can't be compared.
  `compare_models` raises an error if row counts differ. (An earlier LSTM
  couldn't score a seller's first 30 days, which silently removed 39% of
  the test block, with a different late rate from the rest. Left-padding
  the sequences fixed it.)
- **Capture@k and lift@k** answer the operational question: "if we review
  the top k%, what share of late orders do we catch, and how much better is
  that than random?"
- **KS statistic**, the standard separation metric in credit/risk teams.
- **Threshold by cost, not 0.5.** With a missed late order costing 10× a
  wasted review, the threshold that minimises total cost is chosen. The
  F1-optimal threshold is reported for comparison only, since F1 treats
  both errors as equal.

### Ablation: the model vs one column

`scripts/ablation.py` compares the full model with `promised_days` alone,
using the champion model family rather than a fixed random forest (the
forest is the weakest model here, so any subset would look competitive
against it).

Sign matters: `promised_days` scores ROC-AUC 0.2011 as-is, which is as
far from chance as 0.80. A longer promised window means *less* lateness,
so the column is strongly predictive once negated.

Result (see the README's Known limitations): `promised_days` alone
(PR-AUC 0.2531) beats the full model (0.1494). That is the main open
question for this project.

---

## 6. Calibration

Ranking metrics (PR-AUC, ROC-AUC, KS, capture@k) don't change under any
monotone rescaling of the score, so a model can rank perfectly while its
"probabilities" are far off. The raw ensemble's mean score is ~0.59
against a 5.2% observed late rate.

- **Isotonic regression**, fitted on the CALIB block. It is
  non-parametric and there are enough calibration rows for it (Platt
  scaling is the better choice below roughly a thousand samples).
- It works on scores rather than on a model object, so any model can be
  calibrated, including the LSTM (sklearn's `CalibratedClassifierCV` needs
  an sklearn estimator).
- It is applied **per order, before averaging per seller**, because it
  was fitted on orders.
- **Bands use the raw score; the calibrated score is what's displayed.**
  Isotonic output is a step function with many ties, which skewed the
  quantile cuts (the top band came out 5.5% instead of 5%, and "medium" 41%
  instead of 30%). Calibration preserves order, so banding on raw scores
  gives the same ranking with no ties.

---

## 7. Seller scores and risk bands

- Seller risk = mean calibrated risk of their orders in the last 90 days
  (counted back from the newest order in the data, since the dataset ends
  in 2018).
- Sellers with fewer than 5 recent orders are not ranked; an average of
  two orders is noise.
- **Bands are percentiles, not fixed cutoffs.** Critical = top 5%, high =
  next 15%, medium = next 30%. With a ~5% base rate, a calibrated model
  rarely outputs high probabilities, so fixed cutoffs (0.25/0.50/0.75)
  would leave the top bands permanently empty. Percentile bands also match
  how a team works: "who are the worst 5% right now?"
- **SHAP:** the stacking ensemble has no single feature space, so its
  explanations come from the XGBoost model as a proxy. Each seller's top-5
  drivers are the mean SHAP values over their orders.

---

## 8. Serving

- **No inference at request time.** Predictions, bands and SHAP drivers
  are computed offline and stored. Every endpoint except `/chat` just reads
  rows.
- **SQLite snapshot instead of a database server.** The API needs a few
  thousand read-only rows. Render's free Postgres expires after 30 days; a
  committed SQLite file doesn't expire and holds no credentials.
- **Slim serving image** (`requirements-serve.txt`): no torch, xgboost,
  lightgbm, shap, mlflow or optuna, so it fits a 512 MB free instance.
- **Cold starts:** free instances sleep after 15 idle minutes. A GitHub
  cron pings them every 5 minutes. The dashboard probes `/health` once with
  a long timeout and backs off on 429, instead of firing many retries
  (which is what triggered rate limits before). Failed calls are never
  cached.

---

## 9. RAG chat

- **Documents:** one per seller (score, band, SHAP drivers) plus about
  2,000 real customer reviews, kept in Portuguese as in the source data.
- **Embeddings:** ONNX `all-MiniLM-L6-v2` bundled with chromadb (no
  PyTorch), falling back to sentence-transformers, then TF-IDF offline. The
  backend used at index time is saved and checked at query time, because
  mixing backends puts queries and documents in different vector spaces.
- **Generation:** Groq, trying a list of models in order. The original
  model (`llama-3.3-70b-versatile`) was retired mid-project and started
  returning 404, so one pinned model name is a single point of failure.
  Without a key, `/chat` returns the retrieved context instead of an error.
- The prompt tells the model to answer only from the retrieved context and
  to say so when the context isn't enough.

---

## 10. Data provenance

| Data | Source |
|---|---|
| Orders, sellers, products, reviews, late label | Real (Olist) |
| Commodity volatility | Real (FRED WTI oil price) when `FRED_API_KEY` is set |
| Weather, port congestion | Generated: no free source has 2016–2018 history |

Generated signals never look at the label, so they can't fake a
correlation, and every row carries `is_synthetic` (0 = real, 1 = generated).

---

## 11. Orchestration and MLOps

- **Airflow** runs in its own environment because it pins
  SQLAlchemy < 2.0. Each DAG task runs a SENTRIX script through
  `BashOperator` instead of importing project code.
- **MLflow** logs every model and registers the winner as "Production".
- **DVC** versions the raw data (not redistributable, so no public remote).
- **Evidently** compares a reference slice with recent data for drift.
- **CI** runs lint, unit tests and the leakage tests on every push. The
  full retrain + PR-AUC gate is manual, since CI can't access the data.
- **README results** are generated from `artifacts/evaluation/` by
  `scripts/render_results.py`. Hand-copied numbers had drifted out of date
  after retraining.
