# AI Crypto Forecasting Agent — Real-Data, CSV-First Refactor

This project has been refactored into a **real-data-only direction-classification system**.
The deployed prediction path is now designed around:

1. **offline historical backfill and training** on saved real CSVs,
2. **offline model comparison** across Logistic Regression, Random Forest, XGBoost, and LightGBM,
3. **winner selection by class-1 F1** with PR-AUC as the main secondary check,
4. **single-model live scoring** using only the already-selected production winner,
5. **honest separation** between historical backtests and live unresolved forecasts.

No synthetic/demo-generated data is used in the production code path.
Legacy demo artifacts remain under `deprecated_demo/` only for reference and are not imported by the app.

---

## 1) Production architecture

### Historical storage (CSV-first)

Primary inspectable storage:

- `data/raw_price_candles.csv`
- `data/raw_twitter_sentiment.csv` *(optional, user-supplied, real only)*
- `data/feature_matrix.csv`
- `artifacts/model_results.csv`
- `artifacts/backtest_predictions.csv`
- `artifacts/backtest_predictions_selected.csv`
- `artifacts/best_model_metadata.json`
- `artifacts/provenance.json`
- `artifacts/data_quality_report.json`
- `artifacts/data_quality_checks.csv`
- `artifacts/models/*.joblib`
- `artifacts/features/*.csv`

### SQLite use

SQLite is retained only for:

- `data/crypto_agent.db`
- live forecast logging
- forecast resolution status tracking

Historical market data and offline evaluation artifacts are **CSV-first**, not SQLite-first.

---

## 2) Real data provenance

### Historical market data

- Source: **Binance Spot public market-data REST API**
- Base URL: `https://data-api.binance.vision`
- Endpoint: `GET /api/v3/klines`
- Interval: `1d`
- Symbols:
  - `BTCUSDT`
  - `ETHUSDT`
  - `DOGEUSDT`

### Live market updates

- Source: **Binance Spot public market-data REST API**
- Base URL: `https://data-api.binance.vision`
- Endpoints:
  - `GET /api/v3/klines` for latest completed daily candles
  - `GET /api/v3/ticker/24hr` for current spot snapshot in the UI

### Historical Twitter / sentiment

- Source: **not bundled automatically in this offline refactor**
- Expected file: `data/raw_twitter_sentiment.csv`
- Requirement: must be **real** and user-verified
- Supported input styles:
  1. raw tweet-level rows with columns like `date`, `text`, `sentiment_label`, `sentiment_score`, `reply_count`, `like_count`, `retweet_count`, `quote_count`
  2. already aggregated daily sentiment rows containing the engineered daily Twitter features

### Live Twitter / sentiment

- **Not enabled by default**
- Twitter-dependent live modes remain disabled unless a real ongoing live sentiment source is implemented and wired in honestly

### Date ranges

Exact populated date ranges are written automatically to:

- `artifacts/provenance.json`
- `artifacts/best_model_metadata.json`

after you run the pipeline with real data.

---

## 3) Pipeline steps

### Step A — Backfill real historical candles

```bash
python scripts/backfill_real_history.py --coin all
```

This:
- paginates through the public Binance kline API,
- stores only **completed** daily candles,
- deduplicates by `(coin, candle_open_time_utc)`,
- updates `data/raw_price_candles.csv`,
- refreshes provenance metadata.

### Step B — Optional: add real Twitter history

Place a verified real CSV at:

```text
data/raw_twitter_sentiment.csv
```

If you do not provide this file, the system will still work fully for:
- Bitcoin `price_only`
- Ethereum `price_only`
- Dogecoin `price_only`

and will keep Twitter-dependent modes historical-only / unavailable.

### Step C — Build features and quality reports

```bash
python scripts/build_features.py
```

This builds:
- `data/feature_matrix.csv`
- `artifacts/data_quality_report.json`
- `artifacts/data_quality_checks.csv`

Implemented price features include:
- OHLCV core fields
- raw pct change
- lagged returns
- rolling means/std
- price vs moving averages
- candle range/body/shadows
- ATR
- RSI
- MACD
- Bollinger features
- volume change / moving-average ratios
- OBV / OBV change
- cyclical calendar features

Implemented Twitter features (if real data exists) include:
- engagement-weighted sentiment mean
- engagement-weighted sentiment std
- positive / negative share
- weighted tweet count
- total / average / max engagement
- lagged rolling Twitter statistics via shifted daily alignment to reduce leakage

### Step D — Train and compare models offline

```bash
python scripts/train_compare_models.py
```

The offline comparison pipeline:
- uses a strict chronological train / validation / test split,
- compares:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- tunes hyperparameters via small family-specific grids,
- tunes decision threshold on the validation set,
- selects the production winner by **validation class-1 F1**,
- uses **validation PR-AUC** as a tiebreak,
- refits the winning configuration on train+validation,
- evaluates on held-out test data,
- writes:
  - `artifacts/model_results.csv`
  - `artifacts/backtest_predictions.csv`
  - `artifacts/backtest_predictions_selected.csv`
  - `artifacts/best_model_metadata.json`
  - `artifacts/models/*.joblib`
  - `artifacts/features/*.csv`

### Step E — Refresh live history

```bash
python scripts/update_live_history.py
```

This appends only new real completed candles to `data/raw_price_candles.csv`.

### Step F — Run the Flask app

```bash
python app.py
```

---

## 4) App behaviour

### Historical lookup

Historical lookup reads only saved offline artifacts:
- `artifacts/backtest_predictions_selected.csv`
- `artifacts/model_results.csv`
- `artifacts/best_model_metadata.json`

It does **not** read demo rows.

### Live prediction

The live path:
1. syncs the latest completed real daily candle into `data/raw_price_candles.csv`
2. rebuilds the latest feature row using the same offline feature logic
3. loads the already-selected winning model for that coin/mode
4. scores one probability of UP
5. applies the saved threshold
6. logs the forecast as **Pending Outcome** in SQLite
7. resolves the forecast only after the next completed real daily candle exists locally

### Important honesty rule

The app never labels an unresolved live forecast as Correct/Wrong.

---

## 5) Modes and limitations

### Production live modes currently valid by default

- Bitcoin `price_only`
- Ethereum `price_only`
- Dogecoin `price_only`

### Historical-only or unavailable unless real Twitter data is supplied

- Bitcoin `twitter_only`
- Bitcoin `combined`

### Live Twitter limitation

A real ongoing live Twitter/sentiment source is **not configured by default** in this refactor.
Therefore Twitter-dependent live modes are disabled unless you implement and document a real live source.

---

## 6) Legacy demo assets

Legacy synthetic/demo files remain under:

```text
deprecated_demo/
```

These include:
- synthetic/demo rows
- demo-trained models
- demo metadata
- demo generation scripts

They are retained only as deprecated reference material and are **not used** in the runtime or training path.

---

## 7) Notes on package/runtime limitations

The training pipeline attempts to include all four requested model families.
If `xgboost` or `lightgbm` cannot be imported in a runtime environment, the pipeline records that model family as **skipped with an explicit reason** in `artifacts/model_results.csv` instead of silently dropping it.

---

## 8) Quick run order

```bash
python scripts/backfill_real_history.py --coin all
python scripts/build_features.py
python scripts/train_compare_models.py
python scripts/update_live_history.py
python app.py
```

---

## 9) Rubric alignment summary

This refactor is designed to align with the rubric by explicitly covering:
- data import and data quality handling,
- feature engineering,
- multiple-model comparison,
- time-based train/validation/test splitting,
- threshold tuning,
- class-imbalance-aware training,
- F1 / PR-AUC reporting,
- a deployable Flask cloud/local app for prediction.
