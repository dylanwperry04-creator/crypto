# Artifacts directory

This directory contains **offline-generated real-data artifacts** only.

Primary files:
- `model_results.csv`
- `backtest_predictions.csv`
- `backtest_predictions_selected.csv`
- `best_model_metadata.json`
- `provenance.json`
- `data_quality_report.json`
- `data_quality_checks.csv`
- `models/*.joblib`
- `features/*.csv`

These files are created or refreshed by:
- `scripts/build_features.py`
- `scripts/train_compare_models.py`
- `scripts/export_backtests.py`
