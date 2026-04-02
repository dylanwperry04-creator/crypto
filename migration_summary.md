# Migration summary

This archive now uses a CSV-first historical storage design:

- `data/raw_price_candles.csv`
- `data/raw_twitter_sentiment.csv` *(optional real historical input)*
- `data/feature_matrix.csv`
- `artifacts/model_results.csv`
- `artifacts/backtest_predictions.csv`
- `artifacts/backtest_predictions_selected.csv`
- `artifacts/best_model_metadata.json`

SQLite (`data/crypto_agent.db`) is now used only for live forecast logging and later resolution.

Legacy synthetic/demo files remain only under `deprecated_demo/` and are not used in the production path.
