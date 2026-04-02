# Data directory

CSV-first historical storage lives here.

Key files:
- `raw_price_candles.csv` — real daily OHLCV history for BTC/ETH/DOGE
- `raw_twitter_sentiment.csv` — optional real historical Twitter/sentiment input
- `feature_matrix.csv` — engineered offline feature matrix used for model training and live-row reconstruction
- `crypto_agent.db` — SQLite database used only for live forecast logging and later resolution
