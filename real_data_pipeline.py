from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional runtime dependency
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional runtime dependency
    LGBMClassifier = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FEATURES_DIR = ARTIFACTS_DIR / "features"
DB_PATH = DATA_DIR / "crypto_agent.db"

RAW_PRICE_CSV = DATA_DIR / "raw_price_candles.csv"
RAW_TWITTER_CSV = DATA_DIR / "raw_twitter_sentiment.csv"
FEATURE_MATRIX_CSV = DATA_DIR / "feature_matrix.csv"
LEGACY_ROOT_RAW_PRICE_CSV = BASE_DIR / RAW_PRICE_CSV.name
LEGACY_ROOT_RAW_TWITTER_CSV = BASE_DIR / RAW_TWITTER_CSV.name
MODEL_RESULTS_CSV = ARTIFACTS_DIR / "model_results.csv"
BACKTEST_PREDICTIONS_CSV = ARTIFACTS_DIR / "backtest_predictions.csv"
BACKTEST_SELECTED_CSV = ARTIFACTS_DIR / "backtest_predictions_selected.csv"
BEST_MODEL_METADATA_PATH = ARTIFACTS_DIR / "best_model_metadata.json"
DATA_QUALITY_JSON = ARTIFACTS_DIR / "data_quality_report.json"
DATA_QUALITY_CSV = ARTIFACTS_DIR / "data_quality_checks.csv"
PROVENANCE_JSON = ARTIFACTS_DIR / "provenance.json"

COINS = ["Bitcoin", "Ethereum", "Dogecoin"]
PRICE_ONLY_MODE = "price_only"
TWITTER_ONLY_MODE = "twitter_only"
COMBINED_MODE = "combined"
MODE_LABELS = {
    PRICE_ONLY_MODE: "Price Only",
    TWITTER_ONLY_MODE: "Twitter Only",
    COMBINED_MODE: "Price + Twitter",
}
BITCOIN_MODES = [PRICE_ONLY_MODE, TWITTER_ONLY_MODE, COMBINED_MODE]

PUBLIC_MARKET_API_BASE = "https://data-api.binance.vision"
MARKET_DATA_PATH = "/api/v3/klines"
TICKER_24H_PATH = "/api/v3/ticker/24hr"
PRIMARY_HISTORICAL_SOURCE = "Binance Spot public market-data REST API"
PRIMARY_LIVE_SOURCE = "Binance Spot public market-data REST API"
PRIMARY_MARKET_SOURCE = PRIMARY_LIVE_SOURCE
TWITTER_HISTORICAL_SOURCE = "User-supplied real historical Twitter/sentiment CSV"
TWITTER_LIVE_SOURCE = None
TWITTER_LIVE_AVAILABLE = False
TIMEZONE_POLICY = "UTC"
FORECAST_HORIZON_LABEL = "1d_direction"
INTERVAL = "1d"
INTERVAL_MS = 86_400_000
ROLLING_WINDOW = 30

COIN_SYMBOLS = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "Dogecoin": "DOGEUSDT",
}
BACKFILL_DEFAULT_START = {
    "Bitcoin": "2017-01-01",
    "Ethereum": "2017-08-01",
    "Dogecoin": "2019-07-01",
}

PRICE_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "pct_change_raw",
    "return_1",
    "return_3",
    "return_7",
    "return_14",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_30",
    "rolling_std_30",
    "price_vs_ma7",
    "price_vs_ma14",
    "price_vs_ma30",
    "range_pct",
    "upper_shadow_pct",
    "lower_shadow_pct",
    "body_size_pct",
    "atr_14",
    "rsi_7",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_diff",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "bb_pct",
    "volume_change",
    "volume_ma7",
    "volume_vs_ma7",
    "obv",
    "obv_change",
    "lag1_return",
    "lag2_return",
    "lag3_return",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "quarter",
    "is_month_end",
    "is_month_start",
]

TWITTER_AGG_FEATURES = [
    "weighted_sentiment_mean",
    "weighted_sentiment_std",
    "weighted_positive_share",
    "weighted_negative_share",
    "weighted_tweet_count",
    "total_engagement",
    "average_engagement",
    "max_engagement",
    "raw_tweet_count",
    "sent_ma3",
    "sent_ma7",
    "sent_chg3",
    "sent_chg7",
    "tweet_mass_ma3",
    "tweet_mass_ma7",
]

FORECASTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id TEXT PRIMARY KEY,
    identity_key TEXT NOT NULL UNIQUE,
    forecast_context TEXT NOT NULL,
    coin TEXT NOT NULL,
    mode TEXT NOT NULL,
    mode_label TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    resolved_at TEXT,
    forecast_timestamp_utc TEXT NOT NULL,
    feature_date TEXT NOT NULL,
    target_date TEXT NOT NULL,
    forecast_horizon TEXT NOT NULL,
    timezone_policy TEXT NOT NULL,
    exchange_source TEXT,
    model_artifact TEXT NOT NULL,
    model_version TEXT,
    feature_artifact TEXT NOT NULL,
    feature_version TEXT,
    metadata_artifact TEXT,
    metadata_version TEXT,
    pipeline_name TEXT,
    threshold REAL,
    probability REAL,
    predicted_class INTEGER,
    predicted_label TEXT,
    actual_class INTEGER,
    actual_label TEXT,
    reference_close REAL,
    target_close REAL,
    live_spot_price REAL,
    feature_payload_json TEXT,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_forecasts_status ON forecasts(status);
CREATE INDEX IF NOT EXISTS idx_forecasts_coin_mode ON forecasts(coin, mode);
CREATE INDEX IF NOT EXISTS idx_forecasts_target_date ON forecasts(target_date);
"""


@dataclass(frozen=True)
class TrainingOutcome:
    coin: str
    mode: str
    selected_model_family: str
    model_artifact: str
    feature_artifact: str
    threshold: float
    includes_twitter: bool
    live_valid: bool


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def init_db(db_path: Path | None = None) -> None:
    ensure_directories()
    with sqlite3.connect(str(db_path or DB_PATH)) as conn:
        conn.executescript(FORECASTS_SCHEMA_SQL)


def get_db(db_path: Path | None = None) -> sqlite3.Connection:
    ensure_directories()
    conn = sqlite3.connect(str(db_path or DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def symbol_for_coin(coin: str) -> str:
    if coin not in COIN_SYMBOLS:
        raise KeyError(f"Unsupported coin: {coin}")
    return COIN_SYMBOLS[coin]


def parse_ts_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def to_iso_date(value: Any) -> str:
    return parse_ts_utc(value).strftime("%Y-%m-%d")


def safe_read_csv(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=parse_dates or [])
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    ensure_directories()
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    ensure_directories()
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def unique_preserve_order(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for col in columns:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def csv_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        return len(pd.read_csv(path, nrows=1)) > 0
    except pd.errors.EmptyDataError:
        return False
    except Exception:
        return False


def maybe_seed_from_legacy_csv(primary: Path, *legacy_candidates: Path) -> None:
    if csv_has_rows(primary):
        return
    for candidate in legacy_candidates:
        if candidate is None:
            continue
        try:
            if candidate.resolve() == primary.resolve():
                continue
        except Exception:
            pass
        if not csv_has_rows(candidate):
            continue
        write_csv_atomic(pd.read_csv(candidate), primary)
        return


def file_version(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.is_dir():
        return None
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return {
        "file": path.name,
        "sha256_12": digest,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
    }


def model_artifact_path(coin: str, mode: str) -> Path:
    return MODELS_DIR / f"{coin.lower()}_{mode}_model.joblib"


def feature_artifact_path(coin: str, mode: str) -> Path:
    return FEATURES_DIR / f"{coin.lower()}_{mode}_features.csv"


def load_best_model_metadata() -> dict[str, Any]:
    if not BEST_MODEL_METADATA_PATH.exists():
        return {}
    return json.loads(BEST_MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def save_best_model_metadata(payload: dict[str, Any]) -> None:
    write_json_atomic(payload, BEST_MODEL_METADATA_PATH)


def bootstrap_placeholder_files() -> None:
    ensure_directories()
    init_db()
    maybe_seed_from_legacy_csv(RAW_PRICE_CSV, LEGACY_ROOT_RAW_PRICE_CSV)
    maybe_seed_from_legacy_csv(RAW_TWITTER_CSV, LEGACY_ROOT_RAW_TWITTER_CSV)
    if not RAW_PRICE_CSV.exists():
        write_csv_atomic(pd.DataFrame(columns=[
            "coin", "symbol", "candle_open_time_utc", "candle_close_time_utc", "open", "high", "low", "close",
            "volume", "quote_volume", "trade_count", "source", "source_interval", "source_symbol", "is_complete", "ingested_at_utc"
        ]), RAW_PRICE_CSV)
    if not RAW_TWITTER_CSV.exists():
        write_csv_atomic(pd.DataFrame(columns=[
            "date", "text", "sentiment_label", "sentiment_score", "reply_count", "like_count", "retweet_count", "quote_count", "source"
        ]), RAW_TWITTER_CSV)
    if not FEATURE_MATRIX_CSV.exists():
        write_csv_atomic(pd.DataFrame(columns=["coin", "mode", "date", "target_date", "target"]), FEATURE_MATRIX_CSV)
    if not MODEL_RESULTS_CSV.exists():
        write_csv_atomic(pd.DataFrame(columns=[
            "coin", "mode", "model_family", "status", "skip_reason", "selected_for_production",
            "includes_twitter", "live_valid", "feature_count", "train_rows", "validation_rows", "test_rows",
            "train_start_date", "train_end_date", "validation_start_date", "validation_end_date", "test_start_date", "test_end_date",
            "threshold", "val_f1_class1", "val_pr_auc", "val_precision", "val_recall", "val_balanced_accuracy", "val_roc_auc",
            "test_f1_class1", "test_pr_auc", "test_precision", "test_recall", "test_balanced_accuracy", "test_roc_auc",
            "hyperparameters_json", "trained_at_utc"
        ]), MODEL_RESULTS_CSV)
    if not BACKTEST_PREDICTIONS_CSV.exists():
        write_csv_atomic(pd.DataFrame(columns=[
            "coin", "mode", "model_family", "selected_for_production", "split", "feature_date", "target_date",
            "predicted_probability_up", "predicted_class", "predicted_label", "actual_class", "actual_label",
            "was_correct", "threshold", "reference_close", "target_close", "generated_at_utc"
        ]), BACKTEST_PREDICTIONS_CSV)
    if not BACKTEST_SELECTED_CSV.exists():
        write_csv_atomic(pd.DataFrame(columns=[
            "coin", "mode", "model_family", "selected_for_production", "split", "feature_date", "target_date",
            "predicted_probability_up", "predicted_class", "predicted_label", "actual_class", "actual_label",
            "was_correct", "threshold", "reference_close", "target_close", "generated_at_utc"
        ]), BACKTEST_SELECTED_CSV)
    if not DATA_QUALITY_CSV.exists():
        write_csv_atomic(pd.DataFrame(columns=["scope", "check_name", "value", "status", "notes"]), DATA_QUALITY_CSV)
    if not DATA_QUALITY_JSON.exists():
        write_json_atomic({"status": "awaiting_backfill", "generated_at_utc": utc_now_iso()}, DATA_QUALITY_JSON)
    if not PROVENANCE_JSON.exists():
        write_json_atomic(default_provenance_payload(), PROVENANCE_JSON)
    if not BEST_MODEL_METADATA_PATH.exists():
        write_json_atomic(default_metadata_payload(), BEST_MODEL_METADATA_PATH)


# ---------------------------------------------------------------------------
# Provenance and metadata
# ---------------------------------------------------------------------------

def default_provenance_payload() -> dict[str, Any]:
    return {
        "status": "awaiting_backfill",
        "generated_at_utc": utc_now_iso(),
        "price": {
            "historical_source": PRIMARY_HISTORICAL_SOURCE,
            "live_source": PRIMARY_LIVE_SOURCE,
            "api_base": PUBLIC_MARKET_API_BASE,
            "endpoint": MARKET_DATA_PATH,
            "interval": INTERVAL,
            "symbols": COIN_SYMBOLS,
            "configured_backfill_start_dates": BACKFILL_DEFAULT_START,
            "local_csv": str(RAW_PRICE_CSV.relative_to(BASE_DIR)),
        },
        "twitter": {
            "historical_source": TWITTER_HISTORICAL_SOURCE,
            "live_source": TWITTER_LIVE_SOURCE,
            "live_available": TWITTER_LIVE_AVAILABLE,
            "local_csv": str(RAW_TWITTER_CSV.relative_to(BASE_DIR)),
            "notes": "Place a verified real historical Bitcoin tweet/sentiment CSV here to enable historical Twitter-only and combined backtests.",
        },
        "artifacts": {
            "feature_matrix_csv": str(FEATURE_MATRIX_CSV.relative_to(BASE_DIR)),
            "model_results_csv": str(MODEL_RESULTS_CSV.relative_to(BASE_DIR)),
            "backtest_predictions_csv": str(BACKTEST_PREDICTIONS_CSV.relative_to(BASE_DIR)),
            "best_model_metadata_json": str(BEST_MODEL_METADATA_PATH.relative_to(BASE_DIR)),
        },
        "legacy_demo": {
            "path": "deprecated_demo/",
            "in_production_path": False,
        },
    }


def default_metadata_payload() -> dict[str, Any]:
    return {
        "status": "awaiting_backfill_and_training",
        "generated_at_utc": utc_now_iso(),
        "selection_rule": "Choose production winner by validation class-1 F1, using validation PR-AUC as secondary tiebreak. Report final held-out test metrics separately.",
        "price_source": PRIMARY_HISTORICAL_SOURCE,
        "live_price_source": PRIMARY_LIVE_SOURCE,
        "twitter_source": TWITTER_HISTORICAL_SOURCE,
        "twitter_live_available": TWITTER_LIVE_AVAILABLE,
        "selected_models": {},
        "disabled_live_modes": {
            "Bitcoin": {
                TWITTER_ONLY_MODE: "Historical-only unless a real ongoing live Twitter/sentiment source is configured.",
                COMBINED_MODE: "Historical-only unless a real ongoing live Twitter/sentiment source is configured.",
            }
        },
        "artifact_paths": {
            "model_results_csv": str(MODEL_RESULTS_CSV.relative_to(BASE_DIR)),
            "backtest_predictions_csv": str(BACKTEST_PREDICTIONS_CSV.relative_to(BASE_DIR)),
            "backtest_predictions_selected_csv": str(BACKTEST_SELECTED_CSV.relative_to(BASE_DIR)),
            "feature_matrix_csv": str(FEATURE_MATRIX_CSV.relative_to(BASE_DIR)),
        },
    }


# ---------------------------------------------------------------------------
# Data retrieval and storage
# ---------------------------------------------------------------------------

def _request_json(path: str, params: dict[str, Any], timeout: int = 30) -> Any:
    url = f"{PUBLIC_MARKET_API_BASE}{path}"
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_public_klines(symbol: str, start_time_ms: int | None = None, end_time_ms: int | None = None, limit: int = 1000) -> pd.DataFrame:
    params: dict[str, Any] = {"symbol": symbol, "interval": INTERVAL, "limit": int(limit)}
    if start_time_ms is not None:
        params["startTime"] = int(start_time_ms)
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    payload = _request_json(MARKET_DATA_PATH, params=params, timeout=30)
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected kline response for {symbol}: {payload}")
    if not payload:
        return pd.DataFrame()
    cols = [
        "open_time_ms", "open", "high", "low", "close", "volume", "close_time_ms", "quote_volume",
        "trade_count", "taker_buy_base", "taker_buy_quote", "ignore"
    ]
    df = pd.DataFrame(payload, columns=cols)
    for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_count"] = pd.to_numeric(df["trade_count"], errors="coerce")
    df["open_time_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce")
    df["close_time_ms"] = pd.to_numeric(df["close_time_ms"], errors="coerce")
    return df.dropna(subset=["open_time_ms", "close_time_ms", "open", "high", "low", "close", "volume"]).copy()


def fetch_live_ticker(coin: str) -> dict[str, Any]:
    payload = _request_json(TICKER_24H_PATH, {"symbol": symbol_for_coin(coin)}, timeout=20)
    return {
        "coin": coin,
        "symbol": symbol_for_coin(coin),
        "price": float(payload["lastPrice"]),
        "high": float(payload["highPrice"]),
        "low": float(payload["lowPrice"]),
        "volume": float(payload["volume"]),
        "quote_volume": float(payload["quoteVolume"]),
        "source": PRIMARY_LIVE_SOURCE,
        "timestamp": utc_now_iso(),
    }


def normalize_klines(coin: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["coin"] = coin
    out["symbol"] = symbol_for_coin(coin)
    out["candle_open_time_utc"] = pd.to_datetime(out["open_time_ms"], unit="ms", utc=True)
    out["candle_close_time_utc"] = pd.to_datetime(out["close_time_ms"], unit="ms", utc=True)
    now_ts = pd.Timestamp.now(tz="UTC")
    out["is_complete"] = (out["candle_close_time_utc"] < now_ts).astype(int)
    out["source"] = PRIMARY_HISTORICAL_SOURCE
    out["source_interval"] = INTERVAL
    out["source_symbol"] = out["symbol"]
    out["ingested_at_utc"] = utc_now_iso()
    keep = [
        "coin", "symbol", "candle_open_time_utc", "candle_close_time_utc", "open", "high", "low", "close",
        "volume", "quote_volume", "trade_count", "source", "source_interval", "source_symbol", "is_complete", "ingested_at_utc"
    ]
    out = out[keep].copy()
    out["trade_count"] = out["trade_count"].fillna(0).astype(int)
    out["candle_open_time_utc"] = out["candle_open_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["candle_close_time_utc"] = out["candle_close_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out[out["is_complete"] == 1].copy()


def read_price_history(coin: str | None = None) -> pd.DataFrame:
    df = safe_read_csv(RAW_PRICE_CSV, parse_dates=["candle_open_time_utc", "candle_close_time_utc"])
    if df.empty:
        return df
    for col in ["open", "high", "low", "close", "volume", "quote_volume", "trade_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["candle_open_time_utc"] = pd.to_datetime(df["candle_open_time_utc"], utc=True)
    df["candle_close_time_utc"] = pd.to_datetime(df["candle_close_time_utc"], utc=True)
    df["date"] = df["candle_open_time_utc"].dt.strftime("%Y-%m-%d")
    if coin is not None:
        df = df[df["coin"] == coin].copy()
    return df.sort_values(["coin", "candle_open_time_utc"]).reset_index(drop=True)


def write_price_history(df: pd.DataFrame) -> None:
    if df.empty:
        bootstrap_placeholder_files()
        return
    out = df.copy().sort_values(["coin", "candle_open_time_utc"]).drop_duplicates(subset=["coin", "candle_open_time_utc"], keep="last")
    out["candle_open_time_utc"] = pd.to_datetime(out["candle_open_time_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["candle_close_time_utc"] = pd.to_datetime(out["candle_close_time_utc"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    write_csv_atomic(out, RAW_PRICE_CSV)


def append_or_replace_price_rows(new_rows: pd.DataFrame) -> int:
    history = read_price_history()
    before = 0 if history.empty else len(history)
    combined = pd.concat([history, new_rows], ignore_index=True) if not history.empty else new_rows.copy()
    combined = combined.drop_duplicates(subset=["coin", "candle_open_time_utc"], keep="last")
    combined = combined[combined["is_complete"].astype(int) == 1].copy()
    write_price_history(combined)
    after = len(read_price_history())
    return max(after - before, 0)


def latest_stored_candle_time(coin: str) -> pd.Timestamp | None:
    history = read_price_history(coin)
    if history.empty:
        return None
    return pd.to_datetime(history["candle_open_time_utc"], utc=True).max()


def backfill_coin_history(coin: str, start_date: str | None = None, pause_seconds: float = 0.25, batch_limit: int = 1000) -> dict[str, Any]:
    start_date = start_date or BACKFILL_DEFAULT_START[coin]
    existing_latest = latest_stored_candle_time(coin)
    cursor_ms = int(parse_ts_utc(existing_latest + pd.Timedelta(days=1)).timestamp() * 1000) if existing_latest is not None else int(parse_ts_utc(start_date).timestamp() * 1000)
    now_ms = int(utc_now().timestamp() * 1000)
    fetched_rows = 0
    inserted_rows = 0
    loops = 0

    while cursor_ms < now_ms:
        loops += 1
        raw = fetch_public_klines(symbol_for_coin(coin), start_time_ms=cursor_ms, limit=batch_limit)
        if raw.empty:
            break
        normalized = normalize_klines(coin, raw)
        fetched_rows += len(normalized)
        if not normalized.empty:
            inserted_rows += append_or_replace_price_rows(normalized)
            last_open_ms = int(raw["open_time_ms"].max())
            next_cursor = last_open_ms + INTERVAL_MS
            if next_cursor <= cursor_ms:
                break
            cursor_ms = next_cursor
        else:
            cursor_ms += batch_limit * INTERVAL_MS
        if len(raw) < batch_limit:
            break
        time.sleep(pause_seconds)

    history = read_price_history(coin)
    latest = None if history.empty else pd.to_datetime(history["candle_open_time_utc"], utc=True).max()
    update_provenance_from_files()
    return {
        "coin": coin,
        "symbol": symbol_for_coin(coin),
        "source": PRIMARY_HISTORICAL_SOURCE,
        "fetched_rows": int(fetched_rows),
        "stored_rows": int(len(history)),
        "inserted_or_updated": int(inserted_rows),
        "latest_stored_candle": latest.isoformat() if latest is not None else None,
        "iterations": int(loops),
    }


def incremental_update_coin(coin: str, lookback_candles: int = 7) -> dict[str, Any]:
    latest = latest_stored_candle_time(coin)
    if latest is None:
        return backfill_coin_history(coin)
    start_ms = int((latest - pd.Timedelta(days=lookback_candles)).timestamp() * 1000)
    raw = fetch_public_klines(symbol_for_coin(coin), start_time_ms=start_ms, limit=1000)
    normalized = normalize_klines(coin, raw)
    inserted = append_or_replace_price_rows(normalized)
    history = read_price_history(coin)
    latest_after = None if history.empty else pd.to_datetime(history["candle_open_time_utc"], utc=True).max()
    update_provenance_from_files()
    return {
        "coin": coin,
        "symbol": symbol_for_coin(coin),
        "source": PRIMARY_LIVE_SOURCE,
        "fetched_rows": int(len(normalized)),
        "inserted_or_updated": int(inserted),
        "latest_stored_candle": latest_after.isoformat() if latest_after is not None else None,
    }


def incremental_update_all() -> list[dict[str, Any]]:
    return [incremental_update_coin(coin) for coin in COINS]


# ---------------------------------------------------------------------------
# Twitter ingestion and aggregation
# ---------------------------------------------------------------------------

def twitter_history_available() -> bool:
    df = safe_read_csv(RAW_TWITTER_CSV)
    return not df.empty


def read_twitter_raw() -> pd.DataFrame:
    df = safe_read_csv(RAW_TWITTER_CSV)
    if df.empty:
        return df

    if "date" not in df.columns:
        raise ValueError(f"{RAW_TWITTER_CSV.name} must contain a date column.")

    # Safer for dates like 01/01/2022
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"]).copy()

    # If the file is already daily aggregated, pass it through unchanged.
    if all(col in df.columns for col in ["weighted_sentiment_mean", "weighted_sentiment_std"]):
        return df

    for col in ["reply_count", "like_count", "retweet_count", "quote_count", "sentiment_score"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "sentiment_label" not in df.columns:
        # Support already-signed polarity scores.
        signed_score = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
        df["sentiment_score"] = signed_score
        df["sentiment_label"] = np.select(
            [signed_score > 0, signed_score < 0],
            ["positive", "negative"],
            default="neutral",
        )
    else:
        # Convert label + confidence into signed polarity.
        # Positive -> +score, Negative -> -score, Neutral -> 0
        label_norm = df["sentiment_label"].astype(str).str.lower().str.strip()
        raw_score = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)

        df["sentiment_score"] = np.select(
            [
                label_norm == "positive",
                label_norm == "negative",
                label_norm == "neutral",
            ],
            [
                raw_score,
                -raw_score,
                0.0,
            ],
            default=0.0,
        )
        df["sentiment_label"] = label_norm

    df["raw_engagement"] = df[["reply_count", "like_count", "retweet_count", "quote_count"]].sum(axis=1)
    log_cap = 10.0
    df["eng_weight"] = np.log1p(df["raw_engagement"]).clip(upper=log_cap)
    df.loc[df["raw_engagement"] <= 0, "eng_weight"] = 0.01

    label_norm = df["sentiment_label"].astype(str).str.lower().str.strip()
    df["is_positive"] = (label_norm == "positive").astype(float)
    df["is_negative"] = (label_norm == "negative").astype(float)

    return df


def aggregate_twitter_daily(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return raw_df
    if all(col in raw_df.columns for col in TWITTER_AGG_FEATURES):
        daily = raw_df.copy()
        daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.normalize()
        daily["coin"] = "Bitcoin"
        return daily.sort_values("date").reset_index(drop=True)

    df = raw_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.normalize()
    df["w_sentiment"] = df["sentiment_score"] * df["eng_weight"]
    df["w_positive"] = df["is_positive"] * df["eng_weight"]
    df["w_negative"] = df["is_negative"] * df["eng_weight"]
    df["w_sq_sentiment"] = (df["sentiment_score"] ** 2) * df["eng_weight"]
    g = df.groupby("date", as_index=True)
    w_sum = g["eng_weight"].sum().rename("_w_sum")
    daily = pd.DataFrame(index=w_sum.index)
    daily["weighted_sentiment_mean"] = (g["w_sentiment"].sum() / w_sum.replace(0, np.nan)).fillna(0.0)
    mean_sq = (g["w_sq_sentiment"].sum() / w_sum.replace(0, np.nan)).fillna(0.0)
    sq_mean = (daily["weighted_sentiment_mean"] ** 2)
    daily["weighted_sentiment_std"] = np.sqrt((mean_sq - sq_mean).clip(lower=0)).fillna(0.0)
    daily["weighted_positive_share"] = (g["w_positive"].sum() / w_sum.replace(0, np.nan)).fillna(0.0)
    daily["weighted_negative_share"] = (g["w_negative"].sum() / w_sum.replace(0, np.nan)).fillna(0.0)
    daily["weighted_tweet_count"] = w_sum.fillna(0.0)
    daily["total_engagement"] = g["raw_engagement"].sum().fillna(0.0)
    daily["average_engagement"] = g["raw_engagement"].mean().fillna(0.0)
    daily["max_engagement"] = g["raw_engagement"].max().fillna(0.0)
    daily["raw_tweet_count"] = g["raw_engagement"].count().fillna(0.0)
    for window in [3, 7]:
        daily[f"sent_ma{window}"] = daily["weighted_sentiment_mean"].rolling(window, min_periods=1).mean()
        daily[f"sent_chg{window}"] = daily["weighted_sentiment_mean"].diff(window).fillna(0.0)
        daily[f"tweet_mass_ma{window}"] = daily["weighted_tweet_count"].rolling(window, min_periods=1).mean()
    daily = daily.reset_index()
    daily["coin"] = "Bitcoin"
    return daily.sort_values("date").reset_index(drop=True)


def read_twitter_daily() -> pd.DataFrame:
    raw = read_twitter_raw()
    if raw.empty:
        return raw
    daily = aggregate_twitter_daily(raw)
    daily["date"] = pd.to_datetime(daily["date"], utc=True).dt.normalize()
    for col in TWITTER_AGG_FEATURES:
        if col not in daily.columns:
            daily[col] = 0.0
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0.0)
    return daily[["coin", "date"] + TWITTER_AGG_FEATURES].copy()


# ---------------------------------------------------------------------------
# Feature engineering and quality checks
# ---------------------------------------------------------------------------

def _compute_rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _compute_atr(group: pd.DataFrame, window: int = 14) -> pd.Series:
    high = group["high"]
    low = group["low"]
    close = group["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=1).mean().fillna(0.0)


def build_price_feature_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    if price_df.empty:
        return price_df
    df = price_df.copy().sort_values(["coin", "candle_open_time_utc"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["candle_open_time_utc"], utc=True).dt.normalize()

    grp = df.groupby("coin", group_keys=False)
    df["pct_change_raw"] = ((df["close"] - df["open"]) / df["open"].replace(0, np.nan)).fillna(0.0)
    for window in [1, 3, 7, 14]:
        df[f"return_{window}"] = grp["close"].pct_change(window).fillna(0.0)
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = grp["close"].transform(lambda s: s.rolling(window, min_periods=1).mean())
        df[f"rolling_std_{window}"] = grp["close"].transform(lambda s: s.rolling(window, min_periods=1).std(ddof=0)).fillna(0.0)
    df["price_vs_ma7"] = ((df["close"] / df["rolling_mean_7"].replace(0, np.nan)) - 1).fillna(0.0)
    df["price_vs_ma14"] = ((df["close"] / df["rolling_mean_14"].replace(0, np.nan)) - 1).fillna(0.0)
    df["price_vs_ma30"] = ((df["close"] / df["rolling_mean_30"].replace(0, np.nan)) - 1).fillna(0.0)
    df["range_pct"] = ((df["high"] - df["low"]) / df["open"].replace(0, np.nan)).fillna(0.0)
    max_oc = df[["open", "close"]].max(axis=1)
    min_oc = df[["open", "close"]].min(axis=1)
    df["upper_shadow_pct"] = ((df["high"] - max_oc) / df["open"].replace(0, np.nan)).fillna(0.0)
    df["lower_shadow_pct"] = ((min_oc - df["low"]) / df["open"].replace(0, np.nan)).fillna(0.0)
    df["body_size_pct"] = ((df["close"] - df["open"]) / df["open"].replace(0, np.nan)).fillna(0.0)
    df["atr_14"] = grp.apply(lambda g: _compute_atr(g, 14)).reset_index(level=0, drop=True)
    df["rsi_7"] = grp["close"].transform(lambda s: _compute_rsi(s, 7))
    df["rsi_14"] = grp["close"].transform(lambda s: _compute_rsi(s, 14))
    ema12 = grp["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema26 = grp["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd"] = ema12 - ema26
    df["macd_signal"] = grp["macd"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_diff"] = df["macd"] - df["macd_signal"]
    bb_mid = grp["close"].transform(lambda s: s.rolling(20, min_periods=1).mean())
    bb_std = grp["close"].transform(lambda s: s.rolling(20, min_periods=1).std(ddof=0)).fillna(0.0)
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    df["bb_width"] = ((df["bb_upper"] - df["bb_lower"]) / bb_mid.replace(0, np.nan)).fillna(0.0)
    df["bb_pct"] = ((df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)).fillna(0.5)
    df["volume_change"] = grp["volume"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["volume_ma7"] = grp["volume"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    df["volume_vs_ma7"] = ((df["volume"] / df["volume_ma7"].replace(0, np.nan)) - 1).fillna(0.0)

    def _obv(g: pd.DataFrame) -> pd.Series:
        direction = np.sign(g["close"].diff().fillna(0.0))
        return (direction * g["volume"]).cumsum()

    df["obv"] = grp.apply(_obv).reset_index(level=0, drop=True)
    df["obv_change"] = grp["obv"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["lag1_return"] = grp["return_1"].shift(1).fillna(0.0)
    df["lag2_return"] = grp["return_1"].shift(2).fillna(0.0)
    df["lag3_return"] = grp["return_1"].shift(3).fillna(0.0)
    dow = pd.to_datetime(df["date"]).dt.dayofweek
    month = pd.to_datetime(df["date"]).dt.month
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter.astype(int)
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)
    df["is_month_start"] = pd.to_datetime(df["date"]).dt.is_month_start.astype(int)
    df["target_date"] = grp["date"].shift(-1)
    df["next_close"] = grp["close"].shift(-1)
    df["target"] = (df["next_close"] > df["close"]).astype(float)
    df.loc[df["next_close"].isna(), "target"] = np.nan
    return df


def build_feature_matrix() -> pd.DataFrame:
    bootstrap_placeholder_files()
    price_df = read_price_history()
    if price_df.empty:
        write_csv_atomic(pd.DataFrame(columns=["coin", "mode", "date", "target_date", "target"]), FEATURE_MATRIX_CSV)
        return pd.DataFrame()

    feat = build_price_feature_frame(price_df)
    feat["_twitter_exact_match"] = False
    twitter_daily = read_twitter_daily()
    if not twitter_daily.empty:
        twitter_daily = twitter_daily.copy().sort_values("date")
        twitter_daily["date"] = pd.to_datetime(twitter_daily["date"], utc=True).dt.normalize() + pd.Timedelta(days=1)
        feat = feat.merge(twitter_daily, on=["coin", "date"], how="left")
        merged_twitter_cols = [c for c in TWITTER_AGG_FEATURES if c in feat.columns]
        if merged_twitter_cols:
            feat["_twitter_exact_match"] = feat[merged_twitter_cols].notna().any(axis=1)
    for col in TWITTER_AGG_FEATURES:
        if col not in feat.columns:
            feat[col] = 0.0
        feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0.0)

    feature_sets: list[pd.DataFrame] = []
    for coin in COINS:
        coin_df = feat[feat["coin"] == coin].copy()
        if coin_df.empty:
            continue
        price_only_cols = [c for c in PRICE_FEATURES if c in coin_df.columns]
        twitter_cols = [c for c in TWITTER_AGG_FEATURES if c in coin_df.columns]
        base_cols = ["coin", "date", "target_date", "target", "next_close"]

        price_mode_cols = unique_preserve_order(base_cols + ["close"] + price_only_cols)
        price_mode = coin_df.loc[:, price_mode_cols].copy()
        price_mode["mode"] = PRICE_ONLY_MODE
        feature_sets.append(price_mode)

        if coin == "Bitcoin" and twitter_history_available():
            aligned_coin_df = coin_df[coin_df["_twitter_exact_match"].fillna(False)].copy()
            if not aligned_coin_df.empty:
                tw_only_cols = unique_preserve_order(base_cols + ["close"] + twitter_cols)
                tw_only = aligned_coin_df.loc[:, tw_only_cols].copy()
                tw_only["mode"] = TWITTER_ONLY_MODE
                feature_sets.append(tw_only)

                combined_cols = unique_preserve_order(base_cols + ["close"] + price_only_cols + twitter_cols)
                combined = aligned_coin_df.loc[:, combined_cols].copy()
                combined["mode"] = COMBINED_MODE
                feature_sets.append(combined)

    feature_sets = [frame.loc[:, ~frame.columns.duplicated()].copy() for frame in feature_sets]
    feature_matrix = pd.concat(feature_sets, ignore_index=True) if feature_sets else pd.DataFrame()
    if not feature_matrix.empty:
        feature_matrix["date"] = pd.to_datetime(feature_matrix["date"], utc=True).dt.strftime("%Y-%m-%d")
        feature_matrix["target_date"] = pd.to_datetime(feature_matrix["target_date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    write_csv_atomic(feature_matrix, FEATURE_MATRIX_CSV)
    generate_quality_reports(price_df, twitter_daily if not twitter_daily.empty else pd.DataFrame(), feature_matrix)
    update_provenance_from_files()
    return feature_matrix


def read_feature_matrix(coin: str | None = None, mode: str | None = None) -> pd.DataFrame:
    df = safe_read_csv(FEATURE_MATRIX_CSV)
    if df.empty:
        return df
    if coin is not None:
        df = df[df["coin"] == coin].copy()
    if mode is not None:
        df = df[df["mode"] == mode].copy()
    return df.sort_values(["coin", "mode", "date"]).reset_index(drop=True)


def feature_columns_for_mode(coin: str, mode: str, frame: pd.DataFrame) -> list[str]:
    protected = {"coin", "mode", "date", "target_date", "target", "next_close", "close"}
    cols = [c for c in frame.columns if c not in protected]
    if mode == PRICE_ONLY_MODE:
        return [c for c in cols if c in PRICE_FEATURES]
    if mode == TWITTER_ONLY_MODE:
        return [c for c in cols if c in TWITTER_AGG_FEATURES]
    return cols


def latest_feature_row(coin: str, mode: str = PRICE_ONLY_MODE) -> pd.Series | None:
    df = read_feature_matrix(coin, mode)
    if df.empty:
        df = build_feature_matrix()
        df = read_feature_matrix(coin, mode)
    if df.empty:
        return None
    usable = df[df["target"].isna()].copy()
    if usable.empty:
        return df.sort_values("date").iloc[-1]
    return usable.sort_values("date").iloc[-1]


def generate_quality_reports(price_df: pd.DataFrame, twitter_daily: pd.DataFrame, feature_matrix: pd.DataFrame) -> None:
    checks: list[dict[str, Any]] = []

    def add(scope: str, name: str, value: Any, status: str, notes: str = "") -> None:
        checks.append({"scope": scope, "check_name": name, "value": value, "status": status, "notes": notes})

    if price_df.empty:
        add("raw_price", "rows_present", 0, "error", "Run scripts/backfill_real_history.py first.")
    else:
        add("raw_price", "rows_present", len(price_df), "ok")
        dupes = int(price_df.duplicated(subset=["coin", "candle_open_time_utc"]).sum())
        add("raw_price", "duplicate_rows", dupes, "ok" if dupes == 0 else "warn")
        missing = int(price_df[["open", "high", "low", "close", "volume"]].isna().sum().sum())
        add("raw_price", "missing_core_values", missing, "ok" if missing == 0 else "warn")
        incomplete = int((price_df.get("is_complete", 1).astype(int) == 0).sum()) if "is_complete" in price_df.columns else 0
        add("raw_price", "incomplete_candles_present", incomplete, "ok" if incomplete == 0 else "warn")
        outliers = int((price_df["close"].pct_change().abs() > 0.5).sum())
        add("raw_price", "extreme_daily_moves_abs_gt_50pct", outliers, "ok" if outliers == 0 else "warn")

    if twitter_daily.empty:
        add("twitter", "rows_present", 0, "info", "Historical Twitter modes stay disabled until a real CSV is supplied.")
    else:
        add("twitter", "rows_present", len(twitter_daily), "ok")
        dupes = int(twitter_daily.duplicated(subset=["coin", "date"]).sum())
        add("twitter", "duplicate_daily_rows", dupes, "ok" if dupes == 0 else "warn")
        missing = int(twitter_daily[TWITTER_AGG_FEATURES].isna().sum().sum())
        add("twitter", "missing_daily_feature_values", missing, "ok" if missing == 0 else "warn")

    if feature_matrix.empty:
        add("feature_matrix", "rows_present", 0, "error", "Build features after backfilling price history.")
    else:
        add("feature_matrix", "rows_present", len(feature_matrix), "ok")
        dupes = int(feature_matrix.duplicated(subset=["coin", "mode", "date"]).sum())
        add("feature_matrix", "duplicate_rows", dupes, "ok" if dupes == 0 else "warn")
        leakage_target_na = int(feature_matrix[feature_matrix["target_date"].isna()].shape[0])
        add("feature_matrix", "unresolved_target_rows", leakage_target_na, "info", "Latest row is allowed to have no target yet.")
        if "target" in feature_matrix.columns:
            target_non_na = feature_matrix["target"].dropna()
            if not target_non_na.empty:
                pos_rate = float(target_non_na.mean())
                add("feature_matrix", "class1_rate", round(pos_rate, 6), "ok")
        missing = int(feature_matrix.isna().sum().sum())
        add("feature_matrix", "missing_cells_total", missing, "ok" if missing == 0 else "warn", "Twitter-only latest target row may remain NaN until next candle exists.")

    checks_df = pd.DataFrame(checks)
    write_csv_atomic(checks_df, DATA_QUALITY_CSV)
    summary = {
        "generated_at_utc": utc_now_iso(),
        "price_rows": int(len(price_df)),
        "twitter_rows": int(len(twitter_daily)),
        "feature_rows": int(len(feature_matrix)),
        "checks": checks,
    }
    write_json_atomic(summary, DATA_QUALITY_JSON)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def time_split(frame: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("date").reset_index(drop=True)
    n = len(ordered)
    if n < 120:
        raise RuntimeError(f"Need at least 120 completed rows to train robustly; found {n}.")
    train_end = max(int(n * train_ratio), 60)
    val_end = max(int(n * (train_ratio + val_ratio)), train_end + 20)
    val_end = min(val_end, n - 20)
    train_df = ordered.iloc[:train_end].copy()
    val_df = ordered.iloc[train_end:val_end].copy()
    test_df = ordered.iloc[val_end:].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError("Time split produced an empty partition. Increase data history.")
    return train_df, val_df, test_df


def class_weight_ratio(y: pd.Series) -> float:
    positives = max(int((y == 1).sum()), 1)
    negatives = max(int((y == 0).sum()), 1)
    return negatives / positives


def candidate_models(y_train: pd.Series) -> list[dict[str, Any]]:
    ratio = class_weight_ratio(y_train)
    candidates: list[dict[str, Any]] = []
    for c in [0.5, 1.0, 2.0]:
        candidates.append({
            "family": "LogisticRegression",
            "params": {"C": c, "class_weight": "balanced"},
            "estimator": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=c, max_iter=1500, class_weight="balanced", random_state=42)),
            ]),
        })
    for n_estimators, max_depth in [(250, 6), (400, 8), (500, 10)]:
        candidates.append({
            "family": "RandomForest",
            "params": {"n_estimators": n_estimators, "max_depth": max_depth, "class_weight": "balanced_subsample"},
            "estimator": RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        })
    if XGBClassifier is not None:
        for n_estimators, max_depth, learning_rate in [(250, 3, 0.05), (400, 4, 0.03)]:
            candidates.append({
                "family": "XGBoost",
                "params": {
                    "n_estimators": n_estimators,
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "scale_pos_weight": ratio,
                },
                "estimator": XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                    scale_pos_weight=ratio,
                ),
            })
    else:
        candidates.append({"family": "XGBoost", "params": {}, "estimator": None, "skip_reason": "xgboost package unavailable at runtime."})
    if LGBMClassifier is not None:
        for n_estimators, num_leaves, learning_rate in [(250, 31, 0.05), (400, 63, 0.03)]:
            candidates.append({
                "family": "LightGBM",
                "params": {
                    "n_estimators": n_estimators,
                    "num_leaves": num_leaves,
                    "learning_rate": learning_rate,
                    "class_weight": "balanced",
                },
                "estimator": LGBMClassifier(
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    class_weight="balanced",
                    random_state=42,
                    verbosity=-1,
                ),
            })
    else:
        candidates.append({"family": "LightGBM", "params": {}, "estimator": None, "skip_reason": "lightgbm package unavailable at runtime."})
    return candidates


def optimise_threshold(y_true: Iterable[int], probabilities: Iterable[float]) -> tuple[float, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    prob_arr = np.asarray(list(probabilities), dtype=float)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.20, 0.80, 61):
        pred = (prob_arr >= threshold).astype(int)
        score = f1_score(y_true_arr, pred, zero_division=0)
        if score > best_f1 + 1e-12:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def metric_bundle(y_true: Iterable[int], probabilities: Iterable[float], threshold: float) -> dict[str, float | None]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    prob_arr = np.asarray(list(probabilities), dtype=float)
    pred_arr = (prob_arr >= threshold).astype(int)
    metrics: dict[str, float | None] = {
        "f1_class1": float(f1_score(y_true_arr, pred_arr, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true_arr, prob_arr)),
        "precision": float(precision_score(y_true_arr, pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, pred_arr, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, pred_arr)),
        "roc_auc": None,
    }
    if len(np.unique(y_true_arr)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, prob_arr))
    return metrics


def _selected_mode_live_valid(mode: str) -> bool:
    return mode == PRICE_ONLY_MODE or TWITTER_LIVE_AVAILABLE


def train_compare_models() -> list[TrainingOutcome]:
    bootstrap_placeholder_files()
    feature_matrix = read_feature_matrix()
    if feature_matrix.empty:
        feature_matrix = build_feature_matrix()
    if feature_matrix.empty:
        raise RuntimeError("No feature matrix is available. Backfill price history first.")

    all_results: list[dict[str, Any]] = []
    all_backtests: list[pd.DataFrame] = []
    selected_payload = default_metadata_payload()
    selected_payload["status"] = "ready"
    selected_payload["generated_at_utc"] = utc_now_iso()
    selected_payload["selected_models"] = {}

    for coin in COINS:
        coin_modes = [PRICE_ONLY_MODE]
        if coin == "Bitcoin" and twitter_history_available():
            coin_modes.extend([TWITTER_ONLY_MODE, COMBINED_MODE])
        selected_payload["selected_models"].setdefault(coin, {})

        for mode in coin_modes:
            frame = read_feature_matrix(coin, mode)
            if frame.empty:
                continue
            frame = frame.dropna(subset=["target", "target_date"]).copy()
            if frame.empty:
                continue
            frame["target"] = frame["target"].astype(int)
            feature_cols = feature_columns_for_mode(coin, mode, frame)
            train_df, val_df, test_df = time_split(frame)
            X_train = train_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_train = train_df["target"].astype(int)
            X_val = val_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_val = val_df["target"].astype(int)
            X_train_val = pd.concat([X_train, X_val], axis=0)
            y_train_val = pd.concat([y_train, y_val], axis=0)
            X_test = test_df[feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            y_test = test_df["target"].astype(int)

            best_candidate: dict[str, Any] | None = None
            candidate_rows: list[dict[str, Any]] = []

            for candidate in candidate_models(y_train):
                family = candidate["family"]
                estimator = candidate.get("estimator")
                if estimator is None:
                    candidate_rows.append({
                        "coin": coin,
                        "mode": mode,
                        "model_family": family,
                        "status": "skipped",
                        "skip_reason": candidate.get("skip_reason", "Unavailable"),
                        "selected_for_production": False,
                        "includes_twitter": mode in {TWITTER_ONLY_MODE, COMBINED_MODE},
                        "live_valid": _selected_mode_live_valid(mode),
                        "feature_count": len(feature_cols),
                        "train_rows": len(train_df),
                        "validation_rows": len(val_df),
                        "test_rows": len(test_df),
                        "train_start_date": train_df["date"].min(),
                        "train_end_date": train_df["date"].max(),
                        "validation_start_date": val_df["date"].min(),
                        "validation_end_date": val_df["date"].max(),
                        "test_start_date": test_df["date"].min(),
                        "test_end_date": test_df["date"].max(),
                        "threshold": None,
                        "val_f1_class1": None,
                        "val_pr_auc": None,
                        "val_precision": None,
                        "val_recall": None,
                        "val_balanced_accuracy": None,
                        "val_roc_auc": None,
                        "test_f1_class1": None,
                        "test_pr_auc": None,
                        "test_precision": None,
                        "test_recall": None,
                        "test_balanced_accuracy": None,
                        "test_roc_auc": None,
                        "hyperparameters_json": json.dumps(candidate.get("params", {})),
                        "trained_at_utc": utc_now_iso(),
                    })
                    continue

                estimator.fit(X_train, y_train)
                val_prob = estimator.predict_proba(X_val)[:, 1]
                threshold, val_best_f1 = optimise_threshold(y_val, val_prob)
                val_metrics = metric_bundle(y_val, val_prob, threshold)

                final_estimator = candidate["estimator"]
                final_estimator.fit(X_train_val, y_train_val)
                test_prob = final_estimator.predict_proba(X_test)[:, 1]
                test_metrics = metric_bundle(y_test, test_prob, threshold)

                record = {
                    "coin": coin,
                    "mode": mode,
                    "model_family": family,
                    "status": "trained",
                    "skip_reason": "",
                    "selected_for_production": False,
                    "includes_twitter": mode in {TWITTER_ONLY_MODE, COMBINED_MODE},
                    "live_valid": _selected_mode_live_valid(mode),
                    "feature_count": len(feature_cols),
                    "train_rows": len(train_df),
                    "validation_rows": len(val_df),
                    "test_rows": len(test_df),
                    "train_start_date": train_df["date"].min(),
                    "train_end_date": train_df["date"].max(),
                    "validation_start_date": val_df["date"].min(),
                    "validation_end_date": val_df["date"].max(),
                    "test_start_date": test_df["date"].min(),
                    "test_end_date": test_df["date"].max(),
                    "threshold": round(float(threshold), 6),
                    "val_f1_class1": round(float(val_best_f1), 6),
                    "val_pr_auc": round(float(val_metrics["pr_auc"]), 6),
                    "val_precision": round(float(val_metrics["precision"]), 6),
                    "val_recall": round(float(val_metrics["recall"]), 6),
                    "val_balanced_accuracy": round(float(val_metrics["balanced_accuracy"]), 6),
                    "val_roc_auc": None if val_metrics["roc_auc"] is None else round(float(val_metrics["roc_auc"]), 6),
                    "test_f1_class1": round(float(test_metrics["f1_class1"]), 6),
                    "test_pr_auc": round(float(test_metrics["pr_auc"]), 6),
                    "test_precision": round(float(test_metrics["precision"]), 6),
                    "test_recall": round(float(test_metrics["recall"]), 6),
                    "test_balanced_accuracy": round(float(test_metrics["balanced_accuracy"]), 6),
                    "test_roc_auc": None if test_metrics["roc_auc"] is None else round(float(test_metrics["roc_auc"]), 6),
                    "hyperparameters_json": json.dumps(candidate.get("params", {}), sort_keys=True),
                    "trained_at_utc": utc_now_iso(),
                    "_feature_cols": feature_cols,
                    "_estimator": final_estimator,
                    "_test_prob": test_prob,
                    "_test_df": test_df.copy(),
                }
                candidate_rows.append(record)
                if best_candidate is None:
                    best_candidate = record
                else:
                    left = (record["val_f1_class1"], record["val_pr_auc"])
                    right = (best_candidate["val_f1_class1"], best_candidate["val_pr_auc"])
                    if (left[0] > right[0]) or (math.isclose(left[0], right[0], rel_tol=1e-12) and left[1] > right[1]):
                        best_candidate = record

            if best_candidate is None:
                raise RuntimeError(f"All candidate models were skipped for {coin}/{mode}.")

            for row in candidate_rows:
                row["selected_for_production"] = False
            best_candidate["selected_for_production"] = True

            model_path = model_artifact_path(coin, mode)
            feature_path = feature_artifact_path(coin, mode)
            joblib.dump(best_candidate["_estimator"], model_path)
            pd.DataFrame({"feature": best_candidate["_feature_cols"]}).to_csv(feature_path, index=False)

            chosen_test = best_candidate["_test_df"].copy()
            chosen_test["predicted_probability_up"] = best_candidate["_test_prob"].astype(float)
            chosen_test["threshold"] = float(best_candidate["threshold"])
            chosen_test["predicted_class"] = (chosen_test["predicted_probability_up"] >= chosen_test["threshold"]).astype(int)
            chosen_test["predicted_label"] = chosen_test["predicted_class"].map({1: "UP", 0: "DOWN"})
            chosen_test["actual_class"] = chosen_test["target"].astype(int)
            chosen_test["actual_label"] = chosen_test["actual_class"].map({1: "UP", 0: "DOWN"})
            chosen_test["was_correct"] = (chosen_test["predicted_class"] == chosen_test["actual_class"]).astype(int)
            chosen_test["coin"] = coin
            chosen_test["mode"] = mode
            chosen_test["model_family"] = best_candidate["model_family"]
            chosen_test["selected_for_production"] = True
            chosen_test["split"] = "test"
            chosen_test["feature_date"] = chosen_test["date"]
            chosen_test["reference_close"] = chosen_test["close"]
            chosen_test["target_close"] = chosen_test["next_close"]
            chosen_test["generated_at_utc"] = utc_now_iso()
            backtest_keep = [
                "coin", "mode", "model_family", "selected_for_production", "split", "feature_date", "target_date",
                "predicted_probability_up", "predicted_class", "predicted_label", "actual_class", "actual_label",
                "was_correct", "threshold", "reference_close", "target_close", "generated_at_utc"
            ]
            all_backtests.append(chosen_test[backtest_keep].copy())

            selected_payload["selected_models"].setdefault(coin, {})[mode] = {
                "selected_model_family": best_candidate["model_family"],
                "model_artifact": model_path.name,
                "feature_artifact": feature_path.name,
                "model_version": file_version(model_path),
                "feature_version": file_version(feature_path),
                "training_data_date_range": {
                    "start": train_df["date"].min(),
                    "end": val_df["date"].max(),
                },
                "test_date_range": {
                    "start": test_df["date"].min(),
                    "end": test_df["date"].max(),
                },
                "threshold": best_candidate["threshold"],
                "feature_count": len(best_candidate["_feature_cols"]),
                "includes_twitter": mode in {TWITTER_ONLY_MODE, COMBINED_MODE},
                "live_valid": _selected_mode_live_valid(mode),
                "selection_basis": {
                    "rule": "highest validation class-1 F1; PR-AUC tiebreak",
                    "validation_f1_class1": best_candidate["val_f1_class1"],
                    "validation_pr_auc": best_candidate["val_pr_auc"],
                },
                "test_metrics": {
                    "f1_class1": best_candidate["test_f1_class1"],
                    "pr_auc": best_candidate["test_pr_auc"],
                    "precision": best_candidate["test_precision"],
                    "recall": best_candidate["test_recall"],
                    "balanced_accuracy": best_candidate["test_balanced_accuracy"],
                    "roc_auc": best_candidate["test_roc_auc"],
                },
                "source_provenance": {
                    "raw_price_candles_csv": str(RAW_PRICE_CSV.relative_to(BASE_DIR)),
                    "raw_twitter_sentiment_csv": str(RAW_TWITTER_CSV.relative_to(BASE_DIR)) if twitter_history_available() else None,
                    "feature_matrix_csv": str(FEATURE_MATRIX_CSV.relative_to(BASE_DIR)),
                    "model_results_csv": str(MODEL_RESULTS_CSV.relative_to(BASE_DIR)),
                    "backtest_predictions_csv": str(BACKTEST_PREDICTIONS_CSV.relative_to(BASE_DIR)),
                },
                "trained_at_utc": best_candidate["trained_at_utc"],
            }

            for row in candidate_rows:
                row.pop("_feature_cols", None)
                row.pop("_estimator", None)
                row.pop("_test_prob", None)
                row.pop("_test_df", None)
                all_results.append(row)

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        write_csv_atomic(results_df, MODEL_RESULTS_CSV)
    backtests_df = pd.concat(all_backtests, ignore_index=True) if all_backtests else pd.DataFrame(columns=[
        "coin", "mode", "model_family", "selected_for_production", "split", "feature_date", "target_date",
        "predicted_probability_up", "predicted_class", "predicted_label", "actual_class", "actual_label",
        "was_correct", "threshold", "reference_close", "target_close", "generated_at_utc"
    ])
    write_csv_atomic(backtests_df, BACKTEST_PREDICTIONS_CSV)
    export_selected_backtests()
    save_best_model_metadata(selected_payload)
    update_provenance_from_files()

    outcomes: list[TrainingOutcome] = []
    for coin, modes in selected_payload["selected_models"].items():
        for mode, info in modes.items():
            outcomes.append(TrainingOutcome(
                coin=coin,
                mode=mode,
                selected_model_family=info["selected_model_family"],
                model_artifact=info["model_artifact"],
                feature_artifact=info["feature_artifact"],
                threshold=float(info["threshold"]),
                includes_twitter=bool(info["includes_twitter"]),
                live_valid=bool(info["live_valid"]),
            ))
    return outcomes


def export_selected_backtests() -> pd.DataFrame:
    df = safe_read_csv(BACKTEST_PREDICTIONS_CSV)
    if df.empty:
        write_csv_atomic(df, BACKTEST_SELECTED_CSV)
        return df
    selected = df[(df["selected_for_production"].astype(str).str.lower().isin(["true", "1"])) & (df["split"] == "test")].copy()
    write_csv_atomic(selected, BACKTEST_SELECTED_CSV)
    return selected


# ---------------------------------------------------------------------------
# Read artifact helpers used by the app
# ---------------------------------------------------------------------------

def read_model_results(coin: str | None = None, mode: str | None = None) -> pd.DataFrame:
    df = safe_read_csv(MODEL_RESULTS_CSV)
    if df.empty:
        return df
    if coin is not None:
        df = df[df["coin"] == coin].copy()
    if mode is not None:
        df = df[df["mode"] == mode].copy()
    return df.reset_index(drop=True)


def read_backtest_predictions(coin: str | None = None, mode: str | None = None, selected_only: bool = True) -> pd.DataFrame:
    path = BACKTEST_SELECTED_CSV if selected_only else BACKTEST_PREDICTIONS_CSV
    df = safe_read_csv(path)
    if df.empty:
        return df
    if coin is not None:
        df = df[df["coin"] == coin].copy()
    if mode is not None:
        df = df[df["mode"] == mode].copy()
    for col in ["feature_date", "target_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.sort_values(["coin", "mode", "feature_date"]).reset_index(drop=True)


def active_model_record(coin: str, mode: str) -> dict[str, Any] | None:
    meta = load_best_model_metadata()
    return meta.get("selected_models", {}).get(coin, {}).get(mode)


def backtest_rows_for_coin_mode(coin: str, mode: str) -> pd.DataFrame:
    return read_backtest_predictions(coin, mode, selected_only=True)


# ---------------------------------------------------------------------------
# Live scoring support
# ---------------------------------------------------------------------------

def build_live_feature_row(coin: str, mode: str) -> pd.Series | None:
    feature_matrix = build_feature_matrix()
    if feature_matrix.empty:
        return None
    frame = read_feature_matrix(coin, mode)
    if frame.empty:
        return None
    unresolved = frame[frame["target"].isna()].copy()
    if not unresolved.empty:
        return unresolved.sort_values("date").iloc[-1]
    return frame.sort_values("date").iloc[-1]


def update_provenance_from_files() -> None:
    payload = default_provenance_payload()
    price_df = read_price_history()
    if not price_df.empty:
        payload["status"] = "price_history_available"
        payload["price"]["date_range"] = {
            "start": price_df["date"].min(),
            "end": price_df["date"].max(),
        }
        payload["price"]["row_count"] = int(len(price_df))
    twitter_df = read_twitter_daily()
    if not twitter_df.empty:
        payload["twitter"]["date_range"] = {
            "start": twitter_df["date"].dt.strftime("%Y-%m-%d").min(),
            "end": twitter_df["date"].dt.strftime("%Y-%m-%d").max(),
        }
        payload["twitter"]["row_count"] = int(len(twitter_df))
    payload["artifact_versions"] = {
        "raw_price_candles_csv": file_version(RAW_PRICE_CSV),
        "raw_twitter_sentiment_csv": file_version(RAW_TWITTER_CSV),
        "feature_matrix_csv": file_version(FEATURE_MATRIX_CSV),
        "model_results_csv": file_version(MODEL_RESULTS_CSV),
        "backtest_predictions_csv": file_version(BACKTEST_PREDICTIONS_CSV),
        "best_model_metadata_json": file_version(BEST_MODEL_METADATA_PATH),
    }
    write_json_atomic(payload, PROVENANCE_JSON)


bootstrap_placeholder_files()