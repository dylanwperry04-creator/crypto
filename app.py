from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

from real_data_pipeline import (
    BACKTEST_SELECTED_CSV,
    BEST_MODEL_METADATA_PATH,
    COINS,
    DB_PATH,
    FEATURE_MATRIX_CSV,
    FORECAST_HORIZON_LABEL,
    MODE_LABELS,
    PRICE_ONLY_MODE,
    PRIMARY_MARKET_SOURCE,
    PROVENANCE_JSON,
    RAW_PRICE_CSV,
    TIMEZONE_POLICY,
    TWITTER_LIVE_AVAILABLE,
    active_model_record,
    backtest_rows_for_coin_mode,
    bootstrap_placeholder_files,
    build_live_feature_row,
    export_selected_backtests,
    feature_artifact_path,
    fetch_live_ticker,
    file_version,
    get_db,
    incremental_update_coin,
    init_db,
    load_best_model_metadata,
    model_artifact_path,
    read_backtest_predictions,
    read_model_results,
    read_price_history,
    update_provenance_from_files,
)

APP_TITLE = "AI Crypto Forecasting Agent"
BASE_DIR = Path(__file__).resolve().parent
BITCOIN_MODES = ["price_only", "twitter_only", "combined"]
ROLLING_WINDOW = 30

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=str(BASE_DIR))
_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def utc_now() -> datetime:
    return datetime.now(timezone.utc)



def coin_modes(coin: str) -> list[str]:
    return BITCOIN_MODES if coin == "Bitcoin" else [PRICE_ONLY_MODE]



def load_model(path: Path):
    key = str(path.resolve())
    stat = None
    if path.exists():
        st = path.stat()
        stat = (st.st_mtime_ns, st.st_size)
    cached = _CACHE.get(key)
    if cached and cached["stat"] == stat:
        return cached["value"]
    if not path.exists():
        _CACHE[key] = {"stat": stat, "value": None}
        return None
    try:
        value = joblib.load(path)
    except Exception as exc:
        logger.warning("Failed to load model %s: %s", path.name, exc)
        value = None
    _CACHE[key] = {"stat": stat, "value": value}
    return value



def load_feature_list(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Failed to load feature file %s: %s", path.name, exc)
        return []
    return [str(x) for x in frame.get("feature", pd.Series(dtype=str)).dropna().tolist()]



def boolish(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}



def mode_status(coin: str, mode: str) -> dict[str, Any]:
    meta = load_best_model_metadata()
    selected = active_model_record(coin, mode)
    model_path = model_artifact_path(coin, mode)
    feature_path = feature_artifact_path(coin, mode)
    model = load_model(model_path)
    features = load_feature_list(feature_path)
    backtest_df = backtest_rows_for_coin_mode(coin, mode)
    price_df = read_price_history(coin)

    disabled_reason = None
    if coin == "Bitcoin" and mode in {"twitter_only", "combined"} and not selected:
        disabled_reason = "Historical Twitter-dependent modes require a real raw_twitter_sentiment.csv and an offline retrain; live Twitter is disabled unless a real ongoing source is configured."
    elif model is None or not model_path.exists():
        disabled_reason = f"Selected model artifact missing at {model_path.relative_to(BASE_DIR)}. Run the offline pipeline first."
    elif not features:
        disabled_reason = f"Feature list missing at {feature_path.relative_to(BASE_DIR)}. Run the offline pipeline first."
    elif price_df.empty:
        disabled_reason = "No real local OHLCV CSV is stored yet. Run the backfill script first."
    elif backtest_df.empty:
        disabled_reason = "No selected historical backtest rows are available yet. Run the offline training pipeline first."

    live_valid = bool(selected and selected.get("live_valid", False) and model is not None and len(price_df) >= 60)
    if mode in {"twitter_only", "combined"} and not TWITTER_LIVE_AVAILABLE:
        live_valid = False
        if not disabled_reason:
            disabled_reason = "Historical-only mode: live Twitter ingestion is not configured, so this mode is disabled for live scoring."

    return {
        "coin": coin,
        "mode": mode,
        "label": MODE_LABELS[mode],
        "backtest_available": bool(selected and model is not None and features and not backtest_df.empty),
        "live_available": live_valid,
        "disabled_reason": disabled_reason,
        "date_min": None if backtest_df.empty else backtest_df["feature_date"].min().strftime("%Y-%m-%d"),
        "date_max": None if backtest_df.empty else backtest_df["feature_date"].max().strftime("%Y-%m-%d"),
        "model": {
            "artifact": model_path.name if model_path.exists() else None,
            "version": file_version(model_path),
            "pipeline": selected.get("selected_model_family") if selected else None,
        },
        "features": {
            "artifact": feature_path.name if feature_path.exists() else None,
            "version": file_version(feature_path),
            "count": len(features),
            "columns": features,
        },
        "metadata": {
            "artifact": BEST_MODEL_METADATA_PATH.name if BEST_MODEL_METADATA_PATH.exists() else None,
            "version": file_version(BEST_MODEL_METADATA_PATH),
            "threshold": None if not selected else selected.get("threshold"),
        },
        "evaluation_source": {
            "artifact": BACKTEST_SELECTED_CSV.name if BACKTEST_SELECTED_CSV.exists() else None,
            "version": file_version(BACKTEST_SELECTED_CSV),
            "rows": int(len(backtest_df)),
        },
    }



def all_mode_status() -> dict[str, list[dict[str, Any]]]:
    return {coin: [mode_status(coin, mode) for mode in coin_modes(coin)] for coin in COINS}



def validate_mode_request(coin: str, mode: str, for_live: bool, date_str: str | None = None) -> dict[str, Any]:
    if coin not in COINS:
        return {"valid": False, "reason": f"coin must be one of {COINS}"}
    if mode not in coin_modes(coin):
        return {"valid": False, "reason": f"mode '{mode}' is not available for {coin}."}
    status = mode_status(coin, mode)
    if for_live and not status["live_available"]:
        return {"valid": False, "reason": status["disabled_reason"] or "Live forecasting is unavailable."}
    if not for_live and not status["backtest_available"]:
        return {"valid": False, "reason": status["disabled_reason"] or "Historical evaluation is unavailable."}
    if not for_live and date_str and status["date_min"] and status["date_max"] and not (status["date_min"] <= date_str <= status["date_max"]):
        return {"valid": False, "reason": f"Historical evaluation for {coin}/{mode} is available only from {status['date_min']} to {status['date_max']}."}
    return {"valid": True, "reason": ""}



def build_feature_frame(row: pd.Series, features: list[str]) -> pd.DataFrame:
    payload = {feature: pd.to_numeric(row.get(feature, 0), errors="coerce") for feature in features}
    return pd.DataFrame([payload])[features].fillna(0.0)



def score_model(model, features: list[str], row: pd.Series, threshold: float) -> tuple[float, int, str]:
    X = build_feature_frame(row, features)
    prob = float(model.predict_proba(X)[:, 1][0])
    predicted_class = int(prob >= threshold)
    return prob, predicted_class, "UP" if predicted_class == 1 else "DOWN"


# ---------------------------------------------------------------------------
# Historical performance payloads
# ---------------------------------------------------------------------------

def performance_summary_for_mode(coin: str, mode: str) -> dict[str, Any]:
    results = read_model_results(coin, mode)
    frame = backtest_rows_for_coin_mode(coin, mode)
    status = mode_status(coin, mode)
    selected = results[(results.get("selected_for_production", False).astype(str).str.lower().isin(["true", "1"])) & (results["status"] == "trained")].copy() if not results.empty else pd.DataFrame()
    if selected.empty or frame.empty:
        return {
            "coin": coin,
            "mode": mode,
            "label": MODE_LABELS[mode],
            "available": False,
            "reason": status["disabled_reason"],
        }
    row = selected.iloc[0]
    actual = frame["actual_class"].astype(int)
    predicted = frame["predicted_class"].astype(int)
    rolling = []
    for idx in range(len(frame)):
        window = frame.iloc[max(0, idx - ROLLING_WINDOW + 1): idx + 1]
        y_true = window["actual_class"].astype(int)
        y_pred = window["predicted_class"].astype(int)
        rolling.append({
            "date": frame.iloc[idx]["feature_date"].strftime("%Y-%m-%d"),
            "rolling_f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "rolling_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        })
    return {
        "coin": coin,
        "mode": mode,
        "label": MODE_LABELS[mode],
        "available": True,
        "pipeline": row["model_family"],
        "threshold": round(float(row["threshold"]), 6),
        "f1": round(float(row["test_f1_class1"]), 4),
        "accuracy": round(float((frame["was_correct"].astype(int)).mean()), 4),
        "pr_auc": None if pd.isna(row["test_pr_auc"]) else round(float(row["test_pr_auc"]), 4),
        "precision": None if pd.isna(row["test_precision"]) else round(float(row["test_precision"]), 4),
        "recall": None if pd.isna(row["test_recall"]) else round(float(row["test_recall"]), 4),
        "balanced_accuracy": None if pd.isna(row["test_balanced_accuracy"]) else round(float(row["test_balanced_accuracy"]), 4),
        "roc_auc": None if pd.isna(row["test_roc_auc"]) else round(float(row["test_roc_auc"]), 4),
        "rows": int(len(frame)),
        "history": frame[["feature_date", "predicted_probability_up", "actual_class", "predicted_class", "was_correct"]].assign(
            feature_date=lambda d: d["feature_date"].dt.strftime("%Y-%m-%d")
        ).rename(columns={"feature_date": "date", "predicted_probability_up": "probability"}).to_dict(orient="records"),
        "rolling": rolling,
    }



def performance_payload_for_coin(coin: str) -> dict[str, Any]:
    return {
        "coin": coin,
        "metric_source": {
            "kind": "saved_offline_artifacts",
            "metadata_file": BEST_MODEL_METADATA_PATH.name if BEST_MODEL_METADATA_PATH.exists() else None,
            "metadata_version": file_version(BEST_MODEL_METADATA_PATH),
            "evaluation_sources": [
                {
                    "mode": mode,
                    "label": MODE_LABELS[mode],
                    "evaluation_file": BACKTEST_SELECTED_CSV.name if BACKTEST_SELECTED_CSV.exists() else None,
                    "evaluation_version": file_version(BACKTEST_SELECTED_CSV),
                }
                for mode in coin_modes(coin)
            ],
            "generated_at": utc_now().isoformat(),
        },
        "modes": [performance_summary_for_mode(coin, mode) for mode in coin_modes(coin)],
    }



def backtest_lookup(coin: str, mode: str, date_str: str) -> Optional[dict[str, Any]]:
    validation = validate_mode_request(coin, mode, for_live=False, date_str=date_str)
    if not validation["valid"]:
        return {"error": validation["reason"]}
    frame = backtest_rows_for_coin_mode(coin, mode)
    if frame.empty:
        return None
    row = frame[frame["feature_date"].dt.strftime("%Y-%m-%d") == date_str]
    if row.empty:
        return None
    record = row.iloc[0]
    selected = active_model_record(coin, mode)
    return {
        "context": "historical_evaluation",
        "coin": coin,
        "mode": mode,
        "mode_label": MODE_LABELS[mode],
        "forecast_timestamp": f"{date_str}T00:00:00+00:00",
        "feature_date": date_str,
        "target_date": record["target_date"].strftime("%Y-%m-%d"),
        "forecast_horizon": FORECAST_HORIZON_LABEL,
        "status": "Resolved",
        "probability": round(float(record["predicted_probability_up"]), 4),
        "threshold": round(float(record["threshold"]), 6),
        "predicted_class": int(record["predicted_class"]),
        "predicted_label": str(record["predicted_label"]),
        "actual_class": int(record["actual_class"]),
        "actual_label": str(record["actual_label"]),
        "result": "Correct" if int(record["was_correct"]) == 1 else "Wrong",
        "close_price": round(float(record["reference_close"]), 6),
        "pipeline": selected.get("selected_model_family") if selected else None,
        "audit": {
            "model_artifact": None if not selected else selected.get("model_artifact"),
            "model_version": file_version(model_artifact_path(coin, mode)),
            "feature_artifact": None if not selected else selected.get("feature_artifact"),
            "feature_version": file_version(feature_artifact_path(coin, mode)),
            "metadata_version": file_version(BEST_MODEL_METADATA_PATH),
            "evaluation_source": {"file": BACKTEST_SELECTED_CSV.name, "sha256_12": file_version(BACKTEST_SELECTED_CSV)["sha256_12"] if file_version(BACKTEST_SELECTED_CSV) else None},
        },
    }


# ---------------------------------------------------------------------------
# Live forecasts
# ---------------------------------------------------------------------------

def identity_key_for_forecast(coin: str, mode: str, feature_date: str, target_date: str, model_version: str, feature_version: str) -> str:
    return "|".join([coin, mode, FORECAST_HORIZON_LABEL, feature_date, target_date, model_version, feature_version])



def make_forecast_id(identity_key: str) -> str:
    return hashlib.sha256(identity_key.encode("utf-8")).hexdigest()[:24]



def row_to_forecast_dict(row) -> dict[str, Any]:
    actual_class = row["actual_class"]
    predicted_class = row["predicted_class"]
    result = None
    if actual_class is not None and predicted_class is not None:
        result = "Correct" if int(actual_class) == int(predicted_class) else "Wrong"

    feature_payload = None
    raw_feature_payload = row["feature_payload_json"]
    if raw_feature_payload:
        try:
            feature_payload = json.loads(raw_feature_payload)
        except Exception:
            feature_payload = None

    return {
        "forecast_id": row["forecast_id"],
        "forecast_context": row["forecast_context"],
        "coin": row["coin"],
        "mode": row["mode"],
        "mode_label": row["mode_label"],
        "status": row["status"],
        "forecast_timestamp": row["forecast_timestamp_utc"],
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
        "feature_date": row["feature_date"],
        "target_date": row["target_date"],
        "forecast_horizon": row["forecast_horizon"],
        "timezone_policy": row["timezone_policy"],
        "exchange_source": row["exchange_source"],
        "pipeline": row["pipeline_name"],
        "threshold": row["threshold"],
        "probability": row["probability"],
        "predicted_class": predicted_class,
        "predicted_label": row["predicted_label"],
        "actual_class": actual_class,
        "actual_label": row["actual_label"],
        "result": result,
        "reference_close": row["reference_close"],
        "target_close": row["target_close"],
        "live_spot_price": row["live_spot_price"],
        "model_artifact": row["model_artifact"],
        "model_version": row["model_version"],
        "feature_artifact": row["feature_artifact"],
        "feature_version": row["feature_version"],
        "feature_payload": feature_payload,
        "notes": row["notes"],
    }



def create_or_get_live_forecast(coin: str, mode: str) -> dict[str, Any]:
    validation = validate_mode_request(coin, mode, for_live=True)
    if not validation["valid"]:
        return {"error": validation["reason"]}

    selected = active_model_record(coin, mode)
    model_path = model_artifact_path(coin, mode)
    feature_path = feature_artifact_path(coin, mode)
    model = load_model(model_path)
    features = load_feature_list(feature_path)
    if selected is None or model is None or not features:
        return {"error": f"Missing active model artifacts for {coin}/{mode}. Run the offline pipeline first."}

    try:
        incremental_update_coin(coin)
        update_provenance_from_files()
    except Exception as exc:
        return {"error": f"Live market sync failed: {exc}"}

    row = build_live_feature_row(coin, mode)
    if row is None:
        return {"error": "Unable to build the latest live feature row from local real history."}

    feature_date = str(row["date"])
    target_date = str(row["target_date"]) if pd.notna(row.get("target_date")) else (pd.Timestamp(feature_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    threshold = float(selected.get("threshold", 0.5))
    probability, predicted_class, predicted_label = score_model(model, features, row, threshold)
    model_version = (selected.get("model_version") or {}).get("sha256_12") if isinstance(selected.get("model_version"), dict) else None
    feature_version = (selected.get("feature_version") or {}).get("sha256_12") if isinstance(selected.get("feature_version"), dict) else None
    model_version = model_version or (file_version(model_path) or {}).get("sha256_12")
    feature_version = feature_version or (file_version(feature_path) or {}).get("sha256_12")
    identity_key = identity_key_for_forecast(coin, mode, feature_date, target_date, model_version or "na", feature_version or "na")
    forecast_id = make_forecast_id(identity_key)

    with get_db() as conn:
        existing = conn.execute("SELECT * FROM forecasts WHERE identity_key = ?", (identity_key,)).fetchone()
        if existing is not None:
            return row_to_forecast_dict(existing)

        try:
            live_ticker = fetch_live_ticker(coin)
            live_spot_price = float(live_ticker["price"])
        except Exception:
            live_spot_price = None

        payload = {
            feature: None if pd.isna(row.get(feature)) else float(pd.to_numeric(row.get(feature), errors="coerce"))
            for feature in features
        }

        conn.execute(
            """
            INSERT INTO forecasts (
                forecast_id, identity_key, forecast_context, coin, mode, mode_label, status, created_at,
                forecast_timestamp_utc, feature_date, target_date, forecast_horizon, timezone_policy,
                exchange_source, model_artifact, model_version, feature_artifact, feature_version,
                metadata_artifact, metadata_version, pipeline_name, threshold, probability, predicted_class,
                predicted_label, reference_close, live_spot_price, feature_payload_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                forecast_id,
                identity_key,
                "live_forecast",
                coin,
                mode,
                MODE_LABELS[mode],
                "Pending Outcome",
                utc_now().isoformat(),
                utc_now().isoformat(),
                feature_date,
                target_date,
                FORECAST_HORIZON_LABEL,
                TIMEZONE_POLICY,
                PRIMARY_MARKET_SOURCE,
                model_path.name,
                model_version,
                feature_path.name,
                feature_version,
                BEST_MODEL_METADATA_PATH.name,
                (file_version(BEST_MODEL_METADATA_PATH) or {}).get("sha256_12"),
                selected.get("selected_model_family"),
                threshold,
                probability,
                predicted_class,
                predicted_label,
                float(row.get("close", 0.0)),
                live_spot_price,
                json.dumps(payload),
                "Live forecast scored from the latest completed real daily candle after an incremental market-data sync.",
            ),
        )
        conn.commit()
        created = conn.execute("SELECT * FROM forecasts WHERE forecast_id = ?", (forecast_id,)).fetchone()
        return row_to_forecast_dict(created)



def forecast_log_rows() -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM forecasts ORDER BY created_at DESC").fetchall()
    return [row_to_forecast_dict(row) for row in rows]



def forecast_log_summary() -> dict[str, Any]:
    rows = forecast_log_rows()
    pending = [r for r in rows if r["status"] == "Pending Outcome"]
    resolved = [r for r in rows if r["status"] == "Resolved"]
    metrics = None
    if resolved:
        actual = [int(r["actual_class"]) for r in resolved if r["actual_class"] is not None]
        predicted = [int(r["predicted_class"]) for r in resolved if r["predicted_class"] is not None]
        if actual and predicted:
            metrics = {
                "resolved_rows": len(actual),
                "accuracy": round(float(accuracy_score(actual, predicted)), 4),
                "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
            }
    return {
        "storage": {"kind": "sqlite", "path": str(DB_PATH.relative_to(BASE_DIR))},
        "total": len(rows),
        "pending_count": len(pending),
        "resolved_count": len(resolved),
        "metrics_resolved_only": metrics,
        "pending": pending,
        "resolved": resolved,
        "recent": rows[:25],
        "resolution_policy": {
            "horizon": FORECAST_HORIZON_LABEL,
            "timezone": TIMEZONE_POLICY,
            "rule": f"A forecast resolves only after the next completed UTC daily candle exists in {RAW_PRICE_CSV.name} following a sync from {PRIMARY_MARKET_SOURCE}.",
        },
    }



def resolve_pending_live_forecasts() -> dict[str, Any]:
    checked_count = 0
    resolved_count = 0
    errors: list[str] = []
    history = read_price_history()
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM forecasts WHERE status = 'Pending Outcome' ORDER BY created_at ASC").fetchall()
        for row in rows:
            checked_count += 1
            try:
                coin_df = history[history["coin"] == row["coin"]].copy()
                if coin_df.empty:
                    continue
                target_rows = coin_df[coin_df["date"] == row["target_date"]]
                if target_rows.empty:
                    continue
                target_close = float(target_rows.iloc[0]["close"])
                reference_close = float(row["reference_close"])
                actual_class = int(target_close > reference_close)
                actual_label = "UP" if actual_class == 1 else "DOWN"
                conn.execute(
                    "UPDATE forecasts SET status = ?, resolved_at = ?, actual_class = ?, actual_label = ?, target_close = ? WHERE forecast_id = ?",
                    ("Resolved", utc_now().isoformat(), actual_class, actual_label, target_close, row["forecast_id"]),
                )
                resolved_count += 1
            except Exception as exc:
                errors.append(f"{row['coin']}: {exc}")
        conn.commit()
    return {
        "checked_count": checked_count,
        "resolved_count": resolved_count,
        "remaining_pending": forecast_log_summary()["pending_count"],
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Market series payload
# ---------------------------------------------------------------------------

def market_series_payload(coin: str) -> dict[str, Any]:
    if coin not in COINS:
        return {"error": f"coin must be one of {COINS}"}
    points_df = read_price_history(coin)
    if points_df.empty:
        return {
            "coin": coin,
            "source": str(RAW_PRICE_CSV.relative_to(BASE_DIR)),
            "live_market": False,
            "fallback_used": False,
            "unavailable_reason": "No real local OHLCV history is stored yet. Run the backfill script first.",
            "points": [],
            "latest": None,
        }
    # Return enough daily history for the 1Y range selector to differ from 6M when the data exists.
    points_df = points_df.sort_values("candle_open_time_utc").tail(400).copy()
    latest = points_df.iloc[-1]
    return {
        "coin": coin,
        "source": str(RAW_PRICE_CSV.relative_to(BASE_DIR)),
        "live_market": True,
        "fallback_used": False,
        "unavailable_reason": None,
        "points": [
            {
                "date": pd.Timestamp(row["candle_open_time_utc"]).strftime("%Y-%m-%d"),
                "open": round(float(row["open"]), 8),
                "high": round(float(row["high"]), 8),
                "low": round(float(row["low"]), 8),
                "close": round(float(row["close"]), 8),
                "volume": round(float(row["volume"]), 8),
            }
            for _, row in points_df.iterrows()
        ],
        "latest": {
            "date": pd.Timestamp(latest["candle_open_time_utc"]).strftime("%Y-%m-%d"),
            "open": round(float(latest["open"]), 8),
            "high": round(float(latest["high"]), 8),
            "low": round(float(latest["low"]), 8),
            "close": round(float(latest["close"]), 8),
            "volume": round(float(latest["volume"]), 8),
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

bootstrap_placeholder_files()
init_db()
export_selected_backtests()
update_provenance_from_files()


@app.get("/")
def home():
    return render_template("template.html", app_title=APP_TITLE)


@app.get("/app_state")
def app_state():
    provenance = json.loads(PROVENANCE_JSON.read_text(encoding="utf-8")) if PROVENANCE_JSON.exists() else {}
    return jsonify(
        {
            "title": APP_TITLE,
            "config": {
                "PRIMARY_LIVE_SOURCE": PRIMARY_MARKET_SOURCE,
                "TIMEZONE_POLICY": TIMEZONE_POLICY,
                "FORECAST_HORIZON_LABEL": FORECAST_HORIZON_LABEL,
                "DATA_STORE": str(DB_PATH.relative_to(BASE_DIR)),
                "HISTORICAL_STORE": str(RAW_PRICE_CSV.relative_to(BASE_DIR)),
                "FEATURE_MATRIX_STORE": str(FEATURE_MATRIX_CSV.relative_to(BASE_DIR)),
            },
            "coins": COINS,
            "modes": all_mode_status(),
            "forecast_log": forecast_log_summary(),
            "provenance": provenance,
            "resolution_policy": {
                "horizon": FORECAST_HORIZON_LABEL,
                "timezone": TIMEZONE_POLICY,
                "rule": f"Live forecasts are created from the latest completed real daily candle stored locally in {RAW_PRICE_CSV.name} after syncing from {PRIMARY_MARKET_SOURCE}, and remain pending until the next completed candle exists.",
            },
            "legacy_storage": {"deprecated_demo_path": "deprecated_demo/", "in_use": False},
        }
    )


@app.get("/market_series")
def market_series_api():
    coin = request.args.get("coin", "Bitcoin")
    payload = market_series_payload(coin)
    return jsonify(payload), (400 if payload.get("error") else 200)


@app.get("/performance_data")
def performance_data_api():
    coin = request.args.get("coin", "Bitcoin")
    if coin not in COINS:
        return jsonify({"error": f"coin must be one of {COINS}"}), 400
    return jsonify(performance_payload_for_coin(coin))


@app.get("/backtest_lookup")
def backtest_lookup_api():
    coin = request.args.get("coin", "Bitcoin")
    mode = request.args.get("mode", PRICE_ONLY_MODE)
    date_str = request.args.get("date")
    if not date_str:
        return jsonify({"error": "Provide ?date=YYYY-MM-DD"}), 400
    result = backtest_lookup(coin, mode, date_str)
    if result is None:
        return jsonify({"error": f"No historical evaluation row is available for {coin}/{mode} on {date_str}"}), 404
    if result.get("error"):
        return jsonify(result), 400
    return jsonify(result)


@app.route("/predict", methods=["GET", "POST"])
def predict_api():
    args = request.args if request.method == "GET" else (request.get_json(silent=True) or {})
    coin = args.get("coin", "Bitcoin")
    mode = args.get("mode", PRICE_ONLY_MODE)
    live = boolish(args.get("live", "true"))
    if live:
        result = create_or_get_live_forecast(coin, mode)
        return jsonify(result), (400 if result.get("error") else 200)
    date_str = args.get("date")
    if not date_str:
        return jsonify({"error": "Provide ?date=YYYY-MM-DD for historical evaluation lookups."}), 400
    result = backtest_lookup(coin, mode, date_str)
    if result is None:
        return jsonify({"error": f"No historical evaluation row is available for {coin}/{mode} on {date_str}"}), 404
    if result.get("error"):
        return jsonify(result), 400
    return jsonify(result)


@app.get("/prediction_log")
def prediction_log_api():
    return jsonify(forecast_log_summary())


@app.route("/resolve_predictions", methods=["GET", "POST"])
def resolve_predictions_api():
    return jsonify(resolve_pending_live_forecasts())


@app.get("/validate_mode")
def validate_mode_api():
    coin = request.args.get("coin", "Bitcoin")
    mode = request.args.get("mode", PRICE_ONLY_MODE)
    live = boolish(request.args.get("live", "false"))
    date_str = request.args.get("date")
    return jsonify(validate_mode_request(coin, mode, for_live=live, date_str=date_str))


@app.get("/live_price")
def live_price_api():
    coin = request.args.get("coin", "Bitcoin")
    if coin not in COINS:
        return jsonify({"error": f"coin must be one of {COINS}"}), 400
    try:
        return jsonify(fetch_live_ticker(coin))
    except Exception:
        return jsonify({"error": f"Live price is currently unavailable from {PRIMARY_MARKET_SOURCE}."}), 503


@app.get("/coins")
def list_coins_api():
    return jsonify(all_mode_status())


@app.get("/health")
def health_api():
    return jsonify(
        {
            "status": "ok",
            "title": APP_TITLE,
            "storage": {
                "forecast_db": {"path": str(DB_PATH.relative_to(BASE_DIR)), "exists": DB_PATH.exists()},
                "raw_price_csv": {"path": str(RAW_PRICE_CSV.relative_to(BASE_DIR)), "exists": RAW_PRICE_CSV.exists()},
                "feature_matrix_csv": {"path": str(FEATURE_MATRIX_CSV.relative_to(BASE_DIR)), "exists": FEATURE_MATRIX_CSV.exists()},
            },
            "models": {coin: {mode["mode"]: {"backtest_available": mode["backtest_available"], "live_available": mode["live_available"]} for mode in modes} for coin, modes in all_mode_status().items()},
            "provider": PRIMARY_MARKET_SOURCE,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
