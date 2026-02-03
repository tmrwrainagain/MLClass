# api_app.py
# =========================================================
# MODULE G / 4.1 — Flask API (UNIVERSAL TEMPLATE)
# ---------------------------------------------------------
# Цель:
#   - НЕ обучать модели, а загрузить уже сохранённые артефакты (joblib)
#   - дать 3 эндпоинта:
#       GET  /health
#       POST /predict        (1 транзакция)
#       POST /predict_batch  (пачка транзакций)
#       GET  /forecast       (прогноз total_volume на N месяцев)
#
# Важно:
#   - Модели 3.1 сохранены как sklearn Pipeline и ожидают DataFrame
#   - Мы аккуратно приводим типы, добавляем отсутствующие колонки дефолтами
#   - Порт/хост настраиваются через ENV или параметры внизу файла
# =========================================================

import os
import json
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from flask import Flask, request, jsonify

# =========================================================
# 0) НАСТРОЙКИ (обычно меняются на соревнованиях)
# =========================================================

# Корень проекта = папка, где лежит этот файл
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка с моделями (по умолчанию: ./models рядом с проектом)
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models"))

# 3.1 — модели классификации
RISK_MODEL_PATH = os.getenv("RISK_MODEL_PATH", os.path.join(MODEL_DIR, "best_model_risk.joblib"))
CX_MODEL_PATH = os.getenv("CX_MODEL_PATH", os.path.join(MODEL_DIR, "best_model_complexity.joblib"))

# 3.3 — артефакты прогноза total_volume
FORECAST_MODEL_PATH = os.getenv("FORECAST_MODEL_PATH", os.path.join(MODEL_DIR, "forecast_total_volume.joblib"))
FORECAST_HISTORY_PATH = os.getenv(
    "FORECAST_HISTORY_PATH", os.path.join(MODEL_DIR, "forecast_total_volume_history.csv")
)
DEFAULT_FORECAST_MONTHS = int(os.getenv("DEFAULT_FORECAST_MONTHS", "6"))

# Сервер
HOST = os.getenv("API_HOST", "127.0.0.1")
PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("API_DEBUG", "1") == "1"

# =========================================================
# 1) APP
# =========================================================

app = Flask(__name__)

# Загруженные артефакты
risk_model = None
cx_model = None
forecast_model = None
forecast_history = None

# =========================================================
# 2) УТИЛИТЫ: загрузка, валидация, подготовка признаков
# =========================================================


def _require_file(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def load_artifacts():
    """Загружаем всё один раз при старте."""
    global risk_model, cx_model, forecast_model, forecast_history

    _require_file(RISK_MODEL_PATH, "Risk model")
    _require_file(CX_MODEL_PATH, "Complexity model")

    risk_model = joblib.load(RISK_MODEL_PATH)
    cx_model = joblib.load(CX_MODEL_PATH)

    # Прогноз может быть не готов, но по заданию 4.1 — желательно
    if os.path.exists(FORECAST_MODEL_PATH) and os.path.exists(FORECAST_HISTORY_PATH):
        forecast_model = joblib.load(FORECAST_MODEL_PATH)
        forecast_history = pd.read_csv(FORECAST_HISTORY_PATH)

        # ожидаем минимум: month,total_volume
        if "month" not in forecast_history.columns or "total_volume" not in forecast_history.columns:
            raise ValueError(
                "forecast_total_volume_history.csv must contain columns: month,total_volume"
            )

        forecast_history["month"] = pd.to_datetime(forecast_history["month"], errors="coerce")
        forecast_history = forecast_history.dropna(subset=["month"]).sort_values("month").reset_index(drop=True)
    else:
        forecast_model = None
        forecast_history = None


def _safe_to_numeric(s: pd.Series, default=0.0):
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(default)


def _parse_hour_from_tr_datetime(x):
    """Пытаемся извлечь hour из tr_datetime.

    Поддержка:
      - формат соревнований: "0 10:23:26" (day_index + time)
      - ISO-строки: "2020-01-01 10:23:26"
      - если не получилось -> NaN
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    try:
        s = str(x)
        # формат: "<dayIndex> HH:MM:SS"
        if " " in s and ":" in s:
            parts = s.split(" ")
            t = parts[-1]
            hh = int(t.split(":")[0])
            if 0 <= hh <= 23:
                return hh
    except Exception:
        pass

    try:
        dt = pd.to_datetime(x, errors="coerce")
        if pd.notna(dt):
            return int(dt.hour)
    except Exception:
        pass

    return np.nan


def _infer_expected_columns():
    """Берём ожидаемые колонки из fitted sklearn Pipeline, если доступны."""
    cols = set()
    for mdl in (risk_model, cx_model):
        if mdl is None:
            continue
        if hasattr(mdl, "feature_names_in_"):
            try:
                cols |= set(list(mdl.feature_names_in_))
            except Exception:
                pass
    return cols


def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Единая функция подготовки признаков для /predict и /predict_batch.

    Логика:
      1) Стандартизируем базовые поля, которые часто нужны
      2) Если модели сохранили feature_names_in_, выравниваем DataFrame под них
      3) Пропуски заполняем безопасными дефолтами

    ВАЖНО: мы НЕ делаем сложный feature engineering здесь.
    Если на соревнованиях нужно — добавишь 2–3 строки внутри этой функции.
    """
    df = df_raw.copy()

    # --- гарантируем наличие некоторых популярных колонок ---
    if "tr_datetime" in df.columns and "hour" not in df.columns:
        df["hour"] = df["tr_datetime"].apply(_parse_hour_from_tr_datetime)

    # числовые поля
    for col, default in [
        ("amount", 0.0),
        ("mcc_code", 0),
        ("tr_type", 0),
        ("hour", 0),
        ("rule_score", 0.0),
        ("anomaly_score", 0.0),
        ("risk_score", 0.0),
        ("customer_id", 0),
        ("term_id", 0),
    ]:
        if col in df.columns:
            df[col] = _safe_to_numeric(df[col], default=default)

    # категориальные поля
    if "flow" in df.columns:
        df["flow"] = df["flow"].astype(str).fillna("unknown")

    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].astype(str)

    if "verification_complexity" in df.columns:
        df["verification_complexity"] = df["verification_complexity"].astype(str)

    # --- Выравнивание под ожидаемые колонки моделей ---
    expected = _infer_expected_columns()

    if expected:
        # добавим отсутствующие
        for c in expected:
            if c not in df.columns:
                # дефолт: числа -> 0, строки -> "unknown"
                if c in {"flow"}:
                    df[c] = "unknown"
                else:
                    df[c] = 0

        # оставим только expected в том порядке, как сохранён в модели
        # (если order не доступен, сортировка — стабильнее, чем произвольный порядок)
        if hasattr(risk_model, "feature_names_in_"):
            order = list(getattr(risk_model, "feature_names_in_"))
        elif hasattr(cx_model, "feature_names_in_"):
            order = list(getattr(cx_model, "feature_names_in_"))
        else:
            order = sorted(list(expected))

        df = df[order]

    # финальная чистка
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def _predict_one(payload: dict):
    """Предсказание для одного объекта."""
    df = pd.DataFrame([payload])
    X = build_features(df)

    risk = str(risk_model.predict(X)[0])
    cx = str(cx_model.predict(X)[0])

    proba_map = None
    if hasattr(risk_model, "predict_proba"):
        try:
            proba = risk_model.predict_proba(X)[0]
            classes = list(getattr(risk_model, "classes_", []))
            if classes:
                proba_map = {str(c): float(p) for c, p in zip(classes, proba)}
        except Exception:
            proba_map = None

    return risk, cx, proba_map


def _predict_batch(rows: list):
    """Предсказание для пачки объектов."""
    df = pd.DataFrame(rows)
    X = build_features(df)

    risk_pred = risk_model.predict(X)
    cx_pred = cx_model.predict(X)

    proba_rows = None
    if hasattr(risk_model, "predict_proba"):
        try:
            proba = risk_model.predict_proba(X)
            classes = list(getattr(risk_model, "classes_", []))
            if classes:
                proba_rows = pd.DataFrame(proba, columns=classes).to_dict(orient="records")
        except Exception:
            proba_rows = None

    out = []
    for i in range(len(X)):
        item = {
            "risk_level": str(risk_pred[i]),
            "verification_complexity": str(cx_pred[i]),
        }
        if proba_rows is not None:
            item["risk_proba"] = {str(k): float(v) for k, v in proba_rows[i].items()}
        out.append(item)

    return out


# =========================================================
# 3) ПРОГНОЗ total_volume (как в 3.3)
# =========================================================


def _time_features_one(month: pd.Timestamp, month_idx: int) -> dict:
    m = int(month.month)
    return {
        "month_idx": month_idx,
        "sin_m": float(np.sin(2 * np.pi * m / 12)),
        "cos_m": float(np.cos(2 * np.pi * m / 12)),
    }


def forecast_total_volume_next_months(months: int = 6) -> pd.DataFrame:
    """Рекурсивный прогноз total_volume на N месяцев вперёд."""
    if forecast_model is None or forecast_history is None:
        raise RuntimeError(
            "Forecast artifacts are missing. "
            "Need forecast_total_volume.joblib and forecast_total_volume_history.csv in models/"
        )

    hist = (
        forecast_history[["month", "total_volume"]]
        .copy()
        .sort_values("month")
        .reset_index(drop=True)
    )

    last_month = hist["month"].max()
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=months, freq="MS")

    preds = []
    tmp = hist.copy()

    for fm in future_months:
        row = _time_features_one(pd.to_datetime(fm), month_idx=len(tmp))

        for L in (1, 2, 3):
            row[f"lag_{L}"] = float(tmp["total_volume"].iloc[-L]) if len(tmp) >= L else np.nan

        row["roll_mean_3"] = float(tmp["total_volume"].iloc[-3:].mean()) if len(tmp) >= 3 else np.nan

        X_new = pd.DataFrame([row]).fillna(0.0)
        y_hat = float(forecast_model.predict(X_new)[0])

        preds.append(y_hat)
        tmp = pd.concat([tmp, pd.DataFrame({"month": [fm], "total_volume": [y_hat]})], ignore_index=True)

    return pd.DataFrame({"month": future_months, "total_volume_forecast": preds})


# =========================================================
# 4) ENDPOINTS
# =========================================================


@app.get("/health")
def health():
    ok_forecast = forecast_model is not None and forecast_history is not None
    return jsonify(
        {
            "status": "ok",
            "risk_model_loaded": risk_model is not None,
            "complexity_model_loaded": cx_model is not None,
            "forecast_ready": ok_forecast,
            "model_dir": MODEL_DIR,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
    )


@app.post("/predict")
def predict():
    """Предсказание для одной транзакции.

    JSON пример (минимум):
      {
        "amount": -2245.92,
        "mcc_code": 4814,
        "tr_type": 1030,
        "flow": "spend",
        "hour": 10
      }

    Можно присылать и расширенный набор (customer_id, tr_datetime, rule_score, ...)
    """
    try:
        payload = request.get_json(force=True) or {}
        if not isinstance(payload, dict) or len(payload) == 0:
            return jsonify({"error": "Expected JSON object with transaction fields"}), 400

        risk, cx, proba_map = _predict_one(payload)
        return jsonify(
            {
                "risk_level": risk,
                "verification_complexity": cx,
                "risk_proba": proba_map,
            }
        )

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.post("/predict_batch")
def predict_batch():
    """Предсказание для пачки транзакций.

    JSON пример:
      {
        "rows": [
          {"amount": -100, "mcc_code": 4814, "tr_type": 1030, "flow": "spend", "hour": 10},
          {"amount":  500, "mcc_code": 6011, "tr_type": 7010, "flow": "income", "hour": 12}
        ]
      }

    Возвращает список результатов в том же порядке.
    """
    try:
        payload = request.get_json(force=True) or {}
        rows = payload.get("rows")

        if not isinstance(rows, list) or len(rows) == 0:
            return jsonify({"error": "Expected JSON: { rows: [ {...}, ... ] }"}), 400

        result = _predict_batch(rows)
        return jsonify({"count": len(result), "result": result})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.get("/forecast")
def forecast():
    """Прогноз total_volume на N месяцев вперёд.

    Пример:
      GET /forecast?months=6
    """
    try:
        months = int(request.args.get("months", DEFAULT_FORECAST_MONTHS))
        months = max(1, min(months, 24))  # защита 1..24

        fc = forecast_total_volume_next_months(months=months)

        return jsonify(
            {
                "months": months,
                "result": [
                    {
                        "month": pd.to_datetime(d).strftime("%Y-%m-%d"),
                        "total_volume_forecast": float(v),
                    }
                    for d, v in zip(fc["month"], fc["total_volume_forecast"])
                ],
            }
        )

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# =========================================================
# 5) RUN
# =========================================================

if __name__ == "__main__":
    load_artifacts()

    # Подсказка для запуска:
    #   API_PORT=8000 API_HOST=127.0.0.1 python api_app.py
    #   API_PORT=8080 API_DEBUG=0 python api_app.py

    app.run(host=HOST, port=PORT, debug=DEBUG)

    # зависимости:
    #   pip install flask joblib pandas numpy scikit-learn
