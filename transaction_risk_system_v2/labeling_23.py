# ============================================
# MODULE B / 2.3 — Labeling (Risk + Complexity)
# ============================================

import os
import sqlite3
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==============================
# 0) МЕНЯТЬ НА СОРЕВНОВАНИИ
# ==============================
DB_PATH = "db/app.db"              # путь к БД
EVENTS_TABLE = "transactions"      # таблица транзакций

# колонки (переименуй под своё соревнование)
ID_COL = "customer_id"
DT_COL = "tr_datetime"
MCC_COL = "mcc_code"
TYPE_COL = "tr_type"
AMOUNT_COL = "amount"

# если есть уникальный id транзакции — укажи
TX_ID_COL = None  # например "transaction_id" или оставь None

# Ограничение (если БД огромная) — можно None
MAX_ROWS = 2_000_000

# ---------- Бизнес-пороги ----------
X_MED = 50_000       # сумма “подозрительная”
X_HIGH = 150_000     # сумма “очень подозрительная”
Y_FREQ = 200         # частота транзакций по клиенту (за весь период или окно)

RISK_MCC_LIST = {6011, 4829, 5541}   # <-- заменить под задачу (пример)

# ---------- ML параметры ----------
IF_CONTAMINATION = 0.02  # доля аномалий (подбирать 0.01..0.05)
IF_N_ESTIMATORS = 200
RANDOM_STATE = 42

# ---------- Маппинг score -> уровень риска ----------
LOW_THR = 35
HIGH_THR = 70

# ==============================
# 1) Загрузка данных
# ==============================
def get_conn(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")
    return sqlite3.connect(db_path, check_same_thread=False)

def parse_hour(dt_series: pd.Series) -> pd.Series:
    s = dt_series.astype(str)
    parts = s.str.split(" ", n=1, expand=True)
    t = parts[1] if parts.shape[1] == 2 else s
    hour = t.str.split(":", n=1, expand=True)[0]
    return pd.to_numeric(hour, errors="coerce").fillna(-1).astype(int)

con = get_conn(DB_PATH)
limit_sql = f"LIMIT {MAX_ROWS}" if isinstance(MAX_ROWS, int) and MAX_ROWS > 0 else ""

# ВАЖНО: если нет term_id / других полей — убирай
sql = f"""
SELECT
  {ID_COL} AS customer_id,
  {DT_COL} AS tr_datetime,
  {MCC_COL} AS mcc_code,
  {TYPE_COL} AS tr_type,
  {AMOUNT_COL} AS amount
  {',' + TX_ID_COL + ' AS tx_id' if TX_ID_COL else ''}
FROM {EVENTS_TABLE}
{limit_sql}
"""

df = pd.read_sql(sql, con)

# ==============================
# 2) Feature engineering
# ==============================
df["hour"] = parse_hour(df["tr_datetime"])
df["amount_abs"] = df["amount"].abs()
df["flow"] = np.where(df["amount"] < 0, "spend", np.where(df["amount"] > 0, "income", "zero"))
df["is_night"] = ((df["hour"] >= 0) & (df["hour"] <= 5)).astype(int)

# поведение клиента (простое, по всему периоду)
cust_agg = (
    df.groupby("customer_id")
      .agg(
          cust_tx_cnt=("amount", "size"),
          cust_amount_mean=("amount_abs", "mean"),
          cust_amount_std=("amount_abs", "std"),
          cust_amount_sum=("amount_abs", "sum"),
          cust_mcc_nunique=("mcc_code", "nunique"),
      )
      .reset_index()
)

df = df.merge(cust_agg, on="customer_id", how="left")
df["cust_amount_std"] = df["cust_amount_std"].fillna(0.0)

# ==============================
# 3) Бизнес-правила -> rule_score
# ==============================
rule_score = np.zeros(len(df), dtype=float)

# 3.1 сумма
rule_score += np.where(df["amount_abs"] > X_MED, 25, 0)
rule_score += np.where(df["amount_abs"] > X_HIGH, 35, 0)

# 3.2 частота клиента
rule_score += np.where(df["cust_tx_cnt"] > Y_FREQ, 20, 0)

# 3.3 рискованные MCC
rule_score += np.where(df["mcc_code"].isin(RISK_MCC_LIST), 25, 0)

# 3.4 ночная активность + крупная сумма
rule_score += np.where((df["is_night"] == 1) & (df["amount_abs"] > X_MED), 15, 0)

df["rule_score"] = rule_score.clip(0, 100)

# ==============================
# 4) ML-анализ аномалий -> anomaly_score
# ==============================
# ВАЖНО: используем устойчивые фичи без one-hot огромного MCC
ml_features = [
    "amount_abs",
    "hour",
    "cust_tx_cnt",
    "cust_amount_mean",
    "cust_amount_std",
    "cust_mcc_nunique",
    "is_night",
]

X = df[ml_features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(
    n_estimators=IF_N_ESTIMATORS,
    contamination=IF_CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
iso.fit(X_scaled)

# decision_function: чем меньше, тем более аномально
raw = iso.decision_function(X_scaled)
# нормируем в 0..100 (где 100 = самый подозрительный)
anomaly_score = (raw.max() - raw) / (raw.max() - raw.min() + 1e-9) * 100
df["anomaly_score"] = anomaly_score.clip(0, 100)

# ==============================
# 5) Итоговый risk_score -> risk_level
# ==============================
# веса можно менять
df["risk_score"] = (0.6 * df["rule_score"] + 0.4 * df["anomaly_score"]).clip(0, 100)

df["risk_level"] = np.where(
    df["risk_score"] >= HIGH_THR, "high",
    np.where(df["risk_score"] >= LOW_THR, "medium", "low")
)

# ==============================
# 6) Сложность верификации
# ==============================
# Идея:
# - simple: риск объясняется явным правилом (сумма, MCC)
# - medium: частота/поведенческий паттерн
# - hard: в основном ML-аналогия без явных правил

has_big_amount = (df["amount_abs"] > X_HIGH) | (df["amount_abs"] > X_MED)
has_risk_mcc = df["mcc_code"].isin(RISK_MCC_LIST)
has_high_freq = (df["cust_tx_cnt"] > Y_FREQ)
ml_only = (df["rule_score"] < 15) & (df["anomaly_score"] > 60)

df["verification_complexity"] = np.where(
    (has_big_amount | has_risk_mcc) & (df["risk_level"] != "low"), "simple",
    np.where(has_high_freq & (df["risk_level"] != "low"), "medium",
    np.where(ml_only, "hard", "medium"))
)

# ==============================
# 7) Сохранение результата (таблица в БД)
# ==============================
out_cols = [
    "customer_id", "tr_datetime", "mcc_code", "tr_type", "amount",
    "hour", "flow",
    "rule_score", "anomaly_score", "risk_score",
    "risk_level", "verification_complexity"
]
out_df = df[out_cols].copy()

out_df.to_sql("transactions_labeled", con, if_exists="replace", index=False)

print("[OK] Saved to DB table: transactions_labeled")
print(out_df[["risk_level", "verification_complexity"]].value_counts().head(10))