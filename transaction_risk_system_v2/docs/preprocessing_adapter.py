# ============================================================
# UNIVERSAL PREPROCESSING ADAPTER
# Цель:
#   Привести "грязные" входные данные к виду,
#   пригодному для feature engineering и ML
#
# Используется ПЕРЕД шаблоном агрегации признаков
#
# Этот код:
# - не удаляет данные без причины
# - не искажает распределения
# - аккуратно приводит типы
# ============================================================

import pandas as pd
import numpy as np

# -------------------------------
# НАСТРОЙКИ (МЕНЯЮТСЯ ПОД ДАННЫЕ)
# -------------------------------

ID_COL = "customer_id"          # <-- МЕНЯТЬ: ID объекта
AMOUNT_COL = "amount"           # <-- МЕНЯТЬ: числовая колонка
DATETIME_COL = "tr_datetime"    # <-- МЕНЯТЬ: колонка времени (если есть)

# Категориальные колонки (можно добавлять/убирать)
CATEGORICAL_COLS = [
    "mcc_code",                 # <-- МЕНЯТЬ
    "tr_type",                  # <-- МЕНЯТЬ
    "term_id",                  # <-- МЕНЯТЬ или удалить
]

# -------------------------------
# 0) БАЗОВЫЕ ПРОВЕРКИ
# -------------------------------

if ID_COL not in df.columns:
    raise ValueError(f"❌ Нет ID колонки: {ID_COL}")

if AMOUNT_COL not in df.columns:
    raise ValueError(f"❌ Нет числовой колонки: {AMOUNT_COL}")

print("[OK] Base columns present")

# -------------------------------
# 1) ПРИВЕДЕНИЕ ЧИСЕЛ
# -------------------------------
# Частые проблемы:
# - "1 234,56"
# - "1,234.56"
# - числа как строки
# - NULL / empty

df[AMOUNT_COL] = (
    df[AMOUNT_COL]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "NULL": np.nan})
)

df[AMOUNT_COL] = pd.to_numeric(df[AMOUNT_COL], errors="coerce")

print("[OK] Amount column converted to numeric")

# -------------------------------
# 2) РАБОТА С ДАТА / ВРЕМЯ
# -------------------------------
# Если есть datetime → создаём hour
# Если нет — блок спокойно пропускается

if DATETIME_COL in df.columns:
    df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL], errors="coerce")

    df["hour"] = df[DATETIME_COL].dt.hour
    df["day"] = df[DATETIME_COL].dt.day
    df["weekday"] = df[DATETIME_COL].dt.weekday

    print("[OK] Datetime parsed, hour/day/weekday created")
else:
    print("[INFO] No datetime column found — skipping time features")

# -------------------------------
# 3) КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
# -------------------------------
# Приводим строки → category → codes
# Это безопасно для:
# - groupby
# - value_counts
# - share-признаков

for col in CATEGORICAL_COLS:
    if col not in df.columns:
        continue

    if df[col].dtype == "object":
        df[col] = (
            df[col]
              .astype(str)
              .replace({"nan": np.nan, "None": np.nan, "NULL": np.nan})
              .astype("category")
              .cat.codes
        )

        # pandas кодирует NaN как -1 → вернём NaN
        df[col] = df[col].replace(-1, np.nan)

        print(f"[OK] Categorical column encoded: {col}")
    else:
        print(f"[OK] Categorical column already numeric: {col}")

# -------------------------------
# 4) МИНИ-ДИАГНОСТИКА
# -------------------------------

print("\n[DIAG] dtypes:")
display(df.dtypes)

print("\n[DIAG] missing values (top):")
display(df.isna().mean().sort_values(ascending=False).head(10))

print("\n[OK] Preprocessing adapter finished")