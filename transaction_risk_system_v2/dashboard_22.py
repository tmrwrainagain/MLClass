# dashboard_22.py
# ============================================================
# MODULE B / 2.2 — Разработка функционала аналитической системы
# Streamlit dashboard: анализ транзакций + фильтры + графики
# ============================================================

import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st

# (опционально) автообновление
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORF_AVAILABLE = True
except Exception:
    AUTORF_AVAILABLE = False


# ============================================================
# 0) НАСТРОЙКИ (МЕНЯТЬ НА СОРЕВНОВАНИИ)
# ============================================================

DB_PATH = "db/app.db"  # <-- МЕНЯТЬ: путь к базе (зависит от того, где запускаешь `streamlit run`)

# Названия таблиц в БД
EVENTS_TABLE = "transactions"   # <-- МЕНЯТЬ
MCC_TABLE = "mcc_codes"         # <-- МЕНЯТЬ
TYPES_TABLE = "tr_types"        # <-- МЕНЯТЬ
GENDER_TABLE = "gender_train"   # <-- МЕНЯТЬ

# Названия колонок
ID_COL = "customer_id"          # <-- МЕНЯТЬ
DT_COL = "tr_datetime"          # <-- МЕНЯТЬ
MCC_COL = "mcc_code"            # <-- МЕНЯТЬ
TYPE_COL = "tr_type"            # <-- МЕНЯТЬ
AMOUNT_COL = "amount"           # <-- МЕНЯТЬ
TERM_COL = "term_id"            # <-- МЕНЯТЬ/ОТКЛЮЧИТЬ (если нет)

# В справочниках
MCC_DESC_COL = "mcc_description"    # <-- МЕНЯТЬ
TYPE_DESC_COL = "tr_description"    # <-- МЕНЯТЬ
GENDER_COL = "gender"               # <-- МЕНЯТЬ

# Ограничение для производительности
MAX_ROWS = 2_000_000  # <-- МЕНЯТЬ: 200k..5M (или None)

# Код доступа для Analyst (имитация уровней доступа)
ACCESS_CODE = "1234"  # <-- МЕНЯТЬ


# ============================================================
# 1) UI / PAGE
# ============================================================

st.set_page_config(page_title="Transaction Analytics 2.2", layout="wide")
st.title("📊 Transaction Analytics Dashboard (Module B / 2.2)")
st.caption("Фильтры: MCC / пол / время суток / тип операции. Графики и таблицы по заданию 2.2.")




# ============================================================
# 2) ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

@st.cache_resource
def get_conn(db_path: str):
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB file not found: {db_path}")
    return sqlite3.connect(db_path, check_same_thread=False)

def safe_read_sql(con, sql: str) -> pd.DataFrame | None:
    try:
        return pd.read_sql(sql, con)
    except Exception:
        return None

def parse_dt_to_hour(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
    """
    Универсально получаем hour из dt_col.
    Поддержка форматов:
      - "0 10:23:26"
      - "YYYY-MM-DD HH:MM:SS"
      - что-то похожее (пытаемся вытащить час)
    """
    if dt_col not in df.columns:
        df["hour"] = np.nan
        return df

    s = df[dt_col].astype(str)

    # если есть пробел — часто справа время
    parts = s.str.split(" ", n=1, expand=True)
    t = parts[1] if parts.shape[1] == 2 else s

    # час = до двоеточия
    hour = t.str.split(":", n=1, expand=True)[0]
    df["hour"] = pd.to_numeric(hour, errors="coerce")
    return df

def hour_to_tod(hour: pd.Series) -> pd.Series:
    """
    time-of-day:
      night: 0-5
      morning: 6-11
      day: 12-17
      evening: 18-23
    """
    h = hour.fillna(-1).astype(int)
    return pd.Series(
        np.where((h >= 0) & (h <= 5), "night",
        np.where((h >= 6) & (h <= 11), "morning",
        np.where((h >= 12) & (h <= 17), "day",
        np.where((h >= 18) & (h <= 23), "evening", "unknown")))),
        index=hour.index
    )


# ============================================================
# 3) ЗАГРУЗКА ДАННЫХ ИЗ БД
# ============================================================

with st.spinner("⏳ Загружаю данные из БД и готовлю дашборд…"):
    con = get_conn(DB_PATH)

    limit_sql = f"LIMIT {MAX_ROWS}" if isinstance(MAX_ROWS, int) and MAX_ROWS > 0 else ""

    sql_events = f"""
    SELECT
      {ID_COL} AS customer_id,
      {DT_COL} AS tr_datetime,
      {MCC_COL} AS mcc_code,
      {TYPE_COL} AS tr_type,
      {AMOUNT_COL} AS amount,
      {TERM_COL} AS term_id
    FROM {EVENTS_TABLE}
    {limit_sql}
    """
    df_e = safe_read_sql(con, sql_events)
    if df_e is None or df_e.empty:
        st.error("Не удалось загрузить events из БД. Проверь DB_PATH, EVENTS_TABLE и названия колонок.")
        st.stop()

    # справочники (могут отсутствовать)
    df_m = safe_read_sql(con, f"SELECT {MCC_COL} AS mcc_code, {MCC_DESC_COL} AS mcc_description FROM {MCC_TABLE}")
    df_t = safe_read_sql(con, f"SELECT {TYPE_COL} AS tr_type, {TYPE_DESC_COL} AS tr_description FROM {TYPES_TABLE}")
    df_g_raw = safe_read_sql(con, f"SELECT {ID_COL} AS customer_id, {GENDER_COL} AS gender FROM {GENDER_TABLE}")

    # join MCC
    if isinstance(df_m, pd.DataFrame) and not df_m.empty and {"mcc_code", "mcc_description"}.issubset(df_m.columns):
        df_e = df_e.merge(df_m.drop_duplicates("mcc_code"), on="mcc_code", how="left")
    else:
        df_e["mcc_description"] = None

    # join Types
    if isinstance(df_t, pd.DataFrame) and not df_t.empty and {"tr_type", "tr_description"}.issubset(df_t.columns):
        df_e = df_e.merge(df_t.drop_duplicates("tr_type"), on="tr_type", how="left")
    else:
        df_e["tr_description"] = None

    # join Gender (Pylance-safe)
    has_gender = isinstance(df_g_raw, pd.DataFrame) and (not df_g_raw.empty) and {"customer_id", "gender"}.issubset(df_g_raw.columns)
    if has_gender:
        df_g = df_g_raw[["customer_id", "gender"]].drop_duplicates("customer_id")
        df_e = df_e.merge(df_g, on="customer_id", how="left")
    else:
        df_e["gender"] = np.nan

    # базовая предобработка
    df_e = parse_dt_to_hour(df_e, "tr_datetime")
    df_e["tod"] = hour_to_tod(df_e["hour"])

    df_e["flow"] = np.where(df_e["amount"] < 0, "spend", np.where(df_e["amount"] > 0, "income", "zero"))
    df_e["abs_amount"] = df_e["amount"].abs()

    st.caption(f"✅ Загружено событий: {len(df_e):,} | клиентов: {df_e['customer_id'].nunique():,}")




# ============================================================
# 4) SIDEBAR: РОЛИ + ФИЛЬТРЫ (Единственный блок!)
# ============================================================

st.sidebar.header("⚙️ Роль и фильтры 2.2")

# --- РОЛИ ---
if "role_22" not in st.session_state:
    st.session_state.role_22 = "Viewer"

role_choice = st.sidebar.selectbox(
    "Роль",
    ["Viewer", "Analyst"],
    index=0 if st.session_state.role_22 == "Viewer" else 1,
    key="role_sel_22"   # ✅ уникальный key
)

role = "Viewer"
is_analyst = False

if role_choice == "Viewer":
    role = "Viewer"
    is_analyst = False
    st.session_state.role_22 = "Viewer"
    st.sidebar.info("Роль: Viewer")
else:
    code = st.sidebar.text_input(
        "Код доступа (Analyst)",
        type="password",
        key="pwd_22"      # ✅ уникальный key
    )

    # пока код не введён — не валим ошибкой, просто просим ввести
    if code.strip() == "":
        st.sidebar.info("Введите код, чтобы включить Analyst.")
        st.session_state.role_22 = "Viewer"
        st.stop()

    if code.strip() == str(ACCESS_CODE).strip():
        role = "Analyst"
        is_analyst = True
        st.session_state.role_22 = "Analyst"
        st.sidebar.success("Доступ Analyst ✅")
    else:
        st.sidebar.error("Неверный код. Доступ только Viewer.")
        st.session_state.role_22 = "Viewer"
        st.stop()

st.sidebar.caption(f"Текущая роль: {role}")

# --- Автообновление (только Analyst) ---
auto_refresh = st.sidebar.checkbox(
    "Автообновление каждые 30 сек",
    value=False,
    disabled=not is_analyst,
    key="autorefresh_chk_22"     # ✅ уникальный key
)

if auto_refresh:
    if AUTORF_AVAILABLE:
        st_autorefresh(interval=30_000, key="autorefresh_tick_22")
    else:
        st.sidebar.warning("streamlit-autorefresh не установлен")

st.sidebar.divider()

# --- ФИЛЬТРЫ ---
all_mcc = sorted(df_e["mcc_code"].dropna().unique().tolist())
mcc_selected = st.sidebar.multiselect(
    "Категории MCC",
    all_mcc,
    default=[],
    key="mcc_ms_22"              # ✅ уникальный key
)

tod_all = ["night", "morning", "day", "evening", "unknown"]
tod_selected = st.sidebar.multiselect(
    "Время суток",
    tod_all,
    default=["night", "morning", "day", "evening"],
    key="tod_ms_22"              # ✅ уникальный key
)

gender_selected = None
if has_gender and df_e["gender"].notna().any():
    gender_vals = sorted(df_e["gender"].dropna().unique().tolist())
    gender_selected = st.sidebar.multiselect(
        "Пол (gender)",
        gender_vals,
        default=gender_vals,
        key="gender_ms_22"        # ✅ уникальный key
    )

flow_selected = st.sidebar.multiselect(
    "Тип потока",
    ["spend", "income", "zero"],
    default=["spend"],
    key="flow_ms_22"             # ✅ уникальный key
)

# --- ПРИМЕНЯЕМ ФИЛЬТРЫ -> df_f (чтобы потом не было NameError) ---
df_f = df_e.copy()

if mcc_selected:
    df_f = df_f[df_f["mcc_code"].isin(mcc_selected)]

if tod_selected:
    df_f = df_f[df_f["tod"].isin(tod_selected)]

if gender_selected is not None:
    df_f = df_f[df_f["gender"].isin(gender_selected)]

if flow_selected:
    df_f = df_f[df_f["flow"].isin(flow_selected)]

st.sidebar.caption(f"Строк после фильтра: {len(df_f):,}")

# ============================================================
# 5) КЛЮЧЕВЫЕ МЕТРИКИ
# ============================================================

if 'df_f' not in globals() or df_f.empty:
    st.warning("Нет данных для отображения (проверь фильтры)")
else:
    tx_cnt = len(df_f)
    clients_cnt = df_f["customer_id"].nunique()

    avg_check = df_f["abs_amount"].mean()
    avg_check = 0.0 if pd.isna(avg_check) else avg_check

    spend_sum = df_f.loc[df_f["flow"] == "spend", "abs_amount"].sum()
    income_sum = df_f.loc[df_f["flow"] == "income", "amount"].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Транзакции", f"{tx_cnt:,}")
    c2.metric("Клиенты", f"{clients_cnt:,}")
    c3.metric("Средний чек (abs)", f"{avg_check:.2f}")
    c4.metric("Сумма расходов (abs)", f"{spend_sum:.2f}")
    c5.metric("Сумма доходов", f"{income_sum:.2f}")


# ============================================================
# 6) АНАЛИТИКА 2.2
# ============================================================

if df_f.empty:
    st.stop()

st.markdown("## 2.2 Аналитика финансового поведения")

left, right = st.columns([1, 1])

# A) Средний размер транзакций по категориям в разное время суток
with left:
    st.subheader("📌 Средний чек по MCC и времени суток")

    tmp = df_f.copy()
    agg = (
        tmp.groupby(["tod", "mcc_code"], dropna=False)
           .agg(
                avg_amount=("abs_amount", "mean"),
                cnt=("abs_amount", "size"),
                sum_amount=("abs_amount", "sum"),
            )
           .reset_index()
    )

    if "mcc_description" in tmp.columns:
        agg = agg.merge(tmp[["mcc_code", "mcc_description"]].drop_duplicates(), on="mcc_code", how="left")

    st.dataframe(agg.sort_values("sum_amount", ascending=False).head(20), use_container_width=True)

# B) График зависимости суммы операций от времени суток (по часам)
with right:
    st.subheader("📈 Сумма операций по часам")

    tmp = df_f.copy()
    tmp["_spend_abs"] = np.where(tmp["flow"] == "spend", tmp["abs_amount"], 0.0)
    tmp["_income_amt"] = np.where(tmp["flow"] == "income", tmp["amount"], 0.0)

    by_hour = (
        tmp.groupby("hour", dropna=False)
           .agg(
                cnt=("amount", "size"),
                spend_sum=("_spend_abs", "sum"),
                income_sum=("_income_amt", "sum"),
            )
           .reset_index()
           .sort_values("hour")
    )

    st.line_chart(by_hour.set_index("hour")[["cnt", "spend_sum", "income_sum"]], use_container_width=True)
    st.caption("Линии: cnt (кол-во), spend_sum (расходы), income_sum (доходы).")

# C) Влияние категории расходов на активность клиентов
st.subheader("🔥 Влияние категории (MCC) на активность клиентов")

df_spend = df_f[df_f["flow"] == "spend"].copy()
act = (
    df_spend.groupby("mcc_code")
            .agg(
                tx_cnt=("amount", "size"),
                clients=("customer_id", "nunique"),
                spend_sum=("abs_amount", "sum"),
                avg_check=("abs_amount", "mean"),
            )
            .reset_index()
            .sort_values("spend_sum", ascending=False)
)

if "mcc_description" in df_spend.columns:
    act = act.merge(df_spend[["mcc_code", "mcc_description"]].drop_duplicates(), on="mcc_code", how="left")

st.dataframe(act.head(30), use_container_width=True)
st.bar_chart(act.head(30).set_index("mcc_code")["spend_sum"], use_container_width=True)

# D) Взаимосвязь типа транзакции (трата/поступление) с частотой операций
st.subheader("🔁 Частота операций: траты vs поступления")

freq = (
    df_f.groupby("flow")
        .agg(
            tx_cnt=("amount", "size"),
            clients=("customer_id", "nunique"),
            avg_abs=("abs_amount", "mean"),
            sum_abs=("abs_amount", "sum"),
        )
        .reset_index()
)

st.dataframe(freq, use_container_width=True)
st.bar_chart(freq.set_index("flow")["tx_cnt"], use_container_width=True)

# E) Популярные категории трат в разное время суток
st.subheader("🕒 Топ категорий трат по времени суток (spend)")

top_tod = (
    df_spend.groupby(["tod", "mcc_code"])
            .agg(cnt=("amount", "size"), spend_sum=("abs_amount", "sum"))
            .reset_index()
            .sort_values(["tod", "spend_sum"], ascending=[True, False])
)

if "mcc_description" in df_spend.columns:
    top_tod = top_tod.merge(
        df_spend[["mcc_code", "mcc_description"]].drop_duplicates(),
        on="mcc_code",
        how="left"
    )

st.dataframe(top_tod.groupby("tod").head(10), use_container_width=True)

# F) Диаграмма распределения расходов по категориям и полу клиентов
st.subheader("👥 Расходы по MCC и полу (если доступно)")

if has_gender and df_spend["gender"].notna().any():
    gcat = (
        df_spend.groupby(["gender", "mcc_code"])
                .agg(spend_sum=("abs_amount", "sum"), cnt=("amount", "size"))
                .reset_index()
                .sort_values("spend_sum", ascending=False)
    )

    if "mcc_description" in df_spend.columns:
        gcat = gcat.merge(
            df_spend[["mcc_code", "mcc_description"]].drop_duplicates(),
            on="mcc_code",
            how="left"
        )

    st.dataframe(gcat.head(30), use_container_width=True)

    top_mcc = (
        df_spend.groupby("mcc_code")["abs_amount"]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .index
                .tolist()
    )
    gcat_top = gcat[gcat["mcc_code"].isin(top_mcc)].copy()
    pivot = gcat_top.pivot_table(index="mcc_code", columns="gender", values="spend_sum", fill_value=0)
    st.bar_chart(pivot, use_container_width=True)
else:
    st.info("Пол (gender) отсутствует или не подгрузился — блок пропущен.")


# ============================================================
# G) Интерактивная таблица событий (с фильтрами) — только Analyst
# ============================================================

st.subheader("🧾 Таблица транзакций (с фильтрами)")

if not is_analyst:
    st.info("Роль Viewer: доступны агрегаты и графики. Таблица транзакций доступна только Analyst.")
else:
    cols_show = [
        "customer_id", "tr_datetime", "hour", "tod",
        "mcc_code", "mcc_description",
        "tr_type", "tr_description",
        "amount", "flow", "term_id", "gender"
    ]
    cols_show = [c for c in cols_show if c in df_f.columns]

    limit_rows = st.slider("Лимит строк в таблице", 1000, 50000, 20000, step=1000, key="table_limit_22")
    st.dataframe(df_f[cols_show].head(limit_rows), use_container_width=True)

    st.caption("✅ Задание 2.2: фильтрация + вычисления + визуализации выполнены.")

    with st.expander("🔍 Диагностика фильтров"):
        st.write({
            "rows_after_filters": int(len(df_f)),
            "unique_customers": int(df_f["customer_id"].nunique()),
            "mcc_selected": mcc_selected[:10],
            "tod_selected": tod_selected,
            "gender_filter_enabled": bool(gender_selected is not None),
            "flow_selected": flow_selected,
        })


# ============================================================
# 7) ПОДСКАЗКИ ДЛЯ СОРЕВНОВАНИЯ
# ============================================================

with st.expander("🛠 Что менять под другие соревнования (шпаргалка)"):
    st.markdown(
        """
**Меняем чаще всего:**
- `DB_PATH` — путь к базе  
- `EVENTS_TABLE / MCC_TABLE / TYPES_TABLE / GENDER_TABLE` — названия таблиц  
- `ID_COL / DT_COL / MCC_COL / TYPE_COL / AMOUNT_COL` — названия колонок  
- `MAX_ROWS` — ограничение по объёму  
- Логику `flow`: сейчас по знаку `amount`  

**Если нет справочников/пола:**  
код не падает — просто убирает часть визуализаций.
        """
    )