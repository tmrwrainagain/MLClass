# app_dashboard.py
# ============================================================
# МОДУЛЬ B — 2.1: Аналитическая система (дашборд)
# Технология: Streamlit
# Что делает:
#  - подключается к SQLite
#  - читает транзакции + справочники
#  - показывает метрики, графики, фильтры
#  - поддерживает обновление данных (manual + auto-refresh)
#  - простые уровни доступа (Viewer/Analyst/Admin)
# ============================================================

import os
import sqlite3
from datetime import datetime
import pandas as pd
import streamlit as st

# (опционально) авто-обновление
try:
    from streamlit_autorefresh import st_autorefresh
    AUTORF_AVAILABLE = True
except Exception:
    AUTORF_AVAILABLE = False


# =========================
# 1) НАСТРОЙКИ (МЕНЯТЬ НА СОРЕВНОВАНИИ)
# =========================
DB_PATH = "db/app.db"  # <-- МЕНЯТЬ: путь к SQLite базе (если лежит иначе)

# Таблицы и ключевые колонки (меняешь только если у тебя другие названия)
TABLE_EVENTS = "transactions"     # <-- МЕНЯТЬ: таблица событий
TABLE_MCC = "mcc_codes"           # <-- МЕНЯТЬ: справочник категорий
TABLE_TYPES = "tr_types"          # <-- МЕНЯТЬ: справочник типов
TABLE_TARGET = "gender_train"     # <-- МЕНЯТЬ: таргет/разметка (если нужна)

ID_COL = "customer_id"            # <-- МЕНЯТЬ: общий ключ
DATETIME_COL = "tr_datetime"      # <-- МЕНЯТЬ: колонка времени
MCC_COL = "mcc_code"              # <-- МЕНЯТЬ: колонка категории
TYPE_COL = "tr_type"              # <-- МЕНЯТЬ: колонка типа
AMOUNT_COL = "amount"             # <-- МЕНЯТЬ: сумма/значение


# =========================
# 2) НАСТРОЙКИ ДОСТУПА (УРОВНИ)
# =========================
# В реальной системе это делают через логин/токены/БД.
# На соревновании достаточно "роль выбирается по коду".
ACCESS_CODES = {
    "Viewer": "",                 # <-- обычно не меняем: просмотр без кода
    "Analyst": "1234",            # <-- МЕНЯТЬ: простой код доступа
    "Admin": "admin"              # <-- МЕНЯТЬ: код админа
}

# Что разрешаем ролям:
ROLE_CAN_SEE_RAW = {"Analyst", "Admin"}      # сырые данные (таблица транзакций)
ROLE_CAN_EXPORT = {"Admin"}                  # выгрузка csv
ROLE_CAN_SEE_ADVANCED = {"Analyst", "Admin"} # расширенные графики/фичи


# =========================
# 3) ВСПОМОГАТЕЛЬНОЕ (ОБЫЧНО НЕ МЕНЯЕМ)
# =========================
st.set_page_config(
    page_title="Transaction Risk Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def get_connection(db_path: str):
    # Один коннект на сессию. Если база поменялась — перезапусти Streamlit.
    return sqlite3.connect(db_path, check_same_thread=False)

@st.cache_data(ttl=30)
def read_sql(query: str) -> pd.DataFrame:
    # ttl=30 => каждые 30 секунд при новом запросе данные обновятся автоматически
    con = get_connection(DB_PATH)
    return pd.read_sql_query(query, con)

def safe_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


# =========================
# 4) UI — БОКОВАЯ ПАНЕЛЬ (роль, авто-обновление, фильтры)
# =========================
st.title("📊 Transaction Analytics Dashboard (Module B / 2.1)")

if not safe_exists(DB_PATH):
    st.error(f"База не найдена: {DB_PATH}\n\nПроверь путь DB_PATH в начале файла.")
    st.stop()

st.sidebar.header("⚙️ Управление")

# --- Роль ---
role = st.sidebar.selectbox("Роль", ["Viewer", "Analyst", "Admin"], index=0)
code = st.sidebar.text_input("Код доступа (если нужен)", type="password")

if ACCESS_CODES.get(role, "") != code:
    if role != "Viewer":
        st.sidebar.error("Неверный код. Переключись на Viewer или введи правильный код.")
        role = "Viewer"

st.sidebar.success(f"Текущая роль: {role}")

# --- Автообновление ---
auto_refresh = st.sidebar.checkbox("Автообновление каждые 30 сек", value=False)
if auto_refresh and AUTORF_AVAILABLE:
    st_autorefresh(interval=30_000, key="auto_refresh_30s")
elif auto_refresh and not AUTORF_AVAILABLE:
    st.sidebar.warning("Нет streamlit-autorefresh. Автообновление будет только через кеш ttl=30.")

# --- Кнопка обновить ---
if st.sidebar.button("🔄 Обновить сейчас"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

# =========================
# 5) ЗАГРУЗКА ДАННЫХ ИЗ БД (универсально)
# =========================
# Важно: мы не тянем 6.8 млн строк целиком без фильтра — это может убить память.
# Поэтому:
#  - метрики считаем SQL-агрегациями
#  - для таблицы транзакций берём ограниченно (LIMIT) + фильтры

# --- справочники ---
df_mcc = read_sql(f"SELECT * FROM {TABLE_MCC};")
df_types = read_sql(f"SELECT * FROM {TABLE_TYPES};")

# --- дата-диапазон в данных (для фильтра) ---
q_minmax = f"""
SELECT MIN({DATETIME_COL}) AS dt_min, MAX({DATETIME_COL}) AS dt_max
FROM {TABLE_EVENTS};
"""
minmax = read_sql(q_minmax)
dt_min = str(minmax.loc[0, "dt_min"])
dt_max = str(minmax.loc[0, "dt_max"])

st.caption(f"📌 Данные в БД: {dt_min} → {dt_max} (по полю {DATETIME_COL})")


# =========================
# 6) ФИЛЬТРЫ (меняются под задачу)
# =========================
# Универсальный подход:
#  - фильтр по категории (mcc)
#  - фильтр по типу операции
#  - фильтр по знаку суммы (расход/доход)
#  - лимит строк для таблицы

mcc_list = sorted(df_mcc[MCC_COL].dropna().unique().tolist()) if MCC_COL in df_mcc.columns else []
type_list = sorted(df_types[TYPE_COL].dropna().unique().tolist()) if TYPE_COL in df_types.columns else []

selected_mcc = st.sidebar.multiselect("MCC категории", options=mcc_list, default=[])
selected_types = st.sidebar.multiselect("Типы операций", options=type_list, default=[])

amount_mode = st.sidebar.selectbox("Сумма", ["Все", "Только расходы (amount < 0)", "Только доходы (amount > 0)"], index=0)
limit_rows = st.sidebar.slider("Лимит строк в таблице", min_value=1_000, max_value=200_000, value=20_000, step=1_000)

# Строим SQL WHERE
where_parts = []

if selected_mcc:
    where_parts.append(f"{MCC_COL} IN ({','.join(map(str, selected_mcc))})")

if selected_types:
    where_parts.append(f"{TYPE_COL} IN ({','.join(map(str, selected_types))})")

if amount_mode == "Только расходы (amount < 0)":
    where_parts.append(f"{AMOUNT_COL} < 0")
elif amount_mode == "Только доходы (amount > 0)":
    where_parts.append(f"{AMOUNT_COL} > 0")

WHERE_SQL = "WHERE " + " AND ".join(where_parts) if where_parts else ""


# =========================
# 7) КЛЮЧЕВЫЕ МЕТРИКИ (SQL агрегаты — быстро и правильно)
# =========================
q_metrics = f"""
SELECT
    COUNT(*) as tx_count,
    COUNT(DISTINCT {ID_COL}) as customers,
    AVG({AMOUNT_COL}) as avg_amount,
    SUM(CASE WHEN {AMOUNT_COL} < 0 THEN ABS({AMOUNT_COL}) ELSE 0 END) as total_spend,
    SUM(CASE WHEN {AMOUNT_COL} > 0 THEN {AMOUNT_COL} ELSE 0 END) as total_income
FROM {TABLE_EVENTS}
{WHERE_SQL};
"""
metrics = read_sql(q_metrics).iloc[0].to_dict()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Транзакции", f"{int(metrics['tx_count']):,}")
c2.metric("Клиенты", f"{int(metrics['customers']):,}")
c3.metric("Средний amount", f"{metrics['avg_amount']:.2f}" if metrics["avg_amount"] is not None else "—")
c4.metric("Сумма расходов", f"{metrics['total_spend']:.2f}" if metrics["total_spend"] is not None else "—")
c5.metric("Сумма доходов", f"{metrics['total_income']:.2f}" if metrics["total_income"] is not None else "—")


# =========================
# 8) ГРАФИКИ (простые, но “в зачёт”)
# =========================
# 8.1 Топ категорий по сумме расходов
q_top_mcc = f"""
SELECT
    {MCC_COL} as mcc,
    SUM(CASE WHEN {AMOUNT_COL} < 0 THEN ABS({AMOUNT_COL}) ELSE 0 END) as spend_sum,
    COUNT(*) as cnt
FROM {TABLE_EVENTS}
{WHERE_SQL}
GROUP BY {MCC_COL}
ORDER BY spend_sum DESC
LIMIT 15;
"""
top_mcc = read_sql(q_top_mcc).merge(df_mcc, left_on="mcc", right_on=MCC_COL, how="left")

# 8.2 Активность по "часу"
# Важно: у тебя tr_datetime формата "0 10:23:26" (день + время).
# Для универсальности мы берём час как substring после пробела.
q_by_hour = f"""
SELECT
    CAST(substr({DATETIME_COL}, instr({DATETIME_COL}, ' ') + 1, 2) AS INTEGER) as hour,
    COUNT(*) as cnt,
    SUM(CASE WHEN {AMOUNT_COL} < 0 THEN ABS({AMOUNT_COL}) ELSE 0 END) as spend_sum
FROM {TABLE_EVENTS}
{WHERE_SQL}
GROUP BY hour
ORDER BY hour;
"""
by_hour = read_sql(q_by_hour)

# 8.3 Топ типов операций по количеству
q_top_types = f"""
SELECT
    {TYPE_COL} as tr_type,
    COUNT(*) as cnt
FROM {TABLE_EVENTS}
{WHERE_SQL}
GROUP BY {TYPE_COL}
ORDER BY cnt DESC
LIMIT 15;
"""
top_types = read_sql(q_top_types).merge(df_types, left_on="tr_type", right_on=TYPE_COL, how="left")

g1, g2 = st.columns(2)

with g1:
    st.subheader("🏷️ Топ категорий MCC по расходам")
    if not top_mcc.empty:
        # показываем как таблицу + барчарт
        show_cols = ["mcc", "mcc_description", "spend_sum", "cnt"]
        show_cols = [c for c in show_cols if c in top_mcc.columns]
        st.dataframe(top_mcc[show_cols], use_container_width=True, height=300)
        st.bar_chart(top_mcc.set_index("mcc")["spend_sum"])
    else:
        st.info("Нет данных для выбранных фильтров.")

with g2:
    st.subheader("🕒 Активность по часам")
    if not by_hour.empty:
        st.dataframe(by_hour, use_container_width=True, height=300)
        st.line_chart(by_hour.set_index("hour")[["cnt", "spend_sum"]])
    else:
        st.info("Нет данных для выбранных фильтров.")

st.subheader("🧾 Топ типов операций")
if not top_types.empty:
    show_cols = ["tr_type", "tr_description", "cnt"]
    show_cols = [c for c in show_cols if c in top_types.columns]
    st.dataframe(top_types[show_cols], use_container_width=True, height=300)
else:
    st.info("Нет данных для выбранных фильтров.")


# =========================
# 9) СЫРЫЕ ДАННЫЕ (только Analyst/Admin)
# =========================
if role in ROLE_CAN_SEE_RAW:
    st.subheader("📄 Таблица транзакций (ограниченно)")
    q_preview = f"""
    SELECT {ID_COL}, {DATETIME_COL}, {MCC_COL}, {TYPE_COL}, {AMOUNT_COL}, term_id
    FROM {TABLE_EVENTS}
    {WHERE_SQL}
    LIMIT {int(limit_rows)};
    """
    df_preview = read_sql(q_preview)
    st.dataframe(df_preview, use_container_width=True, height=450)

    if role in ROLE_CAN_EXPORT:
        st.download_button(
            "⬇️ Скачать CSV (preview)",
            data=df_preview.to_csv(index=False).encode("utf-8"),
            file_name=f"transactions_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
else:
    st.info("Роль Viewer: сырые данные скрыты. (Это и есть простой уровень доступа).")


# =========================
# 10) СЛУЖЕБНАЯ ИНФОРМАЦИЯ (для отчёта/защиты)
# =========================
with st.expander("ℹ️ Техническая информация (для отчёта)"):
    st.write(
        """
**Что реализовано (2.1):**
- Подключение к SQLite и чтение данных из нескольких таблиц
- Интерактивные фильтры (категория, тип, расход/доход)
- Метрики и графики, формируемые через SQL-агрегации
- Обновление: кнопка обновления + ttl кеша (30 сек) + опциональный авто-рефреш
- Уровни доступа: Viewer / Analyst / Admin (упрощённая модель ролей)

**Как адаптировать под другой датасет:**
- В начале файла поменять DB_PATH и названия таблиц/колонок
- В блоке фильтров заменить MCC/TYPE на нужные поля
- В блоке графиков заменить SQL-агрегации под требования задания
        """
    )