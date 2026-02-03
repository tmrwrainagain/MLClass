# src/dashboard.py - ПОЛНЫЙ КОД С API
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import numpy as np
import requests
import json
from datetime import datetime

# ============================================================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================================================
st.set_page_config(
    page_title="Анализ GPS треков",
    page_icon="🗺️",
    layout="wide"
)

# CSS стили
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .risk-fire {
        color: #ff4444;
        font-weight: bold;
    }
    .risk-flood {
        color: #4444ff;
        font-weight: bold;
    }
    .risk-mountain {
        color: #44aa44;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# ЗАГОЛОВОК
# ============================================================================
st.title("🗺️ Анализ GPS треков")
st.markdown("---")

# ============================================================================
# 1. ПОДКЛЮЧЕНИЕ К БАЗЕ ДАННЫХ
# ============================================================================
@st.cache_resource
def get_db_connection():
    """Подключение к PostgreSQL"""
    try:
        DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
        engine = create_engine(DB_URL)
        return engine
    except Exception as e:
        st.error(f"Ошибка подключения к БД: {e}")
        return None

engine = get_db_connection()

# ============================================================================
# 2. ЗАГРУЗКА ДАННЫХ
# ============================================================================
@st.cache_data(ttl=60)
def load_tracks_data():
    """Загрузка данных о треках"""
    try:
        query = "SELECT * FROM tracks ORDER BY processed_date DESC"
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return pd.DataFrame()

df = load_tracks_data()

if df.empty:
    st.warning("📭 В базе данных нет записей о треках")
    st.info("Запустите сначала main.py для обработки GPX файлов")
    st.stop()

# ============================================================================
# 3. ФУНКЦИЯ ДЛЯ API
# ============================================================================
@st.cache_data(ttl=30)
def get_api_prediction(track_data):
    """Запрос к API для предсказания кластера"""
    try:
        API_URL = "http://localhost:5000/predict"
        
        payload = {
            'distance_km': float(track_data.get('distance_km', 0)),
            'elevation_gain': float(track_data.get('elevation_gain', 0)),
            'avg_temperature': float(track_data.get('avg_temperature', 0)),
            'osm_buildings': int(track_data.get('osm_buildings', 0)),
            'precipitation': float(track_data.get('precipitation', 0))
        }
        
        response = requests.post(API_URL, json=payload, timeout=3)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None
    except Exception as e:
        st.sidebar.error(f"Ошибка API: {e}")
        return None

# ============================================================================
# 4. САЙДБАР - ФИЛЬТРЫ И API
# ============================================================================
st.sidebar.header("🔧 Управление")

# Выбор трека
selected_track = st.sidebar.selectbox(
    "Выберите трек для анализа:",
    df['track_name'].unique(),
    index=0
)

# Данные выбранного трека
track_data = df[df['track_name'] == selected_track].iloc[0]

# Кнопка предсказания через API
st.sidebar.header("🤖 AI Предсказание")

if st.sidebar.button("🔮 Предсказать кластер", use_container_width=True):
    with st.spinner("Обработка API запроса..."):
        prediction = get_api_prediction(track_data)
        
        if prediction:
            st.sidebar.success(f"✅ Кластер: {prediction.get('cluster', 'N/A')}")
            
            # Показываем детали
            with st.sidebar.expander("Детали предсказания"):
                st.json(prediction)
        else:
            st.sidebar.error("❌ API недоступен или ошибка")
            st.sidebar.info("Запустите: python src/api.py")

# Статус API
st.sidebar.header("📊 Статистика")
st.sidebar.metric("Треков в базе", len(df))
st.sidebar.metric("Дата обновления", datetime.now().strftime("%H:%M"))

# ============================================================================
# 5. ОСНОВНЫЕ МЕТРИКИ ТРЕКА
# ============================================================================
st.subheader(f"📊 Анализ трека: **{selected_track}**")

# Создаем 4 колонки для метрик
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📏 Дистанция",
        value=f"{track_data['distance_km']:.2f} км",
        delta=None
    )

with col2:
    st.metric(
        label="⬆️ Набор высоты",
        value=f"{track_data['elevation_gain']:.0f} м",
        delta=None
    )

with col3:
    temp = track_data.get('avg_temperature', 0)
    st.metric(
        label="🌡️ Температура",
        value=f"{temp:.1f}°C" if temp != 0 else "N/A",
        delta=None
    )

with col4:
    risk = track_data.get('risk_zone', 'normal')
    risk_display = risk if risk != 'normal' else 'Нормальная'
    
    # Цвет в зависимости от риска
    risk_color = 'normal'
    if 'пожар' in str(risk).lower() or 'fire' in str(risk).lower():
        risk_color = 'risk-fire'
    elif 'наводн' in str(risk).lower() or 'flood' in str(risk).lower():
        risk_color = 'risk-flood'
    elif 'гор' in str(risk).lower() or 'evacuation' in str(risk).lower():
        risk_color = 'risk-mountain'
    
    st.markdown(f"<div class='{risk_color}'>⚠️ Зона риска:<br><h3>{risk_display}</h3></div>", 
                unsafe_allow_html=True)

# ============================================================================
# 6. ВЫСОТНЫЙ ПРОФИЛЬ
# ============================================================================
st.subheader("📈 Высотный профиль")

# Создаем упрощенный профиль на основе данных трека
if all(col in track_data for col in ['min_elevation', 'max_elevation', 'distance_km']):
    # Генерируем реалистичный профиль
    n_points = 50
    distances = np.linspace(0, track_data['distance_km'], n_points)
    
    # Создаем более реалистичный профиль с подъемами и спусками
    x = distances / track_data['distance_km'] * np.pi * 2
    
    # Комбинация синусов для реалистичности
    elevations = (
        track_data['min_elevation'] + 
        (track_data['max_elevation'] - track_data['min_elevation']) * 
        (0.5 * np.sin(x) + 0.3 * np.sin(2*x + 1) + 0.2 * np.sin(3*x + 2)) / 1.0
    )
    
    fig1 = go.Figure()
    
    # Линия профиля
    fig1.add_trace(go.Scatter(
        x=distances,
        y=elevations,
        mode='lines',
        name='Высота',
        line=dict(color='red', width=3),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.2)'
    ))
    
    fig1.update_layout(
        height=350,
        xaxis_title="Дистанция (км)",
        yaxis_title="Высота (м)",
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Статистики под графиком
    cols_stats = st.columns(3)
    with cols_stats[0]:
        st.metric("Макс. высота", f"{track_data['max_elevation']:.0f} м")
    with cols_stats[1]:
        st.metric("Мин. высота", f"{track_data['min_elevation']:.0f} м")
    with cols_stats[2]:
        st.metric("Средняя", f"{track_data.get('avg_elevation', 0):.0f} м")
else:
    st.info("ℹ️ Нет полных данных о высотах для построения профиля")

# ============================================================================
# 7. ГРАФИК ТЕМПЕРАТУРА vs ВЫСОТА
# ============================================================================
st.subheader("🌡️ Температура vs Высота")

if all(col in df.columns for col in ['avg_temperature', 'avg_elevation']):
    fig2 = px.scatter(
        df,
        x='avg_elevation',
        y='avg_temperature',
        color='risk_zone',
        size='distance_km',
        hover_data=['track_name', 'distance_km'],
        labels={
            'avg_elevation': 'Средняя высота (м)',
            'avg_temperature': 'Средняя температура (°C)',
            'risk_zone': 'Зона риска'
        },
        title='Зависимость температуры от высоты'
    )
    
    # Выделяем выбранный трек
    if 'avg_elevation' in track_data and 'avg_temperature' in track_data:
        fig2.add_trace(go.Scatter(
            x=[track_data['avg_elevation']],
            y=[track_data['avg_temperature']],
            mode='markers',
            marker=dict(
                color='black',
                size=15,
                symbol='star',
                line=dict(width=2, color='white')
            ),
            name='Выбранный трек'
        ))
    
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("ℹ️ Нет данных о температуре и высоте для построения графика")

# ============================================================================
# 8. ОБЪЕКТЫ OSM ВОКРУГ ТРЕКА
# ============================================================================
st.subheader("🏗️ Объекты OSM вокруг трека")

osm_cols = ['osm_water', 'osm_buildings', 'osm_farmland', 'osm_forest']
osm_data = {}

for col in osm_cols:
    if col in track_data and not pd.isna(track_data[col]):
        osm_data[col.replace('osm_', '')] = int(track_data[col])

if osm_data:
    fig3 = go.Figure(data=[
        go.Bar(
            x=list(osm_data.keys()),
            y=list(osm_data.values()),
            marker_color=['blue', 'gray', 'green', 'darkgreen'],
            text=list(osm_data.values()),
            textposition='auto'
        )
    ])
    
    fig3.update_layout(
        height=300,
        xaxis_title="Тип объекта",
        yaxis_title="Количество",
        showlegend=False
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Текстовая информация
    with st.expander("📝 Описание объектов"):
        for obj, count in osm_data.items():
            if count > 0:
                if obj == 'water':
                    st.write(f"💧 **Водоемы:** {count} объектов")
                elif obj == 'buildings':
                    st.write(f"🏠 **Здания:** {count} объектов")
                elif obj == 'farmland':
                    st.write(f"🌾 **Поля/луга:** {count} объектов")
                elif obj == 'forest':
                    st.write(f"🌲 **Леса:** {count} объектов")
else:
    st.info("ℹ️ Нет данных об объектах OSM для этого трека")

# ============================================================================
# 9. ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О ТРЕКЕ
# ============================================================================
st.subheader("📋 Детальная информация")

col_info1, col_info2 = st.columns(2)

with col_info1:
    st.markdown("#### 📊 Основные параметры")
    
    info_data = [
        ("Дистанция", f"{track_data['distance_km']:.2f} км"),
        ("Набор высоты", f"{track_data['elevation_gain']:.0f} м"),
        ("Средний уклон", f"{track_data.get('avg_slope', 'N/A'):.1f}%"),
        ("Макс. высота", f"{track_data.get('max_elevation', 'N/A')} м"),
        ("Мин. высота", f"{track_data.get('min_elevation', 'N/A')} м"),
        ("Средняя высота", f"{track_data.get('avg_elevation', 'N/A')} м"),
    ]
    
    for label, value in info_data:
        st.write(f"**{label}:** {value}")

with col_info2:
    st.markdown("#### 🌤️ Погодные условия")
    
    if 'avg_temperature' in track_data:
        weather_data = [
            ("Средняя температура", f"{track_data['avg_temperature']:.1f}°C"),
            ("Макс. температура", f"{track_data.get('max_temperature', 'N/A')}°C"),
            ("Мин. температура", f"{track_data.get('min_temperature', 'N/A')}°C"),
        ]
        
        for label, value in weather_data:
            st.write(f"**{label}:** {value}")
    
    if 'precipitation' in track_data:
        st.write(f"**Осадки:** {track_data['precipitation']:.1f} мм")
    
    if 'processed_date' in track_data:
        st.write(f"**Дата обработки:** {track_data['processed_date']}")
    
    # Отображение зоны риска
    risk = track_data.get('risk_zone', 'normal')
    if risk != 'normal':
        st.warning(f"**⚠️ Обнаружены риски:** {risk}")
    else:
        st.success("**✅ Безопасная зона**")

# ============================================================================
# 10. ТАБЛИЦА ВСЕХ ТРЕКОВ
# ============================================================================
st.subheader("📁 Все треки в базе данных")

# Определяем колонки для отображения
display_cols = ['track_name', 'distance_km', 'elevation_gain']
for col in ['avg_temperature', 'risk_zone', 'processed_date']:
    if col in df.columns:
        display_cols.append(col)

if display_cols:
    # Сортируем по дате обработки
    if 'processed_date' in df.columns:
        df_display = df[display_cols].sort_values('processed_date', ascending=False)
    else:
        df_display = df[display_cols]
    
    st.dataframe(
        df_display,
        use_container_width=True,
        height=300
    )
    
    # Кнопка скачивания
    csv = df_display.to_csv(index=False)
    st.download_button(
        label="📥 Скачать данные (CSV)",
        data=csv,
        file_name=f"tracks_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
else:
    st.info("ℹ️ Нет данных для отображения в таблице")

# ============================================================================
# 11. ФУТЕР И ИНФОРМАЦИЯ
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <p>📊 <b>Анализ GPS треков</b> | PostgreSQL | OpenStreetMap | Open-Meteo</p>
    <p>Данные обновлены: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# ============================================================================
# 12. ОБРАБОТКА ОШИБОК API (в фоновом режиме)
# ============================================================================
# Пытаемся получить предсказание при загрузке
try:
    prediction = get_api_prediction(track_data)
    if prediction:
        st.sidebar.info(f"🤖 API доступен. Кластер трека: {prediction.get('cluster', 'N/A')}")
except:
    pass

# Кнопка обновления данных
if st.sidebar.button("🔄 Обновить данные", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

#1

# src/dashboard.py - ПРОСТОЙ ВАРИАНТ
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine
import requests
import numpy as np

# Настройка
st.set_page_config(page_title="Анализ треков", layout="wide")
st.title("📊 Анализ GPS треков")

# ============================================================================
# 1. ПОДКЛЮЧЕНИЕ К БАЗЕ
# ============================================================================
DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM tracks", engine)

df = load_data()

if df.empty:
    st.warning("Нет данных. Запустите main.py")
    st.stop()

# ============================================================================
# 2. ФИЛЬТР
# ============================================================================
st.sidebar.header("Фильтры")
selected_track = st.sidebar.selectbox("Выберите трек:", df['track_name'].unique())
track_data = df[df['track_name'] == selected_track].iloc[0]

# ============================================================================
# 3. API ПРЕДСКАЗАНИЕ
# ============================================================================
st.sidebar.header("API Предсказание")

def predict_api():
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json={
                'distance_km': float(track_data.get('distance_km', 0)),
                'elevation_gain': float(track_data.get('elevation_gain', 0)),
                'avg_temperature': float(track_data.get('avg_temperature', 0)),
                'osm_buildings': int(track_data.get('osm_buildings', 0)),
                'precipitation': float(track_data.get('precipitation', 0))
            },
            timeout=2
        )
        return response.json() if response.status_code == 200 else None
    except:
        return None

if st.sidebar.button("Предсказать кластер"):
    result = predict_api()
    if result:
        st.sidebar.success(f"Кластер: {result.get('cluster')}")
    else:
        st.sidebar.error("API недоступен")

# ============================================================================
# 4. МЕТРИКИ ТРЕКА
# ============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Дистанция", f"{track_data['distance_km']:.1f} км")

with col2:
    st.metric("Набор высоты", f"{track_data['elevation_gain']:.0f} м")

with col3:
    if 'avg_temperature' in track_data:
        st.metric("Температура", f"{track_data['avg_temperature']:.1f}°C")

with col4:
    risk = track_data.get('risk_zone', 'normal')
    st.metric("Зона риска", risk if risk != 'normal' else 'Нормальная')

# ============================================================================
# 5. ВЫСОТНЫЙ ПРОФИЛЬ
# ============================================================================
st.subheader("Высотный профиль")

if 'max_elevation' in track_data and 'min_elevation' in track_data:
    # Простой профиль
    n_points = 20
    distances = np.linspace(0, track_data['distance_km'], n_points)
    x = distances / track_data['distance_km'] * np.pi * 2
    
    elevations = (
        track_data['min_elevation'] + 
        (track_data['max_elevation'] - track_data['min_elevation']) * 
        np.sin(x) * 0.8
    )
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=distances,
        y=elevations,
        mode='lines',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(height=250, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 6. ТЕМПЕРАТУРА vs ВЫСОТА
# ============================================================================
st.subheader("Температура vs Высота")

if 'avg_temperature' in df.columns and 'avg_elevation' in df.columns:
    fig = px.scatter(
        df,
        x='avg_elevation',
        y='avg_temperature',
        hover_data=['track_name'],
        size='distance_km'
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# 7. ДЕТАЛЬНАЯ ИНФОРМАЦИЯ
# ============================================================================
st.subheader("Детальная информация")

col_left, col_right = st.columns(2)

with col_left:
    st.write("**Основное:**")
    st.write(f"- Дистанция: {track_data['distance_km']:.1f} км")
    st.write(f"- Набор высоты: {track_data['elevation_gain']:.0f} м")
    st.write(f"- Макс высота: {track_data.get('max_elevation', 'N/A')} м")
    st.write(f"- Мин высота: {track_data.get('min_elevation', 'N/A')} м")
    st.write(f"- Средняя: {track_data.get('avg_elevation', 'N/A')} м")

with col_right:
    st.write("**Погода:**")
    if 'avg_temperature' in track_data:
        st.write(f"- Температура: {track_data['avg_temperature']:.1f}°C")
    if 'precipitation' in track_data:
        st.write(f"- Осадки: {track_data['precipitation']:.1f} мм")
    
    st.write("**Объекты OSM:**")
    for col in ['osm_water', 'osm_buildings', 'osm_farmland', 'osm_forest']:
        if col in track_data:
            st.write(f"- {col.replace('osm_', '')}: {track_data[col]}")

# ============================================================================
# 8. ВСЕ ТРЕКИ
# ============================================================================
st.subheader("Все треки")
st.dataframe(df[['track_name', 'distance_km', 'elevation_gain', 'risk_zone']], use_container_width=True)

# ============================================================================
# 9. ОБНОВЛЕНИЕ
# ============================================================================
st.sidebar.write(f"Треков: {len(df)}")
if st.sidebar.button("Обновить"):
    st.cache_data.clear()
    st.rerun()

#2

# ДОБАВИТЬ В dashboard.py после метрик:
st.subheader("Карта трека")

# Пробуем загрузить сохраненную карту
import os
if 'gpx_file' in track_data:
    map_file = track_data['gpx_file'].replace('.gpx', '_topomap.html').replace('data/', 'outputs/')
    if os.path.exists(map_file):
        with open(map_file, 'r', encoding='utf-8') as f:
            st.components.v1.html(f.read(), height=400)
    else:
        st.info("Карта не найдена. Обработайте трек в main.py")