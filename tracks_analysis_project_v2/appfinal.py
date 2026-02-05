import streamlit as st
import requests
import folium
from streamlit_folium import folium_static
from datetime import datetime
import pandas as pd

st.set_page_config(page_title="Карта рисков", layout="wide")
st.title("⚠️ Карта опасных участков маршрута")

API_URL = "http://localhost:5000"

# Сайдбар
st.sidebar.header("🗓️ Выбор даты")
selected_date = st.sidebar.date_input(
    "Дата прохождения маршрута",
    datetime.now()
)

# Загружаем треки из БД для карты
try:
    response = requests.get(f"{API_URL}/get_tracks", timeout=3)
    if response.status_code == 200:
        tracks = response.json()
        df_tracks = pd.DataFrame(tracks)
        
        st.sidebar.write(f"Треков в БД: {len(df_tracks)}")
    else:
        df_tracks = pd.DataFrame()
        st.sidebar.warning("Не удалось загрузить треки")
except:
    df_tracks = pd.DataFrame()

# Основная часть: Карта
st.header("1. Карта маршрута с зонами риска")

# Создаем карту
m = folium.Map(location=[55.7558, 37.6173], zoom_start=5)

# Добавляем треки из БД
if not df_tracks.empty:
    for _, track in df_tracks.iterrows():
        # Координаты (примерные, нужно чтобы в БД были lat/lon)
        lat = 55.7558 + (track['id'] * 0.1) % 5
        lon = 37.6173 + (track['id'] * 0.1) % 5
        
        # Цвет по существующему риску
        risk = track.get('risk_zone', 'normal')
        if 'fire' in str(risk):
            color = 'red'
            icon = 'fire'
        elif 'flood' in str(risk):
            color = 'blue'
            icon = 'tint'
        elif 'evacuation' in str(risk):
            color = 'orange'
            icon = 'warning-sign'
        else:
            color = 'green'
            icon = 'ok-circle'
        
        folium.Marker(
            [lat, lon],
            popup=f"{track['track_name']}<br>Дистанция: {track['distance_km']} км<br>Риск: {risk}",
            icon=folium.Icon(color=color, icon=icon, prefix='glyphicon')
        ).add_to(m)

# Показываем карту
folium_static(m, width=1000, height=500)

# 2. Предсказание по координатам
st.header("2. Предсказание опасности по координатам")

col1, col2 = st.columns(2)

with col1:
    lat = st.number_input("Широта", value=55.7558)
    lon = st.number_input("Долгота", value=37.6173)

with col2:
    st.write(f"Дата: {selected_date}")
    elevation = st.number_input("Высота над уровнем моря (м)", value=150)

if st.button("🔍 Определить тип опасности", use_container_width=True):
    try:
        response = requests.post(
            f"{API_URL}/predict_danger",
            json={
                'lat': lat,
                'lon': lon,
                'date': selected_date.strftime('%Y-%m-%d')
            },
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            risk_type = result['risk_type']
            
            # Показываем результат
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                if risk_type == 'fire_risk':
                    st.error("🔥 **ПОЖАРООПАСНОСТЬ**")
                    st.write("Высокая температура, сухая растительность")
                elif risk_type == 'flood_risk':
                    st.error("🌊 **РИСК ЗАТОПЛЕНИЯ**")
                    st.write("Обильные осадки, близость водоемов")
                elif risk_type == 'evacuation_hard':
                    st.warning("⛰️ **СЛОЖНАЯ ЭВАКУАЦИЯ**")
                    st.write("Горный рельеф, труднодоступность")
                else:
                    st.success("✅ **НОРМАЛЬНЫЙ УРОВЕНЬ РИСКА**")
                    st.write("Благоприятные условия")
            
            with col_r2:
                params = result.get('parameters_used', {})
                st.write("**Использованные параметры:**")
                st.write(f"- Температура: {params.get('temperature', 0)}°C")
                st.write(f"- Осадки: {params.get('precipitation', 0)} мм")
                st.write(f"- Высота: {params.get('elevation', 0)} м")
                
                st.write(f"**Координаты:** {lat}, {lon}")
                st.write(f"**Дата:** {selected_date}")
                
        else:
            st.error("Ошибка API")
    except Exception as e:
        st.error(f"API недоступен: {e}")

# 3. Оценка сложности эвакуации
st.header("3. Оценка сложности эвакуации")

col_e1, col_e2, col_e3 = st.columns(3)

with col_e1:
    slope = st.number_input("Средний уклон местности (%)", min_value=0.0, value=12.0)

with col_e2:
    buildings = st.number_input("Здания поблизости", min_value=0, value=3)

with col_e3:
    if st.button("📊 Оценить сложность эвакуации"):
        try:
            response = requests.post(
                f"{API_URL}/evacuation_difficulty",
                json={
                    'elevation_gain': elevation,
                    'avg_slope': slope,
                    'osm_buildings': buildings
                },
                timeout=3
            )
            
            if response.status_code == 200:
                result = response.json()
                difficulty = result['evacuation_difficulty']
                
                if difficulty == 'high':
                    st.error("🔴 **ВЫСОКАЯ сложность**")
                    st.write(result['reason'])
                    st.warning("Требуется: вертолет, спасатели")
                elif difficulty == 'medium':
                    st.warning("🟡 **СРЕДНЯЯ сложность**")
                    st.write(result['reason'])
                    st.info("Требуется: внедорожник, аптечка")
                else:
                    st.success("🟢 **НИЗКАЯ сложность**")
                    st.write(result['reason'])
                    st.info("Требуется: стандартная аптечка")
                    
        except:
            st.error("API недоступен")

# 4. Справочная информация по трекам
st.header("4. Справочная информация по трекам")

if not df_tracks.empty:
    selected_track = st.selectbox(
        "Выберите трек для подробной информации:",
        df_tracks['track_name'].tolist()
    )
    
    if selected_track:
        track_info = df_tracks[df_tracks['track_name'] == selected_track].iloc[0]
        
        col_i1, col_i2 = st.columns(2)
        
        with col_i1:
            st.write("**Основные параметры:**")
            st.write(f"- Дистанция: {track_info['distance_km']} км")
            st.write(f"- Набор высоты: {track_info['elevation_gain']} м")
            st.write(f"- Температура: {track_info.get('avg_temperature', 'N/A')}°C")
            st.write(f"- Осадки: {track_info.get('precipitation', 'N/A')} мм")
        
        with col_i2:
            st.write("**Текущая оценка риска:**")
            risk = track_info.get('risk_zone', 'normal')
            if 'fire' in str(risk):
                st.write("🔥 Пожарная опасность")
            elif 'flood' in str(risk):
                st.write("🌊 Риск затопления")
            elif 'evacuation' in str(risk):
                st.write("⛰️ Сложная эвакуация")
            else:
                st.write("✅ Нормальный уровень")

# Информация
st.sidebar.header("ℹ️ Информация")
st.sidebar.write("""
**Типы опасностей:**
- 🔥 Пожар (лето, высокая температура)
- 🌊 Затопление (весна, осадки)
- ⛰️ Сложная эвакуация (горы)

**Как использовать:**
1. Выберите дату на карте
2. Введите координаты для предсказания
3. Оцените сложность эвакуации
4. Посмотрите информацию по трекам
""")

# Проверка
if st.sidebar.button("Проверить API"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            st.sidebar.success("✅ API работает")
        else:
            st.sidebar.error("❌ API ошибка")
    except:
        st.sidebar.error("❌ API недоступен")
