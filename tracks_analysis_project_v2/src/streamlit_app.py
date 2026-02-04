import streamlit as st
import pandas as pd
import requests
import numpy as np
from sqlalchemy import create_engine
import plotly.express as px

st.set_page_config(page_title="GPS Predictor", layout="wide")
st.title("📍 Анализ и прогноз GPS треков")

DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

@st.cache_data
def load_tracks():
    return pd.read_sql("SELECT * FROM tracks", engine)

df = load_tracks()

tab1, tab2 = st.tabs(["📊 Треки из БД", "➕ Новый трек"])

with tab1:
    if not df.empty:
        st.header("Треки из базы данных")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Всего треков", len(df))
        with col2:
            st.metric("Средняя дистанция", f"{df['distance_km'].mean():.1f} км")
        with col3:
            st.metric("Средняя высота", f"{df['elevation_gain'].mean():.0f} м")
        
        selected = st.selectbox("Выберите трек:", df['track_name'].unique())
        
        if selected:
            track = df[df['track_name'] == selected].iloc[0]
            
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.subheader("Параметры трека")
                st.write(f"**Дистанция:** {track['distance_km']:.1f} км")
                st.write(f"**Набор высоты:** {track['elevation_gain']:.0f} м")
                st.write(f"**Макс высота:** {track.get('max_elevation', 'N/A')} м")
                st.write(f"**Мин высота:** {track.get('min_elevation', 'N/A')} м")
                st.write(f"**Средняя высота:** {track.get('avg_elevation', 'N/A')} м")
                st.write(f"**Уклон:** {track.get('avg_slope', 'N/A')}%")
            
            with col_info2:
                st.subheader("Риски и объекты")
                st.write(f"**Риск:** {track.get('risk_zone', 'normal')}")
                st.write(f"**Температура:** {track.get('avg_temperature', 'N/A')}°C")
                st.write(f"**Осадки:** {track.get('precipitation', 'N/A')} мм")
                st.write(f"**Водоемы:** {track.get('osm_water', 0)}")
                st.write(f"**Здания:** {track.get('osm_buildings', 0)}")
                st.write(f"**Лес:** {track.get('osm_forest', 0)}")
                st.write(f"**Поля:** {track.get('osm_farmland', 0)}")
            
            st.subheader("Все треки на графике")
            fig = px.scatter(
                df,
                x='distance_km',
                y='elevation_gain',
                color='risk_zone',
                size='avg_temperature',
                hover_data=['track_name'],
                title="Треки из БД"
            )
            fig.add_scatter(
                x=[track['distance_km']],
                y=[track['elevation_gain']],
                mode='markers',
                marker=dict(size=20, color='red', symbol='star'),
                name='Выбранный трек'
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Создать новый трек")
    
    st.subheader("1. Основные параметры")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_distance = st.number_input("Дистанция (км)", min_value=0.1, value=10.0, key='new_dist')
        new_elevation = st.number_input("Набор высоты (м)", min_value=0, value=500, key='new_elev')
        new_slope = st.number_input("Уклон (%)", min_value=0.0, value=5.0, key='new_slope')
    
    with col2:
        new_temp = st.number_input("Температура (°C)", value=20.0, key='new_temp')
        new_rain = st.number_input("Осадки (мм)", min_value=0.0, value=2.0, key='new_rain')
        max_temp = st.number_input("Макс температура", value=25.0, key='max_temp')
    
    with col3:
        min_temp = st.number_input("Мин температура", value=15.0, key='min_temp')
        max_elev = st.number_input("Макс высота (м)", value=700, key='max_elev')
        min_elev = st.number_input("Мин высота (м)", value=100, key='min_elev')
    
    st.subheader("2. OSM объекты вокруг")
    
    col_osm1, col_osm2 = st.columns(2)
    
    with col_osm1:
        new_water = st.number_input("Водоемы", min_value=0, value=2, key='new_water')
        new_buildings = st.number_input("Здания", min_value=0, value=5, key='new_buildings')
    
    with col_osm2:
        new_forest = st.number_input("Лес", min_value=0, value=3, key='new_forest')
        new_farmland = st.number_input("Поля", min_value=0, value=1, key='new_farmland')
    
    st.subheader("3. Прогноз")
    
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        if st.button("🔮 Предсказать кластер", use_container_width=True):
            try:
                response = requests.post(
                    "http://localhost:5000/predict_cluster",
                    json={
                        'distance_km': new_distance,
                        'elevation_gain': new_elevation,
                        'avg_temperature': new_temp,
                        'osm_buildings': new_buildings,
                        'osm_water': new_water,
                        'precipitation': new_rain
                    },
                    timeout=3
                )
                if response.status_code == 200:
                    cluster = response.json().get('cluster')
                    st.success(f"**Кластер:** {cluster}")
                    
                    cluster_desc = {
                        0: "Равнинный, низкий риск",
                        1: "Горный, средний риск",
                        2: "Сложный, высокий риск",
                        3: "Смешанный тип"
                    }
                    st.info(cluster_desc.get(cluster, "Стандартный маршрут"))
            except:
                st.error("API недоступен")
    
    with col_pred2:
        if st.button("⚠️ Предсказать риск", use_container_width=True):
            try:
                response = requests.post(
                    "http://localhost:5000/predict_risk",
                    json={
                        'distance_km': new_distance,
                        'elevation_gain': new_elevation,
                        'avg_slope': new_slope,
                        'max_elevation': max_elev,
                        'min_elevation': min_elev,
                        'avg_elevation': (max_elev + min_elev) / 2,
                        'osm_water': new_water,
                        'osm_buildings': new_buildings,
                        'osm_farmland': new_farmland,
                        'osm_forest': new_forest,
                        'avg_temperature': new_temp,
                        'max_temperature': max_temp,
                        'min_temperature': min_temp,
                        'precipitation': new_rain
                    },
                    timeout=3
                )
                if response.status_code == 200:
                    risk = response.json().get('risk')
                    
                    risk_colors = {
                        'normal': '🟢',
                        'fire_risk': '🔴',
                        'flood_risk': '🔵',
                        'evacuation_hard': '🟡'
                    }
                    
                    st.success(f"{risk_colors.get(risk, '⚪')} **Риск:** {risk}")
                    
                    recommendations = {
                        'fire_risk': "Избегайте сухой растительности",
                        'flood_risk': "Проверьте прогноз",
                        'evacuation_hard': "Возьмите спутниковый телефон",
                        'normal': "Маршрут безопасен"
                    }
                    
                    st.info(recommendations.get(risk, "Нет рекомендаций"))
            except:
                st.error("API недоступен")
    
    st.subheader("4. Сохранить в БД")
    
    new_name = st.text_input("Название трека", "Новый трек")
    
    if st.button("💾 Сохранить в базу данных", use_container_width=True):
        new_data = pd.DataFrame([{
            'track_name': new_name,
            'distance_km': new_distance,
            'elevation_gain': new_elevation,
            'avg_slope': new_slope,
            'max_elevation': max_elev,
            'min_elevation': min_elev,
            'avg_elevation': (max_elev + min_elev) / 2,
            'osm_water': new_water,
            'osm_buildings': new_buildings,
            'osm_farmland': new_farmland,
            'osm_forest': new_forest,
            'avg_temperature': new_temp,
            'max_temperature': max_temp,
            'min_temperature': min_temp,
            'precipitation': new_rain,
            'risk_zone': 'manual_input',
            'gpx_file': 'manual',
            'processed_date': pd.Timestamp.now()
        }])
        
        try:
            new_data.to_sql('tracks', engine, if_exists='append', index=False)
            st.success(f"Трек '{new_name}' сохранен в БД!")
            st.cache_data.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Ошибка сохранения: {e}")

st.sidebar.header("ℹ️ Информация")
st.sidebar.write("""
1. **Треки из БД** - смотрите сохраненные треки
2. **Новый трек** - вводите параметры и получайте прогноз
3. Сохраняйте новые треки в БД
""")

st.sidebar.header("📈 Статистика")
if not df.empty:
    st.sidebar.write(f"**Треков в БД:** {len(df)}")
    
    if 'risk_zone' in df.columns:
        risk_counts = df['risk_zone'].value_counts()
        st.sidebar.write("**Распределение рисков:**")
        for risk, count in risk_counts.head(5).items():
            st.sidebar.write(f"- {risk}: {count}")

st.sidebar.header("🔧 Настройки")
if st.sidebar.button("🔄 Обновить данные из БД"):
    st.cache_data.clear()
    st.rerun()

api_status = st.sidebar.empty()
if st.sidebar.button("🔍 Проверить API"):
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            api_status.success("API работает")
        else:
            api_status.error("API не отвечает")
    except:
        api_status.error("API недоступен")