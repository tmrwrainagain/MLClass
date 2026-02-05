import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Прогноз рисков", layout="wide")
st.title("⚠️ Прогноз типов рисков по GPS трекам")

# URL API
API_URL = "http://localhost:5000"

st.header("1. Введите параметры трека")

col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Дистанция маршрута (км)", min_value=0.1, value=15.0)
    elevation = st.number_input("Набор высоты (м)", min_value=0, value=800)
    slope = st.number_input("Средний уклон (%)", min_value=0.0, value=8.0)
    
    temp = st.number_input("Температура (°C)", value=28.0)
    rain = st.number_input("Осадки (мм)", min_value=0.0, value=5.0)

with col2:
    water = st.number_input("Водоемы рядом (OSM)", min_value=0, value=3)
    buildings = st.number_input("Здания рядом (OSM)", min_value=0, value=2)
    forest = st.number_input("Лесная зона (OSM)", min_value=0, value=8)
    farmland = st.number_input("Сельхоз угодья (OSM)", min_value=0, value=4)
    
    max_elev = st.number_input("Макс высота (м)", value=elevation + 300)
    min_elev = st.number_input("Мин высота (м)", value=200)

# Кнопка предсказания
if st.button("🔮 Предсказать тип риска", use_container_width=True):
    try:
        response = requests.post(
            f"{API_URL}/predict_risk",
            json={
                'distance_km': distance,
                'elevation_gain': elevation,
                'avg_slope': slope,
                'max_elevation': max_elev,
                'min_elevation': min_elev,
                'avg_elevation': (max_elev + min_elev) / 2,
                'osm_water': water,
                'osm_buildings': buildings,
                'osm_farmland': farmland,
                'osm_forest': forest,
                'avg_temperature': temp,
                'max_temperature': temp + 7,
                'min_temperature': temp - 7,
                'precipitation': rain
            },
            timeout=5
        )
        
        if response.status_code == 200:
            result = response.json()
            risk_type = result.get('risk', 'normal')
            probability = result.get('probability', 0)
            
            # Отображаем в зависимости от типа риска
            if risk_type == 'fire_risk':
                st.error(f"🔥 **ПОЖАРООПАСНОСТЬ**")
                st.write("Высокая температура + лес/поля = риск пожара")
                
                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    st.metric("Температура", f"{temp}°C", "+")
                with col_f2:
                    st.metric("Лес/поля", f"{forest + farmland} объектов", "")
                    
                st.warning("""
                **Рекомендации:**
                - Избегайте маршрутов с сухой растительностью
                - Имейте средства пожаротушения
                - Сообщите маршрут МЧС
                """)
                
            elif risk_type == 'flood_risk':
                st.error(f"🌊 **РИСК ЗАТОПЛЕНИЯ**")
                st.write("Осадки + водоемы = риск наводнения")
                
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    st.metric("Осадки", f"{rain} мм", "+")
                with col_w2:
                    st.metric("Водоемы", f"{water} объектов", "")
                
                st.warning("""
                **Рекомендации:**
                - Проверьте прогноз погоды
                - Избегайте низменностей и русла рек
                - Имейте план эвакуации
                """)
                
            elif risk_type == 'evacuation_hard':
                st.warning(f"⛰️ **СЛОЖНАЯ ЭВАКУАЦИЯ**")
                st.write("Большой набор высоты + крутой уклон")
                
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.metric("Набор высоты", f"{elevation} м", "+")
                with col_e2:
                    st.metric("Уклон", f"{slope}%", "")
                
                st.info("""
                **Рекомендации:**
                - Возьмите спутниковый телефон
                - Сообщите точный маршрут
                - Имейте запас на 2+ дня
                """)
                
            else:  # normal
                st.success(f"✅ **НОРМАЛЬНЫЙ УРОВЕНЬ РИСКА**")
                st.write("Маршрут безопасен")
                
                st.info("""
                **Рекомендации:**
                - Следите за изменениями погоды
                - Сообщите о маршруте
                - Возьмите базовый набор первой помощи
                """)
            
            # Вероятность
            st.metric("Вероятность предсказания", f"{probability:.1%}")
            
        else:
            st.error("Ошибка API")
    except Exception as e:
        st.error(f"API недоступен: {e}")

# Визуализация
st.header("2. Визуализация факторов риска")

fig = go.Figure()

# Температура vs Осадки
fig.add_trace(go.Scatter(
    x=[temp],
    y=[rain],
    mode='markers',
    marker=dict(size=30, color='red'),
    name='Ваш трек'
))

# Зоны рисков
fig.add_trace(go.Scatter(
    x=[30, 35, 25, 20],  # Температура
    y=[5, 2, 15, 25],    # Осадки
    mode='markers',
    marker=dict(size=20, color=['red', 'red', 'blue', 'blue']),
    text=['fire', 'fire', 'flood', 'flood'],
    name='Зоны рисков'
))

fig.update_layout(
    title='Температура vs Осадки (зоны рисков)',
    xaxis_title='Температура (°C)',
    yaxis_title='Осадки (мм)',
    height=300
)

st.plotly_chart(fig, use_container_width=True)

# Статистика
st.header("3. Статистика по типам рисков")

if st.button("📊 Загрузить статистику из БД"):
    try:
        response = requests.get(f"{API_URL}/db_stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            
            col_s1, col_s2, col_s3 = st.columns(3)
            
            with col_s1:
                st.metric("Всего треков", stats['tracks_count'])
            
            with col_s2:
                st.metric("Средняя дистанция", f"{stats['avg_distance']:.1f} км")
            
            with col_s3:
                st.metric("Средний набор высоты", f"{stats['avg_elevation']:.0f} м")
            
            # Распределение рисков
            risk_dist = stats.get('risk_distribution', {})
            if risk_dist:
                st.write("**Распределение типов рисков:**")
                for risk_type, count in risk_dist.items():
                    if risk_type == 'fire_risk':
                        icon = '🔥'
                    elif risk_type == 'flood_risk':
                        icon = '🌊'
                    elif risk_type == 'evacuation_hard':
                        icon = '⛰️'
                    else:
                        icon = '✅'
                    
                    st.write(f"{icon} {risk_type}: {count} треков")
        else:
            st.error("Ошибка загрузки статистики")
    except:
        st.error("API недоступен")

# Переобучение модели
st.sidebar.header("🔄 Управление моделью")

if st.sidebar.button("🔄 Переобучить модель"):
    try:
        response = requests.post(f"{API_URL}/retrain")
        if response.status_code == 200:
            st.sidebar.success("Модель переобучена")
        else:
            st.sidebar.error("Ошибка переобучения")
    except:
        st.sidebar.error("API недоступен")

# Информация о модели
st.sidebar.header("ℹ️ Информация")
st.sidebar.write("""
**Типы рисков:**
- 🔥 **fire_risk** - высокая температура + лес/поля
- 🌊 **flood_risk** - осадки + водоемы  
- ⛰️ **evacuation_hard** - большой набор высоты + уклон
- ✅ **normal** - низкий риск

**Модель использует:**
- RandomForest/SVM/KNN
- Автодообучение при новых данных
- Версионное хранение
""")

# Проверка API
if st.sidebar.button("🔍 Проверить API"):
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            st.sidebar.success("✅ API работает")
        else:
            st.sidebar.error("❌ API не отвечает")
    except:
        st.sidebar.error("❌ API недоступен")
