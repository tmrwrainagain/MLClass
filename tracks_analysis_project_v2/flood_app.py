import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="Прогноз затоплений", layout="wide")
st.title("🌊 Прогноз зон затоплений на год")

# Подключение к API прогнозов
API_URL = "http://localhost:5001"

col1, col2 = st.columns(2)

with col1:
    if st.button("📈 Получить прогноз на 30 дней"):
        try:
            response = requests.get(f"{API_URL}/flood_forecast", timeout=5)
            if response.status_code == 200:
                data = response.json()
                forecast = data['forecast']
                
                # Таблица
                df = pd.DataFrame(forecast)
                st.dataframe(df, use_container_width=True)
                
                # График уровня воды
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['water_level'],
                    mode='lines+markers',
                    name='Уровень воды',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title='Прогноз уровня воды на 30 дней',
                    xaxis_title='Дата',
                    yaxis_title='Уровень воды',
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("API не отвечает")
        except:
            st.error("Сервер прогнозов не запущен")

with col2:
    if st.button("📊 Статистика затоплений"):
        try:
            response = requests.get(f"{API_URL}/flood_stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                
                st.metric("Всего дней в истории", stats['total_days'])
                st.metric("Дней высокого риска", stats['high_risk_days'])
                st.metric("Дней среднего риска", stats['medium_risk_days'])
                st.metric("Макс уровень воды", f"{stats['max_water_level']:.1f}")
                st.metric("Макс осадки", f"{stats['max_precipitation']:.1f} мм")
            else:
                st.error("API не отвечает")
        except:
            st.error("Сервер прогнозов не запущен")

# Ручной прогноз
st.header("🔮 Ручной прогноз")

col_input1, col_input2 = st.columns(2)

with col_input1:
    precipitation = st.number_input("Ожидаемые осадки (мм)", min_value=0.0, value=50.0)
    current_water = st.number_input("Текущий уровень воды", min_value=0.0, value=40.0)

with col_input2:
    days_ahead = st.slider("Прогноз на дней вперед", 7, 365, 30)
    zone_type = st.selectbox("Тип зоны", ["Городская", "Сельская", "Прибрежная"])

if st.button("Рассчитать риск"):
    # Простая логика
    if precipitation > 70 and current_water > 60:
        risk = "🔴 ВЫСОКИЙ РИСК"
        advice = "Эвакуация рекомендуется"
    elif precipitation > 50 and current_water > 40:
        risk = "🟡 СРЕДНИЙ РИСК"
        advice = "Мониторинг ситуации"
    else:
        risk = "🟢 НИЗКИЙ РИСК"
        advice = "Ситуация стабильна"
    
    st.success(f"**Уровень риска:** {risk}")
    st.info(f"**Рекомендация:** {advice}")
    
    # Простой график
    dates = []
    levels = []
    
    today = datetime.now()
    for i in range(days_ahead):
        date = today.replace(day=today.day + i)
        dates.append(date.strftime("%d.%m"))
        # Простая модель: уровень зависит от осадков
        level = current_water + (precipitation * 0.01 * i)
        levels.append(level)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=levels, mode='lines', name='Прогноз'))
    fig.update_layout(title=f'Прогноз уровня воды на {days_ahead} дней',
                     xaxis_title='Дата', yaxis_title='Уровень воды',
                     height=300)
    
    st.plotly_chart(fig, use_container_width=True)

# Информация
st.sidebar.header("ℹ️ Информация")
st.sidebar.write("""
**Как использовать:**
1. Получите прогноз из базы данных
2. Посмотрите статистику прошлых затоплений  
3. Сделайте ручной прогноз по параметрам

**Запустите сервер прогнозов:**
```bash
python flood_api.py
