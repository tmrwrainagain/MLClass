import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine

st.set_page_config(page_title="Track Analyzer", layout="wide")
st.title("🏔️ Анализ туристических треков")

#cайдбар
with st.sidebar:
    st.header("Введите параметры трека")
    distance = st.number_input("Дистанция (км)", 0.0, 100.0, 15.0)
    elevation = st.number_input("Набор высоты (м)", 0, 5000, 800)
    slope = st.number_input("Средний уклон (%)", 0.0, 50.0, 5.0)
    
    if st.button("Анализировать", type="primary"):
        response = requests.post('http://localhost:5000/predict',
                               json={'distance_km': distance,
                                     'elevation_gain': elevation,
                                     'avg_slope': slope})
        if response.status_code == 200:
            st.session_state.result = response.json()

#main space
if 'result' in st.session_state:
    result = st.session_state.result
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Кластер", result['cluster_name'])
    with col2:
        st.metric("ID кластера", result['cluster'])
    with col3:
        st.metric("Признаков", len(result['features_used']))
    
    #VUSIAL
    st.subheader("Визуализация")
    
    # db
    engine = create_engine('postgresql://user:password@localhost:5432/tracks')
    df = pd.read_sql("SELECT * FROM tracks_with_clusters", engine)
    
    # scatter
    fig = px.scatter(df, x='distance_km', y='elevation_gain',
                     color='cluster', hover_data=['track_name'],
                     title='Все треки в базе данных')
    
    # new track
    fig.add_scatter(x=[distance], y=[elevation],
                    mode='markers', name='Ваш трек',
                    marker=dict(size=15, color='red', symbol='star'))
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Введите параметры трека и нажмите 'Анализировать'")

# show graphics
st.subheader("Результаты анализа")
col1, col2 = st.columns(2)
with col1:
    st.image('outputs/track_map.png', caption='Карта трека')
with col2:
    st.image('outputs/distributions.png', caption='Распределения признаков')

col3, col4 = st.columns(2)
with col3:
    st.image('outputs/optimal_k.png', caption='Выбор оптимального k')
with col4:
    st.image('outputs/feature_importance.png', caption='Важность признаков')