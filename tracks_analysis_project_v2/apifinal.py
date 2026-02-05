from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime

app = Flask(__name__)

# Загружаем модель
MODEL_PATH = 'outputs/models/best_classifier.pkl'
SCALER_PATH = 'outputs/models/scaler.pkl'
LABEL_PATH = 'outputs/models/label_encoder.pkl'

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_PATH)

# БД
DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

print("✅ API готов: модель + БД")

@app.route('/predict_danger', methods=['POST'])
def predict_danger():
    """Определение типа опасности по координатам и дате"""
    data = request.json
    
    # Получаем данные
    lat = float(data.get('lat', 55.7558))
    lon = float(data.get('lon', 37.6173))
    date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    # Преобразуем дату
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    month = date_obj.month
    
    # Определяем сезонные параметры
    if month in [12, 1, 2]:  # зима
        temp = -5
        rain = 30
    elif month in [3, 4, 5]:  # весна
        temp = 10
        rain = 50  # больше дождей весной
    elif month in [6, 7, 8]:  # лето
        temp = 25
        rain = 20
    else:  # осень
        temp = 15
        rain = 40
    
    # Координаты влияют на параметры
    # Север = холоднее, юг = теплее
    if lat > 55:  # север
        temp -= 5
        rain += 10
    else:  # юг
        temp += 5
        rain -= 5
    
    # Высота по координатам (примерно)
    if 55 < lat < 56 and 37 < lon < 38:  # Москва
        elevation = 150
        water = 3
        forest = 2
    elif 43 < lat < 44 and 39 < lon < 40:  # Сочи
        elevation = 50
        water = 5
        forest = 8
    else:  # горы
        elevation = 800
        water = 1
        forest = 6
    
    # Формируем признаки для модели
    features = np.array([[
        15.0,  # distance_km
        elevation,  # elevation_gain
        8.0,  # avg_slope
        elevation + 300,  # max_elevation
        100,  # min_elevation
        elevation/2,  # avg_elevation
        water,  # osm_water
        5,  # osm_buildings
        2,  # osm_farmland
        forest,  # osm_forest
        temp,  # avg_temperature
        temp + 5,  # max_temperature
        temp - 5,  # min_temperature
        rain  # precipitation
    ]])
    
    # Предсказываем
    features_scaled = scaler.transform(features)
    pred_num = model.predict(features_scaled)[0]
    risk_type = label_encoder.inverse_transform([pred_num])[0]
    
    return jsonify({
        'coordinates': {'lat': lat, 'lon': lon},
        'date': date_str,
        'risk_type': risk_type,
        'parameters_used': {
            'temperature': temp,
            'precipitation': rain,
            'elevation': elevation
        }
    })

@app.route('/evacuation_difficulty', methods=['POST'])
def evacuation_difficulty():
    """Оценка сложности эвакуации по параметрам"""
    data = request.json
    
    elevation = float(data.get('elevation_gain', 0))
    slope = float(data.get('avg_slope', 0))
    buildings = int(data.get('osm_buildings', 0))
    
    # Логика оценки
    score = 0
    
    if elevation > 1000:
        score += 3
    elif elevation > 500:
        score += 2
    elif elevation > 200:
        score += 1
    
    if slope > 20:
        score += 3
    elif slope > 10:
        score += 2
    elif slope > 5:
        score += 1
    
    if buildings < 5:
        score += 2  # мало зданий = сложнее эвакуация
    
    # Определяем сложность
    if score >= 5:
        difficulty = 'high'
        reason = 'Высокогорье, крутые склоны'
    elif score >= 3:
        difficulty = 'medium'
        reason = 'Горная местность'
    else:
        difficulty = 'low'
        reason = 'Равнинная местность'
    
    return jsonify({
        'evacuation_difficulty': difficulty,
        'reason': reason,
        'score': score,
        'parameters': {
            'elevation_gain': elevation,
            'avg_slope': slope,
            'buildings_nearby': buildings
        }
    })

@app.route('/get_tracks', methods=['GET'])
def get_tracks():
    """Получить все треки для карты"""
    df = pd.read_sql("""
        SELECT id, track_name, distance_km, elevation_gain, 
               avg_temperature, precipitation, risk_zone
        FROM tracks
    """, engine)
    return jsonify(df.to_dict('records'))

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
