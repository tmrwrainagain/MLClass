from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Загружаем модели
kmeans_model = joblib.load('outputs/models/kmeans_model.pkl')
scaler = joblib.load('outputs/models/scaler.pkl')
feature_importance = joblib.load('outputs/models/feature_importance.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """API для предсказания кластера трека"""
    data = request.json
    
    # Формируем признаки
    features = np.array([[
        data.get('distance_km', 0),
        data.get('elevation_gain', 0),
        data.get('avg_temperature', 0),
        data.get('osm_buildings', 0),
        data.get('osm_water', 0),
        data.get('precipitation', 0)
    ]])
    
    # Масштабируем
    features_scaled = scaler.transform(features)
    
    # Предсказываем кластер
    cluster = int(kmeans_model.predict(features_scaled)[0])
    
    return jsonify({
        'cluster': cluster,
        'features_used': list(feature_importance['feature'].head(5)),
        'message': f'Трек отнесен к кластеру {cluster}'
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)