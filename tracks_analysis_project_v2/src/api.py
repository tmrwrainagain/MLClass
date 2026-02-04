from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

app = Flask(__name__)

DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

kmeans_model = joblib.load('outputs/models/kmeans_model.pkl')
scaler = joblib.load('outputs/models/scaler.pkl')
ml_model = joblib.load('outputs/models/best_classifier.pkl')
le = joblib.load('outputs/models/label_encoder.pkl')

@app.route('/predict_cluster', methods=['POST'])
def predict_cluster():
    data = request.json
    
    features = np.array([[
        data['distance_km'],
        data['elevation_gain'],
        data['avg_temperature'],
        data['osm_buildings'],
        data['osm_water'],
        data['precipitation']
    ]])
    
    features_scaled = scaler.transform(features)
    cluster = int(kmeans_model.predict(features_scaled)[0])
    
    return jsonify({'cluster': cluster})

@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    data = request.json
    
    features = np.array([[
        data['distance_km'],
        data['elevation_gain'],
        data['avg_slope'],
        data['max_elevation'],
        data['min_elevation'],
        data['avg_elevation'],
        data['osm_water'],
        data['osm_buildings'],
        data['osm_farmland'],
        data['osm_forest'],
        data['avg_temperature'],
        data['max_temperature'],
        data['min_temperature'],
        data['precipitation']
    ]])
    
    pred = ml_model.predict(features)[0]
    risk = le.inverse_transform([pred])[0]
    
    return jsonify({'risk': risk})

@app.route('/db_stats', methods=['GET'])
def db_stats():
    df = pd.read_sql("SELECT * FROM tracks", engine)
    
    return jsonify({
        'tracks_count': len(df),
        'avg_distance': float(df['distance_km'].mean()),
        'avg_elevation': float(df['elevation_gain'].mean()),
        'risk_distribution': df['risk_zone'].value_counts().to_dict()
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)