from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model_data = joblib.load('outputs/model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[data['distance_km'], data['elevation_gain'], data['avg_slope']]])
    features_scaled = model_data['scaler'].transform(features)
    cluster = int(model_data['kmeans'].predict(features_scaled)[0])
    
    return jsonify({
        'cluster': cluster,
        'cluster_name': model_data['cluster_names'][cluster],
        'features_used': model_data['features']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)