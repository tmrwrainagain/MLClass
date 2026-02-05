from flask import Flask, jsonify
from sqlalchemy import create_engine
import pandas as pd
import json

app = Flask(__name__)

DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

@app.route('/flood_forecast', methods=['GET'])
def flood_forecast():
    df = pd.read_sql("""
        SELECT date, zone_id, water_level, flood_risk 
        FROM flood_predictions 
        ORDER BY date 
        LIMIT 30
    """, engine)
    
    return jsonify({
        'forecast': df.to_dict('records'),
        'count': len(df)
    })

@app.route('/flood_stats', methods=['GET'])
def flood_stats():
    df = pd.read_sql("SELECT * FROM flood_history", engine)
    
    high_risk_days = len(df[df['flood_risk'] == 'high'])
    medium_risk_days = len(df[df['flood_risk'] == 'medium'])
    
    return jsonify({
        'total_days': len(df),
        'high_risk_days': high_risk_days,
        'medium_risk_days': medium_risk_days,
        'max_precipitation': float(df['precipitation'].max()),
        'max_water_level': float(df['water_level'].max())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
