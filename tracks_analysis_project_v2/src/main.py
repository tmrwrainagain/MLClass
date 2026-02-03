import os
import gpxpy
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime
from sqlalchemy import create_engine, text
import psycopg2
import folium
import openmeteo_requests
import requests_cache
from retry_requests import retry
import time

headers = {'User-Agent': 'TrackAnalysis/1.0 (student_project)'}


# parse
gpx_files = []
for file in os.listdir("./data/"):
    if file.endswith(".gpx"):
        gpx_files.append(f"./data/{file}")

print(f"файлов: {len(gpx_files)}") # сколько всего gpx

all_tracks_data = []

# обработка всех gpx

for gpx_file in gpx_files:
    print(f"файл: {gpx_file}")
    
    with open(gpx_file, 'r') as f:
        gpx = gpxpy.parse(f)
    
    points_data = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points_data.append({
                    'lat': point.latitude,
                    'lon': point.longitude,
                    'ele': point.elevation,
                    'time': point.time
                })
    
    df = pd.DataFrame(points_data)
    
    # метрики трека
    lat_rad = np.radians(df['lat'].values)
    lon_rad = np.radians(df['lon'].values)
    
    dlat = lat_rad[1:] - lat_rad[:-1]
    dlon = lon_rad[1:] - lon_rad[:-1]
    
    a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
    distances = 6371 * 2 * np.arcsin(np.sqrt(a)) * 1000
    total_distance = np.sum(distances)
    
    elevation_diff = np.diff(df['ele'].values)
    total_climb = np.sum(elevation_diff[elevation_diff > 0])
    
    track_metrics = {
        'distance_km': total_distance / 1000,
        'elevation_gain': total_climb,
        'avg_slope': (total_climb / total_distance * 100) if total_distance > 0 else 0,
        'max_elevation': np.max(df['ele']),
        'min_elevation': np.min(df['ele']),
        'avg_elevation': np.mean(df['ele'])
    }
    # OSM данные, точка - каждая десятая
    sample_points = df.iloc[::10] 
    print(f"sample points: {len(sample_points)}")
    
    osm_features_total = {'water': 0, 'buildings': 0, 'farmland': 0, 'forest': 0}
    
    for idx, row in sample_points.iterrows():
        lat = row['lat']
        lon = row['lon']
        
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json][timeout:10];
        (
          way["natural"="water"](around:300,{lat},{lon});
          way["building"](around:300,{lat},{lon});
          way["landuse"="farmland"](around:300,{lat},{lon});
          way["landuse"="forest"](around:300,{lat},{lon});
          way["landuse"="meadow"](around:300,{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=30, headers=headers)
        time.sleep(1)
        if response.text and response.text.strip() and response.text.strip()[0] == '{':
            osm_data = response.json()
            print('есть информация')
        else:
            print(f"Нет ответа")
            continue 
        for element in osm_data.get('elements', []):
            tags = element.get('tags', {})
            if 'natural' in tags and tags['natural'] == 'water':
                osm_features_total['water'] += 1
            elif 'building' in tags:
                osm_features_total['buildings'] += 1
            elif 'landuse' in tags:
                if tags['landuse'] == 'farmland':
                    osm_features_total['farmland'] += 1
                elif tags['landuse'] == 'forest':
                    osm_features_total['forest'] += 1
                elif tags['landuse'] == 'meadow':
                    osm_features_total['farmland'] += 1
    #получение погоды
    gpx_date = df['time'].iloc[0] if 'time' in df.columns and not pd.isna(df['time'].iloc[0]) else datetime.now()
    
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    
    sample_lat = df['lat'].iloc[0]
    sample_lon = df['lon'].iloc[0]
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": sample_lat,
        "longitude": sample_lon,
        "start_date": gpx_date.strftime("%Y-%m-%d"),
        "end_date": gpx_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,precipitation",
        "timezone": "auto"
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    hourly = response.Hourly()
    hourly_temperature = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    
    weather_data = {
        'avg_temperature': float(np.mean(hourly_temperature)),
        'max_temperature': float(np.max(hourly_temperature)),
        'min_temperature': float(np.min(hourly_temperature)),
        'precipitation': float(np.sum(hourly_precipitation))
    }

    # зона риска
    risk_rules = []
    
    if weather_data['avg_temperature'] > 25 and (osm_features_total['farmland'] > 0 or osm_features_total['forest'] > 0):
        risk_rules.append('Риск пожара. Температура, лес, сельхоз. земля.')
    
    if gpx_date.month in [3, 4, 5] and osm_features_total['water'] > 0:
        risk_rules.append('Риск наводнения')
    
    if track_metrics['elevation_gain'] > 1000 and track_metrics['avg_slope'] > 15:
        risk_rules.append('Сложно эвакуироваться. Горы.')
    
    if weather_data['precipitation'] > 10 and track_metrics['min_elevation'] < 100:
        risk_rules.append('Риск наводнения')
    
    risk_zone = ', '.join(risk_rules) if risk_rules else 'normal'

#сохранение данных
    
    track_record = {
        'track_name': gpx.tracks[0].name if gpx.tracks and gpx.tracks[0].name else os.path.basename(gpx_file),
        'gpx_file': gpx_file,
        'processed_date': datetime.now(),
        'distance_km': track_metrics['distance_km'],
        'elevation_gain': track_metrics['elevation_gain'],
        'avg_slope': track_metrics['avg_slope'],
        'max_elevation': track_metrics['max_elevation'],
        'min_elevation': track_metrics['min_elevation'],
        'avg_elevation': track_metrics['avg_elevation'],
        'osm_water': osm_features_total['water'],
        'osm_buildings': osm_features_total['buildings'],
        'osm_farmland': osm_features_total['farmland'],
        'osm_forest': osm_features_total['forest'],
        'avg_temperature': weather_data['avg_temperature'],
        'max_temperature': weather_data['max_temperature'],
        'min_temperature': weather_data['min_temperature'],
        'precipitation': weather_data['precipitation'],
        'risk_zone': risk_zone
    }
    
    all_tracks_data.append(track_record)
    
# топографическая карта

    m = folium.Map(
        location=[df['lat'].mean(), df['lon'].mean()],
        zoom_start=13,
        tiles='OpenTopoMap'
    )
    
    points = list(zip(df['lat'], df['lon']))
    folium.PolyLine(points, color='red', weight=2.5, opacity=1).add_to(m)
    
    map_file = gpx_file.replace('.gpx', '_topomap.html').replace('./data/', './outputs/')
    m.save(map_file)
    
    print(f"  {track_metrics['distance_km']:.2f} км") # дистанция
    print(f"  {track_metrics['elevation_gain']:.0f} м") # высота
    print(f"  {track_metrics['avg_elevation']:.0f} м") # avg высота
    print(f"  {weather_data['avg_temperature']:.1f}°C") # temp
    print(f"  {weather_data['precipitation']:.1f} мм") #осадки
    print(f"  вода={osm_features_total['water']}, здания={osm_features_total['buildings']}, поля/луга={osm_features_total['farmland']}, лес={osm_features_total['forest']}") #osm
    print(f" {risk_zone}") #зона риска
    print(f" {map_file}") # файл карты


#db save
DB_URL = "postgresql://username:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

# Создаем таблицу если нет
with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS tracks (
            id SERIAL PRIMARY KEY,
            track_name VARCHAR(255),
            gpx_file VARCHAR(500),
            distance_km FLOAT,
            elevation_gain FLOAT,
            avg_slope FLOAT,
            max_elevation FLOAT,
            min_elevation FLOAT,
            avg_elevation FLOAT,
            osm_water INT,
            osm_buildings INT,
            osm_farmland INT,
            osm_forest INT,
            avg_temperature FLOAT,
            max_temperature FLOAT,
            min_temperature FLOAT,
            precipitation FLOAT,
            risk_zone VARCHAR(100),
            processed_date TIMESTAMP
        )
    """))
    conn.commit()




map_filename = f"outputs/maps/{os.path.basename(gpx_file).replace('.gpx', '_map.html')}"
m.save(map_filename)
df_all_tracks = pd.DataFrame(all_tracks_data)
df_all_tracks.to_sql('tracks', engine, if_exists='append', index=False)
print(f"треков всего {len(all_tracks_data)}")