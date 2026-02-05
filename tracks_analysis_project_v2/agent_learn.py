import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

# Загружаем данные из БД
df = pd.read_sql("SELECT * FROM tracks", engine)
print(f"Треков из БД: {len(df)}")

# Авторазметка
df['label'] = 'normal'
for i in range(len(df)):
    row = df.iloc[i]
    if row['avg_temperature'] > 25 and (row['osm_farmland'] > 0 or row['osm_forest'] > 0):
        df.at[i, 'label'] = 'fire_risk'
    elif row['precipitation'] > 10 and row['osm_water'] > 0:
        df.at[i, 'label'] = 'flood_risk'
    elif row['elevation_gain'] > 1000 and row['avg_slope'] > 15:
        df.at[i, 'label'] = 'evacuation_hard'

# Генерация дополнительных данных
print("Генерирую дополнительные данные...")
n_new = 50
new_data = []

for _ in range(n_new):
    # Случайные параметры
    distance = np.random.uniform(5, 50)
    elevation = np.random.uniform(100, 2000)
    temp = np.random.uniform(-10, 35)
    rain = np.random.uniform(0, 50)
    water = np.random.randint(0, 10)
    forest = np.random.randint(0, 15)
    
    # Определяем риск
    if temp > 25 and forest > 5:
        risk = 'fire_risk'
    elif rain > 15 and water > 3:
        risk = 'flood_risk'
    elif elevation > 800:
        risk = 'evacuation_hard'
    else:
        risk = 'normal'
    
    new_data.append({
        'distance_km': distance,
        'elevation_gain': elevation,
        'avg_slope': np.random.uniform(1, 20),
        'max_elevation': elevation + np.random.uniform(100, 500),
        'min_elevation': np.random.uniform(50, 300),
        'avg_elevation': elevation / 2,
        'osm_water': water,
        'osm_buildings': np.random.randint(0, 20),
        'osm_farmland': np.random.randint(0, 10),
        'osm_forest': forest,
        'avg_temperature': temp,
        'max_temperature': temp + np.random.uniform(2, 10),
        'min_temperature': temp - np.random.uniform(2, 10),
        'precipitation': rain,
        'label': risk
    })

# Добавляем к данным из БД
df_new = pd.DataFrame(new_data)
df_combined = pd.concat([df, df_new], ignore_index=True)

print(f"Всего данных: {len(df_combined)}")
print("Распределение классов:")
print(df_combined['label'].value_counts())

# Подготовка данных
X = df_combined[[
    'distance_km', 'elevation_gain', 'avg_slope',
    'max_elevation', 'min_elevation', 'avg_elevation',
    'osm_water', 'osm_buildings', 'osm_farmland', 'osm_forest',
    'avg_temperature', 'max_temperature', 'min_temperature',
    'precipitation'
]].fillna(0)

y = df_combined['label']

# Разделение на train/test
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Три модели
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

# Предсказания
rf_pred = rf.predict(X_test)
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)

# Метрики
rf_acc = accuracy_score(y_test, rf_pred)
svm_acc = accuracy_score(y_test, svm_pred)
knn_acc = accuracy_score(y_test, knn_pred)

rf_f1 = f1_score(y_test, rf_pred, average='weighted')
svm_f1 = f1_score(y_test, svm_pred, average='weighted')
knn_f1 = f1_score(y_test, knn_pred, average='weighted')

print("\nМетрики моделей:")
print(f"RandomForest: accuracy={rf_acc:.3f}, f1={rf_f1:.3f}")
print(f"SVM: accuracy={svm_acc:.3f}, f1={svm_f1:.3f}")
print(f"KNN: accuracy={knn_acc:.3f}, f1={knn_f1:.3f}")

# Выбор лучшей по f1-score
best_score = max(rf_f1, svm_f1, knn_f1)
if best_score == rf_f1:
    best_model = rf
    model_name = 'RandomForest'
elif best_score == svm_f1:
    best_model = svm
    model_name = 'SVM'
else:
    best_model = knn
    model_name = 'KNN'

print(f"\nЛучшая модель: {model_name} (f1={best_score:.3f})")

# Сохраняем модель
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
model_file = f"outputs/models/model_{timestamp}.pkl"
os.makedirs('outputs/models', exist_ok=True)
joblib.dump(best_model, model_file)

# Логирование
log_entry = {
    'date': timestamp,
    'model': model_name,
    'accuracy': float(best_score),
    'f1_score': float(best_score),
    'samples': len(df_combined),
    'file': model_file,
    'class_distribution': df_combined['label'].value_counts().to_dict()
}

log_file = 'outputs/models/training_log.json'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        logs = json.load(f)
else:
    logs = []

logs.append(log_entry)

with open(log_file, 'w') as f:
    json.dump(logs, f, indent=2)

print(f"\n✅ Модель сохранена: {model_file}")
print(f"📝 Лог обновлен: {log_file}")
