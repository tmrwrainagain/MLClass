import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

df = pd.read_sql("SELECT * FROM tracks", engine)

features = [
    'distance_km', 'elevation_gain', 'avg_slope',
    'max_elevation', 'min_elevation', 'avg_elevation',
    'osm_water', 'osm_buildings', 'osm_farmland', 'osm_forest',
    'avg_temperature', 'max_temperature', 'min_temperature',
    'precipitation'
]

X = df[features].fillna(0)

df['risk_label'] = 'normal'
for i in range(len(df)):
    row = df.iloc[i]
    if row['avg_temperature'] > 25 and (row['osm_farmland'] > 0 or row['osm_forest'] > 0):
        df.at[i, 'risk_label'] = 'fire_risk'
    elif row['precipitation'] > 10 and row['osm_water'] > 0:
        df.at[i, 'risk_label'] = 'flood_risk'
    elif row['elevation_gain'] > 1000 and row['avg_slope'] > 15:
        df.at[i, 'risk_label'] = 'evacuation_hard'

le = LabelEncoder()
y = le.fit_transform(df['risk_label'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

rf.fit(X_train, y_train)
svm.fit(X_train, y_train)
knn.fit(X_train, y_train)

rf_score = rf.score(X_test, y_test)
svm_score = svm.score(X_test, y_test)
knn_score = knn.score(X_test, y_test)

print(f"RandomForest: {rf_score:.3f}")
print(f"SVM: {svm_score:.3f}")
print(f"KNN: {knn_score:.3f}")

best_score = max(rf_score, svm_score, knn_score)
if best_score == rf_score:
    best_model = rf
elif best_score == svm_score:
    best_model = svm
else:
    best_model = knn

print(f"Лучшая модель: {best_model.__class__.__name__}, accuracy: {best_score:.3f}")

joblib.dump(best_model, 'outputs/models/best_classifier.pkl')
joblib.dump(scaler, 'outputs/models/classifier_scaler.pkl')
joblib.dump(le, 'outputs/models/label_encoder.pkl')