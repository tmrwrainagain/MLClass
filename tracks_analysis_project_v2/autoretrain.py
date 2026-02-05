import time
import os
import pandas as pd
from sqlalchemy import create_engine
import subprocess

DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

# Проверяем последний лог
log_file = 'outputs/models/training_log.json'
if os.path.exists(log_file):
    with open(log_file, 'r') as f:
        import json
        logs = json.load(f)
    last_count = logs[-1]['samples'] if logs else 0
else:
    last_count = 0

print(f"Последнее обучение: {last_count} треков")

while True:
    # Проверяем БД
    df = pd.read_sql("SELECT COUNT(*) as cnt FROM tracks", engine)
    current_count = df.iloc[0]['cnt']
    
    print(f"Треков в БД: {current_count}")
    
    # Если добавилось 5+ новых треков
    if current_count > last_count + 5:
        print("🆕 Обнаружены новые данные, запускаю обучение...")
        
        # Запускаем агента
        result = subprocess.run(['python', 'agent_learn.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Модель дообучена")
            # Обновляем счетчик
            last_count = current_count
        else:
            print(f"❌ Ошибка: {result.stderr}")
    
    # Ждем 1 час
    time.sleep(3600)
