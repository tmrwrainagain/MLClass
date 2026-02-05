import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sqlalchemy import create_engine

DB_URL = "postgresql://postgres:password@localhost:5432/tracks_db"
engine = create_engine(DB_URL)

# Создаем таблицу для временных рядов
with engine.connect() as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flood_history (
            id SERIAL PRIMARY KEY,
            date DATE,
            zone_id VARCHAR(50),
            precipitation FLOAT,
            water_level FLOAT,
            flood_risk VARCHAR(20)
        )
    """)
    conn.commit()

# Генерация исторических данных (если нет)
dates = []
start_date = datetime(2023, 1, 1)
for i in range(365):
    date = start_date + timedelta(days=i)
    dates.append(date)

# Случайные данные
np.random.seed(42)
precipitation = np.random.normal(50, 20, 365)
water_level = precipitation * 0.8 + np.random.normal(0, 10, 365)

# Рассчитываем риск
flood_risk = []
for i in range(365):
    if precipitation[i] > 70 and water_level[i] > 60:
        risk = 'high'
    elif precipitation[i] > 50 and water_level[i] > 40:
        risk = 'medium'
    else:
        risk = 'low'
    flood_risk.append(risk)

# Сохраняем в БД
for i in range(365):
    with engine.connect() as conn:
        conn.execute(f"""
            INSERT INTO flood_history (date, zone_id, precipitation, water_level, flood_risk)
            VALUES ('{dates[i].date()}', 'zone_1', {precipitation[i]}, {water_level[i]}, '{flood_risk[i]}')
            ON CONFLICT DO NOTHING
        """)
        conn.commit()

print("Исторические данные сохранены в БД")

# Загружаем для анализа
df_history = pd.read_sql("SELECT * FROM flood_history ORDER BY date", engine)

# Простой прогноз на год вперед (скользящее среднее)
future_dates = []
future_precipitation = []
future_water = []

last_date = df_history['date'].iloc[-1]

for i in range(1, 366):
    future_date = last_date + timedelta(days=i)
    future_dates.append(future_date)
    
    # Простой прогноз: среднее за последние 30 дней + тренд
    last_30_precip = df_history['precipitation'].tail(30).mean()
    future_precipitation.append(last_30_precip * (1 + 0.001 * i))
    
    last_30_water = df_history['water_level'].tail(30).mean()
    future_water.append(last_30_water * (1 + 0.0005 * i))

# Визуализация
plt.figure(figsize=(15, 10))

# График 1: Исторические осадки
plt.subplot(2, 2, 1)
plt.plot(df_history['date'], df_history['precipitation'], label='Исторические')
plt.plot(future_dates, future_precipitation, 'r--', label='Прогноз')
plt.title('Осадки: история и прогноз на год')
plt.xlabel('Дата')
plt.ylabel('Осадки (мм)')
plt.legend()
plt.grid(True)

# График 2: Уровень воды
plt.subplot(2, 2, 2)
plt.plot(df_history['date'], df_history['water_level'], label='Исторические')
plt.plot(future_dates, future_water, 'r--', label='Прогноз')
plt.title('Уровень воды: история и прогноз')
plt.xlabel('Дата')
plt.ylabel('Уровень воды')
plt.legend()
plt.grid(True)

# График 3: Риски затопления
plt.subplot(2, 2, 3)
risk_counts = df_history['flood_risk'].value_counts()
plt.bar(risk_counts.index, risk_counts.values)
plt.title('Распределение рисков (история)')
plt.xlabel('Уровень риска')
plt.ylabel('Количество дней')
plt.grid(True)

# График 4: Сезонность
plt.subplot(2, 2, 4)
df_history['month'] = pd.to_datetime(df_history['date']).dt.month
monthly_avg = df_history.groupby('month')['precipitation'].mean()
plt.plot(monthly_avg.index, monthly_avg.values, 'o-')
plt.title('Сезонность осадков')
plt.xlabel('Месяц')
plt.ylabel('Средние осадки')
plt.grid(True)

plt.tight_layout()
plt.savefig('outputs/flood_forecast.png', dpi=150)
plt.show()

print("✅ Прогноз на год построен и сохранен в outputs/flood_forecast.png")

# Сохраняем прогноз в БД
for i in range(len(future_dates)):
    if future_water[i] > 60:
        risk = 'high'
    elif future_water[i] > 40:
        risk = 'medium'
    else:
        risk = 'low'
    
    with engine.connect() as conn:
        conn.execute(f"""
            INSERT INTO flood_predictions (date, zone_id, water_level, flood_risk)
            VALUES ('{future_dates[i]}', 'zone_1', {future_water[i]}, '{risk}')
        """)
        conn.commit()

print("✅ Прогноз сохранен в таблицу flood_predictions")
