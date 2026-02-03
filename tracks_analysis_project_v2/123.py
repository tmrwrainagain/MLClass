import requests
import json

headers = {'User-Agent': 'TrackAnalysis/1.0 (student_project)'}

# Тестовая точка (Москва)
lat = 55.7558
lon = 37.6173

overpass_query = f"""
[out:json][timeout:30];
(
  node["natural"="water"](around:500,{lat},{lon});
  node["building"](around:500,{lat},{lon});
);
out;
"""

print("Запрос:", overpass_query[:100], "...")

response = requests.post(
    "http://overpass-api.de/api/interpreter",
    data={'data': overpass_query},
    timeout=30,
    headers=headers
)

print(f"Статус: {response.status_code}")
print(f"Длина ответа: {len(response.text)}")
print(f"Первые 200 символов: {response.text[:200]}")

if response.text.strip():
    try:
        data = response.json()
        print(f"Элементов: {len(data.get('elements', []))}")
        for element in data.get('elements', [])[:5]:  # первые 5
            print(f"  - {element.get('tags', {})}")
    except:
        print("Не JSON!")
else:
    print("Пустой ответ!")