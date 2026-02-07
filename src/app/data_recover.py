import requests
from pathlib import Path

# === КОНФИГ ===
BASE_URL = "http://localhost:6333"
COLLECTION_NAME = "nlp2025_chunks"

CURRENT_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = CURRENT_DIR.parent / "qdrant.snapshot"

print(f'Загрузка и восстановление из: {SNAPSHOT_PATH}...')

with open(SNAPSHOT_PATH, "rb") as f:
    response = requests.post(
        f'{BASE_URL}/collections/{COLLECTION_NAME}/snapshots/upload',
        files={"snapshot": f},
        params={"priority": "snapshot"}
    )

if response.status_code == 200:
    print(f'Коллекция "{COLLECTION_NAME}" восстановлена.')
else:
    print(f'Ошибка {response.status_code}: {response.text}')