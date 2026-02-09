import requests
import os
from pathlib import Path

# === КОНФИГ ===
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
BASE_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
COLLECTION_NAME = "nlp2025_chunks"

# Путь к snapshot (внутри контейнера)
SNAPSHOT_PATH = Path("/app/qdrant.snapshot")

if not SNAPSHOT_PATH.exists():
    print(f"ERROR: Snapshot file not found at {SNAPSHOT_PATH}")
    print("Please mount the snapshot file when starting the container")
    exit(1)

print(f'Loading snapshot from: {SNAPSHOT_PATH}...')
print(f'Target Qdrant: {BASE_URL}')

try:
    with open(SNAPSHOT_PATH, "rb") as f:
        response = requests.post(
            f'{BASE_URL}/collections/{COLLECTION_NAME}/snapshots/upload',
            files={"snapshot": f},
            params={"priority": "snapshot"},
            timeout=300  # 5 минут таймаут для больших snapshot'ов
        )

    if response.status_code == 200:
        print(f'✅ Collection "{COLLECTION_NAME}" restored successfully!')
    else:
        print(f'❌ Error {response.status_code}: {response.text}')
        exit(1)
        
except requests.exceptions.ConnectionError:
    print(f"❌ Cannot connect to Qdrant at {BASE_URL}")
    print("Make sure Qdrant container is running")
    exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    exit(1)
