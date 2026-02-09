#!/bin/bash
set -e

QDRANT_HOST="${QDRANT_HOST:-localhost}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_URL="http://${QDRANT_HOST}:${QDRANT_PORT}"
COLLECTION="nlp2025_chunks"
SNAPSHOT="/data/qdrant.snapshot"

echo "=== ArXiv RAG System ==="

# 1. Wait for Qdrant
echo "Waiting for Qdrant at ${QDRANT_URL}..."
until curl -sf "${QDRANT_URL}/readyz" > /dev/null 2>&1; do
    sleep 2
done
echo "Qdrant is ready."

# 2. Restore snapshot if collection doesn't exist
if curl -sf "${QDRANT_URL}/collections/${COLLECTION}" | grep -q '"status":"ok"'; then
    echo "Collection '${COLLECTION}' already exists. Skipping restore."
else
    if [ ! -f "$SNAPSHOT" ]; then
        echo "ERROR: Snapshot not found at ${SNAPSHOT}"
        echo "Make sure qdrant.snapshot is in the project root."
        exit 1
    fi
    echo "Restoring snapshot (~4 GB, this will take a few minutes)..."
    curl -X POST "${QDRANT_URL}/collections/${COLLECTION}/snapshots/upload?priority=snapshot" \
        -H "Content-Type: multipart/form-data" \
        -F "snapshot=@${SNAPSHOT}"
    echo ""
    echo "Snapshot restored."
fi

# 3. Start backend
echo "Starting FastAPI backend..."
cd /app
python -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 4. Start frontend
echo "Starting Streamlit frontend..."
streamlit run frontend.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true &
FRONTEND_PID=$!

echo ""
echo "=== System is ready ==="
echo "  Frontend: http://localhost:8501"
echo "  Backend:  http://localhost:8000/docs"
echo ""

wait $BACKEND_PID $FRONTEND_PID
