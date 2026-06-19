#!/usr/bin/env python3
"""Стадия 06 — индексация в Qdrant.

Читает all_chunks_with_embeddings.parquet и заливает точки в коллекцию Qdrant
(config.COLLECTION_NAME, config.VECTOR_SIZE, COSINE, on_disk, HNSW m=16/ef=128).
Коллекция пересоздаётся (drop-if-exists). В конце сверяет число точек с parquet.

Запуск:
    python src/pipeline/06_index_qdrant.py
    python src/pipeline/06_index_qdrant.py --collection tmp_smoke --input <tmp>.parquet  # смоук
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff,
    PayloadSchemaType, PointStruct,
)
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))   # для config
import config


def iter_points(parquet_path, arrow_batch_size):
    """Генератор PointStruct по parquet, читает Arrow-батчами (экономит RAM)."""
    table = pq.read_table(parquet_path)
    point_id = 0
    for batch in table.to_batches(max_chunksize=arrow_batch_size):
        texts = batch.column("text").to_pylist()
        doc_ids = batch.column("doc_id").to_pylist()
        embeddings = batch.column("embedding").to_pylist()
        metadata_col = batch.column("metadata").to_pylist()
        for text, doc_id, embedding, meta in zip(texts, doc_ids, embeddings, metadata_col):
            payload = {"text": text, "doc_id": doc_id}
            if isinstance(meta, dict):
                payload.update(meta)
            vector = np.array(embedding, dtype=np.float32).tolist()
            yield PointStruct(id=point_id, vector=vector, payload=payload)
            point_id += 1


def main():
    ap = argparse.ArgumentParser(description="Индексация эмбеддингов в Qdrant.")
    ap.add_argument("--input", type=Path,
                    default=DATA / "processed" / "all_chunks_with_embeddings.parquet")
    ap.add_argument("--collection", default=config.COLLECTION_NAME,
                    help="Имя коллекции (для смоука задай временное, чтобы не трогать боевую)")
    ap.add_argument("--host", default=config.QDRANT_HOST)
    ap.add_argument("--port", type=int, default=config.QDRANT_PORT)
    ap.add_argument("--batch-size", type=int, default=1024, help="Точек на один upsert")
    ap.add_argument("--arrow-batch-size", type=int, default=4096)
    args = ap.parse_args()

    client = QdrantClient(host=args.host, port=args.port, timeout=300)
    existing = [c.name for c in client.get_collections().collections]
    print(f"Connected to Qdrant. Collections: {existing}")

    # Пересоздание коллекции (clean slate)
    if client.collection_exists(collection_name=args.collection):
        client.delete_collection(collection_name=args.collection)
        print(f"Collection '{args.collection}' deleted.")
    client.create_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=config.VECTOR_SIZE, distance=Distance.COSINE, on_disk=True),
        hnsw_config=HnswConfigDiff(m=16, ef_construct=128, on_disk=True),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=50_000),  # отложить индекс на bulk-заливке
        on_disk_payload=True,
    )
    print(f"Collection '{args.collection}' created.")

    # Payload-индексы для фильтрованного поиска.
    # NB: индекс на 'Header_1' (с большой буквы) — как в исходном ноутбуке;
    # фактическое поле payload называется 'header_1' (из metadata). Сохранено как есть.
    for field_name, field_type in [("doc_id", PayloadSchemaType.KEYWORD),
                                   ("Header_1", PayloadSchemaType.KEYWORD)]:
        client.create_payload_index(
            collection_name=args.collection, field_name=field_name, field_schema=field_type)
        print(f"Index created: {field_name} ({field_type})")

    total_rows = pq.read_metadata(args.input).num_rows
    print(f"Parquet: {total_rows:,} rows")

    # Батч-заливка
    batch, uploaded = [], 0
    pbar = tqdm(total=total_rows, desc="Uploading", unit="pts")
    for point in iter_points(args.input, args.arrow_batch_size):
        batch.append(point)
        if len(batch) >= args.batch_size:
            client.upsert(collection_name=args.collection, points=batch, wait=True)
            uploaded += len(batch)
            pbar.update(len(batch))
            batch = []
    if batch:
        client.upsert(collection_name=args.collection, points=batch, wait=True)
        uploaded += len(batch)
        pbar.update(len(batch))
    pbar.close()
    print(f"Upload complete: {uploaded:,} points")

    # Вернуть дефолтный порог индексации → HNSW начнёт строиться
    client.update_collection(
        collection_name=args.collection,
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20_000),
    )
    print("Indexing threshold restored (HNSW builds in background).")

    # Сверка количества точек с источником
    info = client.get_collection(collection_name=args.collection)
    print(f"Collection '{args.collection}': points={info.points_count:,} status={info.status}")
    assert info.points_count == total_rows, \
        f"MISMATCH: expected {total_rows:,}, got {info.points_count:,}"
    print(f"[OK] Count matches source parquet ({total_rows:,} rows).")


if __name__ == "__main__":
    main()
