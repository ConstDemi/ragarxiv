#!/usr/bin/env python3
"""Стадия 05 — эмбеддинги чанков.

Читает all_chunks.parquet, кодирует тексты эмбеддером config.EMBED_MODEL
(SentenceTransformer, bf16, L2-normalize) и сохраняет
data/processed/all_chunks_with_embeddings.parquet (+ колонка embedding).

По умолчанию эмбеддится поле `text` (baseline). `--text-field embed_text` — вариант
с контекстом заголовков секции (A/B по ранжированию); меняется только вектор, payload.text остаётся.

Запуск:
    python src/pipeline/05_embed.py
    python src/pipeline/05_embed.py --limit 100 --batch-size 16
"""
import argparse
import sys
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))   # для config
import config


def main():
    ap = argparse.ArgumentParser(description="Векторизация чанков эмбеддером.")
    ap.add_argument("--input", type=Path, default=DATA / "processed" / "all_chunks.parquet")
    ap.add_argument("--output", type=Path,
                    default=DATA / "processed" / "all_chunks_with_embeddings.parquet")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-field", default="text", choices=["text", "embed_text"],
                    help="какое поле эмбедить (A/B: text vs embed_text)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = SentenceTransformer(
        config.EMBED_MODEL,
        device=args.device,
        model_kwargs={"dtype": torch.bfloat16, "trust_remote_code": True},
        tokenizer_kwargs={"padding_side": "left"},
    )

    field = args.text_field
    print(f"Эмбедим поле: {field}")

    def compute_embeddings(batch):
        embeddings = model.encode(
            batch[field],
            batch_size=len(batch[field]),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return {"embedding": embeddings}

    ds = Dataset.from_parquet(str(args.input))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Векторизация {len(ds)} чанков...")

    ds = ds.map(compute_embeddings, batched=True, batch_size=args.batch_size, desc="Embedding")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(str(args.output))
    print(f"SUCCESS! {args.output}")


if __name__ == "__main__":
    main()
