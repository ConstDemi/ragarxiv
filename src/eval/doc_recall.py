#!/usr/bin/env python3
"""Дешёвая retrieval-диагностика: doc_recall@k + MRR/ранг gold.

Без судьи и без LLM-генератора — только эмбеддер + Qdrant (детерминированно, бесплатно).
Зеркалит логику ScienceRAG._retrieve / _group_sources, чтобы числа были сравнимы с run.py.

Метрики:
  - @context_k / @retrieved / @retrieve_k — бинарные hit-rate (грубые при n=50);
  - MRR и медианный ранг gold — непрерывные, чувствительные → главные для A/B ранжирования
    (ловят сдвиг ранга, даже если он не пересёк порог top-k).

Промахи @retrieved печатаются с рангом gold для триажа (recall vs ранжирование vs артефакт разметки).

Запуск:
    python src/eval/doc_recall.py
    python src/eval/doc_recall.py --collection <other>
"""
import argparse
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))         # src/ — для config

import torch
import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import config


def main():
    ap = argparse.ArgumentParser(description="doc_recall@k + MRR/ранг (judge-free, без 7B).")
    ap.add_argument("--collection", default=config.COLLECTION_NAME)
    ap.add_argument("--retrieve-k", type=int, default=config.RETRIEVE_K)
    ap.add_argument("--context-k", type=int, default=config.CONTEXT_K)
    ap.add_argument("--max-papers", type=int, default=config.MAX_PAPERS)
    ap.add_argument("--golden", type=Path, default=HERE.parent.parent / config.GOLDEN_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--show-misses", type=int, default=12)
    args = ap.parse_args()

    golden = pq.read_table(args.golden).to_pylist()
    if args.limit:
        golden = golden[:args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    encoder = SentenceTransformer(config.EMBED_MODEL, trust_remote_code=True, device=device)

    print(f"[doc_recall] collection={args.collection} retrieve_k={args.retrieve_k} "
          f"context_k={args.context_k} max_papers={args.max_papers} | вопросов: {len(golden)}")

    hit_ctx, hit_retr, hit_k, rr, ranks, misses, qtypes = [], [], [], [], [], [], []
    paper_hits = {1: [], 3: [], 5: []}   # строгий k на уровне статей (headroom для реранкера)
    for i, row in enumerate(golden, 1):
        q = row["question"]
        golds = set(row.get("relevant_doc_ids") or [row["doc_id"]])        # мульти-релевантность (v2); фолбэк — single gold
        qt = row.get("qtype", "?")
        # зеркало ScienceRAG._retrieve: query-инструкция Qwen3, без нормализации (Cosine нормализует сам)
        vec = encoder.encode(q, prompt_name="query", convert_to_numpy=True, show_progress_bar=False)
        res = client.query_points(collection_name=args.collection, query=vec.tolist(),
                                  limit=args.retrieve_k, with_payload=True, with_vectors=False)
        chunks = [{"doc_id": (p.payload or {}).get("doc_id"),
                   "title": (p.payload or {}).get("title", ""),
                   "score": p.score} for p in res.points]
        ctx_docs = {c["doc_id"] for c in chunks[:args.context_k]}          # топ-context_k чанков
        seen = []
        for c in chunks:
            if c["doc_id"] not in seen:
                seen.append(c["doc_id"])
        source_docs = set(seen[:args.max_papers])                          # топ-max_papers статей (как _group_sources)
        for K in paper_hits:                                               # recall@1/3/5 — gold среди топ-K статей
            paper_hits[K].append(bool(set(seen[:K]) & golds))
        all_docs = {c["doc_id"] for c in chunks}                           # любой из retrieve_k чанков
        gold_rank = next((j for j, c in enumerate(chunks, 1) if c["doc_id"] in golds), None)  # лучший ранг среди gold

        hit_ctx.append(bool(ctx_docs & golds))
        hit_retr.append(bool(source_docs & golds))
        hit_k.append(bool(all_docs & golds))
        rr.append(1.0 / gold_rank if gold_rank else 0.0)                   # reciprocal rank (miss=0)
        ranks.append(gold_rank)
        qtypes.append(qt)
        if not (source_docs & golds):
            misses.append((i, q, row["doc_id"], row.get("title", ""), gold_rank, chunks[:3]))

    n = len(golden) or 1
    found = [r for r in ranks if r]
    print(f"\n[doc_recall] @context_k  (топ-{args.context_k} чанков)   = {sum(hit_ctx)/n:.3f}  ({sum(hit_ctx)}/{len(golden)})")
    print(f"[doc_recall] @retrieved  (топ-{args.max_papers} статей)    = {sum(hit_retr)/n:.3f}  ({sum(hit_retr)}/{len(golden)})")
    print(f"[doc_recall] @retrieve_k (любой из {args.retrieve_k})    = {sum(hit_k)/n:.3f}  ({sum(hit_k)}/{len(golden)})   ← «нашли ли вообще»")
    print(f"[doc_recall] разрыв @retrieve_k − @retrieved = {(sum(hit_k)-sum(hit_retr))/n:.3f}  (ранжирование; >0 → может помочь reranker)")
    print()
    for K in (1, 3, 5):
        print(f"[doc_recall] recall@{K} статей               = {sum(paper_hits[K])/n:.3f}  ({sum(paper_hits[K])}/{len(golden)})   ← строгий k (арена реранкера)")
    print(f"\n[doc_recall] MRR (gold; miss=0)            = {sum(rr)/n:.3f}   ← ГЛАВНАЯ для A/B ранжирования (чувствительна при n=50)")
    # разбивка по типу вопроса (qtype): Q1 существование vs Q2 метод
    if any(q in ("Q1", "Q2") for q in qtypes):
        for qt in ("Q1", "Q2"):
            idxs = [j for j, x in enumerate(qtypes) if x == qt]
            if idxs:
                cc = sum(hit_ctx[j] for j in idxs) / len(idxs)
                cr = sum(hit_retr[j] for j in idxs) / len(idxs)
                print(f"[doc_recall] {qt}: @context_k={cc:.3f} @retrieved={cr:.3f}  (n={len(idxs)})")
    if found:
        print(f"[doc_recall] ранг gold (среди найденных)   : медиана={statistics.median(found):.0f} "
              f"среднее={sum(found)/len(found):.1f}  (найден в топ-{args.retrieve_k}: {len(found)}/{len(golden)})")

    if misses:
        print(f"\n[doc_recall] промахи @retrieved: {len(misses)} — триаж:")
        for i, q, gold, gtitle, gold_rank, top in misses[:args.show_misses]:
            where = f"в top-{args.retrieve_k} ранг {gold_rank}" if gold_rank else f"НЕ в top-{args.retrieve_k}"
            print(f"  #{i:>2} gold={gold} [{where}] «{gtitle[:55]}»")
            print(f"      Q: {q[:85]}")
            for c in top:
                print(f"      top: {c['doc_id']} s={c['score']:.3f} «{(c['title'] or '')[:55]}»")
    else:
        print("\n[doc_recall] промахов @retrieved нет — ретрив держит.")


if __name__ == "__main__":
    main()
