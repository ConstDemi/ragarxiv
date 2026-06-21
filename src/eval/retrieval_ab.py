#!/usr/bin/env python3
"""Ретрив-A/B (judge-free, без 7B): dense vs +rerank vs +hybrid(BM25,RRF) vs +both.

Меряем на уровне статей: recall@{1,3,5,8} + MRR, мульти-gold через relevant_doc_ids.
Смысл: проверить, помогают ли реранкер/гибрид там, где есть headroom (трудный golden).
Реранкер — cross-encoder Qwen3-Reranker поверх кандидатов; BM25 — in-memory + RRF-фьюз с dense.

Запуск:
    python src/eval/retrieval_ab.py                                          # на трудном golden
    python src/eval/retrieval_ab.py --golden data/eval/golden_dataset50_v2.parquet   # на лёгком
"""
import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))  # src/ — для config
import config

import torch
import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

POOL = 50                                          # кандидатов из каждого ретривера
RERANKER = "BAAI/bge-reranker-v2-m3"  # мультиязычный cross-encoder (запросы RU, доки EN); обычный CrossEncoder API
RRF_K = 60
KS = (1, 3, 5, 8)
CONFIGS = ["dense", "dense+rerank", "hybrid", "hybrid+rerank"]


def tok(s):
    return re.findall(r"[a-zA-Zа-яА-Я0-9]+", (s or "").lower())


def load_bm25(client, collection):
    """Scroll всех чанков из Qdrant → (doc_ids, texts, BM25Okapi)."""
    doc_ids, texts, toks = [], [], []
    offset = None
    while True:
        pts, offset = client.scroll(collection_name=collection, limit=4000, offset=offset,
                                    with_payload=["doc_id", "text"], with_vectors=False)
        for p in pts:
            pl = p.payload or {}
            doc_ids.append(pl.get("doc_id"))
            t = pl.get("text") or ""
            texts.append(t)
            toks.append(tok(t))
        if offset is None:
            break
    return doc_ids, texts, BM25Okapi(toks)


def distinct(doc_ids_in_order):
    seen, out = set(), []
    for d in doc_ids_in_order:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def rrf(*orders, k=RRF_K):
    """Reciprocal Rank Fusion: объединяет ранжированные списки doc_id."""
    sc = defaultdict(float)
    for order in orders:
        for rank, d in enumerate(order, 1):
            sc[d] += 1.0 / (k + rank)
    return [d for d, _ in sorted(sc.items(), key=lambda x: -x[1])]


def metrics(order, golds):
    """order — ранжированные distinct-доки; golds — множество релевантных. recall@K + RR."""
    rank = next((i for i, d in enumerate(order, 1) if d in golds), None)
    out = {f"@{K}": int(bool(set(order[:K]) & golds)) for K in KS}
    out["rr"] = (1.0 / rank if rank else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser(description="Ретрив-A/B: dense/+rerank/+hybrid/+both (judge-free).")
    ap.add_argument("--golden", type=Path, default=ROOT / "data/eval/golden_dataset50_hard.parquet")
    ap.add_argument("--collection", default=config.COLLECTION_NAME)
    ap.add_argument("--pool", type=int, default=POOL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    golden = pq.read_table(args.golden).to_pylist()
    if args.limit:
        golden = golden[:args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    print(f"[ab] golden={args.golden.name} collection={args.collection} pool={args.pool} | вопросов: {len(golden)}")

    print("[ab] загрузка эмбеддера + реранкера...")
    encoder = SentenceTransformer(config.EMBED_MODEL, trust_remote_code=True, device=device)
    reranker = CrossEncoder(RERANKER, trust_remote_code=True, device=device, max_length=512)

    print("[ab] построение BM25-корпуса (scroll Qdrant)...")
    t0 = time.time()
    bm_doc_ids, bm_texts, bm25 = load_bm25(client, args.collection)
    print(f"[ab]   BM25: {len(bm_doc_ids)} чанков за {time.time()-t0:.0f}s")

    agg = {c: defaultdict(list) for c in CONFIGS}
    agg_q = {c: {"Q1": defaultdict(list), "Q2": defaultdict(list)} for c in CONFIGS}

    for r in golden:
        q = r["question"]
        golds = set(r.get("relevant_doc_ids") or [r["doc_id"]])
        qt = r.get("qtype", "?")

        # --- dense ---
        vec = encoder.encode(q, prompt_name="query", convert_to_numpy=True, show_progress_bar=False)
        dres = client.query_points(collection_name=args.collection, query=vec.tolist(), limit=args.pool,
                                   with_payload=["doc_id", "text"], with_vectors=False).points
        dense_chunks = [((p.payload or {}).get("doc_id"), (p.payload or {}).get("text") or "") for p in dres]
        dense_order = distinct([d for d, _ in dense_chunks])

        # --- bm25 ---
        scores = bm25.get_scores(tok(q))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:args.pool]
        bm_chunks = [(bm_doc_ids[i], bm_texts[i]) for i in top]
        bm_order = distinct([d for d, _ in bm_chunks])

        def rerank_chunks(chunks):
            """Реранк на уровне ЧАНКОВ, скор статьи = max по её чанкам (max-pool).
            Иначе статья судится по одному (произвольному) чанку и тонет, если её
            абстракт ниже по dense-рангу, чем таблица/фрагмент."""
            if not chunks:
                return []
            sc = reranker.predict([(q, t) for _, t in chunks])
            best = {}
            for (d, _), s in zip(chunks, sc):
                if d and (d not in best or s > best[d]):
                    best[d] = s
            return [d for d, _ in sorted(best.items(), key=lambda x: -x[1])]

        results = {
            "dense": dense_order,
            "hybrid": rrf(dense_order, bm_order),
            "dense+rerank": rerank_chunks(dense_chunks),
            "hybrid+rerank": rerank_chunks(dense_chunks + bm_chunks),
        }

        for c in CONFIGS:
            for kk, vv in metrics(results[c], golds).items():
                agg[c][kk].append(vv)
                if qt in ("Q1", "Q2"):
                    agg_q[c][qt][kk].append(vv)

    n = len(golden) or 1
    print(f"\n{'config':<15} " + "  ".join(f"R@{K}" for K in KS) + "    MRR    ΔMRR")
    base_mrr = sum(agg['dense']['rr']) / n
    for c in CONFIGS:
        row = [sum(agg[c][f'@{K}']) / n for K in KS]
        mrr = sum(agg[c]['rr']) / n
        d = mrr - base_mrr
        print(f"{c:<15} " + "  ".join(f"{x:.2f}" for x in row) + f"   {mrr:.3f}  {d:+.3f}")

    print("\nпо типам (recall@3 / MRR):")
    for c in CONFIGS:
        parts = []
        for qt in ("Q1", "Q2"):
            qn = len(agg_q[c][qt]['rr']) or 1
            parts.append(f"{qt} R@3={sum(agg_q[c][qt]['@3'])/qn:.2f} MRR={sum(agg_q[c][qt]['rr'])/qn:.2f}")
        print(f"  {c:<15} " + " | ".join(parts))


if __name__ == "__main__":
    main()
