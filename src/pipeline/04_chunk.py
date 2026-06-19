#!/usr/bin/env python3
"""Стадия 04 — чанкование Markdown.

Читает processed_data.parquet (doc_id, md) + метаданные статей, режет на чанки
(MarkdownHeaderTextSplitter → RecursiveCharacterTextSplitter по токенайзеру
эмбеддера config.EMBED_MODEL), обогащает payload-метаданными и сохраняет
data/processed/all_chunks.parquet (text, embed_text, doc_id, metadata).

`text` — чистый контент (идёт в LLM/payload); `embed_text` — путь заголовков
секции + текст (контекстуализированный вариант для эмбеддинга).

Запуск:
    python src/pipeline/04_chunk.py
    python src/pipeline/04_chunk.py --limit 20
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))   # для config
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_batch(batch, md_splitter, text_splitter, tokenizer, paper_metadata):
    """Чанкование батча: чистый text + контекстуализированный embed_text + metadata.
    Импорты/константы внутри — для num_proc (отдельные процессы без глобалов)."""
    import logging
    logger = logging.getLogger(__name__)
    MIN_CHUNK_TOKENS = 10

    chunk_texts, chunk_embed_texts, chunk_doc_ids, chunk_metadata_list = [], [], [], []

    for doc_id, md_content in zip(batch["doc_id"], batch["md"]):
        if not md_content or not isinstance(md_content, str) or len(md_content.strip()) == 0:
            continue
        try:
            sections = md_splitter.split_text(md_content)          # 1. по заголовкам
            chunks = text_splitter.split_documents(sections)       # 2. рекурсивно по токенам
            paper_meta = paper_metadata.get(doc_id, {})            # 3. метаданные статьи

            for chunk_idx, chunk in enumerate(chunks):
                text = chunk.page_content.strip()
                if len(tokenizer.encode(text, add_special_tokens=False)) < MIN_CHUNK_TOKENS:
                    continue
                header_1 = chunk.metadata.get("header_1", "")
                header_2 = chunk.metadata.get("header_2", "")
                header_3 = chunk.metadata.get("header_3", "")
                header_path = " > ".join(h for h in (header_1, header_2, header_3) if h)
                embed_text = f"{header_path}\n\n{text}" if header_path else text
                meta = {
                    "header_1": header_1, "header_2": header_2, "header_3": header_3,
                    "title": paper_meta.get("title", ""),
                    "authors": paper_meta.get("authors", ""),
                    "published": paper_meta.get("published", ""),
                    "chunk_index": chunk_idx,
                }
                chunk_texts.append(text)
                chunk_embed_texts.append(embed_text)
                chunk_doc_ids.append(doc_id)
                chunk_metadata_list.append(meta)
        except Exception as e:
            logger.error(f"Error processing doc_id {doc_id}: {e}")
            continue

    return {
        "text": chunk_texts,
        "embed_text": chunk_embed_texts,
        "doc_id": chunk_doc_ids,
        "metadata": chunk_metadata_list,
    }


def main():
    ap = argparse.ArgumentParser(description="Чанкование Markdown в чанки для эмбеддинга.")
    ap.add_argument("--input", type=Path,
                    default=DATA / "processed" / "parquet" / "processed_data.parquet")
    ap.add_argument("--metadata", type=Path,
                    default=DATA / "metadata" / "arxiv_NLP_2021_2026_metadata.csv")
    ap.add_argument("--output", type=Path, default=DATA / "processed" / "all_chunks.parquet")
    ap.add_argument("--chunk-size", type=int, default=512, help="Размер чанка в токенах")
    ap.add_argument("--overlap", type=int, default=50, help="Перекрытие в токенах")
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--num-proc", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ds = Dataset.from_parquet(str(args.input))
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    logger.info(f"Dataset loaded: {len(ds)} documents")

    meta_df = pd.read_csv(
        args.metadata, usecols=["arxiv_id", "title", "authors", "published"],
        dtype={"arxiv_id": str, "title": str, "authors": str, "published": str},
    )
    paper_metadata = (
        meta_df.dropna(subset=["arxiv_id"]).set_index("arxiv_id")
        [["title", "authors", "published"]].to_dict(orient="index")
    )
    doc_ids_in_ds = set(ds["doc_id"])
    coverage = len(doc_ids_in_ds & set(paper_metadata)) / max(len(doc_ids_in_ds), 1) * 100
    logger.info(f"Paper metadata: {len(paper_metadata)} entries | coverage {coverage:.1f}%")

    tokenizer = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "header_1"), ("##", "header_2"), ("###", "header_3"),
    ])
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, chunk_size=args.chunk_size, chunk_overlap=args.overlap,
    )
    logger.info(f"Splitters ready: chunk_size={args.chunk_size}, overlap={args.overlap}")

    chunked = ds.map(
        process_batch, batched=True, batch_size=args.batch_size,
        remove_columns=ds.column_names, num_proc=args.num_proc,
        fn_kwargs={"md_splitter": md_splitter, "text_splitter": text_splitter,
                   "tokenizer": tokenizer, "paper_metadata": paper_metadata},
    )
    logger.info(f"Chunking complete: {len(chunked)} chunks")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunked.to_parquet(str(args.output))
    logger.info(f"Saved to {args.output}: {len(chunked)} chunks")


if __name__ == "__main__":
    main()
