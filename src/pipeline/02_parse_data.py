#!/usr/bin/env python3
"""Стадия 02 — загрузка HTML статей + упаковка в parquet-шарды.

Читает метаданные, фильтрует по диапазону лет (default 2021–2026), асинхронно качает HTML
(export.arxiv.org → ar5iv fallback, burst + backoff) в data/raw/html/, затем
пакует .html в шарды data/raw/parquet/shard_XXXX.parquet (doc_id, html).
Отчёт об ошибках → data/metadata/download_errors.csv.

Запуск:
    python src/pipeline/02_parse_data.py                # всё: download + shard
    python src/pipeline/02_parse_data.py --limit 3      # смоук: 3 статьи
    python src/pipeline/02_parse_data.py --stage shard  # только упаковка
    python src/pipeline/02_parse_data.py --year-min 2021 --year-max 2026   # весь корпус 2021–2026
"""
import argparse
import asyncio
import random
from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

HEADERS = {
    "User-Agent": "ArXivRAGProject/1.0 (academic capstone; https://github.com/ConstDemi/ragarxiv)",
    "Accept": "text/html",
}
MIN_CONTENT_SIZE = 10_000   # меньше — заглушка/пустая страница
MAX_RETRIES = 3


def get_urls(arxiv_id):
    """Приоритет: ar5iv (LaTeXML-HTML, покрывает весь диапазон 2021–2026) →
    arxiv.org/html (нативный, для свежих статей, которых нет на ar5iv).
    export.arxiv.org убран: он 429-ит и не отдаёт HTML (только холостой backoff)."""
    return [
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/html/{arxiv_id}",
    ]


async def download_one(client, arxiv_id, html_dir, failed):
    """Качает одну статью: основной URL → fallback; backoff на rate limit."""
    last_error = None
    for url in get_urls(arxiv_id):
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    if len(response.text) >= MIN_CONTENT_SIZE:
                        (html_dir / f"{arxiv_id}.html").write_text(response.text, encoding="utf-8")
                        return True
                    last_error = {"status": 200, "type": "SmallContent", "url": url}
                    break
                elif response.status_code == 404:
                    last_error = {"status": 404, "type": "NotFound", "url": url}
                    break
                elif response.status_code in (429, 503, 406):
                    last_error = {"status": response.status_code, "type": "RateLimit", "url": url}
                    await asyncio.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
                    continue
                else:
                    last_error = {"status": response.status_code, "type": "HTTPError", "url": url}
                    break
            except httpx.TimeoutException:
                last_error = {"status": None, "type": "Timeout", "url": url}
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                last_error = {"status": None, "type": "Exception", "url": url, "details": str(e)}
                break
    failed.append({"arxiv_id": arxiv_id, **(last_error or {"type": "Unknown"})})
    return False


async def download_all(ids, html_dir, burst_size, burst_pause):
    """Burst-скачивание по рекомендации arXiv. Возвращает список ошибок."""
    failed = []
    total = len(ids)
    print(f"Скачиваем {total} статей (burst {burst_size}, пауза {burst_pause}с)")
    async with httpx.AsyncClient(
        timeout=30.0, follow_redirects=True, headers=HEADERS,
        limits=httpx.Limits(max_connections=burst_size, max_keepalive_connections=burst_size),
    ) as client:
        pbar = tqdm(total=total, desc="Downloading")
        success = 0
        for i in range(0, total, burst_size):
            burst = ids[i:i + burst_size]
            results = await asyncio.gather(
                *[download_one(client, x, html_dir, failed) for x in burst]
            )
            success += sum(results)
            pbar.update(len(burst))
            if i + burst_size < total:
                await asyncio.sleep(burst_pause)
        pbar.close()
    print(f"Успешно: {success}/{total} | ошибок: {len(failed)}")
    return failed


def pack_shards(html_dir, parquet_dir, shard_size):
    """Пакует .html → parquet-шарды (doc_id, html)."""
    parquet_dir.mkdir(parents=True, exist_ok=True)
    files = list(html_dir.glob("*.html"))
    print(f"HTML файлов для упаковки: {len(files)}")
    schema = pa.schema([("doc_id", pa.string()), ("html", pa.string())])
    for i in tqdm(range(0, len(files), shard_size), desc="Sharding"):
        batch = []
        for f in files[i:i + shard_size]:
            try:
                batch.append({"doc_id": f.stem, "html": f.read_text(encoding="utf-8")})
            except Exception as e:
                print(f"Ошибка в файле {f.name}: {e}")
        if batch:
            table = pa.Table.from_pylist(batch, schema=schema)
            pq.write_table(table, parquet_dir / f"shard_{i // shard_size:04d}.parquet")


def main():
    ap = argparse.ArgumentParser(description="Загрузка HTML arXiv + упаковка в parquet-шарды.")
    ap.add_argument("--metadata", type=Path,
                    default=DATA / "metadata" / "arxiv_NLP_2021_2026_metadata.csv")
    ap.add_argument("--year-min", type=int, default=2021, help="Нижняя граница года (default: 2021)")
    ap.add_argument("--year-max", type=int, default=2026, help="Верхняя граница года (default: 2026)")
    ap.add_argument("--html-dir", type=Path, default=DATA / "raw" / "html")
    ap.add_argument("--parquet-dir", type=Path, default=DATA / "raw" / "parquet")
    ap.add_argument("--limit", type=int, default=None, help="Ограничить число статей (смоук)")
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--burst-size", type=int, default=8)
    ap.add_argument("--burst-pause", type=float, default=0.5)
    ap.add_argument("--stage", choices=["all", "download", "shard"], default="all")
    args = ap.parse_args()

    args.html_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "download"):
        df = pd.read_csv(args.metadata, usecols=["arxiv_id", "title", "published", "html_url"])
        df["published"] = pd.to_datetime(df["published"])
        years = df["published"].dt.year
        df_year = df[(years >= args.year_min) & (years <= args.year_max)]
        downloaded = {f.stem for f in args.html_dir.glob("*.html")}
        ids = [x for x in df_year["arxiv_id"] if x not in downloaded]
        if args.limit:
            ids = ids[:args.limit]
        print(f"Статей {args.year_min}–{args.year_max}: {len(df_year)} | уже скачано: {len(downloaded)} | "
              f"к загрузке: {len(ids)}")
        if ids:
            failed = asyncio.run(download_all(ids, args.html_dir, args.burst_size, args.burst_pause))
            if failed:
                err_path = DATA / "metadata" / "download_errors.csv"
                pd.DataFrame(failed).to_csv(err_path, index=False)
                print(f"Отчёт об ошибках → {err_path}")

    if args.stage in ("all", "shard"):
        pack_shards(args.html_dir, args.parquet_dir, args.shard_size)


if __name__ == "__main__":
    main()
