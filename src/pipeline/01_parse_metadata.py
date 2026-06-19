#!/usr/bin/env python3
"""Стадия 01 — метаданные arXiv.

Парсит arXiv API по категории (default cs.CL) помесячно за диапазон лет,
кладёт год-чекпоинты в data/metadata/temp_years/ и объединяет в один CSV
data/metadata/arxiv_NLP_<start>_<end>_metadata.csv.

Запуск:
    python src/pipeline/01_parse_metadata.py                          # 2021..текущий
    python src/pipeline/01_parse_metadata.py --start-year 2025 --end-year 2025
"""
import argparse
import calendar
import time
from datetime import datetime
from pathlib import Path

import arxiv
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "metadata"


def parse_year(client, cat, year):
    """Парсит один год помесячно → список статей (dict)."""
    year_papers = []
    now = datetime.now()
    for month in tqdm(range(1, 13), desc=f"Year {year}", leave=False):
        if year == now.year and month > now.month:
            break
        last_day = calendar.monthrange(year, month)[1]
        start_str = f"{year}{month:02d}01000000"
        end_str = f"{year}{month:02d}{last_day}235959"
        query = f"cat:{cat} AND submittedDate:[{start_str} TO {end_str}]"
        try:
            search = arxiv.Search(query=query, sort_by=arxiv.SortCriterion.SubmittedDate)
            for paper in client.results(search):
                if paper.primary_category == cat:
                    year_papers.append({
                        "arxiv_id": paper.get_short_id(),
                        "title": paper.title,
                        "authors": ", ".join(a.name for a in paper.authors),
                        "summary": paper.summary.replace("\n", " "),
                        "primary_category": paper.primary_category,
                        "published": paper.published.strftime("%Y-%m-%d"),
                        "updated": paper.updated.strftime("%Y-%m-%d"),
                        "entry_id": paper.entry_id,
                    })
        except Exception as e:
            print(f"Ошибка при парсинге {year}-{month:02d}: {e}")
            time.sleep(10)
    return year_papers


def main():
    ap = argparse.ArgumentParser(description="Парсинг метаданных arXiv по категории и годам.")
    ap.add_argument("--cat", default="cs.CL", help="Категория arXiv (default: cs.CL)")
    ap.add_argument("--start-year", type=int, default=2021)
    ap.add_argument("--end-year", type=int, default=datetime.now().year)
    ap.add_argument("--temp-dir", type=Path, default=META_DIR / "temp_years",
                    help="Папка год-чекпоинтов")
    ap.add_argument("--out", type=Path, default=None,
                    help="Итоговый CSV (default: data/metadata/arxiv_NLP_<start>_<end>_metadata.csv)")
    args = ap.parse_args()

    args.temp_dir.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or META_DIR / f"arxiv_NLP_{args.start_year}_{args.end_year}_metadata.csv"

    client = arxiv.Client(page_size=1000, delay_seconds=5, num_retries=5)

    print(f"Парсинг {args.cat} с {args.start_year} по {args.end_year}.")
    for year in range(args.start_year, args.end_year + 1):
        checkpoint = args.temp_dir / f"arxiv_{args.cat}_{year}.csv"
        if checkpoint.exists():
            print(f"Год {year} уже скачан ({checkpoint.name}). Пропускаем.")
            continue
        t0 = time.time()
        papers = parse_year(client, args.cat, year)
        if papers:
            df_year = pd.DataFrame(papers)
            df_year["html_url"] = "https://arxiv.org/html/" + df_year["arxiv_id"].astype(str)
            df_year.to_csv(checkpoint, index=False, encoding="utf-8")
            print(f"Год {year}: сохранено {len(df_year)} ({time.time() - t0:.1f}с)")
        else:
            print(f"Год {year}: статей не найдено.")

    # Объединение всех год-чекпоинтов категории
    csv_files = sorted(args.temp_dir.glob(f"arxiv_{args.cat}_*.csv"))
    if not csv_files:
        print("Файлы с данными не найдены.")
        return
    full_df = pd.concat([pd.read_csv(f) for f in tqdm(csv_files, desc="Merging")], ignore_index=True)
    full_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSUCCESS! {out_path} — всего статей: {len(full_df)}")


if __name__ == "__main__":
    main()
