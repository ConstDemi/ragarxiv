#!/usr/bin/env python3
"""Стадия 03 — очистка HTML → Markdown.

Читает шарды data/raw/parquet/*.parquet (doc_id, html), чистит LaTeXML-HTML
(BeautifulSoup + markdownify: математика → $...$/$$...$$, картинки/ссылки,
выкидывание References/Acknowledgements/Bibliography) и сохраняет
data/processed/parquet/processed_data.parquet (doc_id, md).

Запуск:
    python src/pipeline/03_preprocess.py
    python src/pipeline/03_preprocess.py --limit 20    # смоук на 20 документах
"""
import argparse
import os
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def process_batch(batch):
    """HTML → Markdown для батча. Импорты внутри — для num_proc (отдельные процессы)."""
    import re
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md

    USELESS_ALT_TEXTS = {"refer to caption", "[uncaptioned image]", "uncaptioned image", ""}
    MIN_FIGURE_ALT_LEN = 5
    TARGET_SECTIONS = re.compile(r"^(acknowledge?ments?|references|bibliography)$", re.IGNORECASE)
    LEADING_NUMBERING = re.compile(r"^\s*[\dIVXLCivxlc]+(\.[\dIVXLCivxlc]+)*\.?\s*")

    processed_htmls = []
    for html_text in batch["html"]:
        # 1. Валидация
        if not html_text or "Conversion to HTML had a Fatal error" in html_text:
            processed_htmls.append(None)
            continue
        soup = BeautifulSoup(html_text, "lxml")
        article = soup.find("article") or soup.find(class_="ltx_page_content")
        if not article:
            processed_htmls.append(None)
            continue

        # 2. Мусорные блоки
        garbage_classes = [
            "ltx_bibliography", "ltx_authors", "ltx_role_footnotetext",
            "ltx_navigation", "ltx_page_footer", "ltx_is_toc",
            "mobile-nav", "header", "ltx_pagination", "ltx_ERROR",
        ]
        for cls in garbage_classes:
            for tag in article.find_all(class_=cls):
                tag.decompose()

        # Acknowledgements / References / Bibliography по тексту заголовка
        # (устойчиво к нумерации "6 References" и вложенным span'ам)
        for header in article.find_all(["h1", "h2", "h3"]):
            heading_text = header.get_text(" ", strip=True)
            cleaned = LEADING_NUMBERING.sub("", heading_text).strip()
            if TARGET_SECTIONS.match(cleaned):
                section = header.find_parent("section")
                if section:
                    section.decompose()

        # 3. Base64 и технический мусор
        for tag in article.find_all("img", src=re.compile(r"^data:")):
            tag.decompose()
        for tag in article.find_all("a", href=re.compile(r"^data:")):
            tag.decompose()

        # 4. Ссылки: текст оставляем, тег убираем
        for a in article.find_all("a"):
            if a.get_text(strip=True):
                a.unwrap()
            else:
                a.decompose()

        # 5. Картинки: осмысленный alt → [Figure: ...], иначе выкидываем
        for img in article.find_all("img"):
            alt = img.get("alt", "")
            if alt.strip().lower() in USELESS_ALT_TEXTS:
                img.decompose()
            elif "math" in str(img.get("class", [])) or len(alt) > MIN_FIGURE_ALT_LEN:
                img.replace_with(f" [Figure: {alt}] ")
            else:
                img.decompose()

        # 6. Математика (LaTeX): inline → $...$, display → $$...$$
        math_registry = {}
        for i, math in enumerate(article.find_all(class_="ltx_Math")):
            latex = math.get("alttext", "")
            if latex:
                placeholder = f"LATEXPH{i}LATEXPH"
                delim = "$$" if math.get("display") == "block" else "$"
                math_registry[placeholder] = f"{delim}{latex}{delim}"
                math.replace_with(f" {placeholder} ")

        # 7. → Markdown, 8. возврат математики + зачистка переносов
        markdown_text = md(str(article), heading_style="ATX")
        for placeholder, original_latex in math_registry.items():
            markdown_text = markdown_text.replace(placeholder, original_latex)
        markdown_text = re.sub(r"\n\s*\n", "\n\n", markdown_text).strip()
        processed_htmls.append(markdown_text)

    batch["html"] = processed_htmls
    return batch


def main():
    ap = argparse.ArgumentParser(description="Очистка HTML → Markdown.")
    ap.add_argument("--input-dir", type=Path, default=DATA / "raw" / "parquet")
    ap.add_argument("--output", type=Path,
                    default=DATA / "processed" / "parquet" / "processed_data.parquet")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--num-proc", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="Взять только N документов (смоук)")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("parquet", data_files=str(args.input_dir / "*.parquet"))["train"]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Документов на вход: {len(ds)}")

    ds = ds.map(process_batch, batched=True, batch_size=args.batch_size, num_proc=args.num_proc)
    ds = ds.rename_column("html", "md")

    total = len(ds)
    if total:
        failed = sum(1 for x in ds["md"] if not x)
        converted = total - failed
        print(f"Сконвертировано в MD: {converted} ({converted / total * 100:.1f}%) | "
              f"провалов/пусто: {failed} ({failed / total * 100:.1f}%)")

    ds.to_parquet(str(args.output))
    print(f"SUCCESS! {args.output}")


if __name__ == "__main__":
    main()
