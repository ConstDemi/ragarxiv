# RAGarxiv — научный RAG-ассистент по arXiv (NLP)

RAG-сервис, помогающий исследователю «сёрфить» статьи arXiv по NLP. Отвечает на два класса вопросов:

- **Q1** — «существует ли статья о X» (наличие работы / датасета / бенчмарка);
- **Q2** — «какие работы помогают с проблемой X и как» (метод + результат).

Ответ всегда опирается на найденные статьи и возвращает источники (arXiv ID + ссылка + релевантные пассажи).

> **Статус: Фаза 1** — фиксация воспроизводимого baseline. Корпус — **только 2025 г.** (~766 статей / ~33K чанков).
> Дорожная карта — [`docs/ROADMAP.md`](docs/ROADMAP.md); журнал A/B-экспериментов — [`docs/experiments.md`](docs/experiments.md).

## Как устроено

```
запрос
  → эмбеддинг (Qwen3-Embedding-0.6B)
  → ретрив top-k из Qdrant (коллекция nlp2025_chunks, 1024-d, cosine)
  → контекст (top context_k чанков)
  → генерация (Qwen2.5-7B-Instruct, 4-bit NF4)
  → ответ + источники (сгруппированы по статьям)
```

- **Генератор:** `Qwen2.5-7B-Instruct` (4-bit) — выбран по eval (faithfulness 0.70 → 0.83 vs 3B).
- **Эмбеддер:** `Qwen3-Embedding-0.6B` (GPU).
- **Векторная БД:** Qdrant.
- **Судья eval:** Claude Haiku (RAGAS).
- Единый конфиг проекта — [`src/config.py`](src/config.py).

## Структура

```
src/
  config.py           # единый конфиг: модели, Qdrant, retrieval, generation, eval
  app/
    main.py           # FastAPI: POST /ask, POST /search
    rag_pipeline.py   # ScienceRAG: retrieve → generate
  eval/
    run.py            # харнес RAGAS + MLflow (дёргает ScienceRAG напрямую)
    metrics.py        # судья (Claude) + метрики RAGAS
    _compat.py        # шим совместимости ragas/langchain
  pipeline/           # офлайн-сборка корпуса, стадии 01→06
  notebooks/
    07_EDA.ipynb      # разведочный анализ корпуса
docs/                 # ROADMAP.md, experiments.md
data/                 # метаданные / raw / processed / eval-golden (gitignored)
docker-compose.yaml   # Qdrant
requirements.txt      # пины (runtime + eval + pipeline)
.env.example          # шаблон окружения
```

## Требования

- Python 3.10 (рекомендуемое имя conda-env — `ragarxiv`).
- NVIDIA GPU ≥ 8 ГБ + драйвер (torch — сборка `cu126`); есть CPU-fallback, но медленно.
- Docker — для Qdrant.

## Установка

```bash
# 1) окружение
conda create -n ragarxiv python=3.10 && conda activate ragarxiv   # или python -m venv
pip install -r requirements.txt                          # torch ставится с индекса cu126 (он прописан в файле)

# 2) ключ Anthropic — нужен ТОЛЬКО eval-судье (приложению не требуется)
cp .env.example .env                                     # впиши ANTHROPIC_API_KEY

# 3) Qdrant
docker compose up -d                                     # поднимает Qdrant на :6333
```

## Запуск API

```bash
cd src/app
uvicorn main:app --host 0.0.0.0 --port 8000
```

Эндпоинты:
- `POST /ask` → `{answer, sources}` — ответ LLM + источники;
- `POST /search` → `{sources}` — только ретрив, без генерации.

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"text": "Есть ли работа о влиянии шума в дифференциально-приватном переписывании текста?"}'
```

> Требует поднятого Qdrant с коллекцией `nlp2025_chunks` (см. ниже) и GPU для загрузки 7B.

## Сборка корпуса (офлайн, [`src/pipeline/`](src/pipeline))

Стадии запускаются по порядку; у каждой есть `--help` и флаг `--limit` для смоука. Поток данных:

```
01 метаданные → 02 HTML+шарды → 03 markdown → 04 чанки → 05 эмбеддинги → 06 индекс Qdrant
```

```bash
python src/pipeline/01_parse_metadata.py      # arXiv API → data/metadata/*.csv      (сеть)
python src/pipeline/02_parse_data.py          # download HTML + parquet-шарды         (сеть)
python src/pipeline/03_preprocess.py          # HTML → markdown
python src/pipeline/04_chunk.py               # markdown → чанки (512 ток., overlap 50)
python src/pipeline/05_embed.py               # чанки → эмбеддинги                    (GPU)
python src/pipeline/06_index_qdrant.py        # → коллекция nlp2025_chunks            (Qdrant)
```

> Корпус сейчас — только 2025 г. (стадия 02 фильтрует год). Каталог `data/` целиком gitignored.

## Оценка (RAGAS + MLflow, [`src/eval/`](src/eval))

Харнес дёргает `ScienceRAG` напрямую, гоняет golden-набор (50 QA-пар: 25 Q1 / 25 Q2) и считает метрики
RAGAS (судья Claude) + детерминированный `doc_recall` по `doc_id`. Golden
(`data/eval/golden_dataset50.parquet`) — локальный, gitignored.

```bash
# впиши вызов track_run(...) в блок __main__ файла src/eval/run.py, затем:
python src/eval/run.py
mlflow ui --backend-store-uri ./mlruns        # просмотр прогонов
```

Текущий baseline — `gen-7b-v2`:

| метрика | значение |
|---|---|
| faithfulness | 0.833 |
| answer_relevancy | 0.719 |
| context_precision | 0.673 |
| context_recall | 0.894 |
| doc_recall@context_k | 0.96 |
| doc_recall@retrieved | 0.98 |

Подробности метода и история A/B — [`docs/experiments.md`](docs/experiments.md).

## Лицензия

См. [`LICENSE`](LICENSE).
