# RAGarxiv — научный RAG-ассистент по arXiv (NLP)

RAG-сервис, помогающий исследователю «сёрфить» статьи arXiv по NLP. Отвечает на два класса вопросов:

- **Q1** — «существует ли статья о X» (наличие работы / датасета / бенчмарка);
- **Q2** — «какие работы помогают с проблемой X и как» (метод + результат).

Ответ всегда опирается на найденные статьи и возвращает источники (arXiv ID + ссылка + релевантные фрагменты).

> **Статус: Фаза 2** (`v0.2.0`). Корпус — **2021–2026** (~4 039 статей / ~149K чанков); при индексации к тексту чанка добавляются заголовки его раздела (поле `embed_text`) — это улучшает ранжирование. Оценка качества — на **MLflow GenAI**.
> Дорожная карта — [`docs/ROADMAP.md`](docs/ROADMAP.md); журнал A/B-экспериментов — [`docs/experiments.md`](docs/experiments.md).

## Как устроено

```
запрос
  → эмбеддинг (Qwen3-Embedding-0.6B)
  → поиск top-k в Qdrant (коллекция nlp2021_2026_embedtext, 1024-d, cosine)
  → контекст (top context_k чанков)
  → генерация (Qwen2.5-7B-Instruct, 4-bit NF4)
  → ответ + источники (сгруппированы по статьям)
```

- **Генератор:** `Qwen2.5-7B-Instruct` (4-bit) — выбран по результатам оценки (метрика faithfulness 0.70 → 0.83 против модели 3B).
- **Эмбеддер:** `Qwen3-Embedding-0.6B` (GPU); документы кодируются вместе с заголовками их раздела (поле `embed_text`).
- **Векторная БД:** Qdrant.
- **Оценка:** MLflow GenAI с LLM-судьёй (Claude Haiku, через `litellm`).
- Единый конфиг проекта — [`src/config.py`](src/config.py).

## Структура

```
src/
  config.py           # единый конфиг: модели, Qdrant, поиск, генерация, оценка
  app/
    main.py           # FastAPI: POST /ask, POST /search
    rag_pipeline.py   # ScienceRAG: поиск → генерация
  eval/
    run.py            # оценка на MLflow GenAI (две фазы: генерация → судьи)
    metrics.py        # метрики MLflow (LLM-судья Claude) + свой doc_recall
    doc_recall.py     # диагностика поиска без LLM-судьи (doc_recall@k + MRR)
  pipeline/           # офлайн-сборка корпуса, стадии 01→06
  notebooks/
    07_EDA.ipynb      # разведочный анализ корпуса
docs/                 # ROADMAP.md, experiments.md
data/                 # метаданные / raw / processed / golden-эталон (gitignored)
docker-compose.yaml   # Qdrant
requirements.txt      # пины (runtime + eval + pipeline)
.env.example          # шаблон окружения
```

## Требования

- Python 3.10 (рекомендуемое имя conda-env — `ragarxiv`).
- NVIDIA GPU ≥ 8 ГБ + драйвер (torch — сборка `cu126`); работает и на CPU, но медленно.
- Docker — для Qdrant.

## Установка

```bash
# 1) окружение
conda create -n ragarxiv python=3.10 && conda activate ragarxiv   # или python -m venv
pip install -r requirements.txt                          # torch ставится с индекса cu126 (он прописан в файле)

# 2) ключ Anthropic — нужен ТОЛЬКО LLM-судье при оценке (приложению не требуется)
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
- `POST /search` → `{sources}` — только поиск, без генерации.

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"text": "Есть ли работа о влиянии шума в дифференциально-приватном переписывании текста?"}'
```

> Требует поднятого Qdrant с коллекцией `nlp2021_2026_embedtext` (см. ниже) и GPU для загрузки 7B.

## Сборка корпуса (офлайн, [`src/pipeline/`](src/pipeline))

Стадии запускаются по порядку; у каждой есть `--help` и флаг `--limit` для пробного прогона. Поток данных:

```
01 метаданные → 02 HTML+parquet → 03 markdown → 04 чанки → 05 эмбеддинги → 06 индекс Qdrant
```

```bash
python src/pipeline/01_parse_metadata.py      # arXiv API → data/metadata/*.csv      (сеть)
python src/pipeline/02_parse_data.py          # скачивание HTML + parquet-файлы       (сеть)
python src/pipeline/03_preprocess.py          # HTML → markdown
python src/pipeline/04_chunk.py               # markdown → чанки (512 ток., перекрытие 50)
python src/pipeline/05_embed.py --text-field embed_text   # чанки → эмбеддинги    (GPU)
python src/pipeline/06_index_qdrant.py        # → коллекция nlp2021_2026_embedtext    (Qdrant)
```

> Корпус — 2021–2026 (стадия 02 по умолчанию; диапазон `--year-min/--year-max`). Каталог `data/` целиком gitignored.

## Оценка (MLflow GenAI, [`src/eval/`](src/eval))

Оценка идёт в две фазы: сначала `ScienceRAG` генерирует ответы на эталонные вопросы (обычный цикл на GPU),
затем `mlflow.genai.evaluate` оценивает их — встроенные LLM-судьи Claude (через `litellm`) плюс
детерминированный `doc_recall` (попал ли нужный документ в выдачу). Эталонный набор (golden) — 50 пар
«вопрос–ответ» (25 Q1 / 25 Q2, `data/eval/golden_dataset50.parquet`), хранится локально, в git не входит.

```bash
python src/eval/run.py                          # прогон (что запускать — задано в __main__)
mlflow ui --backend-store-uri ./mlruns          # просмотр результатов
python src/eval/doc_recall.py                   # быстрая проверка поиска (без LLM-судьи, бесплатно)
```

Текущий результат (прогон `gen-7b-v3` в MLflow; корпус 2021–2026, `embed_text`):

| метрика | что измеряет | значение |
|---|---|---|
| groundedness | ответ опирается на найденный контекст (без выдумок) | 0.84 |
| relevance | найденные документы релевантны вопросу | 0.61 |
| sufficiency | контекста достаточно, чтобы ответить | 0.78 |
| relevance_to_query | ответ по существу вопроса | 0.96 |
| doc_recall | нужный документ попал в контекст | 0.94 |

> ⚠️ В `v0.2.0` инструмент оценки сменился (RAGAS → MLflow GenAI): у них разные LLM-судьи и формулировки,
> поэтому значения метрик нельзя напрямую сравнивать с `v0.1.0`. Одинаково считается только `doc_recall` —
> по нему сравнение корректно.

Подробности метода и история A/B-экспериментов — [`docs/experiments.md`](docs/experiments.md).

## Лицензия

См. [`LICENSE`](LICENSE).
