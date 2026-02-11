# ArXiv Info System: NLP Research Assistant

Вопросно-ответная RAG-система, построенная на корпусе научных статей категории cs.CL (Computation and Language) за 2025 год.

**Автор проекта:** [Демидов Константин](https://github.com/ConstDemi)  
**Руководитель проекта:** [Паточенко Евгений](https://github.com/evgpat)

**Статус проекта:** Baseline-Ready + Docker image

## Технологический стек
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Vector DB: `Qdrant (Dense vectors search)`
- Frameworks, libraries and tools: `arXiv API, Pandas, BeatifulSoup, Markdownify, Pyarrow (Parquet files), HF Datasets, Transformers, LangChain Text Splitters, Sentence Transformers, PyTorch, Streamlit, FastAPI, Uvicorn`
- Data Ops: `DVC, S3 storage`

## Архитектура и этапы пайплайна
1. Сбор метаданных и данных
    - Метаданные (`01_parse_metadata.ipynb`): Парсинг с помощью arXiv API
    - Загрузка HTML (`02_parse_data.ipynb`): Асинхронное скачивание HTML-версий статей. Фильтрация битых и маленьких (пустых) файлов

2. Препроцессинг и чанкование
    - Очистка (`03_preprocess_data.ipynb`): Конвертация HTML в Markdown > удаление шума > сохранение формул в LaTeX формате.

    - Чанкование (`04_chunking_data.ipynb`):
        1. Логическая нарезка по заголовкам (MarkdownHeaderTextSplitter)
        2. Рекурсивная нарезка (RecursiveCharacterTextSplitter)
        3. Удаление частей короче 50 символов

3. Индексация (`05_embedding_data.ipynb`)
    - Создание эмбеддингов и сохранение в Parquet-файл

4. База данных (`06_creating_database.ipynb`)
    - Подключаемся к локальному Qdrant серверу и заливаем туда данные


4. RAG (`rag_pipeline.py` + `main.py` + `frontend.py`)
    - Подтягиваем в `main.py` класс RAG'а из `rag_pipeline.py`
    - `main.py` загружает модели, поднимает backend
    - `frontend.py` поднимает интерфейс на Streamlit
    - Общение между микросервисами реализовано на FastAPI

## Структура репозитория
```
src/Jupyter Notebooks
├── 01_parse_metadata.ipynb         # Парсинг метаданных с помощью API ArXiv
├── 02_parse_data.ipynb             # Асинхронная скачка HTML статей
├── 03_preprocess_data.ipynb        # HTML -> Markdown
├── 04_chunking_data.ipynb          # Чанкование
├── 05_embedding_data.ipynb         # Генерация эмбеддингов
├── 06_creating_database.ipynb      # Заливка обработнных данных в Qdrant
├── 07_get_random_chunks.ipynb      # Скрипт получения случайный чанков из Qdrant коллекции
├── 08_test_RAG.ipynb               # Тестируем RAG в Jupyter Notebook
├── 09_EDA.ipynb                    # Исследовательский анализ данных (EDA)
├── 10_filling_golden_dataset.ipynb # Скрипт для заполнения эталонного датасета ответами RAG-системы
└── 11_validating_RAG.ipynb         # Валидационный тест с помощью RAGAS (Используются облачные GPU)

src/app
├── data_recover.py          # Скрипт восстановления Qdrant коллекции
├── rag_pipeline.py          # Класс RAG'а, импортируется в main.py
├── main.py                  # Бэкенд RAG'а
└── frontend.py              # Фронтенд RAG'а

data/                        # Директория для всех данных проекта (некоторые директории создаются в Juputer Notebooks)
├── eval/                    # Датасеты для валидации RAG'а
├── fig/                     # Графики
├── metadata/                # CSV с метаданными статей
├── processed/               # Обработанные данные, который заливаются в Qdrant
└── raw/                     # Сырые данные, требующие предобработки

arXiv_Presentation.pptx      # Презентация к проекту
docker-compose.yaml          # Файл-оркестратор
requirements.docker.txt      # Зависимости
```


## Запускаем RAG:

### Рекомендуемые системные требования
- 8 VRAM
- До 32 Гб ОЗУ
- 20 GB свободного места на диске
- Git
- Docker


### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/ConstDemi/ArXiv_Info_System
cd ArXiv_Info_System
```

### Шаг 2: Скачиваем данные с S3

1. Положите файл `config.local` в папку `.dvc/` (Файл предоставляется отдельно)

2. Если у вас не установлен DVC: `pip install dvc[s3]`

3. Выполните команду `dvc pull` (Загрузка данных займёт некоторое время)


### Шаг 3: Скачиваем образ и поднимаем контейнер

```bash
docker compose up
```
- Логи можно смотреть командой `docker logs rag-app -f`
- При первом запуске контейнера подождите ~5 минут пока подготовятся все сервисы

После того как сервисы запустятся, RAG ждёт по адресу `localhost:8501`



Выключить RAG можно командой
```bash
docker compose down
````


Дополнительно:
- Образ проекта на Docker Hub - https://hub.docker.com/r/constdemi/ragarxiv
- Фреймворк RAGAS: https://docs.ragas.io/en/stable/ (Paper: https://arxiv.org/abs/2309.15217)
