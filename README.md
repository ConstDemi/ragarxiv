# ArXiv Info System: NLP Research Assistant

Вопросно-ответная RAG-система, построенная на корпусе научных статей категории cs.CL (Computation and Language) за 2025 год.

**Автор проекта:** [Демидов Константин](https://github.com/ConstDemi)  
**Руководитель проекта:** [Паточенко Евгений](https://github.com/evgpat)

**Статус проекта:** Baseline

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
├── 05_embedding_data.ipynb         # Генерация эмбеддингов и FAISS
├── 06_creating_database.ipynb      # Заливка обработнных данных в Qdrant
├── 07_get_random_chunks.ipynb      # Скрипт получения случайный чанков из Qdrant коллекции
├── 08_test_RAG.ipynb               # Тестируем RAG в Jupyter Notebook'
├── 09_EDA.ipynb                    # Исследовательский анализ данных (EDA)
├── 10_filling_golden_dataset.ipynb # Скрипт для заполнения эталонного датасета ответами RAG-системы
└── 11_validating_RAG.ipynb         # Валидационный тест с помощью RAGAS

src/app
├── data_recover.py          # Скрипт восстановления Qdrant коллекции
├── rag_pipeline.py          # Класс RAG'а, импортируется в main.py
├── main.py                  # Бэкенд RAG'а
└── frontend.py              # Фронтенд RAG'а

data/                        # Директория для всех данных проекта
├── eval/                    # Датасеты для валидации RAG'а
├── fig/                     # Графики
├── metadata/                # CSV с метаданными статей
├── processed/               # Обработанные данные, который заливаются в Qdrant
└── raw/                     # Сырые данные, требующие предобработки

arXiv_Presentation.pptx      # Презентация к проекту (TBD)
docker-compose.yaml          # Файл-оркестратор
requirements.txt             # Зависимости проекта
```


## Запускаем RAG:

### Предварительные требования
- Python 3.11+
- Git
- Docker
- 20 GB свободного места на диске

### Шаг 1: Клонирование репозитория
```bash
git clone https://github.com/ConstDemi/ArXiv_Info_System.git
cd ArXiv_Info_System
```

### Шаг 2: Установка зависимостей
```bash
pip install -r requirements.txt
```

### Шаг 3: Настройка доступа к S3 хранилищу

1. Создайте файл `.dvc/config.local` на основе шаблона `.dvc/config.local.example`

2. Отредактируйте `.dvc/config.local`, заполнив переменные предоставленными credentials (доступы от S3 хранилища).

**Для членов комиссии:** credentials предоставляются отдельно.

### Шаг 4: Загрузка датасета (там Qdrant снапшот)
```bash
dvc pull
```

### Шаг 5: Восстанавливаем снапшот Qdrant коллеции

```bash
docker compose up -d

python ./src/app/data_recover.py
```

### Шаг 5: Запуск RAG системы (не закрывайте консоли во время работы RAG)

Первая консоль:
```bash
python ./src/app/main.py
```
Вторая консоль:
```bash
streamlit run ./src/app/frontend.py
```

RAG запущен и ждёт вас по адресу `http://localhost:8501/`

### Остановка RAG
Для остановки RAG просто закройте две консоли и удалите Docker контейнер
```bash
docker compose down -v
````

Для повторного запуска повторите Шаг 5