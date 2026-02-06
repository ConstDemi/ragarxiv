# ArXiv Info System: NLP Research Assistant (2025)

Вопросно-ответная RAG-система, построенная на корпусе научных статей категории cs.CL (Computation and Language) за 2025 год.

**Автор проекта:** [Демидов Константин](https://github.com/ConstDemi)  
**Руководитель проекта:** [Паточенко Евгений](https://github.com/evgpat)

**Статус проекта:** Baseline.

## Технологический стек
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Embeddings: `Qwen/Qwen3-Embedding-0.6B`
- Vector DB: `Qdrant (Dense vectors search)`
- Frameworks, libraries and tools: `arXiv API, Pandas, BeatifulSoup, Markdownify, Pyarrow (Parquet files), HF Datasets, Transformers, LangChain Text Splitters, Sentence Transformers, PyTorch, Streamlit, FastAPI, Uvicorn`
- Data Ops: `DVC, S3 storage`

## Архитектура и этапы пайплайна
1. Сбор метаданных и данных
    - Метаданные (`1metadata_parse.ipynb`): Парсинг с помощью arXiv API
    - Загрузка HTML (`2data_parse.ipynb`): Асинхронное скачивание HTML-версий статей. Фильтрация битых и маленьких (пустых) файлов

2. Препроцессинг и чанкование
    - Очистка (`3html_preprocess.ipynb`): Конвертация HTML в Markdown > удаление шума > сохранение формул в LaTeX формате.

    - Чанкование (`4chunking.ipynb`):
        1. Логическая нарезка по заголовкам (MarkdownHeaderTextSplitter)
        2. Рекурсивная нарезка (RecursiveCharacterTextSplitter)
        3. Удаление частей короче 50 символов

3. Индексация (`5embeddings.ipynb`)
    - Создание эмбеддингов и сохранение в Parquet-файл

4. База данных (`6db.ipynb`)
    - Подключаемся к локальному Qdrant серверу и заливаем туда данные


4. RAG (`rag_pipeline.py` + `main.py` + `frontend.py`)
    - Подтягиваем в `main.py` класс RAG'а
    - `main.py` загружает модели, поднимает backend
    - `frontend.py` поднимает интерфейс на Streamlit
    - Общение между микросервисами реализовано на FastAPI

## Структура репозитория
```
src/
├── 1metadata_parse.ipynb    # Парсинг метаданных с помощью API ArXiv
├── 2html_parse.ipynb        # Асинхронная скачка HTML статей
├── 3html_preprocess.ipynb   # HTML -> MD
├── 4chunking.ipynb          # Чанкование
├── 5embeddings.ipynb        # Генерация эмбеддингов и FAISS
├── 6db.ipynb                # Заливка обработнных данных в Qdrant
├── 7get_random_chunks.ipynb # Скрипт получения случайный чанков из Qdrant коллекции
├── 8RAG.ipynb               # Тестируем RAG в Jupyter Notebook
├── 9create_gd.ipynb         # Скрипт для создания эталонного датасета
├── 10rag_validation.ipynb   # Валидационный тест с помощью RAGAS
|
├── data_recover.py          # Скрипт восстановления Qdrant коллекции
├── rag_pipeline.py          # Класс RAG'а, импортируется в main.py
├── main.py                  # Бэкенд RAG'а
└── frontend.py              # Фронтенд RAG'а

data/                        # Папка для данных, которые участвуют во всём пайплайне от парсинга arxiv до заливки в Qdrant
├── metadata/                # CSV с метаданными статей
├── raw/                     # Сырые данные, требующие предобработки
└── processed/               # Обработанные данные, который заливаются в Qdrant

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

python ./src/data_recover.py
```

### Шаг 5: Запуск RAG системы (не закрывайте консоли во время работы RAG)

Первая консоль:
```bash
python ./src/main.py
```
Вторая консоль:
```bash
streamlit run ./src/frontend.py
```

RAG запущен и ждёт вас по адресу `http://localhost:8501/`

### Остановка RAG
Для остановки RAG просто закройте две консоли и удалите Docker контейнер
```bash
docker compose down -v
````

Для повторного запуска повторите Шаг 5
