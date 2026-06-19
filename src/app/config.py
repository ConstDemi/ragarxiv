# config.py
# Простые настройки проекта. Правятся прямо здесь (без env/.env).
# Когда понадобится переопределять через окружение (например, в Docker) —
# мигрируем на pydantic-settings.

# --- Retrieval ---
RETRIEVE_K = 30    # сколько чанков тянуть из Qdrant (пул для группировки источников)
CONTEXT_K = 10     # сколько лучших чанков отдавать в контекст LLM
MAX_PAPERS = 8     # сколько статей максимум возвращать в sources

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "nlp2025_chunks"

# --- Models ---
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"   # 7B 4-bit — выбран по eval: faithfulness 0.70→0.82 vs 3B (4-bit включается на CUDA)

# --- Generation ---
MAX_NEW_TOKENS = 2000    # максимальная длина ответа (в токенах)
MAX_INPUT_TOKENS = 4096  # предел токенов на вход (контекст обрезается)

# Baseline-промпт (дефолт). Промпт-ось на 7B исчерпана и закрыта:
#   prompt-grounded-v2 (4 правила блоком) → регресс (faithfulness 0.83→0.70, over-refusal + дрейф языка);
#   prompt-lang-v2 (клауза языка)          → ноль (relevancy +0.005, англ 9→8).
# Вывод: 7B плохо слушает нюансы системного промпта → правки промпта не ведём. Детали — раны в MLflow.
SYSTEM_PROMPT = (
    "You are a helpful scientific assistant with knowledge base of NLP arxiv paper for 2025 year. "
    "Use ONLY the provided context to answer the user's question. "
    "If the context doesn't contain enough information, say so explicitly"
)

# --- Eval (LLM-судья RAGAS) ---
JUDGE_MODEL = "claude-haiku-4-5"                  # ЕДИНЫЙ источник модели судьи (run.py берёт только отсюда)
JUDGE_MAX_TOKENS = 4096                           # лимит ответа судьи
GOLDEN_PATH = "data/eval/golden_dataset50.parquet"  # 50 QA-пар (25 Q1 / 25 Q2), относительно корня репо
EVAL_MAX_WORKERS = 4                              # параллелизм RAGAS, параллелит jobs (кол-во метрик X кол-во вопросов); подбираем под лимиты запросов в Claude API
