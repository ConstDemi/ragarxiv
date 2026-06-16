# src/eval/metrics.py
# RAGAS-метрики: судья Claude + эмбеддер Qwen3.
# Модуль не зависит от config — имена моделей передаются из run.py.
import _compat  # noqa: F401 — заглушка vertexai до импорта ragas
import torch
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness, AnswerRelevancy


def build_judge(model: str, max_tokens: int = 4096):
    """LLM-судья на Claude. temperature НЕ передаём — Opus 4.8/4.7 её 400-ят."""
    return LangchainLLMWrapper(ChatAnthropic(model=model, max_tokens=max_tokens, timeout=120))


def build_embeddings(model: str):
    """Эмбеддер для AnswerRelevancy — локальный Qwen3 (без API).
    На GPU, если доступен: грузится ПОСЛЕ rag.cleanup(), т.е. VRAM уже свободна."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=model, model_kwargs={"device": device})
    )


def build_metrics(llm, emb):
    return [
        ContextPrecision(llm=llm),
        ContextRecall(llm=llm),
        Faithfulness(llm=llm),
        AnswerRelevancy(llm=llm, embeddings=emb),
    ]
