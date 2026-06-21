# src/app/gradio_app.py
# Локальная chat-демка (gr.ChatInterface) поверх ScienceRAG — тот же движок, что API и eval.
# Stateless: история сообщений видна в UI, но в модель НЕ подкладывается (промпт не растёт →
# память как у обычного ответа, ~6.74 ГБ). Ответ стримится; источники — сворачиваемым блоком.
#
# Запуск:  python src/app/gradio_app.py   →  http://localhost:7860
# Нужны:   поднятый Qdrant (config.COLLECTION_NAME) + GPU.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src/ → config
import config
from rag_pipeline import ScienceRAG

import gradio as gr

RAG = None  # единый экземпляр движка; создаётся в __main__ перед launch

EXAMPLES = [
    "Есть ли работа о влиянии шума в дифференциально-приватном переписывании текста?",
    "Какие работы снижают вычислительную стоимость самокоррекции в text-to-SQL и как?",
    "Существует ли бенчмарк ASR-моделей для африканских языков?",
    "Как улучшить рабочую память многошагового RAG для сложных рассуждений?",
]

DESCRIPTION = (
    "Научный RAG-ассистент по статьям **arXiv (NLP, 2021–2026)**. Ответ опирается только на "
    "найденные статьи и приводит источники.  \n"
    "**Q1** — «есть ли работа о X» · **Q2** — «какие работы помогают с X и как». "
    "_Контекст к модели — по одному вопросу за раз (история в модель не подаётся)._"
)


def build_rag():
    """Тот же движок и настройки, что в main.py (см. lifespan)."""
    return ScienceRAG(
        qdrant_host=config.QDRANT_HOST, qdrant_port=config.QDRANT_PORT,
        collection_name=config.COLLECTION_NAME, embed_model=config.EMBED_MODEL,
        llm_model=config.LLM_MODEL, system_prompt=config.SYSTEM_PROMPT,
        max_new_tokens=config.MAX_NEW_TOKENS, max_input_tokens=config.MAX_INPUT_TOKENS,
        retrieve_k=config.RETRIEVE_K, context_k=config.CONTEXT_K, max_papers=config.MAX_PAPERS,
    )


def _sources_md(sources) -> str:
    """Список статей (контракт PaperSource) → markdown для сворачиваемого блока источников."""
    lines = []
    for i, p in enumerate(sources, 1):
        title = p.get("title") or "(без заголовка)"
        url = p.get("url") or ""
        head = f"**{i}. [{title}]({url})**" if url else f"**{i}. {title}**"
        meta = f"score {p.get('score', 0.0):.3f}"
        if p.get("published"):
            meta += f" · {p['published']}"
        if p.get("doc_id"):
            meta += f" · {p['doc_id']}"
        snippet = ""
        chunks = p.get("chunks") or []
        if chunks:
            c = chunks[0]
            sec = " › ".join(h for h in (c.get("header_1"), c.get("header_2"), c.get("header_3")) if h)
            text = (c.get("text") or "").strip().replace("\n", " ")
            if len(text) > 300:
                text = text[:300] + "…"
            snippet = f"  \n  > {text}" + (f"  \n  _{sec}_" if sec else "")
        lines.append(f"{head}  \n{meta}{snippet}")
    return "\n\n".join(lines)


def respond(message, history):
    """Чат-обработчик (stateless: history игнорируем). Стримит ответ, в конце добавляет
    сворачиваемый блок источников. yield списка ChatMessage — Gradio заменяет реплику ассистента."""
    q = (message or "").strip()
    if not q:
        yield [gr.ChatMessage(role="assistant", content="Введите вопрос.")]
        return
    try:
        acc, srcs = "", []
        for partial, s in RAG.answer_stream(q):   # ретрив + потоковая генерация (без истории)
            acc, srcs = partial, s
            yield [gr.ChatMessage(role="assistant", content=acc or "…")]
        final = [gr.ChatMessage(role="assistant", content=acc or "_(пустой ответ)_")]
        if srcs:
            final.append(gr.ChatMessage(role="assistant", content=_sources_md(srcs),
                                        metadata={"title": f"📚 Источники ({len(srcs)})"}))
        yield final
    except Exception as e:
        yield [gr.ChatMessage(role="assistant", content=f"⚠️ Ошибка: {e}")]


# Читаемость: чёткий системный шрифт + крупнее базовый текст + просторный межстрочный.
THEME = gr.themes.Soft(
    primary_hue="indigo",
    font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "Arial", "sans-serif"],
    text_size=gr.themes.sizes.text_lg,
)
CSS = """
.gradio-container { font-size: 16px; }
.message .md, .message .prose, .prose, .md p, .md li { font-size: 16px !important; line-height: 1.7 !important; }
"""

demo = gr.ChatInterface(
    fn=respond,
    type="messages",
    title="🔬 RAGarxiv — научный RAG-ассистент по arXiv (NLP)",
    description=DESCRIPTION,
    examples=EXAMPLES,
    cache_examples=False,          # не гонять модель на примерах при старте
    theme=THEME,
    css=CSS,
    fill_height=True,
)


if __name__ == "__main__":
    print("[gradio] загрузка ScienceRAG (модель + эмбеддер, ~1–2 мин)...")
    RAG = build_rag()
    # default_concurrency_limit=1 — на 8 ГБ GPU не пускаем две генерации разом (анти-OOM).
    demo.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860,
                                                   show_error=True)
