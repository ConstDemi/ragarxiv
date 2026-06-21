# src/eval/run.py
# Оценочный харнес на MLflow GenAI: дёргает ScienceRAG напрямую, гоняет golden,
# судит встроенными MLflow-scorers (судья Claude через anthropic:/) + детерминированный doc_recall.
# Без ragas/langchain (и без шима _compat).
import os
# ДО любого импорта (mlflow тянет torch раньше rag_pipeline) → флаг точно активен; меньше
# фрагментации VRAM и ползучего OOM на длинной серии генераций на 8 ГБ.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import logging
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # metrics
sys.path.insert(0, str(HERE.parent / "app"))     # rag_pipeline
sys.path.insert(0, str(HERE.parent))             # config

from dotenv import load_dotenv
load_dotenv(HERE.parent.parent / ".env")         # ANTHROPIC_API_KEY из .env в корне репо

import pyarrow.parquet as pq
import mlflow
from mlflow.entities import Document, SpanType

import config
from rag_pipeline import ScienceRAG
from metrics import build_scorers

GOLDEN = HERE.parent.parent / config.GOLDEN_PATH
mlflow.set_tracking_uri((HERE.parent.parent / "mlruns").as_uri())   # стор в корне репо

for _n in ("httpx", "anthropic", "litellm", "LiteLLM", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)


def _dataset(limit=None):
    """golden (v2) → формат mlflow.genai. relevant_doc_ids — мульти-релевантность для doc_recall
    (попадание = любой из них); expected_facts — атомарные факты для RetrievalSufficiency.
    Фолбэк на старую single-gold схему. Возвращает (rows, data)."""
    rows = pq.read_table(GOLDEN).to_pylist()
    if limit:
        rows = rows[:limit]
    data = [
        # inputs.question → аргумент predict_fn; expectations → эталон для судей:
        #   relevant_doc_ids — для doc_recall; expected_facts — для RetrievalSufficiency
        {"inputs": {"question": r["question"]},
         "expectations": {
             "relevant_doc_ids": list(r.get("relevant_doc_ids") or [r["doc_id"]]),
             "expected_facts": list(r.get("expected_facts") or [r["ground_truth"]]),
         }}
        for r in rows
    ]
    return rows, data


# --- Как устроен прогон: ДВУХФАЗНО (критично на 8 ГБ GPU) ---
# mlflow.genai.evaluate гоняет predict_fn в НЕСКОЛЬКО потоков И включает auto-tracing, который
# удерживает GPU-тензоры генерации → ползучий OOM (диагноз: простой цикл генерит 50 ровно на
# 6.75 ГБ, а через evaluate память доползает до OOM ~на 13-м вопросе).
# Поэтому: ФАЗА 1 — генерируем все ответы простым последовательным циклом (без evaluate → без
# течи), кэшируем и ОСВОБОЖДАЕМ 7B. ФАЗА 2 — evaluate уже по кэшу: predict_fn без генерации,
# только строит RETRIEVER-span для судей; autolog'у нечего держать, GPU свободна.
def evaluate_config(limit=None, scorers=None, **overrides):
    """Оценка одного конфига. Возвращает (rows, EvaluationResult).
    overrides: collection_name / llm_model / system_prompt / retrieve_k / context_k / max_papers.
    scorers: список scorer'ов (для дешёвого doc_recall-проба); по умолчанию — все судьи."""
    collection = overrides.get("collection_name", config.COLLECTION_NAME)
    rag = ScienceRAG(
        qdrant_host=config.QDRANT_HOST, qdrant_port=config.QDRANT_PORT,
        collection_name=collection, embed_model=config.EMBED_MODEL,
        llm_model=overrides.get("llm_model", config.LLM_MODEL),
        system_prompt=overrides.get("system_prompt", config.SYSTEM_PROMPT),
        max_new_tokens=config.MAX_NEW_TOKENS, max_input_tokens=config.MAX_INPUT_TOKENS,
        retrieve_k=overrides.get("retrieve_k", config.RETRIEVE_K),
        context_k=overrides.get("context_k", config.CONTEXT_K),
        max_papers=overrides.get("max_papers", config.MAX_PAPERS),
        rerank=overrides.get("rerank", config.RERANK_ENABLED),
        rerank_model=config.RERANK_MODEL, rerank_pool=config.RERANK_POOL,
    )
    rows, data = _dataset(limit)

    # ФАЗА 1: генерация простым циклом (без mlflow.evaluate → без autolog-течи GPU). Кэшируем ответ+контекст.
    print(f"[eval] коллекция={collection} | генерация {len(rows)} ответов (последовательно)...")
    cache = {}
    for i, r in enumerate(rows, 1):
        res = rag.answer(query=r["question"])
        cache[r["question"]] = {"answer": res["answer"], "ctx": res["context_chunks"]}
        if i % 10 == 0:
            print(f"[eval]   {i}/{len(rows)}")
    rag.cleanup()   # освобождаем 7B/эмбеддер: фаза судей — это Claude API, GPU не нужна

    # ФАЗА 2: predict_fn БЕЗ генерации — отдаёт кэш и строит RETRIEVER-span (для RAG-судей и doc_recall).
    #   Document.id = doc_id; span = top-context_k (что видела LLM, v2-методика).
    @mlflow.trace
    def predict_fn(question):
        c = cache[question]
        with mlflow.start_span(name="retrieve", span_type=SpanType.RETRIEVER) as s:
            s.set_inputs({"query": question})
            s.set_outputs([
                Document(id=x.get("doc_id", ""), page_content=x.get("text", ""),
                         metadata={"title": x.get("title", ""), "score": x.get("score")})
                for x in c["ctx"]
            ])
        return {"response": c["answer"]}

    print(f"[eval] судья={config.JUDGE_MODEL} → scorers (Claude API)...")
    result = mlflow.genai.evaluate(
        data=data, predict_fn=predict_fn,
        scorers=scorers if scorers is not None else build_scorers(config.JUDGE_MODEL),
    )
    return rows, result


def track_run(run_name, tags=None, limit=None, description="", **overrides):
    """Прогон + лог в MLflow под именем/тегами по конвенции (<axis>-<variant>-<ruler>)
    + per-question дамп. mlflow.genai.evaluate логирует метрики/трейсы в активный run."""
    collection = overrides.get("collection_name", config.COLLECTION_NAME)
    # открываем run заранее (имя/теги/параметры по конвенции); evaluate внутри логирует
    # метрики и трейсы в этот же активный run
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params({
            "llm_model": overrides.get("llm_model", config.LLM_MODEL),
            "judge_model": config.JUDGE_MODEL,
            "collection": collection,
            "retrieve_k": overrides.get("retrieve_k", config.RETRIEVE_K),
            "context_k": overrides.get("context_k", config.CONTEXT_K),
            "max_papers": overrides.get("max_papers", config.MAX_PAPERS),
            "golden": config.GOLDEN_PATH,
            "n": limit if limit else "all",
        })
        if description:
            mlflow.set_tag("mlflow.note.content", description)

        rows, result = evaluate_config(limit=limit, **overrides)
        mlflow.log_metrics({k: float(v) for k, v in result.metrics.items()})

        # per-question дамп (gitignored под data/) + артефакт.
        # result_df несёт объекты-трейсы/assessments → parquet их не сериализует;
        # берём только оценки (*/value) + запрос/ответ и стрингуем.
        df = result.result_df.copy()
        df.insert(0, "gold_doc_id", [r["doc_id"] for r in rows])
        keep = (["gold_doc_id"]
                + [c for c in df.columns if c.endswith("/value")]
                + [c for c in ("request", "response") if c in df.columns])
        dump = GOLDEN.parent / f"per_question_{run_name}.parquet"
        df[keep].astype(str).to_parquet(dump, index=False)
        mlflow.log_artifact(str(dump))

    print(f"\n[eval] {run_name} | {result.metrics}")
    print(f"[eval] per-question → {dump}")
    return result.metrics


if __name__ == "__main__":
    # Текущий эксперимент: v3-baseline (корпус 2021–2026, embed_text-коллекция, MLflow-eval).
    # Имя/теги по конвенции: <axis>-<variant>-<ruler> + теги axis/variant/ruler/compare_to.
    print(track_run(
        "gen-7b-v4",
        tags={"axis": "gen", "variant": "7b", "ruler": "v4", "compare_to": "gen-7b-v3"},
        collection_name="nlp2021_2026_embedtext",
        description=(
            "**Итог:** _(заполнить после)_\n\n---\n\n"
            "**Изменение:** переземление golden под корпус 2021–2026 — мульти-релевантность "
            "(relevant_doc_ids, 22/50 вопросов), атомарные expected_facts (5–6/вопрос), qtype.\n\n"
            "**База:** gen-7b-v3 (та же система/корпус) — изменился ТОЛЬКО golden ⇒ линейка v3→v4; "
            "судейские метрики впрямую несравнимы, doc_recall сравним.\n\n"
            "**Golden:** golden_dataset50_v2 (24 Q1 / 26 Q2)."
        ),
    ))
