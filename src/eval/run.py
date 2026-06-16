# src/eval/run.py
# Оценочный харнес: дёргает ScienceRAG НАПРЯМУЮ (не через HTTP API),
# гоняет golden-набор и считает 4 метрики RAGAS (судья Claude, эмбеддер Qwen3).
import sys
import logging
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # для metrics
sys.path.insert(0, str(HERE.parent / "app"))           # для config, rag_pipeline

from dotenv import load_dotenv
load_dotenv(HERE.parent.parent / ".env")               # ANTHROPIC_API_KEY из .env в корне репо

import _compat  # noqa: F401 — заглушка vertexai ДО импорта ragas (ragas 0.4.3 ⟂ langchain 1.x)
import pyarrow.parquet as pq
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
import mlflow

import config
from rag_pipeline import ScienceRAG
from metrics import build_judge, build_embeddings, build_metrics

GOLDEN = HERE.parent.parent / config.GOLDEN_PATH
mlflow.set_tracking_uri((HERE.parent.parent / "mlruns").as_uri())   # стор в корне репо

# глушим шум сторонних логгеров (httpx-запросы, anthropic 429-ретраи), чтобы свои print были видны
for _n in ("httpx", "anthropic", "sentence_transformers", "huggingface_hub"):
    logging.getLogger(_n).setLevel(logging.WARNING)


def _to_scores(result) -> dict:
    """RAGAS EvaluationResult -> dict[str, float] (под log_metrics).
    _repr_dict — официальный агрегат RAGAS (safe_nanmean), совпадает с repr.
    Фолбэк: вручную усредняем per-sample списки (result[metric])."""
    repr_dict = getattr(result, "_repr_dict", None)
    if repr_dict is not None:
        return {k: float(v) for k, v in repr_dict.items()}
    keys = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
    out = {}
    for k in keys:
        vals = [x for x in result[k] if x is not None and x == x]  # x==x отсеивает NaN
        out[k] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def evaluate_config(limit=None, judge_model=None, **overrides):
    """Оценка одного конфига.

    overrides: llm_model / system_prompt / retrieve_k / context_k / max_papers.
    judge_model: переопределить модель судьи (напр. 'claude-haiku-4-5' для смоука).
    Предсказания пересчитываются под текущий конфиг; эталон фиксирован.
    """
    rag = ScienceRAG(
        qdrant_host=config.QDRANT_HOST,
        qdrant_port=config.QDRANT_PORT,
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        llm_model=overrides.get("llm_model", config.LLM_MODEL),
        system_prompt=overrides.get("system_prompt", config.SYSTEM_PROMPT),
        max_new_tokens=config.MAX_NEW_TOKENS,
        max_input_tokens=config.MAX_INPUT_TOKENS,
        retrieve_k=overrides.get("retrieve_k", config.RETRIEVE_K),
        context_k=overrides.get("context_k", config.CONTEXT_K),
        max_papers=overrides.get("max_papers", config.MAX_PAPERS),
    )

    golden = pq.read_table(GOLDEN).to_pylist()
    if limit:
        golden = golden[:limit]

    print(f"\n[eval] конфиг: llm={overrides.get('llm_model', config.LLM_MODEL)} | "
          f"judge={judge_model or config.JUDGE_MODEL} | "
          f"retrieve_k={overrides.get('retrieve_k', config.RETRIEVE_K)} "
          f"context_k={overrides.get('context_k', config.CONTEXT_K)} "
          f"max_papers={overrides.get('max_papers', config.MAX_PAPERS)} | "
          f"вопросов: {len(golden)}")

    samples = []
    hit_ctx, hit_retr = [], []   # точное doc_id-попадание (бесплатный retrieval-recall, без Claude)
    for i, row in enumerate(golden, 1):
        print(f"[eval] генерация {i}/{len(golden)}: {row['question'][:70]}")
        res = rag.answer(query=row["question"])
        samples.append({
            # имена колонок — RAGAS v1.0; на старой версии: question/answer/contexts/ground_truth
            "user_input": row["question"],
            "response": res["answer"],
            # судье отдаём РОВНО то, что видела LLM (context_k чанков), а не все ~retrieve_k:
            # дешевле (меньше токенов и per-chunk вызовов) и корректнее для faithfulness
            "retrieved_contexts": [c["text"] for c in res["context_chunks"]],
            "reference": row["ground_truth"],
        })
        gold = row["doc_id"]
        hit_ctx.append(gold in {c["doc_id"] for c in res["context_chunks"]})   # @context_k: что видела LLM
        hit_retr.append(gold in {s["doc_id"] for s in res["sources"]})         # @retrieved: статьи в выдаче
    rag.cleanup()

    print(f"[eval] генерация готова ({len(samples)} ответов) — запускаю RAGAS-судью "
          f"({judge_model or config.JUDGE_MODEL}, max_workers={config.EVAL_MAX_WORKERS})...")
    llm = build_judge(judge_model or config.JUDGE_MODEL, config.JUDGE_MAX_TOKENS)
    emb = build_embeddings(config.EMBED_MODEL)
    result = evaluate(
        dataset=Dataset.from_list(samples),
        metrics=build_metrics(llm, emb),
        run_config=RunConfig(max_workers=config.EVAL_MAX_WORKERS),  # душим параллелизм → меньше 429
    )
    scores = _to_scores(result)
    # детерминированный retrieval-recall по doc_id (без вызовов Claude);
    # разрыв @retrieved − @context_k = потери на отсечке context_k (нужен reranker / больший context_k)
    scores["doc_recall_at_context_k"] = sum(hit_ctx) / len(hit_ctx)
    scores["doc_recall_at_retrieved"] = sum(hit_retr) / len(hit_retr)
    return scores


def track_run(run_name, limit=None, judge_model=None, prompt_tag="default", **overrides):
    """Прогон + лог в MLflow: params (конфиг) + metrics (4 скора). run_name — метка варианта."""
    scores = evaluate_config(limit=limit, judge_model=judge_model, **overrides)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "llm_model":   overrides.get("llm_model", config.LLM_MODEL),
            "judge_model": judge_model or config.JUDGE_MODEL,
            "retrieve_k":  overrides.get("retrieve_k", config.RETRIEVE_K),
            "context_k":   overrides.get("context_k", config.CONTEXT_K),
            "max_papers":  overrides.get("max_papers", config.MAX_PAPERS),
            "prompt_tag":  prompt_tag,
            "golden":      config.GOLDEN_PATH,
            "n":           limit if limit else "all",
            "judge_context": "context_k",   # судья видит контекст генерации, не весь retrieved (v2-методика)
        })
        mlflow.log_metrics(scores)
    print(f"[eval] залогировано в MLflow: run '{run_name}' | {scores}")
    return scores


if __name__ == "__main__":
    # эксперимент: генератор 7B (4-bit включается автоматически на CUDA), судья Haiku.
    # config.LLM_MODEL (3B) не трогаем — 7B передаём оверрайдом; линейка та же (v2); run "7B-v2"
    print(track_run("7B-v2", judge_model="claude-haiku-4-5",
                    llm_model="Qwen/Qwen2.5-7B-Instruct"))
