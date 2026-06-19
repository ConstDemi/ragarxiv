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


def evaluate_config(limit=None, **overrides):
    """Оценка одного конфига.

    overrides: llm_model / system_prompt / retrieve_k / context_k / max_papers.
    Модель судьи — ТОЛЬКО из config.JUDGE_MODEL (единый источник правды, без override).
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
          f"judge={config.JUDGE_MODEL} | "
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
          f"({config.JUDGE_MODEL}, max_workers={config.EVAL_MAX_WORKERS})...")
    llm = build_judge(config.JUDGE_MODEL, config.JUDGE_MAX_TOKENS)
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

    # по-вопросные оценки: RAGAS уже посчитал их — это чтение result, БЕЗ новых вызовов судьи.
    # колонки: user_input/response/retrieved_contexts/reference + 4 метрики. Дополняем doc_id и hit-флагами.
    per_q = result.to_pandas()
    per_q.insert(0, "q_index", range(len(per_q)))
    per_q["doc_id"] = [row["doc_id"] for row in golden]
    per_q["doc_hit_context_k"] = hit_ctx
    per_q["doc_hit_retrieved"] = hit_retr
    return scores, per_q


def track_run(run_name, tags=None, limit=None, description="", **overrides):
    """Прогон + лог в MLflow. Тонкий логгер: run_name и tags пишешь руками по конвенции
    (имя = <axis>-<variant>-<ruler>; теги axis/variant/ruler/compare_to; status дописываешь в UI).
    description — карточка в поле Description рана."""
    scores, per_q = evaluate_config(limit=limit, **overrides)

    # дамп по-вопросных оценок для диагностики (gitignored под data/) — и как артефакт MLflow
    dump_path = GOLDEN.parent / f"per_question_{run_name}.parquet"
    per_q.to_parquet(dump_path, index=False)

    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)   # axis/variant/ruler/compare_to по конвенции; фильтруются в UI
        mlflow.log_params({
            "llm_model":   overrides.get("llm_model", config.LLM_MODEL),
            "judge_model": config.JUDGE_MODEL,
            "retrieve_k":  overrides.get("retrieve_k", config.RETRIEVE_K),
            "context_k":   overrides.get("context_k", config.CONTEXT_K),
            "max_papers":  overrides.get("max_papers", config.MAX_PAPERS),
            "golden":      config.GOLDEN_PATH,
            "n":           limit if limit else "all",
            "judge_context": "context_k",   # судья видит контекст генерации, не весь retrieved (v2-методика)
        })
        mlflow.log_metrics(scores)
        mlflow.log_artifact(str(dump_path))
        # точный промпт этого рана как артефакт — ран самодостаточен, даже если config позже изменится
        mlflow.log_text(overrides.get("system_prompt", config.SYSTEM_PROMPT), "system_prompt.txt")
        if description:
            mlflow.set_tag("mlflow.note.content", description)   # поле Description на странице рана в UI

    # консольная диагностика: худшие по faithfulness сверху (fa/ar/cp/cr)
    worst = per_q.sort_values("faithfulness").head(8)
    print(f"\n[eval] по-вопросные оценки -> {dump_path}")
    print("[eval] худшие по faithfulness (fa/ar/cp/cr | вопрос):")
    for _, r in worst.iterrows():
        print(f"  #{int(r['q_index']):>2}  fa={r['faithfulness']:.2f} ar={r['answer_relevancy']:.2f} "
              f"cp={r['context_precision']:.2f} cr={r['context_recall']:.2f} | {r['user_input'][:60]}")
    print(f"\n[eval] залогировано в MLflow: run '{run_name}' | {scores}")
    return scores


if __name__ == "__main__":
    # Здесь пишешь ОДИН эксперимент под текущий прогон и запускаешь `python run.py`.
    # Имя и теги — по конвенции: имя <axis>-<variant>-<ruler>; теги axis/variant/ruler/compare_to.
    # Сейчас очереди нет (Фаза 0 закрыта) — вызов закомментирован, запуск ничего не логирует.
    # Шаблон (раскомментируй и правь под эксперимент):
    #
    # print(track_run(
    #     "prompt-xxx-v2",
    #     tags={"axis": "prompt", "variant": "xxx", "ruler": "v2", "compare_to": "gen-7b-v2"},
    #     description=(
    #         "**Итог:** _(заполнить после)_\n\n---\n\n"
    #         "**Изменение:** ...\n\n**База сравнения:** gen-7b-v2 ...\n\n"
    #         "**Гипотеза:** ...\n\n**Ожидания:** ..."
    #     ),
    # ))
    print("Нет активного эксперимента: впиши вызов track_run(...) в __main__ и запусти снова.")
