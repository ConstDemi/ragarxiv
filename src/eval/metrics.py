# src/eval/metrics.py
# MLflow GenAI scorers (судья Claude через anthropic:/) + детерминированный custom doc_recall.
# Без ragas/langchain — судьи встроенные в MLflow.
from mlflow.entities import Feedback, SpanType
from mlflow.genai.scorers import (
    scorer,
    RetrievalGroundedness,   # ≈ faithfulness: ответ обоснован retrieved-контекстом
    RetrievalRelevance,      # ≈ context_precision: релевантность retrieved-доков запросу
    RetrievalSufficiency,    # ≈ context_recall: контекст покрывает expected_facts
    RelevanceToQuery,        # ≈ answer_relevancy: ответ адресует вопрос
)


def judge_uri(model: str) -> str:
    """config.JUDGE_MODEL (id) → провайдер-URI для MLflow-судьи."""
    return model if "/" in model else f"anthropic:/{model}"


@scorer
def doc_recall(expectations, trace):
    """Детерминированный retrieval-recall (без судьи): попадание = ЛЮБОЙ из relevant_doc_ids
    (мульти-релевантность) среди Documents RETRIEVER-спана (top-context_k, что видела LLM).
    Фолбэк на одиночный doc_id для старой схемы golden.

    MLflow подставляет аргументы scorer'а ПО ИМЕНИ из доступного набора
    (inputs / outputs / expectations / trace) — берём только нужные: expectations и trace.
    Возврат Feedback(value=0/1); evaluate усредняет по вопросам → doc_recall/mean."""
    exp = expectations or {}
    golds = set(exp.get("relevant_doc_ids") or ([exp["doc_id"]] if exp.get("doc_id") else []))
    ids = []
    for sp in (trace.search_spans(span_type=SpanType.RETRIEVER) if trace else []):
        for d in (sp.outputs or []):
            did = d.get("id") if isinstance(d, dict) else getattr(d, "id", None)
            if did:
                ids.append(did)
    hit = bool(golds & set(ids))
    return Feedback(value=float(hit),
                    rationale=f"gold∈{sorted(golds)} {'∈' if hit else '∉'} {len(set(ids))} retrieved")


def build_scorers(judge_model: str):
    """Список scorers для mlflow.genai.evaluate. judge_model — id из config.JUDGE_MODEL."""
    j = judge_uri(judge_model)
    return [
        RetrievalGroundedness(model=j),
        RetrievalRelevance(model=j),
        RetrievalSufficiency(model=j),
        RelevanceToQuery(model=j),
        doc_recall,
    ]
