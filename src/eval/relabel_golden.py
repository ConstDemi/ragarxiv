#!/usr/bin/env python3
"""Переземление golden под корпус 2021–2026 (линейка v4).

Зачем: исходный golden заземлён на корпусе-2025 — один gold-документ на вопрос и
ground_truth одним абзацем. На корпусе 2021–2026 это занижает doc_recall (валидных
статей бывает несколько) и sufficiency (судья хочет атомарные факты). Здесь —
инструменты добавить мульти-релевантность, атомарные факты и qtype, не трогая старый файл.

Подкоманды (по порядку):
  inventory  топ-N ретрива по каждому вопросу → data/eval/_relevance_review.json
             (бесплатно: эмбеддер + Qdrant; зеркалит ScienceRAG._retrieve)
  label      Haiku по _relevance_review.json → relevant_doc_ids + expected_facts + qtype
             → data/eval/_relabel_wip.json   (~<$1; нужен ANTHROPIC_API_KEY; через litellm)
  review     печать компактной таблицы _relabel_wip.json для человеческой сверки
  build      собрать data/eval/golden_dataset50_v2.parquet из _relabel_wip.json

    python src/eval/relabel_golden.py inventory
    python src/eval/relabel_golden.py label   [--limit N]
    python src/eval/relabel_golden.py review
    python src/eval/relabel_golden.py build
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))  # src/ — для config
import config

REVIEW_PATH = ROOT / "data/eval/_relevance_review.json"
WIP_PATH = ROOT / "data/eval/_relabel_wip.json"
GOLDEN_OLD = ROOT / config.GOLDEN_PATH
GOLDEN_NEW = ROOT / "data/eval/golden_dataset50_v2.parquet"
HARD_WIP = ROOT / "data/eval/_harden_wip.json"
GOLDEN_HARD = ROOT / "data/eval/golden_dataset50_hard.parquet"

N_CANDIDATES = 15   # сколько различных статей на вопрос показываем разметчику
SNIPPET = 280       # длина текстового фрагмента кандидата для LLM


def _load_golden(path):
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def _parse_json(txt: str) -> dict:
    """Достаёт первый JSON-объект из ответа LLM (снимает ```-обёртки)."""
    t = txt.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    a, b = t.find("{"), t.rfind("}")
    if a != -1 and b != -1:
        t = t[a:b + 1]
    return json.loads(t)


# ───────────────────────── inventory ─────────────────────────
def cmd_inventory(args):
    import torch
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    golden = _load_golden(GOLDEN_OLD)
    if args.limit:
        golden = golden[:args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    encoder = SentenceTransformer(config.EMBED_MODEL, trust_remote_code=True, device=device)
    print(f"[inventory] collection={config.COLLECTION_NAME} retrieve_k={config.RETRIEVE_K} "
          f"N={N_CANDIDATES} | вопросов: {len(golden)}")

    out = []
    for i, row in enumerate(golden):
        q, gold = row["question"], row["doc_id"]
        vec = encoder.encode(q, prompt_name="query", convert_to_numpy=True, show_progress_bar=False)
        res = client.query_points(collection_name=config.COLLECTION_NAME, query=vec.tolist(),
                                  limit=config.RETRIEVE_K, with_payload=True, with_vectors=False)
        seen, cands, gold_rank = set(), [], None
        for rank, p in enumerate(res.points, 1):           # rank по чанкам (как видит ретрив)
            pl = p.payload or {}
            did = pl.get("doc_id")
            if gold_rank is None and did == gold:
                gold_rank = rank
            if did not in seen:                            # одна строка на статью (лучший чанк)
                seen.add(did)
                cands.append({"rank": rank, "doc_id": did,
                              "title": pl.get("title", ""), "published": pl.get("published", ""),
                              "score": round(float(p.score), 4),
                              "snippet": (pl.get("text") or "").replace("\n", " ")[:SNIPPET]})
        out.append({
            "idx": i, "question": q, "ground_truth": row["ground_truth"],
            "gold_doc_id": gold, "gold_title": row.get("title", ""),
            "gold_published": row.get("published", ""), "gold_rank": gold_rank,
            "candidates": cands[:N_CANDIDATES],
        })
        print(f"  #{i:>2} gold_rank={gold_rank} distinct_docs={len(cands)}")

    REVIEW_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[inventory] → {REVIEW_PATH}")


# ───────────────────────── label (Haiku) ─────────────────────────
SYS_LABEL = (
    "Ты — ассистент-разметчик золотого набора для оценки научного RAG по статьям arXiv (NLP). "
    "Отвечай СТРОГО одним JSON-объектом без пояснений и без markdown."
)


def _label_prompt(r: dict) -> str:
    cand_lines = "\n".join(
        f'{c["rank"]}. id={c["doc_id"]} | {c["title"]} | {c["snippet"]}'
        for c in r["candidates"]
    )
    return f"""Вопрос: {r['question']}

Эталонный ответ (ground_truth):
{r['ground_truth']}

Исходный «золотой» документ: {r['gold_doc_id']} — {r['gold_title']}

Кандидаты из ретрива (rank, id, заголовок, фрагмент):
{cand_lines}

Задачи:
1) qtype: "Q1", если вопрос о СУЩЕСТВОВАНИИ работы/датасета/бенчмарка («есть ли», «существует ли», «известен ли»); "Q2", если о МЕТОДЕ/решении проблемы («какие работы», «как», «почему», «насколько»).
2) expected_facts: 3–6 атомарных фактов. Каждый — отдельное проверяемое утверждение, строго из ground_truth; НЕ добавляй ничего нового и не дроби термины ради числа.
3) relevant_doc_ids: id из кандидатов, которые ПРЯМО отвечают на вопрос. ОБЯЗАТЕЛЬНО включи исходный золотой {r['gold_doc_id']} первым элементом.
   - Q1 (существование): добавляй ЕЩЁ документ ТОЛЬКО если это тот же тип артефакта (датасет / бенчмарк / работа) о ТОМ ЖЕ конкретном предмете. Тематической близости или общего слова в названии («Bench», «benchmark») НЕДОСТАТОЧНО. При любом сомнении не добавляй — оставь только золотой.
   - Q2 (метод): добавляй работы, предлагающие метод/решение для ТОЙ ЖЕ проблемы.

Ответ строго в формате JSON:
{{"qtype": "Q1", "expected_facts": ["...", "..."], "relevant_doc_ids": ["{r['gold_doc_id']}"]}}"""


def _label_one(r: dict, model: str) -> dict:
    """Один вызов судьи по записи r → {qtype, expected_facts, relevant_doc_ids}.
    relevant_doc_ids нормализован: gold первым, уникальные, только существующие среди кандидатов."""
    import litellm
    resp = litellm.completion(
        model=model, max_tokens=config.JUDGE_MAX_TOKENS, temperature=0,
        messages=[{"role": "system", "content": SYS_LABEL},
                  {"role": "user", "content": _label_prompt(r)}],
    )
    txt = resp.choices[0].message.content
    try:
        data = _parse_json(txt)
    except Exception as e:
        print(f"  #{r['idx']:>2} ПАРСИНГ FAILED ({e}); сырой ответ:\n{txt[:300]}")
        data = {}
    gold = r["gold_doc_id"]
    valid = {c["doc_id"] for c in r["candidates"]} | {gold}
    rel = [gold] + [d for d in (data.get("relevant_doc_ids") or []) if d != gold]
    rel = [d for d in dict.fromkeys(rel) if d in valid]        # уникальные, существующие, gold первым
    facts = [str(f).strip() for f in (data.get("expected_facts") or []) if str(f).strip()]
    return {"qtype": data.get("qtype", ""), "expected_facts": facts, "relevant_doc_ids": rel}


def cmd_label(args):
    from dotenv import load_dotenv
    load_dotenv()
    model = "anthropic/" + config.JUDGE_MODEL    # тот же судья, что в eval, но через litellm напрямую

    if args.qtype:    # ── ре-разметка подмножества поверх WIP: хирургически меняем только relevant_doc_ids ──
        wip = json.loads(WIP_PATH.read_text())
        targets = [w for w in wip if w.get("qtype") == args.qtype]
        print(f"[label] RE-LABEL qtype={args.qtype} (строгое правило), вопросов: {len(targets)}; "
              f"facts/qtype сохраняю")
        for w in targets:
            res = _label_one(w, model)
            gold = w["gold_doc_id"]
            w["relevant_doc_ids"] = res["relevant_doc_ids"]
            w["added_doc_ids"] = [d for d in res["relevant_doc_ids"] if d != gold]
            print(f"  #{w['idx']:>2} +docs={len(w['added_doc_ids'])}")
        WIP_PATH.write_text(json.dumps(wip, ensure_ascii=False, indent=2))
        print(f"[label] обновлено → {WIP_PATH}")
        return

    # ── свежая разметка с нуля из _relevance_review.json ──
    review = json.loads(REVIEW_PATH.read_text())
    if args.limit:
        review = review[:args.limit]
    print(f"[label] model={model} | вопросов: {len(review)}")
    out = []
    for r in review:
        res = _label_one(r, model)
        gold = r["gold_doc_id"]
        out.append({**r, "qtype": res["qtype"], "expected_facts": res["expected_facts"],
                    "relevant_doc_ids": res["relevant_doc_ids"],
                    "added_doc_ids": [d for d in res["relevant_doc_ids"] if d != gold]})
        print(f"  #{r['idx']:>2} {res['qtype']:<2} facts={len(res['expected_facts'])} "
              f"+docs={len(res['relevant_doc_ids'])-1}")
    WIP_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[label] → {WIP_PATH}")


# ───────────────────────── review ─────────────────────────
def cmd_review(args):
    wip = json.loads(WIP_PATH.read_text())
    multi = 0
    for w in wip:
        tmap = {c["doc_id"]: c["title"] for c in w["candidates"]}
        added = w.get("added_doc_ids", [])
        multi += bool(added)
        print(f"#{w['idx']:>2} [{w.get('qtype','?')}] facts={len(w.get('expected_facts',[]))} "
              f"gold_rank={w.get('gold_rank')} | {w['question'][:68]}")
        for d in added:
            print(f"      + {d}  {tmap.get(d,'')[:72]}")
    qc = {}
    for w in wip:
        qc[w.get("qtype", "?")] = qc.get(w.get("qtype", "?"), 0) + 1
    facts_total = sum(len(w.get("expected_facts", [])) for w in wip)
    print(f"\n[review] вопросов с доп. релевантностью: {multi}/{len(wip)} | qtype={qc} | "
          f"всего фактов={facts_total} (avg {facts_total/max(len(wip),1):.1f}/вопрос)")
    print(f"[review] правьте вручную {WIP_PATH} при необходимости, затем `build`.")


# ───────────────────────── build ─────────────────────────
def cmd_build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    wip = json.loads(WIP_PATH.read_text())
    rows = []
    for w in sorted(wip, key=lambda x: x["idx"]):
        rel = w["relevant_doc_ids"]
        assert rel and rel[0] == w["gold_doc_id"], f"#{w['idx']}: gold не первый в relevant_doc_ids"
        assert w["expected_facts"], f"#{w['idx']}: пустые expected_facts"
        assert w["qtype"] in ("Q1", "Q2"), f"#{w['idx']}: некорректный qtype={w['qtype']!r}"
        rows.append({
            "question": w["question"], "ground_truth": w["ground_truth"],
            "doc_id": w["gold_doc_id"], "title": w["gold_title"], "published": w["gold_published"],
            "relevant_doc_ids": [str(d) for d in rel],
            "expected_facts": [str(f) for f in w["expected_facts"]],
            "qtype": w["qtype"],
        })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, GOLDEN_NEW)
    print(f"[build] {len(rows)} строк → {GOLDEN_NEW}")
    print(f"[build] колонки: {table.schema.names}")
    multi = sum(1 for r in rows if len(r["relevant_doc_ids"]) > 1)
    print(f"[build] мульти-релевантных вопросов: {multi}/{len(rows)}")


# ───────────────────────── harden (трудные вопросы) ─────────────────────────
SYS_HARDEN = (
    "Ты усложняешь вопросы для стресс-теста поиска по статьям arXiv (NLP). "
    "Отвечай СТРОГО одним JSON-объектом без пояснений и markdown."
)


def _harden_prompt(r: dict) -> str:
    return f"""Исходный вопрос: {r['question']}
Целевая статья (на неё вопрос и должен отвечать): {r.get('title','')}

Перепиши вопрос так, чтобы он:
1) звучал как естественный, косвенный вопрос исследователя (разговорно, не «по учебнику»);
2) НЕ называл конкретные методы/датасеты/бенчмарки/модели/аббревиатуры из статьи — никаких имён-подсказок;
3) оставался ОДНОЗНАЧНО ОТВЕЧАЕМЫМ той же статьёй: не про другое и не слишком общий;
4) был на русском, одним предложением.

Ответ строго JSON: {{"question_hard": "..."}}"""


def _harden_one(r: dict, model: str) -> str:
    import litellm
    resp = litellm.completion(
        model=model, max_tokens=512, temperature=0.3,
        messages=[{"role": "system", "content": SYS_HARDEN},
                  {"role": "user", "content": _harden_prompt(r)}],
    )
    try:
        return str(_parse_json(resp.choices[0].message.content).get("question_hard", "")).strip()
    except Exception as e:
        print(f"  #{r.get('idx','?')} ПАРСИНГ FAILED ({e})")
        return ""


def cmd_harden(args):
    from dotenv import load_dotenv
    load_dotenv()
    model = "anthropic/" + config.JUDGE_MODEL
    rows = _load_golden(GOLDEN_OLD)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[harden] model={model} | вопросов: {len(rows)}")
    out = []
    for i, r in enumerate(rows):
        hard = _harden_one({**r, "idx": i}, model)
        out.append({"idx": i, "qtype": r.get("qtype", ""), "gold_doc_id": r["doc_id"],
                    "gold_title": r.get("title", ""), "question_orig": r["question"],
                    "question_hard": hard})
        print(f"  #{i:>2} [{r.get('qtype','?')}] {hard[:72]}")
    HARD_WIP.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[harden] → {HARD_WIP}  (сверь orig→hard командой harden_review)")


def cmd_harden_review(args):
    wip = json.loads(HARD_WIP.read_text())
    for w in wip:
        print(f"#{w['idx']:>2} [{w['qtype']}]")
        print(f"   orig: {w['question_orig']}")
        print(f"   hard: {w['question_hard']}")
    bad = [w["idx"] for w in wip if len(w["question_hard"]) < 10]
    print(f"\n[harden_review] {len(wip)} вопросов" + (f" | ⚠️ подозрительные: {bad}" if bad else " | пустых нет"))
    print(f"[harden_review] правь вручную {HARD_WIP} при необходимости, затем harden_build.")


def cmd_harden_build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    wip = {w["idx"]: w for w in json.loads(HARD_WIP.read_text())}
    rows = _load_golden(GOLDEN_OLD)
    out = []
    for i, r in enumerate(rows):
        w = wip.get(i)
        assert w and w["question_hard"], f"#{i}: нет hard-вопроса"
        out.append({**r, "question_orig": r["question"], "question": w["question_hard"]})
    pq.write_table(pa.Table.from_pylist(out), GOLDEN_HARD)
    print(f"[harden_build] {len(out)} строк → {GOLDEN_HARD}")
    print(f"[harden_build] колонки: {pa.Table.from_pylist(out).schema.names}")


def main():
    ap = argparse.ArgumentParser(description="Переземление golden (линейка v4).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_inv = sub.add_parser("inventory"); p_inv.add_argument("--limit", type=int, default=None)
    p_lab = sub.add_parser("label")
    p_lab.add_argument("--limit", type=int, default=None)
    p_lab.add_argument("--qtype", choices=["Q1", "Q2"], default=None,
                       help="ре-разметить только этот qtype поверх WIP (меняет только relevant_doc_ids)")
    sub.add_parser("review")
    sub.add_parser("build")
    p_hard = sub.add_parser("harden")
    p_hard.add_argument("--limit", type=int, default=None)
    sub.add_parser("harden_review")
    sub.add_parser("harden_build")
    args = ap.parse_args()
    {"inventory": cmd_inventory, "label": cmd_label, "review": cmd_review, "build": cmd_build,
     "harden": cmd_harden, "harden_review": cmd_harden_review, "harden_build": cmd_harden_build}[args.cmd](args)


if __name__ == "__main__":
    main()
