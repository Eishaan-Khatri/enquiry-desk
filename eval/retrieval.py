"""Retrieval quality on the labelled questions: dense, sparse and hybrid, side by side.

    uv run python eval/retrieval.py

A question counts as found at rank r when the r-th result comes from one of its sections.
  top-1   the right section is the first result
  top-4   it is among the four results the agent sees
  MRR     the mean of 1 / rank, with 0 when it is not in the first ten
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

from enquiry_desk.config import get_settings
from enquiry_desk.runtime import make_knowledge_base
from enquiry_desk.tenant import load_tenant

HERE = Path(__file__).parent
MODES = ("dense", "sparse", "hybrid")


def main() -> None:
    settings = get_settings()
    tenant = load_tenant(settings.tenant_dir)
    kb = make_knowledge_base(settings)
    questions = yaml.safe_load((HERE / "retrieval_questions.yaml").read_text(encoding="utf-8"))

    ranks: dict[str, dict[str, list[int | None]]] = {mode: defaultdict(list) for mode in MODES}
    misses = []
    for item in questions:
        for mode in MODES:
            hits = kb.search(tenant.id, item["q"], k=10, mode=mode)
            rank = next((i for i, hit in enumerate(hits, 1) if hit.source in item["sources"]), None)
            ranks[mode][item["kind"]].append(rank)
            if mode == "hybrid" and (rank is None or rank > 4):
                misses.append({"q": item["q"], "kind": item["kind"], "rank": rank, "got": hits[0].source})

    results = {}
    print(f"{'mode':8} {'kind':9} {'n':>3}  {'top-1':>9}  {'top-4':>9}  {'MRR':>5}")
    for mode in MODES:
        by_kind = dict(ranks[mode])
        by_kind["all"] = [rank for kind in ranks[mode].values() for rank in kind]
        results[mode] = {}
        for kind, found in by_kind.items():
            n = len(found)
            top1 = sum(rank == 1 for rank in found)
            top4 = sum(rank is not None and rank <= 4 for rank in found)
            mrr = sum(1 / rank for rank in found if rank) / n
            results[mode][kind] = {"n": n, "top1": top1, "top4": top4, "mrr": round(mrr, 3)}
            print(f"{mode:8} {kind:9} {n:>3}  {top1:>3} of {n:<2}  {top4:>3} of {n:<2}  {mrr:.3f}")
        print()

    print("Not in the top four with hybrid search:")
    for miss in misses:
        print(f"  [{miss['kind']}] {miss['q']}\n      rank {miss['rank']}, first result was {miss['got']}")

    out = HERE / "results" / "retrieval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "dense_model": settings.dense_model,
                "sparse_model": "Qdrant/bm25",
                "qdrant": settings.qdrant_url or "embedded",
                "chunks": kb.count(tenant.id),
                "questions": len(questions),
                "note": "Documents and questions were written by the same author for a fictional institute.",
                "results": results,
                "hybrid_misses": misses,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
