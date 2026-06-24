"""Retrieval evaluation — measure recall@k and MRR, not vibes.

This is the discipline that makes RAG production-grade: before tuning retrieval,
define what "relevant" means and measure it. Each labelled question carries a
`gold` marker — a distinctive string that appears ONLY in the passage that truly
answers it (a balance-sheet figure, or a verbatim MD&A phrase). A retrieved chunk
counts as relevant if it contains the gold marker.

We report, per retrieval mode (vector / bm25 / hybrid):
  - recall@k : fraction of questions whose answer passage is in the top-k
  - MRR      : mean reciprocal rank of the first relevant chunk

The set deliberately mixes TABLE lookups (where dense vectors are weak) and
PROSE questions (where they're strong), so the hybrid advantage is visible.

Run: python src/retrieval_eval.py
"""

from __future__ import annotations

import json
import os

from retrieval import Retriever

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
K = 6

# (question, gold-substring that uniquely marks the answer passage, kind)
GOLD: list[tuple[str, str, str]] = [
    ("What was inventory at year end 2025?", "44,512", "table"),
    ("What was cash and cash equivalents in 2025?", "286,358", "table"),
    ("What were total assets in 2025?", "720,919", "table"),
    ("What was accounts payable in 2025?", "31,979", "table"),
    ("What was net revenue in 2025?", "871,905", "table"),
    ("What was stock-based compensation expense?", "34,536", "table"),
    ("Why did gross margin change year over year?", "basis points", "prose"),
    ("Why did inventory decrease?", "more closely manage stock on hand", "prose"),
    ("What drove the change in operating cash flow?", "operating assets and liabilities", "prose"),
    ("How many stores offer eye exams?", "236 stores", "prose"),
]


def _first_relevant_rank(chunks, gold: str) -> int | None:
    for rank, c in enumerate(chunks, start=1):
        if gold.lower() in c.text.lower():
            return rank
    return None


def evaluate_mode(r: Retriever, mode: str) -> dict:
    hits, rr = 0, 0.0
    per_kind: dict[str, list[int]] = {"table": [], "prose": []}
    for q, gold, kind in GOLD:
        chunks = r.search(q, k=K, mode=mode)
        rank = _first_relevant_rank(chunks, gold)
        hit = rank is not None
        hits += hit
        rr += (1.0 / rank) if rank else 0.0
        per_kind[kind].append(int(hit))
    n = len(GOLD)
    return {
        "recall@k": round(hits / n, 3),
        "mrr": round(rr / n, 3),
        "recall_table": round(sum(per_kind["table"]) / len(per_kind["table"]), 3),
        "recall_prose": round(sum(per_kind["prose"]) / len(per_kind["prose"]), 3),
    }


def main() -> None:
    r = Retriever()
    print(f"Retrieval eval — {len(GOLD)} labelled questions, k={K}\n")
    print(f"{'mode':8} {'recall@k':>9} {'MRR':>6} {'recall(table)':>14} {'recall(prose)':>14}")
    report = {}
    for mode in ("vector", "bm25", "hybrid"):
        m = evaluate_mode(r, mode)
        report[mode] = m
        print(f"{mode:8} {m['recall@k']:>9.2f} {m['mrr']:>6.2f} "
              f"{m['recall_table']:>14.2f} {m['recall_prose']:>14.2f}")
    with open(os.path.join(DATA_DIR, "retrieval_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nReport -> data/retrieval_metrics.json")


if __name__ == "__main__":
    main()
