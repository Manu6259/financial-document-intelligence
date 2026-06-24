"""Evaluation harness — graded against the SEC's own numbers, not ourselves.

Two things finance cares about:

  1. Extraction accuracy. We compare each line item the LLM pulled from the 10-K
     prose/tables to the authoritative XBRL value the company filed with the SEC.
     This is a real answer key — no self-grading.

  2. Guardrail efficacy. We prove the numeric guardrail actually catches a
     hallucinated figure: feed it an answer containing a number absent from the
     cited sources and confirm it's flagged UNSUPPORTED; feed it a grounded
     answer and confirm it passes. This is deterministic, so it runs with or
     without an API key.

Run: python src/evaluate.py
"""

from __future__ import annotations

import json
import os

from extract import LINE_ITEMS, extract_income_statement
from groundtruth import load_ground_truth
from model import EMBED_MOCK, EMBED_MODEL, LLM_MODEL, USING_MOCK
from rag_qa import FilingQA

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _fiscal_year() -> int:
    meta = json.load(open(os.path.join(DATA_DIR, "wrby_meta.json")))
    return int(meta["report_date"][:4])


def eval_extraction() -> dict:
    fy = _fiscal_year()
    truth = load_ground_truth("WRBY", fy)
    pred = extract_income_statement()
    method = pred.pop("_method", "?")
    rows, correct = [], 0
    for item in LINE_ITEMS:
        got = (pred.get(item) or {}).get("value")
        exp = truth.get(item)
        ok = got is not None and exp is not None and got == exp
        # tolerance: exact match expected for reported figures; also allow <0.5%
        # to absorb rounding if the model reports in thousands vs dollars.
        if not ok and got is not None and exp:
            ok = abs(got - exp) / abs(exp) < 0.005
        correct += ok
        rows.append({"item": item, "predicted": got, "truth": exp, "correct": bool(ok)})
    return {"method": method, "fiscal_year": fy,
            "accuracy": round(correct / len(LINE_ITEMS), 3), "items": rows}


def eval_guardrail() -> dict:
    """Deterministic proof the guardrail flags hallucinated numbers."""
    qa = FilingQA()
    sources = qa.chunks[:3]
    # A figure that does not occur in these sources.
    fake = {"answer": "Net revenue was $999,999,999 last year.", "cited_ids": [c.chunk_id for c in sources]}
    grounded = {"answer": f"See discussion. {sources[0].text[:60]}", "cited_ids": [sources[0].chunk_id]}
    caught = not qa._guard(fake, sources)["grounded"]
    passes = qa._guard(grounded, sources)["grounded"]
    return {"hallucinated_number_flagged": bool(caught),
            "grounded_answer_passes": bool(passes),
            "guardrail_working": bool(caught and passes)}


def main() -> None:
    print(f"LLM: {'MOCK' if USING_MOCK else LLM_MODEL}  |  "
          f"Embeddings: {'hash-mock' if EMBED_MOCK else EMBED_MODEL}\n")

    ext = eval_extraction()
    print(f"=== Extraction vs SEC XBRL ({ext['method']}, FY{ext['fiscal_year']}) ===")
    for r in ext["items"]:
        mark = "✅" if r["correct"] else "❌"
        p = f"${r['predicted']:,}" if r["predicted"] is not None else "null"
        t = f"${r['truth']:,}" if r["truth"] is not None else "n/a"
        print(f"  {mark} {r['item']:18} predicted {p:>16}   truth {t:>16}")
    print(f"  Extraction accuracy: {ext['accuracy']:.0%}\n")

    g = eval_guardrail()
    print("=== Hallucination guardrail self-test ===")
    print(f"  Flags a hallucinated number: {'✅' if g['hallucinated_number_flagged'] else '❌'}")
    print(f"  Lets a grounded answer pass: {'✅' if g['grounded_answer_passes'] else '❌'}")

    report = {"llm": "mock" if USING_MOCK else LLM_MODEL,
              "embeddings": "hash-mock" if EMBED_MOCK else EMBED_MODEL,
              "extraction": ext, "guardrail": g}
    with open(os.path.join(DATA_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("\nFull report -> data/metrics.json")


if __name__ == "__main__":
    main()
