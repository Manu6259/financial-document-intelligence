"""The router — one entry point that sends each question to the right engine.

  - Numeric line-item questions ("what was inventory in 2025?") -> structured
    XBRL lookup: exact value, grounded by construction, no LLM, nothing to
    hallucinate.
  - Narrative questions ("why did inventory decline?") -> hybrid RAG with the
    citation + magnitude-aware guardrail.

Routing rule: a "why/how/explain/driver" question is always narrative (it wants
reasoning, not a number); otherwise, if it names a known line item, it's a
numeric lookup. This split is the production pattern for financial filings —
numbers come from structured data, explanations come from the document.
"""

from __future__ import annotations

import re

from rag_qa import FilingQA
from structured_financials import lookup

# Words that signal the user wants an explanation, not a figure.
_NARRATIVE = re.compile(r"\b(why|how did|explain|driver|drove|reason|cause|"
                        r"changed|decline|increase|decrease|grow|fell|rose)\b", re.I)


def _format_structured(res: dict) -> str:
    parts = [f"{y}: ${v:,}" if v is not None else f"{y}: not reported"
             for y, v in sorted(res["values"].items())]
    return f"{res['line_item'].title()} — " + "; ".join(parts) + f"  (source: {res['source']})"


class Ask:
    def __init__(self) -> None:
        self.qa = FilingQA()

    def __call__(self, question: str) -> dict:
        narrative = bool(_NARRATIVE.search(question))
        if not narrative:
            res = lookup(question)
            if res:
                return {
                    "route": "structured",
                    "answer": _format_structured(res),
                    "values": res["values"],
                    "source": res["source"],
                    "grounded": True,           # exact value from SEC data
                    "unsupported_figures": [],
                }
        out = self.qa.answer(question)
        out["route"] = "rag"
        return out


if __name__ == "__main__":
    import sys
    ask = Ask()
    q = sys.argv[1] if len(sys.argv) > 1 else "What was inventory in 2024 and 2025?"
    out = ask(q)
    print(f"Q: {q}")
    print(f"[route: {out['route']}]  grounded: {out['grounded']}")
    print("A:", out["answer"][:500])
    if out.get("unsupported_figures"):
        print("⚠ unsupported:", out["unsupported_figures"])
