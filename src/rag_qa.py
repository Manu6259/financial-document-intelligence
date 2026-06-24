"""Feature 2 — grounded Q&A over the 10-K with citations + a numeric guardrail.

This is RAG used for what RAG is actually good at: pulling the relevant passages
out of a long document the model can't memorise, and answering over them.

The finance-critical addition is the guardrail. In finance a confident wrong
number is the expensive failure, so:

  - the LLM must answer ONLY from retrieved passages and cite them as [id],
  - then a deterministic check verifies that every dollar/percent figure in the
    answer actually appears in one of the cited source chunks. Any number that
    doesn't trace to a source is flagged as UNSUPPORTED.

So the model reasons over text, but it cannot smuggle in an invented figure
without the guardrail catching it.

Embeddings are local (Ollama, nomic-embed-text), cached to disk.
"""

from __future__ import annotations

import os
import re

from model import EMBED_MOCK, USING_MOCK, chat_json
from parse_filing import Chunk
from retrieval import Retriever

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TOP_K = 6

_SYSTEM = """You answer questions about a company using ONLY the numbered source \
passages provided. Rules:
- Use only facts found in the sources. If the answer isn't there, say so.
- Cite the passages you used inline as [id], e.g. [42].
- Quote figures exactly as they appear in the sources. Never estimate or compute \
a number that isn't stated.
Return JSON: {"answer": "<text with [id] citations>", "cited_ids": [<ints>]}"""

# Captures the numeric core of money/percent/scale figures (group 1) and any
# unit (group 2). We hold these to the guardrail.
_NUM_RE = re.compile(
    r"\$?\s?(\d[\d,]*(?:\.\d+)?)\s?(million|billion|thousand|%|bps|basis points)?",
    re.IGNORECASE)


def _sig_digits(num: str) -> str:
    """Significant digits of a number string, magnitude-agnostic.

    "44.5" -> "445", "44,512" -> "44512", "871,905" -> "871905". Stripping commas,
    the decimal point, and leading/trailing zeros lets us compare a rounded
    answer ("$44.5 million") to a table figure ("44,512" in thousands): "445" is
    a prefix of "44512", so they match. This is what makes the guardrail tolerant
    of legitimate rounding/unit conversion without letting a wrong number pass.
    """
    return re.sub(r"\D", "", num).lstrip("0").rstrip("0")


def _figures(text: str) -> list[str]:
    """Financial figures in `text` as raw strings (for reporting). Skips bare
    calendar years (1900–2099 with no unit) — a year isn't a financial claim."""
    out: list[str] = []
    for m in _NUM_RE.finditer(text):
        num = m.group(1).replace(",", "").rstrip(".")
        unit = m.group(2)
        if not unit and re.fullmatch(r"(19|20)\d{2}", num):
            continue
        out.append(m.group(0).strip())
    return out


def _supported(fig: str, source_sigs: list[str]) -> bool:
    """A figure is supported if its significant digits match a source figure's
    (one a prefix of the other, ≥2 digits overlap → tolerates rounding)."""
    s = _sig_digits(fig)
    if len(s) < 2:
        return True  # single-digit counts ("3 stores") aren't worth guarding
    for src in source_sigs:
        if len(src) >= 2 and (s.startswith(src) or src.startswith(s)):
            return True
    return False


class FilingQA:
    """Grounded Q&A over the filing. Retrieval is delegated to the hybrid
    Retriever (lexical + vector); the eval (retrieval_eval.py) is what justifies
    using hybrid. BM25 works even with no embeddings, so this degrades gracefully
    when Ollama is down."""

    def __init__(self, mode: str = "hybrid") -> None:
        self.retriever = Retriever("WRBY")
        self.chunks = self.retriever.chunks
        self.mode = "bm25" if EMBED_MOCK else mode

    def retrieve(self, question: str, k: int = TOP_K) -> list[Chunk]:
        return self.retriever.search(question, k=k, mode=self.mode)

    def answer(self, question: str) -> dict:
        sources = self.retrieve(question)
        context = "\n\n".join(f"[{c.chunk_id}] ({c.section[:40]}) {c.text}" for c in sources)
        if USING_MOCK:
            top = sources[0]
            result = {"answer": f"(offline mock) Most relevant passage [{top.chunk_id}]: "
                                f"{top.text[:280]}…", "cited_ids": [top.chunk_id]}
        else:
            result = chat_json(_SYSTEM, f"Question: {question}\n\nSources:\n{context}")
        return self._guard(result, sources)

    def _guard(self, result: dict, sources: list[Chunk]) -> dict:
        """Verify every figure in the answer appears in a cited source chunk."""
        answer = result.get("answer", "")
        cited_ids = result.get("cited_ids", []) or [c.chunk_id for c in sources]
        by_id = {c.chunk_id: c.text for c in sources}
        cited_text = " ".join(by_id.get(i, "") for i in cited_ids)
        source_sigs = [_sig_digits(f) for f in _figures(cited_text)]

        # Strip [id] citation markers so they aren't mistaken for figures.
        answer_no_cites = re.sub(r"\[\d+\]", "", answer)
        unsupported = sorted({f for f in _figures(answer_no_cites)
                              if not _supported(f, source_sigs)})
        return {
            "answer": answer,
            "cited_ids": cited_ids,
            "sources": [{"id": c.chunk_id, "section": c.section[:50]} for c in sources],
            "unsupported_figures": unsupported,
            "grounded": len(unsupported) == 0,
        }


if __name__ == "__main__":
    import sys
    qa = FilingQA()
    q = sys.argv[1] if len(sys.argv) > 1 else "What was net revenue and how did it change year over year?"
    out = qa.answer(q)
    print("Q:", q)
    print("A:", out["answer"][:600])
    print("cited:", out["cited_ids"], "| grounded:", out["grounded"],
          "| unsupported:", out["unsupported_figures"])
