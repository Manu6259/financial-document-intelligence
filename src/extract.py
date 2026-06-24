"""Feature 1 — structured extraction of the income statement, with provenance.

We locate the Consolidated Statements of Operations in the filing, hand that
text to the LLM, and ask for a clean schema where EACH number carries the source
quote it came from. The number is meaningless without the quote — that pairing
is what makes the output auditable and lets evaluate.py check it against SEC
XBRL.

Offline mock: a deterministic regex parse of the labelled lines, so the pipeline
and eval run with no API key (less accurate than the LLM, which is the point of
the eval comparison).
"""

from __future__ import annotations

import json
import os
import re

from model import USING_MOCK, chat_json
from parse_filing import load_chunks

LINE_ITEMS = ["revenue", "gross_profit", "operating_income", "net_income"]

_SYSTEM = """You extract the income statement from an SEC 10-K. You are given the \
Consolidated Statements of Operations text. Return the MOST RECENT fiscal year's \
figures as JSON, each line item an object with the value in ACTUAL US DOLLARS \
(watch the 'in thousands' header — multiply accordingly) and the exact source \
text you read it from.

Return ONLY these keys: revenue, gross_profit, operating_income, net_income.
Schema: {"revenue": {"value": <number>, "quote": "<exact source text>"}, ...}
If a line item is a loss, the value is negative. Do not compute or estimate — \
read the reported number. If you cannot find one, set value to null."""


_TABLE_SIGNALS = ["cost of goods sold", "gross profit", "from operations",
                  "net revenue", "selling, general", "net income", "net loss"]


def _income_statement_text() -> str:
    """Find the actual Statements of Operations TABLE (not MD&A prose).

    We score each chunk by how many income-statement line labels it contains
    plus whether it has table-style 6-digit comma numbers (e.g. 871,905). The
    real table scores far higher than a prose sentence that merely says
    "net revenue of $871 million", so both the LLM and the mock get the right
    source context.
    """
    chunks = load_chunks("WRBY")

    def score(c) -> int:
        low = c.text.lower()
        s = sum(sig in low for sig in _TABLE_SIGNALS)
        s += 2 * len(re.findall(r"\b\d{3},\d{3}\b", c.text))  # table magnitudes
        return s

    ranked = sorted(chunks, key=score, reverse=True)
    best = [c for c in ranked[:3] if score(c) > 0]
    best.sort(key=lambda c: c.start)  # keep document order for table continuity
    return "\n\n".join(c.text for c in best)


# --- deterministic mock extraction -------------------------------------------
_MOCK_LABELS = {
    "revenue": r"net revenue[s]?",
    "gross_profit": r"gross profit",
    "operating_income": r"(?:income|loss) from operations",
    "net_income": r"net income|net loss",
}


def _mock_extract(text: str) -> dict:
    low = text.lower()
    thousands = "in thousands" in low
    mult = 1000 if thousands else 1
    out = {}
    for item, label in _MOCK_LABELS.items():
        m = re.search(r"(?:" + label + r")[^0-9\(\)-]{0,40}\(?\$?\s*([0-9][0-9,]+)\)?", low)
        if m:
            val = int(m.group(1).replace(",", "")) * mult
            if "(" in m.group(0):  # parenthesised = negative
                val = -val
            out[item] = {"value": val, "quote": m.group(0).strip()[:120]}
        else:
            out[item] = {"value": None, "quote": ""}
    return out


def extract_income_statement() -> dict:
    text = _income_statement_text()
    if USING_MOCK:
        data = _mock_extract(text)
        data["_method"] = "mock-regex"
        return data
    data = chat_json(_SYSTEM, text[:12000])
    # keep only our keys; tolerate the model returning extras
    cleaned = {k: data.get(k, {"value": None, "quote": ""}) for k in LINE_ITEMS}
    cleaned["_method"] = "llm"
    return cleaned


if __name__ == "__main__":
    res = extract_income_statement()
    print(f"method: {res.pop('_method')}")
    for k, v in res.items():
        val = v.get("value")
        print(f"  {k:18} {('$'+format(val,',')) if val is not None else 'null':>16}")
        if v.get("quote"):
            print(f"     ← \"{v['quote'][:90]}\"")
