"""Structured financial line items from SEC XBRL — the numeric-answer path.

The production-correct way to answer "what was inventory in 2025?" is NOT fuzzy
text retrieval over a prose note — it's a lookup against the company's structured
filing data. SEC XBRL gives us exactly that: every statement line item as a
machine-readable value per fiscal year, straight from the regulator.

So numeric questions route here: an exact value, grounded by construction (the
source is the SEC tag), with no LLM arithmetic and nothing to hallucinate. The
narrative "why" questions stay with the RAG path.

Aliases map natural language to us-gaap tags. (Tags vary by filer; this covers
WRBY and would need a small mapping layer to generalise — noted as a limitation.)
"""

from __future__ import annotations

import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Ordered most-specific-first so "total assets" matches before "assets".
ALIASES: list[tuple[str, str]] = [
    ("total current assets", "AssetsCurrent"),
    ("total assets", "Assets"),
    ("total current liabilities", "LiabilitiesCurrent"),
    ("total liabilities", "Liabilities"),
    ("stockholders equity", "StockholdersEquity"),
    ("shareholders equity", "StockholdersEquity"),
    ("accounts payable", "AccountsPayableCurrent"),
    ("accounts receivable", "AccountsReceivableNetCurrent"),
    ("cash and cash equivalents", "CashAndCashEquivalentsAtCarryingValue"),
    ("inventory", "InventoryNet"),
    ("net revenue", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("revenue", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("net sales", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ("gross profit", "GrossProfit"),
    ("operating income", "OperatingIncomeLoss"),
    ("operating loss", "OperatingIncomeLoss"),
    ("income from operations", "OperatingIncomeLoss"),
    ("loss from operations", "OperatingIncomeLoss"),
    ("net income", "NetIncomeLoss"),
    ("net loss", "NetIncomeLoss"),
    ("stock-based compensation", "ShareBasedCompensation"),
    ("stock based compensation", "ShareBasedCompensation"),
    ("depreciation", "DepreciationDepletionAndAmortization"),
]


def _facts(ticker: str = "WRBY") -> dict:
    return json.load(open(os.path.join(DATA_DIR, f"{ticker.lower()}_xbrl_facts.json")))["facts"]["us-gaap"]


def detect_line_item(question: str) -> tuple[str, str] | None:
    """Return (alias_phrase, xbrl_tag) if the question names a known line item."""
    q = question.lower()
    for phrase, tag in ALIASES:
        if phrase in q:
            return phrase, tag
    return None


def _value_for_year(usd: list[dict], year: int) -> int | None:
    """Pick the value for fiscal year `year`, preferring the 10-K. Handles both
    duration items (full-year flows) and instant items (year-end balances)."""
    yr = str(year)
    duration = [u for u in usd if u.get("start", "").startswith(yr)
                and u.get("end", "").startswith(yr) and u.get("form") == "10-K"]
    if duration:
        return duration[-1]["val"]
    instant = [u for u in usd if u.get("end", "").endswith("-12-31")
               and u.get("end", "").startswith(yr) and u.get("form") == "10-K"]
    return instant[-1]["val"] if instant else None


def _years_in(question: str, default_year: int) -> list[int]:
    yrs = sorted({int(y) for y in re.findall(r"\b(20\d{2})\b", question)})
    return yrs or [default_year]


def lookup(question: str, ticker: str = "WRBY", default_year: int = 2025) -> dict | None:
    """Structured numeric answer for a line-item question, or None if not one."""
    hit = detect_line_item(question)
    if not hit:
        return None
    phrase, tag = hit
    facts = _facts(ticker)
    if tag not in facts:
        return None
    usd = facts[tag]["units"].get("USD", [])
    years = _years_in(question, default_year)
    values = {y: _value_for_year(usd, y) for y in years}
    if all(v is None for v in values.values()):
        return None
    return {"line_item": phrase, "xbrl_tag": tag, "values": values, "source": f"SEC XBRL · us-gaap:{tag}"}


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "What was inventory in 2024 and 2025?"
    print(json.dumps(lookup(q), indent=2))
