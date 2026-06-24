"""Load the SEC's own XBRL numbers as ground truth for the eval.

The LLM extracts an income statement from the 10-K *prose/tables*. To know
whether it's right, we compare against the structured values the company filed
with the SEC in XBRL — the authoritative figures. No human labelling, no
guessing: the regulator's data is the answer key.
"""

from __future__ import annotations

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# The income-statement line items we extract, mapped to the us-gaap XBRL tags
# companies file them under. (Tags vary slightly by company; these cover WRBY.)
CONCEPT_TAGS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
}


def load_ground_truth(ticker: str, fiscal_year: int) -> dict[str, int]:
    """Return {line_item: value_in_dollars} for the given fiscal year's 10-K."""
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_xbrl_facts.json")
    facts = json.load(open(path))["facts"]["us-gaap"]
    out: dict[str, int] = {}
    for item, tags in CONCEPT_TAGS.items():
        for tag in tags:
            if tag not in facts:
                continue
            usd = facts[tag]["units"].get("USD", [])
            # Pick the full-year (FY) value from the 10-K for that fiscal year.
            hits = [u for u in usd if u.get("fy") == fiscal_year
                    and u.get("fp") == "FY" and u.get("form") == "10-K"]
            if hits:
                out[item] = hits[-1]["val"]
                break
    return out


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "WRBY"
    fy = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
    gt = load_ground_truth(ticker, fy)
    print(f"Ground truth for {ticker} FY{fy}:")
    for k, v in gt.items():
        print(f"  {k:18} ${v:,}")
