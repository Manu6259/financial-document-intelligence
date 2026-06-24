"""Fetch a real 10-K and its structured XBRL facts from SEC EDGAR.

Everything downstream runs on real, public, audited data — no synthetic inputs.
For any ticker we pull two things:

  1. The 10-K HTML document  -> the messy prose we extract from and answer over.
  2. The company's XBRL facts -> the SEC's own structured numbers, which we use
     as GROUND TRUTH to grade the LLM's extraction. This is what makes the eval
     real rather than self-graded.

SEC asks that every request carry a descriptive User-Agent with contact info;
set SEC_USER_AGENT in your .env.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
UA = os.getenv("SEC_USER_AGENT", "filings-intelligence example@example.com")
DEFAULT_TICKER = "WRBY"  # Warby Parker — a public DTC consumer brand


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def _cik_for(ticker: str) -> str:
    data = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    for row in data.values():
        if row["ticker"].upper() == ticker.upper():
            return f"{row['cik_str']:010d}"
    raise ValueError(f"ticker not found: {ticker}")


def _latest_10k(cik: str) -> dict:
    sub = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik}.json"))
    r = sub["filings"]["recent"]
    for form, date, acc, doc, rpt in zip(
        r["form"], r["filingDate"], r["accessionNumber"], r["primaryDocument"], r["reportDate"]
    ):
        if form == "10-K":
            return {"accession": acc, "doc": doc, "filed": date, "report_date": rpt}
    raise ValueError("no 10-K found")


def fetch(ticker: str = DEFAULT_TICKER) -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    cik = _cik_for(ticker)
    meta = _latest_10k(cik)
    acc_nodash = meta["accession"].replace("-", "")
    cik_int = int(cik)

    doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{meta['doc']}"
    htm_path = os.path.join(DATA_DIR, f"{ticker.lower()}_10k.htm")
    with open(htm_path, "wb") as f:
        f.write(_get(doc_url))
    time.sleep(0.3)  # be polite to SEC

    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    facts_path = os.path.join(DATA_DIR, f"{ticker.lower()}_xbrl_facts.json")
    with open(facts_path, "wb") as f:
        f.write(_get(facts_url))

    meta.update({"ticker": ticker.upper(), "cik": cik, "doc_url": doc_url,
                 "htm_path": htm_path, "facts_path": facts_path})
    with open(os.path.join(DATA_DIR, f"{ticker.lower()}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "facts_path"}, indent=2))
    return meta


if __name__ == "__main__":
    import sys
    fetch(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKER)
