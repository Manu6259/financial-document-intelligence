# Case study: extracting trustworthy financials from real SEC filings

*RAG + structured extraction over a real 10-K, graded against the SEC's own numbers.*

This shows how I use LLMs on **real financial documents** — RAG pipelines,
structured extraction, and evaluation frameworks where hallucinations are
expensive. It runs on **100% real, public, audited data** and grades itself
against an objective answer key, not its own opinion.

---

## Why documents (and why this is the right place for RAG)

Transaction categorization — the task in my companion reconciliation project —
turns out **not** to need RAG: the records are near-duplicate strings, better
served by rules + a classifier. RAG earns its keep when the model must pull facts
out of a **long document it can't memorise** and answer with citations. In
finance that's 10-Ks, contracts, invoices. So I put RAG where it belongs.

## The data is real, top to bottom

- **Input:** Warby Parker's FY2025 10-K, pulled live from SEC EDGAR (~450k chars
  of real, messy prose and tables). Warby Parker is a public DTC consumer brand.
- **Answer key:** the company's **XBRL** filing — the structured figures it
  submitted to the SEC. This is what lets me grade extraction objectively.

## The one principle (same as my other project)

> **The LLM reads and reasons; it never invents a number.**

Two mechanisms enforce it:
1. **Extraction** — every figure the model pulls is compared to authoritative SEC
   XBRL.
2. **Q&A** — the model answers only from retrieved passages, cites them as `[id]`,
   and a deterministic guardrail rejects any figure in the answer that doesn't
   appear in a cited source.

## What I built

- **`fetch_filing.py`** — pulls the 10-K HTML + XBRL facts for any ticker.
- **`parse_filing.py`** — turns the messy HTML into clean text, `Item N.` sections,
  and overlapping **citable chunks** (so every answer can point at an exact span).
- **`extract.py`** — locates the income statement and extracts it into a schema
  where each number carries its **source quote**.
- **`rag_qa.py`** — local-embedding retrieval (Ollama `nomic-embed-text`) +
  grounded answer + the **numeric hallucination guardrail**.
- **`evaluate.py`** — extraction-vs-XBRL accuracy + a guardrail self-test.

No fine-tuning. Embeddings run locally (free, private — which matters for
financial documents). The whole thing runs with **no API key** via a deterministic
mock, so a reviewer can clone and run it.

## Results (offline mock baseline, Warby Parker FY2025)

Extraction graded against SEC XBRL:

| Line item | Extracted | SEC XBRL | ✓ |
|---|--:|--:|:--:|
| revenue | $871,905,000 | $871,905,000 | ✅ |
| gross_profit | $470,579,000 | $470,579,000 | ✅ |
| operating_income | −$5,336,000 | −$5,336,000 | ✅ |
| net_income | null | $1,641,000 | ❌ |

**75% even from a regex baseline** — and the one miss is instructive: the table
reads "Net income **(loss)** 1,641", and the parenthetical breaks naive parsing.
That's exactly the messiness an LLM is there to absorb, and the eval *measures*
the gap rather than hand-waving it. (With the real LLM I'd expect 4/4; the harness
is identical either way.)

Guardrail self-test: feeds a hallucinated `$999,999,999` → **flagged UNSUPPORTED ✅**;
feeds a grounded answer → **passes ✅**. So the protection against confident wrong
numbers is demonstrated, not asserted.

Retrieval sanity check: asking *"why did gross margin change?"* retrieves the exact
MD&A passage ("Gross margin… increased by 80 basis points… driven by faster
growth in…") — real semantic retrieval over the real filing.

## Honest limitations

- One company, one statement (income statement). The pattern extends directly to
  the balance sheet / cash flow and to multi-filing comparison.
- The mock is a regex baseline to keep things runnable offline; the headline
  numbers should be regenerated with the real LLM.
- XBRL tag names vary by company; `groundtruth.py` maps the common ones and would
  need a small mapping layer to generalise across many filers.

## How this pairs with the reconciliation project

Together the two projects cover a broad surface of financial AI:
- **Reconciliation project** — messy *operational* data, deterministic matching,
  categorization, a multi-step agent, eval. (Synthetic, because real paired data
  is proprietary.)
- **This project** — *document* intelligence: RAG, structured extraction, and an
  evaluation graded on real regulator data.

Same spine in both: **the LLM judges/reads; deterministic checks own the numbers;
everything is evaluated against an objective reference.** That's how you make AI
outputs you can actually put in front of a finance team.
