# Financial Document Intelligence

Ask questions and extract structured financials from **real SEC 10-K filings**,
with **citations** and a **hallucination guardrail** — graded against the SEC's
own XBRL numbers.

This is the document-grounded companion to the reconciliation project. Where that
one handles messy *operational* data (necessarily synthetic, because real paired
bank/payout data is proprietary), this one runs entirely on **real, public,
audited data**: a public consumer brand's 10-K and the structured figures it
filed with the regulator.

It's built around the same finance-first opinion:

> **The LLM reads and reasons; it never invents a number.**
> Every figure it reports must trace to a quoted source span (Q&A) or is checked
> against authoritative SEC XBRL (extraction). In finance, a confident wrong
> number is the expensive failure — so numbers are always verified, never trusted.

Default subject: **Warby Parker (WRBY)** — a public DTC consumer brand. Works on
any ticker with a 10-K.

---

## Quickstart

```bash
pip install -r requirements.txt          # into a venv
cp .env.example .env                      # set SEC_USER_AGENT; OPENAI_API_KEY optional

python src/fetch_filing.py WRBY           # 1. pull the real 10-K + XBRL ground truth
python src/parse_filing.py WRBY           # 2. clean -> sections -> citable chunks
python src/extract.py                     # 3. structured income-statement extraction
python src/rag_qa.py "Why did gross margin change?"   # 4. grounded Q&A with citations
python src/evaluate.py                    # 5. extraction vs SEC XBRL + guardrail test
streamlit run src/app.py                  # 6. interactive UI
```

**Runs with no OpenAI key.** Embeddings are local via **Ollama**
(`ollama pull nomic-embed-text`); the LLM falls back to a deterministic mock
(regex extraction + lexical retrieval) so the whole pipeline and eval execute
for free. Add `OPENAI_API_KEY` to `.env` to run the real agent.

---

## The two features

### 1. Structured extraction (`extract.py`)
Locates the Consolidated Statements of Operations, and pulls the income statement
into a clean schema where **each number carries the source quote it came from**.
The number is meaningless without its provenance — that pairing is what makes it
auditable.

### 2. Grounded Q&A with a guardrail (`rag_qa.py`)
Classic RAG used for what RAG is actually good at — retrieving the relevant
passages from a long document and answering over them:
- the LLM answers **only** from retrieved passages and cites them inline as `[id]`;
- a deterministic guardrail then checks that **every dollar/percent figure in the
  answer appears in a cited source chunk**. Anything that doesn't is flagged
  `UNSUPPORTED`. The model can't smuggle in an invented figure.

---

## Why the evaluation is real

`evaluate.py` grades two things against objective references, not itself:

- **Extraction accuracy** — each extracted line item is compared to the
  **SEC XBRL** value the company actually filed (`revenue`, `gross_profit`,
  `operating_income`, `net_income`). The regulator's data is the answer key.
- **Guardrail efficacy** — a deterministic test feeds the guardrail a
  hallucinated number (absent from sources) and confirms it's flagged, and feeds
  a grounded answer and confirms it passes.

Example (offline mock baseline, Warby Parker FY2025):

| Line item | Extracted | SEC XBRL | ✓ |
|---|--:|--:|:--:|
| revenue | $871,905,000 | $871,905,000 | ✅ |
| gross_profit | $470,579,000 | $470,579,000 | ✅ |
| operating_income | −$5,336,000 | −$5,336,000 | ✅ |
| net_income | null | $1,641,000 | ❌ |

The mock's regex even nails 3/4 — and misses `net_income` precisely because the
table reads "Net income **(loss)** 1,641" and the parenthetical breaks naive
parsing. That's the messiness the LLM is there to handle; the eval quantifies the
gap. Guardrail self-test: hallucinated number flagged ✅, grounded answer passes ✅.

---

## Architecture

```
SEC EDGAR ─┬─ 10-K HTML  ─► parse ─► sections + citable chunks ─┬─► extract (LLM) ─► income statement + source quotes
           │                                                    └─► embed (Ollama) ─► retrieve ─► answer + [citations]
           └─ XBRL facts ─────────────────────────────────────────► ground truth ─► evaluate (accuracy + guardrail)
```

| Module | Role |
|---|---|
| `fetch_filing.py` | Pull a real 10-K + XBRL facts for any ticker |
| `parse_filing.py` | HTML → clean text → `Item N.` sections → overlapping citable chunks |
| `groundtruth.py` | Load SEC XBRL figures as the eval answer key |
| `model.py` | LLM (OpenAI) + local embeddings (Ollama) + offline mock; one swap point |
| `extract.py` | Structured income-statement extraction with provenance |
| `rag_qa.py` | Retrieval + grounded answer + numeric hallucination guardrail |
| `evaluate.py` | Extraction-vs-XBRL accuracy + guardrail self-test |
| `app.py` | Streamlit UI |

---

## Capabilities and how they scale

- **RAG pipelines** — grounded Q&A over a real filing (retrieval where it belongs:
  long documents). Scales via hybrid lexical+vector retrieval and reranking
  (already prototyped here) to many documents.
- **Structured extraction from financial documents** — the income-statement
  extractor with provenance; extends to balance sheet / cash flow and to bulk
  ingestion across thousands of filings.
- **Evaluation against real ground truth** — graded against SEC XBRL, plus a
  guardrail that provably catches invented numbers. The harness generalizes to
  any filer and any line item.
- **Handling messiness** — 10-Ks are dense, footnoted, and inconsistent; the
  `(loss)` parenthetical and table-vs-prose retrieval gaps are real examples the
  system is designed around.
