"""Streamlit UI — makes the document-intelligence pipeline tangible.

Two tabs mirroring the two features:
  - Extraction: the income statement pulled from the filing, each number checked
    against the SEC's XBRL value, with the source quote shown.
  - Ask the filing: grounded Q&A with inline [citations], the guardrail verdict,
    and the actual source passages used.

Run:  streamlit run src/app.py
Uses local Ollama embeddings; set OPENAI_API_KEY in .env for real LLM answers.
"""

from __future__ import annotations

import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from ask import Ask
from evaluate import eval_extraction
from groundtruth import load_ground_truth
from model import EMBED_MOCK, EMBED_MODEL, LLM_MODEL, USING_MOCK

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

st.set_page_config(page_title="Filing Intelligence", layout="wide")


@st.cache_resource(show_spinner="Loading the filing…")
def get_ask() -> Ask:
    return Ask()


@st.cache_data
def get_meta() -> dict:
    return json.load(open(os.path.join(DATA_DIR, "wrby_meta.json")))


meta = get_meta()
st.title("Financial Document Intelligence")
st.caption(f"{meta['ticker']} 10-K · filed {meta['filed']} · FY{meta['report_date'][:4]}  ·  "
           f"LLM: {'mock' if USING_MOCK else LLM_MODEL} · "
           f"embeddings: {'hash-mock' if EMBED_MOCK else EMBED_MODEL}")
st.caption("The LLM reads and reasons; every number is verified against a source or SEC XBRL — never invented.")

tab1, tab2 = st.tabs(["📑 Income statement (extracted vs SEC)", "💬 Ask the filing"])

with tab1:
    st.caption("Each figure is extracted from the 10-K text, then checked against the company's filed XBRL value.")
    ext = eval_extraction()
    rows = []
    for r in ext["items"]:
        rows.append({
            "Line item": r["item"],
            "Extracted": f"${r['predicted']:,}" if r["predicted"] is not None else "—",
            "SEC XBRL": f"${r['truth']:,}" if r["truth"] is not None else "—",
            "Match": "✅" if r["correct"] else "❌",
        })
    st.metric("Extraction accuracy vs SEC XBRL", f"{ext['accuracy']:.0%}")
    st.table(rows)

with tab2:
    ask = get_ask()
    q = st.text_input("Ask about the filing",
                      "Why did gross margin change year over year?")
    st.caption("Numeric questions route to structured SEC XBRL (exact); narrative questions route to hybrid RAG.")
    if st.button("Answer") and q:
        with st.spinner("Routing and answering…"):
            out = ask(q)
        st.info(f"Route: **{out['route']}**  ·  " +
                ("structured XBRL lookup — exact value" if out["route"] == "structured"
                 else "hybrid retrieval over the filing"))
        st.markdown("**Answer**")
        st.write(out["answer"])
        if out["grounded"]:
            st.success("✅ Grounded — every figure traces to SEC data or a cited source.")
        else:
            st.error(f"⚠️ Unsupported figures (not found / computed by the model): {out['unsupported_figures']}")
        if out.get("sources"):
            st.markdown("**Sources retrieved**")
            for s in out["sources"]:
                cited = "📌 cited" if s["id"] in out.get("cited_ids", []) else ""
                st.caption(f"[{s['id']}] {s['section']}  {cited}")
