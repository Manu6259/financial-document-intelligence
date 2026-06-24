"""Turn the raw 10-K HTML into clean text, sections, and citable chunks.

A 10-K is messy: nested tables, inline XBRL tags, page furniture, inconsistent
headings. We:
  1. strip it to readable text,
  2. split into the standard "Item N." sections (so answers can cite a section),
  3. window the text into overlapping chunks with stable ids, so retrieval can
     point at an exact, quotable span — the basis for citations.

Every chunk keeps its character offset and section, which is what lets the Q&A
layer say "this number came from Item 7, chunk 42" and lets the guardrail verify
a quoted figure actually appears in a real source span.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, asdict

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# 10-Ks are inline-XBRL (XML-in-HTML); we intentionally parse as HTML for text.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

CHUNK_CHARS = 1100
CHUNK_OVERLAP = 150

# Canonical 10-K item headings we try to segment on.
ITEM_RE = re.compile(r"\bItem\s+(\d+[A-Z]?)\.?\s+([A-Z][A-Za-z'&,\- ]{3,60})")


@dataclass
class Chunk:
    chunk_id: int
    section: str
    start: int
    text: str


def clean_text(htm_path: str) -> str:
    html = open(htm_path, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(" ")
    text = re.sub(r"\xa0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _section_at(text: str, pos: int, headers: list[tuple[int, str]]) -> str:
    cur = "Front matter"
    for start, label in headers:
        if start <= pos:
            cur = label
        else:
            break
    return cur


def chunk_filing(htm_path: str) -> list[Chunk]:
    text = clean_text(htm_path)
    headers = [(m.start(), f"Item {m.group(1)} — {m.group(2).strip()}")
               for m in ITEM_RE.finditer(text)]
    chunks: list[Chunk] = []
    cid = 0
    step = CHUNK_CHARS - CHUNK_OVERLAP
    for start in range(0, len(text), step):
        body = text[start:start + CHUNK_CHARS].strip()
        if len(body) < 200:
            continue
        chunks.append(Chunk(cid, _section_at(text, start, headers), start, body))
        cid += 1
    return chunks


def save_chunks(ticker: str) -> dict:
    htm = os.path.join(DATA_DIR, f"{ticker.lower()}_10k.htm")
    chunks = chunk_filing(htm)
    out = os.path.join(DATA_DIR, f"{ticker.lower()}_chunks.json")
    with open(out, "w") as f:
        json.dump([asdict(c) for c in chunks], f)
    sections = sorted({c.section for c in chunks})
    summary = {"chunks": len(chunks), "sections_found": len(sections),
               "sample_sections": sections[:12]}
    print(json.dumps(summary, indent=2))
    return summary


def load_chunks(ticker: str) -> list[Chunk]:
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_chunks.json")
    return [Chunk(**c) for c in json.load(open(path))]


if __name__ == "__main__":
    import sys
    save_chunks(sys.argv[1] if len(sys.argv) > 1 else "WRBY")
