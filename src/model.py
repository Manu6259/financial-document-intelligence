"""Provider wrapper: LLM (OpenAI) + embeddings (local Ollama) + offline mock.

Design choices that matter for a finance system:
  - One place to call the LLM, so the provider is swappable.
  - Embeddings run LOCALLY via Ollama (nomic-embed-text). Embedding a whole
    filing through a hosted API is wasteful; local is free and private — and for
    financial documents, "private" matters.
  - An offline mock (no key / APP_USE_MOCK=1) keeps everything runnable for a
    reviewer: deterministic extraction + lexical retrieval, so the pipeline and
    eval execute end-to-end with zero credentials.

The LLM's job is to READ and REASON over retrieved text. It is never the source
of truth for a number — every figure it emits is checked against a quoted span
(see rag_qa.py) or against SEC XBRL (see evaluate.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("APP_LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("APP_EMBED_MODEL", "nomic-embed-text")
OLLAMA_URL = os.getenv("APP_OLLAMA_URL", "http://localhost:11434")
_FORCE_MOCK = os.getenv("APP_USE_MOCK", "0") == "1"
_HAS_KEY = bool(os.getenv("OPENAI_API_KEY"))

USING_MOCK = _FORCE_MOCK or not _HAS_KEY

_client = None
if not USING_MOCK:
    try:
        from openai import OpenAI

        _client = OpenAI()
    except Exception:
        USING_MOCK = True


def chat_json(system: str, user: str) -> dict:
    """Call the LLM and parse a JSON object reply. Mock returns {} (callers
    have deterministic fallbacks), so the pipeline still runs without a key."""
    if USING_MOCK:
        return {}
    resp = _client.chat.completions.create(
        model=LLM_MODEL, temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except json.JSONDecodeError:
        return {}


# --- Embeddings ---------------------------------------------------------------

def _ollama_embed(texts: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    req = urllib.request.Request(f"{OLLAMA_URL}/api/embed", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["embeddings"]


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    out = []
    for i in range(dim):
        h = hashlib.sha256(f"{i}:{text.lower()}".encode()).digest()
        out.append((int.from_bytes(h[:4], "big") / 2**32) * 2 - 1)
    return out


def _ollama_up() -> bool:
    """Embeddings are independent of the OpenAI LLM: local Ollama works even with
    no API key. Only APP_USE_MOCK forces the hash fallback."""
    if _FORCE_MOCK:
        return False
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=3).read()
        return True
    except Exception:
        return False


EMBED_MOCK = not _ollama_up()  # True only if Ollama is unavailable


def embed(texts: list[str]) -> list[list[float]]:
    if EMBED_MOCK:
        return [_hash_embed(t) for t in texts]
    try:
        return _ollama_embed(texts)
    except Exception:
        return [_hash_embed(t) for t in texts]
