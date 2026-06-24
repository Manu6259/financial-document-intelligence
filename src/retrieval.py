"""Retrieval layer — vector, lexical (BM25), and hybrid (RRF fusion).

Isolated so the retrieval strategy can evolve without touching Q&A or the eval.

Why hybrid: dense vectors are strong on paraphrased prose ("why did margin
change") but weak on financial tables and exact tokens — a query for "inventory"
doesn't embed close to a wall of numbers. Lexical BM25 nails those exact-token
hits. Reciprocal Rank Fusion (RRF) combines the two rankings so we get both:
semantic recall AND keyword precision. This is what makes the inventory question
retrievable.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

from model import embed
from parse_filing import Chunk, load_chunks

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EMB_CACHE = os.path.join(DATA_DIR, "wrby_chunk_embeddings.npy")
RRF_K = 60  # standard RRF damping constant


def _tokenize(text: str) -> list[str]:
    # Keep numbers (incl. "44,512") as tokens so lexical search can match figures.
    return re.findall(r"[a-z0-9][a-z0-9,\.]*", text.lower())


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-9)


def _embed_chunks(chunks: list[Chunk]) -> np.ndarray:
    if os.path.exists(EMB_CACHE):
        cached = np.load(EMB_CACHE)
        if cached.shape[0] == len(chunks):
            return cached
    vecs = np.asarray(embed([c.text for c in chunks]), dtype="float32")
    np.save(EMB_CACHE, vecs)
    return vecs


class Retriever:
    def __init__(self, ticker: str = "WRBY") -> None:
        self.chunks = load_chunks(ticker)
        self.vecs = _normalize(_embed_chunks(self.chunks))
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in self.chunks])

    # --- individual rankers: return chunk indices best-first ------------------
    def _vector_order(self, query: str) -> list[int]:
        qv = _normalize(np.asarray(embed([query]), dtype="float32"))[0]
        return list(np.argsort(-(self.vecs @ qv)))

    def _bm25_order(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(_tokenize(query))
        return list(np.argsort(-scores))

    # --- public search --------------------------------------------------------
    def search(self, query: str, k: int = 6, mode: str = "hybrid",
               pool: int = 50) -> list[Chunk]:
        if mode == "vector":
            order = self._vector_order(query)[:k]
        elif mode == "bm25":
            order = self._bm25_order(query)[:k]
        elif mode == "hybrid":
            # Weight lexical higher: on filings, exact-token BM25 is the stronger
            # signal (numbers, line items); the eval drove this choice.
            order = self._rrf([(self._vector_order(query)[:pool], 1.0),
                               (self._bm25_order(query)[:pool], 2.0)])[:k]
        else:
            raise ValueError(f"unknown mode: {mode}")
        return [self.chunks[i] for i in order]

    def _rrf(self, weighted_rankings: list[tuple[list[int], float]]) -> list[int]:
        fused: dict[int, float] = defaultdict(float)
        for ranking, weight in weighted_rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] += weight / (RRF_K + rank)
        return [idx for idx, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)]


if __name__ == "__main__":
    import sys
    r = Retriever()
    q = sys.argv[1] if len(sys.argv) > 1 else "What is inventory in 2025?"
    for mode in ("vector", "bm25", "hybrid"):
        hits = r.search(q, k=3, mode=mode)
        print(f"\n[{mode}] top-3 for: {q!r}")
        for c in hits:
            has_inv = "44,512" in c.text or "Inventory" in c.text
            print(f"  [{c.chunk_id}] {c.section[:32]:32} {'★ has inventory' if has_inv else ''}")
