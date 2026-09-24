"""Candidate retrieval. Exactly one retriever is chosen at startup from
``retrieval.mode``; there is no switching between them at runtime.

- ``LexicalRetriever``: BM25 over the FTS index.
- ``HybridRetriever``: BM25 top-N and vector top-N fused with Reciprocal Rank Fusion.

Both return candidates whose ``score`` is the fused score and whose ``signals`` hold the
per-list ranks; final ordering (rerank, graph, exact symbol) happens in ``ranking``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from ai_db.constants import RRF_K
from ai_db.embed.base import EmbeddingProvider
from ai_db.search.query_builder import base_terms, expand_terms
from ai_db.storage.models import SearchResult

SNIPPET_CHARS = 280


class Retriever(ABC):
    mode: str

    def __init__(self) -> None:
        self.last_timings: dict[str, float] = {}

    @abstractmethod
    def candidates(self, text: str, filters: dict[str, Any], pool: int) -> list[SearchResult]:
        """Up to ``pool`` candidates, best first, ``score`` in (0, 1]."""


class LexicalRetriever(Retriever):
    mode = "lexical"

    def __init__(self, db: Any):
        super().__init__()
        self.db = db

    def _bm25(self, text: str, filters: dict[str, Any], pool: int) -> list[SearchResult]:
        terms = expand_terms(text)
        if not terms:
            return []
        return self.db.search_chunks(terms, top_k=pool, core_terms=base_terms(text), **filters)

    def candidates(self, text: str, filters: dict[str, Any], pool: int) -> list[SearchResult]:
        t0 = time.perf_counter()
        hits = self._bm25(text, filters, pool)
        self.last_timings = {"bm25_ms": (time.perf_counter() - t0) * 1000}
        for rank, h in enumerate(hits, start=1):
            h.signals["bm25_rank"] = rank
            h.signals["bm25"] = h.score
            h.score = 1.0 / (RRF_K + rank)
            h.signals["fused"] = h.score
        return hits


class HybridRetriever(LexicalRetriever):
    mode = "hybrid"

    def __init__(self, db: Any, embedder: EmbeddingProvider):
        super().__init__(db)
        self.embedder = embedder

    def candidates(self, text: str, filters: dict[str, Any], pool: int) -> list[SearchResult]:
        t0 = time.perf_counter()
        lexical = self._bm25(text, filters, pool)
        t1 = time.perf_counter()
        qvec = self.embedder.embed_query(text)
        t2 = time.perf_counter()
        vec_hits = self.db.search_vectors(qvec, pool, filters)
        t3 = time.perf_counter()

        by_id: dict[int, SearchResult] = {}
        fused: dict[int, float] = {}
        for rank, h in enumerate(lexical, start=1):
            by_id[h.chunk_id] = h
            h.signals["bm25_rank"] = rank
            h.signals["bm25"] = h.score
            fused[h.chunk_id] = 1.0 / (RRF_K + rank)
        missing = [cid for cid, _ in vec_hits if cid not in by_id]
        for chunk in self.db.get_chunks_by_ids(missing):
            snippet = chunk.content[:SNIPPET_CHARS]
            by_id[chunk.id] = SearchResult(
                chunk_id=chunk.id, filepath=chunk.filepath, name=chunk.name,
                chunk_type=chunk.chunk_type, project=chunk.project,
                start_line=chunk.start_line, end_line=chunk.end_line, score=0.0,
                snippet=snippet, qualified_name=chunk.qualified_name,
                language=chunk.language, parent_id=chunk.parent_id,
            )
        for rank, (cid, distance) in enumerate(vec_hits, start=1):
            if cid not in by_id:
                continue
            by_id[cid].signals["vec_rank"] = rank
            by_id[cid].signals["vec_distance"] = distance
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        out = []
        for cid, score in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:pool]:
            h = by_id[cid]
            h.score = score
            h.signals["fused"] = score
            out.append(h)
        self.last_timings = {
            "bm25_ms": (t1 - t0) * 1000,
            "embed_ms": (t2 - t1) * 1000,
            "vec_ms": (t3 - t2) * 1000,
        }
        return out
