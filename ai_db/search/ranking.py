"""Final ranking of retriever candidates.

Two fixed formulas (weights in ``ai_db.constants``), chosen by whether a reranker is
configured:

- rerank on:  ``RANK_W_RERANK * rerank + RANK_W_FUSED * fused + RANK_W_GRAPH * graph + RANK_W_EXACT * exact``
- rerank off: ``NORERANK_W_FUSED * fused + NORERANK_W_GRAPH * graph + NORERANK_W_EXACT * exact``

``rerank``, ``fused`` and ``graph`` are min-max normalized over the candidate set;
``exact`` is 1 when a query term equals the last component of the chunk's qualified name.
Only the top ``RERANK_TOP`` fused candidates are reranked; the rest keep fused order below
them. A diversity pass caps results per file inside the first ``DIVERSITY_WINDOW``.
"""

from __future__ import annotations

import time
from typing import Any

from ai_db.constants import (
    DIVERSITY_MAX_PER_FILE,
    DIVERSITY_WINDOW,
    NORERANK_W_EXACT,
    NORERANK_W_FUSED,
    NORERANK_W_GRAPH,
    RANK_W_EXACT,
    RANK_W_FUSED,
    RANK_W_GRAPH,
    RANK_W_RERANK,
    RERANK_TOP,
)
from ai_db.embed.base import document_text
from ai_db.rerank.base import RerankProvider
from ai_db.search.query_builder import expand_terms
from ai_db.storage.models import SearchResult


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 if hi > 0 else 0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def last_component(qualified_name: str) -> str:
    return qualified_name.rsplit(".", 1)[-1].split(" ")[0]


def diversify(results: list[SearchResult], window: int = DIVERSITY_WINDOW,
              max_per_file: int = DIVERSITY_MAX_PER_FILE) -> list[SearchResult]:
    """Greedy: fill the first ``window`` slots with at most ``max_per_file`` per file."""
    head: list[SearchResult] = []
    deferred: list[SearchResult] = []
    per_file: dict[str, int] = {}
    for r in results:
        if len(head) < window and per_file.get(r.filepath, 0) < max_per_file:
            head.append(r)
            per_file[r.filepath] = per_file.get(r.filepath, 0) + 1
        else:
            deferred.append(r)
    return head + deferred


class Ranker:
    def __init__(self, db: Any, reranker: RerankProvider | None = None, doc_weight: float = 0.5):
        self.db = db
        self.reranker = reranker
        self.doc_weight = doc_weight
        self.last_timings: dict[str, float] = {}

    def rank(self, query: str, candidates: list[SearchResult],
             allowed_projects: list[str] | None) -> list[SearchResult]:
        self.last_timings = {}
        if not candidates:
            return []
        t0 = time.perf_counter()
        terms = set(expand_terms(query))
        names = [last_component(c.qualified_name or c.name) for c in candidates]
        centrality = self._centrality(names, allowed_projects)
        t1 = time.perf_counter()

        head = candidates[:RERANK_TOP] if self.reranker else candidates
        tail = candidates[RERANK_TOP:] if self.reranker else []

        fused = _minmax([c.score for c in head])
        graph = _minmax([centrality.get(n, 0.0) for n in names[:len(head)]])
        exact = [1.0 if n.lower() in terms else 0.0 for n in names[:len(head)]]

        # Apply doc_weight to md/txt chunks (documentation)
        if self.doc_weight != 1.0:
            for i, c in enumerate(head):
                if c.chunk_type in ("md", "txt"):
                    fused[i] *= self.doc_weight

        if self.reranker is not None:
            chunks = {c.id: c for c in self.db.get_chunks_by_ids([h.chunk_id for h in head])}
            texts = [document_text(c.language, c.qualified_name, chunks[c.chunk_id].content)
                     if c.chunk_id in chunks else c.snippet for c in head]
            raw = self.reranker.score(query, texts)
            if len(raw) != len(head):
                raise ValueError(f"reranker returned {len(raw)} scores for {len(head)} texts")
            rr = _minmax(raw)
            self.last_timings["rerank_ms"] = (time.perf_counter() - t1) * 1000
            scores = [RANK_W_RERANK * r + RANK_W_FUSED * f + RANK_W_GRAPH * g + RANK_W_EXACT * e
                      for r, f, g, e in zip(rr, fused, graph, exact)]
            for c, r in zip(head, raw):
                c.signals["rerank"] = r
        else:
            scores = [NORERANK_W_FUSED * f + NORERANK_W_GRAPH * g + NORERANK_W_EXACT * e
                      for f, g, e in zip(fused, graph, exact)]

        for c, s, g, e in zip(head, scores, graph, exact):
            c.signals["graph"] = g
            c.signals["exact_symbol"] = e
            c.score = round(s, 6)
        ordered = sorted(head, key=lambda c: c.score, reverse=True)
        floor = min((c.score for c in ordered), default=0.0)
        for i, c in enumerate(tail):
            c.score = round(floor * (1 - (i + 1) / (len(tail) + 1)), 6)
        self.last_timings["graph_ms"] = (t1 - t0) * 1000
        return diversify(ordered + tail)

    def _centrality(self, names: list[str], allowed_projects: list[str] | None) -> dict[str, float]:
        if "graph" not in self.db.capabilities():
            return {}
        scores: dict[str, float] = self.db.get_symbol_centrality(names, allowed_projects)
        return scores
