"""Exemplar ranking over run records (pure; backends supply the runs)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_db.scene.models import RunRecord

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def rank_exemplars(runs: list[RunRecord], brief: str, min_score: float, top_k: int) -> list[RunRecord]:
    """Keep only successful runs with judge score >= min_score; rank by brief token overlap
    (Jaccard), ties broken by score then recency. Failed/low-score/unjudged runs are excluded."""
    q = _tokens(brief)
    scored: list[tuple[float, float, float, RunRecord]] = []
    for r in runs:
        s = r.score
        if r.outcome != "success" or s is None or s < min_score:
            continue
        t = _tokens(r.brief)
        sim = len(q & t) / len(q | t) if q | t else 0.0
        scored.append((sim, s, r.created_at, r))
    scored.sort(key=lambda x: x[:3], reverse=True)
    return [x[3] for x in scored[:top_k]]
