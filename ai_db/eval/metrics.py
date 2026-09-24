"""Pure retrieval-quality metrics with binary relevance.

A ranked item is a ``(filepath, name)`` tuple. An expected item is a
``(filepath, symbol_or_None)`` tuple. A ranked item matches an expected item when
the filepaths are equal and, if a symbol is given, the ranked name contains it.
"""

import math
from collections.abc import Sequence

Ranked = Sequence[tuple[str, str]]
Expected = Sequence[tuple[str, str | None]]


def _matches(item: tuple[str, str], exp: tuple[str, str | None]) -> bool:
    filepath, name = item
    exp_path, exp_symbol = exp
    if filepath != exp_path:
        return False
    if exp_symbol is None:
        return True
    return exp_symbol in name


def relevance_vector(ranked: Ranked, expected: Expected) -> list[int]:
    """Return 1/0 per ranked position; each expected item is credited at most once."""
    used = [False] * len(expected)
    rels: list[int] = []
    for item in ranked:
        hit = 0
        for i, exp in enumerate(expected):
            if not used[i] and _matches(item, exp):
                used[i] = True
                hit = 1
                break
        rels.append(hit)
    return rels


def recall_at_k(ranked: Ranked, expected: Expected, k: int) -> float:
    if not expected:
        raise ValueError("expected must not be empty")
    return sum(relevance_vector(list(ranked)[:k], expected)) / len(expected)


def mrr(ranked: Ranked, expected: Expected) -> float:
    for pos, rel in enumerate(relevance_vector(ranked, expected), start=1):
        if rel:
            return 1.0 / pos
    return 0.0


def ndcg_at_k(ranked: Ranked, expected: Expected, k: int) -> float:
    if not expected:
        raise ValueError("expected must not be empty")
    rels = relevance_vector(list(ranked)[:k], expected)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected), k)))
    return dcg / ideal
