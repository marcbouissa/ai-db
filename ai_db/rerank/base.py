"""Rerank provider contract."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RerankProvider(ABC):
    """Scores (query, document) pairs; higher is more relevant. One score per text."""

    model_id: str

    @abstractmethod
    def score(self, query: str, texts: list[str]) -> list[float]: ...
