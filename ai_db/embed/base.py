"""Embedding provider contract."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return [v / norm for v in vec]


class EmbeddingProvider(ABC):
    """Turns text into L2-normalized vectors of a fixed dimension.

    ``model_id`` identifies provider + model (e.g. ``sentence_transformers:Qwen/Qwen3-Embedding-0.6B``)
    and is stored with the index; changing it requires ``ai-db reindex --embeddings``.
    """

    model_id: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]: ...

    #: Pending documents at or above which ``bulk`` switches device; None = never.
    bulk_threshold: int | None = None

    @contextmanager
    def bulk(self, pending: int) -> Iterator[None]:
        """Scope a large embedding run (``pending`` documents to go).

        Providers with a separate bulk device (a GPU for the first index, a CPU
        for day-to-day) switch inside this block; the default does nothing.
        """
        yield


def document_text(language: str, qualified_name: str, content: str) -> str:
    """The exact text embedded for a chunk."""
    return f"{language} {qualified_name}\n{content}"
