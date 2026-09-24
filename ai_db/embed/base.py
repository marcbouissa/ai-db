"""Embedding provider contract."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


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


def document_text(language: str, qualified_name: str, content: str) -> str:
    """The exact text embedded for a chunk."""
    return f"{language} {qualified_name}\n{content}"
