"""Embed chunks that have no vector yet."""

from __future__ import annotations

from typing import Any

from ai_db.embed.base import EmbeddingProvider, document_text
from ai_db.errors import AiDbStorageError


def embed_missing(backend: Any, embedder: EmbeddingProvider, batch_size: int = 64) -> int:
    """Embed every chunk lacking a vector. Returns the number of chunks embedded."""
    total = 0
    while True:
        chunks = backend.chunks_missing_embeddings(limit=batch_size)
        if not chunks:
            return total
        vectors = embedder.embed_documents(
            [document_text(c.language, c.qualified_name, c.content) for c in chunks]
        )
        if len(vectors) != len(chunks):
            raise AiDbStorageError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
        backend.upsert_embeddings([(c.id, v) for c, v in zip(chunks, vectors)])
        total += len(chunks)
