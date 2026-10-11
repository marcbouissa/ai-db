"""Embed chunks that have no vector yet."""

from __future__ import annotations

from typing import Any

from ai_db.embed.base import EmbeddingProvider, document_text
from ai_db.errors import AiDbStorageError


def embed_missing(backend: Any, embedder: EmbeddingProvider, batch_size: int = 64) -> int:
    """Embed every chunk lacking a vector. Returns the number of chunks embedded.

    A backlog of at least ``embedder.bulk_threshold`` chunks (first index,
    ``reindex --embeddings``) runs inside ``embedder.bulk``, which may move the
    model to its bulk device (GPU) for the run and back afterwards.
    """
    threshold = embedder.bulk_threshold
    # Fetching one threshold's worth tells us whether the backlog is large without
    # a separate count query, and is reused as the first batches.
    first = backend.chunks_missing_embeddings(limit=max(batch_size, threshold or 0))
    if not first:
        return 0
    with embedder.bulk(len(first)):
        total = 0
        for start in range(0, len(first), batch_size):
            total += _embed(backend, embedder, first[start:start + batch_size])
        while True:
            chunks = backend.chunks_missing_embeddings(limit=batch_size)
            if not chunks:
                return total
            total += _embed(backend, embedder, chunks)


def _embed(backend: Any, embedder: EmbeddingProvider, chunks: list[Any]) -> int:
    vectors = embedder.embed_documents(
        [document_text(c.language, c.qualified_name, c.content) for c in chunks]
    )
    if len(vectors) != len(chunks):
        raise AiDbStorageError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
    backend.upsert_embeddings([(c.id, v) for c, v in zip(chunks, vectors)])
    return len(chunks)
