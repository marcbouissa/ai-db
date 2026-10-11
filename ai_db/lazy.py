"""Lazy provider proxies: the model loads on first vector/rerank use, not at VectorDB construction.

Exact lookups (symbol, callers, todos, status, outline, trace) are pure SQL and must
never pay the multi-second model load. A missing or misconfigured model still raises
``AiDbConfigError`` -- at the moment a model is actually needed (no fallback).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_db.config import ProviderConfig, RerankConfig
from ai_db.embed.base import EmbeddingProvider
from ai_db.errors import AiDbConfigError
from ai_db.rerank.base import RerankProvider


def configured_model_id(cfg: ProviderConfig | RerankConfig) -> str | None:
    """Identifier derived from config alone (never loads a model)."""
    if cfg.provider == "none":
        return None
    model = cfg.options.get("model")
    return f"{cfg.provider}:{model}" if model else cfg.provider


def release_models() -> int:
    """Drop every cached embedding/rerank model and free its memory (GPU included).

    The registries keep one instance per provider config for the whole process,
    so every VectorDB (each daemon project) shares it. Call this only when none of
    them is in use any more -- the daemon does, once no project is loaded. Returns
    the number of models released.
    """
    from ai_db.embed.registry import _EMBEDDER_CACHE
    from ai_db.rerank.registry import _RERANKER_CACHE

    released = 0
    for cache in (_EMBEDDER_CACHE, _RERANKER_CACHE):
        models = list(cache.values())
        cache.clear()
        for inner in models:
            _release(inner)
            released += 1
    return released


def _release(inner: Any) -> None:
    """Free a model's memory (GPU included)."""
    if inner is not None and hasattr(inner, "model"):
        import torch
        inner.model.cpu()
        del inner.model
        torch.cuda.empty_cache()


class LazyEmbedder(EmbeddingProvider):
    def __init__(self, cfg: ProviderConfig, on_load: Callable[[EmbeddingProvider], None] | None = None):
        self._cfg = cfg
        self._on_load = on_load
        self._inner: EmbeddingProvider | None = None

    @property
    def loaded(self) -> bool:
        return self._inner is not None

    def load(self) -> EmbeddingProvider:
        if self._inner is None:
            from ai_db.embed.registry import build_embedder
            try:
                inner = build_embedder(self._cfg)
            except OSError as exc:  # missing weights, no network, unreadable cache
                raise AiDbConfigError(
                    f"cannot load embedding model {configured_model_id(self._cfg)}: {exc}") from exc
            if inner is None:
                raise AiDbConfigError("embedding provider is 'none' but an embedding was requested")
            self._inner = inner
            if self._on_load is not None:
                self._on_load(inner)
        return self._inner

    @property
    def model_id(self) -> str:  # type: ignore[override]
        return self.load().model_id

    @property
    def dim(self) -> int:  # type: ignore[override]
        return self.load().dim

    @property
    def device(self) -> Any:
        return getattr(self.load(), "device", None)

    @property
    def model(self) -> Any:
        return self.load().model  # type: ignore[attr-defined]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.load().embed_documents(texts)

    @property
    def bulk_threshold(self) -> int | None:  # type: ignore[override]
        return self.load().bulk_threshold

    def bulk(self, pending: int) -> Any:
        return self.load().bulk(pending)

    def embed_query(self, text: str) -> list[float]:
        return self.load().embed_query(text)

    def unload(self) -> None:
        # Only this proxy's reference: the registry shares the instance with every
        # other VectorDB, so freeing it here would break them. release_models() frees.
        self._inner = None


class LazyReranker(RerankProvider):
    def __init__(self, cfg: ProviderConfig | RerankConfig):
        self._cfg = cfg
        self._inner: RerankProvider | None = None

    @property
    def loaded(self) -> bool:
        return self._inner is not None

    def load(self) -> RerankProvider:
        if self._inner is None:
            from ai_db.rerank.registry import build_reranker
            try:
                inner = build_reranker(self._cfg)
            except OSError as exc:  # missing weights, no network, unreadable cache
                raise AiDbConfigError(
                    f"cannot load rerank model {configured_model_id(self._cfg)}: {exc}") from exc
            if inner is None:
                raise AiDbConfigError("rerank provider is 'none' but a rerank was requested")
            self._inner = inner
        return self._inner

    @property
    def model_id(self) -> str:  # type: ignore[override]
        return self.load().model_id

    def score(self, query: str, texts: list[str]) -> list[float]:
        return self.load().score(query, texts)

    def unload(self) -> None:
        # Only this proxy's reference: the registry shares the instance with every
        # other VectorDB, so freeing it here would break them. release_models() frees.
        self._inner = None
