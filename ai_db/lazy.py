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
            inner = build_embedder(self._cfg)
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

    def embed_query(self, text: str) -> list[float]:
        return self.load().embed_query(text)

    def unload(self) -> None:
        inner, self._inner = self._inner, None
        if inner is not None and hasattr(inner, "model"):
            import torch
            inner.model.cpu()
            del inner.model
            torch.cuda.empty_cache()


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
            inner = build_reranker(self._cfg)
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
        inner, self._inner = self._inner, None
        if inner is not None and hasattr(inner, "model"):
            import torch
            inner.model.cpu()
            del inner.model
            torch.cuda.empty_cache()
