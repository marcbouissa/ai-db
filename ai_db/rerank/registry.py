"""Rerank provider registry: built-ins plus the ``ai_db.rerank`` entry-point group.

An entry point loads a callable ``(options: dict) -> RerankProvider``.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from ai_db.config import ProviderConfig
from ai_db.errors import AiDbConfigError
from ai_db.rerank.base import RerankProvider

ENTRY_POINT_GROUP = "ai_db.rerank"


def _load(name: str) -> Callable[[dict[str, Any]], RerankProvider]:
    from ai_db.rerank import providers
    return {
        "sentence_transformers": providers.CrossEncoderReranker,
        "voyage": providers.VoyageReranker,
        "cohere": providers.CohereReranker,
    }[name]


BUILTIN: dict[str, Callable[[dict[str, Any]], RerankProvider]] = {
    name: (lambda opts, _n=name: _load(_n)(opts))
    for name in ("sentence_transformers", "voyage", "cohere")
}


def build_reranker(cfg: ProviderConfig) -> RerankProvider | None:
    """Return the configured reranker, or None only when ``provider == 'none'``."""
    if cfg.provider == "none":
        return None
    factory = BUILTIN.get(cfg.provider)
    if factory is None:
        plugins = {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}
        if cfg.provider not in plugins:
            raise AiDbConfigError(
                f"unknown rerank provider '{cfg.provider}'; available: {sorted([*BUILTIN, *plugins])}"
            )
        factory = plugins[cfg.provider].load()
    options = {k: v for k, v in cfg.options.items() if k != "_note"}
    provider = factory(options)
    if not isinstance(provider, RerankProvider):
        raise AiDbConfigError(f"rerank provider '{cfg.provider}' did not return a RerankProvider")
    return provider
