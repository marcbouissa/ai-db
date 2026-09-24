"""Embedding provider registry: built-ins plus the ``ai_db.embedding`` entry-point group.

An entry point loads a callable ``(options: dict) -> EmbeddingProvider``.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from ai_db.config import ProviderConfig
from ai_db.embed.base import EmbeddingProvider
from ai_db.errors import AiDbConfigError

ENTRY_POINT_GROUP = "ai_db.embedding"


def _st(options: dict[str, Any]) -> EmbeddingProvider:
    from ai_db.embed.st_provider import SentenceTransformersEmbedder
    return SentenceTransformersEmbedder(options)


def _openai(options: dict[str, Any]) -> EmbeddingProvider:
    from ai_db.embed.hosted import OpenAICompatibleEmbedder
    return OpenAICompatibleEmbedder(options)


def _voyage(options: dict[str, Any]) -> EmbeddingProvider:
    from ai_db.embed.hosted import VoyageEmbedder
    return VoyageEmbedder(options)


BUILTIN: dict[str, Callable[[dict[str, Any]], EmbeddingProvider]] = {
    "sentence_transformers": _st,
    "openai_compatible": _openai,
    "voyage": _voyage,
}


def build_embedder(cfg: ProviderConfig) -> EmbeddingProvider | None:
    """Return the configured provider, or None only when ``provider == 'none'``."""
    if cfg.provider == "none":
        return None
    factory = BUILTIN.get(cfg.provider)
    if factory is None:
        plugins = {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}
        if cfg.provider not in plugins:
            raise AiDbConfigError(
                f"unknown embedding provider '{cfg.provider}'; available: "
                f"{sorted([*BUILTIN, *plugins])}"
            )
        factory = plugins[cfg.provider].load()
    options = {k: v for k, v in cfg.options.items() if k != "_note"}
    provider = factory(options)
    if not isinstance(provider, EmbeddingProvider):
        raise AiDbConfigError(f"embedding provider '{cfg.provider}' did not return an EmbeddingProvider")
    return provider
