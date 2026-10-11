"""Recall extensions: extra ranking signals for skill routing and context recall.

A recall provider sees every skill and session context ai-db stores, and contributes
candidates when a prompt is routed to skills (``route_skills``) or contexts are recalled
(``query_contexts``). Keyword search (BM25 + trigger phrases) always runs; providers add
to it.

Providers are found through the ``ai_db.recall`` entry-point group, the same way storage
backends are. The built-in ``vectors`` provider (semantic similarity over embeddings,
``ai_db.ext.vector_recall``) ships with ai-db and is registered in its own pyproject.toml::

    [project.entry-points."ai_db.recall"]
    vectors = "ai_db.ext.vector_recall:VectorRecall.from_options"

A third-party package registers its own the same way; the config picks which run::

    "recall": {"providers": ["vectors"]}

The entry point is a factory ``(backend, embedder, options) -> RecallProvider``;
``embedder`` is None when no embedding provider is configured.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

from ai_db.errors import AiDbConfigError
from ai_db.storage.models import SkillRecord

RECALL_GROUP = "ai_db.recall"


@dataclass(frozen=True)
class SkillHit:
    name: str
    project: str
    similarity: float  # 0..1, higher is closer


@dataclass(frozen=True)
class ContextHit:
    session_id: str
    project: str
    title: str
    summary: str
    timestamp: float
    similarity: float  # 0..1, higher is closer


class RecallProvider(ABC):
    """One extra ranking signal over skills and session contexts."""

    name: str = ""

    @abstractmethod
    def index_skill(self, skill: SkillRecord) -> None:
        """A skill was added or changed (its stored row is already written)."""

    @abstractmethod
    def index_context(self, context_id: int, title: str, summary: str) -> None:
        """A session context was saved (its stored row is already written)."""

    @abstractmethod
    def sync(self) -> int:
        """Bring the provider's index up to date (backfill, model change).
        Returns the number of items (re)indexed."""

    @abstractmethod
    def rank_skills(self, prompt: str, allowed_projects: list[str], limit: int) -> list[SkillHit]:
        """Skills most related to ``prompt``, best first."""

    @abstractmethod
    def rank_contexts(self, query: str, allowed_projects: list[str], limit: int) -> list[ContextHit]:
        """Contexts most related to ``query``, best first."""


def build_recall_providers(names: list[str], backend: Any, embedder: Any) -> list[RecallProvider]:
    """Instantiate the configured providers (``recall.providers``), in order."""
    available = {ep.name: ep for ep in entry_points(group=RECALL_GROUP)}
    providers: list[RecallProvider] = []
    for name in names:
        ep = available.get(name)
        if ep is None:
            raise AiDbConfigError(
                f"recall provider '{name}' is not installed (available: {sorted(available)})")
        factory = ep.load()
        providers.append(factory(backend, embedder, {}))
    return providers
