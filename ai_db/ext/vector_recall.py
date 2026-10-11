"""Built-in recall provider ``vectors``: semantic similarity over embeddings.

Embeds each skill (``description + triggers``) and each session context
(``title + summary``) with the configured embedding provider, and ranks them by cosine
similarity to the prompt. Registered through the ``ai_db.recall`` entry point like any
third-party provider (see ``ai_db.ext.recall``); enabled by default whenever an embedding
provider is configured.

Vectors are tied to the model that made them: when ``embedding.model`` changes, the old
vectors are dropped and everything is re-embedded on first use, so vectors from two
models are never compared. Embedding failures propagate -- they are not skipped.
"""

from __future__ import annotations

from typing import Any

from ai_db.constants import RECALL_EMBED_BATCH, RECALL_VECTORS_STATE_KEY
from ai_db.errors import AiDbConfigError
from ai_db.ext.recall import ContextHit, RecallProvider, SkillHit
from ai_db.storage.models import SkillRecord


def _skill_text(skill: SkillRecord) -> str:
    return f"{skill.description or ''}\n{skill.triggers or ''}".strip() or skill.name


def _context_text(title: str, summary: str) -> str:
    return f"{title}\n{summary}".strip()


class VectorRecall(RecallProvider):
    name = "vectors"

    def __init__(self, backend: Any, embedder: Any):
        self.backend = backend
        self.embedder = embedder
        self._model_checked = False

    @classmethod
    def from_options(cls, backend: Any, embedder: Any, options: dict[str, Any]) -> VectorRecall:
        if embedder is None:
            raise AiDbConfigError(
                "recall provider 'vectors' needs an embedding provider, but embedding.provider "
                "is 'none'. Configure one, or remove 'vectors' from recall.providers.")
        if options:
            raise AiDbConfigError(f"recall provider 'vectors' takes no options, got {sorted(options)}")
        return cls(backend, embedder)

    # --- model guard -----------------------------------------------------------

    def _ensure_model(self) -> None:
        """Drop and rebuild the vectors when they were made by another model."""
        if self._model_checked:
            return
        self._model_checked = True
        current = {"model_id": self.embedder.model_id}
        if self.backend.get_state(RECALL_VECTORS_STATE_KEY) != current:
            self.backend.clear_recall_vectors()
            self.backend.set_state(RECALL_VECTORS_STATE_KEY, current)
            self._backfill()

    def _backfill(self) -> int:
        count = 0
        skills = self.backend.skills_without_vectors()
        for i in range(0, len(skills), RECALL_EMBED_BATCH):
            batch = skills[i:i + RECALL_EMBED_BATCH]
            vectors = self.embedder.embed_documents([_skill_text(s) for s in batch])
            for skill, vec in zip(batch, vectors):
                self.backend.upsert_skill_vector(skill.name, skill.project, vec)
            count += len(batch)
        contexts = self.backend.contexts_without_vectors()
        for i in range(0, len(contexts), RECALL_EMBED_BATCH):
            batch = contexts[i:i + RECALL_EMBED_BATCH]
            vectors = self.embedder.embed_documents([_context_text(t, s) for _, t, s in batch])
            for (context_id, _, _), vec in zip(batch, vectors):
                self.backend.upsert_context_vector(context_id, vec)
            count += len(batch)
        return count

    # --- RecallProvider --------------------------------------------------------

    def index_skill(self, skill: SkillRecord) -> None:
        self._ensure_model()
        vec = self.embedder.embed_documents([_skill_text(skill)])[0]
        self.backend.upsert_skill_vector(skill.name, skill.project, vec)

    def index_context(self, context_id: int, title: str, summary: str) -> None:
        self._ensure_model()
        vec = self.embedder.embed_documents([_context_text(title, summary)])[0]
        self.backend.upsert_context_vector(context_id, vec)

    def sync(self) -> int:
        self._ensure_model()
        return self._backfill()

    def rank_skills(self, prompt: str, allowed_projects: list[str], limit: int) -> list[SkillHit]:
        self._ensure_model()
        query = self.embedder.embed_query(prompt)
        return [SkillHit(name, project, max(0.0, 1.0 - dist)) for name, project, dist
                in self.backend.search_skills_vector(query, allowed_projects, limit)]

    def rank_contexts(self, query: str, allowed_projects: list[str], limit: int) -> list[ContextHit]:
        self._ensure_model()
        vec = self.embedder.embed_query(query)
        return [ContextHit(h["session_id"], h["project"], h["title"] or "", h["summary"],
                           h["timestamp"], max(0.0, 1.0 - h["distance"]))
                for h in self.backend.search_contexts_vector(vec, allowed_projects, limit)]
