"""
Scene Memory Backend Interface.
================================
Follows the exact same pattern as StorageBackend in ai_db.storage.backend.
Pluggable backends for persistent 3D scene state.
"""

from abc import ABC, abstractmethod
from typing import Any

from ai_db_scene.models import PromptContext, RunRecord, SceneState
from ai_db_scene.runs import rank_exemplars


class SceneMemoryBackend(ABC):
    """
    Abstract base for scene memory backends.
    
    Mirrors StorageBackend pattern:
    - backend_name property
    - initialize(config) / close()
    - save_scene / load_scene / list_scenes / delete_scene
    - save_prompt_context / get_scene_history
    - search_scenes_by_prompt (semantic search)
    """
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Unique backend identifier (e.g., 'sqlite_vec', 'json_file')."""
    
    @property
    @abstractmethod
    def capabilities(self) -> set[str]:
        """Capabilities this backend provides."""
    
    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize backend with configuration."""
    
    @abstractmethod
    def close(self) -> None:
        """Close backend connections."""
    
    # --- Scene snapshots ---
    
    @abstractmethod
    def save_scene(self, scene: SceneState) -> str:
        """
        Save a scene snapshot.
        Returns the scene_id.
        """
    
    @abstractmethod
    def load_scene(self, scene_id: str) -> SceneState | None:
        """Load a scene by ID."""
    
    @abstractmethod
    def list_scenes(self, project: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List scenes with metadata (no full objects)."""
    
    @abstractmethod
    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene. Returns True if deleted."""
    
    # --- Prompt-aware history ---
    
    @abstractmethod
    def save_prompt_context(self, ctx: PromptContext) -> int:
        """Save prompt context linking user prompt to scene changes. Returns context ID."""
    
    @abstractmethod
    def get_scene_history(
        self, 
        object_uuid: str | None = None, 
        scene_id: str | None = None, 
        project: str | None = None,
        limit: int = 50
    ) -> list[PromptContext]:
        """Get change history for object, scene, or project."""
    
    # --- Semantic search ---
    
    @abstractmethod
    def search_scenes_by_prompt(
        self, 
        query: str, 
        project: str | None = None, 
        top_k: int = 10
    ) -> list[SceneState]:
        """Semantic search across scene history by prompt text."""
    
    @abstractmethod
    def search_objects_by_description(
        self,
        query: str,
        project: str | None = None,
        top_k: int = 20
    ) -> list[dict[str, Any]]:
        """Search for objects matching description across scenes."""

    # --- Run records (pipeline Stage 7) ---

    @abstractmethod
    def save_run(self, run: RunRecord) -> str:
        """Persist a run record (upsert by run_id). Returns run_id."""

    @abstractmethod
    def load_run(self, run_id: str) -> RunRecord | None:
        pass

    @abstractmethod
    def list_runs(self, project: str | None = None, limit: int = 100) -> list[RunRecord]:
        """Runs newest first."""

    def search_exemplars(
        self, brief: str, min_score: float = 4.0, top_k: int = 3, project: str | None = None
    ) -> list[RunRecord]:
        """Successful, judged runs scoring >= min_score, ranked by brief similarity.

        Failed / low-score / unjudged runs are never returned.
        """
        return rank_exemplars(self.list_runs(project=project, limit=10_000), brief, min_score, top_k)


def get_backend_capabilities(backend_name: str) -> set[str]:
    """Get capabilities for a backend name without instantiating."""
    caps = {
        "json_file": {"scene_storage", "prompt_history", "text_search"},
        "sqlite_vec": {"scene_storage", "prompt_history", "text_search", "vector_search", "sql_queries"},
    }
    return caps.get(backend_name, set())
