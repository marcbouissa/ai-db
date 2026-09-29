"""
Scene Memory Backend Interface.
================================
Follows the exact same pattern as StorageBackend in ai_db.storage.backend.
Pluggable backends for persistent 3D scene state.
"""

from abc import ABC, abstractmethod
from typing import Any
from ai_db.scene.models import SceneState, PromptContext


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
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> set[str]:
        """Capabilities this backend provides."""
        pass
    
    @abstractmethod
    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize backend with configuration."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close backend connections."""
        pass
    
    # --- Scene snapshots ---
    
    @abstractmethod
    def save_scene(self, scene: SceneState) -> str:
        """
        Save a scene snapshot.
        Returns the scene_id.
        """
        pass
    
    @abstractmethod
    def load_scene(self, scene_id: str) -> SceneState | None:
        """Load a scene by ID."""
        pass
    
    @abstractmethod
    def list_scenes(self, project: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List scenes with metadata (no full objects)."""
        pass
    
    @abstractmethod
    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene. Returns True if deleted."""
        pass
    
    # --- Prompt-aware history ---
    
    @abstractmethod
    def save_prompt_context(self, ctx: PromptContext) -> int:
        """Save prompt context linking user prompt to scene changes. Returns context ID."""
        pass
    
    @abstractmethod
    def get_scene_history(
        self, 
        object_uuid: str | None = None, 
        scene_id: str | None = None, 
        project: str | None = None,
        limit: int = 50
    ) -> list[PromptContext]:
        """Get change history for object, scene, or project."""
        pass
    
    # --- Semantic search ---
    
    @abstractmethod
    def search_scenes_by_prompt(
        self, 
        query: str, 
        project: str | None = None, 
        top_k: int = 10
    ) -> list[SceneState]:
        """Semantic search across scene history by prompt text."""
        pass
    
    @abstractmethod
    def search_objects_by_description(
        self,
        query: str,
        project: str | None = None,
        top_k: int = 20
    ) -> list[dict[str, Any]]:
        """Search for objects matching description across scenes."""
        pass


def get_backend_capabilities(backend_name: str) -> set[str]:
    """Get capabilities for a backend name without instantiating."""
    caps = {
        "json_file": {"scene_storage", "prompt_history", "text_search"},
        "sqlite_vec": {"scene_storage", "prompt_history", "text_search", "vector_search", "sql_queries"},
    }
    return caps.get(backend_name, set())
