"""
Scene Memory - High-level API for 3D scene state.
==================================================
Follows ContextMemory pattern from ai_db.memory.context.
"""

from typing import Any
from ai_db.scene.factory import SceneMemoryFactory
from ai_db.scene.models import SceneState, PromptContext
from ai_db.logger import _logger


class SceneMemory:
    """
    High-level 3D scene memory interface.
    
    Mirrors ContextMemory pattern:
    - Wraps pluggable backend
    - Provides semantic operations
    - Links prompts to scene changes
    """
    
    def __init__(self, backend: str = "sqlite_vec", config: dict[str, Any] | None = None):
        self.backend = SceneMemoryFactory.create(backend, config or {})
        self._blender_capture_func = None  # Set by Blender integration
    
    def set_blender_capture(self, capture_func):
        """Set function to capture current Blender scene state."""
        self._blender_capture_func = capture_func
    
    def snapshot_scene(
        self, 
        prompt: str, 
        project: str = "default",
        operation_type: str = "modify",
        objects_affected: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Capture current Blender scene and link to prompt."""
        if not self._blender_capture_func:
            raise RuntimeError("No Blender capture function set. Call set_blender_capture() first.")
        
        # Capture scene from Blender
        scene = self._blender_capture_func()
        scene.project = project
        scene.tags = tags or []
        
        # Create prompt context
        prompt_ctx = PromptContext(
            prompt=prompt,
            project=project,
            scene_id=scene.scene_id,
            objects_affected=objects_affected or [],
            operation_type=operation_type,
            metadata=metadata or {},
        )
        scene.prompt_context = prompt_ctx
        
        # Save
        scene_id = self.backend.save_scene(scene)
        _logger.info(f"Scene snapshot saved: {scene_id} (prompt: {prompt[:50]}...)")
        return scene_id
    
    def restore_scene(self, scene_id: str, mode: str = "replace") -> SceneState:
        """Load scene from memory (Blender integration applies it)."""
        scene = self.backend.load_scene(scene_id)
        if not scene:
            raise ValueError(f"Scene not found: {scene_id}")
        _logger.info(f"Scene loaded for restore: {scene_id}")
        return scene
    
    def list_scenes(self, project: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List available scene snapshots."""
        return self.backend.list_scenes(project=project, limit=limit)
    
    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene snapshot."""
        return self.backend.delete_scene(scene_id)
    
    def get_history(
        self, 
        object_uuid: str | None = None, 
        scene_id: str | None = None, 
        project: str | None = None,
        limit: int = 50
    ) -> list[PromptContext]:
        """Get change history for object, scene, or project."""
        return self.backend.get_scene_history(
            object_uuid=object_uuid, scene_id=scene_id, project=project, limit=limit
        )
    
    def search_by_prompt(self, query: str, project: str | None = None, top_k: int = 10) -> list[SceneState]:
        """Semantic search across scene history by prompt text."""
        return self.backend.search_scenes_by_prompt(query=query, project=project, top_k=top_k)
    
    def search_objects(self, query: str, project: str | None = None, top_k: int = 20) -> list[dict[str, Any]]:
        """Search for objects matching description across scenes."""
        return self.backend.search_objects_by_description(query=query, project=project, top_k=top_k)
    
    def get_scene_versions(self, scene_id: str) -> list[SceneState]:
        """Get version history for a scene (by parent_scene_id chain)."""
        versions = []
        current = self.backend.load_scene(scene_id)
        while current:
            versions.append(current)
            if current.parent_scene_id:
                current = self.backend.load_scene(current.parent_scene_id)
            else:
                break
        return versions
    
    def fork_scene(self, from_scene_id: str, new_prompt: str, project: str | None = None) -> str:
        """Create a new scene version branching from an existing one."""
        parent = self.backend.load_scene(from_scene_id)
        if not parent:
            raise ValueError(f"Parent scene not found: {from_scene_id}")
        
        # Create new scene with same objects but new ID
        new_scene = SceneState(
            project=project or parent.project,
            name=f"{parent.name} (fork)",
            parent_scene_id=parent.scene_id,
            version=parent.version + 1,
            objects=parent.objects.copy(),
            materials=parent.materials.copy(),
            collections=parent.collections.copy(),
            camera=parent.camera,
            tags=parent.tags + ["fork"],
        )
        
        prompt_ctx = PromptContext(
            prompt=new_prompt,
            project=new_scene.project,
            scene_id=new_scene.scene_id,
            operation_type="fork",
            metadata={"forked_from": from_scene_id},
        )
        new_scene.prompt_context = prompt_ctx
        
        return self.backend.save_scene(new_scene)
