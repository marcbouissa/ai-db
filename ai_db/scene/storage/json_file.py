"""
JSON File Scene Memory Backend.
================================
Zero-dependency backend storing scenes as JSON files.
Following StorageBackend pattern - simple, portable, no external deps.
"""

import json
from pathlib import Path
from typing import Any

from ai_db.scene.storage.base import SceneMemoryBackend
from ai_db.scene.models import SceneState, PromptContext
from ai_db.logger import _logger


class JSONFileSceneBackend(SceneMemoryBackend):
    """Scene memory backend using JSON files on disk."""
    
    backend_name = "json_file"
    capabilities = {"scene_storage", "prompt_history", "text_search"}
    
    def __init__(self):
        self.base_path: Path | None = None
        self.scenes_dir: Path | None = None
        self.history_dir: Path | None = None
    
    def initialize(self, config: dict[str, Any]) -> None:
        base = config.get("path", "~/.ai-db/scene_memory")
        self.base_path = Path(base).expanduser()
        self.scenes_dir = self.base_path / "scenes"
        self.history_dir = self.base_path / "history"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        _logger.info(f"JSONFileSceneBackend initialized at {self.base_path}")
    
    def close(self) -> None:
        pass
    
    # --- Scene snapshots ---
    
    def save_scene(self, scene: SceneState) -> str:
        scene_file = self.scenes_dir / f"{scene.scene_id}.json"
        
        # Also save a project index
        project_file = self.scenes_dir / f".index_{scene.project}.json"
        index = {}
        if project_file.exists():
            try:
                index = json.loads(project_file.read_text())
            except Exception:
                pass
        
        index[scene.scene_id] = {
            "name": scene.name,
            "timestamp": scene.timestamp,
            "version": scene.version,
            "object_count": len(scene.objects),
            "tags": scene.tags,
            "prompt": scene.prompt_context.prompt if scene.prompt_context else "",
        }
        
        # Write files
        scene_file.write_text(json.dumps(scene.to_dict(), indent=2))
        project_file.write_text(json.dumps(index, indent=2))
        
        # Save prompt context separately for history
        if scene.prompt_context:
            scene.prompt_context.scene_id = scene.scene_id
            self.save_prompt_context(scene.prompt_context)
        
        return scene.scene_id
    
    def load_scene(self, scene_id: str) -> SceneState | None:
        # Search across all project indexes
        for index_file in self.scenes_dir.glob(".index_*.json"):
            try:
                index = json.loads(index_file.read_text())
                if scene_id in index:
                    scene_file = self.scenes_dir / f"{scene_id}.json"
                    if scene_file.exists():
                        return SceneState.from_dict(json.loads(scene_file.read_text()))
            except Exception:
                continue
        return None
    
    def list_scenes(self, project: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        scenes = []
        
        if project:
            index_files = [self.scenes_dir / f".index_{project}.json"]
        else:
            index_files = list(self.scenes_dir.glob(".index_*.json"))
        
        for index_file in index_files:
            if not index_file.exists():
                continue
            try:
                index = json.loads(index_file.read_text())
                for scene_id, meta in index.items():
                    meta["scene_id"] = scene_id
                    meta["project"] = index_file.stem.replace(".index_", "")
                    scenes.append(meta)
            except Exception:
                continue
        
        # Sort by timestamp desc
        scenes.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return scenes[:limit]
    
    def delete_scene(self, scene_id: str) -> bool:
        deleted = False
        for index_file in self.scenes_dir.glob(".index_*.json"):
            try:
                index = json.loads(index_file.read_text())
                if scene_id in index:
                    del index[scene_id]
                    index_file.write_text(json.dumps(index, indent=2))
                    deleted = True
            except Exception:
                continue
        
        scene_file = self.scenes_dir / f"{scene_id}.json"
        if scene_file.exists():
            scene_file.unlink()
            deleted = True
        
        return deleted
    
    # --- Prompt-aware history ---
    
    def save_prompt_context(self, ctx: PromptContext) -> int:
        # Use timestamp as ID for JSON backend
        ctx.id = int(ctx.timestamp * 1000000)
        history_file = self.history_dir / f"{ctx.project}_{ctx.id}.json"
        history_file.write_text(json.dumps(ctx.__dict__, indent=2))
        return ctx.id
    
    def get_scene_history(
        self, 
        object_uuid: str | None = None, 
        scene_id: str | None = None, 
        project: str | None = None,
        limit: int = 50
    ) -> list[PromptContext]:
        history = []
        pattern = f"{project}_*.json" if project else "*.json"
        
        for history_file in self.history_dir.glob(pattern):
            try:
                data = json.loads(history_file.read_text())
                ctx = PromptContext(**data)
                
                # Filter
                if scene_id and ctx.scene_id != scene_id:
                    continue
                if object_uuid and object_uuid not in ctx.objects_affected:
                    continue
                
                history.append(ctx)
            except Exception:
                continue
        
        history.sort(key=lambda x: x.timestamp, reverse=True)
        return history[:limit]
    
    # --- Text search (no vectors) ---
    
    def search_scenes_by_prompt(
        self, 
        query: str, 
        project: str | None = None, 
        top_k: int = 10
    ) -> list[SceneState]:
        """Simple text search in prompt context."""
        query_lower = query.lower()
        results = []
        
        for ctx in self.get_scene_history(project=project, limit=1000):
            if query_lower in ctx.prompt.lower():
                scene = self.load_scene(ctx.scene_id)
                if scene:
                    results.append(scene)
                    if len(results) >= top_k:
                        break
        
        return results
    
    def search_objects_by_description(
        self,
        query: str,
        project: str | None = None,
        top_k: int = 20
    ) -> list[dict[str, Any]]:
        """Search objects by name/material across scenes."""
        query_lower = query.lower()
        results = []
        seen = set()
        
        for scene_meta in self.list_scenes(project=project, limit=100):
            scene = self.load_scene(scene_meta["scene_id"])
            if not scene:
                continue
            
            for obj in scene.objects.values():
                key = (obj.uuid, scene.scene_id)
                if key in seen:
                    continue
                
                # Match name, type, materials
                searchable = " ".join([
                    obj.name, obj.type, obj.collection,
                    " ".join(obj.materials), " ".join(obj.custom_props.keys())
                ]).lower()
                
                if query_lower in searchable:
                    results.append({
                        "scene_id": scene.scene_id,
                        "scene_name": scene.name,
                        "object": obj.to_dict(),
                        "match_score": 1.0,  # No ranking in JSON backend
                    })
                    seen.add(key)
                    
                    if len(results) >= top_k:
                        break
        
        return results[:top_k]
