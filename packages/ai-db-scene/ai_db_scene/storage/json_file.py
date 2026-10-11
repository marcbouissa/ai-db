"""
JSON File Scene Memory Backend.
================================
Zero-dependency backend storing scenes as JSON files.
Following StorageBackend pattern - simple, portable, no external deps.
"""

import json
from pathlib import Path
from typing import Any, ClassVar

from ai_db.errors import AiDbStorageError
from ai_db.logger import _logger
from ai_db_scene.models import PromptContext, RunRecord, SceneState
from ai_db_scene.storage.base import SceneMemoryBackend


def _read_json(path: Path) -> Any:
    """Read a scene-store JSON file. A corrupt file raises instead of being skipped:
    silently skipping it hides data loss, and ``save_scene`` would overwrite it."""
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise AiDbStorageError(f"corrupt scene store file {path}: {e}") from e

class JSONFileSceneBackend(SceneMemoryBackend):
    """Scene memory backend using JSON files on disk."""
    
    backend_name = "json_file"
    capabilities: ClassVar[set[str]] = {"scene_storage", "prompt_history", "text_search"}
    
    def __init__(self):
        self.base_path: Path | None = None
        self._scenes_dir: Path | None = None
        self._history_dir: Path | None = None
        self._runs_dir: Path | None = None
    
    def _require(self, path: Path | None) -> Path:
        if path is None:
            raise AiDbStorageError("JSONFileSceneBackend not initialized")
        return path

    @property
    def scenes_dir(self) -> Path:
        return self._require(self._scenes_dir)

    @property
    def history_dir(self) -> Path:
        return self._require(self._history_dir)

    @property
    def runs_dir(self) -> Path:
        return self._require(self._runs_dir)

    def initialize(self, config: dict[str, Any]) -> None:
        base = config.get("path", "~/.ai-db/scene_memory")
        self.base_path = Path(base).expanduser()
        self._scenes_dir = self.base_path / "scenes"
        self._history_dir = self.base_path / "history"
        self._runs_dir = self.base_path / "runs"
        self._scenes_dir.mkdir(parents=True, exist_ok=True)
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        self._history_dir.mkdir(parents=True, exist_ok=True)
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
            index = _read_json(project_file)
        
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
            index = _read_json(index_file)
            if scene_id in index:
                scene_file = self.scenes_dir / f"{scene_id}.json"
                if scene_file.exists():
                    return SceneState.from_dict(_read_json(scene_file))
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
            for scene_id, meta in _read_json(index_file).items():
                meta["scene_id"] = scene_id
                meta["project"] = index_file.stem.replace(".index_", "")
                scenes.append(meta)
        
        # Sort by timestamp desc
        scenes.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        return scenes[:limit]
    
    def delete_scene(self, scene_id: str) -> bool:
        deleted = False
        for index_file in self.scenes_dir.glob(".index_*.json"):
            index = _read_json(index_file)
            if scene_id in index:
                del index[scene_id]
                index_file.write_text(json.dumps(index, indent=2))
                deleted = True
        
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
            data = _read_json(history_file)
            try:
                ctx = PromptContext(**data)
            except TypeError as e:
                raise AiDbStorageError(f"corrupt prompt history file {history_file}: {e}") from e

            # Filter
            if scene_id and ctx.scene_id != scene_id:
                continue
            if object_uuid and object_uuid not in ctx.objects_affected:
                continue

            history.append(ctx)
        
        history.sort(key=lambda x: x.timestamp, reverse=True)
        return history[:limit]
    
    # --- Text search (no vectors) ---
    
    def search_scenes_by_prompt(
        self, 
        query: str, 
        project: str | None = None, 
        top_k: int = 10
    ) -> list[SceneState]:
        """Prompts ranked by how many query words they contain (same terms as the
        sqlite backend's FTS query; a whole-query substring match found almost nothing)."""
        from ai_db.search.query_builder import expand_terms

        terms = set(expand_terms(query))
        if not terms:
            return []
        best: dict[str, int] = {}
        for ctx in self.get_scene_history(project=project, limit=1000):
            hits = len(terms & set(expand_terms(ctx.prompt)))
            if hits > best.get(ctx.scene_id, 0):
                best[ctx.scene_id] = hits
        results = []
        for scene_id, _hits in sorted(best.items(), key=lambda kv: -kv[1]):
            scene = self.load_scene(scene_id)
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

    # --- Run records ---

    def save_run(self, run: RunRecord) -> str:
        (self.runs_dir / f"{run.run_id}.json").write_text(json.dumps(run.to_dict(), indent=2))
        return run.run_id

    def load_run(self, run_id: str) -> RunRecord | None:
        f = self.runs_dir / f"{run_id}.json"
        if not f.exists():
            return None
        return RunRecord.from_dict(_read_json(f))

    def list_runs(self, project: str | None = None, limit: int = 100) -> list[RunRecord]:
        runs = [r for f in self.runs_dir.glob("*.json") if (r := self.load_run(f.stem))]
        if project:
            runs = [r for r in runs if r.project == project]
        runs.sort(key=lambda r: r.created_at, reverse=True)
        return runs[:limit]
