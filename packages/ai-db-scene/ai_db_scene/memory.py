"""
Scene Memory - High-level API for 3D scene state.
==================================================
Follows ContextMemory pattern from ai_db.memory.context.
"""

import math
import os
from pathlib import Path
from typing import Any

from ai_db.errors import AiDbError
from ai_db.logger import _logger
from ai_db_scene.factory import SceneMemoryFactory
from ai_db_scene.models import PromptContext, RunRecord, SceneState


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
        self._blender: Any = None  # BlenderClient: geometry write on snapshot, apply on restore
        root = getattr(self.backend, "db_path", None)
        root = Path(root).parent if root else getattr(self.backend, "base_path", None)
        #: <scene_id>.blend per snapshot: the transforms live in the store, the geometry here.
        self.blend_dir: Path | None = Path(root) / "blends" if root else None

    def set_blender_capture(self, capture_func):
        """Set function to capture current Blender scene state."""
        self._blender_capture_func = capture_func

    def set_blender_client(self, client: Any) -> None:
        """Attach a live Blender: snapshots capture state *and* geometry; restores apply."""
        self._blender = client
        self._blender_capture_func = client.extract_scene_state

    def blend_path(self, scene_id: str) -> Path | None:
        return self.blend_dir / f"{scene_id}.blend" if self.blend_dir else None
    
    def snapshot_scene(
        self, 
        prompt: str, 
        project: str = "default",
        operation_type: str = "modify",
        objects_affected: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        parent_scene_id: str | None = None,
    ) -> str:
        """Capture current Blender scene and link to prompt.

        ``parent_scene_id`` records the snapshot as the next version of that scene (after an
        edit), so ``get_scene_versions`` walks back through the edits.
        """
        parent = None
        if parent_scene_id is not None:
            parent = self.backend.load_scene(parent_scene_id)
            if parent is None:
                raise AiDbError(f"parent scene not found: {parent_scene_id}")
        if not self._blender_capture_func:
            raise RuntimeError("No Blender capture function set. Call set_blender_capture() first.")
        
        # Capture scene from Blender
        scene = self._blender_capture_func()
        scene.project = project
        scene.tags = tags or []
        if parent is not None:
            if parent.scene_id == scene.scene_id:
                raise AiDbError(f"scene {scene.scene_id} cannot be its own parent")
            scene.parent_scene_id = parent.scene_id
            scene.version = parent.version + 1
            scene.name = parent.name
        
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
        
        # Geometry first: a stored snapshot must be restorable, so a failed write fails
        # the snapshot instead of saving transforms that can never be applied.
        path = self.blend_path(scene.scene_id)
        if self._blender is not None and path is not None:
            written = self._blender.write_blend(str(path))
            if written != len(scene.objects):
                raise AiDbError(f"snapshot wrote {written} objects but captured {len(scene.objects)}")

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
    
    def apply_scene(self, scene_id: str, mode: str = "replace", tol: float = 1e-3) -> dict[str, Any]:
        """Restore a snapshot into the live Blender scene and verify it.

        ``replace`` empties the scene first; ``additive`` adds the snapshot's objects. Forks
        carry no geometry of their own: the nearest ancestor's is used (the transforms
        of the fork itself are what gets verified). Returns what was restored and every
        object whose transform differs from the snapshot by more than ``tol``.
        """
        if mode not in ("replace", "additive"):
            raise AiDbError(f"mode must be 'replace' or 'additive', got {mode!r}")
        if self._blender is None:
            raise AiDbError("Blender is not connected: start Blender with the ai-db addon server")
        scene = self.restore_scene(scene_id)
        path = self._geometry_path(scene)
        if path is None:
            raise AiDbError(f"snapshot {scene_id} has no stored geometry (taken before geometry "
                            f"was saved, or without Blender); take a new snapshot")
        camera = scene.camera.name if scene.camera else None
        names = self._blender.load_blend(str(path), replace=mode == "replace", camera=camera)
        live = self._blender.extract_scene_state()
        by_name = {o.name: o for o in live.objects.values()}
        mismatches: list[dict[str, Any]] = []
        for want in scene.objects.values():
            got = by_name.get(want.name)
            if got is None:
                mismatches.append({"name": want.name, "problem": "missing"})
                continue
            for field_name in ("location", "rotation", "scale"):
                a, b = getattr(want, field_name), getattr(got, field_name)
                # rotation is stored in degrees, rounded to 0.01
                limit = 0.02 if field_name == "rotation" else tol
                if any(not math.isclose(x, y, abs_tol=limit) for x, y in zip(a, b)):
                    mismatches.append({"name": want.name, "problem": field_name,
                                       "stored": list(a), "live": list(b)})
        _logger.info("Scene %s restored into Blender (%s): %d objects, %d mismatches",
                     scene_id, mode, len(names), len(mismatches))
        return {"scene_id": scene_id, "mode": mode, "restored": len(names),
                "geometry_from": path.stem, "mismatches": mismatches}

    def _load(self, scene_id: str) -> SceneState:
        scene = self.backend.load_scene(scene_id)
        if scene is None:
            raise AiDbError(f"scene not found: {scene_id}")
        return scene

    def _geometry_path(self, scene: SceneState) -> Path | None:
        """The .blend holding ``scene``'s geometry: its own, else the nearest ancestor's."""
        cur: SceneState | None = scene
        visited: set[str] = set()
        while cur is not None and cur.scene_id not in visited:
            visited.add(cur.scene_id)
            candidate = self.blend_path(cur.scene_id)
            if candidate is not None and candidate.exists():
                return candidate
            cur = self.backend.load_scene(cur.parent_scene_id) if cur.parent_scene_id else None
        return None

    def diff_scenes(self, scene_a: str, scene_b: str, tol: float = 1e-3) -> dict[str, Any]:
        """What changed from snapshot ``scene_a`` to ``scene_b``, objects matched by name.

        ``moved`` lists every transform field that differs by more than ``tol`` (rotation:
        0.02 degrees, its stored precision); ``rematerialed`` objects whose material list
        changed; ``reshaped`` objects whose vertex/face counts changed.
        """
        a, b = self._load(scene_a), self._load(scene_b)
        objs_a = {o.name: o for o in a.objects.values()}
        objs_b = {o.name: o for o in b.objects.values()}
        moved: list[dict[str, Any]] = []
        rematerialed: list[dict[str, Any]] = []
        reshaped: list[dict[str, Any]] = []
        for name in sorted(objs_a.keys() & objs_b.keys()):
            oa, ob = objs_a[name], objs_b[name]
            changes = {}
            for field_name in ("location", "rotation", "scale"):
                va, vb = getattr(oa, field_name), getattr(ob, field_name)
                limit = 0.02 if field_name == "rotation" else tol
                if any(not math.isclose(x, y, abs_tol=limit) for x, y in zip(va, vb)):
                    changes[field_name] = {"from": list(va), "to": list(vb)}
            if changes:
                moved.append({"name": name, **changes})
            if list(oa.materials) != list(ob.materials):
                rematerialed.append({"name": name, "from": list(oa.materials), "to": list(ob.materials)})
            if (oa.vertices, oa.faces) != (ob.vertices, ob.faces):
                reshaped.append({"name": name, "from": [oa.vertices, oa.faces],
                                 "to": [ob.vertices, ob.faces]})
        added = sorted(objs_b.keys() - objs_a.keys())
        removed = sorted(objs_a.keys() - objs_b.keys())
        return {"scene_a": scene_a, "scene_b": scene_b, "added": added, "removed": removed,
                "moved": moved, "rematerialed": rematerialed, "reshaped": reshaped,
                "unchanged": not (added or removed or moved or rematerialed or reshaped)}

    def restore_object(self, scene_id: str, name: str, tol: float = 1e-3) -> dict[str, Any]:
        """Put one object back as snapshot ``scene_id`` stored it; the rest of the scene is untouched.

        With stored geometry (the snapshot's .blend or an ancestor's) the live object is
        replaced by the stored one (mesh, materials, modifiers); without it only the transform
        is reset, which needs the object to still exist. The transform always comes from the
        snapshot itself and is verified against the live scene afterwards.
        """
        if self._blender is None:
            raise AiDbError("Blender is not connected: start Blender with the ai-db addon server")
        scene = self._load(scene_id)
        want = next((o for o in scene.objects.values() if o.name == name), None)
        if want is None:
            raise AiDbError(f"object {name!r} is not in snapshot {scene_id}")
        path = self._geometry_path(scene)
        self._blender.restore_object(
            str(path) if path is not None else None, name, list(want.location),
            list(want.rotation), list(want.scale),
        )
        live = self._blender.extract_scene_state()
        got = next((o for o in live.objects.values() if o.name == name), None)
        mismatches: list[dict[str, Any]] = []
        if got is None:
            mismatches.append({"name": name, "problem": "missing"})
        else:
            for field_name in ("location", "rotation", "scale"):
                a, b = getattr(want, field_name), getattr(got, field_name)
                limit = 0.02 if field_name == "rotation" else tol
                if any(not math.isclose(x, y, abs_tol=limit) for x, y in zip(a, b)):
                    mismatches.append({"name": name, "problem": field_name,
                                       "stored": list(a), "live": list(b)})
        _logger.info("Object %s restored from %s (%s)", name, scene_id,
                     "geometry" if path is not None else "transform only")
        return {"scene_id": scene_id, "name": name,
                "geometry_from": path.stem if path is not None else None,
                "mismatches": mismatches}

    def list_scenes(self, project: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """List available scene snapshots."""
        return self.backend.list_scenes(project=project, limit=limit)
    
    def delete_scene(self, scene_id: str) -> bool:
        """Delete a scene snapshot (and its stored geometry)."""
        path = self.blend_path(scene_id)
        if path is not None and path.exists():
            os.remove(path)
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
        seen: set[str] = set()
        current = self.backend.load_scene(scene_id)
        while current and current.scene_id not in seen:  # a corrupt cycle must not hang
            seen.add(current.scene_id)
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

    # --- Run records / exemplars (pipeline Stage 7) ---

    def record_run(self, run: RunRecord) -> str:
        return self.backend.save_run(run)

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.backend.load_run(run_id)

    def search_exemplars(
        self, brief: str, min_score: float = 4.0, top_k: int = 3, project: str | None = None
    ) -> list[RunRecord]:
        return self.backend.search_exemplars(brief, min_score=min_score, top_k=top_k, project=project)
