"""
Scene Memory Models.
====================
Domain models for 3D scene state, following ai-db's DTO pattern (ai_db.storage.models).
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def new_scene_id(timestamp: float | None = None) -> str:
    """Id of a new scene version: ``blender-<unix seconds>-<6 hex>``, the same for snapshots,
    forks and edits (forks used to get a bare 8-character uuid). Ids of either older format stay
    valid: an id is only ever looked up, never parsed."""
    return f"blender-{int(time.time() if timestamp is None else timestamp)}-{uuid4().hex[:6]}"


@dataclass
class ObjectState:
    """Complete state of a single Blender object."""
    uuid: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    type: str = "MESH"           # MESH, CURVE, LIGHT, CAMERA, EMPTY, ARMATURE
    collection: str = "Collection"
    
    # Transform
    location: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)  # Euler degrees
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    
    # Geometry
    mesh_data: str | None = None     # Mesh name if shared
    vertices: int = 0
    faces: int = 0
    
    # Materials
    materials: list[str] = field(default_factory=list)  # Material names
    
    # Modifiers
    modifiers: list[dict[str, Any]] = field(default_factory=list)
    
    # Constraints
    constraints: list[dict[str, Any]] = field(default_factory=list)
    
    # Parent/children
    parent_uuid: str | None = None
    children_uuids: list[str] = field(default_factory=list)
    
    # Custom properties
    custom_props: dict[str, Any] = field(default_factory=dict)
    
    # Visibility
    hide_viewport: bool = False
    hide_render: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "type": self.type,
            "collection": self.collection,
            "location": self.location,
            "rotation": self.rotation,
            "scale": self.scale,
            "mesh_data": self.mesh_data,
            "vertices": self.vertices,
            "faces": self.faces,
            "materials": self.materials,
            "modifiers": self.modifiers,
            "constraints": self.constraints,
            "parent_uuid": self.parent_uuid,
            "children_uuids": self.children_uuids,
            "custom_props": self.custom_props,
            "hide_viewport": self.hide_viewport,
            "hide_render": self.hide_render,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectState":
        obj: ObjectState = _build(cls, data)
        return obj


@dataclass
class MaterialState:
    """Complete state of a Blender material."""
    name: str = ""
    type: str = "PRINCIPLED_BSDF"  # or custom node tree
    nodes: list[dict[str, Any]] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)
    custom_props: dict[str, Any] = field(default_factory=dict)


@dataclass
class CameraState:
    """Camera state."""
    name: str = "Camera"
    location: tuple[float, float, float] = (0.0, 0.0, 10.0)
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lens: float = 50.0
    sensor_width: float = 36.0
    clip_start: float = 0.1
    clip_end: float = 1000.0


@dataclass
class CollectionState:
    """Collection (scene hierarchy) state."""
    name: str = ""
    children: list[str] = field(default_factory=list)  # Collection names
    objects: list[str] = field(default_factory=list)   # Object UUIDs
    hide_viewport: bool = False
    hide_render: bool = False


@dataclass
class PromptContext:
    """Links a user prompt to the scene changes it caused."""
    id: int = 0
    prompt: str = ""
    project: str = "default"
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    scene_id: str = ""
    objects_affected: list[str] = field(default_factory=list)  # Object UUIDs
    operation_type: str = "modify"  # create, modify, delete, transform, material
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneState:
    """Complete snapshot of a Blender scene at a point in time."""
    scene_id: str = field(default_factory=new_scene_id)
    project: str = "default"
    name: str = "Scene"
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    version: int = 1
    
    objects: dict[str, ObjectState] = field(default_factory=dict)  # uuid -> ObjectState
    materials: dict[str, MaterialState] = field(default_factory=dict)
    collections: dict[str, CollectionState] = field(default_factory=dict)
    camera: CameraState = field(default_factory=CameraState)
    
    # Prompt that led to this state
    prompt_context: PromptContext | None = None
    
    # Parent scene (for versioning)
    parent_scene_id: str | None = None
    
    # Tags for organization
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "project": self.project,
            "name": self.name,
            "timestamp": self.timestamp,
            "version": self.version,
            "objects": {k: v.to_dict() for k, v in self.objects.items()},
            "materials": {k: v.__dict__ for k, v in self.materials.items()},
            "collections": {k: v.__dict__ for k, v in self.collections.items()},
            "camera": self.camera.__dict__,
            "prompt_context": self.prompt_context.__dict__ if self.prompt_context else None,
            "parent_scene_id": self.parent_scene_id,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneState":
        # Rows written by older (or newer) versions of the plugin must still load: missing
        # keys take their defaults, unknown keys are ignored (a stored snapshot that cannot
        # be read can never be restored).
        scene: SceneState = _build(cls, {k: v for k, v in data.items()
                             if k not in ("objects", "materials", "collections", "camera", "prompt_context")})
        scene.objects = {k: ObjectState.from_dict(v) for k, v in (data.get("objects") or {}).items()}
        scene.materials = {k: _build(MaterialState, v) for k, v in (data.get("materials") or {}).items()}
        scene.collections = {k: _build(CollectionState, v) for k, v in (data.get("collections") or {}).items()}
        scene.camera = _build(CameraState, data.get("camera") or {})
        if data.get("prompt_context"):
            scene.prompt_context = _build(PromptContext, data["prompt_context"])
        return scene


def _build(cls: Any, data: dict[str, Any]) -> Any:
    """``cls(**data)`` without the keys ``cls`` does not know (stored by another version)."""
    known = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in data.items() if k in known})


RUN_OUTCOMES = ("success", "failed", "partial", "aborted")


@dataclass
class RunRecord:
    """One end-to-end pipeline run (brief to judged scene); source of exemplars."""

    brief: str = ""
    spec: dict[str, Any] = field(default_factory=dict)
    placements: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)  # recipe/asset/free-code + params
    validation_reports: list[Any] = field(default_factory=list)  # one per fix iteration
    judge_scores: list[dict[str, Any]] = field(default_factory=list)  # JudgeResult.to_dict() per call
    token_cost: float = 0.0
    wall_time_s: float = 0.0
    outcome: str = "success"
    created_at: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    project: str = "default"
    run_id: str = field(default_factory=lambda: str(uuid4())[:12])

    @property
    def score(self) -> float | None:
        """Overall score of the last judge call (None if never judged)."""
        for js in reversed(self.judge_scores):
            v = js.get("overall") if isinstance(js, dict) else None
            if isinstance(v, (int, float)):
                return float(v)
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "brief": self.brief,
            "spec": self.spec,
            "placements": self.placements,
            "actions": self.actions,
            "validation_reports": self.validation_reports,
            "judge_scores": self.judge_scores,
            "token_cost": self.token_cost,
            "wall_time_s": self.wall_time_s,
            "outcome": self.outcome,
            "created_at": self.created_at,
            "project": self.project,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
