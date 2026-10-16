"""MCP tools for 3D scenes: scene memory + thin forwards to the Blender addon socket.

Registration is table-driven: (name, description, schema) rows plus a handler.
Spatial / build / render tools are forwarded verbatim to the Blender client.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ai_db.dispatcher import ServiceDispatcher
from ai_db.logger import _logger
from ai_db_scene.blender_client import BlenderClient, BlenderError, try_get_blender_client

if TYPE_CHECKING:
    from ai_db_scene.memory import SceneMemory

_scene_memory: SceneMemory | None = None


def get_scene_memory() -> SceneMemory:
    global _scene_memory
    if _scene_memory is None:
        # Imported here: the scene store pulls in sqlite_vec -> numpy (~120 ms), which
        # every ai-db command would otherwise pay just to register these tools.
        from ai_db_scene.memory import SceneMemory

        _scene_memory = SceneMemory()
    if _scene_memory._blender is None:
        # Checked on every call, not only at creation: the MCP server usually starts before
        # Blender, and a memory created then could never snapshot or restore afterwards.
        client = try_get_blender_client()
        if client is not None:
            _scene_memory.set_blender_client(client)
            _logger.info("[ai-db] Connected 3D scene memory to running Blender instance")
    return _scene_memory


# --- Blender forwarding -------------------------------------------------------

# Tool name -> (blender handler name, params to drop). Default: same name, no drops.
_FORWARD_OVERRIDES: dict[str, tuple[str, tuple[str, ...]]] = {
    "execute_blender_code": ("run_code", ("user_prompt",)),
}


def _forward(tool: str, args: dict[str, Any]) -> Any:
    """Forward a tool call to the Blender socket; structured error dict on failure."""
    op, drop = _FORWARD_OVERRIDES.get(tool, (tool, ()))
    params = {k: v for k, v in args.items() if k not in drop}
    try:
        return {"status": "success", "result": BlenderClient().request(op, params)}
    except BlenderError as e:
        return {
            "status": "error",
            "message": str(e),
            "error_type": e.error_type or type(e).__name__,
            "hint": e.hint,
        }


def _validate(args: dict[str, Any]) -> Any:
    from ai_db_scene.tools.scene_pipeline import _handle_validate_scene

    return _handle_validate_scene(args)


def _make_forward(tool: str) -> Any:
    return lambda args: _forward(tool, args)


def _handle_snapshot_scene(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    scene_id = memory.snapshot_scene(
        prompt=args["prompt"],
        project=args.get("project"),
        operation_type=args.get("operation_type", "modify"),
        objects_affected=args.get("objects_affected"),
        metadata=args.get("metadata"),
        tags=args.get("tags"),
        parent_scene_id=args.get("parent_scene_id"),
        name=args.get("name"),
    )
    return {"scene_id": scene_id, "status": "saved"}


def _handle_restore_scene(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    report = memory.apply_scene(args["scene_id"], args.get("mode", "replace"))
    status = ("restored_with_mismatches" if report["mismatches"]
              else "restored_with_missing_files" if report.get("missing_files") else "restored")
    return {"status": status, **report}


def _handle_diff_scenes(args: dict[str, Any]) -> Any:
    return get_scene_memory().diff_scenes(args["scene_a"], args["scene_b"])


def _handle_restore_object(args: dict[str, Any]) -> Any:
    report = get_scene_memory().restore_object(args["scene_id"], args["name"])
    return {"status": "restored" if not report["mismatches"] else "restored_with_mismatches", **report}


def _handle_scene_snapshot(args: dict[str, Any]) -> Any:
    """Live scene objects (read-only)."""
    client = get_scene_memory()._blender
    if client is None:
        return {"error": "Blender is not connected"}
    objects = client.scene_snapshot()
    return {"count": len(objects), "objects": objects}


def _handle_list_scenes(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    scenes = memory.list_scenes(project=args.get("project"), limit=args.get("limit", 20))
    return {"scenes": scenes, "count": len(scenes)}


def _handle_get_scene_history(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    history = memory.get_history(
        object_uuid=args.get("object_uuid"),
        scene_id=args.get("scene_id"),
        project=args.get("project"),
        limit=args.get("limit", 20),
    )
    return {
        "history": [
            {
                "id": h.id,
                "scene_id": h.scene_id,
                "prompt": h.prompt,
                "timestamp": h.timestamp,
                "operation_type": h.operation_type,
                "objects_affected": h.objects_affected,
            }
            for h in history
        ],
        "count": len(history),
    }


def _handle_search_scenes_by_prompt(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    scenes = memory.search_by_prompt(
        query=args["query"],
        project=args.get("project"),
        top_k=args.get("top_k", 10),
    )
    return {
        "scenes": [
            {
                "scene_id": s.scene_id,
                "name": s.name,
                "project": s.project,
                "timestamp": s.timestamp,
                "object_count": len(s.objects),
                "prompt": s.prompt_context.prompt if s.prompt_context else "",
            }
            for s in scenes
        ],
        "count": len(scenes),
    }


def _handle_search_objects(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    results = memory.search_objects(
        query=args["query"],
        project=args.get("project"),
        top_k=args.get("top_k", 20),
    )
    return {"objects": results, "count": len(results)}


def _handle_fork_scene(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    new_scene_id = memory.fork_scene(
        from_scene_id=args["from_scene_id"],
        new_prompt=args["new_prompt"],
        project=args.get("project"),
    )
    return {"scene_id": new_scene_id, "status": "forked"}


def _handle_get_scene_versions(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    versions = memory.get_scene_versions(args["scene_id"])
    return {
        "versions": [
            {
                "scene_id": v.scene_id,
                "name": v.name,
                "version": v.version,
                "timestamp": v.timestamp,
                "parent_scene_id": v.parent_scene_id,
            }
            for v in versions
        ],
        "count": len(versions),
    }


def _handle_get_blender_libraries(args: dict[str, Any]) -> Any:
    from ai_db_scene.blender_client import try_get_blender_client

    client = try_get_blender_client()
    if not client:
        return {
            "status": "error",
            "message": "Blender is not running or its socket server is not started.",
            "linked_libraries": [],
            "asset_libraries": [],
            "local_assets": [],
        }
    return {
        "status": "success",
        **client.get_available_libraries(),
    }


_MEMORY_TOOLS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "snapshot_scene",
        "Capture current Blender scene state to persistent memory with prompt context",
        {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "What the user asked for / description of changes",
                },
                "project": {
                    "type": "string",
                    "description": "Project name for organization (default: the parent's, else 'default')",
                },
                "name": {
                    "type": "string",
                    "description": "Scene name shown by list_scenes (default: the parent's, else from the prompt)",
                },
                "operation_type": {
                    "type": "string",
                    "enum": ["create", "modify", "delete", "transform", "material", "fork"],
                    "default": "modify",
                },
                "objects_affected": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Object UUIDs that were changed",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for categorization",
                },
                "metadata": {"type": "object", "description": "Additional metadata"},
                "parent_scene_id": {"type": "string",
                                    "description": "Save as the next version of this scene (after an edit)"},
            },
            "required": ["prompt"],
        },
    ),
    (
        "restore_scene",
        "Restore a snapshot into Blender (geometry, materials, transforms) and verify every object's transform",
        {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Scene ID to restore"},
                "mode": {
                    "type": "string",
                    "enum": ["replace", "additive"],
                    "default": "replace",
                    "description": "Replace scene or add to existing",
                },
            },
            "required": ["scene_id"],
        },
    ),
    (
        "diff_scenes",
        ("Compare two snapshots: objects added, removed, moved (location/rotation/scale), "
         "re-materialed or reshaped from scene_a to scene_b (matched by object name)"),
        {
            "type": "object",
            "properties": {
                "scene_a": {"type": "string", "description": "Earlier scene ID"},
                "scene_b": {"type": "string", "description": "Later scene ID"},
            },
            "required": ["scene_a", "scene_b"],
        },
    ),
    (
        "restore_object",
        ("Restore one object as a snapshot stored it (geometry from the snapshot's .blend when "
         "present, else transform only) without touching the rest of the scene"),
        {
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Snapshot to restore from"},
                "name": {"type": "string", "description": "Object name"},
            },
            "required": ["scene_id", "name"],
        },
    ),
    (
        "list_scenes",
        "List available scene snapshots in memory",
        {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Filter by project"},
                "limit": {"type": "integer", "default": 20, "description": "Max results"},
            },
        },
    ),
    (
        "get_scene_history",
        "Get change history for an object, scene, or project",
        {
            "type": "object",
            "properties": {
                "object_uuid": {"type": "string", "description": "Filter by object UUID"},
                "scene_id": {"type": "string", "description": "Filter by scene ID"},
                "project": {"type": "string", "description": "Filter by project"},
                "limit": {"type": "integer", "default": 20, "description": "Max results"},
            },
        },
    ),
    (
        "search_scenes_by_prompt",
        "Semantic search across scene history by prompt text",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "project": {"type": "string", "description": "Filter by project"},
                "top_k": {"type": "integer", "default": 10, "description": "Max results"},
            },
            "required": ["query"],
        },
    ),
    (
        "search_objects",
        "Search for objects matching description across all scenes",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (name, type, material)"},
                "project": {"type": "string", "description": "Filter by project"},
                "top_k": {"type": "integer", "default": 20, "description": "Max results"},
            },
            "required": ["query"],
        },
    ),
    (
        "fork_scene",
        "Create a new scene version branching from an existing one",
        {
            "type": "object",
            "properties": {
                "from_scene_id": {"type": "string", "description": "Scene ID to fork from"},
                "new_prompt": {"type": "string", "description": "Prompt describing the fork"},
                "project": {
                    "type": "string",
                    "description": "Project for new scene (default: same as parent)",
                },
            },
            "required": ["from_scene_id", "new_prompt"],
        },
    ),
    (
        "get_scene_versions",
        "Get version history for a scene",
        {
            "type": "object",
            "properties": {"scene_id": {"type": "string", "description": "Scene ID"}},
            "required": ["scene_id"],
        },
    ),
    (
        "get_blender_libraries",
        "Query Blender for linked libraries (.blend files), asset libraries, and local asset catalog",
        {"type": "object", "properties": {}},
    ),
]

_FORWARDED_TOOLS: list[tuple[str, str, dict[str, Any]]] = [
    (
        "execute_blender_code",
        "Execute arbitrary Python code inside Blender on the main thread via ITC",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute in Blender"},
                "user_prompt": {
                    "type": "string",
                    "default": "",
                    "description": "Optional context or description",
                },
            },
            "required": ["code"],
        },
    ),
    (
        "create_primitive",
        "Create a basic primitive object in Blender (cube, sphere, ico, cylinder, cone, torus, plane, circle) For a whole room or scene use build_scene (layout, real furniture, materials, light, checks) instead; this is for one-off tweaks.",
        {
            "type": "object",
            "properties": {
                "primitive": {
                    "type": "string",
                    "enum": [
                        "cube",
                        "sphere",
                        "ico",
                        "cylinder",
                        "cone",
                        "torus",
                        "plane",
                        "circle",
                        "monkey",
                    ],
                    "description": "Primitive shape",
                },
                "name": {"type": "string", "description": "Name for the created object"},
                "location": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] coordinate",
                },
                "scale": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[sx, sy, sz] scaling",
                },
            },
            "required": ["primitive", "name"],
        },
    ),
    (
        "modify_object",
        "Modify the transform (location, rotation in degrees, scale) of an object in Blender by name",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Object name"},
                "location": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z]",
                },
                "rotation": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[rx, ry, rz] in degrees",
                },
                "scale": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[sx, sy, sz]",
                },
            },
            "required": ["name"],
        },
    ),
    (
        "delete_object",
        "Delete a 3D object from Blender by name",
        {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Object name to delete"}},
            "required": ["name"],
        },
    ),
    (
        "set_material",
        "Create or assign a PBR material to an object in Blender",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Target object name"},
                "material_name": {"type": "string", "description": "Name for the material"},
                "color": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[r, g, b, a] color",
                },
                "roughness": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Roughness 0.0 - 1.0",
                },
                "metallic": {"type": "number", "default": 0.0, "description": "Metallic 0.0 - 1.0"},
                "emission": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Emission [r, g, b, a]",
                },
            },
            "required": ["name"],
        },
    ),
    (
        "add_light",
        "Add a light source to the Blender scene (POINT, SUN, SPOT, AREA)",
        {
            "type": "object",
            "properties": {
                "light_type": {
                    "type": "string",
                    "enum": ["POINT", "SUN", "SPOT", "AREA"],
                    "default": "POINT",
                },
                "name": {"type": "string", "default": "Light"},
                "energy": {
                    "type": "number",
                    "default": 1000.0,
                    "description": "Light power / energy in Watts",
                },
                "color": {"type": "array", "items": {"type": "number"}, "description": "[r, g, b]"},
                "location": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z]",
                },
            },
            "required": ["light_type"],
        },
    ),
    (
        "set_camera",
        "Set Blender camera position, rotation in degrees, and focal length",
        {
            "type": "object",
            "properties": {
                "location": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z]",
                },
                "rotation": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[rx, ry, rz] in degrees",
                },
                "name": {"type": "string", "default": "Camera"},
                "lens": {"type": "number", "default": 50.0, "description": "Focal length in mm"},
            },
            "required": ["location", "rotation"],
        },
    ),
    (
        "clear_scene",
        "Safely clear meshes/curves/lights from the active Blender scene without closing connection",
        {
            "type": "object",
            "properties": {
                "keep_camera": {"type": "boolean", "default": True},
                "purge_orphans": {"type": "boolean", "default": True},
            },
        },
    ),
    (
        "render",
        "Render the current Blender scene to an image file",
        {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Destination image filepath"},
                "resolution": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "[width, height]",
                },
                "samples": {"type": "integer", "description": "Render samples"},
            },
            "required": ["filepath"],
        },
    ),
    (
        "scene_snapshot",
        "List all Blender objects with stable UUIDs, collection, transforms and materials",
        {"type": "object", "properties": {}},
    ),
]


def _obj(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or []}


_XYZ = {"type": "array", "items": {"type": "number"}}


def _vec(desc: str) -> dict[str, Any]:
    return {**_XYZ, "description": desc}


# Schemas mirror blender_mcp/tools.json (guarded by tests/test_scene_forward_schemas.py).
_FORWARDED_TOOLS += [
    (
        "create_complex_asset",
        "Create a complex parametric asset in Blender: rounded_box, pill, staircase, fence, arch, pipe, terrain, rock For a whole room or scene use build_scene (layout, real furniture, materials, light, checks) instead; this is for one-off tweaks.",
        _obj(
            {
                "asset": {
                    "type": "string",
                    "enum": ["rounded_box", "pill", "staircase", "fence", "arch", "pipe", "terrain", "rock"],
                },
                "name": {"type": "string"},
                "location": _vec("[x, y, z]"),
                "scale": _vec("[sx, sy, sz]"),
                "steps": {"type": "integer"},
                "count": {"type": "integer"},
                "radius": {"type": "number"},
                "thickness": {"type": "number"},
                "size": {"type": "number"},
                "height": {"type": "number"},
                "noise_scale": {"type": "number"},
                "strength": {"type": "number"},
            },
            ["asset"],
        ),
    ),
    (
        "get_bounds",
        "Axis-aligned bounds (min, max, center, dimensions) of an object by name",
        _obj({"name": {"type": "string"}}, ["name"]),
    ),
    (
        "get_viewport_screenshot",
        "Render a viewport screenshot; returns image file path",
        _obj(
            {
                "filepath": {"type": "string"},
                "engine": {"type": "string"},
                "keep_file": {"type": "boolean"},
                "resolution": {"type": "array", "items": {"type": "integer"}, "description": "[w, h]"},
            }
        ),
    ),
    (
        "get_object_transform",
        "Get location/rotation/scale/dimensions of an object by name",
        _obj({"name": {"type": "string"}}, ["name"]),
    ),
    (
        "raycast",
        "Cast a ray; returns first hit object, point and normal",
        _obj(
            {
                "origin": _vec("[x, y, z]"),
                "direction": _vec("[dx, dy, dz]"),
                "max_distance": {"type": "number"},
                "ignore": {"type": "array", "items": {"type": "string"}, "description": "Object names to skip"},
            },
            ["origin", "direction"],
        ),
    ),
    (
        "measure_distance",
        "Distance between two objects (mode: surface or center)",
        _obj(
            {"a": {"type": "string"}, "b": {"type": "string"}, "mode": {"type": "string"}},
            ["a", "b"],
        ),
    ),
    (
        "check_collision",
        "Check whether object a overlaps b (or any object if b omitted)",
        _obj(
            {"a": {"type": "string"}, "b": {"type": "string"}, "tolerance": {"type": "number"}},
            ["a"],
        ),
    ),
    (
        "get_camera_view",
        "Objects visible from a camera (default: active camera)",
        _obj({"camera": {"type": "string"}}),
    ),
    (
        "find_objects_in_region",
        "Objects whose bounds intersect an axis-aligned region",
        _obj({"min": _vec("[x, y, z]"), "max": _vec("[x, y, z]")}, ["min", "max"]),
    ),
    (
        "get_spatial_context",
        "Neighbours of an object within a radius, with relative directions",
        _obj({"name": {"type": "string"}, "radius": {"type": "number"}}, ["name"]),
    ),
    (
        "get_scene_info",
        "List all scene objects (name, type, location) and counts",
        _obj({}),
    ),
    (
        "validate_scene",
        (
            "Validate the live Blender scene (interpenetration, floating/sunk, bounds, materials, poly budget) "
            "against an optional SceneSpec; returns issues with fix actions"
        ),
        _obj(
            {
                "spec": {"type": "object", "description": "Optional SceneSpec to check relations/bounds against"},
                "checks": {"type": "array", "items": {"type": "string"}, "description": "Subset of checks to run"},
                "room_bounds": _vec("[x, y, z] room size in meters"),
                "priors": {"type": "object", "description": "Category dimension priors override"},
                "poly_budget": {"type": "object", "description": "Polygon budgets (per-object / total)"},
            }
        ),
    ),
]

_MEMORY_HANDLERS: dict[str, Any] = {
    "snapshot_scene": _handle_snapshot_scene,
    "restore_scene": _handle_restore_scene,
    "diff_scenes": _handle_diff_scenes,
    "restore_object": _handle_restore_object,
    "scene_snapshot": _handle_scene_snapshot,
    "validate_scene": lambda args: _validate(args),
    "list_scenes": _handle_list_scenes,
    "get_scene_history": _handle_get_scene_history,
    "search_scenes_by_prompt": _handle_search_scenes_by_prompt,
    "search_objects": _handle_search_objects,
    "fork_scene": _handle_fork_scene,
    "get_scene_versions": _handle_get_scene_versions,
    "get_blender_libraries": _handle_get_blender_libraries,
}


def scene_tool_table() -> list[tuple[str, str, dict[str, Any], Any]]:
    """All scene tools as (name, description, schema, handler)."""
    rows = [(n, d, s, _MEMORY_HANDLERS[n]) for n, d, s in _MEMORY_TOOLS]
    # A local handler wins over forwarding (scene_snapshot has no addon handler).
    rows += [(n, d, s, _MEMORY_HANDLERS.get(n) or _make_forward(n)) for n, d, s in _FORWARDED_TOOLS]
    from ai_db_scene.tools.scene_pipeline import PIPELINE_TOOLS

    rows += PIPELINE_TOOLS
    return rows


def register_scene_tools(dispatcher: ServiceDispatcher) -> None:
    """Register scene memory + Blender forwarding tools with the dispatcher."""
    for name, description, schema, handler in scene_tool_table():
        dispatcher.register_tool(
            name=name,
            description=description,
            parameters_schema=schema,
            handler=handler,
            category="scene",
        )
