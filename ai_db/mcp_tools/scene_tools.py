"""MCP tools for 3D scenes: scene memory + thin forwards to the Blender (blender_ox) socket.

Registration is table-driven: (name, description, schema) rows plus a handler.
Spatial / build / render tools are forwarded verbatim to the Blender client.
"""

from __future__ import annotations

from typing import Any

from ai_db.dispatcher import ServiceDispatcher
from ai_db.ext.blender_client import BlenderClient, BlenderError, try_get_blender_client
from ai_db.logger import _logger
from ai_db.memory.scene import SceneMemory

_scene_memory: SceneMemory | None = None


def get_scene_memory() -> SceneMemory:
    global _scene_memory
    if _scene_memory is None:
        _scene_memory = SceneMemory()
        client = try_get_blender_client()
        if client is not None:
            _scene_memory.set_blender_capture(client.extract_scene_state)
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


def _make_forward(tool: str) -> Any:
    return lambda args: _forward(tool, args)


def _handle_snapshot_scene(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    scene_id = memory.snapshot_scene(
        prompt=args["prompt"],
        project=args.get("project", "default"),
        operation_type=args.get("operation_type", "modify"),
        objects_affected=args.get("objects_affected"),
        metadata=args.get("metadata"),
        tags=args.get("tags"),
    )
    return {"scene_id": scene_id, "status": "saved"}


def _handle_restore_scene(args: dict[str, Any]) -> Any:
    memory = get_scene_memory()
    scene = memory.restore_scene(args["scene_id"], args.get("mode", "replace"))
    return {
        "scene_id": scene.scene_id,
        "name": scene.name,
        "object_count": len(scene.objects),
        "status": "loaded",
        "message": "Scene loaded. Apply to Blender using blender_mcp execute_blender_code with scene data.",
    }


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
    from ai_db.ext.blender_client import try_get_blender_client

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
                    "default": "default",
                    "description": "Project name for organization",
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
            },
            "required": ["prompt"],
        },
    ),
    (
        "restore_scene",
        "Restore a scene from memory to Blender",
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
        "Create a basic primitive object in Blender (cube, sphere, ico, cylinder, cone, torus, plane, circle)",
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
        "create_complex_asset",
        "Create a complex parametric asset in Blender: rounded_box, pill, staircase, fence, arch, pipe, terrain, rock",
        {
            "type": "object",
            "properties": {
                "asset": {
                    "type": "string",
                    "enum": [
                        "rounded_box",
                        "pill",
                        "staircase",
                        "fence",
                        "arch",
                        "pipe",
                        "terrain",
                        "rock",
                    ],
                    "description": "Complex asset type",
                },
                "name": {"type": "string", "description": "Object name"},
                "location": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z]",
                },
                "scale": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[sx, sy, sz]",
                },
            },
            "required": ["asset"],
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
        "get_viewport_screenshot",
        "Capture a fast render or viewport screenshot from Blender",
        {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Optional destination filepath"}
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
        "get_object_transform",
        "Get full world-space transform (location, rotation, scale), bounding box, and dimensions of an object by name or UUID",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Object name"},
                "uuid": {"type": "string", "description": "Object UUID (alternative to name)"},
                "include_children": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include child objects in bounds calculation",
                },
            },
            "required": [],
        },
    ),
    (
        "raycast",
        "Cast a ray from a point in a direction and return the first object hit with distance and hit location",
        {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] ray origin",
                },
                "direction": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] ray direction (normalized)",
                },
                "distance": {"type": "number", "default": 100.0, "description": "Max ray distance"},
                "exclude_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Object names to ignore",
                },
            },
            "required": ["origin", "direction"],
        },
    ),
    (
        "measure_distance",
        "Measure distance between two objects, an object and a point, or two points in world space",
        {
            "type": "object",
            "properties": {
                "from_name": {"type": "string", "description": "Source object name"},
                "from_uuid": {"type": "string", "description": "Source object UUID"},
                "from_point": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Source point [x, y, z]",
                },
                "to_name": {"type": "string", "description": "Target object name"},
                "to_uuid": {"type": "string", "description": "Target object UUID"},
                "to_point": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Target point [x, y, z]",
                },
            },
            "required": [],
        },
    ),
    (
        "check_collision",
        "Check if two objects intersect (bounding box or precise mesh collision)",
        {
            "type": "object",
            "properties": {
                "name_a": {"type": "string", "description": "First object name"},
                "uuid_a": {"type": "string", "description": "First object UUID"},
                "name_b": {"type": "string", "description": "Second object name"},
                "uuid_b": {"type": "string", "description": "Second object UUID"},
                "precise": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use precise mesh collision (slower)",
                },
            },
            "required": [],
        },
    ),
    (
        "get_camera_view",
        "Get objects visible from the active camera (or specified camera) with screen-space bounds",
        {
            "type": "object",
            "properties": {
                "camera_name": {
                    "type": "string",
                    "description": "Camera object name (default: active scene camera)",
                },
                "frustum_only": {
                    "type": "boolean",
                    "default": True,
                    "description": "Only return objects in camera frustum",
                },
                "max_distance": {
                    "type": "number",
                    "default": 100.0,
                    "description": "Max distance from camera",
                },
            },
        },
    ),
    (
        "get_bounds",
        "Get world-space bounding box of one or more objects (min/max corners, center, dimensions)",
        {
            "type": "object",
            "properties": {
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Object names",
                },
                "uuids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Object UUIDs",
                },
                "include_children": {"type": "boolean", "default": True},
            },
        },
    ),
    (
        "find_objects_in_region",
        "Find all objects within a spatial region (box, sphere, or cylinder)",
        {
            "type": "object",
            "properties": {
                "region_type": {
                    "type": "string",
                    "enum": ["box", "sphere", "cylinder"],
                    "default": "box",
                },
                "center": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] center",
                },
                "size": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] half-extents for box, or [radius, height] for "
                    "cylinder/sphere",
                },
                "rotation": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[rx, ry, rz] degrees for box/cylinder",
                },
            },
            "required": ["center", "size"],
        },
    ),
    (
        "get_spatial_context",
        "Natural language spatial query: 'what is in front of the camera', 'what is near the desk', 'what is to the left of the chair'",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Spatial query in natural language"},
                "reference_object": {
                    "type": "string",
                    "description": "Reference object name for relative queries",
                },
                "camera_relative": {
                    "type": "boolean",
                    "default": False,
                    "description": "Interpret directions relative to active camera",
                },
            },
            "required": ["query"],
        },
    ),
    (
        "get_scene_info",
        "Get a summary of the current Blender scene: object counts, bounds, camera, lights, materials",
        {
            "type": "object",
            "properties": {
                "detailed": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include per-object details",
                }
            },
        },
    ),
    (
        "validate_scene",
        "Validate the live Blender scene (interpenetration, floating/sunk, bounds, materials, poly budget) against an optional SceneSpec; returns issues with fix actions",
        {
            "type": "object",
            "properties": {
                "spec": {
                    "type": "object",
                    "description": "Optional SceneSpec to check relations/bounds against",
                }
            },
        },
    ),
    (
        "scene_snapshot",
        "List all Blender objects with stable UUIDs, locations, dimensions and materials",
        {"type": "object", "properties": {}},
    ),
]

_MEMORY_HANDLERS: dict[str, Any] = {
    "snapshot_scene": _handle_snapshot_scene,
    "restore_scene": _handle_restore_scene,
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
    rows += [(n, d, s, _make_forward(n)) for n, d, s in _FORWARDED_TOOLS]
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
