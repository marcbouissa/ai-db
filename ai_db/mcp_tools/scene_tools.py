"""
MCP Tools for Scene Memory.
============================
Registers with ServiceDispatcher following existing tool patterns.
"""

from typing import Any
from ai_db.dispatcher import ServiceDispatcher
from ai_db.memory.scene import SceneMemory
from ai_db.logger import _logger

# Global scene memory instance (initialized on first use)
_scene_memory: SceneMemory | None = None


def get_scene_memory() -> SceneMemory:
    global _scene_memory
    if _scene_memory is None:
        _scene_memory = SceneMemory()
    return _scene_memory


def register_scene_tools(dispatcher: ServiceDispatcher) -> None:
    """Register scene memory tools with the dispatcher."""
    
    # 1. snapshot_scene
    dispatcher.register_tool(
        name="snapshot_scene",
        description="Capture current Blender scene state to persistent memory with prompt context",
        parameters_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "What the user asked for / description of changes"},
                "project": {"type": "string", "default": "default", "description": "Project name for organization"},
                "operation_type": {"type": "string", "enum": ["create", "modify", "delete", "transform", "material", "fork"], "default": "modify"},
                "objects_affected": {"type": "array", "items": {"type": "string"}, "description": "Object UUIDs that were changed"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization"},
                "metadata": {"type": "object", "description": "Additional metadata"},
            },
            "required": ["prompt"],
        },
        handler=_handle_snapshot_scene,
        category="scene",
    )

    # 2. restore_scene
    dispatcher.register_tool(
        name="restore_scene",
        description="Restore a scene from memory to Blender",
        parameters_schema={
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Scene ID to restore"},
                "mode": {"type": "string", "enum": ["replace", "additive"], "default": "replace", "description": "Replace scene or add to existing"},
            },
            "required": ["scene_id"],
        },
        handler=_handle_restore_scene,
        category="scene",
    )

    # 3. list_scenes
    dispatcher.register_tool(
        name="list_scenes",
        description="List available scene snapshots in memory",
        parameters_schema={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Filter by project"},
                "limit": {"type": "integer", "default": 20, "description": "Max results"},
            },
        },
        handler=_handle_list_scenes,
        category="scene",
    )

    # 4. get_scene_history
    dispatcher.register_tool(
        name="get_scene_history",
        description="Get change history for an object, scene, or project",
        parameters_schema={
            "type": "object",
            "properties": {
                "object_uuid": {"type": "string", "description": "Filter by object UUID"},
                "scene_id": {"type": "string", "description": "Filter by scene ID"},
                "project": {"type": "string", "description": "Filter by project"},
                "limit": {"type": "integer", "default": 20, "description": "Max results"},
            },
        },
        handler=_handle_get_scene_history,
        category="scene",
    )

    # 5. search_scenes_by_prompt
    dispatcher.register_tool(
        name="search_scenes_by_prompt",
        description="Semantic search across scene history by prompt text",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "project": {"type": "string", "description": "Filter by project"},
                "top_k": {"type": "integer", "default": 10, "description": "Max results"},
            },
            "required": ["query"],
        },
        handler=_handle_search_scenes_by_prompt,
        category="scene",
    )

    # 6. search_objects
    dispatcher.register_tool(
        name="search_objects",
        description="Search for objects matching description across all scenes",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (name, type, material)"},
                "project": {"type": "string", "description": "Filter by project"},
                "top_k": {"type": "integer", "default": 20, "description": "Max results"},
            },
            "required": ["query"],
        },
        handler=_handle_search_objects,
        category="scene",
    )

    # 7. fork_scene
    dispatcher.register_tool(
        name="fork_scene",
        description="Create a new scene version branching from an existing one",
        parameters_schema={
            "type": "object",
            "properties": {
                "from_scene_id": {"type": "string", "description": "Scene ID to fork from"},
                "new_prompt": {"type": "string", "description": "Prompt describing the fork"},
                "project": {"type": "string", "description": "Project for new scene (default: same as parent)"},
            },
            "required": ["from_scene_id", "new_prompt"],
        },
        handler=_handle_fork_scene,
        category="scene",
    )

    # 8. get_scene_versions
    dispatcher.register_tool(
        name="get_scene_versions",
        description="Get version history for a scene",
        parameters_schema={
            "type": "object",
            "properties": {
                "scene_id": {"type": "string", "description": "Scene ID"},
            },
            "required": ["scene_id"],
        },
        handler=_handle_get_scene_versions,
        category="scene",
    )

    _logger.info("Registered scene memory MCP tools")


# =========================================================================
# Handlers
# =========================================================================

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
        "message": f"Scene loaded. Apply to Blender using blender_mcp execute_blender_code with scene data."
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
