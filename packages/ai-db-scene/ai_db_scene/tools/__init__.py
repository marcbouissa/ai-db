"""MCP tools of the scene plugin, registered through the ``ai_db.tools`` entry point."""

from typing import Any


def register(dispatcher: Any) -> None:
    """Entry point: add scene memory, scene pipeline and video tools to ``dispatcher``.

    Cheap by design (no torch, no Blender connection): video processors load on the
    first video tool call and the Blender client is resolved per call.
    """
    from ai_db_scene.tools.feedback_tools import register_feedback_tools
    from ai_db_scene.tools.recipe_tools import register_recipe_tools
    from ai_db_scene.tools.scene_tools import register_scene_tools
    from ai_db_scene.tools.video_tools import register_video_tools

    register_scene_tools(dispatcher)
    register_video_tools(dispatcher)
    register_recipe_tools(dispatcher)
    register_feedback_tools(dispatcher)
