"""ai-db-scene: 3D scene and video tools for ai-db, shipped as a separate plugin.

Installs MCP tools (scene memory, the spec -> layout -> Blender pipeline, video
extraction) into every ai-db ServiceDispatcher through the ``ai_db.tools`` entry point.
Settings live in the ``plugins.scene`` section of the ai-db config.
"""


def skills_dir() -> str:
    """Directory of the Blender skills shipped with this package (``ai_db.skills`` entry point)."""
    import os

    return os.path.join(os.path.dirname(__file__), "skills")
