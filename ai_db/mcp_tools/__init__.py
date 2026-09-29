"""
MCP Tools Package.
===================
Registers all MCP tools with the ServiceDispatcher.
"""

from ai_db.mcp_tools.scene_tools import register_scene_tools
from ai_db.mcp_tools.video_tools import register_video_tools

__all__ = ["register_scene_tools", "register_video_tools"]
