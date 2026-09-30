"""
Blender ox-bridge capture function for ai-db scene memory.

Run this instead of `ai-db mcp` to start the MCP server with Blender integration:

    python -m ai_db.blender_oxbridge_capture

This connects to ox-bridge (blender_mcp) on localhost:9878 (MCP streamable HTTP)
and injects a capture function that fetches the current Blender scene via
the `get_scene_info` tool.
"""

import asyncio
import json
import os
import sys
from typing import Any

import httpx

# Add the project root to path so ai_db imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_db.memory.scene import SceneMemory
from ai_db.mcp_tools import scene_tools


class OxBridgeMCPClient:
    """Minimal MCP client to talk to ox-bridge (blender_mcp) on port 9878."""

    def __init__(self, url: str = "http://localhost:9878/"):
        # Ensure trailing slash for streamable HTTP
        if not url.endswith("/"):
            url += "/"
        self.url = url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.request_id = 0

    async def initialize(self) -> bool:
        """Initialize the MCP connection."""
        self.request_id += 1
        resp = await self.client.post(
            self.url,
            json={
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ai-db-blender-capture", "version": "0.1.0"},
                },
            },
        )
        return resp.status_code == 200

    async def list_tools(self) -> list[dict]:
        """List available tools on ox-bridge."""
        self.request_id += 1
        resp = await self.client.post(
            self.url,
            json={
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "tools/list",
                "params": {},
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("result", {}).get("tools", [])
        return []

    async def call_tool(self, name: str, arguments: dict) -> dict | None:
        """Call a tool on ox-bridge."""
        self.request_id += 1
        resp = await self.client.post(
            self.url,
            json={
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        if resp.status_code == 200:
            data = resp.json()
            if "result" in data:
                content = data["result"].get("content", [])
                if content and content[0].get("type") == "text":
                    try:
                        return json.loads(content[0]["text"])
                    except json.JSONDecodeError:
                        return {"raw": content[0]["text"]}
            return data.get("result")
        return None

    async def close(self):
        await self.client.aclose()


async def create_blender_capture(oxbridge_url: str = "http://localhost:9878/"):
    """
    Create a capture function that fetches scene from ox-bridge using get_scene_info.
    """
    client = OxBridgeMCPClient(oxbridge_url)

    # Initialize and discover tools
    await client.initialize()
    tools = await client.list_tools()
    print(f"[ai-db] Connected to ox-bridge, found {len(tools)} tools:")
    for t in tools:
        print(f"  - {t['name']}: {t.get('description', '')[:60]}")

    # The ox-bridge tool for scene capture is `get_scene_info`
    scene_tool = "get_scene_info"
    if not any(t["name"] == scene_tool for t in tools):
        await client.close()
        raise RuntimeError(
            f"Expected tool 'get_scene_info' not found on ox-bridge. Available: {[t['name'] for t in tools]}"
        )

    print(f"[ai-db] Using scene tool: {scene_tool}")

    async def capture() -> dict:
        """Capture current Blender scene via ox-bridge get_scene_info."""
        result = await client.call_tool(scene_tool, {})
        if result is None:
            raise RuntimeError(f"Tool {scene_tool} returned no result")

        # result should be the scene info dict from blender_mcp
        # Convert to SceneState compatible format
        if isinstance(result, str):
            result = json.loads(result)

        # blender_mcp get_scene_info returns:
        # {"status": "success", "objects": [...], "object_count": N, ...}
        # We need to adapt to our SceneState format
        if result.get("status") != "success" and result.get("ok") != 1:
            raise RuntimeError(f"get_scene_info failed: {result}")

        objects = result.get("objects", [])
        scene_data = {
            "scene_id": f"blender-{int(result.get('timestamp', 0)) or __import__('time').time()}",
            "project": "blender",
            "name": "Blender Scene",
            "timestamp": result.get("timestamp", __import__('time').time()),
            "objects": {},
            "materials": {},
            "collections": {},
            "camera": result.get("camera"),
            "prompt_context": None,
        }

        for obj in objects:
            obj_uuid = f"blender-{obj.get('name', 'unnamed')}"
            scene_data["objects"][obj_uuid] = {
                "uuid": obj_uuid,
                "name": obj.get("name", "unnamed"),
                "type": obj.get("type", "MESH"),
                "collection": obj.get("collection", "Collection"),
                "location": obj.get("location", [0, 0, 0]),
                "rotation": obj.get("rotation", [0, 0, 0]),
                "scale": obj.get("scale", [1, 1, 1]),
                "vertices": obj.get("vertices", 0),
                "faces": obj.get("faces", 0),
                "materials": obj.get("materials", []),
                "modifiers": obj.get("modifiers", []),
            }

            # Collect materials
            for mat_name in obj.get("materials", []):
                if mat_name not in scene_data["materials"]:
                    scene_data["materials"][mat_name] = {"name": mat_name}

        return scene_data

    # Store client for cleanup
    capture._client = client
    return capture


def setup_blender_capture(oxbridge_url: str = None):
    """Set up the blender capture function on the global scene memory."""
    if oxbridge_url is None:
        oxbridge_url = os.environ.get("OXBRIDGE_URL", "http://localhost:9878/")

    print(f"[ai-db] Connecting to ox-bridge at {oxbridge_url}...")

    # Run async setup
    capture_func = asyncio.run(create_blender_capture(oxbridge_url))

    # Set on the global scene memory used by MCP tools
    memory = scene_tools.get_scene_memory()
    memory.set_blender_capture(capture_func)

    print("[ai-db] Blender capture function configured")
    return capture_func


def main():
    """Main entry point - configure capture and start MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="Start ai-db MCP server with Blender ox-bridge integration")
    parser.add_argument("--oxbridge-url", default=os.environ.get("OXBRIDGE_URL", "http://localhost:9878/"),
                        help="ox-bridge MCP server URL (default: http://localhost:9878/)")
    parser.add_argument("--daemon-url", help="Connect to ai-db daemon instead of local storage")
    parser.add_argument("--project", help="Project name for daemon mode")
    args = parser.parse_args()

    # Set up blender capture
    try:
        setup_blender_capture(args.oxbridge_url)
    except Exception as e:
        print(f"[ai-db] Warning: Failed to connect to ox-bridge: {e}")
        print("[ai-db] Scene tools will require manual capture function")
        print("[ai-db] Continuing anyway...")

    # Now start the MCP server
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mcp_server import main as mcp_main
    # Override sys.argv for mcp_server.main
    sys.argv = ["ai-db", "mcp"]
    if args.daemon_url:
        sys.argv.extend(["--daemon-url", args.daemon_url])
    if args.project:
        sys.argv.extend(["--project", args.project])

    mcp_main()


if __name__ == "__main__":
    main()