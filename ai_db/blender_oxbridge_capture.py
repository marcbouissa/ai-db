"""
Blender ox-bridge capture function for ai-db scene memory.

Run this instead of `ai-db mcp` to start the MCP server with Blender integration:

    python -m ai_db.blender_oxbridge_capture

This connects to ox-bridge on localhost:9878 (MCP over HTTP)
and injects a capture function that fetches the current Blender scene.
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
    """Minimal MCP client to talk to ox-bridge on port 9878."""

    def __init__(self, url: str = "http://localhost:9878/mcp"):
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
                    return json.loads(content[0]["text"])
            return data.get("result")
        return None

    async def close(self):
        await self.client.aclose()


async def create_blender_capture(oxbridge_url: str = "http://localhost:9878/mcp"):
    """
    Create a capture function that fetches scene from ox-bridge.

    Returns a callable that returns a SceneState compatible dict.
    """
    client = OxBridgeMCPClient(oxbridge_url)

    # Initialize and discover tools
    await client.initialize()
    tools = await client.list_tools()
    print(f"[ai-db] Connected to ox-bridge, found {len(tools)} tools:")
    for t in tools:
        print(f"  - {t['name']}: {t.get('description', '')[:60]}")

    # Look for a scene capture tool (common names)
    scene_tool_names = [
        "get_scene",
        "get_scene_info",
        "blender_get_scene",
        "scene_get",
        "capture_scene",
        "get_current_scene",
    ]
    scene_tool = None
    for t in tools:
        if t["name"] in scene_tool_names:
            scene_tool = t["name"]
            break

    if not scene_tool:
        # Try to find any tool that looks like it returns scene data
        for t in tools:
            desc = t.get("description", "").lower()
            if "scene" in desc and ("get" in desc or "capture" in desc or "export" in desc):
                scene_tool = t["name"]
                break

    if not scene_tool:
        await client.close()
        raise RuntimeError(
            f"No scene capture tool found on ox-bridge. Available: {[t['name'] for t in tools]}"
        )

    print(f"[ai-db] Using scene tool: {scene_tool}")

    async def capture() -> dict:
        """Capture current Blender scene via ox-bridge."""
        result = await client.call_tool(scene_tool, {})
        if result is None:
            raise RuntimeError(f"Tool {scene_tool} returned no result")

        # The ox-bridge tool should return a dict compatible with SceneState
        # If it returns a different format, adapt here
        if isinstance(result, str):
            result = json.loads(result)

        # Ensure it has the required fields for SceneState
        if "scene_id" not in result:
            import uuid
            result["scene_id"] = str(uuid.uuid4())
        if "project" not in result:
            result["project"] = "blender"
        if "name" not in result:
            result["name"] = "Blender Scene"
        if "timestamp" not in result:
            import time
            result["timestamp"] = time.time()
        if "objects" not in result:
            result["objects"] = {}
        if "materials" not in result:
            result["materials"] = {}
        if "collections" not in result:
            result["collections"] = {}
        if "camera" not in result:
            result["camera"] = None

        return result

    # Store client for cleanup
    capture._client = client
    return capture


def setup_blender_capture(oxbridge_url: str = None):
    """Set up the blender capture function on the global scene memory."""
    if oxbridge_url is None:
        oxbridge_url = os.environ.get("OXBRIDGE_URL", "http://localhost:9878/mcp")

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
    parser.add_argument("--oxbridge-url", default=os.environ.get("OXBRIDGE_URL", "http://localhost:9878/mcp"),
                        help="ox-bridge MCP server URL (default: http://localhost:9878/mcp)")
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