#!/usr/bin/env python3
"""Root shim: ``python mcp_server.py`` (README, existing MCP client configs) runs the MCP
server, which lives in ``ai_db.transports.mcp`` (the documented entry is ``ai-db mcp``)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # a source checkout, uninstalled

from ai_db.transports.mcp import main

if __name__ == "__main__":
    main()
