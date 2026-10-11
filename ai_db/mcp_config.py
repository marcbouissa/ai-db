"""MCP configuration file management for ai-db integration."""

import json
from pathlib import Path

from ai_db.errors import AiDbConfigError


def write_mcp_config(
    root: str,
    client: str,
    daemon_url: str | None = None,
) -> str:
    """
    Write or merge MCP configuration for ai-db integration.

    Args:
        root: Project root directory
        client: Client type ("claude-code" or "cursor")
        daemon_url: Optional daemon URL to include in args

    Returns:
        Path to the written MCP config file

    Raises:
        AiDbConfigError: On invalid input or write failure
    """
    if client not in ("claude-code", "cursor"):
        raise AiDbConfigError(f"--client must be 'claude-code' or 'cursor', got '{client}'")

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise AiDbConfigError(f"--root must be a directory: {root}")

    # Determine config path
    if client == "claude-code":
        config_path = root_path / ".mcp.json"
    else:  # cursor
        cursor_dir = root_path / ".cursor"
        cursor_dir.mkdir(exist_ok=True)
        config_path = cursor_dir / "mcp.json"

    # Load existing config or start fresh
    config = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise AiDbConfigError(f"Failed to read existing MCP config: {e}") from e

    # Ensure mcpServers dict exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Build ai-db server config
    ai_db_config: dict = {
        "command": "ai-db",
        "args": ["mcp"],
    }

    if daemon_url:
        ai_db_config["args"].extend(["--daemon-url", daemon_url])

    # Merge in ai-db config (preserving any existing non-ai-db servers)
    config["mcpServers"]["ai-db"] = ai_db_config

    # Write config
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
            f.write("\n")
    except OSError as e:
        raise AiDbConfigError(f"Failed to write MCP config: {e}") from e

    return str(config_path)


def get_claude_md_snippet() -> str:
    """Get the CLAUDE.md snippet for ai-db MCP integration."""
    return """\
## ai-db integration

Use `investigate --fmt stub` before reading files. `expand ref:` only what you need.

- Read the target lines before editing. Trust `stale: true` flags.
- Use qualified names with `callers` (`Class.method`). Check `truncated`.
- Run noisy commands (tests, builds, linters) as `ai-db run --need "<what you need>" -- <cmd>`:
  you get the failures, tracebacks and summary, not the whole log; the exit code is kept.
  Drill in with `ai-db filter --id <id> "<other need>"` or `--lines A-B` instead of re-running.
  Add `--rerank` when your need is phrased differently from what the output would say.
"""
