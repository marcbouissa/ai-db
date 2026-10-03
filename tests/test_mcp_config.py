"""Tests for MCP configuration file writing and merging."""

import json
import tempfile
from pathlib import Path

import pytest

from ai_db.errors import AiDbConfigError
from ai_db.mcp_config import get_claude_md_snippet, write_mcp_config


def test_write_mcp_config_claude_code():
    """Test writing MCP config for claude-code client."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = write_mcp_config(tmp, client="claude-code")

        assert config_path == str(Path(tmp) / ".mcp.json")
        assert Path(config_path).exists()

        with open(config_path) as f:
            config = json.load(f)

        assert "mcpServers" in config
        assert "ai-db" in config["mcpServers"]
        assert config["mcpServers"]["ai-db"]["command"] == "ai-db"
        assert config["mcpServers"]["ai-db"]["args"] == ["mcp"]


def test_write_mcp_config_cursor():
    """Test writing MCP config for cursor client."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = write_mcp_config(tmp, client="cursor")

        expected_path = str(Path(tmp) / ".cursor" / "mcp.json")
        assert config_path == expected_path
        assert Path(config_path).exists()

        with open(config_path) as f:
            config = json.load(f)

        assert "mcpServers" in config
        assert "ai-db" in config["mcpServers"]


def test_write_mcp_config_with_daemon_url():
    """Test writing MCP config with daemon URL."""
    with tempfile.TemporaryDirectory() as tmp:
        daemon_url = "http://localhost:8080"
        config_path = write_mcp_config(tmp, client="claude-code", daemon_url=daemon_url)

        with open(config_path) as f:
            config = json.load(f)

        assert config["mcpServers"]["ai-db"]["args"] == ["mcp", "--daemon-url", daemon_url]


def test_merge_existing_mcp_config():
    """Test that existing servers are preserved when merging."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        config_path = tmp_path / ".mcp.json"

        # Write initial config with another server
        initial_config = {
            "mcpServers": {
                "other-server": {
                    "command": "other",
                    "args": ["run"]
                }
            }
        }
        with open(config_path, "w") as f:
            json.dump(initial_config, f)

        # Merge ai-db config
        write_mcp_config(tmp, client="claude-code")

        # Verify both servers exist
        with open(config_path) as f:
            config = json.load(f)

        assert "other-server" in config["mcpServers"]
        assert "ai-db" in config["mcpServers"]
        assert config["mcpServers"]["other-server"]["command"] == "other"


def test_invalid_client():
    """Test that invalid client raises error."""
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(AiDbConfigError, match="--client must be"):
        write_mcp_config(tmp, client="invalid")


def test_invalid_root():
    """Test that non-existent root raises error."""
    with pytest.raises(AiDbConfigError, match="must be a directory"):
        write_mcp_config("/nonexistent/path", client="claude-code")


def test_claude_md_snippet():
    """Test that CLAUDE.md snippet is generated correctly."""
    snippet = get_claude_md_snippet()

    assert "ai-db integration" in snippet
    assert "investigate --fmt stub" in snippet
    assert "expand ref:" in snippet
    assert "callers" in snippet

    # Should start with markdown section header
    assert snippet.strip().startswith("##")


def test_mcp_config_produces_valid_json():
    """Test that produced config is valid JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        config_path = write_mcp_config(tmp, client="claude-code")

        # Should be readable as JSON and have proper structure
        with open(config_path) as f:
            config = json.load(f)

        # Verify structure
        assert isinstance(config, dict)
        assert isinstance(config["mcpServers"], dict)
        assert isinstance(config["mcpServers"]["ai-db"], dict)
