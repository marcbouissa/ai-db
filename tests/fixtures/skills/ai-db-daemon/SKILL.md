# Skill: ai-db Daemon Management

## Description
Manage and interact with the ai-db daemon server for long-running code intelligence with shared model loading.

## When to Use
- Starting/stopping the daemon background process
- Registering projects with the daemon
- Querying via daemon for faster repeated requests
- Using MCP server with daemon for shared model loading
- Managing multiple projects in a single daemon instance

## Prerequisites
- ai-db installed with `[daemon]` extra: `pip install -e ".[daemon]"`
- Config file created with `ai-db init`

## Operations

### Start Daemon
```bash
# Start in background (default port 8080)
ai-db daemon start

# Start in foreground for debugging
ai-db daemon start --foreground

# Custom host/port
ai-db daemon start --host 0.0.0.0 --port 8080
```

### Register Project
```bash
# Register project with daemon
ai-db daemon register myproject --db-path ~/.local/share/ai-db/myproject.db

# With custom config
ai-db daemon register myproject --db-path ~/.local/share/ai-db/myproject.db --config-path /path/to/config.json
```

### Use Daemon for Queries
```bash
# Query via daemon (models already loaded)
ai-db --daemon-url http://localhost:8080 --project myproject query "search term"
ai-db --daemon-url http://localhost:8080 --project myproject investigate "how does X work"
ai-db --daemon-url http://localhost:8080 --project myproject sync /path/to/repo
```

### MCP Server with Daemon
```bash
# Terminal 1: Start daemon
ai-db daemon start

# Terminal 2: Launch MCP pointing at daemon
ai-db mcp --daemon-url http://localhost:8080 --project myproject
```

### Stop Daemon
```bash
# Graceful shutdown
ai-db daemon stop

# Force kill
ai-db daemon stop --force
```

### Status & Management
```bash
# Check daemon status
ai-db daemon status

# Reload config without restart
ai-db daemon reload

# List registered projects
ai-db daemon register --list  # via status
```

### With Authentication
```bash
# Set API key
export AI_DB_API_KEYS="myproject=mysecretkey"
export AI_DB_API_KEY=mysecretkey

# Daemon with auth
AI_DB_API_KEYS="myproject=mysecretkey" ai-db daemon start

# Client with auth
ai-db --daemon-url http://localhost:8080 --project myproject query "hello"
```

## Environment Variables
- `AI_DB_DAEMON_URL` - Daemon base URL (default: http://localhost:8080)
- `AI_DB_API_KEY` - API key for authenticated requests
- `AI_DB_API_KEYS` - Comma-separated project=key pairs for daemon auth
- `AI_DB_CONFIG` - Config file path

## Configuration
Add to `~/.config/ai-db/config.json`:
```json
{
  "daemon": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8080,
    "auto_discover": true,
    "projects": {
      "myproject": {
        "db_path": "~/.local/share/ai-db/myproject.db",
        "auto_start": true
      }
    }
  }
}
```

## MCP Integration
Configure your AI client (Claude Desktop, Cursor, etc.) to use the MCP server with daemon:

```json
{
  "mcpServers": {
    "ai-db": {
      "command": "ai-db",
      "args": ["mcp", "--daemon-url", "http://localhost:8080", "--project", "myproject"]
    }
  }
}
```

## Notes
- Daemon holds embedding/rerank models in memory, eliminating ~200ms per-CLI model load
- Projects auto-warm on registration (models loaded before first query)
- Idle projects unload after 5 minutes of inactivity
- Hot config reload watches config.json and auto-reloads
- Cross-project queries via `/query/cross-project` endpoint
- WebSocket progress streaming for long-running operations