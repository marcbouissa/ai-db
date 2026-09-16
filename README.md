# ai-db: Zero-Dependency Code Knowledge & Vector Index Engine

A high-speed, zero-dependency SQLite FTS5 + BM25 ranking vector database and knowledge index designed specifically for AI coding assistants (like Antigravity / Gemini / Claude) to **instantly recall codebase architecture, symbols, and files without slow repetitive codebase re-analysis on every new chat**.

## Features
- **Zero External Dependencies**: Powered entirely by the Python 3 standard library (`sqlite3`, `math`, `re`, `hashlib`, `json`, `ast`, `time`).
- **AST Code Validation & Pre-Flight (`ai-db check`)**: Instant detection of broken syntax with exact file, line, and column numbers.
- **Exact Symbol Resolution (`ai-db symbol`)**: Look up exact `class`, `def`, and `interface` definitions without noisy BM25 full-text false positives.
- **Instant File Outlines (`ai-db outline`)**: Extract function signatures, class hierarchies, and key definitions without loading large files into LLM context.
- **Sub-Millisecond Queries (`ai-db query`)**: Instant search across thousands of code chunks using SQLite FTS5 with BM25 ranking and identifier-aware tokenization.
- **Incremental Synchronization (`ai-db sync`, `ai-db sync-all`)**: SHA-256 hash detection indexes only new or modified files, pruning deleted files automatically.
- **Continuous Watch Daemon (`ai-db watch`)**: Background daemon keeping the index fresh on project file changes.

## CLI Usage

### 1. Code Health & Syntax Pre-Flight
Detect syntax errors before executing code or running tests:
```bash
ai-db check .
ai-db check /path/to/file.py
```
Output:
```
SYNTAX_ERROR: path/to/file.py:54:12 Unterminated string literal
Total errors: 1
```

### 2. Exact Symbol Definition Lookup
Jump directly to canonical symbol definitions:
```bash
ai-db symbol execute_command
ai-db symbol VectorDB
```
Output:
```
@blender_ox/handlers.py:L647 (def execute_command)
@vectordb.py:L473 (class VectorDB)
```

### 3. File Skeleton / Outline Extraction
Inspect structure without blowing context tokens:
```bash
ai-db outline blender_ox/handlers.py
```
Output:
```
File: blender_ox/handlers.py (669 lines)
  L56: _PRIMITIVES (dict)
  L74: def create_primitive(d)
  L647: def execute_command(payload)
```

### 4. Semantic & Keyword Search (BM25)
```bash
ai-db query "sqlite fts5 ranking" --top 5
```

### 5. Synchronize / Index a Codebase
```bash
ai-db sync /path/to/project
```

### 6. Multi-Project Synchronization (`sync-all`)
Register projects in `~/.config/ai-db/config.json`:
```json
{
  "auto_sync_paths": [
    "~/.local/share/blender-ai/skills",
    "~/Documents/antigravity/noble-tesla",
    "~/GitRepos/blender_mcp",
    "~/GitRepos/ai-db"
  ]
}
```
Synchronize all registered projects:
```bash
ai-db sync-all
```

### 7. Continuous Background Watch Daemon
```bash
# Foreground watch:
ai-db watch /path/to/project

# Background daemon:
ai-db watch /path/to/project --daemon
```

### 8. Smart Prompt Skill Routing (`ai-db route-skill`)
Analyze user prompts and deterministically route to the best matching skills in `< 2ms` without polluting the context window with dozens of skill descriptions:
```bash
ai-db route-skill "Design an ad banner for LinkedIn with glassmorphism style"
```
Output:
```
[PRIMARY] banner-design (conf: 0.93) -> ~/.gemini/config/skills/banner-design/SKILL.md
  Reason: Matched triggers: design, glassmorphism, banner; High semantic relevance; Banner & creative asset keywords
[SECONDARY #2] design (conf: 0.89) -> ~/.gemini/config/skills/design/SKILL.md
```

You can also output in clean machine formats:
```bash
# Output only the best skill paths
ai-db route-skill "Create 5 presentation slides" --format path

# Output JSON with confidence scores and reasoning
ai-db route-skill "Check if there are broken python files" --format json
```

### 9. Project Scoping & Cross-Project Access Isolation
By default, all code, symbols, syntax checks, and skills are scoped to their respective project (auto-detected from the nearest `.git`, `package.json`, or `pyproject.toml`):
- Operations inside a project only access assets belonging to that project or the `global` scope.
- Project-local skills in `<project>/skills/` or `<project>/.agents/skills/` are automatically synced under the project's scope.
- Other project context is strictly invisible unless explicitly allowed on a read-only basis:
```bash
# Allow read-only access to another project via CLI:
ai-db symbol other_func --allow-project other-project
ai-db query "authentication" --allow-project other-project
ai-db route-skill "custom-workflow" --allow-project other-project
```

To permanently configure cross-project read-only permissions, add `cross_project_access` to `~/.config/ai-db/config.json`:
```json
{
  "cross_project_access": {
    "my-frontend": ["my-backend", "shared-utils"]
  }
}
```

### 10. Session Context & Chat Memory (`ai-db context` / `ai-db remember`)
Store and recall conversation checkpoints, active files, architectural decisions, and open tasks across chats without re-reading the entire repository:

#### Save Session Context:
```bash
ai-db context save session-auth \
  --title "Auth Migration" \
  --summary "Migrating authentication from sessions to JWT and refresh tokens" \
  --files "auth.py,models/user.py" \
  --tasks "Implement token refresh,Add unit tests for expired token"
```

#### Recall Context Memory (`/remember`):
```bash
ai-db remember
# or
ai-db context get
```
Output (token-dense markdown ready for instant injection):
```markdown
### [ai-db Context Memory: my-project / session-auth]
**Title**: Auth Migration
**Summary**: Migrating authentication from sessions to JWT and refresh tokens

**Active Files**:
- auth.py
- models/user.py

**Pending Tasks**:
1. Implement token refresh
2. Add unit tests for expired token
```

#### List & Search Session Contexts:
```bash
ai-db context list
ai-db context query "JWT token"
```

### 11. Index Installed Skills (`ai-db sync-skills`)
```bash
ai-db sync-skills
```

### 12. Check Database Health
```bash
ai-db status
```

### 13. Optimize & Defragment Database (`ai-db optimize`)
Prune stale references, optimize and merge FTS5 inverted indexes, update query planner statistics (`PRAGMA optimize`), and reclaim fragmented disk space (`VACUUM`):
```bash
ai-db optimize
# or:
ai-db vacuum
```

### 14. Token-Optimized Code Analysis (`tokenopt-analyzer v2`)
Minimal token consumption, maximum relevance, progressive disclosure, and zero AI-side blind exploration:

#### Depth Control (`summary | structure | targeted | full`)
```bash
# Outline signatures only (<= 10% of raw file tokens):
ai-db analyze vectordb.py --depth summary

# AST outline with class methods:
ai-db analyze vectordb.py --depth structure

# Targeted: extract bodies only matching query or focus:
ai-db analyze vectordb.py -q "semantic cache" --depth targeted
```

#### Range Targeting (`--span` & `--ctx`)
```bash
# Extract lines 65-75 plus 5 lines of surrounding context:
ai-db analyze vectordb.py --span 65:75 --ctx 5
```

#### Progressive Disclosure (`ai-db expand`)
`analyze` returns opaque handles (`ref:hash`) for discovered AST nodes. Expand bodies on-demand without context bloat:
```bash
ai-db expand ref:b3b64697
# Or expand specific sub-span:
ai-db expand ref:b3b64697 --span 1:10
```

#### Relevance Ranking (`ai-db locate`)
Find top-k candidate files and snippets by natural question or concept via FTS5 BM25:
```bash
ai-db locate "syntax error validation" -k 5
```

#### Output Format Serialization (`--fmt stub | sexp | json`)
Optimize token density based on agent needs:
- `--fmt stub` (**Native Code Skeleton**): Best for LLM code analysis, reasoning, and type inspection. Consumes ~50% fewer tokens than JSON.
- `--fmt sexp` (**S-Expression / Lisp**): Tree-native AST representation for absolute minimum token consumption (~63% fewer tokens).
- `--fmt json` (**Standard JSON**): Traditional format for human inspection and external tool integration.

#### Dynamic Format Optimization & Per-Run Override
The agent can configure and persist the database's default output serialization format to match its preferred reasoning mode, while retaining the ability to override it on any specific run:
- **Set default format**:
  ```bash
  ai-db optimize --default-format sexp   # Persist S-Expression as DB default
  ai-db optimize --default-format stub   # Persist Native Skeleton as DB default
  ```
- **Autonomous Query Execution**:
  When executing `ai-db analyze` or `ai-db locate` without any `--fmt` parameter, `ai-db` automatically queries and outputs in the persisted default format (`stub`, `sexp`, or `json`).
- **Per-Run Override**:
  Passing `--fmt` / `-fmt` dynamically overrides the database default for that single execution:
  ```bash
  ai-db analyze vectordb.py --depth summary --fmt json
  ```
- **MCP Autonomous Configuration**:
  Agents can configure this via the `optimize` tool (`{"default_format": "sexp"}`) and specify `{"format": "stub"}` on `analyze` / `locate` calls.

#### Model Context Protocol (MCP) Server
`ai-db` includes a high-performance, zero-dependency JSON-RPC 2.0 stdio MCP server exposing `analyze`, `expand`, `locate`, `context_save`, `context_recall`, and `optimize` with self-describing schemas under 200 tokens:
```bash
# Launch MCP server manually:
ai-db mcp
# Or:
python3 /home/marc/GitRepos/ai-db/mcp_server.py
```
Configured in `~/.gemini/config/mcp_config.json`:
```json
{
  "mcpServers": {
    "ai-db": {
      "command": "python3",
      "args": ["/home/marc/GitRepos/ai-db/mcp_server.py"]
    }
  }
}
```

## AI Agent Policy Integration
Antigravity automatically checks `ai-db` first on new sessions to route relevant skills, check syntax health, locate symbol definitions, and recall existing project context (`ai-db remember`) without performing slow, full-codebase directory crawls.


