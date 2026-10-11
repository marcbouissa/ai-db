# ai-db-scene

3D scene and video tools for [ai-db](../../README.md), shipped as a separate package so the
core code-knowledge tool stays small.

- **Scene memory**: snapshot, restore, search and fork Blender scenes.
- **Scene pipeline**: `get_scene_spec_schema` → `solve_layout` → `apply_layout`. A small model
  writes the spec; the solver places objects; Blender renders.
- **Video**: keyframes, VLM descriptions and depth from video (optional extras).

## Install

```bash
pip install -e packages/ai-db-scene            # from the ai-db checkout
pip install -e "packages/ai-db-scene[video-base,judge]"   # optional extras
```

Once installed, every ai-db entry point (CLI, `ai-db mcp`, the daemon) exposes the tools
automatically via the `ai_db.tools` entry point. Uninstall the package to remove them.

## Configuration

Settings go in the `plugins.scene` section of the ai-db config:

```json
{"plugins": {"scene": {"judge_model": "claude-opus-5-5"}}}
```

## Layout

| Path | What |
|---|---|
| `ai_db_scene/` | spec, layout solver, priors, judge, scene storage, Blender client |
| `ai_db_scene/tools/` | MCP tool registration (`register` is the entry point) |
| `ai_db_scene/video/` | video processors (loaded on first video tool call) |
| `ai_db_scene/chunkers/` | USD and segmentation-mask chunkers |
| `showcase/` | Blender showcase scripts; rendered outputs are git-ignored |
| `docs/` | 3D pipeline notes and the showcase fix log |
| `tests/` | this package's tests (`pytest packages/ai-db-scene/tests`) |
