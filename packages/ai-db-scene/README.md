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

### Asset providers (optional, off by default)

`build_scene` / `apply_layout` build every object from parametric furniture or a part recipe.
With an asset provider, a real model file is used first when one matches the object's category
(an explicit `asset_map` entry still wins; `geometry: "boxes"` uses none). The object keeps its
id, tags and placement; `asset_source` on it says where the model came from.

**Local library** — folders of your own `.glb`, `.gltf` and `.blend` files:

```json
{"plugins": {"scene": {"asset_dirs": ["~/assets/street", "~/assets/furniture"]}}}
```

Files are matched to categories by the words in their name, numbers ignored, the most specific
category winning: `traffic_cone_01.glb` → `traffic_cone`, `lamp-post double.gltf` →
`lamp_post_double`. A glTF file is imported, scaled uniformly to fit the placement, its base
put on the floor and turned to the placement's yaw. A `.blend` file is appended and must hold an
object named like the file (`park_bench.blend` → object `park_bench`). Every folder must exist.

**Poly Haven** — models from [polyhaven.com](https://polyhaven.com) through its API
(`https://api.polyhaven.com`). The assets are CC0, but using the API commercially needs a
licence from Poly Haven, so it stays off until you say you have one:

```json
{"plugins": {"scene": {"polyhaven": {"enabled": true, "license_acknowledged": true,
                                     "resolution": "1k", "cache_dir": "~/.cache/ai-db/polyhaven"}}}}
```

`enabled` without `license_acknowledged` is a config error. The model list is cached for a day;
a chosen model's glTF and textures are downloaded once into `cache_dir` (default
`$XDG_CACHE_HOME/ai-db/polyhaven`). Requests carry an `ai-db-scene` User-Agent, as the API
requires. A provider that fails (no network, a bad file) is skipped and the recipe is used.

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
