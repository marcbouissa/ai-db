# Side note: ai-db and ox-blender features used in the session of 2026-10-09 to 2026-10-11

Legend: **[new]** added in this session, **[changed]** reworked in this session, **[used]** already
existed and was used. ox-blender is the Blender addon in `~/GitRepos/blender_mcp` (package
`blender_ox`, socket server on :9877, MCP server `blender_ox_mcp`). Its changes from this session
are uncommitted (mixed with the owner's own work there).

## ai-db core (`ai_db/`)

| Feature | Status | Notes |
|---|---|---|
| Daemon shared by all Claude sessions (systemd user unit, `AI_DB_DAEMON_URL`) | [changed] | per-project pools; idle cleanup releases models |
| GPU for big syncs, CPU for small ones (`embedding.bulk_device`, `bulk_threshold`) | [new] | `configure_cpu_threads` caps torch threads |
| Embedding text cap (`embedding.max_seq_length`, default 2048) | [new] | fixed a 16 GiB CUDA OOM on long skills |
| Model sharing across daemon projects (`release_models`) | [new] | |
| Daemon per-user keys (`AI_DB_USER_KEYS`), `/status/html`, auto-watch of sync paths | [new] | core backlog merge |
| Cross-language call linking, parser fixes in 9 languages, one diff engine, storage split into repositories, MCP server in `ai_db/transports/mcp.py` | [changed] | core backlog G1-G10, K1-K9 |
| Schema 13 + NULL-project skill cleanup on open | [changed] | |
| WAL checkpoint skips projects closed by idle cleanup | [changed] | |
| `AI_DB_MCP_TOOLS` allowlist | [new] | |
| Plugin entry points `ai_db.tools`, `ai_db.skills`, `ai_db.recall`, `ai_db.skill_drafts`, `ai_db.skill_sources` | [new] | |
| Skill generation: `create_skill`, `draft_skill`, `list_generated_skills`, `delete_skill` | [new] | |
| Skill drafting from sources: tutorial text/URLs (incl. manual option lists), local videos and subtitles, `.gltf/.glb/.blend` assets | [new] | no downloads of models; private URLs refused |
| `route_skill`, `sync_skills`, `query`, `investigate`, `analyze`, `sync` | [used] | |
| Evaluation harness (`ai-db eval`: lexical, pack, polyglot, symbols, callers) | [changed] | a hit on a method now finds its class; baselines re-recorded |
| Config migration (`ai-db init --migrate`) | [used] | removed `retrieval.doc_weight` |

## ai-db scene plugin (`packages/ai-db-scene/`)

### Tools

| Tool | Status | What it does |
|---|---|---|
| `get_scene_spec_schema` | [changed] | brief-relevant categories, exemplars, styles, **lessons** for the brief |
| `submit_scene_spec`, `solve_layout`, `apply_layout`, `validate_scene` | [changed] | |
| `build_scene` | [new] | layout + repairs + build + validator fix loop + record_run + snapshot; options below |
| `edit_scene` | [new] | ordered edit steps, rebuild, versioned |
| `snapshot_scene`, `restore_scene`, `fork_scene`, `get_scene_versions`, `list_scenes`, `search_scenes_by_prompt` | [changed] | .blend geometry, exact restore |
| `diff_scenes`, `restore_object` | [new] | |
| `export_scene` (glb/gltf/usd/usdz) | [new] | |
| `record_run`, `search_exemplars` | [used] | |
| `create_recipe`, `draft_recipe`, `list_recipes` | [new] | recipes from text, incl. ShapeAssembly; soft-goods shapes |
| `record_feedback`, `list_feedback`, `update_feedback`, `delete_feedback`, `optimize_feedback` | [new] | user complaints become lessons later builds check; global or per project |

### `build_scene` options and automatic steps

| Feature | Status |
|---|---|
| Spec repair (drop conflicting relation, grow room, one copy fewer) | [new] |
| Validator fix loop | [new] |
| Exposure check (probe render, up to 3 corrections) | [new] |
| Camera coverage check (inside the room, heroes visible, alternative poses) | [new] |
| Level interior camera at 1.4 m, 20 mm, framed by lens shift | [new] |
| `judge: true` critique with suggested edit steps | [new] |
| `render: {engine, samples, sky}` (Nishita sky, no HDRI) | [new] |
| `poly_budget` (decimate heaviest non-hero objects) | [new] |
| `physics: true` (rigid bodies, `dynamic` objects) | [new] |
| Ergonomics warnings (coffee-table reach and height, side-table height, dining clearance) | [new] |
| Lessons from user feedback (escalate checks, floating-part tests, verify items) | [new] |

### Spec, layout and rooms

| Feature | Status |
|---|---|
| Relations: `against_wall`, `in_front_of`, `beside`, `facing`, `centered_in`, `on`, `under`, `around`, **`above`** | [changed] |
| `grid` blocks (rows of desks, racks) | [new] |
| Setting shapes rect / ellipse / polygon; openings in curved walls | [changed] |
| Multi-room buildings (`setting.rooms`, doors, inner walls, lintels, door clearance) | [new] |
| `setting.style` presets (modern, scandinavian, industrial, rustic, classic; free text accepted) | [new] |
| `setting.mouldings` (crown, chair rail, baseboard; classic style turns them on) | [new] |
| Ceilings, plank floors, rug pile | [new] |
| Outdoor ground to the horizon | [new] |
| Warehouse rooms: concrete floor and walls, painted aisle and walkway lines | [new] |
| Lighting by room size (exposure by floor area, ceiling fill, lamp lights, window portals) | [new] |

### Assets and materials

| Feature | Status |
|---|---|
| 67 part recipes, including 12 outdoor/industrial | [changed] |
| Rounded edges on every recipe box (~6 mm, real metres) | [new] |
| Seeded recipe variants | [new] |
| Warehouse set with real structure: pallet rack, roller conveyor, forklift, EUR pallet, slatted crate, safety barrier | [new] |
| Office chair (five-star base, casters, cushions, armrests), toilet (bowl, seat, cistern), hollow bathtub (`shell` op) | [new] |
| Kitchen and bath: shaker cabinet run with marble top, cooker (grates, knobs, windowed oven), vanity with hollow basin and mixer | [new] |
| Bedroom: beds from the addon's bed generator (size by width, style preset, seeded); shaker fronts on cabinet, wardrobe, dresser, nightstand, filing cabinet, TV stand | [new] |
| Furniture-scale wood figure on every recipe part | [new] |
| Fridge-freezer (two doors, bar handles, kick plate), whiteboard (aluminium frame, tray, markers), monitor/TV (thin bezel, glass screen) | [new] |
| Floor and table lamps: hollow tapered linen shades with bulbs, weighted base, ceramic body | [new] |
| Picture frame (gilt moulding, mat, painted landscape canvas) and wall mirror (moulded frame, mirror glass) | [new] |
| Classic marble fireplace (hearth, pilasters, capitals, frieze, mantel shelf, sooty firebox with brick cheeks and logs) and a glazed ceramic vase | [new] |
| Car: rounded painted body, glass greenhouse, roof, lights, grille, bumpers, mirrors, rubber tyres with brushed rims | [new] |
| Street lamp (tapered pole, arm, lantern head with lens), street bin (hollow steel body, liner), steel drum (hoops, chimes, bungs), A-frame picnic table | [new] |
| Kitchen island (shaker fronts, overhanging worktop), shower stall (clear glass in chrome frames, tray, head, mixer), straight stair (treads with nosing, risers, stringers) | [new] |
| Boulder (flattened mass with lumps, stone, moss); planters get leaf foliage | [new] |
| Generated furniture varies per scene and follows the style (seed + preset) | [new] |
| Trees grown from a branching skeleton; leaf-card foliage on trees, bushes, hedges, plants | [new] (pine, birch and palm types prototyped, not committed) |
| Material vocabulary incl. `concrete`, `stone`, `foliage`, `plastic`, `paint`, `rubber`, `pbr:<preset>` wear | [new] |
| Fallback to flat colour on an older addon | [new] |
| Asset providers: local library (with CC0 pack filename aliases), Poly Haven (off by default, license acknowledgement) | [new] |

### Skills shipped with the plugin

| Skill | Status |
|---|---|
| `ai-db-scene-pipeline` | [changed] |
| `blender-material-library` | [changed] |
| `pbr-shader-builder` | [new] |
| `recipe-from-text` | [new] |
| `realistic-modeling` | [new] |
| `cloth-and-textiles` | [new] |
| `interior-camera-composition` | [new] |

### Evaluation and benchmark

| Feature | Status |
|---|---|
| `eval/scene_bench.py` (local-model scene benchmark) | [new] |
| Golden spec set (`tests/fixtures/golden/`, solver level + live Blender) | [new] |
| Teacher loop (`eval/teacher.py`, dry run by default) | [new] |
| `eval/agent_bench/` | [new] |

`eval/agent_bench/` covers:
- **Runners:** OpenCode free-model runner and the Strata agent (custom MCP client).
- **Supervision:** a resumable supervisor.
- **Arms:** aidb, aidb2, aidb3 and raw.
- **Scoring:** quality score v2, blind Opus judging (full and paired) and the report.
- **Briefs:** an on-demand `oval_office` brief.

### Infrastructure

- **CI:** `.github/workflows/ci.yml`: tests, lint and evals, plus a headless-Blender job when `BLENDER_ADDON_REPO` is set **[changed]**.
- **Packaging:** `ai-db-scene` builds as a wheel **[new]**.
- **Docs:** `docs/SCENE_TUTORIAL.md` **[new]** and `docs/3D_PIPELINE.md` **[changed]**.

## ox-blender addon (`~/GitRepos/blender_mcp`, package `blender_ox`)

| Handler / module | Status | Used for |
|---|---|---|
| `build_asset` (parts) | [used] | every recipe |
| `build_asset` shape types and ops | [used] | box with `bevel`, cylinder, tapered_cylinder, sphere, rounded_slab, cushion, tufted_panel, raised_panel; `shell` op for hollow tubs and basins |
| `generate_asset` families with `seed` and `preset` | [used] | sofa, armchair, dining_chair, table, desk, bookshelf, bed |
| `apply_material` and the procedural library (`materials.py`) | [changed] | **new kinds** concrete, stone, foliage, plastic, paint, rubber; micro-roughness on `color` |
| `room_shell` (openings, floor, ceiling) | [used] | rooms; ceiling now on |
| `wood_plank_floor`, `carpet_pile` (`surface.py`) | [used] | plank floors, rug pile |
| `daylight_rig`, `set_camera`, `frame_camera`, `render` | [used] / [changed] | `frame_camera` fits the projected extent (`lights_cameras.py`) |
| `validate_scene` (`verify.py`, `scene_checks.py`) | [changed] | structure penetration, coverings (rugs, floors), facing rays, deduplicated pairs |
| `create_primitive`, `modify_object`, `clear_scene`, `append_asset`, `import_file`, `run_code` | [used] | build steps, mouldings, foliage, trees, exposure and camera checks, export |
| Moulding profiles and sweep (`profiles.py`) | [used] | crown, chair rail and baseboard through `run_code` |
| `add_cloth`, `add_collision` (`physics.py`) | [used] | documented in the cloth-and-textiles skill |
| Rigid bodies | [used] | `build_scene(physics: true)` |
| Headless server (`tools/serve_headless.py`), `tools/material_sheet.py`, `tests/run_blender_tests.py` | [used] | private Blender for tests; material sheets; 182 addon tests pass |
| Regression tests (`tests/test_bl_rooms.py`, `tests/test_bl_materials.py`) | [changed] | rooms checks; new material kinds |

## Not finished or left for later

- Tree types (pine, birch, palm): drop the paper-looking flattened sprays and scatter needles and leaflets along twigs instead.
- Opus judging of the "ai-db v2" and "v3" benchmark arms after the weekly reset, then `bench/REPORT.md`.
- The free-model and Strata benchmark runs, when rate limits and the Strata PC allow.
