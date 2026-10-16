# Scene tutorial: from a brief to an exported Blender scene

This walks through the `ai-db-scene` MCP tools end to end. Tool names and parameters are
the ones the MCP server registers; call them from any MCP client connected to `ai-db mcp`.

## 1. Install

```bash
pip install -e ".[dev]" -e packages/ai-db-scene   # or: uv pip install ...
ai-db init
ai-db mcp                                         # the scene tools register automatically
```

A wheel builds with `uv build packages/ai-db-scene` (it ships the skills and `priors.json`).

## 2. Connect Blender

The tools drive Blender through the socket addon from the separate `blender_mcp` repo.
Install the addon in Blender (Edit > Preferences > Add-ons > Install from disk, pick the
addon folder or its zip) and start its server. ai-db connects to `127.0.0.1:9877`;
override with `AI_DB_BLENDER_HOST` / `AI_DB_BLENDER_PORT`. Headless works too:
`blender -b --python <script that starts the addon server>`.

## 3. Write a spec, build it

1. `get_scene_spec_schema {brief: "cozy living room"}` returns the SceneSpec JSON schema,
   the category vocabulary with typical sizes, the relations, and up to 3 exemplar specs
   from earlier successful runs.
2. Write a spec and build it in one call:

```json
build_scene {
  "spec": {
    "setting": {"kind": "indoor", "bounds_m": [5, 4, 2.7], "style": "scandinavian"},
    "objects": [
      {"id": "sofa", "category": "sofa", "hero": true},
      {"id": "table", "category": "coffee_table", "material": "pbr:weathered_wood"},
      {"id": "lamp", "category": "floor_lamp"}
    ],
    "relations": [
      {"a": "sofa", "rel": "against_wall", "b": "wall_north"},
      {"a": "table", "rel": "in_front_of", "b": "sofa", "gap_m": 0.45},
      {"a": "lamp", "rel": "beside", "b": "sofa"}
    ]
  },
  "seed": 1, "prompt": "cozy living room", "judge": true
}
```

`build_scene` validates, lays out (repairs are reported), builds with real furniture,
materials, daylight and a camera, validates again and saves a scene version
(`snapshot: true` by default; the result carries its `scene_id`). Options:

| Parameter | Effect |
|---|---|
| `geometry` | `generated` (default: addon furniture generators, part recipes, asset providers) or `boxes` |
| `clear` | empty the Blender scene first |
| `asset_map` | `id` or `category` -> `{filepath, asset_name}` to use your own files |
| `judge` | render the build camera and return a critique with suggested `edit_scene` steps (`render_path` to keep the render) |
| `poly_budget` | total faces; decimates the heaviest non-hero objects until the scene fits |
| `physics` | rigid bodies: furniture passive, objects with `"dynamic": true` active |

Spec features worth knowing:

- **Styles**: `setting.style` is one of `modern`, `scandinavian`, `industrial`, `rustic`,
  `classic`; it picks floor, wall and per-category materials. Explicit materials win.
- **Materials**: `kind[:variant][:#hex]` strings (`wood:oak`, `fabric:weave:#5a5f66`,
  `metal:steel`), or `pbr:<preset>` for worn PBR looks (`pbr:worn_metal`, `pbr:rusty_steel`,
  `pbr:aged_brass`, `pbr:weathered_wood`, `pbr:scuffed_leather`, `pbr:dusty_plaster`, ...;
  the schema lists them all).
- **Rooms**: an indoor rect setting can hold `rooms: [{id, offset_m, size_m}, ...]`; each
  object then names its `room`, and doors join rooms that share a wall.
- `count`, `grid: [rows, cols, col_gap_m, row_gap_m]`, `detail`, `hero` per object; non-rect
  footprints with `shape` / `points`.

## 4. Edit, version, compare, restore

- `edit_scene {scene_id, steps}` applies ordered steps to the stored spec and rebuilds:
  `add_object`, `remove_object`, `update_object {set: {count|material|hero|grid|category|detail}}`,
  `add_relation`, `remove_relation`, `update_setting {set: {bounds_m|style|floor_material|...}}`.
  Example: `[{"op": "update_object", "id": "sofa", "set": {"material": "velvet:#5b2a4a"}}]`.
  Each edit saves a new version linked to its parent, in the parent's project, rebuilt with the
  version's own seed, geometry mode and furniture variants (untouched objects stay as they were).
  A version saved before these were stored gets them recovered from its stored objects (the
  layout seed that puts them where they are); the result's `build_inferred` says what was
  inferred, or warns that untouched objects may move.
  A fork can be edited too (it uses its ancestor's spec).
- `snapshot_scene {prompt, project?, name?, tags?, parent_scene_id?}` saves the current Blender
  scene by hand (after manual changes). A new scene is named after its prompt (a build without
  one: after its setting, `classic indoor 5x4 m: desk, chair, shelf`), a version keeps its
  parent's name; `list_scenes` shows the names.
- `get_scene_versions {scene_id}` lists the version chain.
- `diff_scenes {scene_a, scene_b}` shows added, removed and changed objects.
- `restore_object {scene_id, name}` brings back one object from a version (with its parts:
  restoring `desk` restores the legs and drawers; a single part is reattached to the live
  parent); `restore_scene {scene_id, mode: replace|additive}` the whole scene. Restores report
  `mismatches` (transform or materials) and `missing_files`. Snapshots pack external textures,
  so a deleted texture cache does not break them. A version's geometry lives in
  `<store dir>/blends/<scene_id>.blend`: move it together with the store.

## 5. Export

`export_scene {path: "living_room.glb"}` writes glb, gltf, usd, usda, usdc or usdz (from `format`
or the extension) under `plugins.scene.export_dir`; paths outside it are refused. With
`scene_id` it exports a stored version (a fork with its own transforms) instead of the live
scene. A separate headless Blender does the export (the live session is left as it is), then
imports the file back: `roundtrip` lists mesh objects that are missing or whose world
transform or materials changed (`status: exported_with_mismatches`). Geometry-nodes instances
(tree leaves) are made real geometry first, so a park exports with its leaves (large files:
~50 MB glb). In USD an object with children comes back as an empty holding its mesh.

## 6. Your own part recipes

A recipe describes a category as parts in relative units (`box`, `cylinder`, `sphere`,
`tapered_cylinder`, each touching a chain to the ground). Recipes come from text, never
from downloaded model files.

- `draft_recipe {category, text}`: from a ShapeAssembly program, or from product-page text
  ("W 180 cm x D 90 cm x H 75 cm, 4 legs") a connected skeleton to finish.
- `create_recipe {category, name, parts, dims?, source: {url, license}, overwrite?}` saves it;
  `list_recipes` lists them. A built-in recipe, and the addon's furniture generator for its
  categories (sofa, chair, tables, desk, bookshelf, ...), win unless you pass `overwrite: true`.

## 7. Your own skills

- `draft_skill {topic, sources?}` drafts a SKILL.md from what ai-db knows plus optional
  sources: `{text, title?, url?, license?}`, `{url}` (fetched as text), `{video: path}`
  (subtitles/keyframes), `{asset: path or object}` (.gltf/.glb/.blend or a Blender object).
  No LLM call.
- Edit the draft, then `create_skill {name, description, body, triggers?, sources?, overwrite?}`;
  `route_skill` finds it at once. `list_generated_skills` / `delete_skill` manage them.

## 8. Asset providers (off by default)

Real model files are used before recipes only when you turn a provider on in the config
(`plugins.scene`), because their licenses are yours to check:

```json
{"plugins": {"scene": {
  "asset_dirs": ["~/assets/furniture"],
  "polyhaven": {"enabled": true, "license_acknowledged": true, "resolution": "1k"}
}}}
```

`asset_dirs` are local folders of `.glb`/`.gltf`/`.blend` files matched by filename. Poly Haven
assets are CC0, but its API terms apply; it stays off until `license_acknowledged` is set.
See [packages/ai-db-scene/README.md](../packages/ai-db-scene/README.md) for details.

## 9. Teach it from feedback

When a result is wrong, say so in your words and let the agent record it:

```
record_feedback(text="the bench back is floating")
record_feedback(text="the clothes are too stiff", subjects=["clothes"], severity="warning")
record_feedback(text="sofas face the wall", project="client_a")   # this project only
```

Each complaint becomes a lesson (objects + kind of problem + the check that sees it).
`get_scene_spec_schema` lists the lessons for the brief's objects under `lessons`, and
`build_scene` checks them afterwards: a matching validator check becomes blocking for those
objects, recipe parts are tested for contact, and problems no check can see come back as
`lesson_verify` items. Lessons are global unless recorded for a project. Manage them with
`list_feedback`, `update_feedback` (`{"enabled": false}` disables), `delete_feedback`, and
`optimize_feedback` (proposes merging duplicates, fixing mistyped categories and retiring
stale lessons; `apply: true` carries the proposals out).

## 10. What makes it look real

- `get_scene_spec_schema` returns `room_kits` for the room types a brief names (living room,
  bedroom, kitchen, bathroom, office, classroom, cafe, warehouse, park): what such rooms hold
  and where it goes. Use them to complete a room; the brief always wins.
- `build_scene` adds `ergonomics` warnings (coffee table out of reach or too tall, side table
  height, room to pull a dining chair out) and `realism` warnings (flat colour on big surfaces,
  no lamp or window, bare walls).
- Materials: prefer kinds with variation (`paint:`, `plastic:`, `concrete`, `stone`,
  `foliage`, `rubber`, wood) over `color:#...`; `pbr:<preset>` adds wear.
- Rooms get a ceiling, plank floors (rect rooms), carpet pile on rugs, painted lines in
  warehouses; trees grow from a branching skeleton and plants get leaf foliage.
- The camera stays level (straight verticals) at 1.4 m on a 20 mm lens; exposure is checked.
- Skills to read: `realistic-modeling`, `cloth-and-textiles`, `interior-camera-composition`,
  `blender-material-library`, `pbr-shader-builder`.
