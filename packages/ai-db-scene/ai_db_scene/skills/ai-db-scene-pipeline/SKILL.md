---
name: ai-db-scene-pipeline
description: Build a whole Blender scene from a short brief with ai-db's pipeline - write a SceneSpec, let the solver place everything, build it with real furniture and materials, validate, then keep versions with snapshot/restore. Fastest reliable path for rooms, outdoor corners and rows of furniture.
license: MIT
metadata:
  author: ai-db
  version: "1.0.0"
  domain: blender
  role: workflow
  triggers: build a scene, layout a room, interior scene, place furniture, scene from brief, scene spec, solve layout, apply layout, validate scene, snapshot scene, restore scene, scene versions, classroom rows, grid of desks
  related-skills: blender-material-library, pbr-shader-builder, archviz, environment-artist
---

# ai-db scene pipeline

The model writes data (a SceneSpec); deterministic code places, builds and checks it.

## Workflow

```
get_scene_spec_schema(brief) -> write the spec -> submit_scene_spec (fix every error)
  -> build_scene(spec) -> read its report -> snapshot_scene -> record_run
```

**`build_scene` is the default.** It lays the spec out and, on a conflict, repairs the spec itself
(drops the named conflicting relation, else enlarges the setting, else places one copy fewer;
every change is listed in `repairs`), then builds (room, real furniture, materials, daylight, lamp
lights, eye-level camera), validates, and applies the validator's own fixes (`fixes_applied`).
Only use the step-by-step tools below when you need control over one stage.

Every build also probes its exposure: a tiny render from the camera, the view exposure stepped
(at most 3 rounds) until the image is neither black nor blown out. The report's `exposure` gives
`rounds, mean, dark_frac, bright_frac, adjusted_by`; still out of band, an `exposure` issue is
left (usually the camera sits in a wall or behind furniture). `exposure_check: false` skips it.

Optional `build_scene` arguments:
- `poly_budget` (total faces): the report's `polys` lists faces per object; over the budget the
  heaviest non-hero objects get a Decimate modifier until the scene fits (`polys.decimated`).
- `physics: true`: rigid bodies on every mesh, passive (walls, floor, furniture) unless the
  object spec says `dynamic: true` (active, convex hull). Validation is unchanged.
- `judge: true`: renders the build camera (`render`, or `render_path`) and returns `critique`:
  open validator issues (missing objects included), repairs, failed commands, a missed poly
  budget, blocking first, each with a `suggest`ed edit_scene step. Look at the render, fix with
  `edit_scene`, build again until the critique has nothing blocking.

1. **`get_scene_spec_schema(brief)`** returns the schema, the category vocabulary with typical
   sizes, the relations, and exemplar specs from past successful runs. Use only listed categories.
2. **Write the spec**:
   - `setting`: `kind` indoor|outdoor|tabletop|single_asset, `bounds_m` [x, y, height],
     `shape` rect|ellipse|polygon, optional `style` (modern|scandinavian|industrial|rustic|classic:
     a material palette for floor, walls and furniture), optional `floor_material` / `wall_material`
     (explicit materials win over the style); any material may also be `pbr:<preset>` (worn_metal, rusty_steel,
     aged_brass, weathered_wood, scuffed_leather, dusty_plaster, worn_concrete): a library
     material with the pbr-shader-builder grunge layer.
   - Several rooms (indoor rect): `rooms: [{id, offset_m: [x, y], size_m: [w, d]}]` inside
     `bounds_m`; every object then gets `room: <id>` and its relations stay in that room (walls
     are that room's wall_north..). Rooms sharing a wall get a door; each is solved on its own.
   - `objects`: `id`, `category`, `count`, `hero`, `material` (see blender-material-library),
     `grid: {rows, cols, gap_m}` for rows of desks, tables or racks (solved as one block).
   - `relations`: `against_wall` (wall_north/south/east/west), `in_front_of` (A faces B at
     `gap_m`), `beside`, `facing`, `centered_in` (a room or a rug: on the rug's footprint), `on`,
     `under`, `around` (chairs around a table), `above` (a wall-mounted object straight over
     another, `gap_m` from its top, default 0.25: mirror above a sink, whiteboard above a desk).
3. **`submit_scene_spec`** returns exact errors; fix all of them and resubmit.
4. **`solve_layout`** returns placements, or the conflicting relations: loosen them (bigger room,
   fewer objects, drop a duplicate constraint), never ignore them.
5. **`apply_layout`** builds the room shell (walls, openings), real furniture where a generator
   exists, textured boxes elsewhere, materials, a key light and a camera. `geometry: "boxes"`
   is the fast blocking-only variant.
6. **`validate_scene(spec)`** checks intersections, floating, sunk, bounds, scale, facing,
   materials, camera, lighting and that every spec object exists. Apply the suggested fixes.
7. **`snapshot_scene(prompt)`** stores transforms and geometry. After an edit, snapshot with
   `parent_scene_id` to make the next version; `restore_scene(id)` puts any version back and
   reports transforms that differ; `fork_scene` branches a variant.
8. **`export_scene(path, format)`** writes the scene as glb, gltf, usd or usdz into the export
   folder (`plugins.scene.export_dir`; other paths are refused); returns size and object count.

## Spec patterns that solve first time

- Seating group: sofa `against_wall`; coffee table `in_front_of` sofa (gap 0.45);
  rug `centered_in` room; table `centered_in` rug; armchairs `facing` the coffee table.
- Dining: table `centered_in` room; chairs with `count` N `around` the table.
- Bathroom: sink `against_wall`; mirror `against_wall` the same wall and `above` the sink.
- Classroom: board `against_wall` wall_north; desks `grid {rows: 3, cols: 4}` `facing` the board;
  teacher desk `in_front_of` the board, gap 1.0.
- Warehouse: racks `grid {rows: 1, cols: 4, gap_m: [3.0, 0]}`; aisles come from the gap.

## Pitfalls

- Chaining `in_front_of` between identical objects to make rows: use `grid`.
- Asking for exact centres in two places (centered_in a rug *and* in_front_of a sofa) used to
  conflict; a rug now only needs the object on its footprint.
- `against_wall` fixes the yaw (back to the wall): an object that must also face something
  needs to be on the wall opposite it.
