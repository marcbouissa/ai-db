# 3D generation pipeline: small-model executor, big-model quality

> **Goal:** a small, cheap agent (Haiku-class) produces 3D scenes and assets in Blender
> whose quality matches what a frontier model (Opus-class) produces when working alone.
>
> **Principle:** a small model fails at 3D for three reasons: weak spatial judgment (sizes,
> placement, overlaps), unreliable `bpy` code, and no self-critique. Each of these is moved out
> of the model and into the system:
>
> | Weakness | Moved into | Owner |
> |---|---|---|
> | Spatial judgment | Dimension priors + a deterministic **layout solver** | code |
> | `bpy` code quality | A vetted, parametric **recipe library** + real assets | code, written once by a big model |
> | Self-critique | **Validators** (geometry) + a sparse **visual judge** | code + big model, 1–2 calls per scene |
> | Learning | **Run records** in scene memory: what worked, scored | ai-db |
>
> The small model only does what it is good at: turning a brief into structured data and
> picking from retrieved options. The big model is used **sparingly** (judge) and **offline**
> (teacher), never in the inner loop.

---

## 1. Pipeline

```
 brief ──► [1 SPEC] ──► [2 RETRIEVE] ──► [3 LAYOUT] ──► [4 BUILD] ──► [5 VALIDATE] ──┐
          small model    ai-db search    solver (code)  small model   code         │
          → SceneSpec    priors, recipes  → Placements   recipes/assets → issues    │
                         assets, exemplars                                          │
                                ▲                                ▲      fix actions  │
                                │                                └───── (loop ≤ N) ◄─┘
                                │                                              │ clean
                                │                                              ▼
                                │                                  [6 JUDGE] render + rubric
                                │                                   big model, 1–2 calls
                                │                                              │
                                └──────────── [7 PERSIST & LEARN] ◄────────────┘
                                              run record + scores → scene memory
                                              ▲
                                  [8 TEACHER, offline] big model writes/repairs recipes,
                                   promoted only after validators + eval pass
```

### Stage 1: Spec (small model, output = data, not code)
The brief becomes a `SceneSpec`, a strict JSON schema validated by code:

```json
{
  "setting": {"kind": "indoor", "bounds_m": [6, 5, 2.8], "style": "scandinavian"},
  "objects": [
    {"id": "sofa", "category": "sofa", "count": 1, "hero": true},
    {"id": "table", "category": "coffee_table", "count": 1},
    {"id": "lamp", "category": "floor_lamp", "count": 1}
  ],
  "relations": [
    {"a": "sofa", "rel": "against_wall", "b": "wall_north"},
    {"a": "table", "rel": "in_front_of", "b": "sofa", "gap_m": 0.45},
    {"a": "lamp", "rel": "beside", "b": "sofa"}
  ],
  "lighting": {"mood": "warm evening"}, "camera": {"focus": "sofa"}
}
```
- The model picks `category` from a closed vocabulary (the categories that have priors or
  recipes), and `rel` from a fixed set (`on`, `against_wall`, `in_front_of`, `beside`,
  `facing`, `centered_in`, `under`, `around`).
- Schema violations go back to the model with the exact error, up to 2 retries.
- Spatial reasoning becomes **classification**, which small models do well.

### Stage 2: Retrieve (ai-db: the existing strength, applied to 3D)
For each object:
- **Dimension prior:** real-world size ranges per category (seat height 0.42–0.48 m,
  door 2.0×0.9 m, …). Stored as data (`ai_db/scene/priors.json`), not hard-coded.
- **Asset:** a matching asset from the Blender asset library (`get_blender_libraries`), or a
  CC0 source such as Poly Haven, ranked by category/style match.
- **Recipe:** if there's no asset, a parametric procedural builder (Stage 4).
- **Exemplars:** past *successful* run records with similar briefs (scene memory search,
  filtered by judge score ≥ threshold).

### Stage 3: Layout (deterministic solver, no LLM)
Relations + priors + room bounds → `Placement` (location, rotation, scale) per object:
- Hard constraints: inside bounds, no overlap (oriented bounding boxes), supported
  (floor or `on` target), wall contact for `against_wall`, clearances (walkway ≥ 0.6 m).
- Soft objectives: `facing`, symmetry, rule-of-thirds relative to the camera focus.
- An unsatisfiable spec returns *which* relation conflicts. The model fixes the spec;
  it never edits coordinates.
- This replaces the current `plan_3d_scene` template (fixed `rounded_box` heroes at
  `x = idx*1.0`, always 4 walls).

### Stage 4: Build (small model selects; code executes)
- `build_object(category, recipe | asset, params, placement)` runs a **vetted recipe**:
  a parametric Python function in `ai_db/scene/recipes/<category>.py` with a JSON-schema for
  its params (e.g. `chair(seat_h, seat_w, leg_style, back_h, material)`), or instantiates
  an asset.
- `execute_blender_code` stays as a last resort. Every free-form call is logged in the run
  record so the teacher can turn repeated patterns into recipes.

### Stage 5: Validate (code; the core of quality for a small model)
`validate_scene(scene_id, spec)` returns a list of issues, each with a suggested fix action:

| Check | Method | Example fix |
|---|---|---|
| Interpenetration | BVH overlap per object pair | move along separating axis |
| Floating / sunk | raycast down from bbox bottom | snap to support |
| Out of bounds | bbox vs room | clamp |
| Scale plausibility | dims vs prior range | rescale |
| Relation satisfied | geometric test per relation | re-solve layout |
| Mesh sanity | non-manifold, flipped normals, zero-area faces | recalc normals / flag recipe |
| Materials | every visible mesh has a material | assign style default |
| Poly budget | per-object and total vs LOD | decimate / lower recipe LOD |
| Camera | hero bbox in frame, not clipped | re-aim |
| Lighting | at least one key light, exposure in range | add/scale |

The loop applies fix actions and re-validates, up to `MAX_FIX_ITERATIONS`. Blocking issues
that survive go to the run record as failures.

### Stage 6: Judge (big model, sparse)
- Render 2–3 low-res Eevee views, then a VLM judge scores them against a fixed rubric
  (composition, realism of proportions, material coherence, lighting, brief fidelity), 1–5 each.
- At most 2 judge calls per scene. The issues it reports are mapped to fix actions or spec edits.
- Pairwise mode for the eval: judge A vs B without knowing which pipeline produced which.

### Stage 7: Persist and learn (ai-db scene memory)
Each run stores a `RunRecord` with the brief, spec, placements, actions taken
(recipe/asset/free code + params), validator reports per iteration, judge scores, token cost
and wall time. Then:
- Runs with a high score become **exemplars** retrieved in Stage 2.
- Recipes accumulate success and failure stats. Recipes that keep failing get flagged for the teacher.

### Stage 8: Teacher (big model, offline, batch)
- Writes new recipes for categories that are missing or failing, and repairs flagged ones,
  using the logged free-form code and validator failures as input.
- A recipe is **promoted** only after it passes the validators on generated param
  sweeps and doesn't regress the 3D eval. This is where frontier-model quality gets
  distilled into reusable code that the small model calls.

---

## 2. Current state → what changes

| Component today | Status | Change |
|---|---|---|
| Scene memory (snapshot/restore/history/search, JSON + sqlite-vec backends) | Works | Add `RunRecord`; filter exemplar search by score |
| Structured actions (`create_primitive`, `set_material`, `add_light`, `set_camera`, …) | Works | Keep; recipes build on them |
| Spatial tools (`raycast`, `check_collision`, `get_bounds`, `measure_distance`, `get_spatial_context`, …: 9 tools) | **Registered, not implemented**. MCP server fails to start (NameError) | Implement first; `validate_scene` is built on them |
| `plan_3d_scene` | Fixed template: every hero is the same box, objects overlap, always indoor | Replace with SceneSpec + solver |
| `plan_3d_asset` | LOD budgets + questions only | Replace with recipe parameter schemas |
| Blender transport | **Two paths**: ox-bridge HTTP MCP (`blender_oxbridge_capture.py`, :9878) and raw socket (`ext/blender_client.py`, :9877/9878) | Pick one (decision D1) |
| Quality measurement | None | 3D eval suite (Phase 6) |

---

## 3. Implementation phases

Rules: as in `TODO_NEXT.md` §0 (no fallbacks, one code path, tunables in
`ai_db/constants.py`). Blender-dependent tests are marked `@pytest.mark.blender` and skipped in
CI. Every pure-Python part (schema, solver, validators' math, scoring) is unit-tested without
Blender.

### Phase 0: Make the base work (blocks everything)
- [ ] **0.1** Decide and implement a single Blender transport (D1). Remove the other.
  **Check:** one client module; `get_scene_info` works live; the transport tests pass.
- [ ] **0.2** Implement the 9 registered-but-missing spatial handlers in
  `ai_db/mcp_tools/scene_tools.py`. **Check:** the 7 currently failing scene/transport tests
  pass; each tool has a live test (`-m blender`).
- [ ] **0.3** `restore_scene` applies the state to Blender (if it only returns data today).
  **Check:** snapshot → clear → restore round-trip is identical (object transforms ±1e-4).

### Phase 1: Spec + priors
- [ ] **1.1** `ai_db/scene/spec.py`: `SceneSpec`, `ObjectSpec`, `Relation` dataclasses + JSON
  schema + validator that returns precise errors. **Check:** fixtures of valid and invalid specs.
- [ ] **1.2** `ai_db/scene/priors.json`: about 60 common categories (furniture, architecture,
  lighting, props, vegetation), each with min/typical/max dims and support type.
  **Check:** a loader validates every entry, and every category appears in the spec vocabulary.
- [ ] **1.3** MCP tool `make_scene_spec(brief)` returns schema + vocabulary + exemplars for the
  model to fill in; `submit_scene_spec(spec)` validates it. **Check:** an invalid spec returns
  actionable errors.

### Phase 2: Layout solver
- [ ] **2.1** `ai_db/scene/layout.py`: OBB placement with hard constraints + soft objectives
  (greedy placement by dependency order, then local search). Pure Python/numpy.
  **Check:** on 20 synthetic specs: zero overlaps, every object supported and in bounds, and
  every satisfiable relation satisfied. Unsatisfiable specs name the conflicting relation.
- [ ] **2.2** Retire `plan_3d_scene` / `plan_3d_asset` in favour of `spec → layout`.
  **Check:** the old tools are removed from the MCP list; no dead code is left.

### Phase 3: Recipes and assets
- [ ] **3.1** Recipe interface (`ai_db/scene/recipes/base.py`): name, category, params JSON
  schema, `build(params) -> bpy code or action list`, declared bbox function, version.
- [ ] **3.2** Teacher-written seed set: 15 recipes (chair, table, sofa, shelf, bed, lamp,
  door, window, stairs, rug, plant, crate, column, railing, cabinet).
  **Check:** param sweep (50 samples each) gives valid meshes, a bbox within the declared bbox,
  and validators clean.
- [ ] **3.3** Asset resolver: Blender library first, then CC0 provider. Each asset is tagged
  with category and its dims are measured. **Check:** resolver returns ranked candidates with
  measured dims.
- [ ] **3.4** MCP tool `build_object(...)`. **Check:** builds a recipe or asset at its
  placement in a live scene.

### Phase 4: Validation loop
- [ ] **4.1** `validate_scene` with the checks in Stage 5, on top of the Phase 0.2 tools.
  **Check:** a seeded-defect scene (overlap, floating, wrong scale, missing material,
  off-camera hero) reports every defect, and nothing in a clean scene.
- [ ] **4.2** Fix-action executor + loop (`MAX_FIX_ITERATIONS` in constants).
  **Check:** the seeded-defect scene is clean after the loop, in ≤ 3 iterations.

### Phase 5: Judge + memory
- [ ] **5.1** Render views + rubric judge (judge model configurable, D2).
  **Check:** fixed rubric prompt; scores and issues are parsed into structured data.
- [ ] **5.2** `RunRecord` in scene storage (both backends + conformance), exemplar retrieval
  filtered by score. **Check:** round-trip; exemplar search never returns failed runs.

### Phase 6: 3D eval (proves the goal)
- [ ] **6.1** `eval/golden/scenes.jsonl`: 20 briefs (8 interiors, 6 exteriors, 6 single
  hero assets), each with a must-have checklist.
- [ ] **6.2** Runner: for each brief, (A) small model + pipeline vs (B) big model alone with only
  the raw tools. Metrics: validator-clean rate, checklist hits, blind pairwise judge
  win/tie rate, tokens, cost, wall time.
  **Target:** A ≥ B on win+tie in ≥ 50% of briefs, at ≤ 30% of B's cost.
- [ ] **6.3** Baseline saved; regression gate (manual or nightly, since it needs Blender).

### Phase 7: Teacher loop
- [ ] **7.1** Batch job: collect free-form code + recipe failures from run records, and have the
  big model propose recipes or repairs. **Check:** a proposal is promoted only when 3.2's sweep
  and 6.2's eval don't regress.

**Order:** 0 → 1 → 2 → 4 (validators on primitives) → 3 → 5 → 6 → 7. Validators come before
recipes so every recipe is checked from day one.

---

## 4. Decisions taken (2026-10-03)

- **D1:** the only transport is the `blender_ox` addon socket (port 9877, ITC main-thread queue)
  from `~/GitRepos/blender_mcp`. ai-db's ox-bridge HTTP path (:9878) is removed.
- **D2:** the judge model is configurable (`scene.judge_model`) and injected as a client, so
  tests use a fake judge.
- **D3:** assets come from the Blender asset library first. CC0 download is a later plug-in.
- **D4:** judge renders use Eevee at low resolution.

### Repo split
| `~/GitRepos/blender_mcp` (runs inside Blender) | `ai-db` (orchestration) |
|---|---|
| One handler registry (`blender_ox/registry.py`), socket → ITC → main thread | Blender client (:9877 only), MCP scene tools |
| Spatial queries + geometric validators (`verify.py`) | SceneSpec, priors, layout solver |
| Shape-operation library, asset builder, recipes (`geometry.py`, `shapes.py`) | Asset spec authoring, retrieval, RunRecord, exemplars |
| Materials, cheap renders (`materials.py`, `render.py`) | Judge, fix loop orchestration, 3D eval, teacher |

Blender-side tests run headless: `blender -b --factory-startup --python <test script>`
(Blender 5.2.1 at `~/bin/blender`).

### Original decision list (kept for reference)

- **D1, Blender transport:** ox-bridge (HTTP MCP, :9878, `get_scene_info`) vs the raw socket
  client (`ext/blender_client.py`). Recommendation: ox-bridge if it supports executing code
  and the spatial queries; keeping two leads to "works in one, fails in the other" bugs.
- **D2, judge model:** a big model with vision, called at most twice per scene. Pairwise
  judging in the eval must be blind.
- **D3, asset sources:** Blender asset library only, or also CC0 downloads (Poly Haven).
  Downloads need caching and licence tagging.
- **D4, render engine for judging:** Eevee (fast, default) vs Cycles (slow, realistic).

---

## 5. Judge options and the local-model plan

**Now:** the judge is the interactive Claude Code session (a person in the loop). The session
renders the probe views, looks at them, and records scores and issues through `record_run`.
There's no API key and no extra cost. This doesn't work unattended.

**Later: local vision model (free, unattended).** Plan:

1. **Runtime:** Ollama (or llama.cpp server) on the local 24 GB GPU, OpenAI-compatible
   endpoint `http://localhost:11434/v1`. Candidate models: Qwen2.5-VL 7B, Gemma 3 12B
   (vision). Both fit next to Blender's Eevee renders.
2. **Code:** `LocalVLMJudge` in `ai_db/scene/judge.py`, implementing the same `Judge` protocol
   (`score`, blind `compare`). Config: `scene.judge = {"provider": "local", "base_url",
   "model"}`. Same rubric and same strict-JSON parsing with one repair retry. Raise
   `AiDbConfigError` if the endpoint is unreachable; no silent fallback.
3. **Make a weak judge reliable:** score one criterion per call (5 short prompts instead of
   one long one); use 3 views and take the median; give anchored examples in the prompt
   (a 1-score and a 5-score reference render per criterion).
4. **Cheap pre-filter:** a local CLIP/SigLIP brief-to-image similarity score, to reject
   obviously off-brief renders before calling the VLM.
5. **Calibrate before trusting it:** have the local judge and the session judge (or the Claude
   API judge, if enabled) score the same 30 renders. Report rank correlation per criterion.
   Use the local judge as a gate only on criteria with correlation ≥ 0.6. The others stay
   advisory.
6. **Eval (Phase 6):** local judge for nightly regression runs. A stronger judge only for
   the headline Haiku-vs-frontier comparison.
