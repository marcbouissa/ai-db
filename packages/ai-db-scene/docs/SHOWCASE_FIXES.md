# Oval Office showcase: gap analysis and fixes

Reference: a press photo of the Oval Office (gold swag curtains, damask sofas, Resolute desk,
seal rug, plaster ceiling). Our render: `showcase/oval_office_out/oval_office_main.png`.
Each fix is a **system** capability (it helps every scene), not a one-off tweak to this scene.
Impact is on perceived quality vs the reference: **H** high, **M** medium, **L** low. Effort is **S/M/L**.

## A. Correctness bugs (fix first)

| # | Problem | Root cause | Fix | Impact | Effort |
|---|---|---|---|---|---|
| A1 | Sofas and armchairs face the walls, not each other | Yaw was set by hand. Nothing turns a `facing` relation into a rotation or checks it. The front = −Y convention isn't enforced. | The layout solver computes yaw from `facing` / `in_front_of` / `against_wall` (back to wall). `validate_scene` gets a `facing` check (front vector vs target, angle ≤ 20°). `build_asset` stores the front axis as a custom property. | H | S |
| A2 | Shape schema advertises `size`/`profile` as strings | `gen_schema` misses `list` params on shapes | Typed arrays in `asset_spec.schema.json` + freshness test | M (small models get confused) | S |
| A3 | `support` vocab differs (`surface`/`ground` in ai-db vs `floor/wall/ceiling` in builder) | Two vocabularies | One shared enum: floor, surface, wall, ceiling, ground. The builder accepts all of them; priors use the same. | M | S |
| A4 | `add_light` can't aim lights | No rotation / target param | `rotation` + `target` (track-to) params | M | S |
| A5 | Wall-mounted items flagged as colliding with their wall | Validator has no host-contact rule | Objects with `support=wall` may touch (≤ 2 cm) their host wall | M | S |
| A6 | Curved rooms are custom code | Spec/solver handle rectangles only | `setting.shape`: rect, ellipse, polygon. Solver bounds/walls for each. Builder op `room_shell(shape, height, thickness, openings[])` with boolean openings | H (unlocks non-box rooms) | M |

## B. Cloth looks like frozen planes (flags, curtains)

| # | Fix | Impact | Effort |
|---|---|---|---|
| B1 | Procedural `drape` shape: a pinned top edge, vertical folds (sine + noise displacement along X, depth decaying toward the pin line), a slight puddle at the floor, and thickness from solidify. Parameters: width, height, fold_count, fold_depth, gather (0–1). | H | M |
| B2 | `swag_valance` shape: catenary swags along the curtain rod, a sweep with a varying profile, plus jabots (cascade tails) at the ends. This is the gold valance in the reference. | H | M |
| B3 | `flag` shape: cloth hanging from a pole with gravity folds (B1 rotated) + fringe (array of small cylinders) + finial + cord with tassels | M | S |
| B4 | Optional real cloth: a headless bake of Blender's cloth simulation (pin group on the top edge, about 60 frames) for hero curtains. Deterministic seed, cached result. | M | M |

## C. Furniture detail (Resolute desk "looks like a box", sofas, chairs)

| # | Fix | Impact | Effort |
|---|---|---|---|
| C1 | Molding profiles library: ogee, bead, cove, dentil, rope. `extrude_profile` / `sweep` take a named profile, so panel moldings, cornices and desk edges get real profiles. | H | M |
| C2 | Panel op `raised_panel(size, border, profile)` for desk fronts, cabinet doors and wainscoting | H | S |
| C3 | Carving / relief via displacement from a grayscale image or procedural pattern (`relief(image, depth)`): desk front carving, plaster ceiling ornaments, the seal | H | M |
| C4 | Upholstery ops: `cushion` (rounded slab + subdivision + edge bulge + optional piping sweep), `rolled_arm` (sweep of a scroll profile), `skirt` (drape, B1, around the base), `tufting` | H | M |
| C5 | Leg library: turned (lathe profiles), tapered square, cabriole (sweep with scale curve), claw foot (from the asset library) | M | S |
| C6 | Recipes rebuilt on C1–C5: Resolute desk (kneehole panel with seal, carved pedestals, leather top), Chippendale armchair (wooden frame, upholstered seat), English roll-arm skirted sofa, drum side table with drawer | H | M |

## D. Materials and patterns (currently flat colours)

| # | Fix | Impact | Effort |
|---|---|---|---|
| D1 | Procedural pattern materials: damask (fabric and wallpaper), stripes, wood grain with figure, plaster, brass, leather, velvet sheen. Parameters: base colour, pattern colour, scale. | H | M |
| D2 | `apply_image_texture(name, image, projection, scale)` + `decal(name, image, at, size)` for flags (stars and stripes), portraits, the rug seal, and the desk seal | H | S |
| D3 | Emblem source: generate a **stylised** seal procedurally (ring text-free, star ring, eagle silhouette from an SVG path), or use a public-domain image downloaded once with your approval and cached in the asset library | M | S |
| D4 | Rug material: a radial gradient field + seal decal + patterned border ring (D1 + D2) | H | S |

## E. Room architecture

| # | Fix | Impact | Effort |
|---|---|---|---|
| E1 | Crown molding: sweep a profiled cornice (C1) + dentil array along the wall top | H | S |
| E2 | Ceiling: plaster relief border + central seal relief (C3), plus a soft cove light (emissive strip behind the cornice) | H | M |
| E3 | Arched wall niches with built-in bookshelves and books/ornaments (boolean niche in `room_shell` + shelf recipe) | M | M |
| E4 | Windows: multi-pane grid (6-over-6 etc.) as a window recipe parameter, deep reveals, an exterior backdrop (garden HDRI or a green gradient plane) | M | S |
| E5 | Doors with pediments/architraves (C1) | L | S |

## F. Set dressing

| # | Fix | Impact | Effort |
|---|---|---|---|
| F1 | Wall art: framed portraits (frame from C1 + D2 image), placed by the solver on wall relations at eye height | M | S |
| F2 | Small props: busts, statuettes, books, vases, desk items. Asset library first, lathe/sculpt fallback. | M | M |
| F3 | Extra flags (military branch flags) + the chairs in front of the desk | M | S |

## G. Lighting and render

| # | Fix | Impact | Effort |
|---|---|---|---|
| G1 | Daylight rig: Nishita sky or HDRI + sun through the windows + window portal area lights + warm lamp emission | H | S |
| G2 | Final render in Cycles with denoising (Eevee stays for judge probes) | H | S |
| G3 | Camera presets matched to the brief/reference ("eye level from the fireplace, 18 mm") + a `match_reference_camera` helper | M | S |

## H. Pipeline (makes all the above reachable by a small model)

| # | Fix | Impact | Effort |
|---|---|---|---|
| H1 | Reference-image mode: the judge compares the render with a user-provided reference photo (pairwise), and returns issues mapped to the categories above | H | M |
| H2 | Detail level in specs: `detail: draft|standard|hero`. Hero assets automatically get molding profiles, upholstery ops and pattern materials. | H | S |
| H3 | Recipes for this showcase are stored as exemplars (RunRecord), so the next "formal office" brief starts from them | M | S |

## Suggested order
1. **A1–A6** (bugs, mostly small; A1 and A6 matter most)
2. **G1–G2 + D1–D2 + D4** (biggest visual jump for the least effort)
3. **B1–B3 + C1–C2 + E1** (cloth and profiles)
4. **C3–C6 + E2** (carving, upholstery, recipes)
5. **H1–H3**, then **E3–E5, F1–F3, B4**

Then rebuild the showcase from a *brief + reference photo* through `build_scene` and compare against
the reference with the judge.
