---
name: blender-material-library
description: Texture objects quickly with the Blender addon's procedural material library (wood, marble, metal, fabric, leather, velvet, plaster, ceramic, glass, brass, gold, chrome, concrete, stone, foliage, plastic, paint, rubber) through apply_material, with tints, scale and projection, and know when to switch to image-based PBR.
license: MIT
metadata:
  author: ai-db
  version: "1.0.0"
  domain: blender
  role: specialist
  triggers: apply material, procedural material, texture object, wood material, marble, metal material, fabric, leather, velvet, plaster walls, glass, brass, chrome, tint color, box projection, texturing
  related-skills: pbr-shader-builder, ai-db-scene-pipeline, materials
---

# Blender material library

`apply_material(name, material)` builds (or reuses) a procedural node material and makes it the
only material of the object, and of every part below an asset root. No texture files, no UVs
needed with box projection.

## When to Use

- Any object needs a believable surface now (placeholders, generated furniture, room shells).
- A spec or brief names a material ("walnut table", "brass lamp", "linen sofa").
- Use **pbr-shader-builder** instead when you have texture files or need a specific photo look.

## Vocabulary

`kind[:variant][:#base[:#pattern]][:scale=N][:proj=uv|xy|xz|yz|box|object]`

| kind | variants | good for |
|---|---|---|
| wood | oak walnut mahogany cherry pine dark | furniture, floors, doors |
| marble | carrara calacatta verde | counters, fireplaces, columns, floors |
| metal | steel copper chrome rust brushed | appliances, racks, fixtures |
| brass, gold, chrome | – | lamps, hardware, trim |
| fabric | weave linen denim canvas | sofas, beds, rugs, curtains |
| leather, velvet | – | armchairs, club chairs |
| plaster | – | walls, ceilings |
| ceramic | – | sinks, toilets, vases, pots |
| concrete | – (tint `:#rrggbb`) | floors, docks, planters, kerbs |
| stone | – | rocks, fountains, walls |
| foliage | – | trees, bushes, hedges, lawns (slight translucency) |
| plastic | – | electronics, cones, toys, whiteboards |
| paint | – | car bodies, forklifts, painted metal (clear coat) |
| rubber | – | tyres, mats, bumpers |
| glass | clear frosted tinted | windows, shower walls, mirrors |
| color | needs `#rrggbb` | plain paint, plastics, screens |

- **Tint:** `fabric:linen:#6b7f99` (dusty blue linen), `plaster:#e8e0d0` (warm white).
- **Scale:** repeats per metre, e.g. `wood:oak:scale=3` for a finer grain on small objects.
- **Projection:** `proj=box` for placeholders, boxes and architecture; `proj=uv` only for
  unwrapped meshes.

## Recipes

- Living room: floor `wood:oak:proj=box`, walls `plaster:#ece6dc`, sofa `fabric:linen:#6b7f99`,
  coffee table `wood:walnut`, lamp `brass`, rug `fabric:weave:#b49a7a`.
- Kitchen: counters `marble:carrara`, fronts `color:#f2f0ea`, fridge `metal:brushed`.
- Warehouse: racks `metal:steel`, barriers `paint:#e0a000`, floor `concrete:#8a8a86`, crates `wood:pine`.
- Bathroom: tub and toilet `ceramic`, shower `glass:frosted`, walls `marble:carrara`.

## Through ai-db

In a SceneSpec give `objects[].material`, `setting.floor_material`, `setting.wall_material`;
`apply_layout` applies them (box projection) and uses a sensible default per category for
anything left out. Generated furniture (sofa, armchair, chairs, tables, desk, bookshelf) keeps
its own multi-part materials.

## Checks

- `validate_scene` reports meshes with no material.
- Render a probe (`render`, 16 samples): plastic-looking wood means the object is tiny and needs
  `scale=`; stretched grain on a box means `proj=box` is missing.

## Why flat colour looks fake

`color:#rrggbb` is a single even colour and sheen, the tell-tale of a placeholder. Real
surfaces vary in colour, roughness and relief even when new. Prefer a kind with its own
variation (concrete, stone, foliage, plastic, paint, rubber) and tint it: `paint:#7a1f1f`
rather than `color:#7a1f1f`. An addon loaded before these kinds existed gets the flat
colour automatically.
