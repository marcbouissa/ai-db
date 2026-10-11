---
name: cloth-and-textiles
description: Make curtains, clothes, throws, tablecloths and flags drape like fabric instead of looking stiff - generator presets with folds, then a short cloth simulation (low stiffness, pinned edge, colliders) applied before rendering. Use when textiles look stiff, flat, rigid or like cardboard.
license: MIT
metadata:
  author: ai-db
  version: "1.0.0"
  domain: blender
  role: guide
  triggers: clothes too stiff, stiff fabric, cloth looks rigid, drape, curtains, cloth simulation, tablecloth, throw blanket, flag, garment, folds, wrinkles
  related-skills: realistic-modeling, ai-db-scene-pipeline, blender-material-library
  sources:
    - {"url": "https://docs.blender.org/manual/en/latest/physics/cloth/settings/index.html", "title": "Blender manual: cloth settings"}
    - {"url": "https://docs.blender.org/api/2.83/bpy.types.ClothSettings.html", "title": "bpy ClothSettings"}
---

# Cloth and textiles

Fabric reads as fake when it has straight edges, even thickness and no folds. The addon's
textile generators (`generate_asset` family `curtain`, `clothes`, `flag`, `industrial_curtain`)
model folds procedurally; garments are soft cushion panels, which look stiff up close. For
anything the camera sees closely, finish with a short cloth simulation.

## 1. Start from a generator preset with folds

- Curtains: `fold_count` 8-14, `fold_depth` 0.04-0.09 m, `puddle` for a break on the floor;
  presets `velvet_theatre`, `modern_track`, `sheer_single`, `classic_damask`.
- Clothes: presets `tshirt_on_hanger`, `denim_on_hanger`, `dress_on_hanger`, `boutique_rack`,
  `folded_tees`. Vary `seed` so a rack is not twelve identical shirts.

## 2. Drape it with a cloth simulation

1. The cloth mesh needs enough vertices to bend: subdivide to roughly 2-3 cm edges first.
2. Colliders: `add_collision` on what it rests on or hangs from (hanger, rod, table, body),
   thickness slightly above the fabric's.
3. `add_cloth(name, quality, mass, stiffness, pin_group)`:
   - stiffness (tension/compression/shear) is what makes fabric stiff: 5-15 for cotton and
     linen, 1-5 for silk and sheers, 20-40 for denim and canvas, 40+ for leather.
   - Higher bending stiffness gives fewer, larger folds; low bending with high tension gives
     stiff sheets with no folds - change them together.
   - Pin the edge that hangs (curtain top, garment shoulders on the hanger) with a vertex
     group; leave the rest free.
   - quality 5 while trying, 8-10 for the final; raise it if cloth passes through colliders.
   - Air damping helps loose fabric settle instead of swinging.
4. Run the simulation for 40-80 frames so it settles, then apply the cloth modifier at a
   settled frame, so a still render does not depend on a cache.
5. Real-world scale matters: cloth on a mis-scaled object behaves wrongly.

## 3. Material

`fabric:linen`, `fabric:canvas`, `fabric:denim`, `velvet`, `damask:fabric`; tint with
`:#rrggbb`. Sheers: low-alpha or `glass:frosted`-like translucency, never opaque.

## Checks before finishing

- No straight bottom hem on a hanging curtain: it should wave with the folds.
- Garments: sleeves fall, the shoulder line follows the hanger, no flat rectangles.
- Nothing intersects: cloth inside a collider means the collider was missing or too thin.
- If a user reported stiff textiles (`record_feedback`), build_scene will remind you: verify
  the drape in a render before calling it done.
