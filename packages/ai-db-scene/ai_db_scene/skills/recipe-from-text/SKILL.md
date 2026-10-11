---
name: recipe-from-text
description: Turn a furniture product page, spec sheet, tutorial or ShapeAssembly program into an ai-db part recipe - read it as text, draft_recipe, finish the parts, create_recipe with source and license. Never downloads model files.
license: MIT
metadata:
  author: ai-db
  version: "1.0.0"
  domain: blender
  role: workflow
  triggers: recipe from text, import recipe, new part recipe, create recipe, recipe from product page, spec sheet dimensions, shapeassembly, furniture from description
  related-skills: ai-db-scene-pipeline, blender-material-library
---

# Recipe from text

A recipe is a list of parts (boxes, cylinders, spheres, tapered cylinders) in relative units
that `build_scene` scales to each placement. Recipes come from text only.

## Rules

- Fetch the source as TEXT (WebFetch of the page, a pasted spec sheet, a ShapeAssembly program).
- Never download, open or convert model files (.glb, .obj, .blend, .fbx, .usd) for a recipe.
- Record where it came from: `source = {"url": ..., "license": ..., "title": ...}`. If the page
  states no license, write `"license": "unknown (dimensions only)"` and keep only measurements.

## Workflow

1. `draft_recipe(category, text)`:
   - a ShapeAssembly program (`Cuboid`/`attach`/`squeeze`/`reflect`/`translate`) is converted
     directly; read `warnings` for skipped statements (rotate, sub-assemblies);
   - other text gives `dims_m` (W x D x H from cm, mm, inches, feet), `part_words`
     ("4 legs", "three shelves") and a connected skeleton in `parts`.
2. Edit the parts. Units: x and y in [-0.5, 0.5], z in [0, 1]; `at` is the bottom centre of
   the part; front is -Y; materials as in blender-material-library (`wood:oak`,
   `metal:steel`, `fabric:linen`, `color:#rrggbb`).
3. Every part must touch a chain of parts that reaches the floor; `issues` in the draft lists
   what `create_recipe` will refuse (floating parts, outside the unit box, unknown materials).
4. `create_recipe(category, name, parts, dims=dims_m, source={...})`. A category with a
   built-in recipe needs `overwrite=true` to use yours. `list_recipes` shows saved ones.

## Pitfalls

- "Seat height 46 cm" is not the overall height: draft_recipe skips part-qualified sizes;
  check `dims_m` against the page.
- A × B × C without labels is read as width × depth × height; product pages sometimes list
  W × H × D. Fix `dims` if so.
- Categories built by the furniture generator (chair, sofa, tables, bookshelf, desk) keep
  using the generator in `build_scene`; a user recipe for them is stored and returned by
  `recipe_spec` for when the generator is unavailable.

## Example

Text: "Oak bench, W 120 cm x D 35 cm x H 45 cm, four legs and a stretcher."

1. `draft_recipe("bench", text)` -> dims_m [1.2, 0.35, 0.45], part_words {legs: 4, rungs: 1}.
2. Keep leg_0..leg_3, top and rung; thicken the top to 0.1.
3. `create_recipe("bench", "oak-bench", parts, dims=[1.2, 0.35, 0.45],
   source={"url": "https://...", "license": "dimensions only"}, overwrite=true)`.
