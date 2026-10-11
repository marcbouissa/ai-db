---
name: realistic-modeling
description: Make Blender furniture and props look real instead of CG - rounded edges, real-world proportions, soft upholstery shapes, uneven materials and believable interior light. Use when modelling or reviewing assets, writing part recipes, or when renders look fake, flat or toy-like.
license: MIT
metadata:
  author: ai-db
  version: "1.0.0"
  domain: blender
  role: guide
  triggers: realistic model, looks fake, looks cg, photoreal furniture, bevel edges, furniture proportions, natural looking, realistic materials, realistic interior render, create_recipe
  related-skills: ai-db-scene-pipeline, blender-material-library, pbr-shader-builder, recipe-from-text
  sources:
    - {"url": "https://blenderartists.org/t/couch-leg-bevel-modelling/1279294", "title": "Couch leg bevel modelling (forum)"}
    - {"url": "https://imeshh.com/blog/blender-28-pro-sofa-modeling-guide-part-3-main-sofa-frame-mo", "title": "Pro sofa modelling guide, part 3"}
    - {"url": "https://blenderartists.org/t/who-has-tips-for-photo-realism-in-cycles/1132314", "title": "Tips for photorealism in Cycles (forum)"}
    - {"url": "https://blenderartists.org/t/interior-render-feedback-and-tips/1621732", "title": "Interior render feedback (forum)"}
    - {"url": "https://roomstogo.com/inspiration/how-to-measure-furniture", "title": "How to measure furniture"}
    - {"url": "https://arcedior.com/blog/coffee-table-size-guide", "title": "Coffee table size and clearance"}
    - {"url": "https://arxiv.org/pdf/2306.09310", "title": "Infinigen: recursive paths + space colonization trees"}
    - {"url": "https://github.com/IRCSS/Trees-With-Geometry-Nodes-Blender", "title": "Trees with geometry nodes"}
---

# Realistic modelling

Why a model reads as CG, in order of how much it costs you: knife-sharp edges, wrong
proportions, hard boxes where things should be soft, perfectly even materials, and light
that no room would have. Fix them in that order.

## 1. No sharp edges

Real objects have no perfectly sharp 90-degree edge; even a 1 mm round catches a highlight
and gives the edge a visible line of light. Without it, edges render as flat, uniform seams.

- Part recipes: ai-db rounds every `box` part automatically (about 6 mm on the built object,
  at most a quarter of the part's thinnest side). Set `bevel` on a part to override it, e.g.
  a 2 cm round on a chunky tabletop, `bevel: 0` for a deliberately crisp edge.
- Keep it small: a large bevel turns a frame into a pillow. 2 segments are enough.
- Upholstery: bevel first, then soften; keep seam lines crisp (they are where fabric wraps
  tight around the frame).

## 2. Real-world proportions

Wrong proportions are the second giveaway. Typical ranges (retail guides; they vary by
market, so treat them as ranges, not standards):

| item | key size |
|---|---|
| sofa | seat height 43-48 cm, seat depth 50-60 cm, overall depth 80-100 cm |
| armrest | slim track arm under 10 cm wide; rolled arms up to 25 cm |
| coffee table | top at seat height or up to 5 cm below; 40-45 cm from the sofa |
| side table | level with the armrest or up to 5 cm below |
| dining chair | seat 45 cm, 40-50 cm deep; 90-100 cm from table edge to wall behind |
| dining table | 75 cm high |
| walkway | 76-91 cm along main paths |

ai-db's category priors (`get_scene_spec_schema`) carry typical sizes; stay inside them.

## 3. Soft things are not boxes

The addon has shapes for soft goods: use them in `create_recipe` parts instead of boxes.

- `cushion` (crown, bulge, piping) for seat and back cushions, beds and pillows.
- `rolled_arm` for classic sofa arms; `tufted_panel` for buttoned backs and headboards.
- `rounded_slab` for mattresses, upholstered seats and soft tops.
- `turned_leg`, `cabriole_leg`, `square_tapered_leg` for furniture legs; `skirt` for valances.
- `raised_panel` for cabinet doors; `cornice` / `sweep` with a moulding `profile` for trims.

## 3b. Plants and trees

- A tree is a skeleton, not a ball on a stick: the trunk tapers (flared at the base) and
  splits where the crown starts; limbs branch recursively, each level thinner and shorter,
  growing up and out with some jitter, thin ends drooping; leaves sit in clumps at the tips
  with gaps where sky and branches show through. `build_scene` grows trees this way (a
  different tree per instance) and scatters leaf cards over the clumps.
- Bushes, hedges and potted plants: uneven lobes in `foliage`, covered with leaf cards.
- Bark is rough and dark (`stone:#4b3b2e` reads better than wood grain, which looks striped).

## 4. Uneven materials

Nothing real is evenly coloured or evenly rough, even when new.

- Prefer a material kind with its own variation over flat `color:#...`: `paint:#7a1f1f` for
  a car body, `plastic:#111111` for a TV, `foliage` for plants, `stone` for rocks,
  `concrete` for floors and docks (see blender-material-library).
- Add wear where hands and feet touch: `pbr:worn_metal`, `pbr:weathered_wood`,
  `pbr:scuffed_leather` (see pbr-shader-builder).
- Glass and metal look fake when pristine: a little smudge or dust helps.

## 5. Believable light

- Daylight through windows plus a few practical lamps; keep the sky/HDRI modest so it does
  not overpower the room's own lights.
- Small windows: use light portals (the build adds them) so Cycles finds the sky.
- Check exposure: `build_scene` measures a probe render and corrects it; a black or blown-out
  render scores worst with any judge.
- Final images: `build_scene(render: {"engine": "cycles", "sky": true})`.

## Pitfalls

- Uniformly thick parts (every board 4 cm): vary them - tops thicker, backs and shelves thinner.
- Everything perfectly aligned and centred: real rooms have a rug slightly off, a chair pulled out.
- Over-bevelling: rounds bigger than the part's thickness make furniture look inflated.
- Flat colour on large surfaces (floors, walls, car bodies) is where fakeness shows most.
