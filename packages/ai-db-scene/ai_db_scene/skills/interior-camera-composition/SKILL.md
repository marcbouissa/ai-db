---
name: interior-camera-composition
description: Frame interior and architectural renders like a photographer - camera height, focal length, level camera with lens shift for straight verticals, corner (two-point) or straight-on (one-point) views, thirds. Use when placing a camera, when walls lean, rooms look distorted or fish-eyed, or the subject is badly framed.
license: MIT
metadata:
  author: ai-db
  version: "1.0.0"
  domain: blender
  role: guide
  triggers: camera placement, frame the room, interior shot, architectural render, walls leaning, keystoning, distorted, fisheye, focal length, camera height, composition, one-point perspective, two-point perspective
  related-skills: ai-db-scene-pipeline, realistic-modeling
  sources:
    - {"url": "https://www.hdrsoft.com/learn/best-height-for-shooting-real-estate-interiors.html", "title": "Best height for shooting interiors"}
    - {"url": "https://architizer.com/blog/practice/tools/the-art-of-rendering-perspectives/amp/", "title": "7 rules for composing architectural perspectives"}
    - {"url": "https://imagen-ai.com/post/architectural-photography/", "title": "Architectural photography"}
---

# Interior camera composition

`build_scene` already does the basics: a level camera at 1.4 m in the clearest corner on a
20 mm lens, framed with lens shift, a coverage check that the heroes are visible, and an
exposure check. Use this when you place or adjust a camera yourself (`set_camera`).

## Height

- 1.2-1.5 m (chest height) for most rooms; slightly lower in kitchens so the underside of
  wall cabinets is not shown. Do not go above head height: it reads as a security camera.
- Furniture does not get taller in tall rooms, so the height stays the same.

## Lens

- 16-35 mm (full-frame equivalent). Wider than ~14-16 mm stretches the corners and makes
  furniture near the edges look smeared. 24-28 mm for a calmer, more natural view; 35-50 mm
  for a detail or a vignette.

## Keep verticals vertical

- Keep the camera level (rotation X = 90 in Blender). Tilting up or down makes walls and door
  frames converge (keystoning) - the most common "this is a render" giveaway.
- To show more floor or ceiling, shift the lens instead of tilting: camera data `shift_y`
  (negative = frame lower). A shift of up to about 0.3 keeps distortion invisible.

## Composition

- Two-point (corner) view: shoot into a corner so two walls recede; most rooms read best
  this way.
- One-point (straight-on): sensor parallel to the back wall, centred, for symmetric rooms
  (a fireplace wall, an oval office desk under its windows).
- Put the hero on a third, leave some floor in the foreground, and do not cut furniture in
  half at the frame edge.
- Real rooms are a little imperfect: a chair pulled out, a throw over an arm.

## Checks

- Verticals parallel to the frame edges.
- The hero is fully in frame and not hidden (build_scene's `camera.visible_heroes`).
- Nothing very close to the lens (a blurred armrest filling a corner).
- Exposure in range (`build_scene`'s `exposure.in_band`).
