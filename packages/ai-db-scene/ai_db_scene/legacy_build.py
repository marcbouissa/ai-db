"""Build options of versions saved before they were stored.

Versions built before build_scene stored its options (seed, geometry mode, furniture
variation key) were rebuilt by edit_scene with seed 0, generated geometry and a variation key
hashed from the *edited* object list: every untouched piece of furniture could move or change.
For such a version the options are recovered from what the version itself stored:

- ``variation``: the key the old code hashed, i.e. ``variation_key`` of the version's own spec;
- ``seed``: the layout seed whose placements match the stored object positions;
- ``geometry``: ``boxes`` when every placed object is a plain 8-vertex box, else ``generated``.
"""

from __future__ import annotations

import math
from typing import Any

from ai_db_scene.models import SceneState

#: seeds tried, in order: build_scene's default first, then small and commonly typed ones
CANDIDATE_SEEDS: tuple[int, ...] = (*range(32), 42, 123, 1234, 2024, 2025, 2026)
#: a stored object "is" its placement when its floor position is this close (m); the
#: validator's fixes may nudge a few objects after the build
MATCH_TOLERANCE_M = 0.05
#: share of placed objects that must match for a seed to count as found
MIN_MATCH = 0.6
_BOX_VERTICES = 8


def infer_build_options(spec_data: dict[str, Any], scene: SceneState) -> tuple[dict[str, Any], str]:
    """Options for rebuilding ``scene`` (built from ``spec_data`` without stored options), and a
    note saying what was inferred or why untouched objects may still change."""
    from ai_db_scene.layout import LayoutError, solve_layout
    from ai_db_scene.tools.scene_pipeline import _parse_spec, frame_offset, variation_key

    spec, _errors = _parse_spec(spec_data)
    if spec is None:
        return {}, "this version predates stored build options and its spec no longer parses"
    out: dict[str, Any] = {"variation": variation_key(spec)}
    by_name = {o.name: o for o in scene.objects.values()}
    off = frame_offset(spec)
    best: tuple[float, int] | None = None
    for seed in CANDIDATE_SEEDS:
        try:
            placements = solve_layout(spec, seed=seed)
        except LayoutError:
            continue
        stored = [(p, by_name[p.id]) for p in placements if p.id in by_name]
        if not stored:
            break  # nothing of the build is left to compare (renamed or deleted objects)
        hits = sum(1 for p, o in stored if math.dist(
            (p.location[0] + off[0], p.location[1] + off[1]), o.location[:2]) <= MATCH_TOLERANCE_M)
        score = hits / len(stored)
        if best is None or score > best[0]:
            best = (score, seed)
        if score == 1.0:
            break
    notes = ["this version predates stored build options"]
    if best is not None and best[0] >= MIN_MATCH:
        out["seed"] = best[1]
        notes.append(f"seed {best[1]} inferred ({best[0]:.0%} of objects where its layout puts them)")
    else:
        notes.append("its layout seed could not be recovered from the stored positions: untouched "
                     "objects may move (pass seed= to choose one)")
    geometry = _geometry(scene, {o.id for o in spec.objects}, by_name)
    if geometry is not None:
        out["geometry"] = geometry
        notes.append(f"geometry {geometry!r} inferred")
    return out, "; ".join(notes)


def _geometry(scene: SceneState, ids: set[str], by_name: dict[str, Any]) -> str | None:
    """``boxes`` when every placed object is a childless 8-vertex mesh, ``generated`` when any is
    not, None when no placed object is stored."""
    placed = [o for name, o in by_name.items() if name in ids or name.rsplit("_", 1)[0] in ids]
    if not placed:
        return None
    parents = {o.parent_uuid for o in scene.objects.values() if o.parent_uuid}
    boxes = all(o.type == "MESH" and o.vertices == _BOX_VERTICES and o.uuid not in parents for o in placed)
    return "boxes" if boxes else "generated"
