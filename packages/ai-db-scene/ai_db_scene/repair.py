"""Automatic repair of an unsatisfiable SceneSpec, one minimal change at a time.

The benchmark showed agents abandoning the pipeline when ``solve_layout`` reported a conflict
(47 solve calls, 18 builds): fixing a spec from a conflict message is exactly what a small model
does badly. ``relax`` turns one LayoutError into one concrete edit and says what it changed:

1. drop the conflicting relation the solver named (positional ones before against_wall / on),
2. otherwise, when the blocker is space (out of bounds, overlaps), give the room more space
   (x/y bounds +20% per step, the floor area at most ``REPAIR_MAX_AREA_GROWTH`` times the
   requested one); a relation conflict is never "fixed" by a bigger room (bench: a picture frame
   with no wall grew a lobby to 20 x 20 m and still failed),
3. otherwise place one instance fewer of the object that could not be placed.

``solve_with_repairs`` runs the whole loop for build_scene: it tries a second seed before it
drops a ``facing`` / ``in_front_of`` relation (the ones judges notice), and lists every dropped
relation so the build can report it instead of calling the scene ``built``.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from ai_db_scene.constants import REPAIR_GROWTH, REPAIR_MAX_AREA_GROWTH
from ai_db_scene.layout import LayoutError, Placement, solve_layout
from ai_db_scene.spec import SceneSpec, validate_spec

_SUFFIX = re.compile(r"_\d+$")
_LABEL = re.compile(r"^(\S+) (\w+) (\S+)")
#: dropped first: they position an object relative to another; walls and support come last
_DROP_ORDER = ("facing", "beside", "around", "in_front_of", "centered_in", "under", "above", "against_wall", "on")
#: blockers a bigger room can actually fix
_SPACE_BLOCKERS = ("out of bounds", "overlaps '", "blocks doorway")
#: relations judges notice when they are gone: another seed is tried before dropping one
_KEEP_HARDER = ("facing", "in_front_of")
_SECOND_SEED = 104729


def _base(iid: str, ids: set[str]) -> str:
    return iid if iid in ids else _SUFFIX.sub("", iid)


def space_bound(err: LayoutError) -> bool:
    """Is lack of space among the reasons the solver gave up?"""
    msg = str(err)
    return "cannot place" in msg and any(b in msg for b in _SPACE_BLOCKERS)


def _drop(data: dict[str, Any], err: LayoutError) -> tuple[dict[str, Any], str, tuple[str, str, str]] | None:
    ids = {o["id"] for o in data.get("objects", [])}
    named: list[tuple[str, str, str]] = []
    for c in err.conflicts:
        m = _LABEL.match(c)
        if m:
            named.append((_base(m.group(1), ids), m.group(2), _base(m.group(3), ids)))
    rels = data.get("relations", [])
    for kind in _DROP_ORDER:
        for a, rel, b in named:
            if rel != kind:
                continue
            for i, r in enumerate(rels):
                if (r.get("a"), r.get("rel"), r.get("b")) == (a, rel, b):
                    del rels[i]
                    return data, f"dropped relation '{a} {rel} {b}'", (a, rel, b)
    return None


def _grow(data: dict[str, Any], err: LayoutError, original_bounds: list[float]) -> tuple[dict[str, Any], str] | None:
    st = data.get("setting", {})
    bounds = st.get("bounds_m")
    if space_bound(err) and st.get("kind") in ("indoor", "outdoor") and st.get("shape", "rect") != "polygon" and bounds:
        area = bounds[0] * bounds[1] * REPAIR_GROWTH ** 2
        if area <= original_bounds[0] * original_bounds[1] * REPAIR_MAX_AREA_GROWTH + 1e-9:
            st["bounds_m"] = [round(bounds[0] * REPAIR_GROWTH, 2), round(bounds[1] * REPAIR_GROWTH, 2), bounds[2]]
            return data, f"enlarged the setting to {st['bounds_m'][0]} x {st['bounds_m'][1]} m"
    return None


def relax(spec: dict[str, Any], err: LayoutError, original_bounds: list[float]) -> tuple[dict[str, Any], str] | None:
    """One edit that makes the conflict go away, as (new spec, description); None if out of ideas."""
    data = copy.deepcopy(spec)
    if _ceiling_object(err) is not None:
        return _relax_ceiling(data, err)
    m = re.search(r"main blockers: \d+x ([^;.]+)", str(err))
    if m and m.group(1).startswith("out of bounds"):
        # the object does not fit anywhere: dropping its relations cannot help, space can
        grown = _grow(data, err, original_bounds)
        if grown is not None:
            return grown
    dropped = _drop(data, err)
    if dropped is not None:
        return dropped[0], dropped[1]
    ids = {o["id"] for o in data.get("objects", [])}
    grown = _grow(data, err, original_bounds)
    if grown is not None:
        return grown
    m = re.search(r"cannot place '([^']+)'", str(err))
    if m:
        base = _base(m.group(1), ids)
        for o in data.get("objects", []):
            if o["id"] == base and o.get("count", 1) > 1 and not o.get("grid"):
                o["count"] -= 1
                return data, f"placed one '{base}' fewer ({o['count']} left)"
    return None


def _ceiling_object(err: LayoutError) -> str | None:
    """The id of the ceiling-mounted object the solver could not place, if that is what failed."""
    from ai_db_scene.priors import load_priors

    m = re.search(r"cannot place '([^']+)' \((\w+),", str(err))
    pr = load_priors().get(m.group(2)) if m else None
    return m.group(1) if m and pr is not None and pr.support == "ceiling" else None


def _relax_ceiling(data: dict[str, Any], err: LayoutError) -> tuple[dict[str, Any], str] | None:
    """A ceiling object never grows the room (lightning aidb5: four lights centered_in room grew
    a 50 x 30 m warehouse to 72 x 43 m and still failed). Drop its own relations (a free ceiling
    object is spread over an even grid), else hang one fewer, else give up with a clear hint."""
    iid = _ceiling_object(err)
    ids = {o["id"] for o in data.get("objects", [])}
    base = _base(iid or "", ids)
    rels = data.get("relations", [])
    mine = [r for r in rels if r.get("a") == base]
    if mine:
        data["relations"] = [r for r in rels if r.get("a") != base]
        labels = ", ".join(f"'{r.get('a')} {r.get('rel')} {r.get('b')}'" for r in mine)
        return data, f"'{base}' hangs on the ceiling grid instead of {labels} (a ceiling object never grows the room)"
    for o in data.get("objects", []):
        if o["id"] == base and o.get("count", 1) > 1 and not o.get("grid"):
            o["count"] -= 1
            return data, f"placed one '{base}' fewer ({o['count']} left)"
    return None


def explain(err: LayoutError, grown: bool) -> str:
    """What the model should change when repair gave up, in one sentence."""
    iid = _ceiling_object(err)
    if iid is not None:
        return (f"ceiling object '{iid}' does not fit on the ceiling: use fewer or smaller ceiling objects, "
                "or drop its relations so it goes on the ceiling grid (a bigger room is never the fix)")
    if space_bound(err):
        return ("the objects do not fit" + (" even in the enlarged room" if grown else "")
                + ": use a larger setting.bounds_m, fewer or smaller objects, or a grid for repeated ones")
    return ("a relation conflict a bigger room cannot fix: change or remove the relation(s) in 'conflicts' "
            "(edit_scene / resubmit), e.g. give a wall-mounted object its own wall")


#: in-between room sizes tried before a full growth step (fractions of the step)
_TIGHT_STEPS = (0.25, 0.5, 0.75)


def _tighter(before: dict[str, Any], grown: dict[str, Any], seed: int) -> tuple[dict[str, Any], str] | None:
    """The smallest room between ``before`` and ``grown`` that solves (a whole growth step left
    a classroom or a lobby with a band of empty floor round the content); None if none does."""
    b0, b1 = before["setting"]["bounds_m"], grown["setting"]["bounds_m"]
    for f in _TIGHT_STEPS:
        data = copy.deepcopy(grown)
        st = data["setting"]
        st["bounds_m"] = [round(b0[0] + (b1[0] - b0[0]) * f, 2), round(b0[1] + (b1[1] - b0[1]) * f, 2), b1[2]]
        spec, _errors = validate_spec(data)
        if spec is None:
            continue
        try:
            solve_layout(spec, seed=seed)
        except LayoutError:
            continue
        return data, f"enlarged the setting to {st['bounds_m'][0]} x {st['bounds_m'][1]} m"
    return None


@dataclass
class Repaired:
    """The outcome of ``solve_with_repairs``."""

    data: dict[str, Any]
    spec: SceneSpec | None = None
    placements: list[Placement] | None = None
    repairs: list[str] = field(default_factory=list)
    #: relations the loop removed, as "a rel b": the build reports each as an issue
    dropped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # validation errors (a repair broke the spec)
    error: LayoutError | None = None  # the conflict it could not repair
    hint: str | None = None


def solve_with_repairs(data: dict[str, Any], seed: int = 0, max_repairs: int = 6,
                       original_bounds: list[float] | None = None) -> Repaired:
    """Solve ``data``, relaxing it one edit at a time until it solves or nothing is left to try."""
    data = copy.deepcopy(data)
    original = list(original_bounds or data["setting"]["bounds_m"])
    out = Repaired(data)
    tried_second_seed = False
    while True:
        spec, errors = validate_spec(data)
        if spec is None:  # a repair must never produce an invalid spec; report it if it does
            out.errors = errors
            return out
        out.spec = spec
        try:
            out.placements = solve_layout(spec, seed=seed)
            return out
        except LayoutError as e:
            err = e
        fix = relax(data, err, original) if len(out.repairs) < max_repairs else None
        if fix is not None and fix[1].startswith("enlarged") and not tried_second_seed:
            # a bigger room is the last thing to try: the sampler may just have been unlucky
            # (qwen aidb7 classroom: 10 x 8 m grew to 12 x 9.6 m on seed 0, seed 1 fits it)
            tried_second_seed = True
            try:
                out.placements = solve_layout(spec, seed=seed + _SECOND_SEED)
                out.repairs.append("solved with a second seed (kept the room size)")
                return out
            except LayoutError:
                pass
        label = re.match(r"dropped relation '(\S+ (\w+) \S+)'", fix[1]) if fix else None
        if label and label.group(2) in _KEEP_HARDER and not tried_second_seed:
            tried_second_seed = True  # judges notice a sofa that does not face the TV: try once more first
            try:
                out.placements = solve_layout(spec, seed=seed + _SECOND_SEED)
                out.repairs.append("solved with a second seed (kept every relation)")
                return out
            except LayoutError:
                pass
        if fix is None:
            out.error = err
            out.hint = explain(err, any(r.startswith("enlarged") for r in out.repairs))
            return out
        if fix[1].startswith("enlarged"):
            tight = _tighter(data, fix[0], seed)
            if tight is not None:  # a step between the two sizes is enough: no emptier room than needed
                fix = tight
        data, note = fix
        out.data = data
        out.repairs.append(note)
        if label:
            out.dropped.append(label.group(1))
