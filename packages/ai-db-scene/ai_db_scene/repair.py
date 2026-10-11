"""Automatic repair of an unsatisfiable SceneSpec, one minimal change at a time.

The benchmark showed agents abandoning the pipeline when ``solve_layout`` reported a conflict
(47 solve calls, 18 builds): fixing a spec from a conflict message is exactly what a small model
does badly. ``relax`` turns one LayoutError into one concrete edit and says what it changed:

1. drop the conflicting relation the solver named (positional ones before against_wall / on),
2. otherwise give the room more space (x/y bounds +20%, up to ``REPAIR_MAX_GROWTH``),
3. otherwise place one instance fewer of the object that could not be placed.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from ai_db_scene.constants import REPAIR_GROWTH, REPAIR_MAX_GROWTH
from ai_db_scene.layout import LayoutError

_SUFFIX = re.compile(r"_\d+$")
_LABEL = re.compile(r"^(\S+) (\w+) (\S+)")
#: dropped first: they position an object relative to another; walls and support come last
_DROP_ORDER = ("facing", "beside", "around", "in_front_of", "centered_in", "under", "against_wall", "on")


def _base(iid: str, ids: set[str]) -> str:
    return iid if iid in ids else _SUFFIX.sub("", iid)


def relax(spec: dict[str, Any], err: LayoutError, original_bounds: list[float]) -> tuple[dict[str, Any], str] | None:
    """One edit that makes the conflict go away, as (new spec, description); None if out of ideas."""
    data = copy.deepcopy(spec)
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
                    return data, f"dropped relation '{a} {rel} {b}'"
    st = data.get("setting", {})
    bounds = st.get("bounds_m")
    if st.get("kind") in ("indoor", "outdoor") and st.get("shape", "rect") != "polygon" and bounds \
            and bounds[0] * REPAIR_GROWTH <= original_bounds[0] * REPAIR_MAX_GROWTH:
        st["bounds_m"] = [round(bounds[0] * REPAIR_GROWTH, 2), round(bounds[1] * REPAIR_GROWTH, 2), bounds[2]]
        return data, f"enlarged the setting to {st['bounds_m'][0]} x {st['bounds_m'][1]} m"
    m = re.search(r"cannot place '([^']+)'", str(err))
    if m:
        base = _base(m.group(1), ids)
        for o in data.get("objects", []):
            if o["id"] == base and o.get("count", 1) > 1 and not o.get("grid"):
                o["count"] -= 1
                return data, f"placed one '{base}' fewer ({o['count']} left)"
    return None
