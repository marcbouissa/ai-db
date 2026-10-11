"""Ordered edit steps on a SceneSpec (``edit_scene``): refactor a built scene in several moves.

Each step is ``{"op": <name>, ...}``; steps apply in order to a copy of the spec, and the
result is validated as a whole (an edit may be temporarily inconsistent, e.g. add an object
in one step and its relations in the next). Errors name the step index.

ops:
  add_object      {"object": {...ObjectSpec...}, "relations": [...optional...]}
  remove_object   {"id": "sofa"}                       (also drops its relations)
  update_object   {"id": "sofa", "set": {"material": "velvet:#5b2a4a", "count": 2, ...}}
  add_relation    {"relation": {"a": ..., "rel": ..., "b": ..., "gap_m": ...}}
  remove_relation {"a": ..., "rel": ..., "b": ...}     (rel optional: all relations a->b)
  update_setting  {"set": {"bounds_m": [...], "style": ..., "floor_material": ..., ...}}
"""

from __future__ import annotations

import copy
from typing import Any

OPS = ("add_object", "remove_object", "update_object", "add_relation", "remove_relation", "update_setting")
_OBJECT_FIELDS = ("category", "count", "hero", "detail", "grid", "material", "dynamic", "room")
_SETTING_FIELDS = ("kind", "bounds_m", "style", "shape", "points", "floor_material", "wall_material", "rooms",
                   "mouldings")


def apply_steps(spec: dict[str, Any], steps: Any) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """(new spec, what each step did, errors). On any error the spec is None."""
    if not isinstance(steps, list) or not steps:
        return None, [], ["steps: expected a non-empty list of {op: ...} objects"]
    data = copy.deepcopy(spec)
    data.setdefault("relations", [])
    done: list[str] = []
    for i, step in enumerate(steps):
        p = f"steps[{i}]"
        if not isinstance(step, dict) or step.get("op") not in OPS:
            return None, done, [f"{p}.op: expected one of {list(OPS)}, got {step!r}"[:300]]
        op = step["op"]
        objs = data["objects"]
        ids = {o["id"] for o in objs}
        if op == "add_object":
            obj = step.get("object")
            if not isinstance(obj, dict) or not obj.get("id"):
                return None, done, [f"{p}.object: expected an object with an id"]
            if obj["id"] in ids:
                return None, done, [f"{p}: object '{obj['id']}' already exists (use update_object)"]
            objs.append(dict(obj))
            data["relations"] += [dict(r) for r in step.get("relations") or []]
            done.append(f"added {obj['id']} ({obj.get('category')})")
        elif op == "remove_object":
            oid = step.get("id")
            if oid not in ids:
                return None, done, [f"{p}: no object '{oid}'"]
            data["objects"] = [o for o in objs if o["id"] != oid]
            before = len(data["relations"])
            data["relations"] = [r for r in data["relations"] if oid not in (r.get("a"), r.get("b"))]
            done.append(f"removed {oid} and {before - len(data['relations'])} relation(s)")
        elif op == "update_object":
            oid, changes = step.get("id"), step.get("set")
            if oid not in ids:
                return None, done, [f"{p}: no object '{oid}'"]
            bad = sorted(set(changes or {}) - set(_OBJECT_FIELDS)) if isinstance(changes, dict) else ["set"]
            if bad or not changes:
                return None, done, [f"{p}.set: fields {bad} not editable; allowed {list(_OBJECT_FIELDS)}"]
            obj = next(o for o in objs if o["id"] == oid)
            obj.update(changes)
            if "grid" in changes and "count" not in changes:
                obj.pop("count", None)  # the grid decides the count
            done.append(f"updated {oid}: {', '.join(sorted(changes))}")
        elif op == "add_relation":
            rel = step.get("relation")
            if not isinstance(rel, dict):
                return None, done, [f"{p}.relation: expected {{a, rel, b, ...}}"]
            data["relations"].append(dict(rel))
            done.append(f"related {rel.get('a')} {rel.get('rel')} {rel.get('b')}")
        elif op == "remove_relation":
            a, rel, b = step.get("a"), step.get("rel"), step.get("b")
            keep = [r for r in data["relations"]
                    if not (r.get("a") == a and r.get("b") == b and (rel is None or r.get("rel") == rel))]
            if len(keep) == len(data["relations"]):
                return None, done, [f"{p}: no relation {a} {rel or '*'} {b}"]
            done.append(f"removed {len(data['relations']) - len(keep)} relation(s) {a} {rel or '*'} {b}")
            data["relations"] = keep
        else:  # update_setting
            changes = step.get("set")
            bad = sorted(set(changes or {}) - set(_SETTING_FIELDS)) if isinstance(changes, dict) else ["set"]
            if bad or not changes:
                return None, done, [f"{p}.set: fields {bad} not editable; allowed {list(_SETTING_FIELDS)}"]
            data["setting"].update(changes)
            done.append(f"setting: {', '.join(sorted(changes))}")
    return data, done, []
