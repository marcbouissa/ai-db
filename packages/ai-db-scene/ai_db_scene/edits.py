"""Ordered edit steps on a SceneSpec (``edit_scene``): refactor a built scene in several moves.

Each step is ``{"op": <name>, ...}``; steps apply in order to a copy of the spec, and the
result is validated as a whole (an edit may be temporarily inconsistent, e.g. add an object
in one step and its relations in the next). Errors name the step index.

ops:
  add_object      {"object": {...ObjectSpec...}, "relations": [...optional...]}
  remove_object   {"id": "sofa"}                       (also drops its relations)
  update_object   {"id": "sofa", "set": {"material": "velvet:#5b2a4a", "count": 2, ...}}
  move            {"id": "table", "to": {"against_wall": "wall_south", "gap_m": 0.5}}
                  (replaces the object's own relations; ``to`` maps relation -> target)
  add_relation    {"relation": {"a": ..., "rel": ..., "b": ..., "gap_m": ...}}
  remove_relation {"a": ..., "rel": ..., "b": ...}     (rel optional: all relations a->b)
  update_setting  {"set": {"bounds_m": [...], "style": ..., "floor_material": ..., ...}}
  rotate          {"id": "teacher_desk", "deg": 180}     (quarter turns on top of the way its
                  relations face it: ObjectSpec.turn_deg; what "rotate it 180 deg" asks for)

Weak models rarely write these shapes exactly (benchmark: 26 of 34 edit_scene calls were
invalid), so ``normalize_step`` first rewrites the forms whose intent is unambiguous: op
aliases (``delete`` -> ``remove_object``, ``add`` -> ``add_object``, ``action`` for ``op``),
``target``/``name`` for ``id``, flat fields instead of the nested ``object``/``relation``/``set``,
``changes`` for ``set``. Each rewrite is reported in the step's line of ``done``.
"""

from __future__ import annotations

import copy
import re
from typing import Any

OPS = ("add_object", "remove_object", "update_object", "move", "add_relation", "remove_relation",
       "update_setting", "rotate")
_OBJECT_FIELDS = ("category", "count", "hero", "detail", "grid", "material", "dynamic", "room", "seats",
                  "seat_category", "variant", "turn_deg")
#: where a rotate step says how far: the first number found (a [x, y, z] rotation: its z)
_TURN_KEYS = ("deg", "degrees", "angle", "by", "turn_deg", "yaw", "yaw_deg", "rotation_deg", "rotation", "rotate",
              "amount", "turn")
_SETTING_FIELDS = ("kind", "bounds_m", "style", "shape", "points", "floor_material", "wall_material", "rooms",
                   "mouldings", "ground", "water")
_RELATION_FIELDS = ("a", "rel", "b", "gap_m", "at_deg")
#: relation names a ``move`` / ``to`` may use as keys (spec.RELATIONS; kept local: no import cycle)
_RELS = ("on", "against_wall", "in_front_of", "beside", "facing", "centered_in", "under", "around", "above")

#: what weak models call the ops (benchmark edit_scene calls)
OP_ALIASES: dict[str, str] = {
    "delete": "remove_object", "delete_object": "remove_object", "remove": "remove_object",
    "del": "remove_object", "drop": "remove_object", "drop_object": "remove_object",
    "add": "add_object", "create": "add_object", "create_object": "add_object", "new_object": "add_object",
    "update": "update_object", "modify": "update_object", "modify_object": "update_object",
    "edit": "update_object", "edit_object": "update_object", "change": "update_object", "set": "update_object",
    "move_object": "move", "place": "move", "reposition": "move", "relocate": "move", "position": "move",
    "relate": "add_relation", "add_rel": "add_relation", "relation": "add_relation",
    "delete_relation": "remove_relation", "unrelate": "remove_relation", "remove_rel": "remove_relation",
    "drop_relation": "remove_relation",
    "setting": "update_setting", "set_setting": "update_setting", "update_room": "update_setting",
    "resize_room": "update_setting",
    "turn": "rotate", "rotate_object": "rotate", "turn_object": "rotate", "spin": "rotate", "flip": "rotate",
    "turn_around": "rotate", "rotate_by": "rotate", "reorient": "rotate",
}
#: ops that ask for something edit_scene does on every call anyway: skipped with a note
_NOOP_OPS = ("validate", "validate_scene", "apply_fixes", "fix", "build", "rebuild", "build_scene", "render",
             "snapshot", "save")
#: absolute transforms: the solver places objects from relations, so these cannot be honoured
#: camera edits (qwen aidb5: update_camera, update_setting {camera: ...}): the build places the camera
_CAMERA_OPS = ("update_camera", "set_camera", "camera", "move_camera", "adjust_camera", "camera_position")
_CAMERA_WHY = ("the build places its camera itself, on the open side looking at the fronts of the seating; to "
               "change the view, call set_camera after the edit (camera.focus in the spec picks what it looks at)")
_TRANSFORM_KEYS = ("location", "position", "translate", "offset", "offset_m", "rotation", "rotation_euler",
                   "rotation_deg", "yaw", "yaw_deg", "scale", "xy", "at")
_SPEC_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_COPY_SUFFIX = re.compile(r"^(.+)_(\d+)$")


def _keys(step: dict[str, Any]) -> list[str]:
    return sorted(step)


def _relations_from_to(oid: str, to: Any) -> list[dict[str, Any]] | str:
    """``{against_wall: wall_south, gap_m: 0.5}`` (or a relation / list of relations) -> relations."""
    if isinstance(to, list):
        out: list[dict[str, Any]] = []
        for t in to:
            got = _relations_from_to(oid, t)
            if isinstance(got, str):
                return got
            out += got
        return out
    if not isinstance(to, dict):
        return f"'to' must be {{<relation>: <target>}}, e.g. {{beside: sofa}}, got {to!r}"[:200]
    if "rel" in to:
        return [{"a": oid, **{k: v for k, v in to.items() if k in _RELATION_FIELDS and k != "a"}}]
    extra = {k: to[k] for k in ("gap_m", "at_deg") if k in to}
    rels = [{"a": oid, "rel": k, "b": v, **extra} for k, v in to.items() if k in _RELS]
    if not rels:
        bad = sorted(set(to) & set(_TRANSFORM_KEYS))
        if bad:
            return (f"{bad} are absolute transforms; the layout solver places objects from relations: "
                    f"give a relation instead, e.g. {{op: move, id: {oid}, to: {{against_wall: wall_north}}}} "
                    f"or {{to: {{beside: <object id>}}}}; relations: {list(_RELS)}")
        return f"'to' names no relation (got keys {sorted(to)}); use keys from {list(_RELS)}"
    return rels


def normalize_step(step: Any) -> tuple[dict[str, Any] | None, list[str], str | None]:
    """(canonical step, rewrites made, error). Rewrites only what has one obvious meaning."""
    if isinstance(step, str):
        return None, [], (f"expected an object {{op: ...}}, got the text {step[:80]!r}; e.g. "
                          "{op: move, id: coffee_table, to: {in_front_of: sofa}}")
    if not isinstance(step, dict):
        return None, [], f"expected an object {{op: ...}}, got {type(step).__name__}"
    s = dict(step)
    notes: list[str] = []
    if "op" not in s:
        for k in ("action", "type", "operation", "kind"):
            if isinstance(s.get(k), str):
                s["op"] = s.pop(k)
                notes.append(f"'{k}' read as 'op'")
                break
    op = s.get("op")
    if not isinstance(op, str):
        return None, notes, (f"needs 'op' (got keys {_keys(step)}); ops: {list(OPS)}")
    op = op.strip().lower()
    if op in _NOOP_OPS:
        return {"op": "noop", "was": op}, notes, None
    if op in _CAMERA_OPS or (op in ("update_setting", "setting", "set_setting") and isinstance(s.get("set"), dict)
                             and set(s["set"]) <= {"camera"}):
        return {"op": "noop", "was": op, "why": _CAMERA_WHY}, notes, None
    if op in OP_ALIASES:
        notes.append(f"op '{op}' read as '{OP_ALIASES[op]}'")
        op = OP_ALIASES[op]
    if op not in OPS:
        return None, notes, (f"op {op!r} is not an edit_scene op; ops: {list(OPS)} "
                             f"(also accepted: {sorted(OP_ALIASES)})")
    s["op"] = op
    # the object a step is about: id, else target / name / object_id / object (a string)
    if op != "add_object" and "id" not in s:
        for k in ("target", "name", "object_id", "object"):
            if isinstance(s.get(k), str):
                s["id"] = s.pop(k)
                notes.append(f"'{k}' read as 'id'")
                break
    if op == "add_object":
        if not isinstance(s.get("object"), dict):
            if isinstance(s.get("target"), dict):
                s["object"] = s.pop("target")
                notes.append("'target' read as 'object'")
            elif "id" in s or "category" in s:
                s["object"] = {k: s.pop(k) for k in ("id", *_OBJECT_FIELDS) if k in s}
                notes.append("flat object fields moved into 'object'")
        rels = s.get("relations")
        if isinstance(s.get("relation"), dict):
            rels = [*(rels or []), s.pop("relation")]
        if "to" in s and isinstance(s.get("object"), dict):
            got = _relations_from_to(str(s["object"].get("id")), s.pop("to"))
            if isinstance(got, str):
                return None, notes, got
            rels = [*(rels or []), *got]
            notes.append("'to' read as relations")
        if rels is not None:
            s["relations"] = rels
    elif op in ("update_object", "update_setting"):
        allowed = _OBJECT_FIELDS if op == "update_object" else _SETTING_FIELDS
        if not isinstance(s.get("set"), dict):
            for k in ("changes", "fields", "values", "update", "props", "properties"):
                if isinstance(s.get(k), dict):
                    s["set"] = s.pop(k)
                    notes.append(f"'{k}' read as 'set'")
                    break
        flat = {k: s.pop(k) for k in list(s) if k in allowed}
        if flat:
            prev = s.get("set")
            s["set"] = {**(prev if isinstance(prev, dict) else {}), **flat}
            notes.append(f"flat fields {sorted(flat)} moved into 'set'")
        if op == "update_object" and ("relations" in s or "to" in s) and not s.get("set"):
            notes.append("update_object with relations read as 'move'")
            s["op"] = op = "move"
        moves = set(s["set"]) if op == "update_object" and isinstance(s.get("set"), dict) else set()
        if moves and moves <= {"rotation", "rotation_euler", "rotation_deg", "yaw", "yaw_deg"}:
            # qwen aidb6 classroom, told "rotate it 180 deg": set {rotation: [0, 0, 180]}
            s = {"op": "rotate", "id": s.get("id"), "deg": s["set"][next(iter(moves))]}
            notes.append(f"update_object {sorted(moves)} read as rotate (a turn from where its relations face it)")
            op = "rotate"
    if op == "rotate":
        if step.get("op") in ("flip", "turn_around") and not any(k in s for k in _TURN_KEYS):
            s["deg"] = 180
        deg = next((_turn_amount(s[k]) for k in _TURN_KEYS if k in s and _turn_amount(s[k]) is not None), None)
        if deg is None:
            return None, notes, (f"rotate needs the turn in degrees, e.g. {{op: rotate, id: {s.get('id', '<id>')}, "
                                 f"deg: 180}} (got keys {_keys(step)})")
        s = {"op": "rotate", "id": s.get("id"), "deg": deg}
    elif op in ("add_relation", "remove_relation"):
        rel = s.get("relation", s.get("relations"))
        if (isinstance(rel, list) and op == "add_relation") or isinstance(rel, dict):
            s.pop("relations", None)
            s["relation"] = rel
        elif any(k in s for k in ("a", "rel", "b")):
            s["relation"] = {k: s.pop(k) for k in _RELATION_FIELDS if k in s}
            notes.append("flat relation fields moved into 'relation'")
        if op == "remove_relation" and isinstance(s.get("relation"), dict):
            s.update({k: v for k, v in s.pop("relation").items() if k in ("a", "rel", "b")})
    return s, notes, None


def _turn_amount(v: Any) -> float | None:
    """180 / "180" / "180 deg" / [0, 0, 180] -> 180.0; None if no number."""
    if isinstance(v, (list, tuple)) and v:
        v = v[-1]
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        v = float(m.group()) if m else None
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _resolve_id(oid: Any, objs: list[dict[str, Any]], p: str, op: str, step: dict[str, Any]) -> tuple[str | None, str | None]:
    """(spec object id, error). ``table_2`` (a built copy) resolves to its spec object ``table``."""
    ids = [o["id"] for o in objs]
    if oid is None:
        return None, f"{p}: {op} needs 'id' (got keys {_keys(step)}); spec object ids: {ids}"
    if oid in ids:
        return str(oid), None
    m = _COPY_SUFFIX.match(str(oid))
    if m and m.group(1) in ids:
        return m.group(1), None
    if not _SPEC_ID.match(str(oid)):
        return None, (f"{p}: '{oid}' is a Blender object, not a spec object; edit_scene edits spec objects "
                      f"(ids: {ids}) and rebuilds the scene, which clears Blender leftovers")
    return None, f"{p}: no object '{oid}' in the spec; spec object ids: {ids}"


def apply_steps(spec: dict[str, Any], steps: Any) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """(new spec, what each step did, errors). On any error the spec is None."""
    if isinstance(steps, dict):
        steps = [steps]
    if not isinstance(steps, list) or not steps:
        return None, [], ["steps: expected a non-empty list of {op: ...} objects"]
    data = copy.deepcopy(spec)
    data.setdefault("relations", [])
    done: list[str] = []
    for i, raw in enumerate(steps):
        p = f"steps[{i}]"
        step, notes, err = normalize_step(raw)
        if step is None:
            return None, done, [f"{p}: {err}"[:500]]
        how = f" ({'; '.join(notes)})" if notes else ""
        op = step["op"]
        objs = data["objects"]
        ids = {o["id"] for o in objs}
        if op == "noop":
            done.append(f"skipped '{step['was']}': " + step.get("why", "edit_scene validates, fixes and rebuilds "
                                                                       "on every call"))
            continue
        if op == "add_object":
            obj = step.get("object")
            if not isinstance(obj, dict) or not obj.get("id"):
                return None, done, [f"{p}: add_object needs 'object': {{id, category, ...}} (got keys {_keys(raw)})"]
            if obj["id"] in ids:
                return None, done, [f"{p}: object '{obj['id']}' already exists (use update_object or move)"]
            if not obj.get("category"):
                return None, done, [f"{p}: add_object '{obj['id']}' needs a 'category'"]
            objs.append(dict(obj))
            rels = step.get("relations") or []
            if not isinstance(rels, list) or not all(isinstance(r, dict) for r in rels):
                return None, done, [f"{p}.relations: expected a list of {{a, rel, b, ...}}"]
            data["relations"] += [{"a": obj["id"], **r} if "a" not in r else dict(r) for r in rels]
            done.append(f"added {obj['id']} ({obj.get('category')}){how}")
            continue
        oid = ""
        if op in ("remove_object", "update_object", "move", "rotate"):
            got, err = _resolve_id(step.get("id"), objs, p, op, raw)
            if got is None and op == "remove_object" and err and "Blender object" in err:
                done.append(f"skipped removing '{step.get('id')}': not a spec object; the rebuild clears it")
                continue
            if got is None:
                return None, done, [err or f"{p}: bad id"]
            oid = got
            if oid != step.get("id"):
                notes.append(f"'{step.get('id')}' read as '{oid}'")
                how = f" ({'; '.join(notes)})"
        if op == "remove_object":
            obj = next(o for o in objs if o["id"] == oid)
            if oid != step.get("id") and obj.get("count", 1) > 1 and not obj.get("grid"):
                obj["count"] -= 1  # one copy of a counted object: one fewer
                done.append(f"{oid}: one fewer ({obj['count']} left){how}")
                continue
            data["objects"] = [o for o in objs if o["id"] != oid]
            before = len(data["relations"])
            data["relations"] = [r for r in data["relations"] if oid not in (r.get("a"), r.get("b"))]
            done.append(f"removed {oid} and {before - len(data['relations'])} relation(s){how}")
        elif op == "update_object":
            changes = step.get("set")
            moves = sorted(set(raw) & set(_TRANSFORM_KEYS)) if isinstance(raw, dict) else []
            if not isinstance(changes, dict) or not changes:
                why = (f"{moves} are absolute transforms; positions come from relations: use "
                       f"{{op: move, id: {oid}, to: {{against_wall: wall_north}}}}" if moves else
                       f"needs 'set': {{field: value}} (got keys {_keys(raw)})")
                return None, done, [f"{p}: update_object {why}; editable fields {list(_OBJECT_FIELDS)}"]
            bad = sorted(set(changes) - set(_OBJECT_FIELDS))
            if bad:
                return None, done, [(f"{p}.set: fields {bad} not editable; allowed {list(_OBJECT_FIELDS)}"
                                      + ("; to reposition use op 'move'" if set(bad) & set(_TRANSFORM_KEYS) else ""))]
            obj = next(o for o in objs if o["id"] == oid)
            obj.update(changes)
            if "grid" in changes and "count" not in changes:
                obj.pop("count", None)  # the grid decides the count
            done.append(f"updated {oid}: {', '.join(sorted(changes))}{how}")
        elif op == "move":
            target = step.get("to")
            if target is None and isinstance(step.get("relations"), list):
                target = step["relations"]
            if target is None and isinstance(step.get("relation"), dict):
                target = step["relation"]
            if target is None:
                flat = {k: step[k] for k in (*_RELS, "gap_m", "at_deg", *_TRANSFORM_KEYS) if k in step}
                target = flat or None
            if target is None:
                return None, done, [(f"{p}: move needs 'to': {{<relation>: <target>}}, e.g. "
                                      f"{{op: move, id: {oid}, to: {{in_front_of: sofa, gap_m: 0.4}}}} "
                                      f"(got keys {_keys(raw)}); relations: {list(_RELS)}")]
            new = _relations_from_to(oid, target)
            if isinstance(new, str):
                return None, done, [f"{p}: move {new}"]
            new = [{**r, "a": oid} for r in new]
            before = len(data["relations"])
            data["relations"] = [r for r in data["relations"] if r.get("a") != oid] + new
            done.append(f"moved {oid}: replaced {before - len(data['relations']) + len(new)} relation(s) with "
                        + ", ".join(f"{r['rel']} {r.get('b')}" for r in new) + how)
        elif op == "rotate":
            obj = next(o for o in objs if o["id"] == oid)
            before = obj.get("turn_deg", 0) if isinstance(obj.get("turn_deg"), (int, float)) else 0
            turn = round((before + step["deg"]) / 90.0) * 90 % 360
            obj["turn_deg"] = turn
            snapped = "" if abs((before + step["deg"]) % 360 - turn) < 1e-9 else f" (quarter turns only: {turn})"
            done.append(f"turned {oid} by {step['deg']:g} deg{snapped}: it now stands {turn} deg from the way its "
                        f"relations face it{how}")
        elif op == "add_relation":
            rel = step.get("relation")
            rels = rel if isinstance(rel, list) else [rel]
            if not rels or not all(isinstance(r, dict) and {"a", "rel", "b"} <= set(r) for r in rels):
                return None, done, [(f"{p}: add_relation needs 'relation': {{a, rel, b, gap_m?}} "
                                      f"(got keys {_keys(raw)})")]
            data["relations"] += [{k: v for k, v in r.items() if k in _RELATION_FIELDS} for r in rels]
            done.append("related " + ", ".join(f"{r.get('a')} {r.get('rel')} {r.get('b')}" for r in rels) + how)
        elif op == "remove_relation":
            a, rel, b = step.get("a"), step.get("rel"), step.get("b")
            if a is None or b is None:
                return None, done, [(f"{p}: remove_relation needs 'a' and 'b' (and optionally 'rel') "
                                      f"(got keys {_keys(raw)})")]
            keep = [r for r in data["relations"]
                    if not (r.get("a") == a and r.get("b") == b and (rel is None or r.get("rel") == rel))]
            if len(keep) == len(data["relations"]):
                have = [f"{r.get('a')} {r.get('rel')} {r.get('b')}" for r in data["relations"] if a in (r.get("a"), r.get("b"))]
                return None, done, [f"{p}: no relation {a} {rel or '*'} {b}; relations of '{a}': {have}"]
            done.append(f"removed {len(data['relations']) - len(keep)} relation(s) {a} {rel or '*'} {b}{how}")
            data["relations"] = keep
        else:  # update_setting
            changes = step.get("set")
            if not isinstance(changes, dict) or not changes:
                return None, done, [(f"{p}: update_setting needs 'set': {{field: value}} (got keys {_keys(raw)}); "
                                      f"editable fields {list(_SETTING_FIELDS)}")]
            if "camera" in changes:  # with other fields: those apply, the camera is the build's
                changes = {k: v for k, v in changes.items() if k != "camera"}
                how += f" (camera skipped: {_CAMERA_WHY})"
            bad = sorted(set(changes) - set(_SETTING_FIELDS))
            if bad:
                return None, done, [f"{p}.set: fields {bad} not editable; allowed {list(_SETTING_FIELDS)}"]
            data["setting"].update(changes)
            done.append(f"setting: {', '.join(sorted(changes))}{how}")
    return data, done, []
