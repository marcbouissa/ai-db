"""SceneSpec: the strict, closed-vocabulary description a small model fills in."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from ai_db.constants import LAYOUT_MAX_BOUND_M, LAYOUT_MAX_COUNT
from ai_db.scene.priors import categories

RELATIONS = ("on", "against_wall", "in_front_of", "beside", "facing", "centered_in", "under", "around")
SETTING_KINDS = ("indoor", "outdoor", "tabletop", "single_asset")
WALLS = ("wall_north", "wall_south", "wall_east", "wall_west")
ROOM = "room"  # target keyword for centered_in
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# Relations that need a wall / an object / either object or the room as their `b`.
_WALL_REL = "against_wall"


@dataclass(frozen=True)
class Setting:
    kind: str
    bounds_m: tuple[float, float, float]  # x (east-west), y (north-south), z (height)
    style: str | None = None


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    category: str
    count: int = 1
    hero: bool = False


@dataclass(frozen=True)
class Relation:
    a: str
    rel: str
    b: str
    gap_m: float | None = None


@dataclass(frozen=True)
class SceneSpec:
    setting: Setting
    objects: tuple[ObjectSpec, ...]
    relations: tuple[Relation, ...] = ()
    lighting: dict[str, Any] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneSpec:
        """Build a spec; raises ValueError listing every validation error."""
        spec, errors = validate_spec(data)
        if spec is None:
            raise ValueError("invalid SceneSpec:\n" + "\n".join(errors))
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "setting": {"kind": self.setting.kind, "bounds_m": list(self.setting.bounds_m), "style": self.setting.style},
            "objects": [{"id": o.id, "category": o.category, "count": o.count, "hero": o.hero} for o in self.objects],
            "relations": [
                {"a": r.a, "rel": r.rel, "b": r.b, **({"gap_m": r.gap_m} if r.gap_m is not None else {})}
                for r in self.relations
            ],
            "lighting": dict(self.lighting),
            "camera": dict(self.camera),
        }


def spec_schema() -> dict[str, Any]:
    """JSON schema (draft 2020-12) of a SceneSpec, with the closed vocabularies inlined."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["setting", "objects"],
        "additionalProperties": False,
        "properties": {
            "setting": {
                "type": "object",
                "required": ["kind", "bounds_m"],
                "additionalProperties": False,
                "properties": {
                    "kind": {"enum": list(SETTING_KINDS)},
                    "bounds_m": {
                        "type": "array",
                        "items": {"type": "number", "exclusiveMinimum": 0, "maximum": LAYOUT_MAX_BOUND_M},
                        "minItems": 3,
                        "maxItems": 3,
                        "description": "[x width, y depth, z height] in meters",
                    },
                    "style": {"type": "string"},
                },
            },
            "objects": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id", "category"],
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "pattern": _ID_RE.pattern},
                        "category": {"enum": categories()},
                        "count": {"type": "integer", "minimum": 1, "maximum": LAYOUT_MAX_COUNT},
                        "hero": {"type": "boolean"},
                    },
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["a", "rel", "b"],
                    "additionalProperties": False,
                    "properties": {
                        "a": {"type": "string", "description": "object id"},
                        "rel": {"enum": list(RELATIONS)},
                        "b": {
                            "type": "string",
                            "description": f"object id; wall name {list(WALLS)} for against_wall (indoor); "
                            f"'{ROOM}' also allowed for centered_in",
                        },
                        "gap_m": {"type": "number", "minimum": 0},
                    },
                },
            },
            "lighting": {"type": "object"},
            "camera": {"type": "object", "properties": {"focus": {"type": "string", "description": "object id"}}},
        },
    }


def _typename(v: object) -> str:
    return type(v).__name__


def _is_num(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _unknown_keys(path: str, d: dict, allowed: tuple[str, ...], errors: list[str]) -> None:
    for k in d:
        if k not in allowed:
            errors.append(f"{path}.{k}: unknown field; allowed fields: {list(allowed)}")


def _suggest(value: str, allowed: list[str]) -> str:
    close = difflib.get_close_matches(value, allowed, n=1)
    return f" Did you mean '{close[0]}'?" if close else ""


def _parse_setting(raw: Any, errors: list[str]) -> Setting | None:
    if not isinstance(raw, dict):
        errors.append(f"setting: expected object, got {_typename(raw)}")
        return None
    n0 = len(errors)
    _unknown_keys("setting", raw, ("kind", "bounds_m", "style"), errors)
    kind = raw.get("kind")
    if kind not in SETTING_KINDS:
        errors.append(f"setting.kind: {kind!r} is not allowed; allowed values: {list(SETTING_KINDS)}")
    b = raw.get("bounds_m")
    bounds: tuple[float, float, float] | None = None
    if not isinstance(b, list) or len(b) != 3:
        errors.append(f"setting.bounds_m: expected [x, y, z] list of 3 positive numbers in meters, got {b!r}")
    else:
        bad = False
        for i, v in enumerate(b):
            if not _is_num(v) or not 0 < v <= LAYOUT_MAX_BOUND_M:
                errors.append(f"setting.bounds_m[{i}]: expected number in (0, {LAYOUT_MAX_BOUND_M:g}], got {v!r}")
                bad = True
        if not bad:
            bounds = (float(b[0]), float(b[1]), float(b[2]))
    style = raw.get("style")
    if style is not None and not isinstance(style, str):
        errors.append(f"setting.style: expected string, got {_typename(style)}")
    if len(errors) > n0 or bounds is None:
        return None
    return Setting(str(kind), bounds, style)


def _parse_objects(raw: Any, errors: list[str]) -> list[ObjectSpec]:
    if not isinstance(raw, list) or not raw:
        errors.append(f"objects: expected a non-empty list, got {raw!r}" if raw is not None else "objects: required field is missing")
        return []
    cats = categories()
    out: list[ObjectSpec] = []
    seen: set[str] = set()
    for i, o in enumerate(raw):
        p = f"objects[{i}]"
        if not isinstance(o, dict):
            errors.append(f"{p}: expected object, got {_typename(o)}")
            continue
        _unknown_keys(p, o, ("id", "category", "count", "hero"), errors)
        oid, cat, count, hero = o.get("id"), o.get("category"), o.get("count", 1), o.get("hero", False)
        ok = True
        if not isinstance(oid, str) or not _ID_RE.match(oid):
            errors.append(f"{p}.id: {oid!r} must match {_ID_RE.pattern} (lowercase snake_case)")
            ok = False
        elif oid in WALLS or oid == ROOM:
            errors.append(f"{p}.id: '{oid}' is reserved (walls {list(WALLS)} and '{ROOM}')")
            ok = False
        elif oid in seen:
            errors.append(f"{p}.id: duplicate id '{oid}'")
            ok = False
        if not isinstance(cat, str) or cat not in cats:
            errors.append(f"{p}.category: {cat!r} is not a known category.{_suggest(str(cat), cats)} allowed values: {cats}")
            ok = False
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= LAYOUT_MAX_COUNT:
            errors.append(f"{p}.count: expected integer in [1, {LAYOUT_MAX_COUNT}], got {count!r}")
            ok = False
        if not isinstance(hero, bool):
            errors.append(f"{p}.hero: expected boolean, got {hero!r}")
            ok = False
        if isinstance(oid, str):
            seen.add(oid)
        if ok:
            out.append(ObjectSpec(str(oid), str(cat), count, hero))
    return out


def _parse_relations(raw: Any, ids: list[str], kind: str, errors: list[str]) -> list[Relation]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"relations: expected a list, got {_typename(raw)}")
        return []
    out: list[Relation] = []
    for i, r in enumerate(raw):
        p = f"relations[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{p}: expected object, got {_typename(r)}")
            continue
        _unknown_keys(p, r, ("a", "rel", "b", "gap_m"), errors)
        a, rel, b, gap = r.get("a"), r.get("rel"), r.get("b"), r.get("gap_m")
        n0 = len(errors)
        if rel not in RELATIONS:
            errors.append(f"{p}.rel: {rel!r} is not allowed; allowed values: {list(RELATIONS)}")
        if a not in ids:
            errors.append(f"{p}.a: {a!r} is not a declared object id.{_suggest(str(a), ids)} declared ids: {ids}")
        if rel == _WALL_REL:
            if kind != "indoor":
                errors.append(f"{p}: against_wall needs setting.kind 'indoor' (walls exist only indoors), got '{kind}'")
            elif b not in WALLS:
                errors.append(f"{p}.b: {b!r} is not a wall; allowed values: {list(WALLS)}")
        elif rel in RELATIONS:
            allowed_b = ids + ([ROOM] if rel == "centered_in" else [])
            if b not in allowed_b:
                errors.append(f"{p}.b: {b!r} is not a valid target for '{rel}'.{_suggest(str(b), allowed_b)} allowed values: {allowed_b}")
        if a == b:
            errors.append(f"{p}: object '{a}' cannot be related to itself")
        if gap is not None and (not _is_num(gap) or gap < 0):
            errors.append(f"{p}.gap_m: expected number >= 0, got {gap!r}")
        if len(errors) == n0:
            out.append(Relation(str(a), str(rel), str(b), None if gap is None else float(gap)))
    return out


def validate_spec(data: Any) -> tuple[SceneSpec | None, list[str]]:
    """Validate a dict. Returns (spec, []) or (None, errors); errors are 'path: problem; allowed'."""
    if not isinstance(data, dict):
        return None, [f"<root>: expected object, got {_typename(data)}"]
    errors: list[str] = []
    _unknown_keys("<root>", data, ("setting", "objects", "relations", "lighting", "camera"), errors)
    if "setting" not in data:
        errors.append("setting: required field is missing")
    if "objects" not in data:
        errors.append("objects: required field is missing")
    setting = _parse_setting(data["setting"], errors) if "setting" in data else None
    objs_present = "objects" in data
    objs = _parse_objects(data.get("objects"), errors) if objs_present else []
    raw_objs = data.get("objects")
    declared = [o["id"] for o in raw_objs if isinstance(o, dict) and isinstance(o.get("id"), str)] if isinstance(raw_objs, list) else []
    kind = setting.kind if setting else ""
    rels = _parse_relations(data.get("relations"), declared, kind, errors) if setting is not None or not errors else []
    lighting = data.get("lighting", {})
    camera = data.get("camera", {})
    if not isinstance(lighting, dict):
        errors.append(f"lighting: expected object, got {_typename(lighting)}")
        lighting = {}
    if not isinstance(camera, dict):
        errors.append(f"camera: expected object, got {_typename(camera)}")
        camera = {}
    focus = camera.get("focus")
    if focus is not None and focus not in declared:
        errors.append(f"camera.focus: {focus!r} is not a declared object id; declared ids: {declared}")
    if errors or setting is None:
        return None, errors
    return SceneSpec(setting, tuple(objs), tuple(rels), lighting, camera), []
