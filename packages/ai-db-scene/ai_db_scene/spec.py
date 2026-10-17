"""SceneSpec: the strict, closed-vocabulary description a small model fills in."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from ai_db_scene.constants import LAYOUT_GRID_GAP_M, LAYOUT_MAX_BOUND_M, LAYOUT_MAX_COUNT
from ai_db_scene.furniture_variants import coerce_variant, families
from ai_db_scene.materials import STYLE_PALETTES, coerce_material, material_vocabulary, style_error
from ai_db_scene.priors import SUPPORTS, categories
from ai_db_scene.shapes import polygon_wall_names

RELATIONS = ("on", "against_wall", "in_front_of", "beside", "facing", "centered_in", "under", "around", "above")
SETTING_KINDS = ("indoor", "outdoor", "tabletop", "single_asset")
WALLS = ("wall_north", "wall_south", "wall_east", "wall_west")
SHAPES = ("rect", "ellipse", "polygon")
#: setting.mouldings -> the profile swept along the walls (profiles of the Blender addon)
MOULDINGS = ("crown", "chair_rail", "baseboard")
DETAILS = ("draft", "standard", "hero")
MAX_SEATS = 12  # ObjectSpec.seats per instance
SUPPORT_VOCAB = SUPPORTS  # floor | surface | wall | ceiling | ground: shared with the Blender builder
NONRECT_WALL = "wall"  # implicit wall of ellipse/polygon rooms (with at_deg); polygons also have wall_1..wall_N
ROOM = "room"  # target keyword for centered_in
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# Relations that need a wall / an object / either object or the room as their `b`.
_WALL_REL = "against_wall"


@dataclass(frozen=True)
class Room:
    """One rectangle of a multi-room building, in the setting's frame (x east, y north)."""

    id: str
    offset_m: tuple[float, float]  # its south-west corner inside setting.bounds_m
    size_m: tuple[float, float]


@dataclass(frozen=True)
class Setting:
    kind: str
    bounds_m: tuple[float, float, float]  # x (east-west), y (north-south), z (height)
    style: str | None = None
    shape: str = "rect"
    points: tuple[tuple[float, float], ...] = ()  # polygon footprint (x, y) in meters
    floor_material: str | None = None  # see ai_db_scene.materials; None = default for the kind
    wall_material: str | None = None
    #: multi-room building (indoor rect only): each object names its room, a door joins
    #: rooms that share a wall (layout.room_doors); empty = one room, as before
    rooms: tuple[Room, ...] = ()
    #: classical mouldings swept along the room outline (MOULDINGS); None = the style's default
    #: (classic: crown + baseboard), () = none
    mouldings: tuple[str, ...] | None = None

    def moulding_list(self) -> tuple[str, ...]:
        """The mouldings a build adds: explicit, else the style's (single indoor rooms only)."""
        if self.mouldings is not None:
            return self.mouldings
        if self.kind == "indoor" and not self.rooms and self.style == "classic":
            return ("crown", "baseboard")
        return ()


@dataclass(frozen=True)
class ObjectSpec:
    id: str
    category: str
    count: int = 1
    hero: bool = False
    detail: str = "standard"
    #: (rows, cols, gap between columns m, gap between rows m): instances in a block, all
    #: facing the same way (classroom desks, cafe tables). count == rows * cols.
    grid: tuple[int, int, float, float] | None = None
    material: str | None = None  # see ai_db_scene.materials; None = the category default
    dynamic: bool = False  # build_scene physics: an active rigid body (else passive)
    room: str | None = None  # Setting.rooms id (required there, refused without rooms)
    #: chairs at every instance (desk, table): the solver places them with it, so each desk has
    #: its chair (chairs solved as a separate block never ended up at the desks); 0 = none
    seats: int = 0
    seat_category: str | None = None  # default: the table's usual seat (layout.SEAT_CATEGORY)
    #: design family of generated seating/tables (furniture_variants.FAMILIES); None = the scene's
    variant: str | None = None
    #: quarter turns (0/90/180/270, counter-clockwise) on top of the yaw the relations give: what
    #: edit_scene's rotate op sets when the validator says "rotate it 180 deg"
    turn_deg: int = 0


@dataclass(frozen=True)
class Relation:
    a: str
    rel: str
    b: str
    gap_m: float | None = None
    at_deg: float | None = None  # against_wall on ellipse/polygon: where on the wall (ellipse angle / polygon ray from center)


@dataclass(frozen=True)
class SceneSpec:
    setting: Setting
    objects: tuple[ObjectSpec, ...]
    relations: tuple[Relation, ...] = ()
    lighting: dict[str, Any] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    #: what validation rewrote instead of refusing (a material alias, a coerced relation), for
    #: the caller to report; not part of the spec itself (to_dict leaves it out)
    notes: tuple[str, ...] = field(default=(), compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneSpec:
        """Build a spec; raises ValueError listing every validation error."""
        spec, errors = validate_spec(data)
        if spec is None:
            raise ValueError("invalid SceneSpec:\n" + "\n".join(errors))
        return spec

    def to_dict(self) -> dict[str, Any]:
        return {
            "setting": {
                "kind": self.setting.kind,
                "bounds_m": list(self.setting.bounds_m),
                "style": self.setting.style,
                **({"shape": self.setting.shape} if self.setting.shape != "rect" else {}),
                **({"points": [list(p) for p in self.setting.points]} if self.setting.points else {}),
                **({"floor_material": self.setting.floor_material} if self.setting.floor_material else {}),
                **({"wall_material": self.setting.wall_material} if self.setting.wall_material else {}),
                **({"rooms": [{"id": r.id, "offset_m": list(r.offset_m), "size_m": list(r.size_m)}
                              for r in self.setting.rooms]} if self.setting.rooms else {}),
                **({"mouldings": list(self.setting.mouldings)} if self.setting.mouldings is not None else {}),
            },
            "objects": [
                {"id": o.id, "category": o.category, "count": o.count, "hero": o.hero, "detail": o.detail,
                 **({"grid": {"rows": o.grid[0], "cols": o.grid[1], "gap_m": [o.grid[2], o.grid[3]]}} if o.grid else {}),
                 **({"material": o.material} if o.material else {}),
                 **({"dynamic": True} if o.dynamic else {}),
                 **({"room": o.room} if o.room else {}),
                 **({"seats": o.seats} if o.seats else {}),
                 **({"seat_category": o.seat_category} if o.seat_category else {}),
                 **({"variant": o.variant} if o.variant else {}),
                 **({"turn_deg": o.turn_deg} if o.turn_deg else {})}
                for o in self.objects
            ],
            "relations": [
                {
                    "a": r.a, "rel": r.rel, "b": r.b,
                    **({"gap_m": r.gap_m} if r.gap_m is not None else {}),
                    **({"at_deg": r.at_deg} if r.at_deg is not None else {}),
                }
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
                    "style": {"type": "string",
                              "description": f"material palette when it names one of {sorted(STYLE_PALETTES)} "
                                             "(floor, walls, furniture); explicit materials win"},
                    "shape": {"enum": list(SHAPES), "description": "room footprint inside bounds_m x/y; default rect"},
                    "floor_material": {"type": "string", "description": "floor surface, e.g. wood:oak, marble:carrara (see objects.material)"},
                    "wall_material": {"type": "string", "description": "wall surface, e.g. plaster:#e8e0d0 (see objects.material)"},
                    "mouldings": {"type": "array", "items": {"enum": list(MOULDINGS)}, "uniqueItems": True,
                                  "description": ("indoor single room: classical profiles swept along the walls "
                                                  "(rect, ellipse or polygon), with gaps at doors and windows; "
                                                  "default: crown + baseboard for style classic, else none; [] = none")},
                    "rooms": {
                        "type": "array",
                        "minItems": 2,
                        "description": ("indoor rect only: rooms of a building, each a rectangle inside bounds_m; "
                                        "rooms sharing a wall get a door; every object then names its room"),
                        "items": {"type": "object", "required": ["id", "offset_m", "size_m"], "additionalProperties": False,
                                  "properties": {"id": {"type": "string", "pattern": _ID_RE.pattern},
                                                 "offset_m": {"type": "array", "items": {"type": "number"}, "minItems": 2,
                                                              "maxItems": 2, "description": "[x, y] of its south-west corner"},
                                                 "size_m": {"type": "array", "items": {"type": "number", "exclusiveMinimum": 0},
                                                            "minItems": 2, "maxItems": 2}}},
                    },
                    "points": {
                        "type": "array",
                        "minItems": 3,
                        "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                        "description": "polygon only: [[x, y], ...] vertices in meters inside [0,x]x[0,y]",
                    },
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
                        "room": {"type": "string", "description": "setting.rooms id (multi-room buildings only)"},
                        "dynamic": {"type": "boolean",
                                    "description": "build_scene physics: falls and moves (active rigid body)"},
                        "seats": {"type": "integer", "minimum": 0, "maximum": MAX_SEATS,
                                  "description": ("chairs at each instance of a desk/table (also each grid cell), "
                                                  "placed with it: front and back sides, then the ends. Use this "
                                                  "instead of a separate chair object, e.g. {id: desks, category: desk, "
                                                  "grid: {rows: 3, cols: 4}, seats: 1}")},
                        "seat_category": {"type": "string",
                                          "description": "the seats' category (default: the table's usual chair)"},
                        "detail": {"enum": list(DETAILS), "description": "modeling detail level; default standard"},
                        "turn_deg": {"enum": [0, 90, 180, 270],
                                     "description": ("turn the object this much (counter-clockwise) from the way its "
                                                     "relations face it, e.g. 180 when it faces a wall; default 0")},
                        "variant": {"type": "string",
                                    "description": ("sofa/armchair/table/chair design (also as 'style'): one of "
                                                    f"{families()}; default: one the setting.style suits, the "
                                                    "same for the whole room")},
                        "material": {"type": "string", "description": ("surface material: kind or kind:variant, optional "
                                                                ":#rrggbb tint, e.g. wood:walnut, fabric:linen:#6b7f99. "
                                                                "Omit for the category default. Allowed: "
                                                                + ", ".join(material_vocabulary()))},
                        "grid": {
                            "type": "object",
                            "required": ["rows", "cols"],
                            "additionalProperties": False,
                            "description": ("rows x cols block of this object, all facing the same way (classroom "
                                            "desks, cafe tables); count = rows*cols. Relations of the object "
                                            "apply to the whole block, e.g. {a: desk, rel: facing, b: whiteboard}"),
                            "properties": {
                                "rows": {"type": "integer", "minimum": 1},
                                "cols": {"type": "integer", "minimum": 1},
                                "gap_m": {"type": "array", "items": {"type": "number", "minimum": 0},
                                          "minItems": 2, "maxItems": 2,
                                          "description": "[between columns, between rows]; default "
                                                         f"{list(LAYOUT_GRID_GAP_M)}"},
                            },
                        },
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
                            "description": f"object id; wall name {list(WALLS)} for against_wall (indoor rect); "
                            f"'{NONRECT_WALL}' (+at_deg) or '{NONRECT_WALL}_<n>' for ellipse/polygon rooms; "
                            f"'{ROOM}' also allowed for centered_in",
                        },
                        "gap_m": {"type": "number", "minimum": 0,
                                  "description": "in_front_of/beside/around: free distance; above: B's top to A's base "
                                                 "(default 0.25)"},
                        "at_deg": {"type": "number", "description": "against_wall in ellipse/polygon rooms: angle on the wall"},
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


def _parse_setting(raw: Any, errors: list[str], notes: list[str] | None = None) -> Setting | None:
    if not isinstance(raw, dict):
        errors.append(f"setting: expected object, got {_typename(raw)}")
        return None
    n0 = len(errors)
    _unknown_keys("setting", raw, ("kind", "bounds_m", "style", "shape", "points", "floor_material",
                                   "wall_material", "rooms", "mouldings"), errors)
    mats = {}
    for key in ("floor_material", "wall_material"):
        if raw.get(key) is not None:
            mat, note = coerce_material(raw[key])
            if note and notes is not None:
                notes.append(f"setting.{key}: {note}")
            if mat:
                mats[key] = mat
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
    if style is not None and style_error(style):
        errors.append(f"setting.style: {style_error(style)}")
    shape = raw.get("shape", "rect")
    if shape not in SHAPES:
        errors.append(f"setting.shape: {shape!r} is not allowed; allowed values: {list(SHAPES)}")
    pts: list[tuple[float, float]] = []
    raw_pts = raw.get("points")
    if shape == "polygon":
        if not isinstance(raw_pts, list) or len(raw_pts) < 3:
            errors.append(f"setting.points: polygon needs a list of >= 3 [x, y] points, got {raw_pts!r}")
        else:
            for i, pt in enumerate(raw_pts):
                if not (isinstance(pt, list) and len(pt) == 2 and all(_is_num(v) for v in pt)):
                    errors.append(f"setting.points[{i}]: expected [x, y] numbers, got {pt!r}")
                elif bounds is not None and not (-1e-9 <= pt[0] <= bounds[0] + 1e-9 and -1e-9 <= pt[1] <= bounds[1] + 1e-9):
                    errors.append(f"setting.points[{i}]: {pt!r} lies outside bounds_m x/y [0, {bounds[0]:g}] x [0, {bounds[1]:g}]")
                else:
                    pts.append((float(pt[0]), float(pt[1])))
    elif raw_pts is not None:
        errors.append("setting.points: only allowed when setting.shape is 'polygon'")
    rooms = _parse_rooms(raw.get("rooms"), kind, shape, bounds, errors)
    mouldings = _parse_mouldings(raw.get("mouldings"), kind, rooms, errors)
    if len(errors) > n0 or bounds is None:
        return None
    return Setting(str(kind), bounds, style, str(shape), tuple(pts), mats.get("floor_material"),
                   mats.get("wall_material"), rooms, mouldings)


def _parse_mouldings(raw: Any, kind: Any, rooms: tuple[Room, ...], errors: list[str]) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or not all(isinstance(m, str) for m in raw):
        errors.append(f"setting.mouldings: expected a list of {list(MOULDINGS)}, got {raw!r}")
        return None
    bad = [m for m in raw if m not in MOULDINGS]
    if bad:
        errors.append(f"setting.mouldings: {bad} not allowed; allowed values: {list(MOULDINGS)}")
    if len(set(raw)) != len(raw):
        errors.append(f"setting.mouldings: duplicates in {raw!r}")
    if raw and (kind != "indoor" or rooms):
        errors.append("setting.mouldings: only for an indoor setting without rooms")
    return tuple(raw)


def _parse_rooms(raw: Any, kind: Any, shape: Any, bounds: tuple[float, float, float] | None,
                 errors: list[str]) -> tuple[Room, ...]:
    if raw is None:
        return ()
    if kind != "indoor" or shape != "rect":
        errors.append("setting.rooms: only allowed for an indoor rect setting")
        return ()
    if not isinstance(raw, list) or len(raw) < 2:
        errors.append(f"setting.rooms: expected a list of >= 2 rooms {{id, offset_m, size_m}}, got {raw!r}"[:300])
        return ()
    out: list[Room] = []
    for i, r in enumerate(raw):
        p = f"setting.rooms[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{p}: expected object, got {_typename(r)}")
            continue
        _unknown_keys(p, r, ("id", "offset_m", "size_m"), errors)
        rid, off, size = r.get("id"), r.get("offset_m"), r.get("size_m")
        if not isinstance(rid, str) or not _ID_RE.match(rid) or rid == ROOM:
            errors.append(f"{p}.id: {rid!r} must match {_ID_RE.pattern} and not be '{ROOM}'")
            continue
        if any(o.id == rid for o in out):
            errors.append(f"{p}.id: duplicate room id '{rid}'")
            continue
        if not (isinstance(off, list) and len(off) == 2 and all(_is_num(v) and v >= 0 for v in off)):
            errors.append(f"{p}.offset_m: expected [x, y] >= 0 in meters, got {off!r}")
            continue
        if not (isinstance(size, list) and len(size) == 2 and all(_is_num(v) and v > 0 for v in size)):
            errors.append(f"{p}.size_m: expected [width, depth] > 0 in meters, got {size!r}")
            continue
        if bounds is not None and (off[0] + size[0] > bounds[0] + 1e-6 or off[1] + size[1] > bounds[1] + 1e-6):
            errors.append(f"{p}: [{off[0]:g}, {off[1]:g}] + [{size[0]:g}, {size[1]:g}] leaves bounds_m "
                          f"[{bounds[0]:g}, {bounds[1]:g}]")
            continue
        room = Room(rid, (float(off[0]), float(off[1])), (float(size[0]), float(size[1])))
        for o in out:
            if (room.offset_m[0] < o.offset_m[0] + o.size_m[0] - 1e-6 and o.offset_m[0] < room.offset_m[0] + room.size_m[0] - 1e-6
                    and room.offset_m[1] < o.offset_m[1] + o.size_m[1] - 1e-6
                    and o.offset_m[1] < room.offset_m[1] + room.size_m[1] - 1e-6):
                errors.append(f"{p}: room '{rid}' overlaps room '{o.id}'")
        out.append(room)
    return tuple(out)


def _parse_objects(raw: Any, errors: list[str], notes: list[str] | None = None) -> list[ObjectSpec]:
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
        _unknown_keys(p, o, ("id", "category", "count", "hero", "detail", "grid", "material", "dynamic", "room",
                             "seats", "seat_category", "variant", "style", "turn_deg"), errors)
        variant, vnote = coerce_variant(o.get("variant", o.get("style")))
        if vnote and notes is not None:
            notes.append(f"{p}.variant: {vnote}")
        material = o.get("material")
        if material is not None:  # never refuse a spec over a material: coerce it, say so
            material, note = coerce_material(material)
            if note and notes is not None:
                notes.append(f"{p}.material: {note}")
        mat_ok = True
        grid, grid_ok = _parse_grid(p, o.get("grid"), o.get("count"), errors)
        oid, cat, hero = o.get("id"), o.get("category"), o.get("hero", False)
        count = o.get("count", grid[0] * grid[1] if grid else 1)
        detail = o.get("detail", "standard")
        ok = True
        if not isinstance(oid, str) or not _ID_RE.match(oid):
            errors.append(f"{p}.id: {oid!r} must match {_ID_RE.pattern} (lowercase snake_case)")
            ok = False
        elif oid in WALLS or oid == ROOM or oid == NONRECT_WALL or re.match(r"^wall_\d+$", oid):
            errors.append(f"{p}.id: '{oid}' is reserved (walls {list(WALLS)}, '{NONRECT_WALL}', '{NONRECT_WALL}_<n>' and '{ROOM}')")
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
        dynamic = o.get("dynamic", False)
        if not isinstance(dynamic, bool):
            errors.append(f"{p}.dynamic: expected boolean, got {dynamic!r}")
            ok = False
        if detail not in DETAILS:
            errors.append(f"{p}.detail: {detail!r} is not allowed; allowed values: {list(DETAILS)}")
            ok = False
        room = o.get("room")
        if room is not None and not isinstance(room, str):
            errors.append(f"{p}.room: expected a setting.rooms id, got {room!r}")
            ok = False
        seats, seat_cat = o.get("seats", 0), o.get("seat_category")
        if not isinstance(seats, int) or isinstance(seats, bool) or not 0 <= seats <= MAX_SEATS:
            errors.append(f"{p}.seats: expected integer in [0, {MAX_SEATS}], got {seats!r}")
            ok = False
        if seat_cat is not None and (not isinstance(seat_cat, str) or seat_cat not in cats):
            errors.append(f"{p}.seat_category: {seat_cat!r} is not a known category.{_suggest(str(seat_cat), cats)}")
            ok = False
        turn, tnote = coerce_turn(o.get("turn_deg", 0))
        if turn is None:
            errors.append(f"{p}.turn_deg: {tnote}")
            ok = False
        elif tnote and notes is not None:
            notes.append(f"{p}.turn_deg: {tnote}")
        if isinstance(oid, str):
            seen.add(oid)
        if ok and grid_ok and mat_ok:
            out.append(ObjectSpec(str(oid), str(cat), count, hero, str(detail), grid, material, dynamic, room,
                                  seats, seat_cat, variant, turn or 0))
    return out


def coerce_turn(v: Any) -> tuple[int | None, str | None]:
    """(quarter turn 0/90/180/270, note): -90 -> 270, 175 -> 180 (noted); None with the reason."""
    if not _is_num(v):
        return None, f"expected degrees (0, 90, 180 or 270), got {v!r}"
    turn = round(float(v) / 90.0) * 90 % 360
    return turn, None if abs(float(v) % 360 - turn) < 1e-9 else f"{v!r} read as {turn} (quarter turns only)"


def _parse_grid(p: str, raw: Any, count: Any, errors: list[str]) -> tuple[tuple[int, int, float, float] | None, bool]:
    if raw is None:
        return None, True
    if not isinstance(raw, dict):
        errors.append(f"{p}.grid: expected object {{rows, cols, gap_m}}, got {_typename(raw)}")
        return None, False
    _unknown_keys(f"{p}.grid", raw, ("rows", "cols", "gap_m"), errors)
    rows, cols, gap = raw.get("rows"), raw.get("cols"), raw.get("gap_m", list(LAYOUT_GRID_GAP_M))
    ok = True
    for name, v in (("rows", rows), ("cols", cols)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            errors.append(f"{p}.grid.{name}: expected integer >= 1, got {v!r}")
            ok = False
    if not (isinstance(gap, list) and len(gap) == 2 and all(_is_num(g) and g >= 0 for g in gap)):
        errors.append(f"{p}.grid.gap_m: expected [between columns, between rows] in meters, got {gap!r}")
        ok = False
    if not ok or not isinstance(rows, int) or not isinstance(cols, int):
        return None, False
    n = rows * cols
    if n > LAYOUT_MAX_COUNT:
        errors.append(f"{p}.grid: {rows}x{cols} = {n} instances, max {LAYOUT_MAX_COUNT}")
        return None, False
    if count is not None and count != n:
        errors.append(f"{p}.count: {count} does not match grid {rows}x{cols} = {n} (omit count)")
        return None, False
    return (rows, cols, float(gap[0]), float(gap[1])), True


def _parse_relations(raw: Any, ids: list[str], setting: Setting | None, errors: list[str]) -> list[Relation]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        errors.append(f"relations: expected a list, got {_typename(raw)}")
        return []
    kind = setting.kind if setting else ""
    shape = setting.shape if setting else "rect"
    walls = list(WALLS) if shape == "rect" else (
        [NONRECT_WALL] if shape == "ellipse" else [NONRECT_WALL, *polygon_wall_names(len(setting.points) if setting else 0)])
    out: list[Relation] = []
    for i, r in enumerate(raw):
        p = f"relations[{i}]"
        if not isinstance(r, dict):
            errors.append(f"{p}: expected object, got {_typename(r)}")
            continue
        _unknown_keys(p, r, ("a", "rel", "b", "gap_m", "at_deg"), errors)
        a, rel, b, gap = r.get("a"), r.get("rel"), r.get("b"), r.get("gap_m")
        at_deg = r.get("at_deg")
        n0 = len(errors)
        if rel not in RELATIONS:
            errors.append(f"{p}.rel: {rel!r} is not allowed; allowed values: {list(RELATIONS)}")
        if a not in ids:
            errors.append(f"{p}.a: {a!r} is not a declared object id.{_suggest(str(a), ids)} declared ids: {ids}")
        if rel == _WALL_REL:
            if kind != "indoor":
                errors.append(f"{p}: against_wall needs setting.kind 'indoor' (walls exist only indoors), got '{kind}'")
            elif b not in walls:
                room_note = "" if shape == "rect" else f" of this {shape} room"
                errors.append(f"{p}.b: {b!r} is not a wall{room_note}; allowed values: {walls}")
        elif rel in RELATIONS:
            allowed_b = ids + ([ROOM] if rel == "centered_in" else [])
            if b not in allowed_b:
                errors.append(f"{p}.b: {b!r} is not a valid target for '{rel}'.{_suggest(str(b), allowed_b)} allowed values: {allowed_b}")
        if a == b:
            errors.append(f"{p}: object '{a}' cannot be related to itself")
        if gap is not None and (not _is_num(gap) or gap < 0):
            errors.append(f"{p}.gap_m: expected number >= 0, got {gap!r}")
        if at_deg is not None and (not _is_num(at_deg) or rel != _WALL_REL or shape == "rect"):
            errors.append(f"{p}.at_deg: only allowed (as a number) on against_wall in ellipse/polygon rooms, got {at_deg!r}")
        if len(errors) == n0:
            out.append(Relation(str(a), str(rel), str(b), None if gap is None else float(gap), None if at_deg is None else float(at_deg)))
    return out


def _check_rooms(setting: Setting, objs: list[ObjectSpec], rels: list[Relation], errors: list[str]) -> None:
    """Every object names an existing room (only when there are rooms); relations stay in one room."""
    rooms = [r.id for r in setting.rooms]
    where = {o.id: o.room for o in objs}
    for o in objs:
        if not rooms and o.room is not None:
            errors.append(f"objects '{o.id}'.room: only allowed when setting.rooms is set")
        elif rooms and o.room not in rooms:
            errors.append(f"objects '{o.id}'.room: {o.room!r} is not a room; allowed values: {rooms}")
    for r in rels:
        if rooms and r.b in where and where[r.a] != where[r.b]:
            errors.append(f"relation {r.a} {r.rel} {r.b}: the objects are in different rooms "
                          f"({where[r.a]}, {where[r.b]}); relations stay inside one room")


def validate_spec(data: Any) -> tuple[SceneSpec | None, list[str]]:
    """Validate a dict. Returns (spec, []) or (None, errors); errors are 'path: problem; allowed'."""
    if not isinstance(data, dict):
        return None, [f"<root>: expected object, got {_typename(data)}"]
    from ai_db_scene.normalize import normalize_spec

    data, notes = normalize_spec(data)  # unambiguous model-side slips are rewritten, with a note
    errors: list[str] = []
    _unknown_keys("<root>", data, ("setting", "objects", "relations", "lighting", "camera"), errors)
    if "setting" not in data:
        errors.append("setting: required field is missing")
    if "objects" not in data:
        errors.append("objects: required field is missing")
    setting = _parse_setting(data["setting"], errors, notes) if "setting" in data else None
    objs_present = "objects" in data
    objs = _parse_objects(data.get("objects"), errors, notes) if objs_present else []
    raw_objs = data.get("objects")
    declared = [o["id"] for o in raw_objs if isinstance(o, dict) and isinstance(o.get("id"), str)] if isinstance(raw_objs, list) else []
    rels = _parse_relations(data.get("relations"), declared, setting, errors) if setting is not None or not errors else []
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
    if setting is not None:
        _check_rooms(setting, objs, rels, errors)
    if errors or setting is None:
        return None, errors
    return SceneSpec(setting, tuple(objs), tuple(rels), lighting, camera, tuple(notes)), []
