"""User part recipes: imported from text, validated like the built-ins, saved as JSON.

No model or binary file is ever downloaded or parsed here: a recipe comes from text the
agent read (a product page, a spec sheet, a ShapeAssembly program) and is checked with the
rules the built-in recipes follow (``recipes.py``): known part shapes and materials, extents
inside the unit box (``bbox_m`` scales it to the placement), and every part connected to the
ground through a chain of touching parts.

One recipe per category, in ``<recipes_dir>/<category>.json`` (``plugins.scene.recipes_dir``,
default ``$XDG_DATA_HOME/ai-db/scene_recipes``). A built-in recipe wins unless the user
recipe was saved with ``overwrite=true``.

- ``parse_shapeassembly``: the Cuboid/attach/squeeze/reflect/translate subset of
  ShapeAssembly (Jones et al. 2020) to parts; other statements are skipped with a warning.
- ``draft_recipe``: dimensions ("W 180 cm x D 90 cm x H 75 cm", inches, mm) and part words
  ("4 legs", "three shelves") from free text, and a connected parts skeleton to finish.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import time
from typing import Any

RECIPE_SHAPES = frozenset({"box", "cylinder", "sphere", "tapered_cylinder"})
_PART_KEYS = frozenset({"id", "shape", "size", "at", "material", "rotation_deg", "radius", "radius_top"})
_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PART_ID = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
RECIPE_MAX_PARTS = 300
RECIPE_MAX_PART_SIZE = 2.0  # relative units
RECIPE_MAX_DIM_M = 30.0
RECIPE_TOUCH = 0.006  # relative units: parts this close count as touching (as the recipe tests)
RECIPE_BOX_TOL = 0.03  # extents may overshoot the unit box by this much
RECIPE_MIN_SPAN = 0.5  # each axis must span at least this much of the unit box


def recipes_dir(app_config: Any | None = None) -> str:
    """Folder of user recipes: ``plugins.scene.recipes_dir`` or the XDG default."""
    from ai_db_scene.config import scene_config

    if app_config is None:
        from ai_db.config import load_config
        from ai_db.errors import AiDbConfigError

        try:
            app_config = load_config(check_env=False)
        except AiDbConfigError:
            app_config = None
    explicit = scene_config(app_config).recipes_dir if app_config is not None else None
    if explicit:
        return explicit
    data = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(data, "ai-db", "scene_recipes")


# --- validation -------------------------------------------------------------------------------

def part_box(part: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Axis-aligned bounds of a part as build_asset places it: size before rotation_deg (Euler
    XYZ), then the rotated bounds' bottom centre put at ``at``."""
    sx, sy, sz = part["size"]
    rx, ry, rz = (math.radians(a) for a in part.get("rotation_deg", (0, 0, 0)))

    def rot(p: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = p
        y, z = y * math.cos(rx) - z * math.sin(rx), y * math.sin(rx) + z * math.cos(rx)
        x, z = x * math.cos(ry) + z * math.sin(ry), -x * math.sin(ry) + z * math.cos(ry)
        x, y = x * math.cos(rz) - y * math.sin(rz), x * math.sin(rz) + y * math.cos(rz)
        return x, y, z

    pts = [rot(c) for c in itertools.product((-sx / 2, sx / 2), (-sy / 2, sy / 2), (-sz / 2, sz / 2))]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    ax, ay, az = part["at"]
    off = (ax - (lo[0] + hi[0]) / 2, ay - (lo[1] + hi[1]) / 2, az - lo[2])
    return [lo[i] + off[i] for i in range(3)], [hi[i] + off[i] for i in range(3)]


def floating_parts(parts: list[dict[str, Any]]) -> list[str]:
    """Ids of parts not connected to the ground through a chain of touching parts."""
    boxes = {p["id"]: part_box(p) for p in parts}
    if not boxes:
        return []
    ground = min(lo[2] for lo, _hi in boxes.values())
    reached = {pid for pid, (lo, _hi) in boxes.items() if lo[2] <= ground + RECIPE_TOUCH}
    grew = True
    while grew:
        grew = False
        for pid, (lo, hi) in boxes.items():
            if pid not in reached and any(
                    all(lo[i] <= boxes[q][1][i] + RECIPE_TOUCH and boxes[q][0][i] <= hi[i] + RECIPE_TOUCH
                        for i in range(3)) for q in reached):
                reached.add(pid)
                grew = True
    return sorted(set(boxes) - reached)


def _vec(v: Any, n: int = 3) -> bool:
    return isinstance(v, list) and len(v) == n and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)


def recipe_errors(parts: Any) -> list[str]:
    """Why ``parts`` is not a usable recipe (empty list: valid)."""
    from ai_db_scene.materials import material_error

    if not isinstance(parts, list) or not parts:
        return ["parts must be a non-empty list"]
    if len(parts) > RECIPE_MAX_PARTS:
        return [f"at most {RECIPE_MAX_PARTS} parts, got {len(parts)}"]
    errors: list[str] = []
    seen: set[str] = set()
    for k, p in enumerate(parts):
        where = f"parts[{k}]"
        if not isinstance(p, dict):
            errors.append(f"{where}: expected an object")
            continue
        unknown = set(p) - _PART_KEYS
        if unknown:
            errors.append(f"{where}: unknown keys {sorted(unknown)}; allowed {sorted(_PART_KEYS)}")
        pid = p.get("id")
        if not isinstance(pid, str) or not _PART_ID.match(pid):
            errors.append(f"{where}.id: expected a short id [A-Za-z0-9_.-], got {pid!r}")
        elif pid in seen:
            errors.append(f"{where}.id: duplicate {pid!r}")
        else:
            seen.add(pid)
        if p.get("shape") not in RECIPE_SHAPES:
            errors.append(f"{where}.shape: one of {sorted(RECIPE_SHAPES)}, got {p.get('shape')!r}")
        size = p.get("size")
        if not _vec(size) or not all(0 < x <= RECIPE_MAX_PART_SIZE for x in size or ()):
            errors.append(f"{where}.size: [x, y, z] in (0, {RECIPE_MAX_PART_SIZE}] relative units, got {size!r}")
        if not _vec(p.get("at")):
            errors.append(f"{where}.at: [x, y, z] (bottom centre, relative units), got {p.get('at')!r}")
        if "rotation_deg" in p and not _vec(p["rotation_deg"]):
            errors.append(f"{where}.rotation_deg: [x, y, z] degrees, got {p['rotation_deg']!r}")
        for key in ("radius", "radius_top"):
            if key in p and (not isinstance(p[key], (int, float)) or not 0 <= p[key] <= 1):
                errors.append(f"{where}.{key}: a number in [0, 1], got {p[key]!r}")
        err = material_error(p.get("material"))
        if err:
            errors.append(f"{where}.material: {err}")
    if errors:
        return errors
    boxes = [part_box(p) for p in parts]
    lo = [min(b[0][i] for b in boxes) for i in range(3)]
    hi = [max(b[1][i] for b in boxes) for i in range(3)]
    t = RECIPE_BOX_TOL
    if lo[0] < -0.5 - t or lo[1] < -0.5 - t or lo[2] < -t or hi[0] > 0.5 + t or hi[1] > 0.5 + t or hi[2] > 1 + t:
        errors.append(f"parts extend outside the unit box x,y in [-0.5, 0.5], z in [0, 1]: "
                      f"lo {[round(v, 3) for v in lo]}, hi {[round(v, 3) for v in hi]}")
    if abs(lo[2]) > t:
        errors.append(f"the lowest part must stand at z=0, lowest is at {lo[2]:.3f}")
    spans = [hi[i] - lo[i] for i in range(3)]
    if min(spans) < RECIPE_MIN_SPAN:
        errors.append(f"parts must fill the unit box (each axis spans >= {RECIPE_MIN_SPAN}), spans {[round(s, 3) for s in spans]}")
    floating = floating_parts(parts)
    if floating:
        errors.append(f"parts not connected to the ground through touching parts: {floating}")
    return errors


# --- store --------------------------------------------------------------------------------------

def _path(category: str, folder: str) -> str:
    if not isinstance(category, str) or not _NAME.match(category):
        raise ValueError(f"category must be a snake_case name, got {category!r}")
    return os.path.join(folder, f"{category}.json")


def load_user_recipe(category: str, folder: str | None = None) -> dict[str, Any] | None:
    try:
        path = _path(category, folder or recipes_dir())
    except ValueError:
        return None
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("parts"), list) else None


def list_user_recipes(folder: str | None = None) -> list[dict[str, Any]]:
    folder = folder or recipes_dir()
    out = []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            if fn.endswith(".json"):
                r = load_user_recipe(fn[:-5], folder)
                if r:
                    out.append({k: r.get(k) for k in ("category", "name", "source", "overrides_builtin", "dims_m")}
                               | {"parts": len(r["parts"])})
    return out


def _source(source: Any) -> dict[str, str]:
    if source is None:
        return {}
    if isinstance(source, str):
        source = {"url": source}
    if not isinstance(source, dict) or not all(isinstance(v, str) for v in source.values()) \
            or set(source) - {"url", "license", "title", "author"}:
        raise ValueError("source must be {url, license, title?, author?} strings")
    url = source.get("url", "")
    if url and not url.startswith(("http://", "https://")):
        raise ValueError(f"source.url must be an http(s) URL, got {url!r}")
    return dict(source)


def create_recipe(category: str, name: str, parts: Any, dims: Any = None, source: Any = None,
                  overwrite: bool = False, folder: str | None = None) -> dict[str, Any]:
    from ai_db_scene.priors import categories
    from ai_db_scene.recipes import _RECIPES

    if category not in set(categories()):
        raise ValueError(f"unknown category {category!r}; it needs a size prior first")
    if not isinstance(name, str) or not _NAME.match(name):
        raise ValueError(f"name must be a short snake/kebab-case name, got {name!r}")
    if dims is not None and (not _vec(dims) or not all(0 < x <= RECIPE_MAX_DIM_M for x in dims)):
        raise ValueError(f"dims must be [w, d, h] in meters (0, {RECIPE_MAX_DIM_M}], got {dims!r}")
    errors = recipe_errors(parts)
    if errors:
        raise ValueError("invalid recipe: " + "; ".join(errors[:12]))
    folder = folder or recipes_dir()
    path = _path(category, folder)
    builtin = category in _RECIPES
    if builtin and not overwrite:
        raise ValueError(f"{category!r} has a built-in recipe; pass overwrite=true to use yours instead")
    existing = load_user_recipe(category, folder)
    if existing and not overwrite:
        raise ValueError(f"a user recipe for {category!r} exists ({existing.get('name')}); pass overwrite=true")
    data = {"category": category, "name": name, "parts": parts,
            "dims_m": [float(x) for x in dims] if dims is not None else None,
            "source": _source(source), "overrides_builtin": builtin, "created_at": time.time()}
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)
    out = {"category": category, "name": name, "path": path, "parts": len(parts),
           "status": "updated" if existing else "created", "overrides_builtin": builtin}
    if not data["source"].get("license"):
        out["warning"] = "no source.license recorded: note where the recipe came from and its terms"
    return out


# --- ShapeAssembly ------------------------------------------------------------------------------

_SA_CALL = re.compile(r"^\s*(?:(\w+)\s*=\s*)?(\w+)\s*\((.*)\)\s*$")
_SA_FACES = {"top": (1, 1), "bot": (1, -1), "left": (0, -1), "right": (0, 1), "front": (2, 1), "back": (2, -1)}
_SA_AXES = {"X": 0, "Y": 1, "Z": 2}


class _Cub:
    def __init__(self, dims: list[float]) -> None:
        self.dims = dims  # ShapeAssembly l (x), h (y, up), w (z)
        self.center: list[float] | None = None

    def point(self, uvw: list[float]) -> list[float]:
        assert self.center is not None
        return [self.center[i] + (uvw[i] - 0.5) * self.dims[i] for i in range(3)]

    def lo(self, i: int) -> float:
        assert self.center is not None
        return self.center[i] - self.dims[i] / 2

    def hi(self, i: int) -> float:
        assert self.center is not None
        return self.center[i] + self.dims[i] / 2


def _sa_args(raw: str) -> list[str]:
    return [a.strip() for a in raw.split(",") if a.strip()]


def is_shapeassembly(text: str) -> bool:
    return bool(re.search(r"\bCuboid\s*\(", text)) and bool(re.search(r"\b(attach|squeeze)\s*\(", text))


def parse_shapeassembly(text: str, material: str = "wood:oak") -> tuple[list[dict[str, Any]], list[str]]:
    """Parts (our relative units, front -Y, z up) and warnings from a ShapeAssembly program.

    Supported: ``Cuboid(l, h, w, aligned)``, the first ``attach`` of a cuboid (later ones
    would rotate/stretch it: skipped), ``squeeze``, ``reflect`` and ``translate``. The
    ``bbox`` cuboid is the frame, centred at the origin; only the first ``def`` is read."""
    cubes: dict[str, _Cub] = {}
    order: list[str] = []
    warnings: list[str] = []
    defs = 0
    for n, line in enumerate(text.splitlines(), 1):
        s = line.split("#")[0].strip()
        if not s or s.startswith("@"):
            continue
        if s.startswith("def "):
            defs += 1
            if defs > 1:
                warnings.append(f"line {n}: sub-assembly {s[4:].split('(')[0]!r} not expanded (only the first def is read)")
            continue
        if defs > 1:
            continue
        m = _SA_CALL.match(s)
        if not m:
            warnings.append(f"line {n}: skipped {s[:60]!r}")
            continue
        var, fn, args = m.group(1), m.group(2), _sa_args(m.group(3))
        try:
            if fn == "Cuboid" and var:
                cubes[var] = _Cub([float(a) for a in args[:3]])
                if var == "bbox":
                    cubes[var].center = [0.0, 0.0, 0.0]
                else:
                    order.append(var)
                if len(args) > 3 and args[3] == "False":
                    warnings.append(f"line {n}: {var} is not axis-aligned; placed axis-aligned")
            elif fn == "attach":
                a, b, nums = cubes[args[0]], cubes[args[1]], [float(x) for x in args[2:8]]
                if a.center is not None:
                    warnings.append(f"line {n}: second attach of {args[0]} skipped (no rotation/stretch)")
                elif b.center is None:
                    warnings.append(f"line {n}: {args[1]} is not placed yet; attach skipped")
                else:
                    target, local = b.point(nums[3:6]), [(nums[i] - 0.5) * a.dims[i] for i in range(3)]
                    a.center = [target[i] - local[i] for i in range(3)]
            elif fn == "squeeze":
                a, b, c = cubes[args[0]], cubes[args[1]], cubes[args[2]]
                axis, sign = _SA_FACES[args[3]]
                u, v = float(args[4]), float(args[5])
                if b.center is None or c.center is None:
                    warnings.append(f"line {n}: squeeze between unplaced cuboids skipped")
                    continue
                if b.lo(axis) <= c.lo(axis) and c.hi(axis) <= b.hi(axis):  # b contains c (bbox)
                    lo, hi = (b.lo(axis), c.lo(axis)) if sign < 0 else (c.hi(axis), b.hi(axis))
                elif c.lo(axis) <= b.lo(axis) and b.hi(axis) <= c.hi(axis):
                    lo, hi = (c.lo(axis), b.lo(axis)) if sign < 0 else (b.hi(axis), c.hi(axis))
                else:
                    lo, hi = sorted((min(b.hi(axis), c.hi(axis)), max(b.lo(axis), c.lo(axis))))
                if hi - lo <= 0:
                    warnings.append(f"line {n}: squeeze of {args[0]} has no gap; skipped")
                    continue
                others = [i for i in range(3) if i != axis]
                a.dims[axis] = hi - lo
                a.center = [0.0, 0.0, 0.0]
                a.center[axis] = (lo + hi) / 2
                for i, t in zip(others, (u, v), strict=True):
                    a.center[i] = b.lo(i) + t * b.dims[i]
            elif fn == "reflect":
                a, axis = cubes[args[0]], _SA_AXES[args[1]]
                if a.center is None:
                    continue
                r = _Cub(list(a.dims))
                r.center = list(a.center)
                r.center[axis] = 2 * cubes["bbox"].center[axis] - a.center[axis]  # type: ignore[index]
                cubes[f"{args[0]}_r"] = r
                order.append(f"{args[0]}_r")
            elif fn == "translate":
                a, axis, count, dist = cubes[args[0]], _SA_AXES[args[1]], int(args[2]), float(args[3])
                if a.center is None:
                    continue
                for k in range(1, count + 1):
                    cp = _Cub(list(a.dims))
                    cp.center = list(a.center)
                    cp.center[axis] += dist * cubes["bbox"].dims[axis] * k / count
                    cubes[f"{args[0]}_t{k}"] = cp
                    order.append(f"{args[0]}_t{k}")
            else:
                warnings.append(f"line {n}: unsupported {fn}() skipped")
        except (KeyError, IndexError, ValueError) as e:
            warnings.append(f"line {n}: could not read {s[:60]!r} ({e})")
    bbox = cubes.get("bbox")
    if bbox is None:
        return [], [*warnings, "no 'bbox = Cuboid(...)' frame"]
    L, H, W = bbox.dims
    parts = []
    for name in order:
        c = cubes[name]
        if c.center is None:
            warnings.append(f"{name} was never attached; dropped")
            continue
        parts.append({"id": name, "shape": "box",
                      "size": [round(c.dims[0] / L, 4), round(c.dims[2] / W, 4), round(c.dims[1] / H, 4)],
                      "at": [round(c.center[0] / L, 4), round(-c.center[2] / W, 4),
                             round((c.center[1] - c.dims[1] / 2 + H / 2) / H, 4)],
                      "material": material})
    return parts, warnings


# --- free-text drafting -------------------------------------------------------------------------

_UNIT_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
           '"': 0.0254, "ft": 0.3048, "feet": 0.3048, "'": 0.3048}
_NUM = r"(\d+(?:[.,]\d+)?)"
_UNIT = r"(mm|cm|m|inches|inch|in|ft|feet|\"|')"
_LABELS = {"w": 0, "width": 0, "wide": 0, "l": 0, "length": 0, "long": 0,
           "d": 1, "depth": 1, "deep": 1, "h": 2, "height": 2, "high": 2, "tall": 2}
_LABELLED = re.compile(r"\b(width|depth|height|length|w|d|h|l)\b\.?\s*[:=]?\s*" + _NUM + r"\s*" + _UNIT + r"?(?![a-z])",
                       re.IGNORECASE)
_TRAILING = re.compile(_NUM + r"\s*" + _UNIT + r"?\s*(wide|deep|high|tall|long)\b", re.IGNORECASE)
_TRIPLE = re.compile(_NUM + r"\s*" + _UNIT + r"?\s*[x×*]\s*" + _NUM + r"\s*" + _UNIT + r"?\s*[x×*]\s*"
                     + _NUM + r"\s*" + _UNIT, re.IGNORECASE)
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
PART_WORDS = {"leg": "legs", "legs": "legs", "feet": "legs", "foot": "legs", "seat": "seat", "back": "back",
              "backrest": "back", "arm": "arms", "arms": "arms", "armrest": "arms", "armrests": "arms",
              "top": "top", "tabletop": "top", "shelf": "shelves", "shelves": "shelves", "drawer": "drawers",
              "drawers": "drawers", "door": "doors", "doors": "doors", "cushion": "cushions", "cushions": "cushions",
              "headboard": "headboard", "mattress": "mattress", "rung": "rungs", "rungs": "rungs",
              "stretcher": "rungs", "stretchers": "rungs", "handle": "handles", "handles": "handles",
              "base": "base", "plinth": "base"}


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _m(value: str, unit: str | None, default_unit: str | None) -> float:
    u = (unit or default_unit or "cm").lower()
    return _num(value) * _UNIT_M.get(u, 0.01)


def extract_dims(text: str) -> tuple[list[float] | None, list[str]]:
    """[w, d, h] in meters from the text (None if not found) and how they were read."""
    notes: list[str] = []
    dims: list[float | None] = [None, None, None]
    for m in _LABELLED.finditer(text):
        label, val, unit = m.groups()
        before = re.findall(r"[a-z]+", text[max(0, m.start() - 20):m.start()].lower())
        if before and before[-1] in PART_WORDS:  # "seat height 46 cm" is not the overall height
            notes.append(f"skipped {before[-1]} {label.lower()}")
            continue
        i = _LABELS[label.lower()]
        if dims[i] is None:
            # "W 180 x D 90 x H 75 cm": a unit later in the same run applies to all
            later = re.search(_UNIT + r"(?![a-z])", text[m.end():m.end() + 40], re.IGNORECASE)
            dims[i] = _m(val, unit or (later.group(1) if later else None), None)
    for val, unit, word in _TRAILING.findall(text):
        i = _LABELS[word.lower()]
        if dims[i] is None:
            dims[i] = _m(val, unit, None)
    if any(d is not None for d in dims):
        notes.append("labelled dimensions")
    if None in dims:
        triple = _TRIPLE.search(text)
        if triple:
            a, ua, b, ub, c, uc = triple.groups()
            vals = [_m(a, ua, uc), _m(b, ub, uc), _m(c, uc, uc)]
            dims = [d if d is not None else v for d, v in zip(dims, vals, strict=True)]
            notes.append("A x B x C read as width x depth x height")
    if None in dims:
        return None, notes + ["dimensions incomplete: give W, D and H"]
    return [round(float(d), 4) for d in dims if d is not None], notes


def extract_part_words(text: str) -> dict[str, int]:
    """Part words in the text with a count when one is stated ("4 legs", "three shelves")."""
    out: dict[str, int] = {}
    toks = re.findall(r"[a-z]+|\d+", text.lower())
    for k, word in enumerate(toks):
        if word not in PART_WORDS:
            continue
        count = 0
        for n in reversed(toks[max(0, k - 2):k]):  # "4 legs", "three adjustable shelves"
            count = int(n) if n.isdigit() and int(n) <= 50 else _WORDS.get(n, 0)
            if count:
                break
        key = PART_WORDS[word]
        out[key] = max(out.get(key, 0), count or 1)
    return out


def skeleton_parts(words: dict[str, int], material: str) -> list[dict[str, Any]]:
    """Connected placeholder parts for the part words (the agent refines sizes/positions)."""
    parts: list[dict[str, Any]] = []
    dark = "color:#2a2a2a"
    cabinet_like = ("doors" in words or "drawers" in words) and "legs" not in words
    if "shelves" in words or cabinet_like:
        if "shelves" in words:
            n = max(2, min(words["shelves"], 8))
            parts += [{"id": f"side_{s}", "shape": "box", "size": [0.04, 1.0, 1.0], "at": [x, 0, 0], "material": material}
                      for s, x in (("l", -0.48), ("r", 0.48))]
            parts += [{"id": f"shelf_{k}", "shape": "box", "size": [0.92, 1.0, 0.03],
                       "at": [0, 0, round(k * 0.97 / (n - 1), 4)], "material": material} for k in range(n)]
        else:
            parts.append({"id": "body", "shape": "box", "size": [1.0, 0.96, 1.0], "at": [0, 0.02, 0], "material": material})
            n = max(1, min(words.get("drawers", 0) or words.get("doors", 1), 8))
            parts += [{"id": f"handle_{k}", "shape": "box", "size": [0.14, 0.03, 0.02],
                       "at": [0, -0.475, round(1 - (k + 0.5) / n, 4)], "material": dark} for k in range(n)]
        return parts
    seat = "seat" in words
    leg_h = 0.45 if seat else 0.94
    n_legs = words.get("legs", 4 if ("top" in words or seat) else 0)
    if n_legs:
        corners = [(-0.45, -0.45), (0.45, -0.45), (-0.45, 0.45), (0.45, 0.45)][:max(1, min(n_legs, 4))]
        if n_legs == 1:
            corners = [(0.0, 0.0)]
        parts += [{"id": f"leg_{i}", "shape": "box", "size": [0.06, 0.06, leg_h], "at": [x, y, 0], "material": material}
                  for i, (x, y) in enumerate(corners)]
        if "rungs" in words:
            parts.append({"id": "rung", "shape": "box", "size": [0.9, 0.04, 0.04], "at": [0, -0.45, 0.2], "material": material})
    else:
        parts.append({"id": "base", "shape": "box", "size": [1.0, 1.0, leg_h], "at": [0, 0, 0], "material": material})
    top = "seat" if seat else "top"
    parts.append({"id": top, "shape": "box", "size": [1.0, 1.0, 0.06], "at": [0, 0, leg_h], "material": material})
    if seat and "back" in words:
        parts.append({"id": "back", "shape": "box", "size": [1.0, 0.08, 1 - leg_h - 0.06],
                      "at": [0, 0.46, leg_h + 0.06], "material": material})
    if seat and "arms" in words:
        parts += [{"id": f"arm_{s}", "shape": "box", "size": [0.08, 0.9, 0.2], "at": [x, -0.05, leg_h + 0.06],
                   "material": material} for s, x in (("l", -0.46), ("r", 0.46))]
    if "cushions" in words and seat:
        parts.append({"id": "cushion", "shape": "box", "size": [0.8, 0.8, 0.08], "at": [0, -0.05, leg_h + 0.06],
                      "material": "fabric:linen"})
    return parts


def default_material(category: str) -> str:
    from ai_db_scene.materials import material_for

    return material_for(category).split(":proj=")[0]


def draft_recipe(category: str, text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is required (the page or program as plain text)")
    material = default_material(category)
    if is_shapeassembly(text):
        parts, warnings = parse_shapeassembly(text, material)
        dims, notes = None, ["ShapeAssembly program"]
        words: dict[str, int] = {}
    else:
        dims, notes = extract_dims(text)
        words = extract_part_words(text)
        parts, warnings = skeleton_parts(words, material), []
    return {"category": category, "dims_m": dims, "part_words": words, "parts": parts,
            "issues": recipe_errors(parts), "warnings": warnings, "notes": notes,
            "next": "Adjust parts (relative units: x,y in [-0.5, 0.5], z in [0, 1], 'at' is the bottom "
                    "centre, front is -Y, parts must touch), then create_recipe(category, name, parts, "
                    "dims, source={url, license})."}
