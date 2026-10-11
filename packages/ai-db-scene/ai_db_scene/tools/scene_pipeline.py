"""MCP tools for the spec -> layout -> build pipeline (docs/3D_PIPELINE.md stages 1, 3, 4, 7).

Table-driven like scene_tools: (name, description, schema) rows plus handlers. Command
construction is pure (``build_layout_commands``) so it is testable without Blender.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ai_db.errors import AiDbConfigError, AiDbError
from ai_db_scene.blender_client import BlenderClient, BlenderError
from ai_db_scene.constants import (
    LAYOUT_DOOR_HEIGHT_M,
    SCENE_CAMERA_HEIGHT_M,
    SCENE_CAMERA_INSET,
    SCENE_EXPOSURE_INDOOR,
    SCENE_EXPOSURE_OUTDOOR,
    SCENE_EXPOSURE_PER_DOUBLING,
    SCENE_EXPOSURE_RANGE,
    SCENE_FILL_CEILING_GAP_M,
    SCENE_FILL_SIZE_FRACTION,
    SCENE_FILL_W_PER_M2,
    SCENE_FILL_W_RANGE,
    SCENE_LAMP_W,
    SCENE_REF_FLOOR_M2,
    SCENE_SUN_ENERGY,
)
from ai_db_scene.layout import LayoutError, Placement, solve_layout
from ai_db_scene.materials import (
    FLOOR_MATERIALS,
    STYLE_PALETTES,
    WALL_MATERIAL,
    material_for,
    pbr_preset,
    style_material,
)
from ai_db_scene.models import RUN_OUTCOMES, RunRecord
from ai_db_scene.priors import load_priors
from ai_db_scene.recipes import has_recipe, recipe_spec, user_overrides
from ai_db_scene.spec import RELATIONS, SETTING_KINDS, SceneSpec, spec_schema, validate_spec

WALL_THICKNESS_M = 0.1
_WALLS = ("wall_north", "wall_south", "wall_east", "wall_west")
_SUFFIX = re.compile(r"_\d+$")
_MAX_EXEMPLARS = 3


def _memory() -> Any:
    from ai_db_scene.tools.scene_tools import get_scene_memory

    return get_scene_memory()


def _err(message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "error", "message": message, **extra}


def _parse_spec(raw: Any) -> tuple[SceneSpec | None, list[str]]:
    return validate_spec(raw)


def parse_placements(raw: Any) -> list[Placement]:
    out = []
    for p in raw or []:
        out.append(
            Placement(
                id=str(p["id"]),
                location=tuple(float(v) for v in p["location"]),  # type: ignore[arg-type]
                rotation_deg_z=float(p.get("rotation_deg_z", 0.0)),
                dims=tuple(float(v) for v in p["dims"]),  # type: ignore[arg-type]
                category=str(p.get("category", "")),
                facing_target=p.get("facing_target"),
            )
        )
    return out


def placement_to_dict(p: Placement) -> dict[str, Any]:
    return {
        "id": p.id,
        "location": list(p.location),
        "rotation_deg_z": p.rotation_deg_z,
        "dims": list(p.dims),
        "category": p.category,
        "yaw_deg": p.rotation_deg_z,
        **({"facing_target": p.facing_target} if p.facing_target else {}),
    }


# --- pure command construction -------------------------------------------------


_OPENING_KINDS = ("window", "door", "niche")


def frame_offset(spec: SceneSpec) -> tuple[float, float, float]:
    """Layout frame -> Blender frame.

    solve_layout places objects in ``[0, w] x [0, d]``; the Blender addon's ``room_shell`` (and
    its validator's ``room_bounds``) centre a sized room on the origin. Without this shift
    every indoor object sat half a room away from its walls. Polygon rooms are drawn at
    their given points, so they keep the layout frame.
    """
    if spec.setting.kind == "single_asset" or spec.setting.shape == "polygon":
        return (0.0, 0.0, 0.0)
    w, d, _h = spec.setting.bounds_m
    return (-w / 2, -d / 2, 0.0)


def _shift(loc: Any, off: tuple[float, float, float]) -> list[float]:
    return [float(loc[0]) + off[0], float(loc[1]) + off[1], float(loc[2]) + off[2]]


def _openings(spec: SceneSpec, placements: list[Placement]) -> list[dict[str, Any]]:
    """Wall openings from window/door/niche objects that have an against_wall relation."""
    base_wall = {r.a: r for r in spec.relations if r.rel == "against_wall"}
    ids = {o.id for o in spec.objects}
    out = []
    for pl in placements:
        if pl.category not in _OPENING_KINDS:
            continue
        base = pl.id if pl.id in ids else _SUFFIX.sub("", pl.id)
        r = base_wall.get(base)
        if r is None:
            continue
        w, d, h = pl.dims
        # room_shell takes the opening's point ("at": snapped to the nearest wall) and its
        # sill; the solver already put the window/door object exactly there.
        out.append({
            "id": pl.id, "kind": pl.category, "wall": r.b,
            "at": _shift(pl.location, frame_offset(spec))[:2],
            "width": w, "height": h, "depth": d, "sill": max(pl.location[2] - h / 2, 0.0),
        })
    return out


def _room_commands(spec: SceneSpec, placements: list[Placement] | None = None) -> list[dict[str, Any]]:
    kind = spec.setting.kind
    if kind == "single_asset":
        return []
    w, d, h = spec.setting.bounds_m
    t = WALL_THICKNESS_M
    if kind == "indoor":
        st = spec.setting
        params: dict[str, Any] = {
            "shape": st.shape, "height": h, "wall_thickness": t,
            "openings": _openings(spec, placements or []),
            "floor": True, "ceiling": False, "trim": False,
        }
        if st.shape == "polygon":
            params["points"] = [list(p) for p in st.points]
        else:
            params["size"] = [w, d]
        return [{"type": "room_shell", "params": params}]
    shp = spec.setting.shape
    return [{
        "type": "create_primitive",
        "params": {"primitive": "cube", "name": "floor", "location": [0.0, 0.0, -t / 2], "scale": [w / 2, d / 2, t / 2]},
    }] if shp == "rect" else [{"type": "room_shell", "params": {
        "shape": shp, "height": 0.0, "wall_thickness": t, "openings": [], "floor": True, "ceiling": False, "trim": False,
        **({"points": [list(p) for p in spec.setting.points]} if shp == "polygon" else {"size": [w, d]})}}]


def _interior_wall_commands(spec: SceneSpec) -> list[dict[str, Any]]:
    """Multi-room buildings: a wall slab on every room edge inside the outer shell (shared with
    another room or facing empty space), with a door gap at each door (layout.room_doors) and a
    lintel above it from LAYOUT_DOOR_HEIGHT_M to the ceiling. A shared edge is built once: by the
    room on its west/south side; a room's west/south edges only get the parts no neighbour builds."""
    from ai_db_scene.layout import room_doors

    rooms = spec.setting.rooms
    if not rooms:
        return []
    w, d, h = spec.setting.bounds_m
    t, off = WALL_THICKNESS_M, frame_offset(spec)
    doors = room_doors(spec)
    mat = material_for("", spec.setting.wall_material or style_material(spec.setting.style, "wall") or WALL_MATERIAL)
    cmds: list[dict[str, Any]] = []

    def slab(name: str, k: int, line: float, a: float, b: float, z0: float, z1: float) -> None:
        o = 1 - k
        centre, half = [0.0, 0.0, (z0 + z1) / 2], [0.0, 0.0, (z1 - z0) / 2]
        centre[k], half[k] = line, t / 2
        centre[o], half[o] = (a + b) / 2, (b - a) / 2
        cmds.append({"type": "create_primitive", "params": {
            "primitive": "cube", "name": name, "location": _shift(centre, off), "scale": half}})
        cmds.append({"type": "apply_material", "params": {"name": name, "material": mat}})

    def minus(lo: float, hi: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
        spans = []
        for a, b in sorted(cuts):
            if b <= lo or a >= hi:
                continue
            spans.append((lo, max(lo, a)))
            lo = max(lo, b)
        spans.append((lo, hi))
        return [s for s in spans if s[1] - s[0] > 1e-3]

    for r in rooms:
        for k, side, far in ((0, "east", True), (1, "north", True), (0, "west", False), (1, "south", False)):
            o = 1 - k
            line = r.offset_m[k] + (r.size_m[k] if far else 0.0)
            if abs(line - (w, d)[k]) < 1e-6 or abs(line) < 1e-6:
                continue  # the outer shell already has this wall
            lo, hi = r.offset_m[o], r.offset_m[o] + r.size_m[o]
            if far:
                cuts = [(dr["at"][o] - dr["width"] / 2, dr["at"][o] + dr["width"] / 2) for dr in doors
                        if abs(dr["at"][k] - line) < 1e-6 and lo < dr["at"][o] < hi]
            else:  # the parts a neighbour's east/north edge on this line does not already build
                cuts = [(n.offset_m[o], n.offset_m[o] + n.size_m[o]) for n in rooms
                        if n is not r and abs(n.offset_m[k] + n.size_m[k] - line) < 1e-6]
            for i, (a, b) in enumerate(minus(lo, hi, cuts)):
                slab(f"wall_{r.id}_{side}_{i}", k, line, a, b, 0.0, h)
    top = min(LAYOUT_DOOR_HEIGHT_M, h)
    for dr in doors:
        if h - top > 1e-3:
            k = 0 if dr["axis"] == "y" else 1
            c = dr["at"][1 - k]
            slab(f"lintel_{dr['id']}", k, dr["at"][k], c - dr["width"] / 2, c + dr["width"] / 2, top, h)
    return cmds


def _lookup_asset(asset_map: dict[str, Any], pl: Placement, base_id: str) -> dict[str, Any] | None:
    for key in (pl.id, base_id, pl.category):
        v = asset_map.get(key)
        if isinstance(v, dict) and v.get("filepath") and v.get("asset_name"):
            return v
    return None


#: first command of every build batch. A name the batch is about to create may still be held by an
#: object outside the current scene (in another scene, or a leftover clear_scene could not remove):
#: Blender would then call the new object "sofa.001", and every later command addressing "sofa"
#: (modify_object, apply_material, the tags below) would hit the leftover. Move such holders aside.
_RESERVE = """
import bpy, json
scene = bpy.context.scene
moved = []
for name in json.loads({payload!r}):
    held = bpy.data.objects.get(name)
    if held is not None and scene.objects.get(name) is None:
        held.name = name + "_stale"
        moved.append(held.name)
result = {{"moved": moved}}
"""

#: after clear_scene in build_scene/edit_scene: drop every object and data block nothing uses any
#: more. clear_scene's own purge (an outliner operator) does nothing without a UI context, so
#: orphans piled up over edits and pushed rebuilt names to ".001".
_PURGE = """
import bpy
removed = 0
for _ in range(8):  # an orphan object frees its mesh, the mesh its material: repeat until stable
    n = 0
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.lights,
                 bpy.data.cameras, bpy.data.collections, bpy.data.node_groups, bpy.data.images):
        for blk in list(coll):
            if blk.users == 0 and not blk.use_fake_user:
                coll.remove(blk)
                n += 1
    removed += n
    if not n:
        break
result = {"purged": removed}
"""

_FINALIZE = """
import bpy, json, re
scene = bpy.context.scene
suffix = re.compile(r"[.][0-9]{{3,}}$")

def fresh(base):
    # the untagged object this batch made for `base` when Blender had to rename it ("base.001")
    for o in scene.objects:
        if o.get("ai_db_id") is None and o.name.startswith(base + ".") and suffix.search(o.name):
            return o
    return None

def claim(obj, name):
    held = bpy.data.objects.get(name)
    if held is not None and held is not obj and scene.objects.get(name) is None:
        held.name = name + "_stale"  # a leftover outside the scene: the scene object gets the name
    obj.name = name

for ent in json.loads({payload!r}):
    obj = scene.objects.get(ent["name"])
    if obj is not None and obj.get("ai_db_id") not in (None, ent["name"]):
        obj = None  # an older object of another id
    if obj is None:
        obj = fresh(ent.get("asset_name") or ent["name"])
        if obj is None and ent.get("asset_name"):
            obj = scene.objects.get(ent["asset_name"])
            obj = obj if obj is not None and obj.get("ai_db_id") is None else None
        if obj is not None:
            claim(obj, ent["name"])
    if obj is None:
        continue
    if ent.get("asset_name"):
        obj.location = ent["location"]
        obj.rotation_euler[2] = ent["rot_z"]
    for k, v in ent["props"].items():
        obj[k] = v
result = {{"tagged": len(json.loads({payload!r}))}}
"""


_ROOM_MATERIAL = """
import bpy
mat = bpy.data.materials.get("ai_db_room") or bpy.data.materials.new("ai_db_room")
mat.diffuse_color = (0.82, 0.8, 0.76, 1.0)
if mat.use_nodes and mat.node_tree.nodes.get("Principled BSDF"):
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.82, 0.8, 0.76, 1.0)
n = 0
for o in bpy.context.scene.objects:
    if o.type == "MESH" and (o.name.startswith("Room_") or o.name == "floor") and not any(o.data.materials):
        o.data.materials.append(mat)
        n += 1
result = {"room_surfaces": n}
"""


#: category -> (generator family, params from the placement's [w, d, h]). Parametric furniture
#: with its own part materials (fabric seat, wooden legs) instead of a textured box.
_GENERATED: dict[str, tuple[str, Any]] = {
    "sofa": ("sofa", lambda w, d, h: {"width": w, "depth": d}),
    "armchair": ("armchair", lambda w, d, h: {"width": w, "depth": d}),
    "dining_chair": ("dining_chair", lambda w, d, h: {"seat_width": w * 0.9, "seat_depth": d * 0.85}),
    "chair": ("dining_chair", lambda w, d, h: {"seat_width": w * 0.9, "seat_depth": d * 0.85}),
    "dining_table": ("table", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "coffee_table": ("table", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "side_table": ("table", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "conference_table": ("table", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "desk": ("desk", lambda w, d, h: {"width": w, "depth": d, "height": h}),
    "bookshelf": ("bookshelf", lambda w, d, h: {"width": w, "depth": d, "height": h}),
}
GEOMETRY_MODES = ("generated", "boxes")
#: parts a spec material must not repaint on generated furniture (wooden legs, the books)
_KEEP_PARTS = {"sofa": ("leg",), "armchair": ("leg",), "bookshelf": ("book",)}

#: after import_file into collection "ai_db_asset_<id>": one root named <id> (an empty parenting
#: the file's roots), scaled uniformly to fit the placement, base centre on the floor point,
#: turned to the placement's yaw. Files come at any scale and origin; the layout does not.
_ADOPT = """
import bpy, json
from mathutils import Vector
ent = json.loads({payload!r})
coll = bpy.data.collections.get(ent["collection"])
objs = list(coll.objects) if coll else []
roots = [o for o in objs if o.parent not in objs]
root = bpy.data.objects.new(ent["name"], None)
(coll or bpy.context.scene.collection).objects.link(root)
for o in roots:
    o.parent = root
bpy.context.view_layer.update()
pts = [o.matrix_world @ Vector(c) for o in objs if o.type == "MESH" for c in o.bound_box]
if pts:
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    size = hi - lo
    s = min(ent["dims"][i] / size[i] for i in range(3) if size[i] > 1e-6)
    for o in roots:  # recentre the content on the root's origin: XY centre, base at z=0
        o.location -= Vector(((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, lo.z))
    root.scale = (s, s, s)
root.location = ent["location"]
root.rotation_euler[2] = ent["rot_z"]
result = {{"root": root.name, "parts": len(objs)}}
"""


_RESKIN = """
import bpy, sys, json
mod = next(m for n, m in sys.modules.items() if n.endswith(".materials") and hasattr(m, "resolve_material"))
done = {{}}
for ent in json.loads({payload!r}):
    root = bpy.data.objects.get(ent["name"])
    if root is None:
        continue
    mat = mod.resolve_material(ent["material"])
    parts = [o for o in [root, *root.children_recursive] if o.type == "MESH"
             and not any(k in o.name.lower() for k in ent["keep"])]
    for o in parts:
        o.data.materials.clear()
        o.data.materials.append(mat)
    done[ent["name"]] = len(parts)
result = {{"reskinned": done}}
"""


_WEAR = """
import json
for ent in json.loads(PAYLOAD):
    root = bpy.data.objects.get(ent["name"])
    if root is None:
        continue
    copies = {}
    for o in [root, *root.children_recursive]:
        for slot in (o.material_slots if o.type == "MESH" else ()):
            m = slot.material
            if m is None or not m.use_nodes or not any(n.type == "BSDF_PRINCIPLED" for n in m.node_tree.nodes):
                continue
            if m.name not in copies:  # library materials are shared: wear a copy
                worn = m.copy()
                worn.name = f"{m.name}_{ent['preset']}"
                copies[m.name] = add_grunge(worn, amount=ent["amount"], scale=ent["scale"])
            slot.material = copies[m.name]
    worn_names.append(ent["name"])
result = {"worn": worn_names}
"""


def _wear_command(wear: list[dict[str, Any]]) -> dict[str, Any]:
    """run_code: the pbr-shader-builder skill's library, then its grunge on each listed object."""
    from ai_db_scene import skills_dir

    text = (Path(skills_dir()) / "pbr-shader-builder" / "SKILL.md").read_text(encoding="utf-8")
    m = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    if m is None:
        raise AiDbError("pbr-shader-builder skill has no code block")
    code = m.group(1) + f"PAYLOAD = {json.dumps(wear)!r}\nworn_names = []\n" + _WEAR
    return {"type": "run_code", "params": {"code": code}}


def build_layout_commands(
    spec: SceneSpec, placements: list[Placement], asset_map: dict[str, Any] | None = None,
    geometry: str = "generated", providers: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Blender batch for room shell + one object per placement. Returns (commands, created names).

    Per placement, the first that applies: an ``asset_map`` entry; with ``geometry="generated"``
    a model file from an asset provider (see providers.py; none unless configured), parametric
    furniture for the categories in ``_GENERATED`` (unless a user recipe was saved for it with
    overwrite=true), a part recipe; else a textured box.
    """
    import math

    from ai_db_scene.providers import AssetFile, find_asset

    asset_map = asset_map or {}
    provided: dict[str, AssetFile | None] = {}

    def provider_file(category: str) -> AssetFile | None:
        if geometry != "generated" or not providers:
            return None
        if category not in provided:  # one lookup (and download) per category per build
            provided[category] = find_asset(providers, category)
        return provided[category]

    priors = load_priors()
    heroes = {o.id for o in spec.objects if o.hero}
    # the style's palette counts as a choice: it wins over generator/category defaults
    materials = {o.id: o.material or style_material(spec.setting.style, o.category) for o in spec.objects}
    details = {o.id: o.detail for o in spec.objects}
    cmds = _room_commands(spec, placements) + _interior_wall_commands(spec)
    names = [c["params"].get("name", "room_shell") for c in cmds if c["type"] != "apply_material"]
    if placements:
        reserve = [pl.id for pl in placements] + [
            a["asset_name"] for pl in placements
            if (a := _lookup_asset(asset_map, pl, _SUFFIX.sub("", pl.id))) is not None]
        reserve += [f.asset_name for pl in placements
                    if (f := provider_file(pl.category)) is not None and f.asset_name]
        if spec.setting.kind != "single_asset":  # shell surfaces and the fill are addressed by name too
            reserve += ["Room_Floor", "Room_Walls", "floor", "ai_db_fill"]
        cmds.insert(0, {"type": "run_code", "params": {"code": _RESERVE.format(payload=json.dumps(reserve))}})
    off = frame_offset(spec)
    reskin: list[dict[str, Any]] = []
    tag: list[dict[str, Any]] = []
    for pl in placements:
        base = _SUFFIX.sub("", pl.id) if pl.id not in {o.id for o in spec.objects} else pl.id
        prior = priors.get(pl.category)
        props = {
            "ai_db_id": pl.id,
            "category": pl.category,
            "hero": base in heroes,
            "support": prior.support if prior else "floor",
            "detail": details.get(base, "standard"),
        }
        if pl.facing_target:
            props["facing_target"] = pl.facing_target
        asset = _lookup_asset(asset_map, pl, base)
        found = None if asset else provider_file(pl.category)
        if found is not None and found.kind == "blend":
            asset = {"filepath": found.filepath, "asset_name": found.asset_name}
        if found is not None:
            props["asset_source"] = found.source
        if found is not None and found.kind == "gltf":
            loc = _shift(pl.location, off)
            coll = f"ai_db_asset_{pl.id}"
            cmds.append({"type": "import_file", "params": {"filepath": found.filepath, "collection": coll}})
            cmds.append({"type": "run_code", "params": {"code": _ADOPT.format(payload=json.dumps({
                "name": pl.id, "collection": coll, "dims": list(pl.dims),
                "location": [loc[0], loc[1], loc[2] - pl.dims[2] / 2],
                "rot_z": math.radians(pl.rotation_deg_z)}))}})
            tag.append({"name": pl.id, "location": loc, "rot_z": 0.0, "props": props})
            if materials.get(base):
                reskin.append({"name": pl.id, "material": material_for(pl.category, materials[base]), "keep": []})
        elif asset:
            cmds.append({"type": "append_asset", "params": {
                "filepath": asset["filepath"], "asset_name": asset["asset_name"], "location": _shift(pl.location, off)}})
            tag.append({"name": pl.id, "asset_name": asset["asset_name"], "location": _shift(pl.location, off),
                        "rot_z": math.radians(pl.rotation_deg_z), "props": props})
        elif geometry == "generated" and pl.category in _GENERATED and not user_overrides(pl.category):
            family, params = _GENERATED[pl.category]
            loc = _shift(pl.location, off)
            cmds.append({"type": "generate_asset", "params": {
                "family": family, "params": params(*pl.dims), "name": pl.id,
                "location": [loc[0], loc[1], loc[2] - pl.dims[2] / 2],  # base centre
                "rotation": [0.0, 0.0, pl.rotation_deg_z]}})
            tag.append({"name": pl.id, "location": loc, "rot_z": 0.0, "props": props})
            if materials.get(base):  # the spec chose a material: it wins over the generator's
                reskin.append({"name": pl.id, "material": material_for(pl.category, materials[base]),
                               "keep": list(_KEEP_PARTS.get(pl.category, ()))})
        elif geometry == "generated" and has_recipe(pl.category):
            loc = _shift(pl.location, off)
            cmds.append({"type": "build_asset", "params": {"spec": recipe_spec(pl.category, pl.id, pl.dims)}})
            cmds.append({"type": "modify_object", "params": {
                "name": pl.id, "location": [loc[0], loc[1], loc[2] - pl.dims[2] / 2],
                "rotation": [0.0, 0.0, pl.rotation_deg_z]}})
            tag.append({"name": pl.id, "location": loc, "rot_z": 0.0, "props": props})
            if materials.get(base):
                reskin.append({"name": pl.id, "material": material_for(pl.category, materials[base]), "keep": []})
        else:
            cmds.append({"type": "create_primitive", "params": {
                "primitive": "cube", "name": pl.id, "location": _shift(pl.location, off),
                "scale": [pl.dims[0] / 2, pl.dims[1] / 2, pl.dims[2] / 2]}})
            if pl.rotation_deg_z:
                cmds.append({"type": "modify_object", "params": {
                    "name": pl.id, "rotation": [0.0, 0.0, pl.rotation_deg_z]}})
            tag.append({"name": pl.id, "location": _shift(pl.location, off), "rot_z": 0.0, "props": props})
            cmds.append({"type": "apply_material", "params": {
                "name": pl.id, "material": material_for(pl.category, materials.get(base))}})
        names.append(pl.id)
    if tag:
        cmds.append({"type": "run_code", "params": {"code": _FINALIZE.format(payload=json.dumps(tag))}})
    if reskin:
        cmds.append({"type": "run_code", "params": {"code": _RESKIN.format(payload=json.dumps(reskin))}})
    if spec.setting.kind != "single_asset":
        st = spec.setting
        indoor_style = st.style if st.kind == "indoor" else None  # outdoor ground stays ground
        floor = st.floor_material or style_material(indoor_style, "floor") or FLOOR_MATERIALS.get(st.kind, "plaster")
        surfaces = [("floor" if st.kind != "indoor" and st.shape == "rect" else "Room_Floor", floor)]
        if st.kind == "indoor":
            surfaces.append(("Room_Walls", st.wall_material or style_material(st.style, "wall") or WALL_MATERIAL))
        for obj, mat in surfaces:
            cmds.append({"type": "apply_material", "params": {"name": obj, "material": material_for("", mat)}})
        # anything else of the shell (ceiling, trim) gets the neutral room material
        cmds.append({"type": "run_code", "params": {"code": _ROOM_MATERIAL}})
    ids = {o.id for o in spec.objects}
    wear = [{"name": pl.id, "preset": chosen[4:], "amount": p[1], "scale": p[2]}
            for pl in placements
            if (chosen := materials.get(pl.id if pl.id in ids else _SUFFIX.sub("", pl.id)))
            and (p := pbr_preset(chosen))]
    if spec.setting.kind != "single_asset":
        wear += [{"name": obj, "preset": mat[4:], "amount": p[1], "scale": p[2]}
                 for obj, mat in surfaces if (p := pbr_preset(mat))]
    if wear:
        cmds.append(_wear_command(wear))
    cmds += lighting_commands(spec, placements, off)
    cmds.append(camera_command(spec, placements, off))
    return cmds, names


def floor_area_m2(spec: SceneSpec) -> float:
    """Floor area of the setting: rectangle, ellipse inscribed in the bounds, or polygon."""
    import math

    st = spec.setting
    w, d, _h = st.bounds_m
    if st.shape == "polygon" and len(st.points) >= 3:
        pts = st.points
        return abs(sum(pts[i][0] * pts[i - 1][1] - pts[i - 1][0] * pts[i][1] for i in range(len(pts)))) / 2
    if st.shape == "ellipse":
        return math.pi * w * d / 4
    return w * d


def _room_centre(spec: SceneSpec, off: tuple[float, float, float]) -> tuple[float, float]:
    st = spec.setting
    if st.shape == "polygon" and st.points:
        xs, ys = [p[0] for p in st.points], [p[1] for p in st.points]
        return ((min(xs) + max(xs)) / 2 + off[0], (min(ys) + max(ys)) / 2 + off[1])
    w, d, _h = st.bounds_m
    return (w / 2 + off[0], d / 2 + off[1])


def indoor_exposure(area_m2: float) -> float:
    """Exposure (stops) for a room: windows light a small room well, the middle of a big one
    much less, so each doubling of floor area over the reference adds a fraction of a stop."""
    import math

    lo, hi = SCENE_EXPOSURE_RANGE
    stops = SCENE_EXPOSURE_INDOOR + SCENE_EXPOSURE_PER_DOUBLING * math.log2(max(area_m2, 1.0) / SCENE_REF_FLOOR_M2)
    return round(min(max(stops, lo), hi), 2)


_AREA_SHAPE = """
import bpy
o = bpy.data.objects.get({name!r})
if o is not None and o.type == "LIGHT" and o.data.type == "AREA":
    o.data.shape = "RECTANGLE"
    o.data.size, o.data.size_y = {sx!r}, {sy!r}
result = {{"fill": o.name if o else None}}
"""


def lighting_commands(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float]) -> list[dict[str, Any]]:
    """Daylight (sky world, sun, a portal per window), an interior fill, and a warm point light in
    every lamp. A lone sun left interiors flat and dark (judges' second most common complaint),
    and window light alone does not reach the middle of a large room: indoors the exposure grows
    with the floor area and a soft ceiling area light (energy by floor area) fills the room."""
    indoor = spec.setting.kind == "indoor"
    area = floor_area_m2(spec)
    cmds: list[dict[str, Any]] = [{"type": "daylight_rig", "params": {
        "sun_elevation": 40.0, "sun_azimuth": 135.0, "strength": SCENE_SUN_ENERGY, "sky": "nishita",
        "window_portals": indoor, "sky_strength": 0.8, "name": "ai_db_daylight",
        "exposure": indoor_exposure(area) if indoor else SCENE_EXPOSURE_OUTDOOR}}]
    if indoor:
        w, d, h = spec.setting.bounds_m
        lo, hi = SCENE_FILL_W_RANGE
        cx, cy = _room_centre(spec, off)
        cmds.append({"type": "add_light", "params": {
            "light_type": "AREA", "name": "ai_db_fill", "energy": round(min(max(SCENE_FILL_W_PER_M2 * area, lo), hi), 1),
            "color": [1.0, 0.96, 0.9], "location": [cx, cy, h - SCENE_FILL_CEILING_GAP_M],
            "rotation": [0.0, 0.0, 0.0]}})  # shines along -Z: down
        cmds.append({"type": "run_code", "params": {"code": _AREA_SHAPE.format(
            name="ai_db_fill", sx=round(w * SCENE_FILL_SIZE_FRACTION, 3), sy=round(d * SCENE_FILL_SIZE_FRACTION, 3))}})
    for pl in placements:
        watts = SCENE_LAMP_W.get(pl.category)
        if watts is None:
            continue
        x, y, z = _shift(pl.location, off)
        cmds.append({"type": "add_light", "params": {
            "light_type": "POINT", "name": f"{pl.id}_light", "energy": watts, "color": [1.0, 0.82, 0.62],
            "location": [x, y, z + pl.dims[2] / 2 - 0.15]}})
    return cmds


def _inside_polygon(x: float, y: float, pts: Any) -> bool:
    inside = False
    for i in range(len(pts)):
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def footprint(spec: SceneSpec, off: tuple[float, float, float], n: int = 48) -> list[list[float]]:
    """The room outline in the Blender frame, counter-clockwise (an ellipse as n points)."""
    import math

    st = spec.setting
    w, d, _h = st.bounds_m
    if st.shape == "polygon" and len(st.points) >= 3:
        pts = [[float(p[0]) + off[0], float(p[1]) + off[1]] for p in st.points]
        area = sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts)))
        return pts if area > 0 else pts[::-1]
    if st.shape == "ellipse":
        return [[w / 2 * (1 + math.cos(2 * math.pi * k / n)) + off[0], d / 2 * (1 + math.sin(2 * math.pi * k / n)) + off[1]]
                for k in range(n)]
    return [[off[0], off[1]], [w + off[0], off[1]], [w + off[0], d + off[1]], [off[0], d + off[1]]]


def _camera_spots(spec: SceneSpec) -> list[tuple[float, float]]:
    """Candidate camera spots (layout frame), all inside the room: the bounding box's corners for a
    rectangle (or outdoors), points on a shrunken outline for an ellipse or polygon (a corner of an
    oval room's bounding box is outside its walls: judges saw the room from outside)."""
    import math

    st = spec.setting
    w, d, _h = st.bounds_m
    i = SCENE_CAMERA_INSET
    if st.kind != "indoor" or st.shape == "rect" or (st.shape == "polygon" and len(st.points) < 3):
        return [(fxr * w, fyr * d) for fxr in (i, 1 - i) for fyr in (i, 1 - i)]
    if st.shape == "ellipse":
        r = 1 - 2 * i
        return [(w / 2 * (1 + r * math.cos(math.radians(a))), d / 2 * (1 + r * math.sin(math.radians(a))))
                for a in (45, 135, 225, 315)]
    pts = st.points
    cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    spots = [(p[0] + (cx - p[0]) * 2 * i, p[1] + (cy - p[1]) * 2 * i) for p in pts]
    return [sp for sp in spots if _inside_polygon(sp[0], sp[1], pts)] or [(cx, cy)]


def camera_poses(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float]) -> list[dict[str, Any]]:
    """set_camera params for every candidate (spot x eye height, then a higher one), best guess
    first: clear of furniture, then the farthest from the focus (the widest view of it), aimed at
    the heroes (else at everything)."""
    import math

    h = spec.setting.bounds_m[2]
    heroes = {o.id for o in spec.objects if o.hero}
    focus = [p for p in placements if _SUFFIX.sub("", p.id) in heroes or p.id in heroes] or placements
    fx = sum(p.location[0] for p in focus) / len(focus)
    fy = sum(p.location[1] for p in focus) / len(focus)
    indoor = spec.setting.kind == "indoor"
    eye = SCENE_CAMERA_HEIGHT_M if indoor else 2.5
    heights = [eye, round(min(eye + 0.8, h - 0.3), 2) if indoor else eye + 2.0]

    def clearance(cx: float, cy: float) -> float:
        return min((max(abs(cx - p.location[0]) - p.dims[0] / 2, abs(cy - p.location[1]) - p.dims[1] / 2, 0.0)
                    for p in placements if p.dims[2] > 0.05), default=99.0)

    spots = sorted(_camera_spots(spec), key=lambda c: (min(clearance(*c), 0.6), math.hypot(c[0] - fx, c[1] - fy)),
                   reverse=True)
    poses = []
    for height in dict.fromkeys(heights):
        for cx, cy in spots:
            dx, dy, dz = fx - cx, fy - cy, 0.8 - height
            yaw = math.degrees(math.atan2(-dx, dy))  # Euler Z: [90, 0, 0] looks along +Y
            pitch = 90.0 + math.degrees(math.atan2(dz, math.hypot(dx, dy)))
            poses.append({"location": _shift((cx, cy, height), off), "rotation": [round(pitch, 2), 0.0, round(yaw, 2)],
                          "name": "ai_db_camera", "lens": 16 if indoor else 24})
    return poses


def camera_command(spec: SceneSpec, placements: list[Placement], off: tuple[float, float, float]) -> dict[str, Any]:
    """An eye-level camera in the room corner with the clearest view, aimed at the heroes (else
    at everything). The old camera looked in from above the walls: walls and framing were the
    judges' most common complaints."""
    if spec.setting.kind in ("single_asset", "tabletop") or not placements:
        # small subjects: a close three-quarter view that frames them, not a room corner
        direction = [0.8, -1.0, 0.5] if spec.setting.kind == "single_asset" else [0.6, -1.0, 0.7]
        return {"type": "frame_camera", "params": {"direction": direction, "margin": 1.2, "lens": 50}}
    return {"type": "set_camera", "params": camera_poses(spec, placements, off)[0]}


#: is the scene camera inside the room, and which targets does it see (in frame, not hidden behind
#: other objects: rays from the camera to the centre and corners of each target's box)? Failing,
#: every candidate pose is scored and the best kept.
_COVERAGE = """
import bpy, json, math
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view
cfg = json.loads({payload!r})
scene = bpy.context.scene
cam = scene.camera
roots = {{o.get("ai_db_id"): o for o in scene.objects if o.get("ai_db_id") in cfg["targets"]}}

def owner(o):
    while o is not None and o.get("ai_db_id") is None:
        o = o.parent
    return o

def box_points(o):
    cs = [m.matrix_world @ Vector(c) for m in [o, *o.children_recursive] if m.type == "MESH" for c in m.bound_box]
    if not cs:
        return [o.matrix_world.translation.copy()]
    lo = Vector([min(c[i] for c in cs) for i in range(3)])
    hi = Vector([max(c[i] for c in cs) for i in range(3)])
    mid = (lo + hi) / 2
    return [mid] + [mid + 0.7 * (Vector((x, y, z)) - mid) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]

points = {{t: box_points(o) for t, o in roots.items()}}

def inside(loc):
    fp = cfg["footprint"]
    if fp is None:
        return True
    x, y, ok = loc[0], loc[1], False
    for i in range(len(fp)):
        (x0, y0), (x1, y1) = fp[i - 1], fp[i]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            ok = not ok
    return ok and 0.0 < loc[2] < cfg["height"]

def score():
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    origin = cam.matrix_world.translation.copy()
    seen = []
    for t, pts in points.items():
        n = 0
        for p in pts:
            co = world_to_camera_view(scene, cam, p)
            if not (0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0 and co.z > 0.0):
                continue
            v = p - origin
            hit, _l, _n, _i, obj, _m = scene.ray_cast(dg, origin, v.normalized(), distance=max(v.length - 0.02, 0.0))
            if not hit or owner(obj) is roots[t]:
                n += 1
        if n * 3 >= len(pts):
            seen.append(t)
    return inside(origin), seen

def pose(p):
    cam.location = p["location"]
    cam.rotation_euler = [math.radians(a) for a in p["rotation"]]
    cam.data.lens = p["lens"]

if cam is None:
    result = {{"skipped": "no scene camera"}}
else:
    best = (*score(), 0)
    tried = 1
    if not (best[0] and len(best[1]) == len(points)):
        for k, p in enumerate(cfg["poses"][1:], 1):
            pose(p)
            got = (*score(), k)
            tried += 1
            if (got[0], len(got[1])) > (best[0], len(best[1])):
                best = got
            if got[0] and len(got[1]) == len(points):
                break
        pose(cfg["poses"][best[2]])
    result = {{"inside": best[0], "visible_heroes": best[1], "targets": sorted(points), "tried": tried, "pose": best[2]}}
"""


def coverage_targets(spec: SceneSpec, placements: list[Placement], n: int = 3) -> list[str]:
    """Hero placement ids, or the n largest objects (openings and lamps' lights aside)."""
    heroes = {o.id for o in spec.objects if o.hero}
    hero_ids = [p.id for p in placements if p.id in heroes or _SUFFIX.sub("", p.id) in heroes]
    if hero_ids:
        return hero_ids
    solid = [p for p in placements if p.category not in _OPENING_KINDS]
    return [p.id for p in sorted(solid, key=lambda p: -(p.dims[0] * p.dims[1] * p.dims[2]))[:n]]


def camera_check(spec: SceneSpec, placements: list[Placement]) -> dict[str, Any] | None:
    """Run _COVERAGE for a room build (None for close-up kinds framed by frame_camera)."""
    if spec.setting.kind in ("single_asset", "tabletop") or not placements:
        return None
    off = frame_offset(spec)
    payload = json.dumps({
        "poses": camera_poses(spec, placements, off), "targets": coverage_targets(spec, placements),
        "footprint": footprint(spec, off) if spec.setting.kind == "indoor" else None,
        "height": spec.setting.bounds_m[2]})
    return dict(BlenderClient().run_code(_COVERAGE.format(payload=payload), timeout=120.0))


def camera_issue(cam: dict[str, Any]) -> dict[str, Any] | None:
    if "inside" not in cam:
        return None
    missing = sorted(set(cam.get("targets", [])) - set(cam.get("visible_heroes", [])))
    if cam["inside"] and not missing:
        return None
    where = "" if cam["inside"] else "camera outside the room; "
    return {"check": "camera", "severity": "warning", "objects": missing, "fix": None,
            "message": f"{where}not visible from any of {cam.get('tried')} camera spots: {', '.join(missing) or '-'}"}


# --- handlers ------------------------------------------------------------------


def validator_args(spec: SceneSpec) -> dict[str, Any]:
    """A SceneSpec in the shape the Blender addon's validate_scene reads.

    The validator knows nothing of SceneSpec: given one directly it found no room bounds,
    priors or usable relations, so out_of_bounds/scale never ran.
    """
    priors = load_priors()
    out: dict[str, Any] = {
        "relations": [{"a": r.a, "rel": r.rel, "b": r.b} for r in spec.relations
                      if r.rel in ("facing", "in_front_of")],
        "priors": {o.category: {"min": list(priors[o.category].min_m), "max": list(priors[o.category].max_m)}
                   for o in spec.objects if o.category in priors},
    }
    if spec.setting.kind in ("indoor", "outdoor") and spec.setting.shape != "polygon":
        out["room_bounds"] = list(spec.setting.bounds_m)  # centred: see frame_offset
    return out


_LIST_IDS = """
import bpy
result = [{"name": o.name, "ai_db_id": o.get("ai_db_id")} for o in bpy.context.scene.objects]
"""


def missing_objects(spec: SceneSpec, present_ids: list[str]) -> list[dict[str, Any]]:
    """Spec objects with fewer instances in the scene than ``count`` (by ai_db_id tag)."""
    have: dict[str, int] = {}
    ids = {o.id for o in spec.objects}
    for pid in present_ids:
        base = pid if pid in ids else _SUFFIX.sub("", pid)
        have[base] = have.get(base, 0) + 1
    issues = []
    for o in spec.objects:
        n = have.get(o.id, 0)
        if n < o.count:
            issues.append({"check": "missing_objects", "severity": "blocking", "objects": [o.id],
                           "message": f"{o.id}: {n} of {o.count} in the scene", "fix": None})
    return issues


def _handle_validate_scene(args: dict[str, Any]) -> Any:
    from ai_db_scene.tools.scene_tools import _forward

    raw = args.get("spec")
    if not (isinstance(raw, dict) and "setting" in raw):
        return _forward("validate_scene", args)  # validator-native spec or none
    spec, errors = _parse_spec(raw)
    if spec is None:
        return {"status": "invalid", "errors": errors}
    # Explicit args (checks, room_bounds, priors, ...) win over the spec, as in the validator.
    res = _forward("validate_scene", {**{k: v for k, v in args.items() if k != "spec"},
                                      "spec": validator_args(spec)})
    if res.get("status") != "success":
        return res
    from ai_db_scene.blender_client import BlenderClient, BlenderError
    try:
        objs = BlenderClient().run_code(_LIST_IDS)
    except BlenderError as e:
        return {"status": "error", "message": str(e)}
    extra = missing_objects(spec, [o["ai_db_id"] for o in objs if o.get("ai_db_id")])
    result = res["result"]
    if extra:
        result["issues"] = list(result.get("issues", [])) + extra
        result["ok"] = False
        stats = result.setdefault("stats", {})
        stats["blocking"] = stats.get("blocking", 0) + len(extra)
    return res



def _handle_get_scene_spec_schema(args: dict[str, Any]) -> Any:
    priors = load_priors()
    exemplars: list[dict[str, Any]] = []
    try:
        runs = _memory().search_exemplars(args.get("brief", ""), min_score=4.0, top_k=_MAX_EXEMPLARS)
        exemplars = [{"brief": r.brief, "spec": r.spec, "score": r.score} for r in runs if r.spec]
    except Exception:  # noqa: BLE001  memory unavailable must not block schema discovery
        exemplars = []
    # Sizes only for the categories the brief is likely to need; every name stays listed, so the
    # vocabulary is complete but the reply is ~60% smaller (ai-db runs spent the most tokens here).
    brief_words = {w.rstrip("s") for w in re.findall(r"[a-z]+", str(args.get("brief", "")).lower())}
    relevant = {n for n in priors if any(part.rstrip("s") in brief_words for part in n.split("_"))}
    if not relevant:
        relevant = set(priors)  # no brief: everything, as before
    return {
        "schema": spec_schema(),
        "categories": {
            n: {"typical_m": list(p.typical_m), "support": p.support, "can_support_others": p.can_support_others}
            for n, p in sorted(priors.items()) if n in relevant
        },
        "other_categories": sorted(set(priors) - relevant),
        "relations": list(RELATIONS),
        "setting_kinds": list(SETTING_KINDS),
        "styles": sorted(STYLE_PALETTES),
        "exemplars": exemplars[:_MAX_EXEMPLARS] or [{"brief": "a small reading corner", "spec": _EXAMPLE_SPEC}],
        "next": "submit_scene_spec(spec) to check it, then build_scene(spec)",
    }


_EXAMPLE_SPEC: dict[str, Any] = {
    "setting": {"kind": "indoor", "bounds_m": [4.0, 3.5, 2.7], "floor_material": "wood:oak"},
    "objects": [{"id": "armchair", "category": "armchair", "hero": True, "material": "velvet:#3d5a80"},
                {"id": "lamp", "category": "floor_lamp"}, {"id": "shelf", "category": "bookshelf"},
                {"id": "rug", "category": "rug"}],
    "relations": [{"a": "shelf", "rel": "against_wall", "b": "wall_north"},
                  {"a": "armchair", "rel": "facing", "b": "shelf"},
                  {"a": "lamp", "rel": "beside", "b": "armchair"},
                  {"a": "rug", "rel": "centered_in", "b": "room"}]}


def _handle_submit_scene_spec(args: dict[str, Any]) -> Any:
    spec, errors = _parse_spec(args.get("spec"))
    if spec is None:
        return {"status": "invalid", "errors": errors}
    return {"status": "ok", "spec": spec.to_dict()}


def _handle_solve_layout(args: dict[str, Any]) -> Any:
    spec, errors = _parse_spec(args.get("spec"))
    if spec is None:
        return {"status": "invalid", "errors": errors}
    kw: dict[str, Any] = {}
    if args.get("seed") is not None:
        kw["seed"] = int(args["seed"])
    try:
        pls = solve_layout(spec, **kw)
    except LayoutError as e:
        return {"status": "conflict", "message": str(e), "conflicts": e.conflicts}
    from ai_db_scene.layout import room_doors

    return {"status": "ok", "placements": [placement_to_dict(p) for p in pls],
            **({"doors": room_doors(spec)} if spec.setting.rooms else {})}


def _handle_apply_layout(args: dict[str, Any]) -> Any:
    spec, errors = _parse_spec(args.get("spec"))
    if spec is None:
        return {"status": "invalid", "errors": errors}
    try:
        pls = parse_placements(args.get("placements"))
    except (KeyError, TypeError, ValueError) as e:
        return _err(f"bad placements: {e!r}")
    geometry = args.get("geometry", "generated")
    if geometry not in GEOMETRY_MODES:
        return _err(f"geometry must be one of {list(GEOMETRY_MODES)}")
    try:
        from ai_db_scene.providers import providers_from_config

        providers = providers_from_config()
    except AiDbConfigError as e:  # a bad plugins.scene section: say so instead of ignoring it
        return _err(str(e))
    cmds, names = build_layout_commands(spec, pls, args.get("asset_map"), geometry, providers)
    try:
        results = BlenderClient().batch(cmds)
    except BlenderError as e:
        return {"status": "error", "message": str(e), "error_type": e.error_type or type(e).__name__, "hint": e.hint}
    failed = [{"index": i, "type": cmds[i]["type"], "message": r.get("message")}
              for i, r in enumerate(results) if not r.get("ok", r.get("status") == "success")]
    return {"status": "partial" if failed else "success", "created": names, "commands": len(cmds), "failed": failed}


def _handle_record_run(args: dict[str, Any]) -> Any:
    outcome = args.get("outcome", "success")
    if outcome not in RUN_OUTCOMES:
        return _err(f"outcome must be one of {list(RUN_OUTCOMES)}")
    fields = {k: args[k] for k in (
        "brief", "spec", "placements", "actions", "validation_reports", "judge_scores",
        "token_cost", "wall_time_s", "project") if k in args}
    run = RunRecord(outcome=outcome, **fields)
    return {"status": "saved", "run_id": _memory().record_run(run)}


def _handle_search_exemplars(args: dict[str, Any]) -> Any:
    runs = _memory().search_exemplars(
        args["brief"], min_score=float(args.get("min_score", 4.0)),
        top_k=int(args.get("top_k", 3)), project=args.get("project"))
    return {"count": len(runs), "exemplars": [r.to_dict() | {"score": r.score} for r in runs]}


_SPEC = {"type": "object", "description": "SceneSpec (see get_scene_spec_schema)"}

def _issues(res: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((res or {}).get("result") or {}).get("issues", []))


def _handle_build_scene(args: dict[str, Any]) -> Any:
    """spec -> layout (repairing conflicts) -> build -> validate -> apply the validator's fixes."""
    from ai_db_scene.constants import BUILD_FIX_ROUNDS, BUILD_MAX_REPAIRS
    from ai_db_scene.repair import relax

    spec, errors = _parse_spec(args.get("spec"))
    if spec is None:
        return {"status": "invalid", "errors": errors}
    data = spec.to_dict()
    original = list(spec.setting.bounds_m)
    repairs: list[str] = []
    seed = int(args.get("seed", 0))
    while True:
        spec, errors = _parse_spec(data)
        if spec is None:  # a repair must never produce an invalid spec; report it if it does
            return {"status": "invalid", "errors": errors, "repairs": repairs}
        try:
            placements = solve_layout(spec, seed=seed)
            break
        except LayoutError as e:
            fix = relax(data, e, original) if len(repairs) < BUILD_MAX_REPAIRS else None
            if fix is None:
                return {"status": "conflict", "message": str(e), "conflicts": e.conflicts, "repairs": repairs,
                        "spec": data}
            data, note = fix
            repairs.append(note)
    if args.get("clear"):
        try:
            BlenderClient().request("clear_scene", {"keep_camera": False, "purge_orphans": True})
            BlenderClient().request("run_code", {"code": _PURGE})
        except BlenderError as e:
            return {"status": "error", "message": f"clear_scene failed: {e}", "repairs": repairs}
    built = _handle_apply_layout({"spec": data, "placements": [placement_to_dict(p) for p in placements],
                                  "asset_map": args.get("asset_map"), "geometry": args.get("geometry", "generated")})
    if built.get("status") not in ("success", "partial"):
        return {**built, "repairs": repairs, "spec": data}
    report = _handle_validate_scene({"spec": data})
    first = _issues(report)
    applied: list[str] = []
    for _ in range(BUILD_FIX_ROUNDS):
        fixes = [i for i in _issues(report) if isinstance(i.get("fix"), dict) and i["fix"].get("handler")]
        if not fixes:
            break
        for issue in fixes:
            try:
                BlenderClient().request(issue["fix"]["handler"], issue["fix"].get("params") or {})
                applied.append(f"{issue.get('check')}: {issue.get('message')}")
            except BlenderError as e:
                applied.append(f"fix failed for {issue.get('check')}: {e}")
        report = _handle_validate_scene({"spec": data})
    left = _issues(report)
    from ai_db_scene import constants as _c

    camera: dict[str, Any] | None = None
    try:
        camera = camera_check(spec, placements)
    except Exception as e:  # noqa: BLE001 - a check must never fail a finished build
        camera = {"error": str(e)}
    cam_issue = camera_issue(camera) if camera else None
    if cam_issue:
        left.append(cam_issue)
    exposure: dict[str, Any] | None = None
    if args.get("exposure_check", _c.BUILD_EXPOSURE_CHECK):
        try:
            exposure = exposure_check()
        except Exception as e:  # noqa: BLE001 - a check must never fail a finished build
            exposure = {"error": str(e)}
        exp_issue = exposure_issue(exposure)
        if exp_issue:
            left.append(exp_issue)
    out = {"status": "built" if not any(i.get("severity") == "blocking" for i in left) else "built_with_issues",
           "objects": len(placements), "created": built.get("created"), "failed_commands": built.get("failed"),
           "repairs": repairs, "fixes_applied": applied, "issues_found": len(first),
           "issues_left": left[:20], "spec": data if repairs else None}
    if camera is not None:
        out["camera"] = camera
    if exposure is not None:
        out["exposure"] = exposure
    from ai_db_scene.constants import BUILD_POLY_BUDGET

    budget = args.get("poly_budget", BUILD_POLY_BUDGET)
    if budget is not None:
        try:
            out["polys"] = poly_report(int(budget))
        except (BlenderError, KeyError, TypeError, ValueError) as e:
            out["polys"] = {"error": str(e)}
    if args.get("physics"):
        try:
            out["physics"] = BlenderClient().request("run_code", physics_commands(spec, placements)["params"])["result"]
        except (BlenderError, KeyError, TypeError) as e:
            out["physics"] = {"error": str(e)}
    if args.get("judge"):
        import tempfile
        import uuid

        out["critique"] = critique(left, repairs, built.get("failed"), out.get("polys"))
        target = Path(args.get("render_path") or Path(tempfile.gettempdir()) / "ai_db_judge" / f"{uuid.uuid4().hex}.png")
        try:
            out["render"] = judge_render(target)
        except BlenderError as e:
            out["render_error"] = str(e)
    # every build is a run: exemplar search (get_scene_spec_schema, search_exemplars) learns
    # from successes without anyone calling record_run (4 calls in 23 benchmark runs)
    blocking = sum(1 for i in left if i.get("severity") == "blocking")
    try:
        out["run_id"] = _memory().record_run(RunRecord(
            brief=str(args.get("prompt") or ""), spec=data, placements=[placement_to_dict(p) for p in placements],
            validation_reports=[{"issues_found": len(first), "issues_left": left[:20], "fixes": applied,
                                 "repairs": repairs}],
            judge_scores=[{"overall": max(1.0, 5.0 - blocking), "source": "validator"}],
            outcome="success" if not blocking else "partial", project=args.get("project", "default")))
    except Exception as e:  # noqa: BLE001 - recording must not fail a finished build
        out["record_error"] = str(e)
    if args.get("snapshot", True):
        # the spec travels with the version: edit_scene starts from it
        try:
            out["scene_id"] = _memory().snapshot_scene(
                prompt=str(args.get("prompt") or "build_scene"), project=args.get("project", "default"),
                operation_type=args.get("operation_type", "build"),
                metadata={"spec": data, "repairs": repairs, **(args.get("metadata") or {})},
                tags=["build_scene"], parent_scene_id=args.get("parent_scene_id"))
        except Exception as e:  # noqa: BLE001 - the scene is built; say why it was not saved
            out["snapshot_error"] = str(e)
    return out


#: faces per tagged object (its whole hierarchy, modifiers applied) and, over the budget, a
#: Decimate modifier on the heaviest non-hero objects, each cut just enough (ratio floor
#: BUILD_DECIMATE_MIN_RATIO) until the scene total fits. Untagged meshes (the room shell) count
#: towards the total but are never decimated.
_POLY_BUDGET = """
import bpy, json
budget, floor = json.loads({payload!r})
dg = bpy.context.evaluated_depsgraph_get()

def faces(o):
    return len(o.evaluated_get(dg).data.polygons) if o.type == "MESH" else 0

def parts(root):
    return [o for o in [root, *root.children_recursive] if o.type == "MESH"]

tagged = [o for o in bpy.context.scene.objects if o.get("ai_db_id") is not None]
inside = {{p.name for o in tagged for p in parts(o)}}
per = {{o.name: sum(faces(p) for p in parts(o)) for o in tagged}}
total = sum(per.values()) + sum(faces(o) for o in bpy.context.scene.objects if o.name not in inside)
before, changed = total, []
if budget is not None and total > budget:
    for o in sorted((o for o in tagged if not o.get("hero")), key=lambda o: -per[o.name]):
        if total <= budget or per[o.name] == 0:
            break
        ratio = max(floor, (per[o.name] - (total - budget)) / per[o.name])
        for p in parts(o):
            mod = p.modifiers.get("ai_db_decimate") or p.modifiers.new("ai_db_decimate", "DECIMATE")
            mod.ratio = ratio
        dg.update()
        now = sum(faces(p) for p in parts(o))
        changed.append({{"object": o.name, "ratio": round(ratio, 3), "faces_before": per[o.name], "faces_after": now}})
        total -= per[o.name] - now
        per[o.name] = now
result = {{"faces_total": total, "faces_before": before, "budget": budget, "over_budget": budget is not None and total > budget,
          "faces_per_object": dict(sorted(per.items(), key=lambda kv: -kv[1])), "decimated": changed}}
"""


def poly_report(budget: int | None) -> dict[str, Any]:
    """Faces per object after a build; with a budget, decimate down to it (see _POLY_BUDGET)."""
    from ai_db_scene.constants import BUILD_DECIMATE_MIN_RATIO

    return dict(BlenderClient().run_code(_POLY_BUDGET.format(
        payload=json.dumps([budget, BUILD_DECIMATE_MIN_RATIO])), timeout=120.0))


#: rigid bodies on every mesh of the scene: active (convex hull) for the parts of objects the
#: spec marks dynamic, passive for everything else: furniture as convex hulls, the floor as a box,
#: the walls as their mesh (a box around a room's walls would fill the room).
_PHYSICS = """
import bpy, json
dynamic = set(json.loads({payload!r}))
scene = bpy.context.scene
if scene.rigidbody_world is None:
    with bpy.context.temp_override(scene=scene):
        bpy.ops.rigidbody.world_add()

def owner(o):
    while o is not None and o.get("ai_db_id") is None:
        o = o.parent
    return o

counts = {{"active": 0, "passive": 0}}
for o in list(scene.objects):
    if o.type != "MESH":
        continue
    root = owner(o)
    active = root is not None and root.get("ai_db_id") in dynamic
    if o.rigid_body is None:
        with bpy.context.temp_override(object=o, active_object=o, selected_objects=[o]):
            bpy.ops.rigidbody.object_add()
    rb = o.rigid_body
    rb.type = "ACTIVE" if active else "PASSIVE"
    rb.collision_shape = ("CONVEX_HULL" if root is not None else
                          "MESH" if o.name.startswith("Room_Wall") else "BOX")
    counts["active" if active else "passive"] += 1
result = counts
"""


def physics_commands(spec: SceneSpec, placements: list[Placement]) -> dict[str, Any]:
    """run_code adding the rigid bodies of _PHYSICS; dynamic ids are the placements of dynamic objects."""
    ids = {o.id for o in spec.objects}
    dynamic = {o.id for o in spec.objects if o.dynamic}
    payload = [pl.id for pl in placements if (pl.id if pl.id in ids else _SUFFIX.sub("", pl.id)) in dynamic]
    return {"type": "run_code", "params": {"code": _PHYSICS.format(payload=json.dumps(payload))}}


#: validator check -> the edit_scene step that usually clears it (a hint, not an order)
_CRITIQUE_FIX: dict[str, str] = {
    "missing_objects": "update_setting {set: {bounds_m: larger}} or update_object {set: {count: fewer}}",
    "out_of_bounds": "update_setting {set: {bounds_m: larger}} or remove_relation that pushes it out",
    "interpenetration": "add_relation beside / in_front_of with gap_m, or remove_object one of them",
    "floating": "add_relation {a: <id>, rel: on, b: <support>} or update_object {set: {category: a floor one}}",
    "sunk": "remove_relation on, or update_object {set: {category: ...}} with a matching support",
    "facing": "remove_relation against_wall that fixes its yaw, or move the target (add_relation in_front_of)",
    "scale": "update_object {set: {category: the right one}}",
    "door_clearance": "remove_relation against_wall near the door, or update_setting {set: {bounds_m: larger}}",
    "camera": ("hidden from every camera spot: add_relation against_wall for the tall object in front of it, "
               "or update_setting {set: {bounds_m: larger}}"),
    "exposure": ("a black render: the camera may sit in a wall or behind furniture (move it, or add lamps); "
                 "blown out: fewer lights or an outdoor kind without sky"),
}


def critique(left: list[dict[str, Any]], repairs: list[str], failed: list[dict[str, Any]] | None,
             polys: dict[str, Any] | None) -> list[dict[str, Any]]:
    """What a calling agent should look at after a build, worst first: validator issues still
    open (missing objects included), spec repairs build_scene made, failed commands, a poly
    budget it could not reach. Each item: source, severity, objects, message, suggest."""
    out = [{"source": "validator", "severity": i.get("severity", "warning"), "check": i.get("check"),
            "objects": i.get("objects") or [], "message": i.get("message", ""),
            "suggest": _CRITIQUE_FIX.get(str(i.get("check")))} for i in left]
    out += [{"source": "repair", "severity": "warning", "objects": [], "message": r,
             "suggest": "edit the spec so it solves without this repair"} for r in repairs]
    out += [{"source": "build", "severity": "blocking", "objects": [], "message": f"{f.get('type')}: {f.get('message')}",
             "suggest": None} for f in failed or []]
    if polys and polys.get("over_budget"):
        out.append({"source": "polys", "severity": "warning", "objects": [],
                    "message": f"{polys['faces_total']} faces, budget {polys['budget']} (heroes are never decimated)",
                    "suggest": "update_object {set: {detail: draft}} or {hero: false} on heavy objects"})
    rank = {"blocking": 0, "warning": 1}
    return sorted(out, key=lambda c: rank.get(str(c["severity"]), 2))


#: probe renders from the scene camera (Cycles, tiny, few samples; the scene's render settings are
#: restored after) and view exposure steps until the mean display luminance is in band. A step
#: assumes display ~ linear^(1/2.2): k times brighter on screen is 2.2*log2(k) stops.
_EXPOSURE = """
import bpy, json, math, os, tempfile
import numpy as np
cfg = json.loads({payload!r})
scene = bpy.context.scene
r, cy, vs = scene.render, scene.cycles, scene.view_settings
if scene.camera is None:
    result = {{"skipped": "no scene camera"}}
else:
    saved = (r.resolution_x, r.resolution_y, r.resolution_percentage, r.engine, r.filepath,
             r.image_settings.file_format, cy.samples, cy.use_denoising)
    path = os.path.join(tempfile.gettempdir(), "ai_db_exposure_probe_%d.png" % os.getpid())
    start, rounds, stats = vs.exposure, 0, None
    try:
        r.resolution_x, r.resolution_y = cfg["res"]
        r.resolution_percentage, r.engine, r.filepath = 100, "CYCLES", path
        r.image_settings.file_format = "PNG"
        cy.samples, cy.use_denoising = cfg["samples"], False
        lo, hi = cfg["band"]
        while True:
            bpy.ops.render.render(write_still=True)
            img = bpy.data.images.load(path, check_existing=False)
            try:
                px = np.empty(len(img.pixels), dtype=np.float32)
                img.pixels.foreach_get(px)
            finally:
                bpy.data.images.remove(img)
            lum = px.reshape(-1, 4)[:, :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
            stats = {{"mean": round(float(lum.mean()), 3),
                     "dark_frac": round(float((lum < cfg["clip"][0]).mean()), 3),
                     "bright_frac": round(float((lum > cfg["clip"][1]).mean()), 3)}}
            too_bright = stats["bright_frac"] > cfg["max_bright"]
            if (lo <= stats["mean"] <= hi and not too_bright) or rounds >= cfg["rounds"]:
                break
            step = 2.2 * math.log2(cfg["target"] / max(stats["mean"], 0.005))
            if too_bright:
                step = min(step, -0.5)
            step = max(-cfg["max_step"], min(cfg["max_step"], step))
            new = max(start - cfg["max_total"], min(start + cfg["max_total"], vs.exposure + step))
            if abs(new - vs.exposure) < 0.05:
                break  # at the limit: more rounds would render the same image
            vs.exposure = new
            rounds += 1
    finally:
        (r.resolution_x, r.resolution_y, r.resolution_percentage, r.engine, r.filepath,
         r.image_settings.file_format, cy.samples, cy.use_denoising) = saved
        if os.path.exists(path):
            os.remove(path)
    result = {{"rounds": rounds, **(stats or {{}}), "adjusted_by": round(vs.exposure - start, 2),
              "exposure": round(vs.exposure, 2)}}
"""


def exposure_check() -> dict[str, Any]:
    """Probe the scene camera's exposure and fix it (see _EXPOSURE). Adds ``in_band``."""
    from ai_db_scene import constants as c

    payload = json.dumps({
        "res": list(c.BUILD_EXPOSURE_PROBE_RESOLUTION), "samples": c.BUILD_EXPOSURE_PROBE_SAMPLES,
        "band": list(c.BUILD_EXPOSURE_BAND), "target": c.BUILD_EXPOSURE_TARGET, "clip": list(c.BUILD_EXPOSURE_CLIP),
        "max_bright": c.BUILD_EXPOSURE_MAX_BRIGHT_FRAC, "rounds": c.BUILD_EXPOSURE_MAX_ROUNDS,
        "max_step": c.BUILD_EXPOSURE_MAX_STEP, "max_total": c.BUILD_EXPOSURE_MAX_TOTAL})
    res = dict(BlenderClient(timeout=300.0).run_code(_EXPOSURE.format(payload=payload), timeout=300.0))
    if "mean" in res:
        lo, hi = c.BUILD_EXPOSURE_BAND
        res["in_band"] = lo <= res["mean"] <= hi and res.get("bright_frac", 0.0) <= c.BUILD_EXPOSURE_MAX_BRIGHT_FRAC
    return res


def exposure_issue(exp: dict[str, Any]) -> dict[str, Any] | None:
    """A validator-style warning when the probe is still out of band after the adjustments."""
    if exp.get("in_band", True):
        return None
    dark = exp.get("mean", 0.0) < 0.3
    return {"check": "exposure", "severity": "warning", "objects": [], "fix": None,
            "message": (f"render too {'dark' if dark else 'bright'} after {exp.get('rounds')} exposure rounds: "
                        f"mean {exp.get('mean')}, {exp.get('dark_frac')} clipped dark, "
                        f"{exp.get('bright_frac')} clipped bright")}


def judge_render(path: Path) -> str:
    """Render the build's eye-level camera to ``path`` (small and quick: a probe, not a final)."""
    from ai_db_scene.constants import BUILD_JUDGE_RESOLUTION, BUILD_JUDGE_SAMPLES

    path.parent.mkdir(parents=True, exist_ok=True)
    res = BlenderClient(timeout=300.0).request("render", {
        "filepath": str(path), "resolution": list(BUILD_JUDGE_RESOLUTION), "samples": BUILD_JUDGE_SAMPLES,
        "denoise": False})
    return str(res.get("filepath") or path) if isinstance(res, dict) else str(path)


def _stored_spec(scene_id: str) -> tuple[dict[str, Any] | None, str | None]:
    scene = _memory().restore_scene(scene_id)
    meta = scene.prompt_context.metadata if scene.prompt_context else {}
    spec = meta.get("spec") if isinstance(meta, dict) else None
    if not isinstance(spec, dict):
        return None, (f"scene {scene_id} has no stored spec (it was not made by build_scene/edit_scene); "
                      "pass the spec to build_scene first")
    return spec, None


def _handle_edit_scene(args: dict[str, Any]) -> Any:
    """Apply ordered edit steps to a built scene's spec, rebuild it, save the next version."""
    from ai_db_scene.edits import apply_steps

    scene_id = args.get("scene_id")
    if not scene_id:
        return _err("scene_id is required")
    try:
        spec, why = _stored_spec(scene_id)
    except ValueError as e:
        return _err(str(e))
    if spec is None:
        return _err(why or "no stored spec")
    data, done, errors = apply_steps(spec, args.get("steps"))
    if data is None:
        return {"status": "invalid", "errors": errors, "applied_before_error": done}
    parsed, errors = _parse_spec(data)
    if parsed is None:
        return {"status": "invalid", "errors": errors, "steps_applied": done}
    safety = None
    try:  # keep manual Blender edits made since the last version before the scene is cleared
        safety = _memory().snapshot_scene(prompt=f"before edit of {scene_id}", project=args.get("project", "default"),
                                          operation_type="pre_edit", tags=["pre_edit"])
    except Exception:  # noqa: BLE001 - no Blender or empty scene: nothing to keep
        safety = None
    out = _handle_build_scene({"spec": data, "seed": args.get("seed", 0), "geometry": args.get("geometry", "generated"),
                               "asset_map": args.get("asset_map"), "clear": True, "snapshot": True,
                               "prompt": args.get("prompt") or "; ".join(done), "project": args.get("project", "default"),
                               "operation_type": "edit", "parent_scene_id": scene_id,
                               "metadata": {"steps": args.get("steps")}})
    return {**out, "steps_applied": done, "previous_scene_id": scene_id, "safety_snapshot": safety}


_EXPORT = """
import bpy, os
path, fmt = {path!r}, {fmt!r}
os.makedirs(os.path.dirname(path), exist_ok=True)
if fmt in ("glb", "gltf"):
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB" if fmt == "glb" else "GLTF_SEPARATE")
else:
    try:
        bpy.ops.wm.usd_export.get_rna_type()
    except KeyError:
        raise RuntimeError("this Blender has no USD exporter (bpy.ops.wm.usd_export)")
    bpy.ops.wm.usd_export(filepath=path)
files = [path] + ([path[:-5] + ".bin"] if fmt == "gltf" else [])
result = {{"bytes": sum(os.path.getsize(f) for f in files if os.path.exists(f)),
          "objects": len(bpy.context.scene.objects),
          "meshes": sum(1 for o in bpy.context.scene.objects if o.type == "MESH")}}
"""


def export_root() -> Path:
    """The one folder export_scene writes into: $AI_DB_SCENE_EXPORT_DIR, plugins.scene.export_dir
    or SCENE_EXPORT_DIR."""
    import os

    from ai_db_scene.constants import SCENE_EXPORT_DIR

    configured = os.environ.get("AI_DB_SCENE_EXPORT_DIR")
    if not configured:
        from ai_db.config import load_config
        from ai_db_scene.config import scene_config

        try:
            configured = scene_config(load_config(check_env=False)).export_dir
        except AiDbConfigError:
            configured = None
    return Path(configured or SCENE_EXPORT_DIR).expanduser().resolve()


def export_path(path: str, fmt: str | None, root: Path) -> tuple[Path, str]:
    """Resolve ``path`` (relative to ``root``, or absolute inside it) and the format.

    Raises ValueError for a path that leaves ``root`` once resolved ("..", symlinks) or an
    unknown format; a missing extension is added from the format.
    """
    from ai_db_scene.constants import SCENE_EXPORT_FORMATS

    p = Path(path).expanduser()
    suffix = p.suffix.lower()
    fmt = fmt or next((f for f, ext in SCENE_EXPORT_FORMATS.items() if ext == suffix), None)
    if fmt not in SCENE_EXPORT_FORMATS:
        raise ValueError(f"format must be one of {list(SCENE_EXPORT_FORMATS)} (or a path ending in one)")
    if suffix != SCENE_EXPORT_FORMATS[fmt]:
        p = p.with_name(p.name + SCENE_EXPORT_FORMATS[fmt])
    target = (p if p.is_absolute() else root / p).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"{path!r} is outside the export folder {root} (set plugins.scene.export_dir)")
    return target, fmt


def _handle_export_scene(args: dict[str, Any]) -> Any:
    from ai_db_scene.constants import SCENE_EXPORT_TIMEOUT_S

    root = export_root()
    try:
        target, fmt = export_path(str(args.get("path") or ""), args.get("format"), root)
    except ValueError as e:
        return _err(str(e))
    try:
        res = BlenderClient().run_code(_EXPORT.format(path=str(target), fmt=fmt), timeout=SCENE_EXPORT_TIMEOUT_S)
    except BlenderError as e:
        return {"status": "error", "message": str(e), "error_type": e.error_type or type(e).__name__, "hint": e.hint}
    return {"status": "success", "path": str(target), "format": fmt, "bytes": res["bytes"],
            "objects": res["objects"], "meshes": res["meshes"]}


PIPELINE_TOOLS: list[tuple[str, str, dict[str, Any], Any]] = [
    ("get_scene_spec_schema",
     "SceneSpec JSON schema + category vocabulary with typical dims + relations + up to 3 exemplar specs from past successful runs",
     {"type": "object", "properties": {"brief": {"type": "string", "description": "Brief, to retrieve similar exemplars"}}},
     _handle_get_scene_spec_schema),
    ("submit_scene_spec",
     "Validate a SceneSpec; returns exact errors, or ok + the normalized spec",
     {"type": "object", "properties": {"spec": _SPEC}, "required": ["spec"]},
     _handle_submit_scene_spec),
    ("solve_layout",
     "Deterministically place every object of a SceneSpec; returns placements or the conflicting relations",
     {"type": "object", "properties": {"spec": _SPEC, "seed": {"type": "integer"}}, "required": ["spec"]},
     _handle_solve_layout),
    ("build_scene",
     ("One call from SceneSpec to finished scene: validates the spec, lays it out (repairing conflicts by "
      "dropping the conflicting relation, enlarging the setting or placing fewer copies, each change reported), "
      "builds it with real furniture, materials, daylight, lamp lights and an eye-level camera, validates it and "
      "applies the validator's own fixes. Prefer this to solve_layout + apply_layout"),
     {"type": "object",
      "properties": {
          "spec": _SPEC,
          "seed": {"type": "integer"},
          "geometry": {"enum": list(GEOMETRY_MODES)},
          "asset_map": {"type": "object", "description": "id|category -> {filepath, asset_name}"},
          "clear": {"type": "boolean", "description": "empty the Blender scene first (default false)"},
          "snapshot": {"type": "boolean", "description": "save the result as a scene version with its spec (default true)"},
          "prompt": {"type": "string", "description": "what this build is, for the scene history"},
          "project": {"type": "string"},
          "poly_budget": {"type": "integer", "minimum": 1,
                          "description": ("total faces; reports faces per object and decimates the heaviest "
                                          "non-hero objects until the scene fits (report: polys)")},
          "physics": {"type": "boolean",
                      "description": "add rigid bodies: passive structure and furniture, active objects with dynamic: true"},
          "judge": {"type": "boolean",
                    "description": ("render the build camera (render) and return a structured critique (validator issues, "
                                    "missing objects, repairs, failed commands) with suggested edit_scene steps")},
          "render_path": {"type": "string", "description": "judge: where to write the render (default: a temp file)"},
          "exposure_check": {"type": "boolean",
                             "description": ("default true: a tiny probe render from the camera; the view exposure is "
                                             "adjusted until the image is neither black nor blown out (report: exposure)")}},
      "required": ["spec"]},
     _handle_build_scene),
    ("edit_scene",
     ("Refactor a scene made by build_scene in ordered steps: add_object, remove_object, update_object "
      "{set: count|material|hero|grid|category|detail}, add_relation, remove_relation, update_setting "
      "{set: bounds_m|style|floor_material|wall_material|shape|points|kind}. Applies them to the scene's stored "
      "spec, rebuilds the scene (same seed: untouched objects mostly stay), validates, and saves the result as the "
      "next version (restore_scene on an earlier id undoes it). The current Blender scene is snapshotted first"),
     {"type": "object",
      "properties": {
          "scene_id": {"type": "string", "description": "a version saved by build_scene or edit_scene"},
          "steps": {"type": "array", "items": {"type": "object"},
                    "description": "e.g. [{op: add_object, object: {id: lamp, category: floor_lamp}, relations: [{a: lamp, rel: beside, b: sofa}]}, {op: update_object, id: sofa, set: {material: velvet:#5b2a4a}}]"},
          "seed": {"type": "integer"}, "geometry": {"enum": list(GEOMETRY_MODES)},
          "prompt": {"type": "string"}, "project": {"type": "string"}},
      "required": ["scene_id", "steps"]},
     _handle_edit_scene),
    ("apply_layout",
     ("Create the room shell (floor + walls if indoor) and one object per placement in Blender as one batch; "
      "placeholder cubes unless asset_map maps an object id / category to {filepath, asset_name}. "
      "Tags custom props category/hero/support"),
     {"type": "object",
      "properties": {
          "spec": _SPEC,
          "placements": {"type": "array", "items": {"type": "object"}, "description": "From solve_layout"},
          "asset_map": {"type": "object", "description": "id|category -> {filepath, asset_name}"},
          "geometry": {"enum": list(GEOMETRY_MODES),
                       "description": "generated (default): parametric furniture where a generator exists; boxes: placeholders only"}},
      "required": ["spec", "placements"]},
     _handle_apply_layout),
    ("record_run",
     "Persist a pipeline run record (brief, spec, placements, actions, validation reports, judge scores, cost, outcome)",
     {"type": "object",
      "properties": {
          "brief": {"type": "string"}, "spec": {"type": "object"},
          "placements": {"type": "array"}, "actions": {"type": "array"},
          "validation_reports": {"type": "array"}, "judge_scores": {"type": "array"},
          "token_cost": {"type": "number"}, "wall_time_s": {"type": "number"},
          "outcome": {"type": "string", "enum": list(RUN_OUTCOMES)}, "project": {"type": "string"}},
      "required": ["brief"]},
     _handle_record_run),
    ("export_scene",
     ("Export the current Blender scene to glb, gltf, usd or usdz (format from `format` or the path's "
      "extension). `path` is relative to the export folder (plugins.scene.export_dir); paths outside it "
      "are refused. Returns the file size and object count"),
     {"type": "object",
      "properties": {"path": {"type": "string", "description": "e.g. living_room.glb"},
                     "format": {"enum": ["glb", "gltf", "usd", "usdz"]}},
      "required": ["path"]},
     _handle_export_scene),
    ("search_exemplars",
     "Find past successful, judge-approved runs similar to a brief (never returns failed or low-score runs)",
     {"type": "object",
      "properties": {"brief": {"type": "string"}, "min_score": {"type": "number", "default": 4.0},
                     "top_k": {"type": "integer", "default": 3}, "project": {"type": "string"}},
      "required": ["brief"]},
     _handle_search_exemplars),
]


def _handle_resolve_asset(args: dict[str, Any]) -> Any:
    from ai_db_scene.assets import AssetResolver
    from ai_db_scene.blender_client import try_get_blender_client

    client = try_get_blender_client()
    if client is None:
        return _err("Blender is not running or its socket server is not started.")
    try:
        cands = AssetResolver(client).resolve_asset(args["category"], args.get("style"), int(args.get("top_k", 3)))
    except BlenderError as e:
        return {"status": "error", "message": str(e), "error_type": e.error_type or type(e).__name__, "hint": e.hint}
    return {"status": "success", "candidates": [c.to_dict() for c in cands]}


PIPELINE_TOOLS.append(
    ("resolve_asset",
     ("Find Blender library assets for a category/style, ranked by name/tag match, with measured dimensions "
      "(appends candidates to measure them). Pass the result to apply_layout's asset_map"),
     {"type": "object",
      "properties": {"category": {"type": "string"}, "style": {"type": "string"},
                     "top_k": {"type": "integer", "default": 3}},
      "required": ["category"]},
     _handle_resolve_asset)
)
