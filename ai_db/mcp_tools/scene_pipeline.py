"""MCP tools for the spec -> layout -> build pipeline (docs/3D_PIPELINE.md stages 1, 3, 4, 7).

Table-driven like scene_tools: (name, description, schema) rows plus handlers. Command
construction is pure (``build_layout_commands``) so it is testable without Blender.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ai_db.ext.blender_client import BlenderClient, BlenderError
from ai_db.scene.layout import LayoutError, Placement, solve_layout
from ai_db.scene.models import RUN_OUTCOMES, RunRecord
from ai_db.scene.priors import load_priors
from ai_db.scene.spec import RELATIONS, SETTING_KINDS, SceneSpec, spec_schema, validate_spec

WALL_THICKNESS_M = 0.1
_WALLS = ("wall_north", "wall_south", "wall_east", "wall_west")
_SUFFIX = re.compile(r"_\d+$")
_MAX_EXEMPLARS = 3


def _memory() -> Any:
    from ai_db.mcp_tools.scene_tools import get_scene_memory

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
        out.append({
            "id": pl.id, "kind": pl.category, "wall": r.b, **({"at_deg": r.at_deg} if r.at_deg is not None else {}),
            "center": [pl.location[0], pl.location[1]], "yaw_deg": pl.rotation_deg_z,
            "width": w, "height": h, "depth": d, "base_z": pl.location[2] - h / 2,
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
        "params": {"primitive": "cube", "name": "floor", "location": [w / 2, d / 2, -t / 2], "scale": [w / 2, d / 2, t / 2]},
    }] if shp == "rect" else [{"type": "room_shell", "params": {
        "shape": shp, "height": 0.0, "wall_thickness": t, "openings": [], "floor": True, "ceiling": False, "trim": False,
        **({"points": [list(p) for p in spec.setting.points]} if shp == "polygon" else {"size": [w, d]})}}]


def _lookup_asset(asset_map: dict[str, Any], pl: Placement, base_id: str) -> dict[str, Any] | None:
    for key in (pl.id, base_id, pl.category):
        v = asset_map.get(key)
        if isinstance(v, dict) and v.get("filepath") and v.get("asset_name"):
            return v
    return None


_FINALIZE = """
import bpy, json
for ent in json.loads({payload!r}):
    obj = bpy.data.objects.get(ent["name"])
    if obj is None and ent.get("asset_name"):
        for o in bpy.data.objects:
            if o.get("ai_db_id") is None and (o.name == ent["asset_name"] or o.name.startswith(ent["asset_name"] + ".")):
                obj = o
                break
        if obj is not None:
            obj.name = ent["name"]
    if obj is None:
        continue
    if ent.get("asset_name"):
        obj.location = ent["location"]
        obj.rotation_euler[2] = ent["rot_z"]
    for k, v in ent["props"].items():
        obj[k] = v
result = {{"tagged": len(json.loads({payload!r}))}}
"""


def build_layout_commands(
    spec: SceneSpec, placements: list[Placement], asset_map: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Blender batch for room shell + one object per placement. Returns (commands, created names)."""
    import math

    asset_map = asset_map or {}
    priors = load_priors()
    heroes = {o.id for o in spec.objects if o.hero}
    details = {o.id: o.detail for o in spec.objects}
    cmds = _room_commands(spec, placements)
    names = [c["params"].get("name", "room_shell") for c in cmds]
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
        if asset:
            cmds.append({"type": "append_asset", "params": {
                "filepath": asset["filepath"], "asset_name": asset["asset_name"], "location": list(pl.location)}})
            tag.append({"name": pl.id, "asset_name": asset["asset_name"], "location": list(pl.location),
                        "rot_z": math.radians(pl.rotation_deg_z), "props": props})
        else:
            cmds.append({"type": "create_primitive", "params": {
                "primitive": "cube", "name": pl.id, "location": list(pl.location),
                "scale": [pl.dims[0] / 2, pl.dims[1] / 2, pl.dims[2] / 2]}})
            if pl.rotation_deg_z:
                cmds.append({"type": "modify_object", "params": {
                    "name": pl.id, "rotation": [0.0, 0.0, pl.rotation_deg_z]}})
            tag.append({"name": pl.id, "location": list(pl.location), "rot_z": 0.0, "props": props})
        names.append(pl.id)
    if tag:
        cmds.append({"type": "run_code", "params": {"code": _FINALIZE.format(payload=json.dumps(tag))}})
    return cmds, names


# --- handlers ------------------------------------------------------------------


def _handle_get_scene_spec_schema(args: dict[str, Any]) -> Any:
    priors = load_priors()
    exemplars: list[dict[str, Any]] = []
    try:
        runs = _memory().search_exemplars(args.get("brief", ""), min_score=4.0, top_k=_MAX_EXEMPLARS)
        exemplars = [{"brief": r.brief, "spec": r.spec, "score": r.score} for r in runs if r.spec]
    except Exception:  # noqa: BLE001  memory unavailable must not block schema discovery
        exemplars = []
    return {
        "schema": spec_schema(),
        "categories": {
            n: {"typical_m": list(p.typical_m), "support": p.support, "can_support_others": p.can_support_others}
            for n, p in sorted(priors.items())
        },
        "relations": list(RELATIONS),
        "setting_kinds": list(SETTING_KINDS),
        "exemplars": exemplars[:_MAX_EXEMPLARS],
    }


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
    return {"status": "ok", "placements": [placement_to_dict(p) for p in pls]}


def _handle_apply_layout(args: dict[str, Any]) -> Any:
    spec, errors = _parse_spec(args.get("spec"))
    if spec is None:
        return {"status": "invalid", "errors": errors}
    try:
        pls = parse_placements(args.get("placements"))
    except (KeyError, TypeError, ValueError) as e:
        return _err(f"bad placements: {e!r}")
    cmds, names = build_layout_commands(spec, pls, args.get("asset_map"))
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
    ("apply_layout",
     ("Create the room shell (floor + walls if indoor) and one object per placement in Blender as one batch; "
      "placeholder cubes unless asset_map maps an object id / category to {filepath, asset_name}. "
      "Tags custom props category/hero/support"),
     {"type": "object",
      "properties": {
          "spec": _SPEC,
          "placements": {"type": "array", "items": {"type": "object"}, "description": "From solve_layout"},
          "asset_map": {"type": "object", "description": "id|category -> {filepath, asset_name}"}},
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
    ("search_exemplars",
     "Find past successful, judge-approved runs similar to a brief (never returns failed or low-score runs)",
     {"type": "object",
      "properties": {"brief": {"type": "string"}, "min_score": {"type": "number", "default": 4.0},
                     "top_k": {"type": "integer", "default": 3}, "project": {"type": "string"}},
      "required": ["brief"]},
     _handle_search_exemplars),
]


def _handle_resolve_asset(args: dict[str, Any]) -> Any:
    from ai_db.ext.blender_client import try_get_blender_client
    from ai_db.scene.assets import AssetResolver

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
