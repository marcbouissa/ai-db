"""``draft_skill`` sources of the scene plugin (core ``ai_db.skill_sources`` entry point).

- ``{"video": path}``: a local video, never a URL. Its subtitles (``.vtt``/``.srt`` with the
  same stem, or ``"subtitles": path``) become timestamped Workflow steps; keyframes (the
  ``frame_extractor`` video processor) are listed under Examples when ``"keyframes": true``
  or when there are no subtitles.
- ``{"asset": path or object name}``: how an asset is built. ``.gltf``/``.glb`` are read in
  pure Python (the JSON part only); a ``.blend`` file or an object name is summarised in the
  connected Blender through ``run_code``: modifier stacks, geometry-node and shader node
  trees (node types, unlinked input values, links), dimensions and materials. A ``.blend`` is
  read through ``bpy.data.libraries.load`` and its datablocks are removed afterwards, so the
  open scene is not changed.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Any

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
SUB_EXTS = (".vtt", ".srt")
MAX_CUES = 40
MAX_KEYFRAMES = 12
KEYFRAME_INTERVAL_S = 10.0
MAX_ASSET_LINES = 60
_CUE_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->")


def _mmss(t: float) -> str:
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def parse_subtitles(text: str) -> list[tuple[float, str]]:
    """(start seconds, text) per cue of a WebVTT or SRT file; markup and repeats dropped."""
    cues: list[tuple[float, str]] = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        m = _CUE_TIME.search(block)
        if not m:
            continue
        h, mnt, sec, _ms = m.groups()
        start = int(h or 0) * 3600 + int(mnt) * 60 + int(sec)
        after = block[m.end():].split("\n", 1)
        body = re.sub(r"<[^>]+>", "", after[1] if len(after) > 1 else "")
        line = " ".join(body.split())
        if line and (not cues or cues[-1][1] != line):
            cues.append((float(start), line))
    return cues


def _local_path(raw: Any, what: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{what} must be a local path")
    if "://" in raw:
        raise ValueError(f"{what} must be a local file, not a URL ({raw!r}): nothing is downloaded")
    p = Path(os.path.expanduser(raw)).resolve()
    if not p.is_file():
        raise ValueError(f"{what} {raw!r} is not a file")
    return p


def read_video(source: dict[str, Any]) -> dict[str, Any]:
    path = _local_path(source.get("video"), "video")
    warnings: list[str] = []
    subs = source.get("subtitles")
    sub_path = _local_path(subs, "subtitles") if subs else next(
        (path.with_suffix(e) for e in SUB_EXTS if path.with_suffix(e).is_file()), None)
    workflow: list[str] = []
    if sub_path:
        cues = parse_subtitles(sub_path.read_text(encoding="utf-8", errors="replace"))
        workflow = [f"[{_mmss(t)}] {line}" for t, line in cues[:MAX_CUES]]
        if len(cues) > MAX_CUES:
            warnings.append(f"video: {len(cues) - MAX_CUES} later subtitle cues not shown")
    else:
        warnings.append(f"video: no .vtt/.srt next to {path.name}; only keyframes")
    examples: list[str] = []
    if source.get("keyframes") or not sub_path:
        try:
            examples = [f"keyframe [{_mmss(k['timestamp'])}]: {k['filepath']}" for k in _keyframes(path)]
        except Exception as e:  # noqa: BLE001 - optional video deps (opencv) may be missing
            warnings.append(f"video: keyframes unavailable ({e})")
    rec = {"video": str(path), **({"license": str(source["license"])} if source.get("license") else {}),
           **({"title": str(source["title"])} if source.get("title") else {})}
    return {"sections": {"workflow": workflow, "examples": examples}, "source": rec, "warnings": warnings}


def _keyframes(path: Path) -> list[dict[str, Any]]:
    from ai_db_scene.video.processors.base import get_video_processor

    proc = get_video_processor("frame_extractor")
    if proc is None:
        raise RuntimeError("frame_extractor processor not installed")
    missing = proc.check_dependencies()
    if missing:
        raise RuntimeError(f"missing {missing}")
    res = proc.process(path, {"interval": KEYFRAME_INTERVAL_S, "max_frames": MAX_KEYFRAMES})
    return [{"timestamp": kf.timestamp, "filepath": kf.filepath} for kf in res.keyframes]


# --- assets ---------------------------------------------------------------------------------------

def read_gltf_json(path: Path) -> dict[str, Any]:
    """The glTF JSON of a .gltf, or the JSON chunk of a .glb (binary buffers are not read)."""
    if path.suffix.lower() == ".gltf":
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data
    with path.open("rb") as f:
        head = f.read(20)
        if len(head) < 20 or head[:4] != b"glTF":
            raise ValueError(f"{path.name} is not a binary glTF")
        _version, _length, chunk_len, chunk_type = struct.unpack("<IIII", head[4:20])
        if chunk_type != 0x4E4F534A:  # "JSON"
            raise ValueError(f"{path.name}: first chunk is not JSON")
        out: dict[str, Any] = json.loads(f.read(chunk_len).decode("utf-8"))
        return out


def summarize_gltf(doc: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    acc = doc.get("accessors") or []
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for mesh in doc.get("meshes") or []:
        prims = mesh.get("primitives") or []
        attrs = sorted({a for p in prims for a in (p.get("attributes") or {})})
        lines.append(f"mesh {mesh.get('name', '?')}: {len(prims)} primitive(s), attributes {', '.join(attrs)}")
        for p in prims:
            i = (p.get("attributes") or {}).get("POSITION")
            a = acc[i] if isinstance(i, int) and i < len(acc) else {}
            if a.get("min") and a.get("max"):
                lo = [min(lo[k], a["min"][k]) for k in range(3)]
                hi = [max(hi[k], a["max"][k]) for k in range(3)]
    if lo[0] != float("inf"):
        # glTF is Y-up: report Blender's x, y (depth), z (height)
        lines.insert(0, f"dimensions (mesh space, m): x {hi[0] - lo[0]:.3f}, y {hi[2] - lo[2]:.3f}, z {hi[1] - lo[1]:.3f}")
    for node in doc.get("nodes") or []:
        tr = {k: node[k] for k in ("translation", "rotation", "scale") if k in node}
        if "mesh" in node or tr:
            lines.append(f"node {node.get('name', '?')}: mesh {node.get('mesh', '-')}"
                         + (f", {json.dumps(tr)}" if tr else "")
                         + (f", children {node['children']}" if node.get("children") else ""))
    textures = doc.get("textures") or []
    images = doc.get("images") or []

    def tex(ref: Any) -> str:
        if not isinstance(ref, dict) or not isinstance(ref.get("index"), int) or ref["index"] >= len(textures):
            return "-"
        src = textures[ref["index"]].get("source")
        img = images[src] if isinstance(src, int) and src < len(images) else {}
        return str(img.get("uri") or img.get("name") or f"image {src}")

    for m in doc.get("materials") or []:
        pbr = m.get("pbrMetallicRoughness") or {}
        parts = [f"baseColor {pbr.get('baseColorFactor', [1, 1, 1, 1])}",
                 f"metallic {pbr.get('metallicFactor', 1.0)}", f"roughness {pbr.get('roughnessFactor', 1.0)}"]
        for key, ref in (("baseColor tex", pbr.get("baseColorTexture")),
                         ("metal/rough tex", pbr.get("metallicRoughnessTexture")),
                         ("normal tex", m.get("normalTexture")), ("emissive tex", m.get("emissiveTexture"))):
            if ref:
                parts.append(f"{key} {tex(ref)}")
        if m.get("alphaMode"):
            parts.append(f"alpha {m['alphaMode']}")
        lines.append(f"material {m.get('name', '?')}: " + ", ".join(parts))
    if doc.get("extensionsUsed"):
        lines.append(f"extensions: {', '.join(doc['extensionsUsed'])}")
    return lines[:MAX_ASSET_LINES]


#: Runs inside Blender: ``TARGET`` is a .blend path or an object name. Sets ``result``.
BLEND_SUMMARY_CODE = r'''
import bpy, os
TARGET = {target!r}
MAX_NODES = 40

def val(v):
    try:
        if hasattr(v, "__len__") and not isinstance(v, str):
            return [round(float(x), 4) for x in v][:4]
        if isinstance(v, float):
            return round(v, 4)
        if isinstance(v, (int, bool, str)):
            return v
    except (TypeError, ValueError):
        pass
    return None

def tree(nt):
    if nt is None:
        return None
    nodes = []
    for n in list(nt.nodes)[:MAX_NODES]:
        ins = {{}}
        for s in n.inputs:
            if not s.is_linked and hasattr(s, "default_value"):
                v = val(s.default_value)
                if v is not None:
                    ins[s.name] = v
        nodes.append({{"name": n.name, "type": n.bl_idname, "inputs": dict(list(ins.items())[:8])}})
    links = [f"{{l.from_node.name}}.{{l.from_socket.name}} -> {{l.to_node.name}}.{{l.to_socket.name}}"
             for l in list(nt.links)[:MAX_NODES]]
    return {{"name": nt.name, "nodes": nodes, "links": links}}

def mod(m):
    props = {{}}
    for p in m.bl_rna.properties:
        if p.identifier in ("rna_type", "name", "type") or p.is_readonly:
            continue
        if p.type in ("INT", "FLOAT", "BOOLEAN", "ENUM"):
            v = val(getattr(m, p.identifier, None))
            if v not in (None, False):
                props[p.identifier] = v
    out = {{"name": m.name, "type": m.type, "props": dict(list(props.items())[:10])}}
    if m.type == "NODES":
        out["node_group"] = tree(m.node_group)
    return out

def obj(o):
    mats = []
    for slot in getattr(o, "material_slots", []):
        mt = slot.material
        if mt is not None:
            mats.append({{"name": mt.name, "tree": tree(mt.node_tree) if mt.use_nodes else None}})
    return {{"name": o.name, "type": o.type, "dimensions": val(o.dimensions),
             "modifiers": [mod(m) for m in o.modifiers], "materials": mats,
             "children": [c.name for c in o.children]}}

if TARGET.lower().endswith(".blend"):
    before = set(bpy.data.libraries)
    with bpy.data.libraries.load(TARGET, link=True) as (src, dst):
        dst.objects = list(src.objects)[:20]
    objs = [o for o in dst.objects if o is not None]
    result = {{"objects": [obj(o) for o in objs]}}
    for lib in set(bpy.data.libraries) - before:
        bpy.data.libraries.remove(lib)
else:
    o = bpy.data.objects.get(TARGET)
    if o is None:
        result = {{"error": "no object named " + TARGET}}
    else:
        result = {{"objects": [obj(x) for x in [o, *o.children_recursive][:20]]}}
'''


def summarize_blend(summary: dict[str, Any]) -> list[str]:
    """Lines of the ``BLEND_SUMMARY_CODE`` result."""
    if summary.get("error"):
        raise ValueError(str(summary["error"]))
    lines: list[str] = []
    for o in summary.get("objects") or []:
        lines.append(f"object {o.get('name')} ({o.get('type')}), dimensions {o.get('dimensions')} m"
                     + (f", children {o['children']}" if o.get("children") else ""))
        for k, m in enumerate(o.get("modifiers") or [], 1):
            lines.append(f"  modifier {k}: {m.get('type')} '{m.get('name')}' {json.dumps(m.get('props') or {})}")
            if m.get("node_group"):
                lines += _tree_lines(m["node_group"], "    geometry nodes")
        for mat in o.get("materials") or []:
            lines.append(f"  material {mat.get('name')}")
            if mat.get("tree"):
                lines += _tree_lines(mat["tree"], "    shader")
    return lines[:MAX_ASSET_LINES]


def _tree_lines(tree: dict[str, Any], label: str) -> list[str]:
    out = [f"{label} '{tree.get('name')}': {len(tree.get('nodes') or [])} nodes"]
    out += [f"{label} node {n.get('type')} '{n.get('name')}'" + (f" {json.dumps(n['inputs'])}" if n.get("inputs") else "")
            for n in tree.get("nodes") or []]
    out += [f"{label} link {lk}" for lk in tree.get("links") or []]
    return out


def read_asset(source: dict[str, Any], client: Any = None) -> dict[str, Any]:
    raw = source.get("asset")
    if not isinstance(raw, str) or not raw:
        raise ValueError("asset must be a .gltf/.glb/.blend path or a Blender object name")
    if "://" in raw:
        raise ValueError("asset must be local (a path or a Blender object name): nothing is downloaded")
    looks_like_path = raw.lower().endswith((".gltf", ".glb", ".blend")) or os.sep in raw
    if looks_like_path:
        path = _local_path(raw, "asset")
        rec = {"asset": str(path)}
        if path.suffix.lower() in (".gltf", ".glb"):
            lines = summarize_gltf(read_gltf_json(path))
        elif path.suffix.lower() == ".blend":
            lines = summarize_blend(_blender(client).run_code(BLEND_SUMMARY_CODE.format(target=str(path)), timeout=60.0))
        else:
            raise ValueError(f"unsupported asset type {path.suffix!r}: .gltf, .glb or .blend")
    else:
        rec = {"asset": raw}
        lines = summarize_blend(_blender(client).run_code(BLEND_SUMMARY_CODE.format(target=raw), timeout=60.0))
    if source.get("license"):
        rec["license"] = str(source["license"])
    return {"sections": {"asset_build": lines}, "source": rec, "warnings": []}


def _blender(client: Any) -> Any:
    if client is not None:
        return client
    from ai_db_scene.blender_client import BlenderClient

    return BlenderClient()


def read_source(source: dict[str, Any], config: Any = None) -> dict[str, Any] | None:
    """``ai_db.skill_sources`` hook: video and asset sources; None for anything else."""
    if "video" in source:
        return read_video(source)
    if "asset" in source:
        from ai_db_scene.blender_client import BlenderError

        try:
            return read_asset(source)
        except BlenderError as e:
            raise ValueError(f"asset: Blender not reachable or failed ({e})") from e
    return None
