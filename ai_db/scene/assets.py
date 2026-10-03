"""Asset resolver (pipeline Stage 2): find Blender library assets for a category + style.

``rank_assets`` is pure; ``AssetResolver`` is the thin client part (list libraries, append,
measure with get_bounds).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

_TOKEN = re.compile(r"[a-z0-9]+")
_LIST_CODE = """
import bpy
with bpy.data.libraries.load({fp!r}) as (src, _dst):
    result = {{"objects": list(src.objects)}}
"""


def _tokens(*texts: str) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out |= set(_TOKEN.findall(t.lower().replace("_", " ")))
    return out


@dataclass
class AssetCandidate:
    name: str
    score: float
    filepath: str | None = None  # None: already in the open .blend
    library: str = ""
    tags: list[str] = field(default_factory=list)
    dims: tuple[float, float, float] | None = None  # measured [w, d, h] in meters
    measure_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "score": self.score, "filepath": self.filepath, "library": self.library,
                "tags": self.tags, "dims": list(self.dims) if self.dims else None,
                "measure_error": self.measure_error}


def _score(category: str, style: str | None, a: dict[str, Any]) -> float:
    cat = _tokens(category)
    cat_str = category.lower().replace("_", " ")
    name_t = _tokens(a.get("name", ""))
    tag_t = _tokens(*(a.get("tags") or []), a.get("description", "") or "")
    path_t = _tokens(a.get("library", "") or "", os.path.basename(a.get("filepath") or ""))
    s = 0.0
    if cat <= name_t:
        s += 3.0
    elif cat_str.replace(" ", "") in a.get("name", "").lower().replace("_", "").replace(" ", ""):
        s += 2.0
    elif cat & name_t:
        s += 1.0 * len(cat & name_t) / len(cat)
    if cat <= tag_t:
        s += 1.5
    elif cat & tag_t:
        s += 0.5
    if cat <= path_t:
        s += 1.0
    if s == 0:
        return 0.0  # style alone never makes an asset a match
    for st in _tokens(style or ""):
        if st in name_t:
            s += 1.0
        if st in tag_t:
            s += 0.75
        if st in path_t:
            s += 0.5
    return s


def rank_assets(category: str, style: str | None, assets: list[dict[str, Any]]) -> list[AssetCandidate]:
    """Rank assets ({name, filepath?, library?, tags?, description?, type?}) by name/tag match.

    Non-object assets and zero-score assets are dropped. Ties break by name for determinism.
    """
    out = []
    for a in assets:
        if a.get("type", "OBJECT") != "OBJECT" or not a.get("name"):
            continue
        s = _score(category, style, a)
        if s > 0:
            out.append(AssetCandidate(a["name"], s, a.get("filepath"), a.get("library", "") or "",
                                      list(a.get("tags") or [])))
    out.sort(key=lambda c: (-c.score, c.name))
    return out


class AssetResolver:
    """Resolve a (category, style) to ranked candidates with measured dimensions."""

    def __init__(self, client: Any) -> None:
        self.client = client

    def collect_assets(self) -> list[dict[str, Any]]:
        libs = self.client.get_available_libraries()
        assets: list[dict[str, Any]] = [dict(a) for a in libs.get("local_assets", [])]
        assets += [dict(a) for a in libs.get("available_assets", [])]
        for lib in libs.get("asset_libraries", []):
            root = lib.get("path") or ""
            if not lib.get("exists", True) or not os.path.isdir(root):
                continue
            for dirpath, _dirs, files in os.walk(root):
                for f in sorted(files):
                    if not f.endswith(".blend"):
                        continue
                    fp = os.path.join(dirpath, f)
                    rel = os.path.relpath(fp, root)
                    try:
                        names = self.client.run_code(_LIST_CODE.format(fp=fp)).get("objects", [])
                    except Exception:  # noqa: BLE001, S112
                        continue
                    assets += [{"name": n, "filepath": fp, "library": lib.get("name", ""),
                                "tags": list(_tokens(rel))} for n in names]
        return assets

    def measure(self, cand: AssetCandidate) -> AssetCandidate:
        """Append (if from a library file) and read get_bounds; failures are recorded, not raised."""
        try:
            name = cand.name
            if cand.filepath:
                name = self.client.request("append_asset", {"filepath": cand.filepath, "asset_name": cand.name})["name"]
            b = self.client.request("get_bounds", {"name": name})
            size = b.get("size") or b.get("obb_dims")
            cand.dims = (float(size[0]), float(size[1]), float(size[2]))
        except Exception as e:  # noqa: BLE001
            cand.measure_error = str(e)
        return cand

    def resolve_asset(self, category: str, style: str | None = None, top_k: int = 3) -> list[AssetCandidate]:
        ranked = rank_assets(category, style, self.collect_assets())[:top_k]
        return [self.measure(c) for c in ranked]
