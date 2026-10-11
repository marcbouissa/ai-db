"""Asset providers: real model files for a category, used before part recipes when found.

Both are off by default (see ``plugins.scene`` in config.py):

- ``LocalLibrary``: folders of ``.glb``/``.gltf``/``.blend`` files, matched to categories by
  filename tokens (``traffic_cone_01.glb`` -> traffic_cone). A ``.blend`` must hold an object
  named like the file's stem; it is appended (``append_asset``). glTF files are imported
  (``import_file``) and fitted to the placement.
- ``PolyHaven``: models from https://api.polyhaven.com, downloaded once (glTF + its textures)
  into a cache dir. Disabled unless ``polyhaven.enabled`` and ``polyhaven.license_acknowledged``.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ai_db_scene.constants import (
    ASSET_FILE_TYPES,
    POLYHAVEN_API,
    POLYHAVEN_CACHE_DIR,
    POLYHAVEN_INDEX_TTL_S,
    POLYHAVEN_TIMEOUT_S,
    POLYHAVEN_USER_AGENT,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"^\d+$")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower().replace("_", " ")) if not _NUMBER.match(t)}


@dataclass(frozen=True)
class AssetFile:
    """A model file for one category. ``kind`` is ``blend`` (append ``asset_name``) or ``gltf``."""

    category: str
    filepath: str
    kind: str
    asset_name: str | None = None
    source: str = "local"


class AssetProvider(Protocol):
    name: str

    def find(self, category: str) -> AssetFile | None:
        """A model file for ``category``, or None."""
        ...


def match_category(stem: str, categories: list[str]) -> str | None:
    """The category whose words all occur in ``stem``; the most specific (most words) wins
    (``lamp_post_double_02`` -> lamp_post_double, not street_lamp), ties by name. Numbers are
    ignored."""
    have = _tokens(stem)
    hits = [c for c in categories if _tokens(c) and _tokens(c) <= have]
    return min(hits, key=lambda c: (-len(_tokens(c)), c)) if hits else None


class LocalLibrary:
    """Index of the model files under ``dirs`` by category (built once, on first use)."""

    name = "local"

    def __init__(self, dirs: list[str] | tuple[str, ...], categories: list[str] | None = None) -> None:
        self.dirs = [str(d) for d in dirs]
        self._categories = categories
        self._index: dict[str, list[AssetFile]] | None = None

    def index(self) -> dict[str, list[AssetFile]]:
        if self._index is None:
            if self._categories is None:
                from ai_db_scene.priors import categories

                self._categories = categories()
            found: dict[str, list[AssetFile]] = {}
            for root in self.dirs:
                for dirpath, dirnames, files in os.walk(root):
                    dirnames.sort()
                    for fn in sorted(files):
                        stem, ext = os.path.splitext(fn)
                        if ext.lower() not in ASSET_FILE_TYPES:
                            continue
                        cat = match_category(stem, self._categories)
                        if cat is None:
                            continue
                        blend = ext.lower() == ".blend"
                        found.setdefault(cat, []).append(AssetFile(
                            cat, os.path.join(dirpath, fn), "blend" if blend else "gltf",
                            stem if blend else None))
            self._index = found
        return self._index

    def find(self, category: str) -> AssetFile | None:
        files = self.index().get(category)
        # deterministic: the shortest (least decorated) name, then the path
        return min(files, key=lambda f: (len(os.path.basename(f.filepath)), f.filepath)) if files else None


Fetch = Callable[[str], bytes]


def _http_get(url: str) -> bytes:
    # Poly Haven rejects requests without a User-Agent naming the client
    req = urllib.request.Request(url, headers={"User-Agent": POLYHAVEN_USER_AGENT})
    with urllib.request.urlopen(req, timeout=POLYHAVEN_TIMEOUT_S) as resp:
        data: bytes = resp.read()
        return data


def default_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, POLYHAVEN_CACHE_DIR)


class PolyHaven:
    """Poly Haven models: pick one by name/tags/categories, download its glTF once."""

    name = "polyhaven"

    def __init__(self, cache_dir: str | None = None, resolution: str = "1k", fetch: Fetch | None = None,
                 api: str = POLYHAVEN_API) -> None:
        self.cache = Path(cache_dir or default_cache_dir())
        self.resolution = resolution
        self.api = api.rstrip("/")
        self._fetch = fetch or _http_get
        self._models: dict[str, Any] | None = None

    def _json(self, path: str) -> Any:
        return json.loads(self._fetch(f"{self.api}{path}").decode("utf-8"))

    def models(self) -> dict[str, Any]:
        """id -> asset info of every model; cached on disk for POLYHAVEN_INDEX_TTL_S."""
        if self._models is None:
            idx = self.cache / "models.json"
            if idx.is_file() and time.time() - idx.stat().st_mtime < POLYHAVEN_INDEX_TTL_S:
                self._models = json.loads(idx.read_text(encoding="utf-8"))
            else:
                self._models = self._json("/assets?t=models")
                self.cache.mkdir(parents=True, exist_ok=True)
                idx.write_text(json.dumps(self._models), encoding="utf-8")
        return self._models or {}

    def best_model(self, category: str) -> str | None:
        """The model whose id/name/tags/categories cover the category's words best."""
        want = _tokens(category)
        ranked: list[tuple[float, str]] = []
        for mid, info in self.models().items():
            name_t = _tokens(mid) | _tokens(str(info.get("name", "")))
            tag_t: set[str] = set()
            for t in [*(info.get("tags") or []), *(info.get("categories") or [])]:
                tag_t |= _tokens(str(t))
            if want <= name_t:
                score = 3.0
            elif want <= name_t | tag_t:
                score = 2.0
            else:
                continue
            ranked.append((-(score - 0.01 * len(name_t - want)), mid))  # fewer extra words: the plain object
        return min(ranked)[1] if ranked else None

    def _download(self, mid: str) -> str:
        """Download the model's glTF and every file it includes; return the .gltf path."""
        files = self._json(f"/files/{mid}")
        try:
            entry = files["gltf"][self.resolution]["gltf"]
        except (KeyError, TypeError) as e:
            raise ValueError(f"Poly Haven model {mid!r} has no {self.resolution} glTF") from e
        root = (self.cache / mid / self.resolution).resolve()
        main = root / os.path.basename(str(entry["url"]).split("?")[0])
        if main.is_file():
            return str(main)
        for rel, inc in sorted((entry.get("include") or {}).items()):
            dest = (root / rel).resolve()
            if root not in dest.parents:  # never write outside this model's folder
                raise ValueError(f"Poly Haven model {mid!r}: unsafe include path {rel!r}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self._fetch(inc["url"]))
        root.mkdir(parents=True, exist_ok=True)
        tmp = main.with_suffix(".part")
        tmp.write_bytes(self._fetch(entry["url"]))
        tmp.replace(main)  # the .gltf last: its presence means the download is complete
        return str(main)

    def find(self, category: str) -> AssetFile | None:
        mid = self.best_model(category)
        if mid is None:
            return None
        return AssetFile(category, self._download(mid), "gltf", None, f"polyhaven:{mid}")


def providers_from_config(app_config: Any | None = None) -> list[AssetProvider]:
    """The providers ``plugins.scene`` turns on: local library first, then Poly Haven."""
    from ai_db_scene.config import scene_config

    if app_config is None:
        from ai_db.config import load_config
        from ai_db.errors import AiDbConfigError

        try:
            app_config = load_config(check_env=False)
        except AiDbConfigError:
            return []  # no config file: nothing configured, nothing on
    cfg = scene_config(app_config)
    out: list[AssetProvider] = []
    if cfg.asset_dirs:
        out.append(LocalLibrary(cfg.asset_dirs))
    if cfg.polyhaven.enabled and cfg.polyhaven.license_acknowledged:
        out.append(PolyHaven(cfg.polyhaven.cache_dir, cfg.polyhaven.resolution))
    return out


def find_asset(providers: list[AssetProvider], category: str) -> AssetFile | None:
    """First provider's match. A provider that fails (network, bad file) is skipped: the
    recipe or generated geometry is always there to fall back on."""
    for p in providers:
        try:
            found = p.find(category)
        except (OSError, ValueError, KeyError) as e:
            import logging

            logging.getLogger(__name__).warning("asset provider %s failed for %s: %s", p.name, category, e)
            continue
        if found is not None:
            return found
    return None
