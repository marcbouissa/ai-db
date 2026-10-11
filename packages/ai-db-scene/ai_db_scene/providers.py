"""Asset providers: real model files for a category, used before part recipes when found.

Both are off by default (see ``plugins.scene`` in config.py):

- ``LocalLibrary``: folders of ``.glb``/``.gltf``/``.blend`` files, matched to categories by
  filename tokens (``traffic_cone_01.glb`` -> traffic_cone). A ``.blend`` must hold an object
  named like the file's stem; it is appended (``append_asset``). glTF files are imported
  (``import_file``) and fitted to the placement. CC0 pack names are understood too:
  camelCase/snake words are split (``chairCushion``, ``Bed_Double``) and ``CATEGORY_ALIASES``
  maps pack vocabulary to ours (``bookcaseOpen`` -> bookshelf, ``chairDesk`` -> office_chair);
  ``plugins.scene.aliases`` adds or overrides entries.
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


_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

#: File-name vocabulary of CC0 furniture packs (Kenney, Quaternius, ...) -> our category.
#: Keys are space-separated words, matched in any order like category names; the match with
#: the most words wins, so "lounge chair" beats "chair" and "cabinet television" beats "tv".
CATEGORY_ALIASES: dict[str, str] = {
    "bookcase": "bookshelf", "book case": "bookshelf", "shelf": "shelving_unit", "shelves": "shelving_unit",
    "books": "book_stack",
    "couch": "sofa", "loveseat": "sofa", "lounge chair": "armchair", "arm chair": "armchair",
    "chair desk": "office_chair", "desk chair": "office_chair", "chair office": "office_chair",
    "chair dining": "dining_chair", "table": "dining_table", "table dining": "dining_table",
    "table side": "side_table", "end table": "side_table", "night stand": "nightstand",
    "bedside table": "nightstand", "drawer": "dresser", "drawers": "dresser", "chest drawers": "dresser",
    "closet": "wardrobe", "cupboard": "cabinet",
    "television": "tv", "cabinet television": "tv_stand", "tv cabinet": "tv_stand", "television stand": "tv_stand",
    "computer screen": "monitor", "screen": "monitor",
    "fridge": "refrigerator", "kitchen fridge": "refrigerator", "kitchen cabinet": "kitchen_counter",
    "kitchen bar": "kitchen_island", "oven": "stove", "kitchen stove": "stove",
    "lamp ceiling": "ceiling_light", "light ceiling": "ceiling_light", "lamp wall": "wall_sconce",
    "lamp pendant": "pendant_lamp", "lamp desk": "table_lamp", "light floor": "floor_lamp",
    "trashcan": "trash_can", "trash": "trash_can", "bin": "trash_can", "garbage": "trash_can",
    "plant": "potted_plant", "plant pot": "potted_plant", "houseplant": "potted_plant",
    "shower": "shower_stall", "bath": "bathtub", "tub": "bathtub", "bathroom sink": "sink",
    "carpet": "rug", "doorway": "door", "box": "crate", "cardboard box": "crate", "frame": "picture_frame",
    "painting": "picture_frame", "clock": "wall_clock",
}


def split_words(text: str) -> list[str]:
    """Lowercase words of a file stem: camelCase, snake_case, kebab and digits split apart
    (``lampRoundFloor`` -> lamp round floor, ``TVStand_02`` -> tv stand 02)."""
    return [w.lower() for w in _WORD.findall(text)]


def _tokens(text: str) -> set[str]:
    return {t for t in split_words(text.replace("_", " ")) if not _NUMBER.match(t)}


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


def match_category(stem: str, categories: list[str], aliases: dict[str, str] | None = None) -> str | None:
    """The category whose words (or an alias's words) all occur in ``stem``; the most
    specific (most words) wins (``lamp_post_double_02`` -> lamp_post_double, not street_lamp;
    ``chairDesk`` -> office_chair, not chair or desk). At equal words a category name beats
    an alias, then ties go by name. Numbers are ignored. ``aliases`` defaults to
    ``CATEGORY_ALIASES``; aliases to categories not in ``categories`` are skipped."""
    have = _tokens(stem)
    known = set(categories)
    hits = [(len(_tokens(c)), 0, c) for c in categories if _tokens(c) and _tokens(c) <= have]
    for words, cat in (CATEGORY_ALIASES if aliases is None else aliases).items():
        if cat in known and _tokens(words) and _tokens(words) <= have:
            hits.append((len(_tokens(words)), 1, cat))
    return min(hits, key=lambda h: (-h[0], h[1], h[2]))[2] if hits else None


class LocalLibrary:
    """Index of the model files under ``dirs`` by category (built once, on first use)."""

    name = "local"

    def __init__(self, dirs: list[str] | tuple[str, ...], categories: list[str] | None = None,
                 aliases: dict[str, str] | None = None) -> None:
        self.dirs = [str(d) for d in dirs]
        self._categories = categories
        #: built-in aliases, then the user's (``plugins.scene.aliases``) on top
        self.aliases = {**CATEGORY_ALIASES, **{" ".join(split_words(k.replace("_", " "))): v
                                               for k, v in (aliases or {}).items()}}
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
                        cat = match_category(stem, self._categories, self.aliases)
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
        out.append(LocalLibrary(cfg.asset_dirs, aliases=dict(cfg.aliases)))
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
