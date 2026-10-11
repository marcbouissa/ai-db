"""Dimension priors per object category (data in ``priors.json``)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ai_db.errors import AiDbConfigError

SUPPORTS = ("floor", "surface", "wall", "ceiling", "ground")
_PRIORS_PATH = Path(__file__).with_name("priors.json")
_Dims = tuple[float, float, float]


@dataclass(frozen=True)
class CategoryPrior:
    """Real-world size range of a category. Dims are [width, depth, height] in meters."""

    name: str
    min_m: _Dims
    typical_m: _Dims
    max_m: _Dims
    support: str
    can_support_others: bool
    default_clearance_m: float
    mount_height_m: float = 0.0  # base height for wall-supported categories


def _dims(name: str, key: str, raw: object) -> _Dims:
    if (
        not isinstance(raw, list)
        or len(raw) != 3
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in raw)
    ):
        raise AiDbConfigError(f"priors[{name}].{key}: expected 3 positive numbers [w, d, h], got {raw!r}")
    return (float(raw[0]), float(raw[1]), float(raw[2]))


def parse_priors(data: dict) -> dict[str, CategoryPrior]:
    """Validate a priors document and return {category: CategoryPrior}."""
    cats = data.get("categories")
    if not isinstance(cats, dict) or not cats:
        raise AiDbConfigError("priors: 'categories' must be a non-empty object")
    out: dict[str, CategoryPrior] = {}
    for name, e in cats.items():
        if not isinstance(e, dict):
            raise AiDbConfigError(f"priors[{name}]: expected an object")
        lo, ty, hi = (_dims(name, k, e.get(k)) for k in ("min_m", "typical_m", "max_m"))
        for i, axis in enumerate(("w", "d", "h")):
            if not lo[i] <= ty[i] <= hi[i]:
                raise AiDbConfigError(
                    f"priors[{name}]: need min <= typical <= max on {axis}, got "
                    f"{lo[i]} / {ty[i]} / {hi[i]}"
                )
        sup = e.get("support")
        if sup not in SUPPORTS:
            raise AiDbConfigError(f"priors[{name}].support: {sup!r} not in {list(SUPPORTS)}")
        cso = e.get("can_support_others")
        if not isinstance(cso, bool):
            raise AiDbConfigError(f"priors[{name}].can_support_others: expected bool, got {cso!r}")
        cl = e.get("default_clearance_m")
        if not isinstance(cl, (int, float)) or isinstance(cl, bool) or cl < 0:
            raise AiDbConfigError(f"priors[{name}].default_clearance_m: expected number >= 0, got {cl!r}")
        mh = e.get("mount_height_m", 0.0)
        if not isinstance(mh, (int, float)) or isinstance(mh, bool) or mh < 0:
            raise AiDbConfigError(f"priors[{name}].mount_height_m: expected number >= 0, got {mh!r}")
        out[name] = CategoryPrior(name, lo, ty, hi, sup, cso, float(cl), float(mh))
    return out


@lru_cache(maxsize=1)
def load_priors() -> dict[str, CategoryPrior]:
    """Load and validate the bundled priors. Raises AiDbConfigError if invalid."""
    try:
        data = json.loads(_PRIORS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AiDbConfigError(f"cannot load {_PRIORS_PATH}: {exc}") from exc
    return parse_priors(data)


def categories() -> list[str]:
    """Sorted closed vocabulary of categories."""
    return sorted(load_priors())
