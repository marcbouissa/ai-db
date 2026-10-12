"""Room kits: typical contents for the room types a brief names, all known categories."""

from __future__ import annotations

from ai_db_scene.priors import load_priors
from ai_db_scene.room_kits import KITS, kits_for
from ai_db_scene.tools import scene_pipeline as sp


def test_every_kit_uses_known_categories():
    known = set(load_priors())
    assert all(c in known for _, items, _ in KITS.values() for c, _, _ in items)


def test_brief_selects_kits_and_schema_returns_them():
    assert [k["room"] for k in kits_for("A cozy living room next to a small kitchen")] == ["living_room", "kitchen"]
    assert kits_for("an abstract sculpture") == []
    out = sp._handle_get_scene_spec_schema({"brief": "a bedroom for two"})
    assert out["room_kits"][0]["room"] == "bedroom" and any(i["category"] == "nightstand" for i in out["room_kits"][0]["typical"])
