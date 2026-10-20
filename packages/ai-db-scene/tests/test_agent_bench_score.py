"""Agent benchmark quality score (eval/agent_bench/bench.py) on synthetic inventories."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "eval" / "agent_bench"


@pytest.fixture(scope="module")
def bench() -> Any:
    sys.path.insert(0, str(BENCH_DIR))
    try:
        return importlib.import_module("bench")
    finally:
        sys.path.remove(str(BENCH_DIR))


def _obj(name: str, typ: str = "MESH", parent: str | None = None, faces: int = 0,
         category: str = "", size: list[float] | None = None, mats: int = 1) -> dict[str, Any]:
    return {"name": name, "type": typ, "category": category, "parent": parent, "mats": mats,
            "tris": 2 * faces, "faces": faces, "size_m": size}


def _asset(name: str, category: str, size: list[float], parts: int = 3) -> list[dict[str, Any]]:
    """An asset root (empty) with `parts` box children: what build_asset makes."""
    root = _obj(name, "EMPTY", category=category, size=size, mats=0)
    return [root] + [_obj(f"{name}_p{i}", parent=name, faces=6) for i in range(parts)]


def _inv(objs: list[dict[str, Any]]) -> dict[str, Any]:
    return {"objects": objs, "camera": "cam", "lights": 1}


def _living_room(size_ok: bool = True) -> list[dict[str, Any]]:
    sofa = [2.1, 0.95, 0.85] if size_ok else [6.0, 3.0, 0.85]
    return (_asset("sofa", "sofa", sofa) + _asset("tv_stand", "tv_stand", [1.5, 0.45, 0.55])
            + _asset("tv", "tv", [1.2, 0.08, 0.7]) + _asset("coffee_table", "coffee_table", [1.1, 0.6, 0.42])
            + _asset("rug", "rug", [2.0, 1.4, 0.01]) + _asset("armchair_1", "armchair", [0.85, 0.9, 0.9])
            + _asset("armchair_2", "armchair", [0.9, 0.85, 0.9]) + _asset("floor_lamp", "floor_lamp", [0.4, 0.4, 1.6]))


def test_geometry_weight(bench: Any) -> None:
    box = _obj("box", faces=6)
    detailed = _obj("blob", faces=200)
    empty = _obj("e", "EMPTY")
    inv = _inv([box, detailed, empty, *_asset("sofa", "sofa", [2, 1, 1], parts=2)])
    assert bench.geometry_weight(inv, box) == 0.5
    assert bench.geometry_weight(inv, detailed) == 1.0
    assert bench.geometry_weight(inv, empty) == 0.0
    assert bench.geometry_weight(inv, inv["objects"][3]) == 1.0
    # an inventory from before "faces" existed: 12 triangles is still a box
    old = {"name": "b", "type": "MESH", "category": "", "parent": None, "mats": 1, "tris": 12}
    assert bench.geometry_weight(_inv([old]), old) == 0.5


def test_full_scene_scores_high(bench: Any) -> None:
    q = bench.score("living_room", _inv(_living_room()), {"issues": []})
    assert q["formula"] == "v2"
    assert q["checklist_pct"] == 100.0 and q["size_pct"] == 100.0 and q["coverage_pct"] == 100.0
    assert q["score"] == 100.0
    # old fields stay for older report code
    for key in ("checklist_pct", "validator_pct", "basics_pct", "blocking", "warnings", "issue_kinds",
                "checklist", "meshes", "triangles"):
        assert key in q


def test_named_boxes_do_not_saturate(bench: Any) -> None:
    """The old score gave named 6-face boxes full checklist credit."""
    names = ["sofa", "tv_stand", "tv", "coffee_table", "rug", "armchair_1", "armchair_2", "floor_lamp"]
    q = bench.score("living_room", _inv([_obj(n, faces=6) for n in names]), {"issues": []})
    assert q["checklist_name_pct"] == 100.0
    assert q["checklist_pct"] < 60.0  # half credit per box ("tv" also matches the tv_stand box)
    assert q["size_pct"] is None  # no categories: size weight moves to the checklist
    assert q["coverage_pct"] == 50.0  # 8 meshes for a need of 8 objects x 2 parts
    assert q["score"] < 75.0  # the old formula gave 100


def test_implausible_size_costs(bench: Any) -> None:
    good = bench.score("living_room", _inv(_living_room()), {"issues": []})
    bad = bench.score("living_room", _inv(_living_room(size_ok=False)), {"issues": []})
    assert bad["size_pct"] == pytest.approx(100 * 7 / 8, abs=0.1)
    assert bad["score"] < good["score"]


def test_size_plausible_accepts_turned_footprint(bench: Any) -> None:
    from ai_db_scene.priors import load_priors

    sofa = load_priors()["sofa"]
    assert bench.size_plausible([2.1, 0.95, 0.85], sofa)
    assert bench.size_plausible([0.95, 2.1, 0.85], sofa)  # turned 90 degrees
    assert not bench.size_plausible([2.1, 0.95, 2.5], sofa)  # far too tall


def test_validator_and_basics_still_count(bench: Any) -> None:
    issues = {"issues": [{"severity": "blocking", "check": "overlap"}, {"severity": "warning", "check": "x"}]}
    q = bench.score("living_room", _inv(_living_room()), issues)
    assert q["blocking"] == 1 and q["warnings"] == 1 and q["validator_pct"] == 90.0
    assert q["score"] == pytest.approx(100 - 0.25 * 10, abs=0.1)


def _found(bench: Any, brief: str, objs: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {i["item"]: i["objects"] for i in bench.checklist(brief, _inv(objs))}


def test_a_fence_named_parasol_is_no_parasol(bench: Any) -> None:
    """qwen aidb6 beach_cove (pre-terrain): a fence_section named "parasol" scored as the parasol."""
    got = _found(bench, "beach_cove", _asset("parasol", "fence_section", [2.0, 0.06, 1.2])
                 + _asset("umbrella_1", "parasol", [2.8, 2.8, 2.45]))
    assert got["parasol"] == ["umbrella_1"]


def test_street_lamps_are_no_trees(bench: Any) -> None:
    lamps = _asset("street_lamp_1", "street_lamp", [0.4, 0.4, 4.5]) + [_obj("street_lamp_2", faces=200,
                                                                           size=[0.4, 0.4, 4.5])]
    got = _found(bench, "park_corner", lamps + _asset("tree_1", "tree", [3, 3, 6]))
    assert got["trees"] == ["tree_1"]
    assert sorted(got["street lamps"]) == ["street_lamp_1", "street_lamp_2"]


def test_a_name_counts_for_its_head_item(bench: Any) -> None:
    """Untagged objects: desk_chair is a chair, bar_stool a stool, tv_stand a stand, pallet_rack a rack."""
    def untagged(n: str, size: list[float]) -> dict[str, Any]:
        return _obj(n, faces=200, size=size)

    assert _found(bench, "classroom", [untagged("desk_chair_1", [0.45, 0.5, 0.9])])["desks"] == []
    assert _found(bench, "classroom", [untagged("desk_chair_1", [0.45, 0.5, 0.9])])["chairs"] == ["desk_chair_1"]
    cafe = _found(bench, "cafe", [untagged("Bar_Stool_1", [0.4, 0.4, 0.75])])
    assert cafe["counter"] == [] and cafe["stools"] == ["Bar_Stool_1"]
    living = _found(bench, "living_room", [untagged("TVStand", [1.5, 0.45, 0.55])])
    assert living["tv"] == [] and living["tv stand"] == ["TVStand"]
    wh = _found(bench, "warehouse", [untagged("pallet_rack_1", [2.7, 1.1, 4.5])])
    assert wh["pallets"] == [] and wh["pallet racks"] == ["pallet_rack_1"]
    # the room's baseboard is no whiteboard
    assert _found(bench, "classroom", [untagged("Room_Baseboard", [8, 6, 0.15])])["whiteboard"] == []


def test_a_name_match_needs_a_plausible_size(bench: Any) -> None:
    """Name matching is the fallback for arms that tag nothing; a size prior gates it (x2 either way)."""
    fence = _obj("parasol", faces=200, size=[3.0, 0.06, 1.2])
    real = _obj("parasol", faces=200, size=[2.5, 2.5, 2.4])
    assert _found(bench, "beach_cove", [fence])["parasol"] == []
    assert _found(bench, "beach_cove", [real])["parasol"] == ["parasol"]
    # unknown size: no gate (old inventories); a flat board modelled as a plane still counts
    assert _found(bench, "beach_cove", [_obj("parasol", faces=200)])["parasol"] == ["parasol"]
    assert _found(bench, "classroom", [_obj("whiteboard", faces=1, size=[2.4, 0.0, 1.2])])["whiteboard"]


def test_lights_and_cameras_never_count(bench: Any) -> None:
    objs = [_obj("ai_db_daylight_Sun", "LIGHT"), _obj("lamp1_light", "LIGHT"), _obj("pendant_lamp_light", "LIGHT")]
    got = _found(bench, "park_corner", objs)
    assert got["street lamps"] == []
    assert _found(bench, "round_lobby", objs)["pendant lamp"] == []


def test_a_broad_category_lets_the_name_decide(bench: Any) -> None:
    """A "table" named tv_stand is the TV stand; a "chair" named student_desk_seat_3 (ai-db's desk
    seats) is a chair; "table_3" tagged table in a cafe is still a table; a "table" named side table
    is no coffee table."""
    living = _found(bench, "living_room", _asset("tv_stand", "table", [1.5, 0.45, 0.55])
                    + _asset("Side_Table_1", "table", [0.5, 0.5, 0.55]))
    assert living["tv stand"] == ["tv_stand"] and living["coffee table"] == []
    cls = _found(bench, "classroom", _asset("student_desk_seat_3", "chair", [0.45, 0.5, 0.9]))
    assert cls["chairs"] == ["student_desk_seat_3"] and cls["desks"] == []
    assert _found(bench, "cafe", _asset("t3", "table", [0.8, 0.8, 0.75]))["tables"] == ["t3"]


def test_old_name_count_is_kept_for_comparison(bench: Any) -> None:
    q = bench.score("beach_cove", _inv(_asset("parasol", "fence_section", [2.0, 0.06, 1.2])), {"issues": []})
    item = next(i for i in q["checklist"] if i["item"] == "parasol")
    assert item["found"] == 0 and item["name_found"] == 1
    assert q["checklist_name_pct"] > q["checklist_pct"]
