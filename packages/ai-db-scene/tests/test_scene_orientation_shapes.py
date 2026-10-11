"""A1 yaw from relations (front = -Y), A3 support vocabulary, A6 room shapes, H2 detail."""

from __future__ import annotations

import math
import random

from ai_db_scene.layout import check_layout, solve_layout
from ai_db_scene.priors import SUPPORTS, load_priors
from ai_db_scene.spec import SceneSpec, validate_spec
from ai_db_scene.tools import scene_pipeline as sp


def _front(yaw_deg: float) -> tuple[float, float]:
    y = math.radians(yaw_deg)
    return math.sin(y), -math.cos(y)


def _dir_err(pl, other) -> float:
    dx, dy = other.location[0] - pl.location[0], other.location[1] - pl.location[1]
    fx, fy = _front(pl.rotation_deg_z)
    return math.degrees(math.acos(max(-1, min(1, (fx * dx + fy * dy) / math.hypot(dx, dy)))))


def test_two_sofas_facing_each_other():
    spec = SceneSpec.from_dict({
        "setting": {"kind": "indoor", "bounds_m": [8, 7, 3]},
        "objects": [{"id": "s1", "category": "sofa"}, {"id": "s2", "category": "sofa"}],
        "relations": [{"a": "s1", "rel": "facing", "b": "s2"}, {"a": "s2", "rel": "facing", "b": "s1"}],
    })
    a, b = {p.id: p for p in solve_layout(spec)}.values()
    assert abs(((a.rotation_deg_z - b.rotation_deg_z) % 360) - 180) < 1.0
    assert _dir_err(a, b) < 1.0 and _dir_err(b, a) < 1.0
    assert not [i for i in check_layout(spec, [a, b]) if i.startswith("soft: ")]
    assert a.facing_target == "s2"


def test_against_wall_front_points_into_room():
    walls = {"wall_north": (0, -1), "wall_south": (0, 1), "wall_east": (-1, 0), "wall_west": (1, 0)}
    for wall, n in walls.items():
        spec = SceneSpec.from_dict({
            "setting": {"kind": "indoor", "bounds_m": [6, 5, 3]},
            "objects": [{"id": "sofa", "category": "sofa"}],
            "relations": [{"a": "sofa", "rel": "against_wall", "b": wall}],
        })
        (pl,) = solve_layout(spec)
        fx, fy = _front(pl.rotation_deg_z)
        assert fx * n[0] + fy * n[1] > 0.999, wall


def test_in_front_of_faces_target_and_beside_matches_yaw_and_default_faces_center():
    spec = SceneSpec.from_dict({
        "setting": {"kind": "indoor", "bounds_m": [8, 7, 3]},
        "objects": [{"id": "sofa", "category": "sofa"}, {"id": "table", "category": "coffee_table"},
                    {"id": "lamp", "category": "floor_lamp"}, {"id": "plant", "category": "potted_plant"}],
        "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_north"},
                      {"a": "table", "rel": "in_front_of", "b": "sofa"},
                      {"a": "lamp", "rel": "beside", "b": "sofa"}],
    })
    pl = {p.id: p for p in solve_layout(spec)}
    assert abs((pl["table"].rotation_deg_z - pl["sofa"].rotation_deg_z) % 360 - 180) < 1
    assert abs((pl["lamp"].rotation_deg_z - pl["sofa"].rotation_deg_z + 180) % 360 - 180) < 1
    centre = type("C", (), {"location": (4.0, 3.5, 0)})()
    assert _dir_err(pl["plant"], centre) < 20


def test_check_layout_flags_off_facing():
    spec = SceneSpec.from_dict({
        "setting": {"kind": "indoor", "bounds_m": [8, 7, 3]},
        "objects": [{"id": "s1", "category": "sofa"}, {"id": "s2", "category": "sofa"}],
        "relations": [{"a": "s1", "rel": "facing", "b": "s2"}],
    })
    pls = solve_layout(spec)
    bad = [type(p)(p.id, p.location, p.rotation_deg_z + 40, p.dims, p.category) for p in pls]
    assert any(i.startswith("soft:") and "facing" in i for i in check_layout(spec, bad))


def test_support_vocab_shared():
    assert SUPPORTS == ("floor", "surface", "wall", "ceiling", "ground")
    assert {p.support for p in load_priors().values()} <= set(SUPPORTS)


def _ellipse(rng: random.Random, i: int) -> dict:
    w, d = rng.uniform(7, 12), rng.uniform(6, 10)
    objs = [{"id": "sofa", "category": "sofa", "hero": True}, {"id": "desk", "category": "desk"},
            {"id": "chairs", "category": "armchair", "count": 2}, {"id": "rug", "category": "rug"},
            {"id": "shelf", "category": "bookshelf"}, {"id": "lamp", "category": "floor_lamp"},
            {"id": "win", "category": "window"}, {"id": "plant", "category": "potted_plant"}]
    rels = [{"a": "sofa", "rel": "against_wall", "b": "wall", "at_deg": 90 + 20 * (i % 3)},
            {"a": "shelf", "rel": "against_wall", "b": "wall", "at_deg": 270},
            {"a": "win", "rel": "against_wall", "b": "wall", "at_deg": 0},
            {"a": "plant", "rel": "against_wall", "b": "wall"},
            {"a": "rug", "rel": "centered_in", "b": "room"},
            {"a": "desk", "rel": "centered_in", "b": "room"},
            {"a": "chairs", "rel": "facing", "b": "desk"},
            {"a": "lamp", "rel": "beside", "b": "sofa"}]
    return {"setting": {"kind": "indoor", "shape": "ellipse", "bounds_m": [w, d, 3]}, "objects": objs, "relations": rels}


def test_ten_ellipse_rooms_zero_hard_violations():
    rng = random.Random(7)
    for i in range(10):
        spec = SceneSpec.from_dict(_ellipse(rng, i))
        pls = solve_layout(spec)
        assert not [x for x in check_layout(spec, pls) if x.startswith("hard")], i
        sofa = next(p for p in pls if p.id == "sofa")
        d = spec.setting.bounds_m[1]
        # at_deg 90+: sofa sits on the north (+y) side with its front pointing into the room
        assert sofa.location[1] > d / 2 and _front(sofa.rotation_deg_z)[1] < 0


def test_polygon_room_and_validation():
    pts = [[0, 0], [8, 0], [8, 4], [4, 7], [0, 4]]
    data = {
        "setting": {"kind": "indoor", "shape": "polygon", "points": pts, "bounds_m": [8, 7, 3]},
        "objects": [{"id": "sofa", "category": "sofa"}, {"id": "bed", "category": "bed"}],
        "relations": [{"a": "sofa", "rel": "against_wall", "b": "wall_4"}, {"a": "bed", "rel": "against_wall", "b": "wall", "at_deg": 270}],
    }
    spec = SceneSpec.from_dict(data)
    pls = solve_layout(spec)
    assert not [x for x in check_layout(spec, pls) if x.startswith("hard")]
    sofa = next(p for p in pls if p.id == "sofa")
    assert sofa.location[0] < 4 and sofa.location[1] > 4  # wall_4 runs (4,7)->(0,4)
    bad = {**data, "setting": {**data["setting"], "points": [[0, 0], [9, 0], [0, 4]]}}
    assert validate_spec(bad)[0] is None
    assert validate_spec({**data, "setting": {**data["setting"], "shape": "blob"}})[0] is None
    assert validate_spec({**data, "setting": {**data["setting"], "shape": "ellipse", "points": pts}})[0] is None


def test_apply_layout_room_shell_openings_detail_and_facing_prop():
    spec = SceneSpec.from_dict({
        "setting": {"kind": "indoor", "shape": "ellipse", "bounds_m": [10, 8, 3]},
        "objects": [{"id": "s1", "category": "sofa", "detail": "hero"}, {"id": "s2", "category": "sofa"},
                    {"id": "win", "category": "window"}],
        "relations": [{"a": "s1", "rel": "facing", "b": "s2"}, {"a": "s2", "rel": "facing", "b": "s1"},
                      {"a": "win", "rel": "against_wall", "b": "wall", "at_deg": 0}],
    })
    pls = solve_layout(spec)
    cmds, names = sp.build_layout_commands(spec, pls)
    shell = cmds[0]
    assert shell["type"] == "room_shell" and names[0] == "room_shell"
    prm = shell["params"]
    assert prm["shape"] == "ellipse" and prm["size"] == [10.0, 8.0] and prm["height"] == 3.0 and "wall_thickness" in prm
    assert len(prm["openings"]) == 1 and prm["openings"][0]["kind"] == "window" and prm["openings"][0]["wall"] == "wall"
    code = next(c for c in cmds if c["type"] == "run_code" and "ai_db_id" in c["params"]["code"])["params"]["code"]
    op = prm["openings"][0]
    assert "at" in op and "sill" in op and "center" not in op  # room_shell's opening keys
    assert '"facing_target": "s2"' in code and '"detail": "hero"' in code
    assert "yaw_deg" in sp.placement_to_dict(pls[0])
    assert SceneSpec.from_dict(spec.to_dict()).objects[0].detail == "hero"
    assert validate_spec({"setting": {"kind": "indoor", "bounds_m": [4, 4, 3]},
                          "objects": [{"id": "a", "category": "sofa", "detail": "ultra"}]})[0] is None
