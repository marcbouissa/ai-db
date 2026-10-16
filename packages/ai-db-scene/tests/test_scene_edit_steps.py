"""edit_scene steps as weak models write them (bench mining 2026-10-16, item 3).

Each case is the argument shape of a real failing edit_scene call from the benchmark
(``bench/<arm>/<brief>#i`` in docs/WEAK_MODEL_FAILURES_2026-10-16.md), on a neutral spec.
"""

from __future__ import annotations

from ai_db_scene.edits import apply_steps

SPEC = {"setting": {"kind": "indoor", "bounds_m": [8.0, 6.0, 3.0]},
        "objects": [{"id": "counter", "category": "kitchen_counter"},
                    {"id": "table", "category": "dining_table", "count": 3},
                    {"id": "sofa", "category": "sofa"}],
        "relations": [{"a": "counter", "rel": "against_wall", "b": "wall_north"},
                      {"a": "table", "rel": "in_front_of", "b": "counter"},
                      {"a": "sofa", "rel": "against_wall", "b": "wall_south"}]}


def _ok(steps):
    data, done, errors = apply_steps(SPEC, steps)
    assert data is not None and not errors, errors
    return data, done


def _err(steps) -> str:
    data, _done, errors = apply_steps(SPEC, steps)
    assert data is None and len(errors) == 1
    return errors[0]


def test_op_aliases_and_target_or_name_for_id():
    # {op: delete, target: X}, {op: delete_object, name: X}
    data, done = _ok([{"op": "delete", "target": "sofa"}])
    assert "sofa" not in {o["id"] for o in data["objects"]} and "read as 'remove_object'" in done[0]
    data, _ = _ok([{"op": "delete_object", "name": "sofa"}])
    assert "sofa" not in {o["id"] for o in data["objects"]}
    # {action: add_relation, a, rel, b} (action for op, flat relation fields)
    data, _ = _ok([{"action": "add_relation", "a": "sofa", "rel": "facing", "b": "counter"}])
    assert {"a": "sofa", "rel": "facing", "b": "counter"} in data["relations"]


def test_blender_objects_are_skipped_not_reported_as_none():
    # {op: remove_object, name: Cube} used to say "no object 'None'"
    data, done = _ok([{"op": "remove_object", "name": "Cube"}, {"op": "validate"}])
    assert data["objects"] == SPEC["objects"]
    assert "not a spec object" in done[0] and "skipped 'validate'" in done[1]
    # an unknown lowercase id still fails, naming the spec ids instead of None
    msg = _err([{"op": "remove_object", "name": "lamp"}])
    assert "no object 'lamp'" in msg and "'counter'" in msg and "None" not in msg
    msg = _err([{"op": "remove_object"}])
    assert "needs 'id'" in msg and "['op']" in msg


def test_flat_fields_are_nested():
    # {op: update_object, id, material} -> set
    data, done = _ok([{"op": "update_object", "id": "counter", "material": "wood:walnut"},
                      {"op": "add_relation", "a": "counter", "rel": "against_wall", "b": "wall_east", "gap_m": 0.5}])
    assert next(o for o in data["objects"] if o["id"] == "counter")["material"] == "wood:walnut"
    assert "moved into 'set'" in done[0]
    # {op: add_object, id, category, material}
    data, _ = _ok([{"op": "add_object", "id": "lamp_1", "category": "pendant_lamp", "material": "metal:brushed"}])
    assert data["objects"][-1] == {"id": "lamp_1", "category": "pendant_lamp", "material": "metal:brushed"}
    # {op: add, target: {object}, relations}
    data, _ = _ok([{"op": "add", "target": {"id": "clock", "category": "wall_clock"},
                    "relations": [{"a": "clock", "rel": "against_wall", "b": "wall_east"}]}])
    assert data["objects"][-1]["id"] == "clock" and data["relations"][-1]["a"] == "clock"
    # {op: update_object, target, changes: {...}}
    data, _ = _ok([{"op": "update_object", "target": "counter", "changes": {"hero": True}}])
    assert next(o for o in data["objects"] if o["id"] == "counter")["hero"] is True
    # {op: remove_relation, relation: {a, rel, b}}
    data, _ = _ok([{"op": "remove_relation", "relation": {"a": "sofa", "rel": "against_wall", "b": "wall_south"}}])
    assert len(data["relations"]) == 2


def test_add_with_to_becomes_relations():
    # {op: add, id, category, material, to: {above: table_1, gap_m}}
    data, _ = _ok([{"op": "add", "id": "pendant_1", "category": "pendant_lamp", "to": {"above": "table", "gap_m": 1.2}}])
    assert data["relations"][-1] == {"a": "pendant_1", "rel": "above", "b": "table", "gap_m": 1.2}


def test_move_replaces_the_objects_relations():
    # {op: move, id, to: {against_wall: wall_south, gap_m: 0.5}}
    data, done = _ok([{"op": "move", "id": "table_2", "to": {"against_wall": "wall_south", "gap_m": 0.5}}])
    assert [r for r in data["relations"] if r["a"] == "table"] == [
        {"a": "table", "rel": "against_wall", "b": "wall_south", "gap_m": 0.5}]
    assert "'table_2' read as 'table'" in done[0]
    # {op: update_object, id, relations: [...]} means the same
    data, _ = _ok([{"op": "update_object", "id": "counter",
                    "relations": [{"a": "counter", "rel": "against_wall", "b": "wall_east"}]}])
    assert [r["b"] for r in data["relations"] if r["a"] == "counter"] == ["wall_east"]


def test_absolute_transforms_get_a_precise_error():
    # {op: move, id, location: [...], rotation_euler: [...]} and {op: move, target, offset_m}
    for step in ({"op": "move", "id": "sofa", "location": [1.5, 2.8, 0], "rotation_euler": [0, 0, 0.3]},
                 {"op": "move", "target": "counter", "offset_m": [0, 0.5, 0]},
                 {"op": "update_object", "target": "sofa", "location": [2, 2, 0]}):
        msg = _err([step])
        assert "relation" in msg and "None" not in msg and "fields []" not in msg, msg


def test_update_without_set_names_the_missing_field():
    msg = _err([{"op": "update_object", "id": "sofa"}])
    assert "needs 'set'" in msg and "fields ['set']" not in msg


def test_unknown_ops_and_text_steps():
    assert "not an edit_scene op" in _err([{"op": "explode"}])
    msg = _err(["Place the table on top of the rug"])
    assert "expected an object" in msg
