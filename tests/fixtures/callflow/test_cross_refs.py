from ai_db.parser.ts_graph import extract_graph

with open("tests/fixtures/callflow/stage_manager.py") as f:
    STAGE_MANAGER = f.read()


def test_extract_cross_refs_stage_manager():
    _, refs, _ = extract_graph("tests/fixtures/callflow/stage_manager.py", STAGE_MANAGER)

    calls = [r for r in refs if r["ref_type"] == "call"]
    assert len(calls) >= 6

    launch_refs = [r for r in calls if r["caller_name"] == "module.launch_stage_ov"]
    assert len(launch_refs) == 3
    callee_names = [r["callee_name"] for r in launch_refs]
    assert callee_names == ["open_stage", "wait_for_stage_load", "preload_prefabs"]

    for r in launch_refs:
        assert r["await_kind"] == "await"
        assert r["seq"] is not None
        assert r["call_col"] is not None

    open_refs = [r for r in calls if r["caller_name"] == "module.open_stage"]
    assert len(open_refs) == 3
    callee_names = [r["callee_name"] for r in open_refs]
    assert callee_names == ["create_task", "_load_async", "_init_ui"]

    # create_task is a spawn function
    create_task_refs = [r for r in open_refs if r["callee_name"] == "create_task"]
    assert len(create_task_refs) == 1
    assert create_task_refs[0]["await_kind"] == "spawn"

    # _load_async() is called as argument to create_task, not directly awaited
    load_async_refs = [r for r in open_refs if r["callee_name"] == "_load_async"]
    assert len(load_async_refs) == 1
    assert load_async_refs[0]["await_kind"] == "sync"

    init_ui_refs = [r for r in open_refs if r["callee_name"] == "_init_ui"]
    assert len(init_ui_refs) == 1
    assert init_ui_refs[0]["await_kind"] == "sync"


def test_extract_cross_refs_has_seq():
    _, refs, _ = extract_graph("tests/fixtures/callflow/stage_manager.py", STAGE_MANAGER)

    for r in refs:
        if r["ref_type"] == "call" and r["caller_name"].startswith("module."):
            assert r["seq"] is not None, f"Missing seq for {r}"
            assert isinstance(r["seq"], int)
            assert r["seq"] > 0


def test_extract_cross_refs_guard():
    content = """
def foo():
    if condition:
        bar()
    while x:
        baz()
"""
    _, refs, _ = extract_graph("test.py", content)
    calls = [r for r in refs if r["ref_type"] == "call"]
    bar_ref = next(r for r in calls if r["callee_name"] == "bar")
    assert bar_ref["guard"] == "condition"
    baz_ref = next(r for r in calls if r["callee_name"] == "baz")
    assert baz_ref["guard"] == "x"


def test_extract_cross_refs_receiver():
    content = """
class Foo:
    def method(self):
        self.bar()
        other.baz()
"""
    _, refs, _ = extract_graph("test.py", content)
    calls = [r for r in refs if r["ref_type"] == "call"]
    bar_ref = next(r for r in calls if r["callee_name"] == "bar")
    assert bar_ref["receiver"] == "self.bar"
    baz_ref = next(r for r in calls if r["callee_name"] == "baz")
    assert baz_ref["receiver"] == "other.baz"