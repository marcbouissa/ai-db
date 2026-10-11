"""`analyze`: full spans from the chunker (G6)."""
import pytest

from ai_db import VectorDB

CODE = '''\
import os


class Store:
    def save(self, item):
        a = 1
        b = 2
        return a + b


def helper(x):
    y = x + 1
    z = y * 2
    return z
'''


@pytest.fixture
def vdb(tmp_path):
    db = VectorDB(str(tmp_path / "t.db"))
    yield db
    db.close()


def test_full_depth_returns_the_whole_function(vdb, tmp_path):
    path = tmp_path / "m.py"
    path.write_text(CODE)
    res = vdb.analyze_file(str(path), depth="full", bypass_cache=True)
    syms = {s["name"]: s for s in res["symbols"]}
    assert syms["helper"]["span"] == [11, 14]
    assert syms["helper"]["body"].splitlines() == CODE.splitlines()[10:14]  # all 4 lines
    assert syms["helper"]["sig"] == "def helper(x):"
    assert syms["Store.save"]["span"] == [5, 8]
    assert syms["Store"]["span"] == [4, 8]
    assert syms["Store"]["kind"] == "class"


def test_ref_ids_are_48_bits_and_a_collision_raises(vdb):
    """G9: a forced id collision must raise instead of overwriting another body."""
    import time

    from ai_db.errors import AiDbStorageError
    from ai_db.storage.models import AnalysisRefRecord

    store = vdb.analyzer_engine.ref_store
    ref_id = store.store_ref("/x/a.py", "f", 1, 2, "function", "def f():\n    pass")
    assert len(ref_id) == len("ref:") + 12
    # storing the same body again is idempotent
    assert store.store_ref("/x/a.py", "f", 1, 2, "function", "def f():\n    pass") == ref_id
    other = AnalysisRefRecord(ref_id=ref_id, filepath="/x/b.py", name="g", start_line=5,
                              end_line=6, kind="function", body_text="def g(): ...",
                              timestamp=time.time())
    with pytest.raises(AiDbStorageError, match="collision"):
        vdb.backend.store_analysis_ref(other)
    assert vdb.backend.get_analysis_ref(ref_id).body_text == "def f():\n    pass"


def test_text_files_yield_symbols(vdb, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("first line\nsecond line\n")
    res = vdb.analyze_file(str(path), depth="full", bypass_cache=True)
    assert [s["span"] for s in res["symbols"]] == [[1, 2]]
    assert "second line" in res["symbols"][0]["body"]
