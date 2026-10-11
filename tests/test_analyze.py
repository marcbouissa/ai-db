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


def test_text_files_yield_symbols(vdb, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("first line\nsecond line\n")
    res = vdb.analyze_file(str(path), depth="full", bypass_cache=True)
    assert [s["span"] for s in res["symbols"]] == [[1, 2]]
    assert "second line" in res["symbols"][0]["body"]
