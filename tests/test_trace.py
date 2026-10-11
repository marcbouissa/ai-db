"""Call-flow trace: entry resolution, call order and lines, async markers, link precision."""

import subprocess
import sys

import pytest

from ai_db import VectorDB
from ai_db.dispatcher import ServiceDispatcher
from ai_db.parser.ts_graph import extract_graph

APP = '''\
import asyncio


class Store:
    def save(self, item):
        return item


class Service:
    def __init__(self, store: Store):
        self.store = store

    async def run(self, items):
        prepare()
        for item in items:
            self.store.save(item)
        await notify()
        _VAR.reset(1)
        return finish()


def prepare():
    return 1


async def notify():
    return 2


def finish():
    return 3


class Unrelated:
    def reset(self, x):
        return x
'''


@pytest.fixture
def db(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "app.py").write_text(APP)
    vdb = VectorDB(str(tmp_path / "t.db"))
    vdb.sync(str(repo), project="proj", verbose=False)
    yield vdb, repo
    vdb.close()


def _trace(db, entry, cwd, **kw):
    disp = ServiceDispatcher(db=db)
    return disp.execute("trace", {"entry": entry, "format": "json", "depth": 1,
                                  "cwd": str(cwd), **kw})


def _children(out):
    import json
    root = json.loads(out)["root"]
    return [(c["qualified_name"], c.get("call_line"), c.get("await_kind")) for c in root["children"]]


def test_qualified_entry_resolves_and_calls_come_in_source_order(db):
    vdb, repo = db
    kids = _children(_trace(vdb, "Service.run", repo))
    assert [k[0] for k in kids] == ["prepare", "Store.save", "notify", "finish"]
    assert [k[1] for k in kids] == [14, 16, 17, 19]  # the call sites, not the def line (13)
    assert {k[0]: k[2] for k in kids}["notify"] == "await"


def test_unknown_receiver_does_not_link_to_an_unrelated_method(db):
    vdb, repo = db
    names = [k[0] for k in _children(_trace(vdb, "Service.run", repo))]
    assert "Unrelated.reset" not in names  # `_VAR.reset()` is not Unrelated.reset


def test_default_scope_is_the_callers_project(db):
    vdb, repo = db
    assert "Entry point not found" not in _trace(vdb, "finish", repo)


def test_file_line_picks_the_innermost_definition(db):
    vdb, repo = db
    import json
    inside_method = json.loads(_trace(vdb, "app.py:16", repo))["root"]["qualified_name"]
    assert inside_method == "Service.run"  # not the class header that also spans line 16
    blank_above = json.loads(_trace(vdb, "app.py:21", repo))["root"]["qualified_name"]
    assert blank_above == "prepare"  # a blank line just above a def snaps to it


def test_extract_graph_is_deterministic():
    first = extract_graph("app.py", APP)[1]
    second = extract_graph("app.py", APP)[1]
    assert first == second  # `seq` came from a never-reset module-global counter


def test_cli_import_stays_light():
    """Every command imports ai_db.cli; the daemon (fastapi), httpx and numpy are only
    needed by a few, and used to cost ~400 ms at every start."""
    code = ("import sys, ai_db.cli\n"
            "print([m for m in ('fastapi', 'httpx', 'numpy', 'ai_db.daemon.server') "
            "if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "[]"


@pytest.mark.parametrize("line,expected", [
    ("# TODO: fix the cache", ("todo", "fix the cache")),
    ("x = 1  # FIXME later", ("fixme", "later")),
    ("// HACK: until v2", ("hack", "until v2")),
    ("# todo: lowercase with a colon", ("todo", "lowercase with a colon")),
    ("# todos", None),
    ("# note that this is prose", None),
    ("# Notes on the design", None),
])
def test_annotation_tags_are_real_tags(line, expected):
    from ai_db.parser.annotations import extract_annotations

    hits = [(h["kind"], h["content"]) for h in extract_annotations("m.js", line)]
    assert hits == ([expected] if expected else [])
