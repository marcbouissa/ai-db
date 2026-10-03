"""
tests/test_callers.py
`callers`: real call-site lines, receiver-aware qualified queries, no duplicates,
reported truncation.
"""
import pytest

from ai_db import VectorDB
from ai_db.parser.ts_graph import extract_graph

SRC = '''\
class A:
    def execute(self, q):
        return q

    def run(self):
        self.execute(1)          # line 6: A.execute


class B:
    def execute(self, q):
        return q

    def run(self, a: A):
        self.execute(2)          # line 14: B.execute
        a.execute(3)             # line 15: A.execute (annotated param)


def use_cursor(cur):
    cur.execute("select 1")      # line 19: unresolved


def main():
    b = B()
    b.execute(4)                 # line 24: B.execute (local ctor)
    A.execute(None, 5)           # line 25: A.execute (class receiver)
    cur = None
    cur.execute("x")             # line 27: unresolved
'''


def _calls(refs, name):
    return [r for r in refs if r["ref_type"] == "call" and r["callee_name"] == name]


def test_extract_records_call_site_line_and_resolution():
    _, refs, _ = extract_graph("m.py", SRC)
    got = {(r["call_line"], r["callee_resolved"]) for r in _calls(refs, "execute")}
    assert got == {
        (6, "A.execute"), (14, "B.execute"), (15, "A.execute"), (19, None),
        (24, "B.execute"), (25, "A.execute"), (27, None),
    }
    # caller_line is still the enclosing def, call_line is the call site
    r6 = next(r for r in _calls(refs, "execute") if r["call_line"] == 6)
    assert r6["caller_line"] == 5 and r6["callee_qualifier"] == "self"


@pytest.fixture
def vdb(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text(SRC)
    db = VectorDB(str(tmp_path / "db.sqlite"))
    db.sync(str(src), project="p", verbose=False)
    yield db


def test_qualified_query_returns_only_that_class_with_real_lines(vdb):
    out = vdb.query_callers("A.execute", project="p")
    assert sorted(r["line"] for r in out["results"]) == [6, 15, 25]
    assert out["total"] == 3 and out["returned"] == 3 and out["truncated"] is False
    assert all(r["resolved"] == "A.execute" for r in out["results"])
    out_b = vdb.query_callers("B.execute", project="p")
    assert sorted(r["line"] for r in out_b["results"]) == [14, 24]


def test_each_call_site_appears_exactly_once(vdb):
    out = vdb.query_callers("execute", project="p")
    sites = [(r["file"], r["line"], r["col"]) for r in out["results"]]
    assert len(sites) == len(set(sites)) == 7


def test_bare_query_groups_by_resolved_with_unresolved_bucket(vdb):
    out = vdb.query_callers("execute", project="p")
    assert out["groups"] == {"A.execute": 3, "B.execute": 2, "unresolved": 2}


def test_truncation_is_reported(vdb):
    out = vdb.query_callers("execute", project="p", top_k=3)
    assert out["total"] == 7 and out["returned"] == 3 and out["truncated"] is True


def test_cli_prints_truncation_and_limit(vdb, tmp_path, capsys):
    from ai_db.cli import main
    path = str(tmp_path / "db.sqlite")
    vdb.close()
    assert main(["callers", "execute", "--limit", "3", "--db", path, "--project", "p"]) == 0
    out = capsys.readouterr().out
    assert "... 4 more (use --limit)" in out
    assert "# receivers: A.execute (3), B.execute (2), unresolved (2)" in out
