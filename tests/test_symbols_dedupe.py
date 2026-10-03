"""One symbol per definition with the right kind; unique index as a safety net."""
from ai_db import VectorDB
from ai_db.parser.ts_graph import extract_graph

SRC = (
    "def top(): pass\n"
    "class A:\n"
    "    def m(self):\n"
    "        def inner(): pass\n"
    "    @staticmethod\n"
    "    def s(): pass\n"
    "    async def am(self): pass\n"
)


def test_python_one_symbol_per_def_with_correct_kind():
    symbols, _, _ = extract_graph("a.py", SRC)
    got = sorted((s["line"], s["name"], s["symbol_type"]) for s in symbols)
    assert got == [
        (1, "top", "function"), (2, "A", "class"), (3, "m", "method"),
        (4, "inner", "function"), (6, "s", "method"), (7, "am", "method"),
    ]


def test_indexed_symbols_have_no_duplicates(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.py").write_text(SRC)
    db = VectorDB(str(tmp_path / "db.sqlite"))
    db.sync(str(src), project="p", verbose=False)
    n, distinct = db.backend.conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT filepath || ':' || line || ':' || name) FROM symbols").fetchone()
    assert n == distinct == 6
    assert [r["symbol_type"] for r in db.query_symbol("top", project="p")] == ["function"]
    db.close()


def test_duplicate_symbol_insert_is_ignored(tmp_path):
    from ai_db.storage import SymbolRecord
    db = VectorDB(str(tmp_path / "db.sqlite"))
    db.backend.conn.execute("PRAGMA foreign_keys = OFF")
    db.backend.insert_symbols([SymbolRecord("f", "function", "/r/a.py", 1, "def f()", "p"),
                               SymbolRecord("f", "method", "/r/a.py", 1, "def f()", "p")])
    assert db.backend.conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0] == 1
    db.close()
