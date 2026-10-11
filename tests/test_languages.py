"""index.languages: pick which tree-sitter grammars are used, per language."""

import json

import pytest

from ai_db import VectorDB
from ai_db.config import parse_config
from ai_db.config_template import build_template
from ai_db.constants import PARALLEL_PARSE_MIN_FILES
from ai_db.errors import AiDbConfigError
from ai_db.parser import linters
from ai_db.parser.chunker import chunk_file
from ai_db.parser.languages import SUPPORTED_LANGUAGES, enabled_language_for
from ai_db.parser.linters import parse_source

PY = "def alpha():\n    return beta()\n\n\ndef beta():\n    return 1\n"
GO = "package main\n\nfunc gamma() int {\n\treturn 1\n}\n"


def _config(tmp_path, languages):
    data = build_template(languages=languages)
    data["storage"]["options"]["path"] = str(tmp_path / "x.db")
    return parse_config(data, check_env=False)


def test_default_is_every_supported_language():
    assert parse_config(build_template(), check_env=False).index.languages == SUPPORTED_LANGUAGES
    assert build_template()["index"]["languages"] == sorted(SUPPORTED_LANGUAGES)


def test_unknown_language_is_rejected():
    with pytest.raises(AiDbConfigError, match="cobol"):
        build_template(languages=["python", "cobol"])


def test_enabled_language_for():
    assert enabled_language_for("a.py", {"python"}) == "python"
    assert enabled_language_for("a.go", {"python"}) is None
    assert enabled_language_for("a.txt", SUPPORTED_LANGUAGES) is None


def test_deselected_language_is_chunked_as_text():
    assert {c["chunk_type"] for c in chunk_file("m.py", PY, {"python"})} >= {"code"}
    assert {c["chunk_type"] for c in chunk_file("m.py", PY, {"go"})} == {"txt"}


def test_deselected_language_is_not_parsed_or_linted(monkeypatch):
    monkeypatch.setattr(linters, "get_linter", lambda *a: pytest.fail("linter must not run"))
    assert parse_source("m.go", GO, {"python"}) == ([], [], None)
    symbols, _, _ = parse_source("m.go", GO, {"go"})
    assert [s["name"] for s in symbols] == ["gamma"]


def _names(db):
    rows = db.backend.conn.execute("SELECT name FROM symbols").fetchall()
    return {r[0] for r in rows}


def test_toggling_a_language_reparses_its_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text(PY)
    (repo / "b.go").write_text(GO)
    db = VectorDB(str(tmp_path / "x.db"), config=_config(tmp_path, ["python"]))
    db.sync(str(repo), project="p", verbose=False)
    assert _names(db) == {"alpha", "beta"}
    db.close()

    db = VectorDB(str(tmp_path / "x.db"), config=_config(tmp_path, ["python", "go"]))
    result = db.sync(str(repo), project="p", verbose=False)
    assert result["updated"] == 1  # only b.go: its grammar was turned on
    assert _names(db) == {"alpha", "beta", "gamma"}
    db.close()

    db = VectorDB(str(tmp_path / "x.db"), config=_config(tmp_path, ["python"]))
    db.sync(str(repo), project="p", verbose=False)
    assert _names(db) == {"alpha", "beta"}
    assert db.backend.get_file(str(repo / "b.go")) is not None  # still indexed, as text
    db.close()


def test_parallel_parse_path_honours_the_selection(tmp_path):
    repo = tmp_path / "many"
    repo.mkdir()
    for i in range(PARALLEL_PARSE_MIN_FILES + 2):
        (repo / f"m{i}.py").write_text(f"def fn_{i}():\n    return {i}\n")
    db = VectorDB(str(tmp_path / "x.db"), config=_config(tmp_path, ["go"]))
    db.sync(str(repo), project="p", verbose=False)
    assert _names(db) == set()
    db.close()


def test_grammars_command_lists_selection(tmp_path, monkeypatch, capsys):
    from ai_db import cli

    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps(build_template(languages=["rust"])))
    assert cli.main(["--config", str(cfg), "grammars"]) == 0
    out = capsys.readouterr().out
    rust = next(line for line in out.splitlines() if line.startswith("rust"))
    python = next(line for line in out.splitlines() if line.startswith("python"))
    assert rust.split()[1] == "yes" and python.split()[1] == "no"
