"""K3: package inits export lazily, so importing one module imports little else."""

import subprocess
import sys

import pytest


def _loaded_after(statement: str) -> set[str]:
    code = (f"import sys\n{statement}\n"
            "print('\\n'.join(sorted(m for m in sys.modules if m.startswith('ai_db'))))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def test_leaf_modules_import_only_themselves():
    assert _loaded_after("import ai_db.constants") == {"ai_db", "ai_db._lazy", "ai_db.constants"}
    assert _loaded_after("import ai_db.errors") == {"ai_db", "ai_db._lazy", "ai_db.errors"}


def test_cli_does_not_import_the_facade_stack():
    loaded = _loaded_after("import ai_db.cli")
    for heavy in ("ai_db.facade", "ai_db.storage.sqlite_backend", "ai_db.search.indexer",
                  "ai_db.analyzer.engine", "ai_db.parser.ts_graph"):
        assert heavy not in loaded, heavy


@pytest.mark.parametrize("package", ["ai_db", "ai_db.parser", "ai_db.search", "ai_db.storage",
                                     "ai_db.analyzer", "ai_db.analysis"])
def test_every_listed_export_resolves(package):
    import importlib

    mod = importlib.import_module(package)
    for name in mod.__all__:
        assert getattr(mod, name) is not None, f"{package}.{name}"
        assert name in dir(mod)
    with pytest.raises(AttributeError):
        _ = mod.definitely_not_exported


def test_public_imports_still_work():
    from ai_db import VectorDB, load_config
    from ai_db.storage import SQLiteBackend

    assert VectorDB.__module__ == "ai_db.facade"
    assert callable(load_config) and SQLiteBackend.__name__ == "SQLiteBackend"
