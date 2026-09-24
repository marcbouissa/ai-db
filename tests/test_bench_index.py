"""Indexing throughput benchmark. Run with: pytest -m bench tests/test_bench_index.py -s"""

import os
import shutil
import time

import pytest

from ai_db import VectorDB
from ai_db.search import indexer as indexer_mod

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _corpus(tmp_path, copies=5):
    root = tmp_path / "corpus"
    for i in range(copies):
        shutil.copytree(os.path.join(REPO, "ai_db"), root / f"c{i}" / "ai_db",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(os.path.join(REPO, "tests"), root / f"c{i}" / "tests",
                        ignore=shutil.ignore_patterns("__pycache__"))
    return str(root)


def _timed_sync(db_path, root):
    db = VectorDB(db_path)
    t0 = time.perf_counter()
    res = db.sync(root, verbose=False)
    elapsed = time.perf_counter() - t0
    db.close()
    return res, elapsed


@pytest.mark.bench
def test_parallel_sync_beats_serial(tmp_path, monkeypatch):
    root = _corpus(tmp_path)
    monkeypatch.setattr(indexer_mod, "PARALLEL_PARSE_MIN_FILES", 10**9)
    serial_res, serial_s = _timed_sync(str(tmp_path / "serial.db"), root)
    monkeypatch.setattr(indexer_mod, "PARALLEL_PARSE_MIN_FILES", 32)
    par_res, par_s = _timed_sync(str(tmp_path / "par.db"), root)
    print(f"\nfiles={par_res['added']} serial={serial_s:.2f}s parallel={par_s:.2f}s "
          f"speedup={serial_s / par_s:.1f}x")
    assert serial_res == par_res
    if (os.cpu_count() or 1) >= 4:
        assert serial_s / par_s >= 1.5
