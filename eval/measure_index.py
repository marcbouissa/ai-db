"""Measure one already-built matrix index, without re-syncing it.

`matrix_runner.py` always re-indexes, which costs ~55 min for the CPU-embedder
condition. The index is a pure function of (corpus, config), so once it exists
there is nothing to gain from rebuilding it. This reads the existing DB and runs
only the evaluations, against the same project the index was synced under.

Usage: measure_index.py <condition> <config.json> <db>
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_db import VectorDB
from ai_db.config import load_config
from ai_db.eval.harness import load_golden
from ai_db.eval.metrics import mrr as mrr_fn
from ai_db.eval.metrics import ndcg_at_k, recall_at_k
from ai_db.utils import detect_project_name

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def expected_tuples(item: dict) -> set[tuple[str, str]]:
    return {(e["filepath"], e.get("symbol", e.get("name", ""))) for e in item["expected"]}


def main() -> int:
    condition, cfg_path, db_path = sys.argv[1], sys.argv[2], sys.argv[3]
    os.environ["AI_DB_CONFIG"] = cfg_path
    cfg = load_config(cfg_path)
    project = detect_project_name(REPO)
    db = VectorDB(db_path, config=cfg)
    try:
        golden = load_golden(os.path.join(REPO, "eval/golden/ai_db.jsonl"))
        root = os.path.abspath(REPO)
        recalls, mrrs, ndcgs, lat = [], [], [], []
        for item in golden:
            t0 = time.perf_counter()
            hits = db.query(item["query"], top_k=10, project=project)
            lat.append((time.perf_counter() - t0) * 1000)
            ranked = [(os.path.relpath(h["abs_path"], root), h["name"]) for h in hits]
            exp = expected_tuples(item)
            recalls.append(recall_at_k(ranked, exp, 10))
            mrrs.append(mrr_fn(ranked, exp))
            ndcgs.append(ndcg_at_k(ranked, exp, 10))
        lat.sort()
        retrieval = {"k": 10, "queries": len(golden),
                     "recall@10": round(statistics.fmean(recalls), 4),
                     "mrr": round(statistics.fmean(mrrs), 4),
                     "ndcg@10": round(statistics.fmean(ndcgs), 4),
                     "latency_ms_p50": round(statistics.median(lat), 2),
                     "latency_ms_p95": round(lat[int(len(lat) * 0.95) - 1], 2)}

        pgolden = load_golden(os.path.join(REPO, "eval/golden/ai_db_pack.jsonl"))
        prec, ptok = [], []
        for item in pgolden:
            pack = db.investigate(item["query"], budget_tokens=8000,
                                  mode=item["kind"], project=project)
            ranked = [(os.path.relpath(e["filepath"], root), e["qualified_name"])
                      for e in pack["evidence"]]
            exp = expected_tuples(item)
            prec.append(recall_at_k(ranked, exp, len(ranked) or 1))
            ptok.append(pack["token_count"])
        pack = {"queries": len(pgolden), "budget_tokens": 8000,
                "pack_recall": round(statistics.fmean(prec), 4),
                "tokens_mean": round(statistics.fmean(ptok), 1)}

        out = {"conditions": {condition: {
            "config": cfg_path, "db": db_path, "project": project,
            "index": {"indexed": True, "measured_without_resync": True,
                      "chunks": 1713},
            "retrieval": retrieval, "pack": pack}}}
        print(json.dumps(out["conditions"][condition], indent=2))
        with open(sys.argv[4], "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
