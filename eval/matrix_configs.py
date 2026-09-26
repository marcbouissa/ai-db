"""Generate the config matrix used by eval_matrix.py.

One JSON config per retrieval condition, all sharing the same storage options
shape so only the retrieval-relevant keys differ. Written to a directory so a
run is reproducible and the exact configs are inspectable.
"""
from __future__ import annotations

import json
import os
import sys

EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
QUERY_PROMPT = "Instruct: Given a code search query, retrieve relevant code\nQuery: "

# name -> (retrieval mode, embedding provider, rerank provider, vector_index, device, dtype)
#
# `dtype` is only meaningful with a real embedding provider; it is carried here
# so a single generator covers the whole config-mode matrix. `auto` leaves the
# choice to the timing probe in `ai_db.device.resolve_dtype`.
CONDITIONS = {
    "lexical":            ("lexical", "none",                    "none",    "exact", None, None),
    "lexical_vec0":       ("lexical", "none",                    "none",    "vec0",  None, None),
    "hybrid_cuda":        ("hybrid", "sentence_transformers",   "none",    "exact", "cuda", None),
    "hybrid_cuda_vec0":   ("hybrid", "sentence_transformers",   "none",    "vec0",  "cuda", None),
    "hybrid_cuda_rerank": ("hybrid", "sentence_transformers",   "sentence_transformers", "exact", "cuda", None),
    "hybrid_cpu":         ("hybrid", "sentence_transformers",   "none",    "exact", "cpu", None),
    # dtype sweep: same model and device, only the tensor dtype differs.
    "hybrid_cuda_fp32":   ("hybrid", "sentence_transformers",   "none",    "exact", "cuda", "float32"),
    "hybrid_cuda_bf16":   ("hybrid", "sentence_transformers",   "none",    "exact", "cuda", "bfloat16"),
    "hybrid_cuda_fp16":   ("hybrid", "sentence_transformers",   "none",    "exact", "cuda", "float16"),
}


def build(name: str, db_path: str) -> dict:
    mode, embed, rerank, vindex, device, dtype = CONDITIONS[name]
    embedding: dict = {"provider": embed}
    if embed != "none":
        embedding.update({
            "model": EMBED_MODEL,
            "device": device,
            "batch_size": 32 if device == "cuda" else 8,
            "query_prompt": QUERY_PROMPT,
        })
        if dtype is not None:
            embedding["dtype"] = dtype
    # Provider options are siblings of "provider", as in the `embedding`
    # section. `top_n` is a sibling too, not an option.
    rerank_cfg: dict = {"provider": rerank}
    if rerank != "none":
        rerank_cfg.update({"model": RERANK_MODEL, "device": device or "cpu",
                           "top_n": 10})
    return {
        "version": 2,
        "storage": {"provider": "sqlite",
                    "options": {"path": db_path, "vector_index": vindex}},
        "retrieval": {"mode": mode},
        "embedding": embedding,
        "rerank": rerank_cfg,
        "access": {"cross_project": {}},
        "auto_sync_paths": [],
        "trace": {"wait_patterns": None, "wait_patterns_extend": None},
    }


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/matrix"
    db_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/matrix/db"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    index = {}
    for name in list(CONDITIONS):
        db_path = os.path.join(db_dir, f"{name}.db")
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(build(name, db_path), fh, indent=2)
        mode, embed, rerank, vindex, device, dtype = CONDITIONS[name]
        index[name] = {"config": path, "db": db_path, "retrieval_mode": mode,
                       "vector_index": vindex, "device": device, "dtype": dtype,
                       "rerank": rerank, "embedding": embed}
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    print(json.dumps(index, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
