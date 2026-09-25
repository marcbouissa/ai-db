import glob
import hashlib
import json
import os
import time
from typing import Any

from ai_db.analyzer.references import ReferenceStore
from ai_db.parser.ts_graph import extract_graph, language_for
from ai_db.utils import compute_sha256, tokenize


class AnalyzerEngine:
    def __init__(self, db: Any, query_engine: Any = None):
        self.db = db
        self.backend = getattr(db, "backend", getattr(db, "db", db))
        # Injected by VectorDB so locate shares the configured retriever/ranker
        # (lexical or hybrid) instead of reaching into the backend directly.
        self.query_engine = query_engine
        self.ref_store = ReferenceStore(db=self.backend)
        self._evict_stale_refs()

    def _evict_stale_refs(self):
        """Evicts analysis refs older than 7 days to prevent unbounded table growth."""
        self.backend.evict_stale_analysis_refs(older_than_seconds=86400 * 7)

    def _store_analysis_ref(self, filepath: str, name: str, start_line: int, end_line: int, kind: str, body_text: str) -> str:
        return self.ref_store._store_analysis_ref(filepath, name, start_line, end_line, kind, body_text)

    def expand_ref(self, ref_id: str, depth: str = "full", span: tuple[int, int] | None = None) -> dict[str, Any] | None:
        return self.ref_store.expand_ref(ref_id, depth, span)

    def _diff_spans(self, filepath: str, current_content: str, since: str | None) -> dict[str, Any]:
        return self.ref_store._diff_spans(filepath, current_content, since)

    def get_session_state(self, key: str) -> Any | None:
        if hasattr(self.backend, "get_state"):
            return self.backend.get_state(key)
        if hasattr(self.db, "get_session_state"):
            return self.db.get_session_state(key)
        return None

    def set_session_state(self, key: str, value: Any):
        if hasattr(self.backend, "set_state"):
            return self.backend.set_state(key, value)
        if hasattr(self.db, "set_session_state"):
            return self.db.set_session_state(key, value)

    def analyze_file(self, filepath: str, depth: str = "structure",
                     span: tuple[int, int] | None = None,
                     focus: str | None = None,
                     q: str | None = None,
                     since: str | None = None,
                     ctx_lines: int = 10,
                     bypass_cache: bool = False,
                     no_cache: bool = False) -> dict[str, Any]:
        """Analyzes a single file according to RFC tokenopt-analyzer v2."""
        bypass_cache = bypass_cache or no_cache
        abs_path = os.path.abspath(os.path.expanduser(filepath))
        if not os.path.exists(abs_path):
            return {
                "file": filepath,
                "error": f"File not found: {filepath}",
                "symbols": [],
                "notes": ["File not found"],
                "meta": {"tokens_in": 0, "tokens_out": 0, "cached": False, "truncated": False, "conf": 0.0}
            }

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:  # reported to the caller as a per-file error
            return {
                "file": filepath,
                "error": str(e),
                "symbols": [],
                "notes": [f"Error reading file: {e}"],
                "meta": {"tokens_in": 0, "tokens_out": 0, "cached": False, "truncated": False, "conf": 0.0}
            }

        file_hash = compute_sha256(abs_path)
        tokens_in = max(1, len(content) // 4)
        all_lines = content.splitlines()
        total_lines = len(all_lines)

        # F9 Semantic Cache check
        norm_q = (q or "").strip().lower()
        cache_key_raw = f"{abs_path}:{file_hash}:{depth}:{span}:{focus}:{norm_q}:{since}:{ctx_lines}"
        cache_key = hashlib.sha256(cache_key_raw.encode("utf-8")).hexdigest()

        if not bypass_cache:
            cached_res = self.backend.get_semantic_cache(cache_key, file_hash)
            if cached_res is not None:
                if isinstance(cached_res, dict) and "meta" in cached_res:
                    cached_res["meta"]["cached"] = True
                return dict(cached_res)

        # F8 Diff Mode check
        diff_info = self._diff_spans(abs_path, content, since) if since else None

        # F2 Range Target extraction
        if span:
            start_req, end_req = span
            start_bounded = max(1, start_req - ctx_lines)
            end_bounded = min(total_lines, end_req + ctx_lines)
            snippet_lines = all_lines[start_bounded - 1 : end_bounded]
            snippet_text = "\n".join(snippet_lines)
            ref_id = self._store_analysis_ref(abs_path, f"span:{start_bounded}-{end_bounded}", start_bounded, end_bounded, "range", snippet_text)
            symbols = [{
                "name": f"L{start_bounded}-{end_bounded}",
                "kind": "range",
                "span": [start_bounded, end_bounded],
                "ref": ref_id,
                "body": snippet_text
            }]
            tokens_out = max(1, len(snippet_text) // 4)
            res = {
                "file": filepath,
                "symbols": symbols,
                "diff": diff_info,
                "notes": [f"Extracted span L{start_bounded}-{end_bounded} (ctx={ctx_lines})"],
                "meta": {"tokens_in": tokens_in, "tokens_out": tokens_out, "cached": False, "truncated": False, "conf": 0.99}
            }
            self._save_to_semantic_cache(cache_key, file_hash, res)
            return res

        # Extract symbols using tree-sitter for supported languages
        symbols_found: list[dict[str, Any]] = []
        notes = []
        q_tokens = set(tokenize(q or "")) if q else set()

        # Get symbols from tree-sitter for supported languages
        lang = language_for(abs_path)
        if lang is not None:
            ts_symbols, _, _ = extract_graph(abs_path, content)
            for sym in ts_symbols:
                sym_name = sym["name"]
                sym_type = sym["symbol_type"]
                sym_line = sym["line"]
                signature = sym.get("signature")
                # Find end line from chunks
                end_line = sym_line
                for chunk in self.db.get_chunks_for_file(abs_path):
                    if chunk.name == sym_name and chunk.start_line == sym_line:
                        end_line = chunk.end_line
                        break
                body_block = "\n".join(all_lines[sym_line - 1 : end_line])
                ref_id = self._store_analysis_ref(abs_path, sym_name, sym_line, end_line, sym_type, body_block)
                symbols_found.append({
                    "name": sym_name,
                    "kind": sym_type,
                    "sig": signature or sym_name,
                    "span": [sym_line, end_line],
                    "ref": ref_id,
                    "body": body_block
                })
        else:
            # For unsupported languages (markdown, text), build symbols from chunks
            chunks = self.db.get_chunks_for_file(abs_path)
            for chunk in chunks:
                if chunk.chunk_type in ("md", "section", "text", "lib_meta"):
                    ref_id = self._store_analysis_ref(abs_path, chunk.name, chunk.start_line, chunk.end_line, chunk.chunk_type, chunk.content)
                    symbols_found.append({
                        "name": chunk.name,
                        "kind": chunk.chunk_type,
                        "sig": chunk.qualified_name,
                        "span": [chunk.start_line, chunk.end_line],
                        "ref": ref_id,
                        "body": chunk.content
                    })

        # F1 Depth Control & F3 Question-Driven Filtering
        filtered_symbols = []
        conf = 0.95

        for sym in symbols_found:
            is_match = True
            if q_tokens:
                combined_text = f"{sym['name']} {sym.get('sig', '')} {sym.get('body', '')}".lower()
                matched_q = any(tok in combined_text for tok in q_tokens if len(tok) > 2)
                if not matched_q:
                    is_match = False

            if focus and focus.lower() not in sym["name"].lower():
                is_match = False

            item: dict[str, Any] = {
                "name": sym["name"],
                "kind": sym["kind"],
                "sig": sym.get("sig", sym["name"]),
                "span": sym["span"],
                "ref": sym["ref"]
            }

            if depth == "summary":
                # Signatures only, strictly no body
                pass
            elif depth == "structure":
                # Structure: signature + submethods if any
                if "methods" in sym:
                    item["methods"] = [{
                        "name": mt["name"],
                        "kind": mt["kind"],
                        "sig": mt["sig"],
                        "span": mt["span"],
                        "ref": mt["ref"]
                    } for mt in sym["methods"]]
            elif depth == "targeted":
                # Bodies included only if matched filter/question
                if is_match:
                    item["body"] = sym.get("body", "")
                if "methods" in sym:
                    item["methods"] = []
                    for mt in sym["methods"]:
                        m_item: dict[str, Any] = {
                            "name": mt["name"],
                            "kind": mt["kind"],
                            "sig": mt["sig"],
                            "span": mt["span"],
                            "ref": mt["ref"]
                        }
                        if q_tokens and any(tok in f"{mt['name']} {mt['sig']} {mt['body']}".lower() for tok in q_tokens if len(tok) > 2):
                            m_item["body"] = mt["body"]
                        item["methods"].append(m_item)
            elif depth == "full":
                item["body"] = sym.get("body", "")
                if "methods" in sym:
                    item["methods"] = sym["methods"]

            if q_tokens or focus:
                if is_match or (depth != "targeted" and depth != "summary"):
                    filtered_symbols.append(item)
            else:
                filtered_symbols.append(item)

        if not filtered_symbols and (q_tokens or focus):
            conf = 0.4
            notes.append("No symbols matched filter; suggest depth=full or widening query")

        # Compute token estimates
        rendered_json = json.dumps(filtered_symbols)
        tokens_out = max(1, len(rendered_json) // 4)

        result = {
            "file": filepath,
            "symbols": filtered_symbols,
            "diff": diff_info,
            "notes": notes,
            "meta": {
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cached": False,
                "truncated": False,
                "conf": conf
            }
        }
        self._save_to_semantic_cache(cache_key, file_hash, result)
        return result

    def _save_to_semantic_cache(self, cache_key: str, file_hash: str, result: dict[str, Any]):
        """Saves result to semantic cache table."""
        self.backend.set_semantic_cache(cache_key, file_hash, result)

    def analyze_batch(self, targets: list[str], depth: str = "structure",
                      q: str | None = None, focus: str | None = None,
                      span: tuple[int, int] | None = None,
                      since: str | None = None,
                      max_out: int | None = None,
                      cursor: str | None = None,
                      ctx_lines: int = 10) -> dict[str, Any]:
        """F4 Batching + F5 Token Budgeting: Analyzes multiple targets up to n<=32 with cursor continuation."""
        expanded_targets = []
        for t in targets:
            if any(char in t for char in ["*", "?", "["]):
                matched = glob.glob(os.path.expanduser(t), recursive=True)
                expanded_targets.extend([m for m in matched if os.path.isfile(m)])
            else:
                expanded_targets.append(t)

        expanded_targets = list(dict.fromkeys(expanded_targets))[:32]  # Cap n<=32

        # F13 Session State: Save targets and query
        self.set_session_state("last_analysis", {
            "targets": expanded_targets,
            "depth": depth,
            "q": q,
            "focus": focus,
            "since": since
        })

        results: dict[str, Any] = {}
        total_tokens_out = 0
        total_tokens_in = 0
        is_truncated = False
        next_cursor = None

        # Check continuation cursor
        start_index = 0
        if cursor:
            cur_data = self.get_session_state(f"cursor:{cursor}")
            if cur_data:
                start_index = cur_data.get("next_index", 0)

        for idx in range(start_index, len(expanded_targets)):
            target = expanded_targets[idx]
            file_res = self.analyze_file(target, depth=depth, span=span, focus=focus, q=q, since=since, ctx_lines=ctx_lines)
            tokens_this = file_res["meta"]["tokens_out"]
            total_tokens_in += file_res["meta"]["tokens_in"]

            # F5 Token Budget Enforcement
            if max_out and (total_tokens_out + tokens_this > max_out) and results:
                is_truncated = True
                cursor_id = f"cur_{int(time.time()*1000)}"
                self.set_session_state(f"cursor:{cursor_id}", {"next_index": idx, "targets": expanded_targets})
                next_cursor = cursor_id
                break

            results[target] = file_res
            total_tokens_out += tokens_this

        return {
            "results": results,
            "meta": {
                "tokens_in": total_tokens_in,
                "tokens_out": total_tokens_out,
                "truncated": is_truncated,
                "cursor": next_cursor,
                "targets_count": len(results)
            }
        }

    def locate_targets(self, q: str, scope: str = ".", k: int = 5,
                       path_prefix: str | None = None) -> list[dict[str, Any]]:
        """F10 Relevance Rank: Finds top-k matching files/snippets without dumping entire directory scans."""
        if not tokenize(q):
            return []

        scope_abs = os.path.abspath(os.path.expanduser(scope))
        filters: dict[str, Any] = {"allowed_projects": None, "path_prefix": scope_abs}
        if path_prefix:
            filters["path_prefix"] = os.path.abspath(os.path.expanduser(path_prefix))

        if self.query_engine is not None:
            search_results = self.query_engine.search(q, filters, k)
        else:
            # Standalone use (tests, tools): no configured retriever/ranker available.
            search_results = self.backend.search_chunks(
                tokenize(q), top_k=k, **filters)

        hits = []
        for r in search_results:
            snippet = getattr(r, "snippet", "") or ""
            score = round(float(getattr(r, "score", 1.0)), 3)
            hits.append({
                "file": os.path.relpath(r.filepath, os.getcwd()),
                "name": r.name,
                "span": [r.start_line, r.end_line],
                "score": score,
                "snippet": snippet[:180].strip()
            })
        return hits
