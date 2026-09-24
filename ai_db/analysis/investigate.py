"""``investigate``: replace the agent's grep/read/reason loop with one deterministic call.

Pipeline (no LLM):

1. seeds: the configured retriever + ranker, top ``SEED_K`` chunks
2. structure: the enclosing ``class_header`` of every method seed
3. graph (by mode):
   - ``locate``: nothing more
   - ``explain``: one hop of callees of every seed
   - ``impact``: callers of every seed, transitively up to ``IMPACT_DEPTH``
4. tests (explain/impact): test functions that call a seed
5. recent changes: ``git log`` for the seed files (only when they are in a git repo)
6. packing under ``budget_tokens``: seeds get full bodies, everything else a stub plus a
   ``ref:<id>`` handle for ``expand``; lowest-scored bodies are demoted to stubs, then
   dropped into ``omitted``, until the pack fits.
"""

from __future__ import annotations

import builtins
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

from ai_db.analysis.pack import EntryPoint, Evidence, InvestigationPack
from ai_db.constants import (
    COMPOUND_MAX_DEFINITIONS,
    DELEGATE_MAX_TOKENS,
    EXPLAIN_CALLER_SEEDS,
    IMPACT_DEPTH,
    INVESTIGATE_BODY_SHARE,
    INVESTIGATE_MIN_BUDGET,
    MAX_EXPANDED_PER_SEED,
    MAX_OMITTED,
    MAX_TESTS,
    MODULE_SEED_FACTOR,
    SEED_K,
)
from ai_db.parser.chunker import count_tokens
from ai_db.search.query_builder import split_identifier
from ai_db.search.ranking import last_component
from ai_db.storage.models import ChunkRecord, SearchResult
from ai_db.utils import get_allowed_projects

MODES = ("locate", "explain", "impact")
ROLE_ORDER = {"seed": 0, "parent": 1, "callee": 2, "caller": 3, "test": 4}
_BUILTINS = frozenset(dir(builtins)) | frozenset({
    "append", "extend", "get", "items", "keys", "values", "join", "split", "strip", "format",
    "startswith", "endswith", "replace", "lower", "upper", "update", "pop", "add", "execute",
    "fetchall", "fetchone", "cursor", "commit", "encode", "decode", "read", "write", "close",
})


def is_test_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    base = parts[-1]
    return "tests" in parts[:-1] or "test" in parts[:-1] or base.startswith("test_") \
        or base.endswith(("_test.py", ".test.ts", ".test.js", ".spec.ts", ".spec.js", "_test.go"))


def scope_of(qualified_name: str) -> str:
    """Chunk qualified name -> cross-ref caller scope (``module.Class.method``)."""
    return f"module.{qualified_name}"


def qualified_of(scope: str) -> str | None:
    """Cross-ref caller scope -> chunk qualified name (None for module-level code)."""
    if scope == "module":
        return None
    return scope.removeprefix("module.")


@dataclass
class _Item:
    chunk: ChunkRecord
    role: str
    score: float
    why: list[str] = field(default_factory=list)
    order: int = 0


class Investigator:
    def __init__(self, vdb: Any):
        self.vdb = vdb
        self.db = vdb.backend
        self._import_cache: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ api

    def investigate(self, query: str, budget_tokens: int = 8000, mode: str = "explain",
                    project: str | None = None, allowed_projects: list[str] | None = None,
                    languages: list[str] | None = None, chunk_types: list[str] | None = None,
                    modified_since: float | None = None,
                    relative_to: str | None = None) -> InvestigationPack:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if budget_tokens < INVESTIGATE_MIN_BUDGET:
            raise ValueError(f"budget_tokens must be >= {INVESTIGATE_MIN_BUDGET}")
        if not query.strip():
            raise ValueError("query must not be empty")

        allowed = get_allowed_projects(project or "global", allowed_projects,
                                       self.vdb.query_engine.cross_project)
        filters = {"allowed_projects": allowed, "languages": languages,
                   "chunk_types": chunk_types, "modified_since": modified_since}
        seeds = self.vdb.query_engine.search(query, filters, SEED_K[mode])

        items: dict[int, _Item] = {}
        edges: list[list[str]] = []
        unresolved: set[str] = set()
        tests: list[dict[str, Any]] = []
        self._add_seeds(seeds, items)
        seed_items = sorted((it for it in items.values() if it.role == "seed"),
                            key=lambda it: (-it.score, it.order))

        if mode in ("explain", "impact"):
            self._add_parents(seed_items, items)
            self._add_tests(seed_items, allowed, items, tests, relative_to)
        if mode == "explain":
            self._add_callees(seed_items, allowed, items, edges, unresolved)
            # direct users of the top seeds show how the code is entered
            self._add_callers(seed_items[:EXPLAIN_CALLER_SEEDS], allowed, items, edges, depth_limit=1)
        if mode == "impact":
            self._add_callers(seed_items, allowed, items, edges, depth_limit=IMPACT_DEPTH)

        pack = InvestigationPack(query=query, mode=mode, budget_tokens=budget_tokens)
        pack.retrieval = {
            "mode": self.vdb.config.retrieval_mode,
            "embedding_model": self.vdb.embedder.model_id if self.vdb.embedder else None,
            "rerank_model": self.vdb.reranker.model_id if self.vdb.reranker else None,
        }
        pack.entry_points = [
            EntryPoint(it.chunk.qualified_name, self._display(it.chunk.filepath, relative_to),
                       f"L{it.chunk.start_line}-{it.chunk.end_line}", "; ".join(it.why))
            for it in seed_items[:5]
        ]
        pack.tests = tests
        pack.unresolved = sorted(unresolved)
        pack.recent_changes = self._recent_changes([it.chunk.filepath for it in seed_items])
        self._pack(pack, items, edges, relative_to)
        return pack

    # ------------------------------------------------------------ gathering

    def _add_seeds(self, seeds: list[SearchResult], items: dict[int, _Item]) -> None:
        chunks = {c.id: c for c in self.db.get_chunks_by_ids([s.chunk_id for s in seeds])}
        for rank, s in enumerate(seeds, start=1):
            chunk = chunks.get(s.chunk_id)
            if chunk is None:
                continue
            why = [f"rank {rank}"]
            sig = s.signals
            if sig.get("exact_symbol"):
                why.append(f"name matches '{last_component(chunk.qualified_name)}'")
            if "bm25_rank" in sig:
                why.append(f"bm25 #{int(sig['bm25_rank'])}")
            if "vec_rank" in sig:
                why.append(f"vector #{int(sig['vec_rank'])}")
            if "rerank" in sig:
                why.append(f"rerank {sig['rerank']:.2f}")
            if sig.get("graph"):
                why.append(f"graph {sig['graph']:.2f}")
            score = float(s.score) or 1e-6
            if chunk.chunk_type == "module":
                score *= MODULE_SEED_FACTOR  # import/global blocks rarely answer a question
                why.append("module header")
            items[chunk.id] = _Item(chunk, "seed", score, why, rank)

    def _add(self, items: dict[int, _Item], chunk: ChunkRecord, role: str, score: float,
             why: str) -> None:
        existing = items.get(chunk.id)
        if existing is not None:
            if why not in existing.why:
                existing.why.append(why)
            existing.score = max(existing.score, score)
            return
        items[chunk.id] = _Item(chunk, role, score, [why], len(items) + 1)

    def _add_parents(self, seeds: list[_Item], items: dict[int, _Item]) -> None:
        parent_ids = [s.chunk.parent_id for s in seeds if s.chunk.parent_id is not None]
        parents = {c.id: c for c in self.db.get_chunks_by_ids(parent_ids)}
        for s in seeds:
            parent = parents.get(s.chunk.parent_id) if s.chunk.parent_id is not None else None
            if parent is not None and parent.chunk_type == "class_header":
                self._add(items, parent, "parent", s.score * 0.5, f"class of {s.chunk.qualified_name}")

    def _imports(self, filepath: str) -> set[str]:
        """Dotted-name parts imported anywhere in ``filepath`` (cached per call)."""
        cached = self._import_cache.get(filepath)
        if cached is None:
            cached = set()
            for ref in self.db.get_refs_from(filepath, None, ("import",)):
                cached.update(ref.callee_name.split("."))
            self._import_cache[filepath] = cached
        return cached

    def _linked(self, user_file: str, definition: ChunkRecord) -> bool:
        """``user_file`` can reach ``definition``: same file, or it imports the definition's
        module (file stem) or its top-level name."""
        if user_file == definition.filepath:
            return True
        stem = os.path.splitext(os.path.basename(definition.filepath))[0]
        top = definition.qualified_name.split(".")[0]
        return bool(self._imports(user_file) & {stem, top})

    def _resolve(self, names: set[str], near: ChunkRecord,
                 found: dict[str, list[ChunkRecord]]) -> dict[str, list[ChunkRecord]]:
        """Deterministic name resolution (never the caller itself): a definition in the same
        file, else one the file imports, else the only definition of that name, else — for
        compound names only — up to COMPOUND_MAX_DEFINITIONS definitions. Other names stay
        unresolved."""
        out: dict[str, list[ChunkRecord]] = {}
        for name in names:
            cands = [c for c in found.get(name, []) if c.id != near.id]
            same = [c for c in cands if c.filepath == near.filepath]
            linked = [c for c in cands if self._linked(near.filepath, c)]
            if same or linked:
                out[name] = sorted(same or linked, key=lambda c: c.id)[:1]
            elif len(cands) == 1:
                out[name] = cands
            elif cands and len(split_identifier(name)) >= 2:
                # compound name, several definitions (interface + implementation): keep both
                out[name] = sorted(cands, key=lambda c: c.id)[:COMPOUND_MAX_DEFINITIONS]
        return out

    def _caller_links_to(self, ref: Any, target: ChunkRecord) -> bool:
        """A caller counts if the called name is a compound identifier (``search_chunks``,
        ``parseConfig``: collisions are rare, so duck-typed calls like ``self.db.search_chunks``
        are kept), or if its file can reach the target (same file or import). Single-word
        names need the link (``re.search`` must not count as ``Engine.search``)."""
        name = last_component(target.qualified_name)
        return len(split_identifier(name)) >= 2 or self._linked(ref.caller_filepath, target)

    def _add_callees(self, seeds: list[_Item], allowed: list[str], items: dict[int, _Item],
                     edges: list[list[str]], unresolved: set[str]) -> None:
        """One hop of callees; a second hop through thin delegates (small bodies)."""
        frontier = [(s.chunk, s.score) for s in seeds]
        for hop in (1, 2):
            per_src: list[tuple[ChunkRecord, float, list[str]]] = []
            all_names: set[str] = set()
            for chunk, score in frontier:
                refs = self.db.get_refs_from(chunk.filepath, scope_of(chunk.qualified_name))
                # call order (first occurrence) is the reading order of the body
                names = list(dict.fromkeys(r.callee_name for r in refs if r.callee_name not in _BUILTINS))
                per_src.append((chunk, score, names))
                all_names |= set(names)
            found = self.db.find_chunks_by_symbol(sorted(all_names), allowed)
            next_frontier = []
            for chunk, score, names in per_src:
                resolved = self._resolve(set(names), chunk, found)
                if hop == 1:
                    unresolved |= set(names) - set(resolved)
                targets = [c for n in names if n in resolved for c in resolved[n]]
                for callee in targets[:MAX_EXPANDED_PER_SEED]:
                    self._add(items, callee, "callee", score * (0.6 ** hop),
                              f"called by {chunk.qualified_name}")
                    edges.append([chunk.qualified_name, callee.qualified_name])
                    if hop == 1 and chunk.token_count <= DELEGATE_MAX_TOKENS:
                        next_frontier.append((callee, score))
            frontier = next_frontier
            if not frontier:
                return

    def _callers_of(self, name: str, allowed: list[str]) -> list[Any]:
        refs = self.db.query_symbol_callers(callee_name=name, allowed_projects=allowed, limit=100)
        return [r for r in refs if r.ref_type in ("call", "inherit")]

    def _add_callers(self, seeds: list[_Item], allowed: list[str], items: dict[int, _Item],
                     edges: list[list[str]], depth_limit: int) -> None:
        frontier = [(s.chunk, s.score) for s in seeds]
        seen = set(items)
        for depth in range(1, depth_limit + 1):
            next_frontier = []
            for target, score in frontier:
                name = last_component(target.qualified_name)
                added = 0
                for ref in self._callers_of(name, allowed):
                    qn = qualified_of(ref.caller_name)
                    if qn is None or added >= MAX_EXPANDED_PER_SEED:
                        continue
                    if not self._caller_links_to(ref, target):
                        continue
                    caller = self.db.get_chunk_by_qualified_name(ref.caller_filepath, qn)
                    if caller is None:
                        continue
                    edges.append([caller.qualified_name, target.qualified_name])
                    if caller.id in seen:
                        continue
                    seen.add(caller.id)
                    added += 1
                    role = "test" if is_test_path(caller.filepath) else "caller"
                    self._add(items, caller, role, score * (0.6 ** depth),
                              f"calls {target.qualified_name} (depth {depth})")
                    next_frontier.append((caller, score))
            frontier = next_frontier
            if not frontier:
                return

    def _add_tests(self, seeds: list[_Item], allowed: list[str], items: dict[int, _Item],
                   tests: list[dict[str, Any]], relative_to: str | None) -> None:
        listed: set[int] = set()
        for s in seeds:
            if is_test_path(s.chunk.filepath):
                continue
            name = last_component(s.chunk.qualified_name)
            for ref in self._callers_of(name, allowed):
                qn = qualified_of(ref.caller_name)
                if qn is None or not is_test_path(ref.caller_filepath):
                    continue
                if not self._caller_links_to(ref, s.chunk):
                    continue
                if len(tests) >= MAX_TESTS:
                    return
                chunk = self.db.get_chunk_by_qualified_name(ref.caller_filepath, qn)
                if chunk is None or chunk.id in listed:
                    continue
                listed.add(chunk.id)
                self._add(items, chunk, "test", s.score * 0.4, f"tests {s.chunk.qualified_name}")
                tests.append({"test": chunk.qualified_name,
                              "filepath": self._display(chunk.filepath, relative_to),
                              "lines": f"L{chunk.start_line}-{chunk.end_line}",
                              "covers": s.chunk.qualified_name})

    # ------------------------------------------------------------- git

    def _recent_changes(self, files: list[str], per_file: int = 5) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in list(dict.fromkeys(files))[:8]:
            directory = os.path.dirname(path) or "."
            if not os.path.isdir(directory):
                continue
            inside = subprocess.run(["git", "-C", directory, "rev-parse", "--is-inside-work-tree"],
                                    capture_output=True, text=True, timeout=5, check=False)
            if inside.returncode != 0 or inside.stdout.strip() != "true":
                continue  # not version controlled: nothing to report for this file
            log = subprocess.run(
                ["git", "-C", directory, "log", f"-n{per_file}", "--date=short",
                 "--format=%h%x1f%s%x1f%ad", "--", os.path.basename(path)],
                capture_output=True, text=True, timeout=10, check=True)
            commits = []
            for line in log.stdout.splitlines():
                h, subject, date = line.split("\x1f")
                commits.append({"commit": h, "subject": subject, "date": date})
            if commits:
                out.append({"filepath": path, "commits": commits})
        return out

    # ------------------------------------------------------------- packing

    @staticmethod
    def _stub(chunk: ChunkRecord) -> str:
        lines = [ln for ln in chunk.content.splitlines() if ln.strip()]
        head: list[str] = []
        for ln in lines:
            head.append(ln.rstrip())
            if len(head) >= 3 or ln.rstrip().endswith((":", "{")) and len(head) > 1:
                break
        return "\n".join(head)[:300]

    @staticmethod
    def _display(path: str, relative_to: str | None) -> str:
        return os.path.relpath(path, relative_to) if relative_to else path

    def _measure(self, pack: InvestigationPack) -> int:
        return count_tokens(json.dumps(pack.to_dict(), ensure_ascii=False))

    def _finalize(self, pack: InvestigationPack, edges: list[list[str]]) -> None:
        included = {e.qualified_name for e in pack.evidence}
        pack.call_graph = {
            "nodes": sorted(included),
            "edges": [e for e in edges if e[0] in included and e[1] in included],
        }
        pack.files_touched = sorted({e.filepath for e in pack.evidence})
        pack.token_count = self._measure(pack)

    def _pack(self, pack: InvestigationPack, items: dict[int, _Item], edges: list[list[str]],
              relative_to: str | None) -> None:
        """Greedy: walk items by (role, score); give a body (seed/parent) while bodies stay
        under ``INVESTIGATE_BODY_SHARE`` of the budget, else a stub, else list it in
        ``omitted``. Then trim until the finished pack fits."""
        ordered = sorted(items.values(), key=lambda it: (ROLE_ORDER[it.role], -it.score, it.order))
        self._finalize(pack, edges)
        used = pack.token_count
        body_spans: dict[str, list[tuple[int, int]]] = {}
        seen_edges_overhead = 12  # call-graph node + files_touched entry per evidence item
        for it in ordered:
            c = it.chunk
            if c.chunk_type != "class_header" and any(
                    a <= c.start_line and c.end_line <= b for a, b in body_spans.get(c.filepath, [])):
                continue  # already fully shown inside another body
            ref = self.vdb.analyzer_engine._store_analysis_ref(
                c.filepath, c.qualified_name, c.start_line, c.end_line, it.role, c.content)
            ev = Evidence(ref=ref, filepath=self._display(c.filepath, relative_to),
                          lines=f"L{c.start_line}-{c.end_line}", qualified_name=c.qualified_name,
                          role=it.role, score=round(it.score, 6), why="; ".join(it.why))
            base = count_tokens(json.dumps(ev.__dict__, ensure_ascii=False)) + seen_edges_overhead
            body_cost = base + count_tokens(json.dumps(c.content, ensure_ascii=False))
            stub = self._stub(c)
            stub_cost = base + count_tokens(json.dumps(stub, ensure_ascii=False))
            body_cap = pack.budget_tokens * INVESTIGATE_BODY_SHARE
            wants_body = it.role in ("seed", "parent") and c.chunk_type != "module"
            if wants_body and used + body_cost <= body_cap:
                ev.body = c.content
                used += body_cost
                if c.chunk_type != "class_header":
                    body_spans.setdefault(c.filepath, []).append((c.start_line, c.end_line))
            elif used + stub_cost <= pack.budget_tokens or not pack.evidence:
                # the best item is always returned, at least as a stub
                ev.stub = stub
                used += stub_cost
            else:
                self._omit(pack, c.qualified_name, ref, it.role)
                continue
            pack.evidence.append(ev)

        self._finalize(pack, edges)
        # Estimates are per item; correct any overshoot from the last items first.
        while pack.token_count > pack.budget_tokens and len(pack.evidence) > 1:
            last = pack.evidence[-1]
            if last.body is not None:
                last.stub, last.body = self._stub_from_body(last.body), None
            else:
                pack.evidence.pop()
                self._omit(pack, last.qualified_name, last.ref, last.role, front=True)
            self._finalize(pack, edges)
        # Tiny budgets: shed secondary sections, then the last body.
        while pack.token_count > pack.budget_tokens:
            if pack.omitted:
                pack.omitted.pop()
            elif pack.tests:
                pack.tests.pop()
            elif pack.recent_changes:
                pack.recent_changes.pop()
            elif len(pack.entry_points) > 1:
                pack.entry_points.pop()
            elif pack.unresolved:
                pack.unresolved.clear()
            elif pack.evidence and pack.evidence[0].body is not None:
                first = pack.evidence[0]
                first.stub, first.body = self._stub_from_body(first.body or ""), None
            else:
                break
            self._finalize(pack, edges)

    @staticmethod
    def _omit(pack: InvestigationPack, qualified_name: str, ref: str, role: str,
              front: bool = False) -> None:
        """Record an item that did not fit; only the first MAX_OMITTED are listed."""
        pack.omitted_count += 1
        entry = {"ref": ref, "qualified_name": qualified_name, "role": role}
        if front:
            pack.omitted.insert(0, entry)
            del pack.omitted[MAX_OMITTED:]
        elif len(pack.omitted) < MAX_OMITTED:
            pack.omitted.append(entry)

    @staticmethod
    def _stub_from_body(body: str) -> str:
        lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
        return "\n".join(lines[:3])[:300]
