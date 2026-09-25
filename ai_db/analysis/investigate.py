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
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

from ai_db.analysis.pack import EntryPoint, Evidence, InvestigationPack
from ai_db.analysis.resolve import Resolver, scope_of
from ai_db.constants import (
    DELEGATE_MAX_TOKENS,
    DIFF_QUERY_BOOST,
    DIFF_SEEDS_PER_FILE,
    EXPLAIN_CALLER_SEEDS,
    GIT_TIMEOUT_S,
    IMPACT_DEPTH,
    INVESTIGATE_BODY_SHARE,
    INVESTIGATE_MIN_BUDGET,
    MAX_EXPANDED_PER_SEED,
    MAX_OMITTED,
    MAX_TESTS,
    MODULE_SEED_FACTOR,
    SEED_K,
    TRACE_CONTEXT_LINES,
    TRACE_DEPTH,
    TRACE_MAX_NODES,
)
from ai_db.errors import AiDbConfigError
from ai_db.parser.chunker import count_tokens
from ai_db.search.ranking import last_component
from ai_db.search.retriever import SNIPPET_CHARS
from ai_db.storage.models import ChunkRecord, SearchResult
from ai_db.utils import get_allowed_projects

MODES = ("locate", "explain", "impact", "flow", "diff")
ROLE_ORDER = {"seed": 0, "parent": 1, "callee": 2, "caller": 3, "test": 4}
_BUILTINS = frozenset(dir(builtins)) | frozenset({
    "append", "extend", "get", "items", "keys", "values", "join", "split", "strip", "format",
    "startswith", "endswith", "replace", "lower", "upper", "update", "pop", "add", "execute",
    "fetchall", "fetchone", "cursor", "commit", "encode", "decode", "read", "write", "close",
})


def _cid(chunk: ChunkRecord) -> int:
    """Id of a stored chunk (every chunk read from the backend has one)."""
    if chunk.id is None:
        raise ValueError(f"chunk {chunk.qualified_name} has no id")
    return chunk.id


def is_test_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    base = parts[-1]
    return "tests" in parts[:-1] or "test" in parts[:-1] or base.startswith("test_") \
        or base.endswith(("_test.py", ".test.ts", ".test.js", ".spec.ts", ".spec.js", "_test.go"))


def _git_changed_spans(since: str, root: str) -> dict[str, list[int]]:
    """Return {absolute_filepath: [changed line numbers]} for `git diff <since>`.

    `--unified=0` drops context lines, so the hunk headers carry the whole
    signal. Raises AiDbConfigError when git cannot answer: a diff-mode
    investigation with no diff would silently return an empty pack, which
    reads as "nothing to review" rather than "I could not look".
    """
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S, cwd=root, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        raise AiDbConfigError(f"mode 'diff' needs a git repository: {err}") from err
    if toplevel.returncode != 0:
        raise AiDbConfigError(f"mode 'diff' needs a git repository: {root} is not in one")

    repo_root = toplevel.stdout.strip()
    try:
        diff = subprocess.run(
            ["git", "diff", "--unified=0", "--no-color", since, "--"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S, cwd=repo_root, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as err:
        raise AiDbConfigError(f"git diff against {since!r} failed: {err}") from err
    if diff.returncode != 0:
        raise AiDbConfigError(f"git diff against {since!r} failed: {diff.stderr.strip()}")

    spans: dict[str, list[int]] = {}
    current: str | None = None
    hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
    for line in diff.stdout.splitlines():
        if line.startswith("+++ "):
            # "+++ b/path" for a change, "+++ /dev/null" for a deletion
            target = line[4:].strip()
            if target == "/dev/null":
                current = None
            elif target.startswith("b/"):
                current = os.path.join(repo_root, target[2:])
            else:
                current = os.path.join(repo_root, target)
            continue
        if current is None:
            continue
        m = hunk.match(line)
        if m is None:
            continue
        start = int(m.group(1))
        count = int(m.group(2) or 1)
        spans.setdefault(current, []).extend(range(start, start + count))
    return spans


def _query_tokens(query: str) -> set[str]:
    """Split a diff-mode query into comparable identifier tokens.

    "symbol centrality rebuild after sync" has to match a chunk whose text says
    "rebuild_symbol_centrality", so a literal substring test is useless here.
    """
    return {t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2}


def _tokens_present(needle: set[str], chunk: ChunkRecord) -> bool:
    """True when the chunk mentions enough of the query to be what was asked for.

    Every token counts, on the chunk's text and its qualified name: a caller who
    says "centrality rebuild" wants that function, not the file it sits in.
    """
    text = f"{chunk.qualified_name}\n{chunk.name}\n{chunk.content}".lower()
    return all(token in text for token in needle)


def _line_ranges(lines: list[int]) -> str:
    """Collapse [1,2,3,7,9,10] into "L1-3 L7 L9-10" for display."""
    if not lines:
        return ""
    ordered = sorted(set(lines))
    parts: list[str] = []
    start = prev = ordered[0]
    for ln in ordered[1:]:
        if ln == prev + 1:
            prev = ln
            continue
        parts.append(f"L{start}" if start == prev else f"L{start}-{prev}")
        start = prev = ln
    parts.append(f"L{start}" if start == prev else f"L{start}-{prev}")
    return " ".join(parts)


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
        self._resolver = Resolver(self.db)

    # ------------------------------------------------------------------ api

    def investigate(self, query: str, budget_tokens: int = 8000, mode: str = "explain",
                    project: str | None = None, allowed_projects: list[str] | None = None,
                    languages: list[str] | None = None, chunk_types: list[str] | None = None,
                    modified_since: float | None = None,
                    relative_to: str | None = None,
                    since: str | None = None,
                    root: str | None = None) -> InvestigationPack:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if budget_tokens < INVESTIGATE_MIN_BUDGET:
            raise ValueError(f"budget_tokens must be >= {INVESTIGATE_MIN_BUDGET}")
        if not query.strip() and mode != "diff":
            raise ValueError("query must not be empty")
        if mode == "diff" and not since:
            raise AiDbConfigError("mode 'diff' requires 'since' (a git ref to diff against)")

        allowed = get_allowed_projects(project or "global", allowed_projects,
                                       self.vdb.query_engine.cross_project)
        filters = {"allowed_projects": allowed, "languages": languages,
                   "chunk_types": chunk_types, "modified_since": modified_since}

        if mode == "diff":
            # Seeds come from the diff, not from retrieval.
            spans = _git_changed_spans(since or "", root or ".")
            seeds = self._diff_seeds(spans, query, allowed)
            changes = [{"filepath": path,
                        "lines": _line_ranges(lines),
                        "changed_lines": len(lines)}
                       for path, lines in sorted(spans.items())]
        else:
            seeds = self.vdb.query_engine.search(query, filters, SEED_K[mode])
            changes = []

        items: dict[int, _Item] = {}
        edges: list[list[str]] = []
        unresolved: set[str] = set()
        tests: list[dict[str, Any]] = []
        self._add_seeds(seeds, items)
        seed_items = sorted((it for it in items.values() if it.role == "seed"),
                            key=lambda it: (-it.score, it.order))

        if mode in ("explain", "impact", "flow", "diff"):
            self._add_parents(seed_items, items)
            self._add_tests(seed_items, allowed, items, tests, relative_to)
        if mode == "explain":
            self._add_callees(seed_items, allowed, items, edges, unresolved)
            # direct users of the top seeds show how the code is entered
            self._add_callers(seed_items[:EXPLAIN_CALLER_SEEDS], allowed, items, edges, depth_limit=1)
        if mode in ("impact", "diff"):
            # diff mode answers "what breaks because of this change"
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
        pack.changes = changes

        if mode == "flow":
            # For flow mode: pick the best entry-like seed and run trace from it
            from ai_db.analysis.trace import TraceEngine
            from ai_db.analysis.trace_format import format_tree
            entry_seed = self._pick_entry_seed(seed_items)
            if entry_seed:
                trace_engine = TraceEngine(
                    self.db,
                    allowed_projects=allowed,
                )
                trace_result = trace_engine.trace(
                    entry=entry_seed.chunk.qualified_name,
                    depth=TRACE_DEPTH,
                    max_nodes=TRACE_MAX_NODES,
                    direction="down",
                    include_tests=False,
                )
                # Store trace in pack for later output
                pack.trace_result = trace_result
                pack.trace_output = format_tree(trace_result, with_code=True, context_lines=TRACE_CONTEXT_LINES)

        self._pack(pack, items, edges, relative_to)
        return pack

    def _pick_entry_seed(self, seed_items: list[_Item]) -> _Item | None:
        """Pick the most entry-like seed for flow tracing.
        Prefers: module-level code, __main__, then highest-scored seed."""
        for it in seed_items:
            if it.chunk.chunk_type == "module":
                return it
            if "__main__" in it.chunk.qualified_name:
                return it
        return seed_items[0] if seed_items else None

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
        existing = items.get(_cid(chunk))
        if existing is not None:
            if why not in existing.why:
                existing.why.append(why)
            existing.score = max(existing.score, score)
            return
        items[_cid(chunk)] = _Item(chunk, role, score, [why], len(items) + 1)

    # ----------------------------------------------------------- diff seeds

    def _diff_seeds(self, spans: dict[str, list[int]], query: str,
                    allowed: list[str]) -> list[SearchResult]:
        """Rank the symbols touched by a diff.

        ``query`` is used only to break ties, preferring chunks whose text
        contains it.
        """
        if not spans:
            return []

        needle = _query_tokens(query)
        by_file: list[list[SearchResult]] = []
        for filepath, lines in spans.items():
            try:
                chunks = self.db.get_chunks_for_file(filepath)
            except FileNotFoundError:
                continue
            per_file: list[tuple[float, int, int, SearchResult]] = []
            for chunk in chunks:
                if chunk.project not in allowed:
                    continue
                if chunk.chunk_type == "module":
                    continue
                hit = sum(1 for ln in lines if chunk.start_line <= ln <= chunk.end_line)
                if hit == 0:
                    continue
                # Normalize by chunk size: a 400-line class that had three
                # lines touched must not outrank the 5-line function the change
                # is actually about. hit/sqrt(span) rises for focused chunks
                # and falls as the containing chunk grows around the edit.
                size = max(1, chunk.end_line - chunk.start_line + 1)
                score = hit / size ** 0.5
                # The query is the caller telling us which part of the diff
                # they care about ("just the centrality rebuild"). A tie-break
                # nudge loses to raw diff density, so it has to be a multiplier.
                if needle and _tokens_present(needle, chunk):
                    score *= DIFF_QUERY_BOOST
                per_file.append((score, size, hit, SearchResult(
                    chunk_id=chunk.id or 0,
                    filepath=chunk.filepath,
                    name=chunk.name,
                    chunk_type=chunk.chunk_type,
                    project=chunk.project,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    score=score,
                    snippet=chunk.content[:SNIPPET_CHARS],
                    qualified_name=chunk.qualified_name,
                    language=chunk.language,
                    parent_id=chunk.parent_id,
                    signals={"diff_lines": float(hit),
                             "diff_file": float(len(lines))},
                )))
            # A class_header spans every method in its file, so it always covers
            # at least as many changed lines as the method that was actually
            # edited. Drop any chunk whose changed lines are fully covered by a
            # strictly smaller one, so the seed is the most specific symbol
            # available rather than its container.
            specific = [entry for entry in per_file
                        if not any(other[2] >= entry[2] and other[1] < entry[1]
                                   for other in per_file if other[3] is not entry[3])]
            specific.sort(key=lambda row: (-row[0], row[1]))
            if specific:
                by_file.append([row[3] for row in specific[:DIFF_SEEDS_PER_FILE]])
        # Round-robin one chunk per file at a time, and do NOT re-sort: the
        # interleaved order is the ranking. A commit that rewrites a 400-line
        # markdown file would otherwise fill every seed slot and hide the code.
        ordered = sorted(by_file, key=lambda group: -group[0].score)
        results: list[SearchResult] = []
        for depth in range(DIFF_SEEDS_PER_FILE):
            for group in ordered:
                if depth < len(group):
                    results.append(group[depth])
        return results[:SEED_K["diff"]]

    def _add_parents(self, seeds: list[_Item], items: dict[int, _Item]) -> None:
        parent_ids = [s.chunk.parent_id for s in seeds if s.chunk.parent_id is not None]
        parents = {c.id: c for c in self.db.get_chunks_by_ids(parent_ids)}
        for s in seeds:
            parent = parents.get(s.chunk.parent_id) if s.chunk.parent_id is not None else None
            if parent is not None and parent.chunk_type == "class_header":
                self._add(items, parent, "parent", s.score * 0.5, f"class of {s.chunk.qualified_name}")

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
            self.db.find_chunks_by_symbol(sorted(all_names), allowed)  # populate resolver cache
            next_frontier = []
            for chunk, score, names in per_src:
                result = self._resolver.resolve_callees(set(names), chunk, allowed)
                if hop == 1:
                    unresolved |= result.unresolved
                targets = [c for n in names if n in result.resolved for c in result.resolved[n]]
                for callee in targets[:MAX_EXPANDED_PER_SEED]:
                    self._add(items, callee, "callee", score * (0.6 ** hop),
                              f"called by {chunk.qualified_name}")
                    edges.append([chunk.qualified_name, callee.qualified_name])
                    if hop == 1 and chunk.token_count <= DELEGATE_MAX_TOKENS:
                        next_frontier.append((callee, score))
            frontier = next_frontier
            if not frontier:
                return

    def _add_callers(self, seeds: list[_Item], allowed: list[str], items: dict[int, _Item],
                     edges: list[list[str]], depth_limit: int) -> None:
        frontier = [(s.chunk, s.score) for s in seeds]
        seen = set(items)
        for depth in range(1, depth_limit + 1):
            next_frontier = []
            for target, score in frontier:
                name = last_component(target.qualified_name)
                callers = self._resolver.resolve_callers(name, target, allowed, limit=100)
                added = 0
                for caller in callers:
                    if added >= MAX_EXPANDED_PER_SEED:
                        break
                    edges.append([caller.qualified_name, target.qualified_name])
                    if caller.id is None or caller.id in seen:
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
            test_callers = self._resolver.find_test_callers(name, s.chunk, allowed, limit=100)
            for chunk in test_callers:
                if len(tests) >= MAX_TESTS:
                    return
                if chunk.id is None or chunk.id in listed:
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
