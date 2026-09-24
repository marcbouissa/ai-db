"""Indexing: discover files, parse them in worker processes, write from one process.

Parsing (chunking, symbols, cross-refs, annotations, syntax checks) is pure and runs in
a process pool for large syncs. All storage writes happen in the calling process inside
a single transaction per sync.
"""

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from ai_db.constants import HARD_IGNORE_DIRS, PARALLEL_PARSE_MIN_FILES
from ai_db.ignorer import AidbIgnore
from ai_db.logger import _logger
from ai_db.parser.annotations import extract_annotations
from ai_db.parser.ast_visitor import extract_symbols
from ai_db.parser.chunker import chunk_file
from ai_db.parser.cross_refs import extract_cross_refs
from ai_db.parser.syntax import validate_python_syntax
from ai_db.storage.models import (
    AnnotationRecord,
    ChunkRecord,
    FileRecord,
    ParsedFile,
    SymbolRecord,
    SymbolRefRecord,
    SyntaxErrorRecord,
)
from ai_db.utils import compute_sha256, detect_project_name, should_index_path


def parse_file(filepath: str, file_hash: str, project: str) -> ParsedFile:
    """Read and fully parse one file. Pure: touches no storage."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    mtime = os.path.getmtime(filepath)
    parsed = ParsedFile(filepath=filepath, sha256=file_hash, last_modified=mtime, project=project)

    ext = os.path.splitext(filepath)[1].lower()
    if ext in (".py", ".pyi"):
        err = validate_python_syntax(content, filepath)
    else:
        from ai_db.parser.linters import get_linter
        err = get_linter().validate(filepath, content)
    if err:
        line, col, msg = err
        parsed.syntax_error = SyntaxErrorRecord(
            filepath=filepath, line=line, col=col, message=msg, timestamp=time.time(), project=project
        )

    parsed.chunks = [
        ChunkRecord(
            filepath=filepath,
            chunk_type=c["chunk_type"],
            name=c["name"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            content=c["content"],
            project=project,
        )
        for c in chunk_file(filepath, content)
    ]
    parsed.symbols = [
        SymbolRecord(
            name=s["name"], symbol_type=s["symbol_type"], filepath=s["filepath"],
            line=s["line"], signature=s.get("signature"), project=project,
        )
        for s in extract_symbols(filepath, content)
    ]
    parsed.refs = [
        SymbolRefRecord(
            caller_filepath=filepath, caller_name=r["caller_name"], caller_line=r["caller_line"],
            callee_name=r["callee_name"], ref_type=r["ref_type"], project=project,
        )
        for r in extract_cross_refs(filepath, content)
    ]
    parsed.annotations = [
        AnnotationRecord(
            filepath=filepath, line=a["line"], kind=a["kind"], content=a["content"],
            symbol=a.get("symbol"), project=project,
        )
        for a in extract_annotations(filepath, content)
    ]
    return parsed


def _parse_job(job: tuple[str, str, str]) -> ParsedFile:
    return parse_file(*job)


class Indexer:
    def __init__(self, db: Any):
        self.db = db
        self.db_path = getattr(db, "db_path", getattr(db, "backend_name", "storage"))
        self.post_sync_hooks: list[Any] = []

    def scan_directory(self, root_dir: str) -> list[str]:
        candidates = []
        root_dir = os.path.abspath(root_dir)
        ignorer = AidbIgnore(root_dir)
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in HARD_IGNORE_DIRS]
            for f in files:
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, root_dir)
                if ignorer.should_ignore(rel_path):
                    _logger.debug(f"Ignoring (aidbignore): {rel_path}")
                    continue
                if should_index_path(rel_path, f):
                    candidates.append(full_path)
        return candidates

    def _parse_many(self, jobs: list[tuple[str, str, str]]) -> list[ParsedFile]:
        if len(jobs) < PARALLEL_PARSE_MIN_FILES:
            return [_parse_job(j) for j in jobs]
        workers = min(os.cpu_count() or 1, max(1, len(jobs) // 8))
        # 'fork' does not re-import the caller's __main__ (library-safe); Windows has only 'spawn'.
        method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        ctx = multiprocessing.get_context(method)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            return list(pool.map(_parse_job, jobs, chunksize=8))

    def sync(self, root_dir: str, project: str | None = None, verbose: bool = True) -> dict[str, int]:
        root_dir = os.path.abspath(root_dir)
        if project is None:
            project = detect_project_name(root_dir)

        all_disk_files = set(self.scan_directory(root_dir))
        stored_files = self.db.get_files_by_prefix(root_dir)

        to_prune = [p for p in stored_files if p not in all_disk_files]
        jobs: list[tuple[str, str, str]] = []
        updated_paths = set()
        skipped = 0
        for filepath in sorted(all_disk_files):
            try:
                current_sha = compute_sha256(filepath)
            except OSError as e:
                _logger.debug(f"Skipping unreadable file {filepath}: {e}")
                continue
            stored_sha = stored_files.get(filepath)
            if stored_sha == current_sha:
                skipped += 1
                continue
            if stored_sha is not None:
                updated_paths.add(filepath)
            jobs.append((filepath, current_sha, project))

        parsed_files = self._parse_many(jobs)

        with self.db.transaction():
            for stored_path in to_prune:
                self.prune_file(stored_path)
            for parsed in parsed_files:
                if parsed.filepath in updated_paths:
                    self.prune_file(parsed.filepath)
                self._write_parsed(parsed)

        changed = bool(to_prune or parsed_files)
        for hook in self.post_sync_hooks:
            hook(changed)

        proj_skill_dirs = []
        for candidate_skill_dir in ["skills", ".agents/skills", ".agent/skills"]:
            candidate_full = os.path.join(root_dir, candidate_skill_dir)
            if os.path.isdir(candidate_full):
                proj_skill_dirs.append(candidate_full)
        if proj_skill_dirs and hasattr(self.db, "sync_skills"):
            self.db.sync_skills(skill_dirs=proj_skill_dirs, project=project, verbose=verbose)

        added = len(parsed_files) - len(updated_paths)
        result = {"added": added, "updated": len(updated_paths), "pruned": len(to_prune), "skipped": skipped}
        if verbose:
            db_name = os.path.basename(str(self.db_path))
            print(f"[{db_name}] Sync ({project}): +{added} ~{len(updated_paths)} -{len(to_prune)} ={skipped}")
        return result

    def _write_parsed(self, parsed: ParsedFile) -> None:
        if parsed.syntax_error is not None:
            self.db.upsert_syntax_error(parsed.syntax_error)
        else:
            self.db.delete_syntax_error(parsed.filepath)
        self.db.upsert_file(FileRecord(
            filepath=parsed.filepath, sha256=parsed.sha256, last_modified=parsed.last_modified,
            chunk_count=len(parsed.chunks), project=parsed.project,
        ))
        if parsed.chunks:
            self.db.insert_chunks(parsed.chunks)
        if parsed.symbols:
            self.db.insert_symbols(parsed.symbols)
        if parsed.refs:
            self.db.insert_symbol_refs(parsed.refs)
        if parsed.annotations:
            self.db.insert_annotations(parsed.annotations)

    def _index_file(self, filepath: str, file_hash: str, project: str = "global") -> None:
        with self.db.transaction():
            self._write_parsed(parse_file(filepath, file_hash, project))
        for hook in self.post_sync_hooks:
            hook(True)

    def prune_file(self, filepath: str):
        """Prunes file and cascades deletion through storage backend."""
        self.db.delete_file(filepath)
