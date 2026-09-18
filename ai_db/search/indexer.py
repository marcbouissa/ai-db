import os
import time
from typing import List, Dict, Any, Optional
from ai_db.constants import HARD_IGNORE_DIRS
from ai_db.logger import _logger
from ai_db.ignorer import AidbIgnore
from ai_db.utils import detect_project_name, compute_sha256, should_index_path
from ai_db.parser.syntax import validate_python_syntax
from ai_db.parser.ast_visitor import extract_symbols
from ai_db.parser.chunker import chunk_file
from ai_db.storage.models import (
    FileRecord, ChunkRecord, SymbolRecord,
    SymbolRefRecord, AnnotationRecord, SyntaxErrorRecord
)


class Indexer:
    def __init__(self, db: Any):
        self.db = db
        self.db_path = getattr(db, "db_path", getattr(db, "backend_name", "storage"))

    def scan_directory(self, root_dir: str) -> List[str]:
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

    def sync(self, root_dir: str, project: Optional[str] = None, verbose: bool = True) -> Dict[str, int]:
        root_dir = os.path.abspath(root_dir)
        if project is None:
            project = detect_project_name(root_dir)

        all_disk_files = set(self.scan_directory(root_dir))

        stored_files = self.db.get_files_by_prefix(root_dir)

        added = 0
        updated = 0
        pruned = 0
        skipped = 0

        # Prune deleted files
        for stored_path in list(stored_files.keys()):
            if stored_path not in all_disk_files:
                self.prune_file(stored_path)
                pruned += 1

        # Process on-disk files
        for filepath in all_disk_files:
            try:
                current_sha = compute_sha256(filepath)
            except Exception as e:
                _logger.debug(f"Skipping unreadable file {filepath}: {e}")
                continue

            stored_sha = stored_files.get(filepath)

            if stored_sha == current_sha:
                skipped += 1
                continue

            if stored_sha is not None:
                self.prune_file(filepath)
                updated += 1
            else:
                added += 1

            self._index_file(filepath, current_sha, project=project)

        # Auto-sync project-local skills if any exist in project
        proj_skill_dirs = []
        for candidate_skill_dir in ["skills", ".agents/skills", ".agent/skills"]:
            candidate_full = os.path.join(root_dir, candidate_skill_dir)
            if os.path.isdir(candidate_full):
                proj_skill_dirs.append(candidate_full)

        if proj_skill_dirs and hasattr(self.db, "sync_skills"):
            self.db.sync_skills(skill_dirs=proj_skill_dirs, project=project, verbose=verbose)

        if verbose:
            db_name = os.path.basename(str(self.db_path))
            print(f"[{db_name}] Sync ({project}): +{added} ~{updated} -{pruned} ={skipped}")
        return {"added": added, "updated": updated, "pruned": pruned, "skipped": skipped}

    def _index_file(self, filepath: str, file_hash: str, project: str = "global"):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            _logger.debug(f"_index_file: cannot read {filepath}: {e}")
            return

        # AST Validation for Python files
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".py", ".pyi"):
            syntax_err = validate_python_syntax(content, filepath)
            if syntax_err:
                line, col, msg = syntax_err
                self.db.upsert_syntax_error(
                    SyntaxErrorRecord(filepath=filepath, line=line, col=col, message=msg, timestamp=time.time(), project=project)
                )
            else:
                self.db.delete_syntax_error(filepath)
        else:
            # External linter validation for JS/TS/Shell/YAML etc.
            from ai_db.parser.linters import get_linter
            lint_err = get_linter().validate(filepath, content)
            if lint_err:
                line, col, msg = lint_err
                self.db.upsert_syntax_error(
                    SyntaxErrorRecord(filepath=filepath, line=line, col=col, message=msg, timestamp=time.time(), project=project)
                )
            else:
                self.db.delete_syntax_error(filepath)

        chunks_data = chunk_file(filepath, content)
        mtime = os.path.getmtime(filepath)

        # 1. Upsert file record
        self.db.upsert_file(
            FileRecord(filepath=filepath, sha256=file_hash, last_modified=mtime, chunk_count=len(chunks_data), project=project)
        )

        # 2. Insert chunks (backend handles compression & FTS indexing)
        chunk_records = [
            ChunkRecord(
                filepath=filepath,
                chunk_type=c["chunk_type"],
                name=c["name"],
                start_line=c["start_line"],
                end_line=c["end_line"],
                content=c["content"],
                project=project,
            )
            for c in chunks_data
        ]
        if chunk_records:
            self.db.insert_chunks(chunk_records)

        # 3. Symbol extraction & indexing
        symbols_data = extract_symbols(filepath, content)
        symbol_records = [
            SymbolRecord(
                name=s["name"],
                symbol_type=s["symbol_type"],
                filepath=s["filepath"],
                line=s["line"],
                signature=s.get("signature"),
                project=project,
            )
            for s in symbols_data
        ]
        if symbol_records:
            self.db.insert_symbols(symbol_records)

        # 4. Cross-references (calls, imports, inheritance)
        from ai_db.parser.cross_refs import extract_cross_refs
        cross_refs = extract_cross_refs(filepath, content)
        if cross_refs:
            ref_records = [
                SymbolRefRecord(
                    caller_filepath=filepath,
                    caller_name=r["caller_name"],
                    caller_line=r["caller_line"],
                    callee_name=r["callee_name"],
                    ref_type=r["ref_type"],
                    project=project,
                )
                for r in cross_refs
            ]
            self.db.insert_symbol_refs(ref_records)

        # 5. Annotation extraction (TODO/FIXME/HACK + docstrings)
        from ai_db.parser.annotations import extract_annotations
        annotations_data = extract_annotations(filepath, content)
        if annotations_data:
            annotation_records = [
                AnnotationRecord(
                    filepath=filepath,
                    line=a["line"],
                    kind=a["kind"],
                    content=a["content"],
                    symbol=a.get("symbol"),
                    project=project,
                )
                for a in annotations_data
            ]
            self.db.insert_annotations(annotation_records)

    def prune_file(self, filepath: str):
        """Prunes file and cascades deletion through storage backend."""
        self.db.delete_file(filepath)
