import os
import time
import zlib
import sqlite3
from typing import List, Dict, Any, Optional
from ai_db.constants import HARD_IGNORE_DIRS
from ai_db.logger import _logger
from ai_db.ignorer import AidbIgnore
from ai_db.utils import detect_project_name, compute_sha256, should_index_path
from ai_db.parser.syntax import validate_python_syntax
from ai_db.parser.ast_visitor import extract_symbols
from ai_db.parser.chunker import chunk_file

class Indexer:
    def __init__(self, db):
        self.db = db
        self.conn = db.conn
        self.db_path = db.db_path

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

        cur = self.conn.cursor()
        cur.execute(
            "SELECT filepath, sha256 FROM files WHERE filepath = ? OR filepath LIKE ?",
            (root_dir, f"{root_dir}{os.sep}%")
        )
        stored_files = {row["filepath"]: row["sha256"] for row in cur.fetchall()}

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

        if proj_skill_dirs:
            self.db.sync_skills(skill_dirs=proj_skill_dirs, project=project, verbose=verbose)

        self.conn.commit()
        if verbose:
            print(f"[{os.path.basename(self.db_path)}] Sync ({project}): +{added} ~{updated} -{pruned} ={skipped}")
        return {"added": added, "updated": updated, "pruned": pruned, "skipped": skipped}

    def _index_file(self, filepath: str, file_hash: str, project: str = "global"):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            _logger.debug(f"_index_file: cannot read {filepath}: {e}")
            return

        cur = self.conn.cursor()

        # AST Validation for Python files
        ext = os.path.splitext(filepath)[1].lower()
        if ext in (".py", ".pyi"):
            syntax_err = validate_python_syntax(content, filepath)
            if syntax_err:
                line, col, msg = syntax_err
                cur.execute(
                    """INSERT OR REPLACE INTO syntax_errors (filepath, line, col, message, timestamp, project)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (filepath, line, col, msg, time.time(), project)
                )
            else:
                cur.execute("DELETE FROM syntax_errors WHERE filepath = ?", (filepath,))
        else:
            # F1: External linter validation for JS/TS/Shell/YAML etc.
            from ai_db.parser.linters import get_linter
            lint_err = get_linter().validate(filepath, content)
            if lint_err:
                line, col, msg = lint_err
                cur.execute(
                    """INSERT OR REPLACE INTO syntax_errors (filepath, line, col, message, timestamp, project)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (filepath, line, col, msg, time.time(), project)
                )
            else:
                cur.execute("DELETE FROM syntax_errors WHERE filepath = ?", (filepath,))


        chunks = chunk_file(filepath, content)
        mtime = os.path.getmtime(filepath)

        cur.execute(
            "INSERT OR REPLACE INTO files (filepath, sha256, last_modified, chunk_count, project) VALUES (?, ?, ?, ?, ?)",
            (filepath, file_hash, mtime, len(chunks), project)
        )

        for c in chunks:
            raw_bytes = c["content"].encode("utf-8")
            z_blob = zlib.compress(raw_bytes, level=9)
            cur.execute(
                """INSERT INTO chunks (filepath, chunk_type, name, start_line, end_line, zcontent, project)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (filepath, c["chunk_type"], c["name"], c["start_line"], c["end_line"], z_blob, project)
            )
            chunk_id = cur.lastrowid
            cur.execute(
                "INSERT INTO fts_index (content, filepath, name, chunk_id) VALUES (?, ?, ?, ?)",
                (f"{c['name']} {c['content']}", filepath, c["name"], chunk_id)
            )

        # Symbol extraction & indexing
        symbols = extract_symbols(filepath, content)
        for s in symbols:
            cur.execute(
                """INSERT INTO symbols (name, symbol_type, filepath, line, signature, project)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (s["name"], s["symbol_type"], s["filepath"], s["line"], s["signature"], project)
            )

        # F2: Cross-reference extraction (calls, imports, inheritance)
        from ai_db.parser.cross_refs import extract_cross_refs
        cross_refs = extract_cross_refs(filepath, content)
        if cross_refs:
            cur.executemany(
                """INSERT INTO symbol_refs (caller_filepath, caller_name, caller_line, callee_name, ref_type, project)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(filepath, r["caller_name"], r["caller_line"], r["callee_name"], r["ref_type"], project)
                 for r in cross_refs]
            )

        # F10: Annotation extraction (TODO/FIXME/HACK + docstrings)
        from ai_db.parser.annotations import extract_annotations
        annotations = extract_annotations(filepath, content)
        if annotations:
            cur.executemany(
                """INSERT INTO annotations (filepath, line, kind, symbol, content, project)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(filepath, a["line"], a["kind"], a.get("symbol"), a["content"], project)
                 for a in annotations]
            )

    def prune_file(self, filepath: str):
        cur = self.conn.cursor()
        # Delete FTS rows first (they reference chunk ids)
        cur.execute("DELETE FROM fts_index WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM chunks WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM symbols WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM syntax_errors WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM analysis_refs WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM symbol_refs WHERE caller_filepath = ?", (filepath,))
        cur.execute("DELETE FROM annotations WHERE filepath = ?", (filepath,))
        cur.execute("DELETE FROM files WHERE filepath = ?", (filepath,))



