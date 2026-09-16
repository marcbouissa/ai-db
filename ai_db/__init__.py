"""
ai_db: High-Performance Vector & Code Intelligence Database.
Modular architecture with zero external dependencies (Python standard library only).
"""

import os
from typing import List, Dict, Any, Tuple, Optional

from ai_db.constants import (
    DEFAULT_DB_FILE,
    DEFAULT_CONFIG_FILE,
    DEFAULT_SKILL_DIRS,
    INDEXABLE_EXTENSIONS,
    HARD_IGNORE_DIRS,
    VENDOR_NOISE_EXTENSIONS
)

from ai_db.utils import (
    load_config,
    detect_project_name,
    get_allowed_projects,
    compute_sha256,
    tokenize,
    strip_code_bloat,
    should_index_path
)

from ai_db.parser.syntax import validate_python_syntax
from ai_db.parser.ast_visitor import extract_symbols, extract_file_outline
from ai_db.parser.chunker import chunk_file

from ai_db.storage.database import Database
from ai_db.storage.state import get_session_state, set_session_state
from ai_db.memory.context import ContextMemory
from ai_db.search.indexer import Indexer
from ai_db.search.query import QueryEngine
from ai_db.search.skills import SkillRouter
from ai_db.analyzer.formatters import format_as_stub, format_as_sexp, Formatters
from ai_db.analyzer.references import ReferenceStore
from ai_db.analyzer.engine import AnalyzerEngine
from ai_db.watcher import run_watch as _run_watch

class VectorDB:
    """Unified Facade for ai-db, maintaining 100% backward compatibility."""

    def __init__(self, db_path: str = DEFAULT_DB_FILE):
        self.db = Database(db_path)
        self.db_path = self.db.db_path
        self.conn = self.db.conn
        self.context_memory = ContextMemory(self.conn)
        self.indexer = Indexer(self)
        self.query_engine = QueryEngine(self.conn)
        self.skill_router = SkillRouter(self.conn, self.db_path)
        self.analyzer_engine = AnalyzerEngine(self)
        self.formatters = Formatters

    # Storage & DB management
    def status(self) -> Dict[str, Any]:
        return self.db.status()

    def optimize(self, prune_missing: bool = True, default_format: Optional[str] = None) -> Dict[str, Any]:
        return self.db.optimize(prune_missing=prune_missing, default_format=default_format)

    def get_session_state(self, key: str) -> Optional[Any]:
        return get_session_state(self.conn, key)

    def set_session_state(self, key: str, value: Any):
        return set_session_state(self.conn, key, value)

    def close(self):
        self.db.close()

    # Indexing
    def scan_directory(self, root_dir: str) -> List[str]:
        return self.indexer.scan_directory(root_dir)

    def sync(self, root_dir: str, project: Optional[str] = None, verbose: bool = True) -> Dict[str, int]:
        return self.indexer.sync(root_dir, project=project, verbose=verbose)

    def _index_file(self, filepath: str, file_hash: str, project: str = "global"):
        return self.indexer._index_file(filepath, file_hash, project=project)

    def prune_file(self, filepath: str):
        return self.indexer.prune_file(filepath)

    # Search & Code Query
    def query(self, search_text: str, top_k: int = 5, relative_to: Optional[str] = None,
              project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.query_engine.query(search_text, top_k=top_k, relative_to=relative_to,
                                       project=project, allowed_projects=allowed_projects)

    def query_symbol(self, name: str, relative_to: Optional[str] = None,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.query_engine.query_symbol(name, relative_to=relative_to,
                                              project=project, allowed_projects=allowed_projects)

    def check_syntax(self, target_path: Optional[str] = None, relative_to: Optional[str] = None,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.query_engine.check_syntax(target_path, relative_to=relative_to,
                                              project=project, allowed_projects=allowed_projects)

    # Skill discovery & routing
    def sync_skills(self, skill_dirs: Optional[List[str]] = None, project: str = "global", verbose: bool = True) -> Dict[str, int]:
        return self.skill_router.sync_skills(skill_dirs=skill_dirs, project=project, verbose=verbose)

    def route_skills(self, prompt: str, top_k: int = 3,
                     project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.skill_router.route_skills(prompt, top_k=top_k, project=project, allowed_projects=allowed_projects)

    # Context Memory
    def save_context(self, session_id: str, summary: str, project: Optional[str] = None,
                     title: Optional[str] = None, active_files: Optional[List[str]] = None,
                     open_tasks: Optional[List[str]] = None, full_notes: Optional[str] = None) -> Dict[str, Any]:
        return self.context_memory.save_context(session_id, summary, project=project,
                                                title=title, active_files=active_files,
                                                open_tasks=open_tasks, full_notes=full_notes)

    def get_context(self, session_id: Optional[str] = None, project: Optional[str] = None,
                    allowed_projects: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        return self.context_memory.get_context(session_id=session_id, project=project, allowed_projects=allowed_projects)

    def list_contexts(self, project: Optional[str] = None, allowed_projects: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return self.context_memory.list_contexts(project=project, allowed_projects=allowed_projects)

    def query_contexts(self, query_text: str, project: Optional[str] = None,
                       allowed_projects: Optional[List[str]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        return self.context_memory.query_contexts(query_text, project=project, allowed_projects=allowed_projects, top_k=top_k)

    # Output Formatters
    @staticmethod
    def format_as_stub(data: Dict[str, Any]) -> str:
        return format_as_stub(data)

    @staticmethod
    def format_as_sexp(data: Dict[str, Any]) -> str:
        return format_as_sexp(data)

    # Analyzer Engine (RFC: tokenopt-analyzer v2)
    def _store_analysis_ref(self, filepath: str, name: str, start_line: int, end_line: int, kind: str, body_text: str) -> str:
        return self.analyzer_engine._store_analysis_ref(filepath, name, start_line, end_line, kind, body_text)

    def expand_ref(self, ref_id: str, depth: str = "full", span: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
        return self.analyzer_engine.expand_ref(ref_id, depth=depth, span=span)

    def _diff_spans(self, filepath: str, current_content: str, since: Optional[str]) -> Dict[str, Any]:
        return self.analyzer_engine._diff_spans(filepath, current_content, since=since)

    def analyze_file(self, filepath: str, depth: str = "structure",
                     span: Optional[Tuple[int, int]] = None,
                     focus: Optional[str] = None,
                     q: Optional[str] = None,
                     since: Optional[str] = None,
                     ctx_lines: int = 10,
                     bypass_cache: bool = False,
                     no_cache: bool = False) -> Dict[str, Any]:
        return self.analyzer_engine.analyze_file(filepath, depth=depth, span=span, focus=focus, q=q,
                                                since=since, ctx_lines=ctx_lines,
                                                bypass_cache=bypass_cache, no_cache=no_cache)

    def analyze_batch(self, targets: List[str], depth: str = "structure",
                      q: Optional[str] = None, focus: Optional[str] = None,
                      span: Optional[Tuple[int, int]] = None,
                      since: Optional[str] = None,
                      max_out: Optional[int] = None,
                      cursor: Optional[str] = None,
                      ctx_lines: int = 10) -> Dict[str, Any]:
        return self.analyzer_engine.analyze_batch(targets, depth=depth, q=q, focus=focus, span=span,
                                                 since=since, max_out=max_out, cursor=cursor, ctx_lines=ctx_lines)

    def locate_targets(self, q: str, scope: str = ".", k: int = 5) -> List[Dict[str, Any]]:
        return self.analyzer_engine.locate_targets(q=q, scope=scope, k=k)


def run_watch(db_path: str, target_dir: str, interval: float = 2.0):
    return _run_watch(VectorDB, db_path, target_dir, interval)
