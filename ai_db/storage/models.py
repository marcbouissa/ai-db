"""Domain Data Transfer Objects (DTOs) for the ai-db storage layer.

Decouples storage records from concrete database row representations
(e.g., sqlite3.Row, PyMySQL dicts) using strongly typed, memory-efficient
slots-based dataclasses.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass(slots=True)
class FileRecord:
    """Record representing a tracked source file in the index."""
    filepath: str
    sha256: str
    last_modified: float
    chunk_count: int
    project: str = "global"


@dataclass(slots=True)
class ChunkRecord:
    """Record representing a code block or section chunk.
    
    Content is stored uncompressed in the DTO; the backend handles
    transparent compression and decompression.
    """
    filepath: str
    chunk_type: str
    name: str
    start_line: int
    end_line: int
    content: str
    project: str = "global"
    id: Optional[int] = None
    qualified_name: str = ""
    language: str = ""
    token_count: int = 0
    content_hash: str = ""
    parent_id: Optional[int] = None
    # Index of the parent within the list passed to insert/replace (transient).
    parent_index: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.content_hash:
            import hashlib
            self.content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if not self.qualified_name:
            self.qualified_name = self.name


@dataclass(slots=True)
class SymbolRecord:
    """Record representing an extracted code symbol (class, function, method)."""
    name: str
    symbol_type: str
    filepath: str
    line: int
    signature: Optional[str] = None
    project: str = "global"
    id: Optional[int] = None


@dataclass(slots=True)
class SymbolRefRecord:
    """Record representing a cross-reference call site, import, or inheritance."""
    caller_filepath: str
    caller_name: str
    caller_line: int
    callee_name: str
    ref_type: str  # 'call', 'import', 'inherit'
    project: str = "global"
    id: Optional[int] = None


@dataclass(slots=True)
class AnnotationRecord:
    """Record representing inline annotations (TODO, FIXME, HACK, docstrings)."""
    filepath: str
    line: int
    kind: str  # 'todo', 'fixme', 'hack', 'note', 'xxx', 'docstring'
    content: str
    symbol: Optional[str] = None
    project: str = "global"
    id: Optional[int] = None


@dataclass(slots=True)
class SyntaxErrorRecord:
    """Record representing a parser syntax error or diagnostic."""
    filepath: str
    line: int
    col: int
    message: str
    timestamp: float
    project: str = "global"


@dataclass(slots=True)
class SkillRecord:
    """Record representing a discovered assistant skill and routing triggers."""
    name: str
    description: str
    filepath: str
    triggers: str
    sha256: str
    last_modified: float
    content: str = ""
    project: str = "global"


@dataclass(slots=True)
class ContextRecord:
    """Record representing a preserved conversation session context snapshot."""
    session_id: str
    project: str = "global"
    title: Optional[str] = None
    summary: str = ""
    active_files: List[str] = field(default_factory=list)
    open_tasks: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    full_notes: str = ""
    id: Optional[int] = None


@dataclass(slots=True)
class AnalysisRefRecord:
    """Record representing an opaque token-optimized progressive disclosure ref."""
    ref_id: str
    filepath: str
    name: str
    start_line: int
    end_line: int
    kind: str
    body_text: str
    timestamp: float


@dataclass(slots=True)
class SearchResult:
    """Ranked search hit returned from chunk full-text BM25 queries."""
    chunk_id: int
    filepath: str
    name: str
    chunk_type: str
    project: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    qualified_name: str = ""
    language: str = ""
    parent_id: Optional[int] = None
    # Per-stage ranking signals, e.g. {"bm25_rank": 1, "vec_rank": 3, "rrf": 0.03}
    signals: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedFile:
    """Everything extracted from one source file, produced by a parser worker process
    and written by the single writer (the main process)."""
    filepath: str
    sha256: str
    last_modified: float
    project: str
    chunks: List[ChunkRecord] = field(default_factory=list)
    symbols: List[SymbolRecord] = field(default_factory=list)
    refs: List[SymbolRefRecord] = field(default_factory=list)
    annotations: List[AnnotationRecord] = field(default_factory=list)
    syntax_error: Optional[SyntaxErrorRecord] = None
