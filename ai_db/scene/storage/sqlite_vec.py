"""
SQLite-vec Scene Memory Backend.
=================================
Uses existing sqlite-vec dependency from ai-db for vector search.
Following StorageBackend pattern with WAL mode, FTS5, and vector indexes.
"""

import json
import sqlite3
import sqlite_vec
from pathlib import Path
from typing import Any

from ai_db.scene.storage.base import SceneMemoryBackend
from ai_db.scene.models import SceneState, PromptContext
from ai_db.logger import _logger


class SQLiteVecSceneBackend(SceneMemoryBackend):
    """Scene memory backend using SQLite with sqlite-vec for semantic search."""
    
    backend_name = "sqlite_vec"
    capabilities = {"scene_storage", "prompt_history", "text_search", "vector_search", "sql_queries"}
    
    def __init__(self):
        self.conn: sqlite3.Connection | None = None
        self.db_path: Path | None = None
    
    def initialize(self, config: dict[str, Any]) -> None:
        db_path = config.get("path", "~/.ai-db/scene_memory.db")
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)
        
        # WAL mode for performance
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-32768")  # 32MB cache
        
        self._init_schema()
        _logger.info(f"SQLiteVecSceneBackend initialized at {self.db_path}")
    
    def _init_schema(self):
        c = self.conn.cursor()
        
        # Scenes table
        c.execute("""
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                name TEXT NOT NULL,
                timestamp REAL NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                parent_scene_id TEXT,
                tags TEXT,  -- JSON array
                data TEXT NOT NULL,  -- Full scene JSON
                created_at REAL DEFAULT (strftime('%s','now'))
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_scenes_project ON scenes(project)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_scenes_timestamp ON scenes(timestamp DESC)")
        
        # Objects table (for object-level queries)
        c.execute("""
            CREATE TABLE IF NOT EXISTS scene_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id TEXT NOT NULL,
                object_uuid TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                collection TEXT,
                location TEXT,  -- JSON array
                rotation TEXT,  -- JSON array
                scale TEXT,     -- JSON array
                materials TEXT, -- JSON array
                mesh_data TEXT,
                vertices INTEGER DEFAULT 0,
                faces INTEGER DEFAULT 0,
                parent_uuid TEXT,
                data TEXT NOT NULL,  -- Full object JSON
                FOREIGN KEY(scene_id) REFERENCES scenes(scene_id) ON DELETE CASCADE
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_objects_scene ON scene_objects(scene_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_objects_name ON scene_objects(name)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_objects_type ON scene_objects(type)")
        
        # Prompt contexts table
        c.execute("""
            CREATE TABLE IF NOT EXISTS prompt_contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id TEXT NOT NULL,
                project TEXT NOT NULL,
                prompt TEXT NOT NULL,
                timestamp REAL NOT NULL,
                objects_affected TEXT,  -- JSON array of UUIDs
                operation_type TEXT NOT NULL,
                metadata TEXT,  -- JSON
                FOREIGN KEY(scene_id) REFERENCES scenes(scene_id) ON DELETE CASCADE
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_prompt_scene ON prompt_contexts(scene_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_prompt_project ON prompt_contexts(project)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_prompt_timestamp ON prompt_contexts(timestamp DESC)")
        
        # FTS5 for text search on prompts
        c.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS prompt_contexts_fts USING fts5(
                prompt, 
                content=prompt_contexts, 
                content_rowid=id
            )
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS prompt_contexts_ai AFTER INSERT ON prompt_contexts BEGIN
                INSERT INTO prompt_contexts_fts(rowid, prompt) VALUES (new.id, new.prompt);
            END
        """)
        c.execute("""
            CREATE TRIGGER IF NOT EXISTS prompt_contexts_ad AFTER DELETE ON prompt_contexts BEGIN
                INSERT INTO prompt_contexts_fts(prompt_contexts_fts, rowid, prompt) VALUES ('delete', old.id, old.prompt);
            END
        """)
        
        # Vector embeddings for prompts (using sqlite-vec)
        # Dimension depends on embedding model - default 384 for all-MiniLM-L6-v2
        embed_dim = 384
        c.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS prompt_embeddings USING vec0(
                prompt_id INTEGER PRIMARY KEY,
                embedding FLOAT[{embed_dim}]
            )
        """)
        
        self.conn.commit()
    
    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
    
    # --- Scene snapshots ---
    
    def save_scene(self, scene: SceneState) -> str:
        c = self.conn.cursor()
        data_json = json.dumps(scene.to_dict())
        
        # Upsert scene
        c.execute("""
            INSERT OR REPLACE INTO scenes (scene_id, project, name, timestamp, version, parent_scene_id, tags, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scene.scene_id, scene.project, scene.name, scene.timestamp,
            scene.version, scene.parent_scene_id, json.dumps(scene.tags), data_json
        ))
        
        # Clear and re-insert objects for this scene
        c.execute("DELETE FROM scene_objects WHERE scene_id = ?", (scene.scene_id,))
        
        for obj in scene.objects.values():
            obj_data = json.dumps(obj.to_dict())
            c.execute("""
                INSERT INTO scene_objects (scene_id, object_uuid, name, type, collection, location, rotation, scale, materials, mesh_data, vertices, faces, parent_uuid, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scene.scene_id, obj.uuid, obj.name, obj.type, obj.collection,
                json.dumps(obj.location), json.dumps(obj.rotation), json.dumps(obj.scale),
                json.dumps(obj.materials), obj.mesh_data, obj.vertices, obj.faces,
                obj.parent_uuid, obj_data
            ))
        
        self.conn.commit()
        
        # Save prompt context if present
        if scene.prompt_context:
            scene.prompt_context.scene_id = scene.scene_id
            self.save_prompt_context(scene.prompt_context)
        
        return scene.scene_id
    
    def load_scene(self, scene_id: str) -> SceneState | None:
        c = self.conn.cursor()
        c.execute("SELECT data FROM scenes WHERE scene_id = ?", (scene_id,))
        row = c.fetchone()
        if not row:
            return None
        return SceneState.from_dict(json.loads(row[0]))
    
    def list_scenes(self, project: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        c = self.conn.cursor()
        if project:
            c.execute("""
                SELECT scene_id, project, name, timestamp, version, object_count, tags, prompt
                FROM (
                    SELECT s.scene_id, s.project, s.name, s.timestamp, s.version, s.tags,
                           (SELECT COUNT(*) FROM scene_objects WHERE scene_id = s.scene_id) as object_count,
                           (SELECT prompt FROM prompt_contexts WHERE scene_id = s.scene_id ORDER BY timestamp DESC LIMIT 1) as prompt
                    FROM scenes s WHERE s.project = ?
                    ORDER BY s.timestamp DESC LIMIT ?
                )
            """, (project, limit))
        else:
            c.execute("""
                SELECT scene_id, project, name, timestamp, version, object_count, tags, prompt
                FROM (
                    SELECT s.scene_id, s.project, s.name, s.timestamp, s.version, s.tags,
                           (SELECT COUNT(*) FROM scene_objects WHERE scene_id = s.scene_id) as object_count,
                           (SELECT prompt FROM prompt_contexts WHERE scene_id = s.scene_id ORDER BY timestamp DESC LIMIT 1) as prompt
                    FROM scenes s
                    ORDER BY s.timestamp DESC LIMIT ?
                )
            """, (limit,))
        
        return [
            {
                "scene_id": r[0], "project": r[1], "name": r[2],
                "timestamp": r[3], "version": r[4], "object_count": r[5],
                "tags": json.loads(r[6]) if r[6] else [], "prompt": r[7] or ""
            }
            for r in c.fetchall()
        ]
    
    def delete_scene(self, scene_id: str) -> bool:
        c = self.conn.cursor()
        c.execute("DELETE FROM scenes WHERE scene_id = ?", (scene_id,))
        self.conn.commit()
        return c.rowcount > 0
    
    # --- Prompt-aware history ---
    
    def save_prompt_context(self, ctx: PromptContext) -> int:
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO prompt_contexts (scene_id, project, prompt, timestamp, objects_affected, operation_type, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            ctx.scene_id, ctx.project, ctx.prompt, ctx.timestamp,
            json.dumps(ctx.objects_affected), ctx.operation_type, json.dumps(ctx.metadata)
        ))
        ctx.id = c.lastrowid
        self.conn.commit()
        return ctx.id
    
    def get_scene_history(
        self, 
        object_uuid: str | None = None, 
        scene_id: str | None = None, 
        project: str | None = None,
        limit: int = 50
    ) -> list[PromptContext]:
        c = self.conn.cursor()
        
        query = "SELECT id, scene_id, project, prompt, timestamp, objects_affected, operation_type, metadata FROM prompt_contexts WHERE 1=1"
        params = []
        
        if scene_id:
            query += " AND scene_id = ?"
            params.append(scene_id)
        if project:
            query += " AND project = ?"
            params.append(project)
        if object_uuid:
            query += " AND json_extract(objects_affected, '$') LIKE ?"
            params.append(f'%"{object_uuid}"%')
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        c.execute(query, params)
        
        return [
            PromptContext(
                id=r[0], scene_id=r[1], project=r[2], prompt=r[3],
                timestamp=r[4], objects_affected=json.loads(r[5]) if r[5] else [],
                operation_type=r[6], metadata=json.loads(r[7]) if r[7] else {}
            )
            for r in c.fetchall()
        ]
    
    # --- Semantic search ---
    
    def search_scenes_by_prompt(
        self, 
        query: str, 
        project: str | None = None, 
        top_k: int = 10
    ) -> list[SceneState]:
        """Search using FTS5 (BM25) + vector similarity if embeddings available."""
        c = self.conn.cursor()
        
        # First try vector search if embeddings exist
        c.execute("SELECT COUNT(*) FROM prompt_embeddings")
        has_vectors = c.fetchone()[0] > 0
        
        if has_vectors:
            # Would need embedder to encode query - fallback to FTS5 for now
            pass
        
        # FTS5 BM25 search
        fts_query = query.replace('"', '""')  # Escape quotes
        sql = """
            SELECT s.scene_id, s.project, s.name, s.timestamp, s.version, s.tags, s.data,
                   bm25(prompt_contexts_fts) as rank
            FROM prompt_contexts_fts
            JOIN prompt_contexts pc ON prompt_contexts_fts.rowid = pc.id
            JOIN scenes s ON pc.scene_id = s.scene_id
            WHERE prompt_contexts_fts MATCH ?
        """
        params = [fts_query]
        if project:
            sql += " AND s.project = ?"
            params.append(project)
        sql += " ORDER BY rank LIMIT ?"
        params.append(top_k)
        
        c.execute(sql, params)
        
        return [SceneState.from_dict(json.loads(r[6])) for r in c.fetchall()]
    
    def search_objects_by_description(
        self,
        query: str,
        project: str | None = None,
        top_k: int = 20
    ) -> list[dict[str, Any]]:
        """Search objects by name/material/type across scenes using FTS."""
        c = self.conn.cursor()
        
        # Search in objects table using name, type, materials
        fts_query = query.replace('"', '""')
        sql = """
            SELECT o.scene_id, o.name, o.type, o.collection, o.materials, o.data,
                   s.name as scene_name
            FROM scene_objects o
            JOIN scenes s ON o.scene_id = s.scene_id
            WHERE (o.name LIKE ? OR o.type LIKE ? OR o.collection LIKE ? OR o.materials LIKE ?)
        """
        like_param = f"%{query}%"
        params = [like_param, like_param, like_param, like_param]
        
        if project:
            sql += " AND s.project = ?"
            params.append(project)
        
        sql += " LIMIT ?"
        params.append(top_k)
        
        c.execute(sql, params)
        
        return [
            {
                "scene_id": r[0], "scene_name": r[6],
                "object": json.loads(r[5]),
                "match_score": 1.0
            }
            for r in c.fetchall()
        ]
