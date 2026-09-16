import json
import time
import sqlite3
from typing import Any, Optional

def get_session_state(conn: sqlite3.Connection, key: str) -> Optional[Any]:
    cur = conn.cursor()
    cur.execute("SELECT value_json FROM session_state WHERE key = ?", (key,))
    row = cur.fetchone()
    if row:
        try:
            return json.loads(row["value_json"])
        except Exception:
            return None
    return None

def set_session_state(conn: sqlite3.Connection, key: str, value: Any):
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO session_state (key, value_json, updated)
               VALUES (?, ?, ?)""",
            (key, json.dumps(value), time.time())
        )
        conn.commit()
    except Exception:
        pass
