"""
nova.memory.context_store
----------------------------
Short-term context: the current conversation's turns, task state, etc.
Session-scoped and safe to clear anytime -- this is deliberately separate
from LongTermMemory so clearing a conversation never risks deleting a
stored user preference.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional

from nova.config.loader import get_config
from nova.logging_setup import get_logger

log = get_logger("memory.context")


class ShortTermContext:
    def __init__(self, config: Optional[dict] = None, db_path: Optional[str] = None):
        cfg = config or get_config()
        mem_cfg = cfg["brain"]["memory"]
        path = db_path or mem_cfg.get("short_term_db_path", "./data/short_term.db")

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO turns (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        self._conn.commit()

    def get_recent(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        cursor = self._conn.execute(
            "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cursor.fetchall()
        rows.reverse()  # oldest first, so it reads as a normal conversation
        return [{"role": role, "content": content} for role, content in rows]

    def clear_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
        self._conn.commit()
        log.info("Cleared short-term context for session '%s'", session_id)

    def close(self) -> None:
        self._conn.close()
