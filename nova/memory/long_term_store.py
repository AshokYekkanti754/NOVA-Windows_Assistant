"""
nova.memory.long_term_store
------------------------------
Long-term memory: durable facts/preferences that persist across sessions
indefinitely (until explicitly forgotten). Looked up by exact/keyword
match rather than semantic similarity -- for "what's my name" style
recall, a plain SQL lookup is simpler and faster than a vector search.
Semantic recall over open-ended documents is RagKnowledgeStore's job.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from nova.config.loader import get_config
from nova.logging_setup import get_logger

log = get_logger("memory.long_term")


class LongTermMemory:
    def __init__(self, config: Optional[dict] = None, db_path: Optional[str] = None):
        cfg = config or get_config()
        mem_cfg = cfg["brain"]["memory"]
        path = db_path or mem_cfg.get("long_term_db_path", "./data/long_term.db")

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def remember(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        try:
            self._conn.execute(
                "INSERT OR IGNORE INTO facts (fact, created_at) VALUES (?, ?)",
                (fact, time.time()),
            )
            self._conn.commit()
            log.info("Remembered fact: %r", fact)
        except sqlite3.Error:
            log.exception("Failed to store fact %r", fact)

    def recall_all(self) -> List[str]:
        cursor = self._conn.execute("SELECT fact FROM facts ORDER BY id ASC")
        return [row[0] for row in cursor.fetchall()]

    def search(self, keyword: str) -> List[str]:
        cursor = self._conn.execute(
            "SELECT fact FROM facts WHERE fact LIKE ? ORDER BY id ASC",
            (f"%{keyword}%",),
        )
        return [row[0] for row in cursor.fetchall()]

    def forget(self, fact: str) -> None:
        self._conn.execute("DELETE FROM facts WHERE fact = ?", (fact.strip(),))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
