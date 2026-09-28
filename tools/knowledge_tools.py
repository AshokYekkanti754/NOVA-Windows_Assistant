"""
NOVA - Knowledge / Memory Tools (Step 6c)
Stack: SQLite (preferences + facts) + ChromaDB + sentence-transformers (RAG)

These are the *tool-facing* wrappers. If your Memory Manager from Step 5 already
owns these databases, point DB_PATH / CHROMA_PATH at the same locations, or replace
the bodies below with calls into your Memory Manager.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import List

from .base import tool, ok, fail, ToolResult

CATEGORY = "knowledge"

NOVA_HOME = Path.home() / ".nova"
DB_PATH = NOVA_HOME / "memory.db"
CHROMA_PATH = NOVA_HOME / "chroma"
EMBED_MODEL = "all-MiniLM-L6-v2"

NOVA_HOME.mkdir(parents=True, exist_ok=True)

_chroma = {"client": None, "docs": None, "embedder": None}


# ----------------------------------------------------------------------------
# SQLite (preferences + facts)
# ----------------------------------------------------------------------------
def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS preferences (key TEXT PRIMARY KEY, value TEXT, updated REAL)")
    con.execute("CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT UNIQUE, tag TEXT, created REAL)")
    return con


@tool("store_preference", "Remember a user preference", {"key": "e.g. favorite_music", "value": "the preference"}, CATEGORY)
def store_preference(key: str, value: str) -> ToolResult:
    with _db() as con:
        con.execute("INSERT OR REPLACE INTO preferences VALUES (?,?,?)", (key.lower().strip(), value, time.time()))
    return ok(f"Got it. I'll remember that your {key} is {value}.")


@tool("get_preference", "Recall a stored preference", {"key": "preference name"}, CATEGORY)
def get_preference(key: str) -> ToolResult:
    with _db() as con:
        row = con.execute("SELECT value FROM preferences WHERE key=?", (key.lower().strip(),)).fetchone()
        if not row:
            row = con.execute("SELECT value FROM preferences WHERE key LIKE ?", (f"%{key.lower().strip()}%",)).fetchone()
    return ok(f"Your {key} is {row[0]}.", row[0]) if row else fail(f"I don't have anything saved for {key}.")


@tool("list_preferences", "List all stored preferences", {}, CATEGORY)
def list_preferences() -> ToolResult:
    with _db() as con:
        rows = con.execute("SELECT key, value FROM preferences ORDER BY key").fetchall()
    if not rows:
        return fail("I haven't saved any preferences yet.")
    return ok("I know: " + "; ".join(f"{k} is {v}" for k, v in rows[:6]), rows)


@tool("save_fact", "Save an important fact to long-term memory", {"text": "the fact", "tag": "optional category"}, CATEGORY)
def save_fact(text: str, tag: str = "general") -> ToolResult:
    with _db() as con:
        con.execute("INSERT OR IGNORE INTO facts (text, tag, created) VALUES (?,?,?)", (text.strip(), tag, time.time()))
    return ok("Saved. I'll remember that.")


@tool("recall_facts", "Search saved facts", {"query": "keywords"}, CATEGORY)
def recall_facts(query: str) -> ToolResult:
    with _db() as con:
        rows = con.execute("SELECT text FROM facts WHERE text LIKE ? ORDER BY created DESC LIMIT 5", (f"%{query}%",)).fetchall()
    facts = [r[0] for r in rows]
    return ok("Here's what I remember: " + " ".join(facts), facts) if facts else fail(f"I don't remember anything about {query}.")


@tool("forget_fact", "Delete saved facts matching text", {"query": "keywords"}, CATEGORY, dangerous=True)
def forget_fact(query: str) -> ToolResult:
    with _db() as con:
        n = con.execute("DELETE FROM facts WHERE text LIKE ?", (f"%{query}%",)).rowcount
    return ok(f"Forgot {n} item(s).") if n else fail("Nothing matched.")


# ----------------------------------------------------------------------------
# ChromaDB (RAG)
# ----------------------------------------------------------------------------
def _collection():
    if _chroma["docs"] is None:
        import chromadb
        from sentence_transformers import SentenceTransformer
        _chroma["embedder"] = SentenceTransformer(EMBED_MODEL)
        _chroma["client"] = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _chroma["docs"] = _chroma["client"].get_or_create_collection("nova_knowledge")
    return _chroma["docs"], _chroma["embedder"]


def _chunk(text: str, size: int = 800, overlap: int = 120) -> List[str]:
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


def _read_document(p: Path) -> str:
    ext = p.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    if ext == ".docx":
        import docx  # python-docx
        return "\n".join(par.text for par in docx.Document(str(p)).paragraphs)
    return p.read_text(encoding="utf-8", errors="ignore")   # txt, md, code, notes...


@tool("ingest_document", "Add a document (pdf, docx, txt, md, code) to the knowledge base", {"path": "file path"}, CATEGORY)
def ingest_document(path: str) -> ToolResult:
    from .windows_tools import resolve_path
    p = resolve_path(path)
    if not p.is_file():
        return fail(f"I can't find {p}.")
    text = _read_document(p)
    if not text.strip():
        return fail("That file has no readable text.")
    col, emb = _collection()
    chunks = _chunk(text)
    ids = [hashlib.md5(f"{p}:{i}:{c[:40]}".encode()).hexdigest() for i, c in enumerate(chunks)]
    col.upsert(
        ids=ids,
        documents=chunks,
        embeddings=emb.encode(chunks).tolist(),
        metadatas=[{"source": p.name, "path": str(p), "chunk": i} for i in range(len(chunks))],
    )
    return ok(f"I've learned {p.name}. That's {len(chunks)} sections.")


@tool("ingest_folder", "Add every supported document in a folder", {"path": "folder path"}, CATEGORY)
def ingest_folder(path: str) -> ToolResult:
    from .windows_tools import resolve_path
    root = resolve_path(path)
    exts = {".pdf", ".docx", ".txt", ".md", ".py", ".js", ".json"}
    files = [f for f in root.rglob("*") if f.suffix.lower() in exts][:50]
    done = 0
    for f in files:
        if ingest_document(str(f)).ok:
            done += 1
    return ok(f"Learned {done} of {len(files)} documents.")


@tool("search_knowledge", "Semantic search over ingested documents (RAG)", {"query": "question or topic", "k": "results (default 3)"}, CATEGORY)
def search_knowledge(query: str, k: int = 3) -> ToolResult:
    col, emb = _collection()
    if col.count() == 0:
        return fail("My knowledge base is empty. Ask me to learn a document first.")
    res = col.query(query_embeddings=emb.encode([query]).tolist(), n_results=int(k))
    hits = [
        {"text": d, "source": m["source"], "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    context = "\n---\n".join(f"[{h['source']}] {h['text']}" for h in hits)
    return ok(f"Found {len(hits)} relevant passages.", {"context": context, "hits": hits})


@tool("knowledge_stats", "How much is in the knowledge base", {}, CATEGORY)
def knowledge_stats() -> ToolResult:
    col, _ = _collection()
    with _db() as con:
        nf = con.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        npf = con.execute("SELECT COUNT(*) FROM preferences").fetchone()[0]
    return ok(f"I have {col.count()} document sections, {nf} facts and {npf} preferences.")
