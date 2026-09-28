"""
Milestone 4 tests: the three memory stores and MemoryManager's
orchestration of them. ShortTermContext/LongTermMemory use SQLite's
":memory:" mode (no files touched); RagKnowledgeStore uses a fake
embedding function and a fake in-memory collection, so no
sentence-transformers weights or ChromaDB instance are needed.

Run with: pytest tests/test_milestone4_memory.py -v
(or run this file's functions directly with plain python -- see bottom)
"""

from nova.memory.context_store import ShortTermContext
from nova.memory.long_term_store import LongTermMemory
from nova.memory.manager import MemoryManager
from nova.memory.rag_store import RagKnowledgeStore


# ---------------------------------------------------------------------------
# Fakes for RagKnowledgeStore
# ---------------------------------------------------------------------------

class FakeCollection:
    """Stands in for a ChromaDB collection. Ignores real vector similarity --
    just returns whatever was stored, most-recent-first, up to n_results."""

    def __init__(self):
        self._docs = {}  # id -> (document, metadata)

    def upsert(self, ids, embeddings, documents, metadatas):
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            self._docs[doc_id] = (doc, meta)

    def query(self, query_embeddings, n_results):
        docs = [doc for doc, _meta in list(self._docs.values())[-n_results:]]
        return {"documents": [docs]}


def fake_embed_fn(texts):
    # Real embeddings encode meaning; this fake just needs a consistent shape.
    return [[float(len(t))] for t in texts]


# ---------------------------------------------------------------------------
# ShortTermContext tests
# ---------------------------------------------------------------------------

def test_short_term_context_add_and_get_recent():
    store = ShortTermContext(db_path=":memory:")
    store.add_turn("s1", "user", "hello")
    store.add_turn("s1", "assistant", "hi there")

    turns = store.get_recent("s1")
    assert turns == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_short_term_context_isolates_sessions():
    store = ShortTermContext(db_path=":memory:")
    store.add_turn("s1", "user", "for session 1")
    store.add_turn("s2", "user", "for session 2")

    assert [t["content"] for t in store.get_recent("s1")] == ["for session 1"]
    assert [t["content"] for t in store.get_recent("s2")] == ["for session 2"]


def test_short_term_context_clear_session():
    store = ShortTermContext(db_path=":memory:")
    store.add_turn("s1", "user", "hello")
    store.clear_session("s1")

    assert store.get_recent("s1") == []


def test_short_term_context_respects_limit():
    store = ShortTermContext(db_path=":memory:")
    for i in range(5):
        store.add_turn("s1", "user", f"message {i}")

    recent = store.get_recent("s1", limit=2)
    assert [t["content"] for t in recent] == ["message 3", "message 4"]


# ---------------------------------------------------------------------------
# LongTermMemory tests
# ---------------------------------------------------------------------------

def test_long_term_memory_remember_and_recall():
    memory = LongTermMemory(db_path=":memory:")
    memory.remember("my favorite color is blue")

    assert memory.recall_all() == ["my favorite color is blue"]


def test_long_term_memory_deduplicates_identical_facts():
    memory = LongTermMemory(db_path=":memory:")
    memory.remember("my favorite color is blue")
    memory.remember("my favorite color is blue")

    assert memory.recall_all() == ["my favorite color is blue"]


def test_long_term_memory_search_by_keyword():
    memory = LongTermMemory(db_path=":memory:")
    memory.remember("my favorite color is blue")
    memory.remember("my dog's name is Rex")

    results = memory.search("favorite")
    assert results == ["my favorite color is blue"]


def test_long_term_memory_forget():
    memory = LongTermMemory(db_path=":memory:")
    memory.remember("my favorite color is blue")
    memory.forget("my favorite color is blue")

    assert memory.recall_all() == []


# ---------------------------------------------------------------------------
# RagKnowledgeStore tests
# ---------------------------------------------------------------------------

def test_rag_store_add_and_query():
    store = RagKnowledgeStore(
        config={"brain": {"memory": {"persist_directory": ":memory:", "embedding_model": "fake"}}},
        embed_fn=fake_embed_fn,
        collection=FakeCollection(),
    )
    store.add_document("doc1", "NOVA is a voice assistant for Windows.")

    results = store.query("what is NOVA?", k=1)
    assert results == ["NOVA is a voice assistant for Windows."]


# ---------------------------------------------------------------------------
# MemoryManager orchestration tests
# ---------------------------------------------------------------------------

def make_manager():
    return MemoryManager(
        short_term=ShortTermContext(db_path=":memory:"),
        long_term=LongTermMemory(db_path=":memory:"),
        rag=RagKnowledgeStore(
            config={"brain": {"memory": {"persist_directory": ":memory:", "embedding_model": "fake"}}},
            embed_fn=fake_embed_fn,
            collection=FakeCollection(),
        ),
    )


def test_memory_manager_stores_fact_on_remember_trigger():
    manager = make_manager()
    manager.record_turn("s1", "user", "remember that my favorite color is blue")

    assert manager.long_term.recall_all() == ["my favorite color is blue"]


def test_memory_manager_does_not_store_fact_without_trigger_phrase():
    manager = make_manager()
    manager.record_turn("s1", "user", "what's the weather like?")

    assert manager.long_term.recall_all() == []


def test_memory_manager_build_context_includes_known_facts():
    manager = make_manager()
    manager.long_term.remember("my favorite color is blue")

    context = manager.build_context("s1", "what's my favorite color?")

    fact_messages = [m for m in context if "Known facts" in m["content"]]
    assert len(fact_messages) == 1
    assert "my favorite color is blue" in fact_messages[0]["content"]


def test_memory_manager_build_context_includes_recent_turns():
    manager = make_manager()
    manager.record_turn("s1", "user", "hello")
    manager.record_turn("s1", "assistant", "hi there")

    context = manager.build_context("s1", "how are you?")

    assert {"role": "user", "content": "hello"} in context
    assert {"role": "assistant", "content": "hi there"} in context


if __name__ == "__main__":
    tests = [
        test_short_term_context_add_and_get_recent,
        test_short_term_context_isolates_sessions,
        test_short_term_context_clear_session,
        test_short_term_context_respects_limit,
        test_long_term_memory_remember_and_recall,
        test_long_term_memory_deduplicates_identical_facts,
        test_long_term_memory_search_by_keyword,
        test_long_term_memory_forget,
        test_rag_store_add_and_query,
        test_memory_manager_stores_fact_on_remember_trigger,
        test_memory_manager_does_not_store_fact_without_trigger_phrase,
        test_memory_manager_build_context_includes_known_facts,
        test_memory_manager_build_context_includes_recent_turns,
    ]
    for t in tests:
        t()
        print(f"{t.__name__}: PASS")
