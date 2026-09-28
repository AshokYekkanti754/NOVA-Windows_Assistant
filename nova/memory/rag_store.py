"""
nova.memory.rag_store
------------------------
Semantic search over documents/code/notes the user has fed NOVA. Unlike
LongTermMemory (exact/keyword lookup), this is looked up by *meaning* --
an embedding model turns text into a vector, and ChromaDB finds the
nearest stored vectors.

Both the embedding function and the ChromaDB collection are injectable,
so tests can run without downloading sentence-transformers weights or
starting a real ChromaDB instance.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from nova.config.loader import get_config
from nova.logging_setup import get_logger

log = get_logger("memory.rag")

EmbedFn = Callable[[List[str]], List[List[float]]]


def load_sentence_transformer_embedder(model_name: str) -> EmbedFn:
    """Loads the real sentence-transformers model and wraps it as a plain text -> vectors function."""
    from sentence_transformers import SentenceTransformer  # heavy import, deferred

    model = SentenceTransformer(model_name)

    def _embed(texts: List[str]) -> List[List[float]]:
        return model.encode(texts).tolist()

    return _embed


class RagKnowledgeStore:
    def __init__(
        self,
        config: Optional[dict] = None,
        embed_fn: Optional[EmbedFn] = None,
        collection=None,
    ):
        cfg = config or get_config()
        mem_cfg = cfg["brain"]["memory"]

        self.persist_directory: str = mem_cfg["persist_directory"]
        self.embedding_model_name: str = mem_cfg["embedding_model"]
        self._embed_fn = embed_fn  # injected fake in tests
        self._collection = collection  # injected fake in tests

    def _ensure_embed_fn(self) -> EmbedFn:
        if self._embed_fn is None:
            self._embed_fn = load_sentence_transformer_embedder(self.embedding_model_name)
        return self._embed_fn

    def _ensure_collection(self):
        if self._collection is None:
            import chromadb  # heavy import, deferred

            client = chromadb.PersistentClient(path=self.persist_directory)
            self._collection = client.get_or_create_collection("nova_knowledge")
            log.info("Opened ChromaDB collection at %s", self.persist_directory)
        return self._collection

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        embed_fn = self._ensure_embed_fn()
        collection = self._ensure_collection()
        embedding = embed_fn([text])[0]
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
        )
        log.info("Indexed document '%s' (%d chars)", doc_id, len(text))

    def query(self, text: str, k: int = 3) -> List[str]:
        embed_fn = self._ensure_embed_fn()
        collection = self._ensure_collection()
        embedding = embed_fn([text])[0]
        results = collection.query(query_embeddings=[embedding], n_results=k)
        documents = results.get("documents", [[]])
        return documents[0] if documents else []
