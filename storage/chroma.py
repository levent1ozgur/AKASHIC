"""
storage/chroma.py
ChromaDB client wrapper.
Manages two named collections: "documents" and "vault".
All pipeline components interact with ChromaDB through this module only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION_DOCUMENTS = "documents"
COLLECTION_VAULT     = "vault"

VALID_COLLECTIONS = {COLLECTION_DOCUMENTS, COLLECTION_VAULT}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    chunk_id:     str
    chroma_id:    str
    text:         str
    metadata:     dict
    distance:     float          # lower = more similar (L2) or higher = more similar (cosine)
    collection:   str            # "documents" | "vault"


# ---------------------------------------------------------------------------
# ChromaDB wrapper
# ---------------------------------------------------------------------------

class ChromaStore:
    """
    Thin wrapper around ChromaDB.

    Provides:
      - Persistent storage at data/vector_db/{documents,vault}
      - Upsert, query, delete, and count operations
      - Consistent metadata schema enforcement
      - Collection-level isolation between documents and vault

    Usage:
        store = ChromaStore("data/vector_db")
        store.init()
        store.upsert(
            collection="documents",
            chroma_id="uuid-1",
            embedding=[0.1, 0.2, ...],
            text="Chapter 1 content...",
            metadata={"source": "report.pdf", "heading_path": "Chapter 1"},
        )
        results = store.query(
            collection="documents",
            query_embedding=[0.1, 0.2, ...],
            top_k=10,
        )
    """

    def __init__(self, base_path: str = "data/vector_db"):
        self.base_path = Path(base_path)
        self._clients: dict[str, chromadb.PersistentClient] = {}
        self._collections: dict[str, chromadb.Collection] = {}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init(self) -> None:
        """
        Create persistent ChromaDB clients and collections.
        Safe to call on every startup — collections are created if absent.
        """
        for name in VALID_COLLECTIONS:
            persist_dir = self.base_path / name
            persist_dir.mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,   # no phone-home
                    allow_reset=True,
                ),
            )
            self._clients[name] = client

            # cosine distance is more reliable than L2 for text embeddings
            collection = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[name] = collection
            logger.info(
                "ChromaDB collection '%s' ready — %d existing entries",
                name, collection.count(),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collection(self, name: str) -> chromadb.Collection:
        if name not in VALID_COLLECTIONS:
            raise ValueError(
                f"Unknown collection '{name}'. "
                f"Valid options: {sorted(VALID_COLLECTIONS)}"
            )
        if name not in self._collections:
            raise RuntimeError(
                f"ChromaStore not initialised. Call store.init() first."
            )
        return self._collections[name]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(
        self,
        collection: str,
        chroma_id: str,
        embedding: list[float],
        text: str,
        metadata: dict,
    ) -> None:
        """
        Insert or update a single entry.
        If chroma_id already exists, the entry is overwritten.

        metadata must be flat (str/int/float/bool values only —
        ChromaDB does not support nested dicts or lists as metadata values).
        """
        safe_meta = _flatten_metadata(metadata)
        self._collection(collection).upsert(
            ids=[chroma_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[safe_meta],
        )

    def upsert_batch(
        self,
        collection: str,
        chroma_ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Batch upsert — significantly faster than individual upserts
        for large ingestion runs.
        """
        if not (len(chroma_ids) == len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("All batch lists must have the same length.")

        safe_metas = [_flatten_metadata(m) for m in metadatas]
        self._collection(collection).upsert(
            ids=chroma_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=safe_metas,
        )
        logger.debug(
            "Upserted %d entries to '%s'", len(chroma_ids), collection
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        where: Optional[dict] = None,
    ) -> list[QueryResult]:
        """
        Semantic search against a collection.

        Args:
            collection:       "documents" or "vault"
            query_embedding:  embedding vector for the query
            top_k:            number of results to return
            where:            optional ChromaDB metadata filter,
                              e.g. {"source": "report.pdf"}

        Returns:
            List of QueryResult sorted by relevance (best first).
            With cosine distance: lower distance = more similar.
        """
        coll = self._collection(collection)
        count = coll.count()
        if count == 0:
            return []

        # Clamp top_k to available entries to avoid ChromaDB errors
        effective_k = min(top_k, count)

        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw = coll.query(**kwargs)

        results: list[QueryResult] = []
        ids       = raw["ids"][0]
        docs      = raw["documents"][0]
        metas     = raw["metadatas"][0]
        distances = raw["distances"][0]

        for chroma_id, text, meta, dist in zip(ids, docs, metas, distances):
            results.append(QueryResult(
                chunk_id=meta.get("chunk_id", chroma_id),
                chroma_id=chroma_id,
                text=text,
                metadata=meta,
                distance=dist,
                collection=collection,
            ))

        return results

    def get_by_id(
        self,
        collection: str,
        chroma_id: str,
    ) -> Optional[QueryResult]:
        """Fetch a single entry by its chroma_id. Returns None if not found."""
        coll = self._collection(collection)
        raw = coll.get(
            ids=[chroma_id],
            include=["documents", "metadatas"],
        )
        if not raw["ids"]:
            return None
        return QueryResult(
            chunk_id=raw["metadatas"][0].get("chunk_id", chroma_id),
            chroma_id=chroma_id,
            text=raw["documents"][0],
            metadata=raw["metadatas"][0],
            distance=0.0,
            collection=collection,
        )

    def get_by_metadata(
        self,
        collection: str,
        where: dict,
        limit: int = 100,
    ) -> list[QueryResult]:
        """
        Fetch entries matching a metadata filter without a query vector.
        Useful for graph expansion (fetching linked vault notes by path).
        """
        coll = self._collection(collection)
        if coll.count() == 0:
            return []

        raw = coll.get(
            where=where,
            limit=limit,
            include=["documents", "metadatas"],
        )
        results = []
        for chroma_id, text, meta in zip(
            raw["ids"], raw["documents"], raw["metadatas"]
        ):
            results.append(QueryResult(
                chunk_id=meta.get("chunk_id", chroma_id),
                chroma_id=chroma_id,
                text=text,
                metadata=meta,
                distance=0.0,
                collection=collection,
            ))
        return results

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete(self, collection: str, chroma_ids: list[str]) -> None:
        """Delete entries by chroma_id list."""
        if not chroma_ids:
            return
        self._collection(collection).delete(ids=chroma_ids)
        logger.debug("Deleted %d entries from '%s'", len(chroma_ids), collection)

    def delete_by_metadata(self, collection: str, where: dict) -> None:
        """
        Delete all entries matching a metadata filter.
        Used when re-indexing a document: delete old chunks before writing new ones.
        Example: where={"doc_id": "some-uuid"}
        """
        coll = self._collection(collection)
        if coll.count() == 0:
            return
        coll.delete(where=where)
        logger.debug(
            "Deleted entries matching %s from '%s'", where, collection
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self, collection: str) -> int:
        """Return the number of entries in a collection."""
        return self._collection(collection).count()

    def reset(self, collection: str) -> None:
        """
        Wipe and recreate a collection.
        Use only for testing or full re-index operations.
        """
        client = self._clients[collection]
        client.delete_collection(collection)
        new_coll = client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        self._collections[collection] = new_coll
        logger.warning("Collection '%s' has been reset.", collection)


# ---------------------------------------------------------------------------
# Internal: metadata flattening
# ---------------------------------------------------------------------------

def _flatten_metadata(meta: dict) -> dict:
    """
    ChromaDB requires flat metadata (str/int/float/bool values only).
    Lists and nested dicts are serialised to strings.
    None values are replaced with empty string (ChromaDB rejects None).
    """
    flat = {}
    for k, v in meta.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        elif isinstance(v, list):
            # Serialise lists as comma-separated strings
            flat[k] = ",".join(str(item) for item in v)
        else:
            flat[k] = str(v)
    return flat


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os, math

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        store = ChromaStore(base_path=tmp)
        store.init()

        # Confirm both collections exist and are empty
        assert store.count(COLLECTION_DOCUMENTS) == 0
        assert store.count(COLLECTION_VAULT) == 0
        print("Collections initialised and empty.")

        # Create a tiny fake embedding (4 dimensions for testing)
        def fake_embedding(seed: float) -> list[float]:
            raw = [seed, seed * 0.5, seed * 0.25, seed * 0.1]
            norm = math.sqrt(sum(x**2 for x in raw))
            return [x / norm for x in raw]

        # Upsert a single entry
        store.upsert(
            collection=COLLECTION_DOCUMENTS,
            chroma_id="doc-chunk-1",
            embedding=fake_embedding(1.0),
            text="Chapter 1: Introduction to the system.",
            metadata={
                "chunk_id": "uuid-1",
                "doc_id": "doc-uuid-1",
                "source": "report.pdf",
                "heading_path": "Introduction",
                "page_estimate": 1,
                "token_count": 42,
                "tags": ["intro", "overview"],   # list → flattened to string
                "source_type": "document",
            },
        )
        assert store.count(COLLECTION_DOCUMENTS) == 1
        print("Single upsert: OK")

        # Batch upsert
        store.upsert_batch(
            collection=COLLECTION_DOCUMENTS,
            chroma_ids=["doc-chunk-2", "doc-chunk-3"],
            embeddings=[fake_embedding(0.8), fake_embedding(0.6)],
            texts=["Chapter 2: Setup.", "Chapter 3: Usage."],
            metadatas=[
                {"chunk_id": "uuid-2", "doc_id": "doc-uuid-1",
                 "source": "report.pdf", "heading_path": "Setup",
                 "source_type": "document"},
                {"chunk_id": "uuid-3", "doc_id": "doc-uuid-1",
                 "source": "report.pdf", "heading_path": "Usage",
                 "source_type": "document"},
            ],
        )
        assert store.count(COLLECTION_DOCUMENTS) == 3
        print("Batch upsert: OK")

        # Query
        results = store.query(
            collection=COLLECTION_DOCUMENTS,
            query_embedding=fake_embedding(0.95),
            top_k=2,
        )
        assert len(results) == 2
        assert results[0].collection == COLLECTION_DOCUMENTS
        print(f"Query returned {len(results)} results. Top: '{results[0].text}'")

        # get_by_id
        entry = store.get_by_id(COLLECTION_DOCUMENTS, "doc-chunk-1")
        assert entry is not None
        assert entry.text == "Chapter 1: Introduction to the system."
        print("get_by_id: OK")

        # get_by_metadata
        by_meta = store.get_by_metadata(
            COLLECTION_DOCUMENTS,
            where={"doc_id": "doc-uuid-1"},
        )
        assert len(by_meta) == 3
        print(f"get_by_metadata: returned {len(by_meta)} entries")

        # Vault collection is independent
        store.upsert(
            collection=COLLECTION_VAULT,
            chroma_id="vault-note-1",
            embedding=fake_embedding(0.7),
            text="My research note about architecture.",
            metadata={"chunk_id": "v-uuid-1", "source": "Research/arch.md",
                      "source_type": "vault"},
        )
        assert store.count(COLLECTION_VAULT) == 1
        assert store.count(COLLECTION_DOCUMENTS) == 3   # unchanged
        print("Collection isolation: OK")

        # Delete by metadata
        store.delete_by_metadata(
            COLLECTION_DOCUMENTS,
            where={"doc_id": "doc-uuid-1"},
        )
        assert store.count(COLLECTION_DOCUMENTS) == 0
        print("delete_by_metadata: OK")

        # Reset
        store.reset(COLLECTION_VAULT)
        assert store.count(COLLECTION_VAULT) == 0
        print("reset: OK")

        print("\nAll ChromaStore assertions passed.")
