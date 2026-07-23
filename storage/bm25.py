"""
storage/bm25.py
BM25S-backed sparse keyword search index with persistence.
Maintains two independent indexes: "documents" and "vault".
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import bm25s

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INDEX_DOCUMENTS = "documents"
INDEX_VAULT     = "vault"
VALID_INDEXES   = {INDEX_DOCUMENTS, INDEX_VAULT}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BM25Result:
    chroma_id:  str
    text:       str
    metadata:   dict
    score:      float
    index:      str     # "documents" | "vault"

# ---------------------------------------------------------------------------
# Internal index state
# ---------------------------------------------------------------------------

@dataclass
class _IndexState:
    """Holds the BM25S retriever plus parallel metadata list."""
    retriever:  bm25s.BM25 | None
    chroma_ids: list[str]       # parallel to corpus
    texts:      list[str]       # parallel to corpus
    metadatas:  list[dict]      # parallel to corpus

    @property
    def size(self) -> int:
        return len(self.chroma_ids)

# ---------------------------------------------------------------------------
# BM25 store
# ---------------------------------------------------------------------------

class BM25Store:
    """
    Sparse keyword search using BM25S.

    Two independent in-memory indexes are maintained (documents + vault).
    Each index is persisted to disk after every write so restarts are safe.

    Usage:
        store = BM25Store("data/bm25")
        store.init()
        store.add(
            index="documents",
            chroma_id="uuid-1",
            text="Chapter 1 content...",
            metadata={"source": "report.pdf", "heading_path": "Chapter 1"},
        )
        results = store.query(index="documents", query="authentication errors", top_k=10)
    """

    def __init__(self, base_path: str = "data/bm25"):
        self.base_path = Path(base_path)
        self._indexes: dict[str, _IndexState] = {}

    # ------------------------------------------------------------------
    # Init / persistence
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Load persisted indexes from disk, or create empty ones."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        for name in VALID_INDEXES:
            state = self._load(name)
            self._indexes[name] = state
            logger.info(
                "BM25 index '%s' ready — %d existing entries",
                name, state.size,
            )

    def _index_dir(self, name: str) -> Path:
        return self.base_path / name

    def _meta_path(self, name: str) -> Path:
        return self._index_dir(name) / "meta.json"

    def _load(self, name: str) -> _IndexState:
        """Load index from disk. Returns empty state if not found."""
        meta_path = self._meta_path(name)
        index_dir = self._index_dir(name)
        index_dir.mkdir(parents=True, exist_ok=True)

        if not meta_path.exists():
            return _IndexState(
                retriever=None,
                chroma_ids=[],
                texts=[],
                metadatas=[],
            )

        try:
            with open(meta_path) as f:
                meta = json.load(f)

            retriever = bm25s.BM25.load(str(index_dir), load_corpus=False)

            return _IndexState(
                retriever=retriever,
                chroma_ids=meta["chroma_ids"],
                texts=meta["texts"],
                metadatas=meta["metadatas"],
            )
        except Exception as e:
            logger.warning(
                "Failed to load BM25 index '%s': %s. Starting fresh.", name, e
            )
            return _IndexState(
                retriever=None,
                chroma_ids=[],
                texts=[],
                metadatas=[],
            )

    def _save(self, name: str) -> None:
        """Persist index to disk."""
        state = self._indexes[name]
        index_dir = self._index_dir(name)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Save BM25S retriever (handles its own format)
        if state.retriever is not None:
            state.retriever.save(str(index_dir))

        # Save parallel metadata as JSON
        with open(self._meta_path(name), "w") as f:
            json.dump({
                "chroma_ids": state.chroma_ids,
                "texts":      state.texts,
                "metadatas":  state.metadatas,
            }, f)

    def _rebuild(self, name: str) -> None:
        """
        Rebuild the BM25S retriever from the current corpus.
        Must be called after any add/delete operation.
        """
        state = self._indexes[name]
        if not state.texts:
            state.retriever = None
            return

        tokenized = bm25s.tokenize(state.texts, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(tokenized)
        state.retriever = retriever

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _check(self, name: str) -> _IndexState:
        if name not in VALID_INDEXES:
            raise ValueError(
                f"Unknown index '{name}'. Valid: {sorted(VALID_INDEXES)}"
            )
        if name not in self._indexes:
            raise RuntimeError("BM25Store not initialised. Call store.init() first.")
        return self._indexes[name]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(
        self,
        index: str,
        chroma_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        """
        Add or update a single entry.
        If chroma_id already exists, the old entry is replaced.
        """
        state = self._check(index)

        # Replace if chroma_id already exists
        if chroma_id in state.chroma_ids:
            self.delete(index, [chroma_id])
            state = self._indexes[index]   # re-fetch after delete

        state.chroma_ids.append(chroma_id)
        state.texts.append(text)
        state.metadatas.append(metadata)

        self._rebuild(index)
        self._save(index)

    def add_batch(
        self,
        index: str,
        chroma_ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Batch add — much faster than individual adds for large ingestion.
        Existing chroma_ids are replaced.
        """
        if not (len(chroma_ids) == len(texts) == len(metadatas)):
            raise ValueError("All batch lists must have the same length.")

        state = self._check(index)

        # Remove any existing entries with these ids first
        existing = set(state.chroma_ids)
        to_replace = [cid for cid in chroma_ids if cid in existing]
        if to_replace:
            self.delete(index, to_replace)
            state = self._indexes[index]

        state.chroma_ids.extend(chroma_ids)
        state.texts.extend(texts)
        state.metadatas.extend(metadatas)

        self._rebuild(index)
        self._save(index)
        logger.debug("Added %d entries to BM25 index '%s'", len(chroma_ids), index)

    def delete(self, index: str, chroma_ids: list[str]) -> int:
        """
        Remove entries by chroma_id list.
        Returns number of entries actually removed.
        """
        if not chroma_ids:
            return 0

        state = self._check(index)
        id_set = set(chroma_ids)
        before = state.size

        keep = [
            i for i, cid in enumerate(state.chroma_ids)
            if cid not in id_set
        ]

        state.chroma_ids = [state.chroma_ids[i] for i in keep]
        state.texts      = [state.texts[i]      for i in keep]
        state.metadatas  = [state.metadatas[i]  for i in keep]

        removed = before - state.size
        if removed > 0:
            self._rebuild(index)
            self._save(index)
            logger.debug(
                "Deleted %d entries from BM25 index '%s'", removed, index
            )
        return removed

    def delete_by_doc_id(self, index: str, doc_id: str) -> int:
        """
        Remove all entries belonging to a document.
        Matches on metadata["doc_id"].
        """
        state = self._check(index)
        to_delete = [
            cid for cid, meta in zip(state.chroma_ids, state.metadatas)
            if meta.get("doc_id") == doc_id
        ]
        return self.delete(index, to_delete)

    def reset(self, index: str) -> None:
        """Wipe an index completely. Use for testing or full re-index."""
        self._check(index)
        self._indexes[index] = _IndexState(
            retriever=None,
            chroma_ids=[],
            texts=[],
            metadatas=[],
        )
        self._save(index)
        logger.warning("BM25 index '%s' has been reset.", index)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def query(
        self,
        index: str,
        query: str,
        top_k: int = 10,
    ) -> list[BM25Result]:
        """
        Keyword search using BM25.

        Args:
            index:  "documents" or "vault"
            query:  raw query string (tokenised internally)
            top_k:  number of results to return

        Returns:
            List of BM25Result sorted by score descending (best first).
            Returns empty list if index is empty or query has no matches.
        """
        state = self._check(index)

        if state.retriever is None or state.size == 0:
            return []

        query_clean = _clean_query(query)
        if not query_clean:
            return []

        effective_k = min(top_k, state.size)

        try:
            tokenized_query = bm25s.tokenize(
                [query_clean], stopwords="en", show_progress=False
            )
            results, scores = state.retriever.retrieve(
                tokenized_query, k=effective_k
            )
        except Exception as e:
            logger.warning("BM25 query failed: %s", e)
            return []

        output: list[BM25Result] = []
        for idx, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            output.append(BM25Result(
                chroma_id=state.chroma_ids[idx],
                text=state.texts[idx],
                metadata=state.metadatas[idx],
                score=float(score),
                index=index,
            ))

        return output

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self, index: str) -> int:
        """Return the number of entries in an index."""
        return self._check(index).size

    def contains(self, index: str, chroma_id: str) -> bool:
        """Check whether a chroma_id exists in an index."""
        return chroma_id in self._check(index).chroma_ids


# ---------------------------------------------------------------------------
# Internal: query cleaning
# ---------------------------------------------------------------------------

def _clean_query(query: str) -> str:
    """Lowercase and strip punctuation for more reliable BM25 tokenisation."""
    query = query.lower().strip()
    query = re.sub(r"[^\w\s]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    return query


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        store = BM25Store(base_path=tmp)
        store.init()

        assert store.count(INDEX_DOCUMENTS) == 0
        assert store.count(INDEX_VAULT) == 0
        print("Indexes initialised and empty.")

        # Add entries
        store.add_batch(
            index=INDEX_DOCUMENTS,
            chroma_ids=["c1", "c2", "c3"],
            texts=[
                "Authentication tokens expire after 24 hours.",
                "Rate limits apply to all API endpoints.",
                "Error codes are documented in Appendix B.",
            ],
            metadatas=[
                {"doc_id": "doc-1", "source": "api.pdf",
                 "heading_path": "Authentication"},
                {"doc_id": "doc-1", "source": "api.pdf",
                 "heading_path": "Rate Limits"},
                {"doc_id": "doc-1", "source": "api.pdf",
                 "heading_path": "Error Codes"},
            ],
        )
        assert store.count(INDEX_DOCUMENTS) == 3
        print("Batch add: OK")

        # Query
        results = store.query(INDEX_DOCUMENTS, "authentication token", top_k=5)
        assert len(results) > 0
        assert results[0].index == INDEX_DOCUMENTS
        print(f"Query 'authentication token': top result = '{results[0].text[:50]}...'")
        print(f"  Score: {results[0].score:.4f}")

        # Vault is independent
        store.add(
            index=INDEX_VAULT,
            chroma_id="v1",
            text="My research note on system design.",
            metadata={"doc_id": "vault-note-1", "source": "Research/design.md"},
        )
        assert store.count(INDEX_VAULT) == 1
        assert store.count(INDEX_DOCUMENTS) == 3
        print("Index isolation: OK")

        # Persistence: save and reload
        store2 = BM25Store(base_path=tmp)
        store2.init()
        assert store2.count(INDEX_DOCUMENTS) == 3
        assert store2.count(INDEX_VAULT) == 1
        results2 = store2.query(INDEX_DOCUMENTS, "rate limit", top_k=3)
        assert len(results2) > 0
        print(f"Persistence reload: OK — {store2.count(INDEX_DOCUMENTS)} entries")

        # Delete by doc_id
        removed = store2.delete_by_doc_id(INDEX_DOCUMENTS, "doc-1")
        assert removed == 3
        assert store2.count(INDEX_DOCUMENTS) == 0
        print(f"delete_by_doc_id: removed {removed} entries, OK")

        # Empty query returns nothing gracefully
        results3 = store2.query(INDEX_DOCUMENTS, "anything", top_k=5)
        assert results3 == []
        print("Empty index query: graceful empty return OK")

        # Reset
        store2.reset(INDEX_VAULT)
        assert store2.count(INDEX_VAULT) == 0
        print("Reset: OK")

        print("\nAll BM25Store assertions passed.")
