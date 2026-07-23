"""
storage/registry.py
Document registry backed by SQLite.
Tracks every document and chunk through the ingestion pipeline.
"""

import sqlite3
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    doc_id: str
    source_path: str
    source_type: str          # "document" | "vault"
    status: str               # see STATUS_* constants below
    git_hash: Optional[str] = None
    file_hash: Optional[str] = None
    chunk_count: int = 0
    token_count: int = 0
    conversion_method: Optional[str] = None   # markitdown | ocr | passthrough
    language: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chroma_id: str
    heading_path: str
    page_estimate: Optional[int]
    token_count: int
    source_type: str          # "document" | "vault"
    chunk_index: int


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

STATUS_PENDING     = "pending"
STATUS_CONVERTING  = "converting"
STATUS_CHUNKING    = "chunking"
STATUS_EMBEDDING   = "embedding"
STATUS_INDEXED     = "indexed"
STATUS_FAILED      = "failed"

SOURCE_DOCUMENT    = "document"
SOURCE_VAULT       = "vault"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class Registry:
    """
    SQLite-backed document and chunk registry.

    Usage:
        registry = Registry("data/registry.db")
        registry.init()
        doc_id = registry.create_document("report.pdf", "document", file_hash="abc123")
        registry.update_status(doc_id, STATUS_INDEXED, chunk_count=42)
    """

    def __init__(self, db_path: str = "data/registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _conn(self):
        """Yield a connection with row_factory set, auto-commit on success."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # Enable WAL for better concurrent read performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create tables if they do not exist. Safe to call on every startup."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id            TEXT PRIMARY KEY,
                    source_path       TEXT NOT NULL,
                    source_type       TEXT NOT NULL
                                      CHECK(source_type IN ('document','vault')),
                    status            TEXT NOT NULL
                                      CHECK(status IN (
                                        'pending','converting','chunking',
                                        'embedding','indexed','failed')),
                    git_hash          TEXT,
                    file_hash         TEXT,
                    chunk_count       INTEGER DEFAULT 0,
                    token_count       INTEGER DEFAULT 0,
                    conversion_method TEXT,
                    language          TEXT,
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL,
                    error_message     TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_documents_source_path
                    ON documents(source_path);

                CREATE INDEX IF NOT EXISTS idx_documents_status
                    ON documents(status);

                CREATE INDEX IF NOT EXISTS idx_documents_file_hash
                    ON documents(file_hash);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id      TEXT PRIMARY KEY,
                    doc_id        TEXT NOT NULL
                                    REFERENCES documents(doc_id)
                                    ON DELETE CASCADE,
                    chroma_id     TEXT NOT NULL,
                    heading_path  TEXT NOT NULL DEFAULT '',
                    page_estimate INTEGER,
                    token_count   INTEGER NOT NULL DEFAULT 0,
                    source_type   TEXT NOT NULL,
                    chunk_index   INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
                    ON chunks(doc_id);

                CREATE INDEX IF NOT EXISTS idx_chunks_chroma_id
                    ON chunks(chroma_id);
            """)

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_document(
        self,
        source_path: str,
        source_type: str,
        file_hash: Optional[str] = None,
        git_hash: Optional[str] = None,
    ) -> str:
        """
        Insert a new document record in PENDING status.
        Returns the new doc_id (uuid).
        """
        doc_id = str(uuid.uuid4())
        now = self._now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, source_path, source_type, status,
                     file_hash, git_hash, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (doc_id, source_path, source_type, STATUS_PENDING,
                 file_hash, git_hash, now, now),
            )
        return doc_id

    def update_status(
        self,
        doc_id: str,
        status: str,
        *,
        chunk_count: Optional[int] = None,
        token_count: Optional[int] = None,
        conversion_method: Optional[str] = None,
        language: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update status and any optional fields on a document record."""
        fields = ["status = ?", "updated_at = ?"]
        values: list = [status, self._now()]

        if chunk_count is not None:
            fields.append("chunk_count = ?")
            values.append(chunk_count)
        if token_count is not None:
            fields.append("token_count = ?")
            values.append(token_count)
        if conversion_method is not None:
            fields.append("conversion_method = ?")
            values.append(conversion_method)
        if language is not None:
            fields.append("language = ?")
            values.append(language)
        if error_message is not None:
            fields.append("error_message = ?")
            values.append(error_message)

        values.append(doc_id)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(fields)} WHERE doc_id = ?",
                values,
            )

    def get_document(self, doc_id: str) -> Optional[DocumentRecord]:
        """Fetch a single document by doc_id. Returns None if not found."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
        if row is None:
            return None
        return DocumentRecord(**dict(row))

    def get_document_by_path(self, source_path: str) -> Optional[DocumentRecord]:
        """Fetch document by source path (most recent if duplicates exist)."""
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM documents
                WHERE source_path = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (source_path,),
            ).fetchone()
        if row is None:
            return None
        return DocumentRecord(**dict(row))

    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentRecord]:
        """
        Look up a document by its SHA-256 file hash.
        Used for deduplication on upload.
        Returns None if not found.
        """
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM documents
                WHERE file_hash = ? AND status = 'indexed'
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (file_hash,),
            ).fetchone()
        if row is None:
            return None
        return DocumentRecord(**dict(row))

    def list_documents(
        self,
        source_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[DocumentRecord]:
        """List all documents, optionally filtered by source_type or status."""
        query = "SELECT * FROM documents WHERE 1=1"
        params: list = []
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [DocumentRecord(**dict(r)) for r in rows]

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and all its chunks (CASCADE).
        Returns True if a row was deleted, False if doc_id not found.
        """
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
            )
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    def add_chunk(
        self,
        doc_id: str,
        chroma_id: str,
        heading_path: str,
        token_count: int,
        source_type: str,
        chunk_index: int,
        page_estimate: Optional[int] = None,
    ) -> str:
        """Insert a chunk record. Returns the new chunk_id."""
        chunk_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chunks
                    (chunk_id, doc_id, chroma_id, heading_path,
                     page_estimate, token_count, source_type, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, doc_id, chroma_id, heading_path,
                 page_estimate, token_count, source_type, chunk_index),
            )
        return chunk_id

    def get_chunks_for_document(self, doc_id: str) -> list[ChunkRecord]:
        """Return all chunk records for a given document, ordered by index."""
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM chunks
                WHERE doc_id = ?
                ORDER BY chunk_index ASC
                """,
                (doc_id,),
            ).fetchall()
        return [ChunkRecord(**dict(r)) for r in rows]

    def delete_chunks_for_document(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE doc_id = ?", (doc_id,)
            )
        return cursor.rowcount

    # ------------------------------------------------------------------
    # Summary / health
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return counts useful for the /status API endpoint."""
        with self._conn() as conn:
            total_docs = conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
            indexed_docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = 'indexed'"
            ).fetchone()[0]
            failed_docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE status = 'failed'"
            ).fetchone()[0]
            total_chunks = conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            vault_docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE source_type = 'vault'"
            ).fetchone()[0]
            doc_docs = conn.execute(
                "SELECT COUNT(*) FROM documents WHERE source_type = 'document'"
            ).fetchone()[0]

        return {
            "total_documents": total_docs,
            "indexed_documents": indexed_docs,
            "failed_documents": failed_docs,
            "total_chunks": total_chunks,
            "vault_documents": vault_docs,
            "uploaded_documents": doc_docs,
        }


# ---------------------------------------------------------------------------
# Utility: file hashing
# ---------------------------------------------------------------------------

def hash_file(path: str | Path, chunk_size: int = 65536) -> str:
    """Return the SHA-256 hex digest of a file. Used for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(chunk_size):
            h.update(data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Smoke test (run directly: python storage/registry.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test_registry.db")
        reg = Registry(db_path)
        reg.init()

        # Create a document
        doc_id = reg.create_document("report.pdf", SOURCE_DOCUMENT, file_hash="abc123")
        print(f"Created document: {doc_id}")

        # Update status through the pipeline stages
        reg.update_status(doc_id, STATUS_CONVERTING, conversion_method="markitdown")
        reg.update_status(doc_id, STATUS_CHUNKING)
        reg.update_status(doc_id, STATUS_EMBEDDING)

        # Add some chunks
        for i in range(3):
            reg.add_chunk(
                doc_id=doc_id,
                chroma_id=f"chroma_{i}",
                heading_path=f"Section {i+1}",
                token_count=256,
                source_type=SOURCE_DOCUMENT,
                chunk_index=i,
                page_estimate=i + 1,
            )

        reg.update_status(doc_id, STATUS_INDEXED, chunk_count=3, token_count=768)

        # Fetch and verify
        doc = reg.get_document(doc_id)
        assert doc is not None
        assert doc.status == STATUS_INDEXED
        assert doc.chunk_count == 3

        # Deduplication check
        dup = reg.get_document_by_hash("abc123")
        assert dup is not None
        assert dup.doc_id == doc_id

        # Summary
        s = reg.summary()
        assert s["total_documents"] == 1
        assert s["indexed_documents"] == 1
        assert s["total_chunks"] == 3

        # Delete
        deleted = reg.delete_document(doc_id)
        assert deleted is True
        assert reg.get_document(doc_id) is None
        # Chunks should be cascade-deleted
        chunks = reg.get_chunks_for_document(doc_id)
        assert len(chunks) == 0

        print("All assertions passed.")
        print(f"Summary: {s}")
