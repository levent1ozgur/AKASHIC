"""
pipeline/ingestion/embedder.py
Embeds chunks via qwen3-embedding:0.6b (Ollama API),
then writes to ChromaDB and BM25S.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from pipeline.ingestion.chunker import Chunk
from storage.chroma import ChromaStore, COLLECTION_DOCUMENTS, COLLECTION_VAULT
from storage.bm25 import BM25Store, INDEX_DOCUMENTS, INDEX_VAULT
from storage.registry import Registry, STATUS_EMBEDDING, STATUS_INDEXED, STATUS_FAILED

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE    = 32
DEFAULT_OLLAMA_URL    = "http://localhost:11434"
DEFAULT_EMBED_MODEL   = "qwen3-embedding:0.6b"
MAX_RETRIES           = 3
RETRY_BACKOFF_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class EmbedResult:
    doc_id:         str
    chunks_total:   int
    chunks_embedded: int
    chunks_failed:  int
    collection:     str


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """
    Embeds a list of Chunk objects and writes them to ChromaDB + BM25S.

    Usage:
        embedder = Embedder(chroma=store, bm25=bm25, registry=registry)
        result = await embedder.embed(chunks, doc_id="uuid-1")
    """

    def __init__(
        self,
        chroma:       ChromaStore,
        bm25:         BM25Store,
        registry:     Registry,
        ollama_url:   str = DEFAULT_OLLAMA_URL,
        model:        str = DEFAULT_EMBED_MODEL,
        batch_size:   int = DEFAULT_BATCH_SIZE,
    ):
        self.chroma     = chroma
        self.bm25       = bm25
        self.registry   = registry
        self.ollama_url = ollama_url.rstrip("/")
        self.model      = model
        self.batch_size = batch_size

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def embed(self, chunks: list[Chunk], doc_id: str) -> EmbedResult:
        """
        Embed all chunks for a document and write to storage.
        Updates registry status throughout.
        Returns EmbedResult with counts.
        """
        if not chunks:
            logger.warning("No chunks to embed for doc_id=%s", doc_id)
            return EmbedResult(
                doc_id=doc_id,
                chunks_total=0,
                chunks_embedded=0,
                chunks_failed=0,
                collection="",
            )

        source_type = chunks[0].source_type
        collection  = COLLECTION_VAULT if source_type == "vault" else COLLECTION_DOCUMENTS
        bm25_index  = INDEX_VAULT      if source_type == "vault" else INDEX_DOCUMENTS

        self.registry.update_status(doc_id, STATUS_EMBEDDING)

        # Delete any existing chunks for this doc before writing new ones
        self.chroma.delete_by_metadata(collection, where={"doc_id": doc_id})
        self.bm25.delete_by_doc_id(bm25_index, doc_id)

        chunks_embedded = 0
        chunks_failed   = 0

        # Process in batches
        for batch_start in range(0, len(chunks), self.batch_size):
            batch = chunks[batch_start: batch_start + self.batch_size]
            texts = []
            for c in batch:
                if c.heading_path:
                    texts.append(f"{c.heading_path}\n{c.text}")
                elif c.section_context:
                    texts.append(f"[{c.section_context}]\n{c.text}")
                else:
                    texts.append(c.text)

            # Get embeddings from Ollama
            embeddings = self._embed_texts(texts)
            if embeddings is None:
                logger.error(
                    "Embedding failed for batch %d-%d of doc %s",
                    batch_start, batch_start + len(batch), doc_id,
                )
                chunks_failed += len(batch)
                # Mark individual chunks as failed in registry
                for chunk in batch:
                    self.registry.update_status(
                        doc_id, STATUS_FAILED,
                        error_message=f"Embedding failed for chunk {chunk.chunk_id}",
                    )
                continue

            # Verify we got the right number of embeddings
            if len(embeddings) != len(batch):
                logger.error(
                    "Embedding count mismatch: expected %d, got %d",
                    len(batch), len(embeddings),
                )
                chunks_failed += len(batch)
                continue

            # Write to ChromaDB
            chroma_ids = [c.chunk_id for c in batch]
            metadatas  = [_chunk_to_metadata(c) for c in batch]
            # Tag each chunk with the embedding model that produced it
            for md in metadatas:
                md.update({"embedding_model": self.model})

            try:
                self.chroma.upsert_batch(
                    collection=collection,
                    chroma_ids=chroma_ids,
                    embeddings=embeddings,
                    texts=texts,
                    metadatas=metadatas,
                )
            except Exception as e:
                logger.error("ChromaDB write failed: %s", e)
                chunks_failed += len(batch)
                continue

            # Write to BM25
            try:
                self.bm25.add_batch(
                    index=bm25_index,
                    chroma_ids=chroma_ids,
                    texts=texts,
                    metadatas=metadatas,
                )
            except Exception as e:
                logger.error("BM25 write failed: %s", e)
                # BM25 failure is non-fatal — ChromaDB write succeeded
                logger.warning(
                    "Chunks written to ChromaDB but BM25 failed for batch."
                )

            # Write chunk records to registry
            for chunk in batch:
                self.registry.add_chunk(
                    doc_id=doc_id,
                    chroma_id=chunk.chunk_id,
                    heading_path=chunk.heading_path,
                    token_count=chunk.token_count,
                    source_type=chunk.source_type,
                    chunk_index=chunk.chunk_index,
                    page_estimate=chunk.page_estimate,
                )

            chunks_embedded += len(batch)
            logger.debug(
                "Embedded batch %d-%d (%d chunks)",
                batch_start, batch_start + len(batch), len(batch),
            )

        # Final status
        if chunks_failed == 0:
            self.registry.update_status(
                doc_id, STATUS_INDEXED,
                chunk_count=chunks_embedded,
                token_count=sum(c.token_count for c in chunks),
            )
        else:
            self.registry.update_status(
                doc_id, STATUS_FAILED,
                error_message=f"{chunks_failed} chunks failed to embed",
                chunk_count=chunks_embedded,
            )

        logger.info(
            "Embedded doc %s: %d/%d chunks OK, %d failed → collection='%s'",
            doc_id, chunks_embedded, len(chunks), chunks_failed, collection,
        )

        return EmbedResult(
            doc_id=doc_id,
            chunks_total=len(chunks),
            chunks_embedded=chunks_embedded,
            chunks_failed=chunks_failed,
            collection=collection,
        )

    # ------------------------------------------------------------------
    # Ollama API
    # ------------------------------------------------------------------

    def _embed_texts(self, texts: list[str]) -> Optional[list[list[float]]]:
        """
        Call Ollama /api/embed and return embedding vectors.
        Retries up to MAX_RETRIES times on transient errors.
        Returns None if all retries fail.
        """
        url     = f"{self.ollama_url}/api/embed"
        payload = {"model": self.model, "input": texts}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    timeout=120.0,      # embedding can be slow for large batches
                )
                response.raise_for_status()
                data = response.json()

                embeddings = data.get("embeddings")
                if not embeddings:
                    logger.warning(
                        "Ollama returned no embeddings (attempt %d): %s",
                        attempt, data,
                    )
                    if attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue

                return embeddings

            except httpx.TimeoutException:
                logger.warning(
                    "Ollama embed timeout (attempt %d/%d)",
                    attempt, MAX_RETRIES,
                )
            except httpx.HTTPStatusError as e:
                logger.warning(
                    "Ollama embed HTTP error %d (attempt %d/%d): %s",
                    e.response.status_code, attempt, MAX_RETRIES, e,
                )
            except Exception as e:
                logger.warning(
                    "Ollama embed error (attempt %d/%d): %s",
                    attempt, MAX_RETRIES, e,
                )

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

        logger.error(
            "All %d embedding attempts failed for batch of %d texts.",
            MAX_RETRIES, len(texts),
        )
        return None

    def health_check(self) -> bool:
        """
        Verify Ollama is reachable and the embedding model is available.
        Returns True if healthy.
        """
        try:
            response = httpx.get(
                f"{self.ollama_url}/api/tags",
                timeout=5.0,
            )
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            if self.model not in models:
                logger.warning(
                    "Embedding model '%s' not found in Ollama. "
                    "Available: %s", self.model, models,
                )
                return False
            return True
        except Exception as e:
            logger.error("Ollama health check failed: %s", e)
            return False


# ---------------------------------------------------------------------------
# Metadata serialisation
# ---------------------------------------------------------------------------

def _chunk_to_metadata(chunk: Chunk) -> dict:
    """
    Convert a Chunk to a flat metadata dict for ChromaDB/BM25.
    Lists are serialised to comma-separated strings (ChromaDB requirement).
    """
    return {
        "chunk_id":      chunk.chunk_id,
        "doc_id":        chunk.doc_id,
        "source":        chunk.source,
        "source_type":   chunk.source_type,
        "chunk_index":   chunk.chunk_index,
        "heading_path":  chunk.heading_path,
        "heading_level": chunk.heading_level,
        "page_estimate": chunk.page_estimate or 0,
        "contains_table": int(chunk.contains_table),
        "contains_code":  int(chunk.contains_code),
        "contains_list":  int(chunk.contains_list),
        "is_protected":   int(chunk.is_protected),
        "token_count":    chunk.token_count,
        "tags":           ",".join(chunk.tags),
        "wikilinks":      ",".join(chunk.wikilinks),
        "language":       chunk.language,
        "created_at":              chunk.created_at,
        "chunking_version":           chunk.chunking_version,
        "context_enrichment_version": chunk.context_enrichment_version,
    }


# ---------------------------------------------------------------------------
# Smoke test (requires Ollama running with qwen3-embedding:0.6b)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, tempfile
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    # Quick connectivity check
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        r.raise_for_status()
        print("Ollama: reachable")
    except Exception as e:
        print(f"Ollama not reachable: {e}")
        print("Start Ollama and ensure qwen3-embedding:0.6b is pulled.")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        # Init storage
        chroma = ChromaStore(base_path=f"{tmp}/vector_db")
        chroma.init()
        bm25   = BM25Store(base_path=f"{tmp}/bm25")
        bm25.init()
        reg    = Registry(db_path=f"{tmp}/registry.db")
        reg.init()

        embedder = Embedder(
            chroma=chroma,
            bm25=bm25,
            registry=reg,
            model=DEFAULT_EMBED_MODEL,
            batch_size=8,
        )

        # Health check
        healthy = embedder.health_check()
        assert healthy, "Embedder health check failed — is qwen3-embedding:0.6b pulled?"
        print("Health check: OK")

        # Create test chunks
        from pipeline.ingestion.chunker import Chunk
        from datetime import datetime, timezone

        def make_chunk(idx: int, text: str, source_type: str = "document") -> Chunk:
            return Chunk(
                chunk_id=f"chunk-{idx}",
                doc_id="test-doc-1",
                source="test.md",
                source_type=source_type,
                chunk_index=idx,
                text=text,
                heading_path=f"Section {idx}",
                heading_level=2,
                page_estimate=idx + 1,
                contains_table=False,
                contains_code=False,
                contains_list=False,
                is_protected=False,
                token_count=len(text) // 4,
                tags=["test"],
                wikilinks=[],
                language="en",
                created_at=datetime.now(timezone.utc).isoformat(),
            )

        chunks = [
            make_chunk(0, "Authentication tokens expire after 24 hours."),
            make_chunk(1, "Rate limits apply to all API endpoints globally."),
            make_chunk(2, "Error codes are documented in the appendix section."),
        ]

        # Register doc
        doc_id = reg.create_document("test.md", "document", file_hash="abc123")

        # Embed
        result = embedder.embed(chunks, doc_id=doc_id)

        assert result.chunks_total    == 3, f"total: {result.chunks_total}"
        assert result.chunks_embedded == 3, f"embedded: {result.chunks_embedded}"
        assert result.chunks_failed   == 0, f"failed: {result.chunks_failed}"
        assert result.collection == "documents"
        print(f"Embed result: OK  "
              f"({result.chunks_embedded}/{result.chunks_total} chunks)")

        # Verify ChromaDB has the chunks
        count = chroma.count("documents")
        assert count == 3, f"ChromaDB count: {count}"
        print(f"ChromaDB count: {count}  OK")

        # Verify BM25 has the chunks
        bm25_count = bm25.count("documents")
        assert bm25_count == 3, f"BM25 count: {bm25_count}"
        print(f"BM25 count: {bm25_count}  OK")

        # Verify registry was updated
        doc = reg.get_document(doc_id)
        assert doc.status == "indexed", f"status: {doc.status}"
        assert doc.chunk_count == 3,    f"chunk_count: {doc.chunk_count}"
        print(f"Registry status: '{doc.status}'  chunk_count={doc.chunk_count}  OK")

        # Verify semantic search works
        query_result = chroma.query(
            collection="documents",
            query_embedding=embedder._embed_texts(["authentication token"])[0],
            top_k=3,
        )
        assert len(query_result) > 0
        print(f"Semantic query: OK  — top result: '{query_result[0].text[:60]}...'")

        # Verify BM25 search works
        bm25_results = bm25.query("documents", "rate limit", top_k=3)
        assert len(bm25_results) > 0
        print(f"BM25 query: OK  — top result: '{bm25_results[0].text[:60]}...'")

        # Test re-embed (idempotent — old chunks deleted, new ones written)
        result2 = embedder.embed(chunks, doc_id=doc_id)
        assert chroma.count("documents") == 3   # same count after re-embed
        print(f"Re-embed idempotency: OK")

        print("\nAll Embedder assertions passed.")
