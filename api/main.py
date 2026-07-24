"""
api/main.py
FastAPI application — wires all pipeline components together
and exposes the REST API.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yaml
import dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from api.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, ChunkResult, AdminResponse, PipelineStatus, DocumentStatus
from pipeline.ingestion.validator import FileValidator
from pipeline.ingestion.converter import DocumentConverter
from pipeline.ingestion.metadata import MetadataExtractor
from pipeline.ingestion.chunker import HierarchicalChunker
from pipeline.ingestion.embedder import Embedder
from pipeline.retrieval.router import QueryRouter, Route
from pipeline.retrieval.fusion import RecipRankFusion
from pipeline.retrieval.reranker import Reranker
from pipeline.retrieval.prompt_builder import PromptBuilder
from pipeline.vault.graph_builder import VaultGraphBuilder
from pipeline.vault.watcher import VaultWatcher, VaultEvent, VaultEventType
from storage.registry import Registry, SOURCE_DOCUMENT, SOURCE_VAULT
from storage.chroma import ChromaStore, COLLECTION_DOCUMENTS, COLLECTION_VAULT
from storage.bm25 import BM25Store, INDEX_DOCUMENTS, INDEX_VAULT
from telemetry.recorder import TelemetryRecorder, make_record

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_config(path: str = "config.yaml") -> dict:
    """Load config.yaml, return defaults if file not found."""
    defaults = {
        "embedding_model":           "qwen3-embedding:0.6b",
        "ollama_base_url":           "http://host.docker.internal:11434",
        "embedding_batch_size":      32,
        "reranker_model":            "Qwen/Qwen3-Reranker-0.6B",
        "reranker_top_k":            6,
        "target_chunk_tokens":       512,
        "chunk_overlap_tokens":      64,
        "small_doc_threshold_tokens": 20000,
        "dense_top_k":               10,
        "sparse_top_k":              10,
        "fusion_top_k":              20,
        "high_confidence_threshold": 0.7,
        "low_confidence_threshold":  0.4,
        "graph_expansion_enabled":   True,
        "graph_expansion_depth":     1,
        "graph_expansion_max_per_chunk": 3,
        "hyde_enabled":              False,
        "hyde_min_query_tokens":     6,
        "compression_enabled":       False,
        "query_cache_enabled":       True,
        "query_cache_ttl_seconds":   3600,
        "vault_path":                "/vault",
        "vault_exclude_patterns":    [".git/**", ".opencode/node_modules/**"],
        "vault_watcher_debounce_seconds": 2,
        "max_file_size_mb":          100,
        "telemetry_enabled":         True,
    }
    try:
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        defaults.update(user_config)
    except FileNotFoundError:
        logger.warning("config.yaml not found — using defaults.")
    return defaults

# ---------------------------------------------------------------------------
# App state (shared across requests)
# ---------------------------------------------------------------------------

class AppState:
    config:     dict
    registry:   Registry
    chroma:     ChromaStore
    bm25:       BM25Store
    validator:  FileValidator
    converter:  DocumentConverter
    meta_extractor: MetadataExtractor
    chunker:    HierarchicalChunker
    embedder:   Embedder
    router:     QueryRouter
    fusion:     RecipRankFusion
    reranker:   Reranker
    prompt_builder: PromptBuilder
    graph:      VaultGraphBuilder
    watcher:    Optional[VaultWatcher]
    telemetry:  Optional[TelemetryRecorder]
    query_cache: dict    # simple in-memory TTL cache

state = AppState()

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all pipeline components on startup, clean up on shutdown."""
    cfg = _load_config()
    state.config = cfg

    logger.info("Initialising RAG pipeline...")
    # ── Environment variable overrides (matches .env.example) ──
    dotenv.load_dotenv()
    for key in ("OLLAMA_BASE_URL", "EMBEDDING_MODEL", "RERANKER_MODEL",
                "VAULT_PATH", "TARGET_CHUNK_TOKENS", "CHUNK_OVERLAP_TOKENS",
                "DENSE_TOP_K", "SPARSE_TOP_K", "CONFIDENCE_THRESHOLD",
                "CACHE_TTL_SECONDS", "MAX_FILE_SIZE_MB", "LOG_LEVEL"):
        if key in os.environ:
            cfg_key = key.lower()
            if cfg_key in cfg or key == "OLLAMA_BASE_URL":
                val = os.environ[key]
                if val.replace(".", "").replace("-", "").isdigit():
                    val = float(val) if "." in val else int(val)
                cfg[cfg_key if cfg_key in cfg else "ollama_base_url"] = val


    # Storage
    state.registry = Registry("data/registry.db")
    state.registry.init()

    state.chroma = ChromaStore("data/vector_db")
    state.chroma.init()

    state.bm25 = BM25Store("data/bm25")
    state.bm25.init()

    # Ingestion components
    state.validator = FileValidator(
        max_size_bytes=cfg["max_file_size_mb"] * 1024 * 1024
    )
    state.converter      = DocumentConverter()
    state.meta_extractor = MetadataExtractor()
    state.chunker        = HierarchicalChunker(
        target_tokens=cfg["target_chunk_tokens"],
        overlap_tokens=cfg["chunk_overlap_tokens"],
    )
    state.embedder = Embedder(
        chroma=state.chroma,
        bm25=state.bm25,
        registry=state.registry,
        ollama_url=cfg["ollama_base_url"],
        model=cfg["embedding_model"],
        batch_size=cfg["embedding_batch_size"],
    )

    # Retrieval components
    known_sources = [
        doc.source_path
        for doc in state.registry.list_documents(source_type=SOURCE_DOCUMENT)
        if doc.status == "indexed"
    ]
    state.router  = QueryRouter(
        known_sources=known_sources,
        hyde_min_tokens=cfg["hyde_min_query_tokens"],
    )
    state.fusion  = RecipRankFusion()
    state.reranker = Reranker(model_name=cfg["reranker_model"])
    state.prompt_builder = PromptBuilder()
    state.query_cache    = {}

    # Vault graph
    vault_path = Path(cfg["vault_path"]).expanduser()
    state.graph = VaultGraphBuilder(vault_path=vault_path)
    graph_cache = Path("data/vault_graph.json")
    if graph_cache.exists():
        state.graph.load(str(graph_cache))
        logger.info("Vault graph loaded from cache.")
    elif vault_path.exists():
        count = state.graph.build(
            exclude_patterns=cfg["vault_exclude_patterns"]
        )
        state.graph.save(str(graph_cache))
        logger.info("Vault graph built: %d notes", count)

    # Vault watcher
    state.watcher = None
    if vault_path.exists():
        state.watcher = VaultWatcher(
            vault_path=vault_path,
            on_event=_on_vault_event,
            exclude_patterns=cfg["vault_exclude_patterns"],
            debounce_seconds=cfg["vault_watcher_debounce_seconds"],
            startup_sync=True,
        )
        state.watcher.start()
        logger.info("Vault watcher started.")

    # Telemetry
    state.telemetry = None
    if cfg["telemetry_enabled"]:
        state.telemetry = TelemetryRecorder("data/telemetry.db")
        state.telemetry.start()

    logger.info("RAG pipeline ready.")
    yield

    # Shutdown
    if state.watcher:
        state.watcher.stop()
    if state.telemetry:
        state.telemetry.stop()
    logger.info("RAG pipeline shut down.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AKASHIC RAG Pipeline",
    description="Local-first document and vault retrieval API",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload and ingest a document.
    Accepts: PDF, DOCX, PPTX, XLSX, HTML, EPUB, MD, TXT
    Returns immediately with doc_id; ingestion runs in the background.
    """
    import asyncio, uuid

    # Read file bytes
    data = await file.read()

    # Validate
    result = state.validator.validate_bytes(data, file.filename or "upload")
    if not result.valid:
        raise HTTPException(status_code=400, detail=result.error)

    # Deduplication
    from storage.registry import hash_file
    import hashlib
    file_hash = hashlib.sha256(data).hexdigest()
    existing  = state.registry.get_document_by_hash(file_hash)
    if existing and existing.status == "indexed":
        return IngestResponse(
            doc_id=existing.doc_id,
            status="already_indexed",
            message=f"Document already indexed as {existing.source_path}.",
        )

    # Save to uploads/
    safe_name  = result.filename
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    upload_path = upload_dir / safe_name
    upload_path.write_bytes(data)

    # Register
    doc_id = state.registry.create_document(
        source_path=safe_name,
        source_type=SOURCE_DOCUMENT,
        file_hash=file_hash,
    )

    # Run ingestion in background
    asyncio.create_task(_ingest_document(doc_id, upload_path, safe_name))

    return IngestResponse(
        doc_id=doc_id,
        status="pending",
        message="Document queued for ingestion.",
    )

def invalidate_query_cache() -> None:
    """Clear the in-memory query cache when vault data changes."""
    if state.query_cache:
        n = len(state.query_cache)
        state.query_cache.clear()
        logger.info("Query cache cleared (%d entries)", n)

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """
    Query the knowledge base.
    Searches documents, vault, or both depending on the query.
    Returns relevant chunks + an assembled prompt ready for an LLM.
    """
    import time

    t0    = time.time()
    cfg   = state.config
    q     = req.query.strip()
    top_k = req.top_k or cfg["reranker_top_k"]

    # Cache check
    cache_key = f"{q}:{req.route}:{top_k}"
    cache_hit = False
    if cfg["query_cache_enabled"] and cache_key in state.query_cache:
        cached_at, cached_resp = state.query_cache[cache_key]
        if time.time() - cached_at < cfg["query_cache_ttl_seconds"]:
            cache_hit = True
            cached_resp.cache_hit = True
            return cached_resp

    # Route
    router_result = state.router.route(q)
    if req.route:
        _route_map = {"docs": Route.DOCUMENTS, "vault": Route.VAULT, "both": Route.BOTH}
        router_result.route = _route_map.get(req.route, router_result.route)
    if req.hyde is not None:
        router_result.hyde_enabled = req.hyde

    route = router_result.route

    # Embed query
    query_vecs = state.embedder._embed_texts([q])
    if not query_vecs:
        raise HTTPException(status_code=503, detail="Embedding service unavailable.")
    query_vec = query_vecs[0]

    # Dense retrieval
    dense_docs  = []
    dense_vault = []
    if route in (Route.DOCUMENTS, Route.BOTH):
        dense_docs = state.chroma.query(
            COLLECTION_DOCUMENTS, query_vec,
            top_k=cfg["dense_top_k"],
            where=router_result.metadata_filter,
        )
    if route in (Route.VAULT, Route.BOTH):
        dense_vault = state.chroma.query(
            COLLECTION_VAULT, query_vec,
            top_k=cfg["dense_top_k"],
        )

    # Sparse retrieval
    sparse_docs  = []
    sparse_vault = []
    if route in (Route.DOCUMENTS, Route.BOTH):
        sparse_docs = state.bm25.query(INDEX_DOCUMENTS, q, top_k=cfg["sparse_top_k"])
    if route in (Route.VAULT, Route.BOTH):
        sparse_vault = state.bm25.query(INDEX_VAULT, q, top_k=cfg["sparse_top_k"])

    # Graph expansion
    graph_expanded = []
    graph_used     = False
    if cfg["graph_expansion_enabled"] and route in (Route.VAULT, Route.BOTH):
        for vault_result in dense_vault[:3]:
            source = vault_result.metadata.get("source", "")
            if source:
                expansion = state.graph.expand(
                    source,
                    depth=cfg["graph_expansion_depth"],
                    max_results=cfg["graph_expansion_max_per_chunk"],
                )
                for linked_path in expansion.linked_paths:
                    linked = state.chroma.get_by_metadata(
                        COLLECTION_VAULT,
                        where={"source": linked_path},
                        limit=2,
                    )
                    graph_expanded.extend(linked)
                    graph_used = True

    # RRF fusion
    fused = state.fusion.fuse(
        dense_docs=dense_docs,
        sparse_docs=sparse_docs,
        dense_vault=dense_vault,
        sparse_vault=sparse_vault,
        graph_expanded=graph_expanded,
        top_k=cfg["fusion_top_k"],
    )

    # Rerank
    reranked = state.reranker.rerank(query=q, candidates=fused, top_k=top_k)

    # Dense confidence: best (lowest) cosine distance across all dense results
    all_dense = dense_docs + dense_vault
    min_dense_distance = min((d.distance for d in all_dense), default=1.0)

    # Confidence: configurable source
    confidence_source = cfg.get("confidence_source", "dense")
    if confidence_source == "dense":
        top_confidence = 1.0 - min_dense_distance
        dense_threshold = cfg.get("dense_confidence_threshold", 0.55)
        not_found = min_dense_distance > dense_threshold or not reranked
    else:
        top_confidence = reranked[0].reranker_score if reranked else 0.0
        not_found = not reranked or top_confidence < cfg["low_confidence_threshold"]

    # Build prompt
    prompt = state.prompt_builder.build(query=q, results=reranked)

    latency_ms = int((time.time() - t0) * 1000)

    # Telemetry
    if state.telemetry:
        state.telemetry.record(make_record(
            query=q,
            router_decision=route.value,
            cache_hit=False,
            hyde_used=router_result.hyde_enabled,
            graph_expansion_used=graph_used,
            dense_results_count=len(dense_docs) + len(dense_vault),
            sparse_results_count=len(sparse_docs) + len(sparse_vault),
            fused_results_count=len(fused),
            reranked_results_count=len(reranked),
            top_confidence_score=top_confidence,
            not_found=not_found,
            final_chunk_count=prompt.chunk_count,
            final_token_count=prompt.total_tokens,
            latency_ms=latency_ms,
        ))

    # Build response
    chunks = [
        ChunkResult(
            text=r.text,
            source=r.metadata.get("source", ""),
            heading_path=r.metadata.get("heading_path", ""),
            collection=r.collection,
            confidence=1.0 - min_dense_distance,
            page_estimate=r.metadata.get("page_estimate") or None,
        )
        for r in reranked
    ]

    response = QueryResponse(
        query=q,
        chunks=chunks,
        prompt=prompt.full_prompt,
        confidence=top_confidence,
        not_found=not_found,
        route_used=route.value,
        cache_hit=False,
        latency_ms=latency_ms,
    )

    # Cache
    if cfg["query_cache_enabled"]:
        state.query_cache[cache_key] = (time.time(), response)

    return response

@app.get("/status", response_model=PipelineStatus)
async def status():
    """Return pipeline health and document counts."""
    reg_summary = state.registry.summary()
    telemetry   = {}
    if state.telemetry:
        telemetry = state.telemetry.summary()

    return PipelineStatus(
        **reg_summary,
        vault_watcher_running=state.watcher.is_running() if state.watcher else False,
        queue_depth=state.watcher.queue_depth() if state.watcher else 0,
        telemetry=telemetry,
    )

@app.get("/status/{doc_id}", response_model=DocumentStatus)
async def document_status(doc_id: str):
    """Return status of a specific document by doc_id."""
    doc = state.registry.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return DocumentStatus(**{
        k: v for k, v in vars(doc).items()
        if k in DocumentStatus.model_fields
    })

@app.delete("/document/{doc_id}", response_model=AdminResponse)
async def delete_document(doc_id: str):
    """Remove a document from all indexes."""
    doc = state.registry.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    collection = COLLECTION_VAULT if doc.source_type == SOURCE_VAULT else COLLECTION_DOCUMENTS
    bm25_index = INDEX_VAULT     if doc.source_type == SOURCE_VAULT else INDEX_DOCUMENTS

    state.chroma.delete_by_metadata(collection, where={"doc_id": doc_id})
    state.bm25.delete_by_doc_id(bm25_index, doc_id)
    state.registry.delete_document(doc_id)

    return AdminResponse(
        success=True,
        message=f"Document '{doc.source_path}' deleted.",
        doc_id=doc_id,
    )

@app.post("/reindex", response_model=AdminResponse)
async def reindex(doc_id: Optional[str] = None):
    """Re-ingest a specific document or all documents."""
    import asyncio

    if doc_id:
        doc = state.registry.get_document(doc_id)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
        upload_path = Path("data/uploads") / doc.source_path
        if not upload_path.exists():
            raise HTTPException(status_code=404,
                                detail=f"Source file not found: {doc.source_path}")
        asyncio.create_task(_ingest_document(doc_id, upload_path, doc.source_path))
        return AdminResponse(success=True, message="Re-ingestion queued.", doc_id=doc_id)

    # Re-ingest all
    docs = state.registry.list_documents(source_type=SOURCE_DOCUMENT)
    queued = 0
    for doc in docs:
        upload_path = Path("data/uploads") / doc.source_path
        if upload_path.exists():
            asyncio.create_task(_ingest_document(doc.doc_id, upload_path, doc.source_path))
            queued += 1

    return AdminResponse(success=True, message=f"Re-ingestion queued for {queued} documents.")

# ---------------------------------------------------------------------------
# Background ingestion
# ---------------------------------------------------------------------------

async def _ingest_document(doc_id: str, path: Path, source_name: str) -> None:
    """Background task: convert → chunk → embed."""
    import asyncio

    loop = asyncio.get_event_loop()
    try:
        # Run CPU-bound ingestion in thread pool to avoid blocking event loop
        await loop.run_in_executor(None, _ingest_sync, doc_id, path, source_name)
    except Exception as e:
        logger.error("Ingestion failed for %s: %s", source_name, e)
        state.registry.update_status(doc_id, "failed", error_message=str(e))

def _ingest_sync(doc_id: str, path: Path, source_name: str) -> None:
    """Synchronous ingestion pipeline (runs in thread pool)."""
    # Convert
    state.registry.update_status(doc_id, "converting")
    conv_result = state.converter.convert(
        path, output_dir=Path("data/markdown")
    )
    if not conv_result.markdown.strip():
        state.registry.update_status(
            doc_id, "failed",
            error_message="Conversion produced empty output."
        )
        return

    state.registry.update_status(
        doc_id, "chunking",
        conversion_method=conv_result.method,
        language=conv_result.markdown[:500],  # rough first 500 chars for lang detect
    )

    # Extract metadata
    meta = state.meta_extractor.extract(conv_result.markdown)

    # Chunk
    chunks = state.chunker.chunk(
        text=conv_result.markdown,
        doc_id=doc_id,
        source=source_name,
        source_type=SOURCE_DOCUMENT,
        metadata=meta,
    )

    if not chunks:
        state.registry.update_status(
            doc_id, "failed",
            error_message="Chunking produced no chunks."
        )
        return

    # Embed
    state.embedder.embed(chunks, doc_id=doc_id)

    # Update router with new source
    indexed = state.registry.list_documents(source_type=SOURCE_DOCUMENT)
    state.router.update_sources([d.source_path for d in indexed if d.status == "indexed"])

    logger.info("Ingestion complete: %s (%d chunks)", source_name, len(chunks))

# ---------------------------------------------------------------------------
# Vault event handler
# ---------------------------------------------------------------------------

def _on_vault_event(event: VaultEvent) -> None:
    """Handle vault file changes sequentially to prevent race conditions."""
    _handle_vault_event_sync(event)

def _handle_vault_event_sync(event: VaultEvent) -> None:
    """Synchronous vault event handler (runs in a thread)."""
    vault_path = Path(state.config["vault_path"]).expanduser()
    rel_path   = event.path

    if event.event_type == VaultEventType.DELETED:
        doc = state.registry.get_document_by_path(rel_path)
        if doc:
            state.chroma.delete_by_metadata(COLLECTION_VAULT, {"doc_id": doc.doc_id})
            state.bm25.delete_by_doc_id(INDEX_VAULT, doc.doc_id)
            state.registry.delete_document(doc.doc_id)
            state.graph.remove_node(rel_path)
            invalidate_query_cache()
            logger.info("Vault note deleted from index: %s", rel_path)
        return

    if event.event_type == VaultEventType.MOVED:
        if event.old_path:
            old_doc = state.registry.get_document_by_path(event.old_path)
            if old_doc:
                state.registry.delete_document(old_doc.doc_id)
                state.chroma.delete_by_metadata(COLLECTION_VAULT, {"doc_id": old_doc.doc_id})
                state.bm25.delete_by_doc_id(INDEX_VAULT, old_doc.doc_id)
                invalidate_query_cache()
                state.graph.remove_node(event.old_path)

    # CREATED or MODIFIED or MOVED (new path)
    abs_path = vault_path / rel_path
    if not abs_path.exists():
        return

    try:
        text = abs_path.read_text(encoding="utf-8", errors="replace")
        meta = state.meta_extractor.extract(text)

        # Check if already indexed and unchanged
        existing = state.registry.get_document_by_path(rel_path)
        if existing and existing.status == "indexed":
            doc_id = existing.doc_id
            state.chroma.delete_by_metadata(COLLECTION_VAULT, {"doc_id": doc_id})
            state.bm25.delete_by_doc_id(INDEX_VAULT, doc_id)
            state.registry.delete_chunks_for_document(doc_id)
        else:
            doc_id = state.registry.create_document(
                rel_path, SOURCE_VAULT
            )

        chunks = state.chunker.chunk(
            text=text,
            doc_id=doc_id,
            source=rel_path,
            source_type=SOURCE_VAULT,
            metadata=meta,
        )
        if chunks:
            state.embedder.embed(chunks, doc_id=doc_id)
            invalidate_query_cache()

        # Update graph
        state.graph.update_node(rel_path, abs_path)
        state.graph.save("data/vault_graph.json")

        logger.info("Vault note indexed: %s (%d chunks)", rel_path, len(chunks))

    except Exception as e:
        logger.error("Failed to index vault note %s: %s", rel_path, e)
