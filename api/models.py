"""Request/response models for AKASHIC API."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class IngestRequest(BaseModel):
    """Upload a document from a URL (optional — file is preferred)."""
    url: Optional[str] = None
    filename: Optional[str] = None


class IngestResponse(BaseModel):
    """Response after queueing a document for ingestion."""
    doc_id: str
    status: str
    message: str


class QueryRequest(BaseModel):
    """Search the knowledge base."""
    query: str
    route: str = "both"
    top_k: Optional[int] = None


class ChunkResult(BaseModel):
    """A single retrieved chunk."""
    chunk_id: str
    doc_id: str
    source: str
    source_type: str
    text: str
    heading_path: str = ""
    score: float = 0.0


class QueryResponse(BaseModel):
    """Full query result."""
    query: str
    route: str
    response: str
    chunks: list[ChunkResult]
    total_chunks: int
    not_found: bool
    confidence: float
    cache_hit: bool = False
    latency_ms: float


class AdminResponse(BaseModel):
    """Generic admin action response."""
    success: bool
    message: str
    doc_id: Optional[str] = None

class PipelineStatus(BaseModel):
    """Pipeline health and document counts."""
    total_documents: int
    indexed_documents: int
    failed_documents: int
    total_chunks: int
    vault_documents: int
    uploaded_documents: int
    vault_watcher_running: bool
    queue_depth: int
    telemetry: dict


class DocumentStatus(BaseModel):
    """Status of a single document in the pipeline."""
    doc_id: str
    source_path: str
    source_type: str
    status: str
    chunk_count: int
    token_count: int
    created_at: str
    updated_at: str
    error_message: Optional[str] = None

