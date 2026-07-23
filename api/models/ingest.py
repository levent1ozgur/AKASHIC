"""
api/models/query.py + api/models/ingest.py — combined for single download.
Pydantic request/response models for the FastAPI layer.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest models
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    doc_id:   str
    status:   str
    message:  str


class DocumentStatus(BaseModel):
    doc_id:             str
    source_path:        str
    source_type:        str
    status:             str
    chunk_count:        int
    token_count:        int
    conversion_method:  Optional[str]
    language:           Optional[str]
    created_at:         Optional[str]
    updated_at:         Optional[str]
    error_message:      Optional[str]


# ---------------------------------------------------------------------------
# Query models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query:  str                     = Field(..., min_length=1, max_length=2000)
    route:  Optional[str]           = Field(None, pattern="^(docs|vault|both)$")
    top_k:  Optional[int]           = Field(None, ge=1, le=20)
    hyde:   Optional[bool]          = None


class ChunkResult(BaseModel):
    text:           str
    source:         str
    heading_path:   str
    collection:     str
    confidence:     float
    page_estimate:  Optional[int]


class QueryResponse(BaseModel):
    query:          str
    chunks:         list[ChunkResult]
    prompt:         str
    confidence:     float
    not_found:      bool
    route_used:     str
    cache_hit:      bool
    latency_ms:     int


# ---------------------------------------------------------------------------
# Status models
# ---------------------------------------------------------------------------

class PipelineStatus(BaseModel):
    total_documents:    int
    indexed_documents:  int
    failed_documents:   int
    total_chunks:       int
    vault_documents:    int
    uploaded_documents: int
    vault_watcher_running: bool
    queue_depth:        int
    telemetry:          dict


class AdminResponse(BaseModel):
    success:    bool
    message:    str
    doc_id:     Optional[str] = None
