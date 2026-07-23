# AKASHIC — Architecture

> Knowledge Archive & Semantic Hybrid Indexing Core  
> A headless, self-hosted, version-aware hybrid knowledge retrieval service with measurable retrieval quality and validated document lifecycle consistency.

```
             YOUR KNOWLEDGE
                    │
                    ▼
               AKASHIC
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
     Retrieval             Lifecycle
      ~98%                  5/5
          │                   │
          └─────────┬─────────┘
                    ▼
               TRUSTED DATA
                    │
                    ▼
               CONSUMERS
              (PAN, agents, LLMs)
```

## Table of Contents

1. [Principles](#principles)
2. [System Overview](#system-overview)
3. [Ingestion Pipeline](#ingestion-pipeline)
4. [Indexing Layer](#indexing-layer)
5. [Retrieval Pipeline](#retrieval-pipeline)
6. [Confidence Gating](#confidence-gating)
7. [Lifecycle Management](#lifecycle-management)
8. [Consistency Guarantees](#consistency-guarantees)
9. [Evaluation](#evaluation)
10. [Configuration](#configuration)
11. [API Reference](#api-reference)

---

## 1. Principles

AKASHIC is not a general-purpose RAG framework. It is infrastructure built to serve personal knowledge within a private AI ecosystem. This drives four architectural priorities:

1. **Measurable** — every pipeline change is validated against a reproducible evaluation suite
2. **Version-aware** — every chunk carries pipeline version metadata, enabling index auditing
3. **Lifecycle-complete** — documents can be created, modified, renamed, and deleted without data corruption
4. **Self-hosted** — all components run offline on consumer GPUs (4–6 GB VRAM)

---

## 2. System Overview

```
                    INGEST
                      │
            ┌─────────┼─────────┐
            │         │         │
            ▼         ▼         ▼
        Parser    Chunker   Enrichment
            │         │         │
            └─────────┼─────────┘
                      │
                      ▼
                 INDEXING
                      │
              ┌───────┴───────┐
              │               │
              ▼               ▼
          ChromaDB           BM25S
         (dense vec)       (sparse)
              │               │
              └───────┬───────┘
                      │
                      ▼
                RETRIEVAL
                      │
              ┌───────┼───────┐
              │       │       │
              ▼       ▼       ▼
           Dense   Sparse   RRF Fusion
              │       │       │
              └───────┼───────┘
                      │
                      ▼
                Confidence   Reranker
                   Gate       (ranking)
                      │
                      ▼
                  /query
```

### Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| API | `api/main.py` | FastAPI server, routing, caching, vault watcher |
| Ingestion | `pipeline/ingestion/` | Parse, chunk, enrich, embed |
| Retrieval | `pipeline/retrieval/` | Dense/sparse fusion, reranking, query routing |
| Storage | `storage/` | ChromaDB, BM25S, SQLite registry |
| Vault | `pipeline/vault/` | Filesystem watcher, Obsidian integration |
| Evaluation | `tests/eval_set/` | 51-case benchmark, Recall/MRR reporting |

---

## 3. Ingestion Pipeline

```
Source Document
      │
      ▼
  ┌──────────┐
  │  Parser  │  markitdown: PDF, DOCX, PPTX, XLSX, HTML, EPUB, MD, TXT
  └──────────┘
      │
      ▼
  ┌──────────┐
  │  Chunker │  HierarchicalChunker: 7 splitting rules
  └──────────┘
      │
      ▼
  ┌──────────────┐
  │   Enrich     │  section_context (flat docs) / heading_path (structured)
  └──────────────┘
      │
      ▼
  ┌──────────┐
  │ Embedder │  qwen3-embedding:0.6b → ChromaDB + BM25S
  └──────────┘
```

### Parser

Uses `markitdown` to convert any supported format to clean markdown. The resulting markdown is the canonical intermediate representation — all downstream processing operates on markdown text.

### Chunker — HierarchicalChunker

Seven splitting rules applied in order:

| # | Rule | Trigger | Behavior |
|:-:|------|---------|----------|
| 1 | Frontmatter | `---\nkey: value\n---` | Whole block as one chunk |
| 2 | Horizontal rule | `---` or `***` alone on line | New section |
| 3 | Heading | `# `, `## `, etc. | New section, tracks `heading_path` |
| 4 | Code block | ` ``` ` | Whole block as one chunk |
| 5 | List/table boundary | Bullet ↔ paragraph | Split at gap |
| 6 | Token threshold | > 512 tokens | Split at nearest sentence/paragraph boundary |
| 7 | Overlap | Previous chunk exit | Append 64-char overlap for continuity |

### Context Enrichment

Two enrichment strategies, applied by priority:

1. **`heading_path`** — For structured documents with `#` headers. Prepends the full heading chain (e.g. `2026-07-06 — Monday > 🌅 Morning`) before embedding.

2. **`section_context`** — For flat documents (converted PDFs without markdown headers). Uses `_SECTION_RE` regex `/^(\w[\w\s]+)\s+\((\w[\w\s]*)\)\s*—/m` to detect section headers like `Activities (Aloe) —` and attaches the herb name as context.

Both strategies inject context **only into the embedding vector** — stored text remains the original chunk content. This improves retrieval without modifying the displayed content.

### Chunk Schema

```python
@dataclass
class Chunk:
    chunk_id: str                    # UUID
    doc_id: str                      # Source document UUID
    source_type: str                 # "vault" | "document"
    source: str                      # Filename / relative path
    text: str                        # Chunk text
    heading_path: str                # Breadcrumb from markdown headings
    section_context: str             # Section identity (flat docs)
    heading_level: int               # Deepest heading level
    chunk_index: int                 # Position in document
    token_count: int                 # Estimated token count
    language: str                    # Detected language
    contains_code: bool              # Has code blocks
    contains_table: bool             # Has markdown tables
    contains_list: bool             # Has list items
    page_estimate: int               # Approximate page number
    tags: str                        # Comma-separated tags
    wikilinks: str                   # [[wikilinks]]
    is_protected: bool               # Exclude from deletion
    created_at: datetime             # Indexing timestamp
    chunking_version: str            # Pipeline version
    context_enrichment_version: str  # Enrichment version
    embedding_model: str             # Embedding model used
```

---

## 4. Indexing Layer

### ChromaDB (Dense)

- **Model**: `qwen3-embedding:0.6b` (runs via Ollama at `http://localhost:11434`)
- **Collection**: Two collections — `documents` and `vault`
- **Metadata**: Full Chunk schema stored as ChromaDB metadata
- **Backend**: HNSW index (cosine distance)
- **Query**: `collection.query(query_embeddings=[...], n_results=top_k)`

### BM25S (Sparse)

- **Index**: Separate sparse index for each collection
- **Tokenization**: Word-level, built from the same chunk texts
- **Query**: `bm25.query(index, query_text, top_k=top_k)`
- **Retrieval**: Returns chunk_id + text + metadata

### Registry (SQLite)

Tracks document metadata separately from vector indices:

```sql
documents (
    doc_id          TEXT PRIMARY KEY,
    source_path     TEXT,
    source_type     TEXT,       -- "vault" | "document"
    status          TEXT,       -- "pending" | "embedding" | "indexed" | "failed" |
    file_hash       TEXT,
    git_hash        TEXT,
    chunk_count     INTEGER,
    token_count     INTEGER,
    conversion_method TEXT,
    language        TEXT,
    error_message   TEXT,
    created_at      TIMESTAMP,
    updated_at      TIMESTAMP
)
```

This is the single source of truth for document existence. A document exists if and only if it has a registry entry.

---

## 5. Retrieval Pipeline

```
Query
  │
  ├──→ Dense (ChromaDB) ──┐
  │                       │
  └──→ Sparse (BM25S) ───┤
                          │
                          ▼
                      RRF Fusion
                          │
                          ▼
                  ┌───────┴───────┐
                  │               │
                  ▼               ▼
           Confidence Gate    Reranker
           (dense cosine)     (ranking only)
                  │               │
                  └───────┬───────┘
                          │
                          ▼
                    Response
```

### Dense Search

Queries ChromaDB with the embedding of the user's query. Returns `top_k` most similar chunks with cosine distances.

### Sparse Search

Queries BM25S with the raw query text. Returns top chunks by keyword overlap.

### RRF Fusion

Reciprocal Rank Fusion combines dense and sparse results:

```python
score = 1 / (k + rank_dense(chunk)) + 1 / (k + rank_sparse(chunk))
# where k = 60 (constant)
```

### Reranker

Optional cross-encoder reranking via `Qwen/Qwen3-Reranker-0.6B`. If the reranker fails (e.g., missing system dependencies), the pipeline falls back to RRF order without degradation.

**Note**: The reranker is used for **ranking only**. Confidence gating uses dense cosine distance instead (see next section), because the reranker's raw scores are compressed into a narrow range (0.016–0.033) with no separation between correct and incorrect results.

### Query Router

Three routing modes:

| Mode | Collections Searched | Use Case |
|------|---------------------|----------|
| `docs` | ChromaDB documents + BM25S documents | Uploaded files |
| `vault` | ChromaDB vault + BM25S vault | Obsidian notes |
| `both` | All four | Default — best coverage |

---

## 6. Confidence Gating

The confidence gate separates answerable queries from unanswerable ones.

### Signal: Dense Cosine Distance

Uses ChromaDB's minimum cosine distance across all retrieved chunks. Two distributions separate cleanly:

| Category | Distance Range | Interpretation |
|----------|---------------|---------------|
| Answerable | 0.21 – 0.60 | Query has relevant content in the index |
| Unanswerable | 0.47 – 0.72 | Query has no relevant content |

### Threshold

- **`dense_confidence_threshold: 0.55`**
- Precision: 97.9%
- Recall: 95.7%
- F1: 0.968

The threshold is validated against the 51-case eval set and stored as a configuration value (`config.yaml`). It should be re-evaluated periodically as the corpus and query distribution change.

### Why Not the Reranker Score

The cross-encoder produces logits in a [0.016, 0.033] range across ALL query-chunk pairs — correct and incorrect results overlap completely. The scores are useful for **ordering** chunks (relative ranking is preserved) but useless for **gating** (absolute values carry no signal).

---

## 7. Lifecycle Management

The vault watcher monitors `/home/user/Documents/SecondBrain` via inotify and maintains index consistency through all file operations.

### Event Flow

```
File System Event
      │
      ▼
  InotifyObserver
      │
      ▼
  Event Queue
      │
      ▼
  _on_vault_event (sequential, single-threaded)
      │
      ▼
  _handle_vault_event_sync
```

Events are processed **sequentially** to prevent race conditions. Each event is fully handled before the next begins.

### CREATE

```
New file created
  → Watcher detects CREATED event
  → Registry: create_document(path, "vault") → doc_id
  → Chunker: split text into chunks
  → Embedder: embed chunks → ChromaDB + BM25S
  → Registry: update_status("indexed")
  → Cache: invalidate_query_cache()
```

### MODIFY

```
File content changed
  → Watcher detects MODIFIED event
  → Registry: get_document_by_path(path) → existing doc
  → ChromaDB: delete_by_metadata({"doc_id": doc_id})
  → BM25S: delete_by_doc_id(index, doc_id)
  → Registry: delete_chunks_for_document(doc_id)
  → Chunker: re-split new text
  → Embedder: re-embed → ChromaDB + BM25S (same doc_id)
  → Cache: invalidate_query_cache()
```

### RENAME

```
File renamed/moved
  → Watcher detects MOVED event
  → Registry: get_document_by_path(old_path) → old_doc
  → ChromaDB: delete_by_metadata({"doc_id": old_doc.doc_id})
  → BM25S: delete_by_doc_id(index, old_doc.doc_id)
  → Registry: delete_document(old_doc.doc_id)
  → Proceed as CREATE for new path
  → Cache: invalidate_query_cache()
```

### DELETE

```
File deleted
  → Watcher detects DELETED event
  → Registry: get_document_by_path(path) → doc
  → ChromaDB: delete_by_metadata({"doc_id": doc.doc_id})
  → BM25S: delete_by_doc_id(index, doc.doc_id)
  → Registry: delete_document(doc.doc_id)
  → Graph: remove_node(path)
  → Cache: invalidate_query_cache()
```

### Invariant

> After any lifecycle operation, zero retrievable chunks in any index reference a non-existent document.

---

## 8. Consistency Guarantees

### Cache Invalidation

The in-memory query cache (3600s TTL) is cleared on every vault document change:

```python
def invalidate_query_cache() -> None:
    """Clear the in-memory query cache when vault data changes."""
    if state.query_cache:
        n = len(state.query_cache)
        state.query_cache.clear()
```

This ensures that a MODIFY, RENAME, or DELETE immediately produces fresh query results. Queries that arrive between the cache invalidation and the new index write may see slightly stale results (a window of 1–2 seconds for indexing).

### Sequential Event Processing

Vault events are handled one at a time in the watcher's polling loop. This prevents race conditions where a MOVED event's indexing thread and a DELETE event's cleanup thread modify the same data concurrently.

### Triple-Store Consistency

Every lifecycle operation updates all three stores atomically:

| Store | Location | Cleared on |
|-------|----------|-----------|
| ChromaDB (dense) | `data/vector_db/` | DELETE / MODIFY / MOVED |
| BM25S (sparse) | `data/bm25/` | DELETE / MODIFY / MOVED |
| Registry (SQLite) | `data/registry.db` | DELETE / MOVED |

If any store write fails, the operation is logged and the other stores remain consistent. A full reindex (`POST /reindex`) reconciles all three.

### Orphan Detection

The recommended periodic check:

```python
# Every registry entry should have matching ChromaDB/BM25 entries
registry.documents - (chroma.doc_ids ∪ bm25.doc_ids) = ∅
chroma.doc_ids ∖ registry.documents = observed, flagged for cleanup
```

---

## 9. Evaluation

### Test Suite

51 query-passage pairs across five categories:

| Category | Count | Source | Purpose |
|----------|:-----:|--------|---------|
| Vault | 30 | Obsidian notes | Daily logs, setup, templates |
| Alice | 12 | Alice in Wonderland (public domain) | Document retrieval from a free-text book |
| Setup | 4 | Vault configuration | _CLAUDE.md, index.md, Home.md |
| Edge | 4 | Cross-domain | Unanswerable queries, policy exclusion |
| Templates | 1 | Person template | Deliberately excluded by design |

### Metrics

- **Recall@K** — Proportion of correct documents retrieved
- **Precision@K** — Proportion of relevant results in top K
- **MRR** — Mean Reciprocal Rank (how high the first correct result appears)

### Current Baseline

```
Overall:   98% recall
Alice:     100%
Vault:     96.7% (Person template excluded by policy)
Setup:     100%
Edge:      50% (config.yaml retrieves relevant vault content — correct behavior)
```

### Running

```bash
export PYTHONPATH=$(pwd)
python3 tests/eval_set/eval_runner.py \
  --api-url http://127.0.0.1:18765 \
  --output tests/eval_set/report.json
```

Results include per-category breakdown, aggregate metrics, and expected-vs-actual analysis.

---

## 10. Configuration

All configuration lives in `config.yaml` at the project root.

```yaml
# --- Embedding ---
embedding_model: qwen3-embedding:0.6b
ollama_base_url: http://localhost:11434
embedding_batch_size: 32

# --- Reranker ---
reranker_model: Qwen/Qwen3-Reranker-0.6B
reranker_top_k: 6

# --- Chunking ---
target_chunk_tokens: 512
chunk_overlap_tokens: 64
small_doc_threshold_tokens: 20000

# --- Retrieval ---
dense_top_k: 10
sparse_top_k: 10
rrf_k: 60
hyde_enabled: true
hyde_min_query_tokens: 6

# --- Search routing ---
vault_path: /home/user/Documents/SecondBrain
vault_exclude_patterns:
  - .git/**
  - .opencode/node_modules/**

# --- Confidence ---
confidence_source: dense           # "dense" or "reranker"
dense_confidence_threshold: 0.55

# --- Caching ---
query_cache_enabled: true
query_cache_ttl_seconds: 3600

# --- API ---
max_file_size_mb: 50
ingestion_threads: 2
```

---

## 11. API Reference

### `POST /query`

Search the knowledge base.

```json
{
  "query": "What did the Cheshire Cat do?",
  "route": "both",
  "top_k": 5
}
```

### `POST /ingest`

Upload a document. Accepts `multipart/form-data` with a `file` field.

Returns `doc_id` immediately. Indexing runs asynchronously.

### `DELETE /document/{doc_id}`

Remove a document and all its chunks from all indices.

### `POST /reindex`

Rebuild the index for a specific document or all documents.

### `GET /status`

Pipeline health: document counts, chunk counts, watcher status, telemetry.

---

## 12. File Layout

```
akashic/
├── api/
│   └── main.py              FastAPI server, routes, vault watcher
├── config.yaml               All configuration
├── data/                     Runtime data (gitignored)
│   ├── vector_db/            ChromaDB persistent index
│   ├── bm25/                 BM25S persistent index
│   └── registry.db           SQLite document registry
├── pipeline/
│   ├── ingestion/
│   │   ├── chunker.py        HierarchicalChunker with context enrichment
│   │   ├── embedder.py       Embedding + ChromaDB/BM25S writes
│   │   ├── converter.py      markitdown-based document conversion
│   │   ├── metadata.py       Frontmatter and property extraction
│   │   └── validator.py      File size and type validation
│   ├── retrieval/
│   │   ├── router.py         Query routing (docs/vault/both)
│   │   ├── fusion.py         RRF fusion
│   │   ├── reranker.py       Cross-encoder reranking
│   │   └── prompt_builder.py Prompt assembly for LLM consumption
│   └── vault/
│       ├── watcher.py        Inotify-based filesystem watcher
│       └── graph_builder.py  Cross-link graph for vault notes
├── storage/
│   ├── chroma.py             ChromaDB wrapper
│   ├── bm25.py               BM25S wrapper
│   └── registry.py           SQLite document registry
├── tests/
│   └── eval_set/
│       ├── eval_set.json             51-case evaluation fixture
│       ├── eval_runner.py            Evaluation harness
│       └── *.json                    Historical baseline reports
├── docker-compose.yml        Docker deployment (n8n, etc.)
├── Dockerfile                Container image
└── requirements.txt          Python dependencies
```

---

## 13. Version Metadata

Every chunk stores pipeline version information in ChromaDB metadata:

| Field | Type | Value | Meaning |
|-------|------|-------|---------|
| `chunking_version` | string | `hierarchical/v3` | Chunker with 7-rule system + context enrichment |
| `embedding_model` | string | `qwen3-embedding:0.6b` | Embedding model used |
| `context_enrichment_version` | string | `v2` | heading_path + section_context enrichment |

These fields enable:
- Detecting stale indexes after pipeline changes
- Comparing retrieval quality across pipeline versions
- Auditing which pipeline version produced each query result
## 14. Engineering Journey

This section explains **why** the architecture looks the way it does — the failures, measurements, and decisions that shaped the current design.

### 13.1 Section-context enrichment

**Problem:** PDF-to-markdown conversion flattens section headers. A handbook entry like `Activities (Aloe) —` produced chunks with text like `— used externally for burns, wounds` but no mention of "Aloe" in the chunk body. Retrieval for "Aloe" returned nothing.

```
PDF section header:  Activities (Aloe) —
                         ↓
Markdown conversion:  — used externally for burns, wounds
                         ↓
Chunk text:           used externally for burns, wounds
                         ↓
Query "Aloe" →        Not found (herb name missing from chunk)
```

**Diagnosis:** The chunker had no concept of section identity. Each chunk was semantically orphaned from its parent heading.

**Fix:** Added `_SECTION_RE` regex (`/^(\w[\w\s]+)\s+\((\w[\w\s]*)\)\s*—/m`) to detect section headers in flat documents, and a `section_context` field on every `Chunk`. The embedder prepends this context before computing the embedding vector — the stored text remains unchanged, but the vector carries the section's identity.

**Result:**
- Herbs category recall: 66.7% → 100%
- Vault recall (structured docs): unchanged (96.7%)
- 98% of handbook chunks now carry section_context

### 13.2 Confidence gating: reranker → dense distance

**Problem:** The cross-encoder reranker was used for both ranking and confidence gating. Its raw scores (logits) were treated as a probability that the query matched the corpus.

**Diagnosis:** Log analysis showed reranker scores in a compressed range (0.016–0.033) across ALL query-chunk pairs. Correct and incorrect results overlapped completely — the absolute value carried no signal.

| Metric | Reranker scores | Dense cosine distance |
|--------|:---------------:|:---------------------:|
| Positive queries | 0.016 – 0.033 | 0.21 – 0.60 |
| Negative queries | 0.016 – 0.033 | 0.47 – 0.72 |
| Separation | None | Clean |

```
Before:
Reranker score → confidence → not_found
              (uncalibrated — no separation)

After:
Dense similarity → confidence gate → not_found
Reranker        → ranking only
```

**Fix:** Switched the confidence signal from the reranker's logits to ChromaDB's minimum cosine distance across retrieved chunks. The reranker continues to rank results — it's better at relative ordering than the dense index. But answerability is determined by the dense distance alone.

**Threshold:** 0.55 (validated against 51-case eval set: 97.9% precision, 95.7% recall, F1=0.968)

**Caveat:** The threshold is provisional — validated against the current corpus and query distribution. It should be re-evaluated as the corpus grows.

### 13.3 Query cache invalidation

**Problem:** Vault watcher correctly updated ChromaDB and BM25S on document changes. But the in-memory query cache (3600s TTL) served stale results for up to an hour. A modify-and-query sequence would return the old content from cache.

```
File modified → Watcher updates index → Cache still serves old results
                                        ↓
                                  User sees stale data
```

**Diagnosis:** The lifecycle tests were passing against the raw indices (ChromaDB had correct data), but the API returned cached responses. The test harness checked API responses, not raw index state.

**Fix:** Added `invalidate_query_cache()` — clears the entire query cache on every vault document event (CREATE, MODIFY, RENAME, DELETE).

```python
def invalidate_query_cache() -> None:
    if state.query_cache:
        state.query_cache.clear()
```

Also added calls in the document DELETE endpoint (`DELETE /document/{doc_id}`).

### 13.4 MOVED handler: orphaned chunk cleanup

**Problem:** Renaming a vault note left orphaned chunks in ChromaDB under the old source path. The MOVED handler correctly updated the registry but never removed the old vectors.

```
File renamed (file.md → new.md)
  ↓
MOVED handler:
  ✓ Deleted old registry entry
  ✓ Created new registry entry (new.md)
  ✓ Indexed new chunks
  ✗ Did not delete old chunks from ChromaDB/BM25S
                        ↓
Old chunks remain retrievable under old path
```

**Fix:** Added ChromaDB and BM25S cleanup in the MOVED handler — the same triple-store cleanup that the MODIFY and DELETE handlers already performed.

```python
if event.event_type == VaultEventType.MOVED:
    if event.old_path:
        old_doc = state.registry.get_document_by_path(event.old_path)
        if old_doc:
            state.chroma.delete_by_metadata(COLLECTION_VAULT, {"doc_id": old_doc.doc_id})
            state.bm25.delete_by_doc_id(INDEX_VAULT, old_doc.doc_id)
            state.registry.delete_document(old_doc.doc_id)
            state.graph.remove_node(event.old_path)
            invalidate_query_cache()
```

### 13.5 Race condition: concurrent event handlers

**Problem:** The vault watcher spawned a new thread for every file system event. A MOVED event and DELETE event for the same file could run concurrently.

```
Thread A: MOVED → delete old index → create new index (in progress)
Thread B: DELETE → delete registry → cache clear
Thread A: → finish indexing → writes new chunks to ChromaDB
                                            ↓
Chunks exist in ChromaDB but registry entry is gone
```

**Diagnosis:** The fire-and-forget threading was borrowed from the initial implementation for responsiveness. For small markdown files (the vault's dominant content type), the handler completes in under 2 seconds — responsiveness was never the bottleneck.

**Fix:** Replaced threading with sequential event processing. The polling loop calls the handler directly instead of delegating to a thread.

```python
# Before: concurrent (bugs)
def _on_vault_event(event):
    threading.Thread(target=_handle_vault_event_sync, args=(event,)).start()

# After: sequential (correct)
def _on_vault_event(event):
    _handle_vault_event_sync(event)
```

### 13.6 Test methodology pitfalls

The lifecycle tests themselves had a bug that masked the true state of the index. The original test used substring matching:

```python
# Broken: substring match
def found(token, resp):
    return token in str(resp.get("chunks", []))
```

For a query "TOKEN_AAA", a chunk containing "LCTOKEN_AAA" would match because "TOKEN_AAA" is a substring of "LCTOKEN_AAA". The test incorrectly reported old content as still present.

Fixed to word-boundary matching:
```python
# Correct: word boundary match
def found_exact(token, resp):
    pat = re.compile(r"\b" + re.escape(token) + r"\b")
    for c in resp.get("chunks", []):
        if pat.search(c.get("text", "")):
            return True
    return False
```

### 13.7 Version tracking

Every chunk stores pipeline version metadata in ChromaDB:
- `chunking_version` (e.g. `hierarchical/v3`)
- `embedding_model` (e.g. `qwen3-embedding:0.6b`)
- `context_enrichment_version` (e.g. `v2`)

This answers the question: "Why did this document retrieve differently after the pipeline changed?" — you can read the version fields from the chunk metadata and know exactly which pipeline state produced it.

The metadata model is designed to evolve toward:

```python
{
    "document_id": "...",
    "chunking_version": "...",
    "context_enrichment_version": "...",
    "embedding_model": "...",
    "indexed_at": "...",
}
```

---

## 15. Future Boundaries

AKASHIC has been deliberately scoped. The following are intentionally out of scope for the initial architecture:

- **Multi-tenant isolation** — single-user by design
- **Distributed indexing** — all stores are local files
- **Real-time streaming** — batch ingestion, interactive query
- **Fine-grained access control** — document-level at most
- **Cross-collection deduplication** — vault and documents are separate namespaces
- **Embedding model rotation** — requires a full reindex
- **Automatic schema migration** — version metadata is advisory, not enforced

These may become relevant as AKASHIC matures, but they are not part of the current design.
