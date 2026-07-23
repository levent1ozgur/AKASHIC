# AKASHIC — Knowledge Archive & Semantic Hybrid Indexing Core

> **A local-first, headless knowledge retrieval service for self-hosted AI systems — combining dense and sparse retrieval, cross-encoder reranking, confidence gating, and Obsidian vault synchronization.**

AKASHIC turns your documents, notes, and research into a searchable knowledge base — served via REST API or MCP protocol to agents, LLMs, and other applications. Everything runs on your own hardware — no cloud calls at query time.

---

## Why AKASHIC exists

Most RAG projects are frameworks: they give you components and expect you to build the system. AKASHIC is the opposite — it is a **finished, deployable knowledge retrieval service** with:

- A measured retrieval baseline (98% recall across 51 test cases)
- Validated document lifecycle (create / modify / rename / delete all tested)
- A reproducible evaluation suite for detecting regressions
- Documented pipeline version metadata for index auditing

It is built for personal knowledge infrastructure — your own notes, documents, and research, served by a self-hosted pipeline that never phones home.

---

## Architecture overview

```
                         Query
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
         Dense Retrieval       Sparse Retrieval
           ChromaDB                BM25S
                │                     │
                └──────────┬──────────┘
                           ▼
                       RRF Fusion
                           │
                           ▼
                   Cross-Encoder
                    (ranking only)
                           │
                           ▼
                  Ranked Evidence
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      Dense Distance                Ranked Chunks
      Confidence Gate                    │
             │                           │
             └─────────────┬─────────────┘
                           ▼
                        Response
```

The pipeline has three phases:

1. **Ingestion** — parses documents (PDF, DOCX, MD, etc.), chunks them with structural context, embeds each chunk, and writes to dense (ChromaDB) and sparse (BM25S) indices
2. **Retrieval** — routes queries across both indices, fuses results via RRF, re-ranks with a cross-encoder (ranking only), and gates answerability via dense cosine distance (separate from ranking)
3. **Lifecycle** — an inotify-based vault watcher keeps the index synchronized with file system changes (create, modify, rename, delete)

**[Read the full architecture →](ARCHITECTURE.md)**

---

## Core capabilities

| Capability | Implementation |
|---|---|
| **Multi-format ingestion** | PDF, DOCX, PPTX, XLSX, HTML, EPUB, MD, TXT via [MarkItDown](https://github.com/microsoft/markitdown) |
| **Hybrid retrieval** | Dense embeddings (ChromaDB) + sparse keywords (BM25S) fused via RRF |
| **Cross-encoder reranking** | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) — GPU optional, CPU fallback |
| **Confidence gating** | Dense cosine distance (threshold 0.55) — not reranker score, which is uncalibrated |
| **Obsidian vault sync** | Inotify-based watcher — auto-indexes, auto-updates on rename, auto-removes on delete |
| **Context enrichment** | `heading_path` for structured docs, `section_context` for flat PDF conversions |
| **Pipeline versioning** | `chunking_version`, `embedding_model`, `context_enrichment_version` in every chunk's metadata |
| **Query cache** | Configurable TTL with automatic invalidation on vault changes |
| **MCP protocol** | Tools: `search_knowledge_base`, `ingest_document`, `pipeline_status` |
| **REST API** | FastAPI — query, ingest, status, admin |
| **HyDE** | Hypothetical Document Embeddings — generate-then-retrieve |
| **Graph expansion** | Wikilink graph from vault notes for related-context retrieval |

---

## Engineering validation

AKASHIC includes a reproducible evaluation suite and automated lifecycle tests.

### Retrieval evaluation — 51 cases

```
51 test cases
├── 30 vault queries (daily logs, setup, templates)
├── 12 document queries (Alice in Wonderland)
├── 4 setup queries
├── 4 edge cases (unanswerable / policy / ambiguity)
└── 1 template exclusion (policy)

Results: 98% overall Recall@K (all currently indexed answerable retrieval cases pass; policy-excluded and expected-unanswerable cases are evaluated separately)
```

Metrics tracked: Recall@K, Precision@K, MRR — per-category and aggregate.

### Lifecycle tests — 5/5

```
CREATE  → note appears in search   ✓
MODIFY  → old content gone, new    ✓
RENAME  → searchable at new path   ✓
DELETE  → content not retrievable  ✓
```

Validates index consistency across ChromaDB, BM25S, the document registry, and the query cache.

**[Evaluation methodology and lifecycle design →](ARCHITECTURE.md#evaluation)**

---

## Quick start

### Prerequisites

```bash
# Python 3.12+ and Ollama
ollama pull qwen3-embedding:0.6b
```

### 1. Configure

```yaml
# config.yaml
ollama_base_url: http://localhost:11434
vault_path: /path/to/your/obsidian/vault
```

### 2. Run

**Docker:**
```bash
docker compose up -d
```
Starts REST API on port **8765** and MCP server on port **8766**.

**Manual:**
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8765
```

### 3. Ingest and query

```bash
curl -X POST http://localhost:8765/ingest -F "file=@document.pdf"
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does this cover?", "top_k": 6}'
```

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a document |
| `POST` | `/query` | Query the knowledge base |
| `GET` | `/status` | Pipeline health and document counts |
| `GET` | `/status/{doc_id}` | Status for a specific document |
| `DELETE` | `/document/{doc_id}` | Remove a document |

---

## MCP

Register in any MCP-compatible client:
```
URL: http://host:8766/mcp
Transport: streamable-http
```

Tools: `search_knowledge_base`, `ingest_document`, `pipeline_status`

---

## Configuration

Environment variables override `config.yaml` at runtime (set in `.env` or shell):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Embeddings endpoint |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Reranker model |
| `VAULT_PATH` | — | Obsidian vault directory |
| `DENSE_CONFIDENCE_THRESHOLD` | `0.55` | Dense cosine threshold for `not_found` |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size |

---

## Project scope & maintenance

AKASHIC is built primarily for personal knowledge infrastructure. It is:

- **Measured** — 51-case eval suite detects regressions before they affect usage
- **Lifecycle-validated** — create, modify, rename, and delete all tested
- **Self-hosted** — all components run offline on consumer GPUs (4–6 GB VRAM)

It is not:

- A general-purpose RAG framework
- A multi-tenant or horizontally scalable knowledge service
- Guaranteed to have active upstream development

Contributions are welcome — especially evaluation cases that expose new failure modes.

---

## Tech stack

FastAPI · ChromaDB · BM25S · Sentence-Transformers · MarkItDown · Ollama · FastMCP · Docker

## License

MIT
