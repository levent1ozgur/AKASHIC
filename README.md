# AKASHIC — Knowledge Archive & Semantic Hybrid Indexing Core

> **A local-first, retrieval-augmented generation pipeline with dense + sparse retrieval, cross-encoder reranking, and Obsidian vault integration.**

AKASHIC turns your documents and notes into a searchable knowledge base — hybrid search (dense embeddings + BM25), reranked by a cross-encoder, served via REST API or MCP protocol. Everything runs on your own hardware — no cloud calls at query time.

---

## Architecture

```
                         ┌──────────────────────┐
                         │       AKASHIC        │
                         │    Local-first RAG   │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
         ┌──────────────────┐              ┌──────────────────┐
         │  Documents       │              │    Obsidian      │
         │  PDF, DOCX,      │              │      Vault       │
         │  PPTX, HTML…     │              │  Watcher + Sync  │
         └────────┬─────────┘              └────────┬─────────┘
                  │                                 │
                  └────────────────┬────────────────┘
                                   ▼
                        ┌─────────────────────┐
                        │     Ingestion       │
                        │                     │
                        │ Parse / Normalize   │
                        │    (MarkItDown)     │
                        │         ↓           │
                        │      Chunking       │
                        │         ↓           │
                        │      Embedding      │◄──── Ollama /
                        │                     │      Embedding API
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │     Index Store     │
                        │                     │
                        │ ChromaDB │  BM25S   │
                        │  Dense   │  Sparse  │
                        └──────────┬──────────┘
                                   │
                                   ▼
                        ┌──────────────────────────────┐
                        │       Query Pipeline         │
                        │                              │
                        │ Query → Query Cache          │
                        │          │                   │
                        │          └─ miss → HyDE      │
                        │                    ↓         │
                        │          Dense + Sparse      │
                        │             Retrieval        │
                        │                    ↓         │
                        │              RRF Fusion      │
                        │                    ↓         │
                        │          Cross-Encoder       │
                        │             Reranker         │
                        │                    ↓         │
                        │        Results + Confidence  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │          API Layer           │
                        │                              │
                        │ FastAPI (REST) :8765         │
                        │ FastMCP (MCP)  :8766         │
                        └──────────────────────────────┘
```

### Pipeline flow

```
Query → Query Router
        ├── Dense: ChromaDB (semantic similarity)
        └── Sparse: BM25 (keyword matching)
        → Reciprocal Rank Fusion
        → Cross-Encoder Reranker
            → Prompt Builder → Response with citations
```

---

## Features

| Feature | Details |
|---|---|
| **Multi-format ingestion** | PDF, DOCX, PPTX, XLSX, HTML, EPUB, Markdown, TXT — powered by [MarkItDown](https://github.com/microsoft/markitdown) |
| **Dense retrieval** | ChromaDB with embedding model (Ollama or any OpenAI-compatible endpoint) |
| **Sparse retrieval** | BM25 via [BM25S](https://github.com/dorianbrown/rank_bm25) for keyword matching |
| **Hybrid fusion** | Reciprocal Rank Fusion combines dense + sparse results into a single ranked list |
| **Cross-encoder reranking** | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) re-scores top candidates (GPU optional, falls back to CPU) |
| **Obsidian vault integration** | Watches a vault directory via inotify, auto-indexes new/changed notes, follows `[[wikilinks]]` for graph-aware retrieval |
| **Graph expansion** | Builds a wikilink graph from vault notes — retrieval follows links to surface related context |
| **HyDE** | Hypothetical Document Embeddings — generates a synthetic answer first, then retrieves against it |
| **Query cache** | Caches frequent queries with configurable TTL |
| **Confidence scoring** | Each result includes a reranker confidence score |
| **MCP protocol** | Exposes `search_knowledge_base`, `ingest_document`, `pipeline_status` via the Model Context Protocol |
| **REST API** | Full FastAPI with endpoints for query, ingest, status, admin |
| **Telemetry** | Query latency, cache hit rates, not-found rates, confidence tracking |
| **Dockerized** | Single-container deployment with GPU support |

---

## Quick start

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com) with an embedding model:
  ```bash
  ollama pull qwen3-embedding:0.6b
  ```
- (Optional) NVIDIA GPU with CUDA for reranker acceleration

### 1. Configuration

Edit `config.yaml`:

```yaml
ollama_base_url: http://localhost:11434
vault_path: /path/to/your/obsidian/vault
```

The reranker model (`Qwen/Qwen3-Reranker-0.6B`) downloads automatically from HuggingFace on first use.

### 2. Run

#### Docker (recommended)

```bash
docker compose up -d
```

This starts both the REST API (port 8765) and MCP server (port 8766).

#### Manual (development)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — API:
uvicorn api.main:app --host 0.0.0.0 --port 8765

# Terminal 2 — MCP server:
python mcp_servers/rag_pipeline_server.py
```

### 3. Ingest a document

```bash
curl -X POST http://localhost:8765/ingest \
  -F "file=@report.pdf"
```

### 4. Query

```bash
curl -X POST http://localhost:8765/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What does this document cover?", "top_k": 6}'
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/ingest` | Upload a document for indexing |
| `POST` | `/query` | Query the knowledge base |
| `GET` | `/status` | Pipeline status (doc counts, index sizes, cache stats) |
| `GET` | `/status/{doc_id}` | Status for a specific document |
| `DELETE` | `/document/{doc_id}` | Remove a document and its vectors |

### Query example

```json
{
  "query": "How do authentication tokens expire?",
  "top_k": 6,
  "hyde": false,
  "graph_expansion": true
}
```

---

## MCP Server

AKASHIC includes an MCP (Model Context Protocol) server on port **8766**. Register it in any MCP-compatible client:

```
URL: http://host:8766/mcp
Transport: streamable-http
```

Available tools: `search_knowledge_base`, `ingest_document`, `pipeline_status`

---

## Environment variables

Optional overrides — set these in `.env` or as shell variables to override `config.yaml` at runtime:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama/embeddings endpoint |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Embedding model name |
| `RERANKER_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Reranker model path or HuggingFace ID |
| `VAULT_PATH` | — | Obsidian vault directory |
| `TELEMETRY_ENABLED` | `true` | Enable query telemetry |

---

## Directory structure

```
AKASHIC/
├── api/
│   ├── main.py             # FastAPI app, all routes
│   └── models.py           # Pydantic schemas
├── pipeline/
│   ├── ingestion/          # Validation, conversion (MarkItDown), chunking, embedding
│   ├── retrieval/          # Dense, sparse, fusion, reranking, HyDE, prompt builder
│   └── vault/              # Obsidian vault watcher + wikilink graph builder
├── storage/                # ChromaDB, BM25, document registry
├── mcp_servers/            # MCP protocol server (FastMCP)
├── telemetry/              # Query logging and stats
├── config.yaml             # All tunable parameters
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Requirements

| | Minimum | Recommended |
|---|---|---|
| **Python** | 3.12 | 3.12+ |
| **RAM** | 4 GB | 8 GB+ |
| **Disk** | 3 GB (models + index) | 10 GB+ |
| **GPU** | CPU only | NVIDIA GPU with CUDA |
| **Embedding model** | Any Ollama model | `qwen3-embedding:0.6b` |
| **Reranker model** | — | `Qwen/Qwen3-Reranker-0.6B` (downloads on first use) |

---

## Tech stack

- **FastAPI** — REST API framework
- **FastMCP** — MCP protocol server
- **ChromaDB** — Vector database
- **BM25S** — Sparse keyword retrieval
- **Sentence-Transformers** — Cross-encoder reranker
- **MarkItDown (Microsoft)** — Document ingestion
- **Ollama** — Embedding inference
- **Docker** — Containerization

---

## License

MIT
