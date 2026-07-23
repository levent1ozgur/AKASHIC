# AKASHIC — Knowledge Archive & Semantic Hybrid Indexing Core

> **A local-first, headless knowledge retrieval service for self-hosted AI systems, combining dense and sparse retrieval, cross-encoder reranking, confidence gating, and Obsidian vault synchronization.**

AKASHIC turns your documents, notes, and research into a searchable knowledge base that is served through a REST API or MCP to agents, LLMs, and other applications. Everything runs on your own hardware, with no cloud calls at query time.

---

## Why AKASHIC exists

Most RAG projects are frameworks. They provide components and expect you to build the system around them. AKASHIC takes the opposite approach. It is a **finished, deployable knowledge retrieval service** with:

* A measured retrieval baseline of 98% recall across 51 test cases
* Validated document lifecycle operations covering create, modify, rename, and delete
* A reproducible evaluation suite for detecting regressions
* Documented pipeline version metadata for index auditing

It is built for personal knowledge infrastructure: your own notes, documents, and research, served by a self-hosted pipeline that never phones home.

---

## Architecture overview

```text
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

1. **Ingestion** — Parses documents (PDF, DOCX, MD, etc.), chunks them with structural context, embeds each chunk, and writes to dense (ChromaDB) and sparse (BM25S) indices.
2. **Retrieval** — Routes queries across both indices, fuses results via RRF, reranks with a cross-encoder for ranking only, and gates answerability using dense cosine distance as a separate confidence signal.
3. **Lifecycle** — An inotify-based vault watcher keeps the index synchronized with filesystem changes, including create, modify, rename, and delete events.

**[Read the full architecture →](ARCHITECTURE.md)**

---

## Core capabilities

| Capability                  | Implementation                                                                                                         |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Multi-format ingestion**  | PDF, DOCX, PPTX, XLSX, HTML, EPUB, MD, TXT via [MarkItDown](https://github.com/microsoft/markitdown)                   |
| **Hybrid retrieval**        | Dense embeddings (ChromaDB) + sparse keywords (BM25S) fused via RRF                                                    |
| **Cross-encoder reranking** | [Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) with optional GPU acceleration and CPU fallback |
| **Confidence gating**       | Dense cosine distance with a threshold of 0.55; reranker scores are not used because they are uncalibrated             |
| **Obsidian vault sync**     | Inotify-based watcher that automatically indexes, updates, renames, and removes vault content                          |
| **Context enrichment**      | `heading_path` for structured documents and `section_context` for flat PDF conversions                                 |
| **Pipeline versioning**     | `chunking_version`, `embedding_model`, and `context_enrichment_version` in every chunk's metadata                      |
| **Query cache**             | Configurable TTL with automatic invalidation on vault changes                                                          |
| **MCP protocol**            | Tools: `search_knowledge_base`, `ingest_document`, `pipeline_status`                                                   |
| **REST API**                | FastAPI endpoints for querying, ingestion, status, and administration                                                  |
| **HyDE**                    | Hypothetical Document Embeddings for generate-then-retrieve workflows                                                  |
| **Graph expansion**         | Wikilink graph extraction from vault notes for related-context retrieval                                               |

---

## Engineering validation

AKASHIC includes a reproducible evaluation suite and automated lifecycle tests.

### Retrieval evaluation: 51 cases

```text
51 test cases
├── 30 vault queries (daily logs, setup, templates)
├── 12 document queries (Alice in Wonderland)
├── 4 setup queries
├── 4 edge cases (unanswerable / policy / ambiguity)
└── 1 template exclusion (policy)

Results: 98% overall Recall@K
All currently indexed answerable retrieval cases pass.
Policy-excluded and expected-unanswerable cases are evaluated separately.
```

Metrics tracked: Recall@K, Precision@K, and MRR, reported both per category and in aggregate.

### Lifecycle tests: 5/5

```text
CREATE  → note appears in search   ✓
MODIFY  → old content gone, new    ✓
RENAME  → searchable at new path   ✓
DELETE  → content not retrievable  ✓
```

Lifecycle testing validates index consistency across ChromaDB, BM25S, the document registry, and the query cache.

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

Starts the REST API on port **8765** and the MCP server on port **8766**.

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

| Method   | Path                 | Description                         |
| -------- | -------------------- | ----------------------------------- |
| `POST`   | `/ingest`            | Upload a document                   |
| `POST`   | `/query`             | Query the knowledge base            |
| `GET`    | `/status`            | Pipeline health and document counts |
| `GET`    | `/status/{doc_id}`   | Status for a specific document      |
| `DELETE` | `/document/{doc_id}` | Remove a document                   |

---

## MCP

Register AKASHIC in any MCP-compatible client:

```text
URL: http://host:8766/mcp
Transport: streamable-http
```

Tools: `search_knowledge_base`, `ingest_document`, `pipeline_status`

---

## Configuration

Environment variables override `config.yaml` at runtime. Set them in `.env` or in the shell.

| Variable                     | Default                    | Description                            |
| ---------------------------- | -------------------------- | -------------------------------------- |
| `OLLAMA_BASE_URL`            | `http://localhost:11434`   | Embeddings endpoint                    |
| `EMBEDDING_MODEL`            | `qwen3-embedding:0.6b`     | Embedding model                        |
| `RERANKER_MODEL`             | `Qwen/Qwen3-Reranker-0.6B` | Reranker model                         |
| `VAULT_PATH`                 | —                          | Obsidian vault directory               |
| `DENSE_CONFIDENCE_THRESHOLD` | `0.55`                     | Dense cosine threshold for `not_found` |
| `MAX_FILE_SIZE_MB`           | `50`                       | Maximum upload size                    |

---

## Project scope & maintenance

AKASHIC is built primarily for my own personal knowledge infrastructure and self-hosted AI systems. It exists to solve problems I encounter in my own environment and to serve as a component of the systems I build.

Development priorities are therefore driven by my own needs. If I need a new feature, encounter a bug that affects my usage, or change how my infrastructure works, I will improve AKASHIC accordingly. If I do not need something myself, I may not implement it, regardless of how useful it might be to someone else.

This means AKASHIC is not maintained as a product with a public roadmap, guaranteed response times, or a commitment to implement feature requests. Issues and pull requests may be reviewed when they align with my own needs and priorities, but there is no promise of ongoing maintenance for external use cases.

That is intentional. AKASHIC is infrastructure I build for myself and share publicly because others may find it useful.

If you need functionality that falls outside my priorities, you are free to fork the project and adapt it to your own requirements.

It is:

* **Measured** — A 51-case evaluation suite detects regressions before they affect usage
* **Lifecycle-validated** — Create, modify, rename, and delete operations are tested
* **Self-hosted** — All components run on your own hardware, including consumer GPUs with 4–6 GB of VRAM

It is not:

* A general-purpose RAG framework
* A multi-tenant or horizontally scalable knowledge service
* A product with a public feature roadmap or guaranteed maintenance

---

## Tech stack

FastAPI · ChromaDB · BM25S · Sentence-Transformers · MarkItDown · Ollama · FastMCP · Docker

## License

MIT
