"""
mcp_servers/rag_pipeline_server.py
MCP server that exposes your RAG pipeline to AKASHIC.

Run on host:
    python3.12 mcp_servers/rag_pipeline_server.py

Register in AKASHIC → Settings → Integrations → Add MCP Server:
    Transport: http (or streamable-http)
    URL: http://host.docker.internal:8766/mcp
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAG_BASE_URL = "http://localhost:8765"
TIMEOUT      = 120.0
MCP_HOST     = "0.0.0.0"
MCP_PORT     = 8766

# ---------------------------------------------------------------------------
# MCP server — host/port passed at construction time (required by mcp 1.x)
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="odysseus-rag-pipeline",
    host=MCP_HOST,
    port=MCP_PORT,
    instructions=(
        "RAG pipeline for searching personal documents and Obsidian vault notes. "
        "Use search_knowledge_base when the user asks about their documents, notes, "
        "research, or anything that might be in their knowledge base. "
        "Use ingest_document to add new files to the knowledge base. "
        "Use pipeline_status to check how many documents are indexed."
    ),
)


@mcp.tool()
async def search_knowledge_base(
    query: str,
    route: str = "both",
    top_k: int = 6,
) -> str:
    """
    Search the personal knowledge base (uploaded documents + Obsidian vault).

    Use this when the user asks about anything in their documents, PDFs,
    reports, notes, research, decisions, or project notes.

    Args:
        query: The search query — use the user's question as-is
        route: Where to search: "docs" (uploaded files only),
               "vault" (Obsidian notes only), or "both" (default)
        top_k: Number of results to retrieve (1-10, default 6)

    Returns:
        JSON with retrieved chunks, assembled prompt, and confidence score.
    """
    if route not in ("docs", "vault", "both"):
        route = "both"
    top_k = max(1, min(10, top_k))

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
                f"{RAG_BASE_URL}/query",
                json={"query": query, "route": route, "top_k": top_k},
            )
            response.raise_for_status()
            data = response.json()

            if data.get("not_found"):
                return json.dumps({
                    "found": False,
                    "message": (
                        f"No relevant information found for: '{query}'. "
                        "The document may not be ingested yet, or try rephrasing."
                    ),
                })

            return json.dumps({
                "found":       True,
                "confidence":  round(data.get("confidence", 0), 3),
                "route_used":  data.get("route_used"),
                "chunk_count": len(data.get("chunks", [])),
                "latency_ms":  data.get("latency_ms"),
                "chunks": [
                    {
                        "source":     c["source"],
                        "section":    c["heading_path"],
                        "collection": c["collection"],
                        "confidence": round(c["confidence"], 3),
                    }
                    for c in data.get("chunks", [])
                ],
                "prompt": data.get("prompt", ""),
            })

        except httpx.ConnectError:
            return json.dumps({"error": f"Cannot connect to RAG pipeline at {RAG_BASE_URL}."})
        except Exception as e:
            return json.dumps({"error": f"RAG query failed: {e}"})


@mcp.tool()
async def ingest_document(file_path: str) -> str:
    """
    Add a document to the knowledge base for future retrieval.
    Supports: PDF, DOCX, PPTX, XLSX, HTML, EPUB, MD, TXT

    Args:
        file_path: Absolute path to the file, e.g. /home/user/Downloads/report.pdf
    """
    path = Path(file_path).expanduser()
    if not path.exists():
        return json.dumps({"error": f"File not found: {file_path}"})

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            with open(path, "rb") as f:
                response = await client.post(
                    f"{RAG_BASE_URL}/ingest",
                    files={"file": (path.name, f, "application/octet-stream")},
                )
            response.raise_for_status()
            data = response.json()
            return json.dumps({"doc_id": data.get("doc_id"), "status": data.get("status"),
                               "message": data.get("message"), "file": path.name})
        except httpx.ConnectError:
            return json.dumps({"error": f"Cannot connect to RAG pipeline at {RAG_BASE_URL}."})
        except Exception as e:
            return json.dumps({"error": f"Ingest failed: {e}"})


@mcp.tool()
async def pipeline_status() -> str:
    """Check the RAG pipeline health and document index counts."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{RAG_BASE_URL}/status")
            response.raise_for_status()
            return json.dumps(response.json(), indent=2)
        except httpx.ConnectError:
            return json.dumps({"error": f"RAG pipeline not reachable at {RAG_BASE_URL}."})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if "--stdio" in sys.argv:
        print("Running in stdio mode", flush=True)
        mcp.run(transport="stdio")
    else:
        print(f"Starting RAG MCP server → http://{MCP_HOST}:{MCP_PORT}")
        print(f"RAG pipeline at: {RAG_BASE_URL}")
        print(f"Register in AKASHIC: http://host.docker.internal:{MCP_PORT}/mcp")
        mcp.run(transport="streamable-http")
