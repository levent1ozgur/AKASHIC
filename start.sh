#!/bin/bash
# AKASHIC startup script — starts REST API and optional MCP server.
set -e

# Accept MCP port from env (default 8766)
export MCP_PORT="${MCP_PORT:-8766}"

echo "Starting AKASHIC REST API on port ${API_PORT:-8765}..."
uvicorn api.main:app --host 0.0.0.0 --port "${API_PORT:-8765}" --log-level info &

API_PID=$!

# Start MCP server if the module exists
if python3 -c "import mcp_servers.rag_pipeline_server" 2>/dev/null; then
    echo "Starting AKASHIC MCP server on port $MCP_PORT..."
    python3 -m uvicorn mcp_servers.rag_pipeline_server:mcp_app         --host 0.0.0.0 --port "$MCP_PORT" --log-level info &
    MCP_PID=$!
else
    echo "MCP server module not found — skipping."
    MCP_PID=""
fi

# Trap to clean up both on exit
cleanup() {
    echo "Shutting down..."
    kill "$API_PID" 2>/dev/null
    [ -n "$MCP_PID" ] && kill "$MCP_PID" 2>/dev/null
    wait
    echo "Done."
}
trap cleanup SIGTERM SIGINT

# Wait for either to exit
wait
