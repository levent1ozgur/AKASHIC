"""
telemetry/recorder.py
Async SQLite telemetry recorder.
Logs per-query metrics without blocking the retrieval path.
Queries are hashed for privacy — raw query text is never stored.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry record
# ---------------------------------------------------------------------------

@dataclass
class TelemetryRecord:
    """Metrics captured for a single retrieval request."""
    query_hash:             str         # SHA-256 of query (privacy-safe)
    timestamp:              str         # ISO-8601 UTC
    router_decision:        str         # "docs" | "vault" | "both"
    cache_hit:              bool
    hyde_used:              bool
    metadata_filter_applied: bool
    graph_expansion_used:   bool
    dense_results_count:    int
    sparse_results_count:   int
    fused_results_count:    int
    reranked_results_count: int
    top_confidence_score:   float
    expansion_triggered:    bool
    not_found:              bool
    compression_used:       bool
    final_chunk_count:      int
    final_token_count:      int
    latency_ms:             int


def make_record(
    query:                  str,
    router_decision:        str         = "both",
    cache_hit:              bool        = False,
    hyde_used:              bool        = False,
    metadata_filter_applied: bool       = False,
    graph_expansion_used:   bool        = False,
    dense_results_count:    int         = 0,
    sparse_results_count:   int         = 0,
    fused_results_count:    int         = 0,
    reranked_results_count: int         = 0,
    top_confidence_score:   float       = 0.0,
    expansion_triggered:    bool        = False,
    not_found:              bool        = False,
    compression_used:       bool        = False,
    final_chunk_count:      int         = 0,
    final_token_count:      int         = 0,
    latency_ms:             int         = 0,
) -> TelemetryRecord:
    """Convenience constructor — hashes the query automatically."""
    return TelemetryRecord(
        query_hash=_hash_query(query),
        timestamp=datetime.now(timezone.utc).isoformat(),
        router_decision=router_decision,
        cache_hit=cache_hit,
        hyde_used=hyde_used,
        metadata_filter_applied=metadata_filter_applied,
        graph_expansion_used=graph_expansion_used,
        dense_results_count=dense_results_count,
        sparse_results_count=sparse_results_count,
        fused_results_count=fused_results_count,
        reranked_results_count=reranked_results_count,
        top_confidence_score=top_confidence_score,
        expansion_triggered=expansion_triggered,
        not_found=not_found,
        compression_used=compression_used,
        final_chunk_count=final_chunk_count,
        final_token_count=final_token_count,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class TelemetryRecorder:
    """
    Async, non-blocking telemetry recorder backed by SQLite.

    Records are queued in memory and written to disk by a background thread,
    so the retrieval path is never blocked waiting for disk I/O.

    Usage:
        recorder = TelemetryRecorder("data/telemetry.db")
        recorder.start()

        record = make_record(query="auth tokens", latency_ms=412, ...)
        recorder.record(record)     # non-blocking

        recorder.stop()             # flush and close
    """

    def __init__(
        self,
        db_path:    str = "data/telemetry.db",
        queue_size: int = 1000,
    ):
        self.db_path   = Path(db_path)
        self._queue:   queue.Queue = queue.Queue(maxsize=queue_size)
        self._thread:  Optional[threading.Thread] = None
        self._stop:    threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Create the database schema and start the background writer thread."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._writer,
            name="telemetry-writer",
            daemon=True,
        )
        self._thread.start()
        logger.info("TelemetryRecorder started → %s", self.db_path)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the writer to stop and wait for the queue to drain."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("TelemetryRecorder stopped.")

    # ------------------------------------------------------------------
    # Public write interface
    # ------------------------------------------------------------------

    def record(self, rec: TelemetryRecord) -> None:
        """
        Enqueue a record for async writing.
        Non-blocking — drops the record if the queue is full (logged as warning).
        """
        try:
            self._queue.put_nowait(rec)
        except queue.Full:
            logger.warning(
                "Telemetry queue full — dropping record for query_hash=%s",
                rec.query_hash[:8],
            )

    # ------------------------------------------------------------------
    # Background writer
    # ------------------------------------------------------------------

    def _writer(self) -> None:
        """Background thread: drain queue and write to SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            while not self._stop.is_set() or not self._queue.empty():
                try:
                    rec = self._queue.get(timeout=0.5)
                    self._write_one(conn, rec)
                    self._queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error("Telemetry write error: %s", e)
        finally:
            conn.close()

    def _write_one(self, conn: sqlite3.Connection, rec: TelemetryRecord) -> None:
        conn.execute(
            """
            INSERT INTO queries (
                query_hash, timestamp, router_decision,
                cache_hit, hyde_used, metadata_filter_applied,
                graph_expansion_used, dense_results_count,
                sparse_results_count, fused_results_count,
                reranked_results_count, top_confidence_score,
                expansion_triggered, not_found, compression_used,
                final_chunk_count, final_token_count, latency_ms
            ) VALUES (
                :query_hash, :timestamp, :router_decision,
                :cache_hit, :hyde_used, :metadata_filter_applied,
                :graph_expansion_used, :dense_results_count,
                :sparse_results_count, :fused_results_count,
                :reranked_results_count, :top_confidence_score,
                :expansion_triggered, :not_found, :compression_used,
                :final_chunk_count, :final_token_count, :latency_ms
            )
            """,
            asdict(rec),
        )
        conn.commit()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                query_hash              TEXT NOT NULL,
                timestamp               TEXT NOT NULL,
                router_decision         TEXT NOT NULL,
                cache_hit               INTEGER NOT NULL DEFAULT 0,
                hyde_used               INTEGER NOT NULL DEFAULT 0,
                metadata_filter_applied INTEGER NOT NULL DEFAULT 0,
                graph_expansion_used    INTEGER NOT NULL DEFAULT 0,
                dense_results_count     INTEGER NOT NULL DEFAULT 0,
                sparse_results_count    INTEGER NOT NULL DEFAULT 0,
                fused_results_count     INTEGER NOT NULL DEFAULT 0,
                reranked_results_count  INTEGER NOT NULL DEFAULT 0,
                top_confidence_score    REAL    NOT NULL DEFAULT 0.0,
                expansion_triggered     INTEGER NOT NULL DEFAULT 0,
                not_found               INTEGER NOT NULL DEFAULT 0,
                compression_used        INTEGER NOT NULL DEFAULT 0,
                final_chunk_count       INTEGER NOT NULL DEFAULT 0,
                final_token_count       INTEGER NOT NULL DEFAULT 0,
                latency_ms              INTEGER NOT NULL DEFAULT 0
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries(timestamp)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_queries_hash ON queries(query_hash)"
        )
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Read / analytics
    # ------------------------------------------------------------------

    def summary(self, last_n: int = 100) -> dict:
        """
        Return aggregate metrics over the last N queries.
        Used by the /status API endpoint.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(f"""
                SELECT
                    COUNT(*)                        AS total_queries,
                    AVG(latency_ms)                 AS avg_latency_ms,
                    SUM(cache_hit)                  AS cache_hits,
                    SUM(not_found)                  AS not_found_count,
                    AVG(top_confidence_score)       AS avg_confidence,
                    AVG(final_chunk_count)          AS avg_chunks,
                    SUM(expansion_triggered)        AS expansions_triggered,
                    SUM(hyde_used)                  AS hyde_used_count,
                    SUM(graph_expansion_used)       AS graph_used_count
                FROM (
                    SELECT * FROM queries
                    ORDER BY id DESC
                    LIMIT {last_n}
                )
            """).fetchone()

            total = row["total_queries"] or 0
            return {
                "total_queries":        total,
                "avg_latency_ms":       round(row["avg_latency_ms"] or 0, 1),
                "cache_hit_rate":       round((row["cache_hits"] or 0) / max(total, 1), 3),
                "not_found_rate":       round((row["not_found_count"] or 0) / max(total, 1), 3),
                "avg_confidence":       round(row["avg_confidence"] or 0, 3),
                "avg_chunks_per_query": round(row["avg_chunks"] or 0, 1),
                "expansions_triggered": row["expansions_triggered"] or 0,
                "hyde_used_count":      row["hyde_used_count"] or 0,
                "graph_used_count":     row["graph_used_count"] or 0,
            }
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Privacy helper
# ---------------------------------------------------------------------------

def _hash_query(query: str) -> str:
    """SHA-256 hex digest of the query string (truncated to 16 chars)."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, time

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        db_path = f"{tmp}/telemetry.db"
        recorder = TelemetryRecorder(db_path=db_path)
        recorder.start()

        # Write several records
        queries = [
            ("authentication token expiry", "docs",  False, 9, 412),
            ("my notes on chunking",        "vault", False, 6, 198),
            ("rate limits api",             "both",  True,  0, 12),   # cache hit
            ("kubernetes deployment yaml",  "docs",  False, 5, 523),
            ("",                            "both",  False, 0, 8),    # empty query
        ]

        for q, route, cache, chunks, latency in queries:
            rec = make_record(
                query=q,
                router_decision=route,
                cache_hit=cache,
                final_chunk_count=chunks,
                final_token_count=chunks * 128,
                latency_ms=latency,
                top_confidence_score=0.85 if chunks > 0 else 0.0,
                not_found=(chunks == 0),
            )
            recorder.record(rec)

        # Give writer time to flush
        time.sleep(0.5)
        recorder.stop()

        # Verify records written
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        conn.close()

        assert count == 5, f"Expected 5 records, got {count}"
        print(f"Records written: {count}  OK")

        # Privacy: verify raw query not stored
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT query_hash FROM queries").fetchall()
        conn.close()
        for (h,) in rows:
            assert len(h) == 16, f"Hash length wrong: {len(h)}"
            assert "authentication" not in h
        print(f"Privacy (no raw query stored): OK")

        # Summary
        recorder2 = TelemetryRecorder(db_path=db_path)
        s = recorder2.summary(last_n=100)
        assert s["total_queries"] == 5
        assert s["cache_hit_rate"] == 0.2   # 1/5
        assert s["not_found_rate"] > 0, f"not_found_rate should be > 0: {s['not_found_rate']}"
        assert s["avg_latency_ms"] > 0
        print(f"Summary: OK  {s}")

        # Queue full → drop gracefully (no crash)
        tiny = TelemetryRecorder(db_path=db_path, queue_size=2)
        tiny.start()
        for i in range(10):   # overflow the queue
            tiny.record(make_record(f"query {i}", latency_ms=1))
        time.sleep(0.3)
        tiny.stop()
        print(f"Queue overflow: OK  (no crash)")

        print(f"\nAll TelemetryRecorder assertions passed.")
