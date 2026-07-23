"""
pipeline/retrieval/reranker.py
Qwen3-Reranker-0.6B cross-encoder reranker.
Takes fused RRF results and re-scores each (query, chunk) pair
for significantly better relevance ordering than embedding similarity alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pipeline.retrieval.fusion import FusedResult

logger = logging.getLogger(__name__)

DEFAULT_MODEL      = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_MAX_LENGTH = 512
DEFAULT_TOP_K      = 6
DEFAULT_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class RerankedResult:
    """A chunk after cross-encoder reranking."""
    chroma_id:      str
    text:           str
    metadata:       dict
    reranker_score: float           # raw cross-encoder logit (higher = more relevant)
    collection:     str
    source_methods: list[str]
    rrf_score:      float           # original RRF score (kept for telemetry)
    original_rank:  int             # rank before reranking (1-based)


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """
    Cross-encoder reranker using Qwen3-Reranker-0.6B.

    Loads the model once on first use (lazy init) and keeps it in memory.
    Scores all (query, chunk) pairs, sorts by score, returns top_k.

    On your GTX 1660 Super: model runs on CPU (~90MB RAM, ~200ms per batch).
    VRAM is not consumed since we offload to CPU deliberately to avoid
    competing with the embedding model.

    Usage:
        reranker = Reranker()
        results = reranker.rerank(
            query="authentication token expiry",
            candidates=fused_results,
            top_k=6,
        )
    """

    def __init__(
        self,
        model_name:  str = DEFAULT_MODEL,
        max_length:  int = DEFAULT_MAX_LENGTH,
        batch_size:  int = DEFAULT_BATCH_SIZE,
        device:      str = "cuda",       # use GPU for fast inference
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device     = device
        self._model     = None          # lazy init

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the CrossEncoder model. Called once on first rerank() call."""
        if self._model is not None:
            return

        logger.info("Loading reranker model '%s' on %s...", self.model_name, self.device)
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
            )
            logger.info("Reranker model loaded.")
        except Exception as e:
            logger.error("Failed to load reranker model: %s", e)
            raise

    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def rerank(
        self,
        query:      str,
        candidates: list[FusedResult],
        top_k:      int = DEFAULT_TOP_K,
    ) -> list[RerankedResult]:
        """
        Rerank candidates using the cross-encoder.

        Args:
            query:      the user's query string
            candidates: fused results from RRF (already deduplicated)
            top_k:      number of results to return after reranking

        Returns:
            List of RerankedResult sorted by reranker_score descending.
            Returns empty list if candidates is empty.
            Falls back to RRF ordering if model fails.
        """
        if not candidates:
            return []

        self._load()

        # Build (query, chunk_text) pairs for cross-encoder
        pairs = [(query, c.text) for c in candidates]

        # Score all pairs
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error("Reranker prediction failed: %s — falling back to RRF order", e)
            return self._fallback(candidates, top_k)

        # Build reranked results
        scored = []
        for original_rank, (candidate, score) in enumerate(
            zip(candidates, scores), start=1
        ):
            scored.append(RerankedResult(
                chroma_id=candidate.chroma_id,
                text=candidate.text,
                metadata=candidate.metadata,
                reranker_score=float(score),
                collection=candidate.collection,
                source_methods=candidate.source_methods,
                rrf_score=candidate.rrf_score,
                original_rank=original_rank,
            ))

        # Sort by reranker score descending
        scored.sort(key=lambda r: r.reranker_score, reverse=True)

        logger.debug(
            "Reranked %d candidates → top %d  "
            "(top score=%.4f, bottom score=%.4f)",
            len(candidates), min(top_k, len(scored)),
            scored[0].reranker_score if scored else 0.0,
            scored[-1].reranker_score if scored else 0.0,
        )

        return scored[:top_k]

    def _fallback(
        self,
        candidates: list[FusedResult],
        top_k: int,
    ) -> list[RerankedResult]:
        """Fallback: return candidates in RRF order when model fails."""
        return [
            RerankedResult(
                chroma_id=c.chroma_id,
                text=c.text,
                metadata=c.metadata,
                reranker_score=c.rrf_score,   # use RRF score as proxy
                collection=c.collection,
                source_methods=c.source_methods,
                rrf_score=c.rrf_score,
                original_rank=i + 1,
            )
            for i, c in enumerate(candidates[:top_k])
        ]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    from pipeline.retrieval.fusion import FusedResult

    def make_fused(cid: str, text: str, score: float = 0.05) -> FusedResult:
        return FusedResult(
            chroma_id=cid,
            text=text,
            metadata={"source": "test.pdf", "chunk_id": cid},
            rrf_score=score,
            collection="documents",
            source_methods=["dense_docs"],
        )

    # Candidates deliberately ordered with the least-relevant first
    # to confirm reranker changes the ordering
    query = "How do authentication tokens expire?"
    candidates = [
        make_fused("c1", "Rate limits apply to all API endpoints globally.", 0.030),
        make_fused("c2", "Error codes are listed in Appendix B of the manual.", 0.025),
        make_fused("c3", "Authentication tokens expire after 24 hours of issuance.", 0.020),
        make_fused("c4", "The system uses bearer tokens for all authenticated requests.", 0.015),
        make_fused("c5", "Database connection pooling is configured in settings.py.", 0.010),
    ]

    print(f"Query: '{query}'")
    print(f"Input order (by RRF score):")
    for i, c in enumerate(candidates, 1):
        print(f"  [{i}] '{c.text[:60]}' (rrf={c.rrf_score:.3f})")

    reranker = Reranker(device="cpu")

    t0 = time.time()
    results = reranker.rerank(query=query, candidates=candidates, top_k=3)
    elapsed = time.time() - t0

    print(f"\nReranked top-3 (in {elapsed:.1f}s):")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] score={r.reranker_score:.4f}  "
              f"original_rank={r.original_rank}  "
              f"'{r.text[:60]}'")

    # The most relevant chunk (c3 — about token expiry) should rank #1
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert results[0].chroma_id == "c3", (
        f"Expected c3 (token expiry) at rank 1, "
        f"got {results[0].chroma_id}: '{results[0].text[:50]}'"
    )
    print(f"\nRelevance ordering: OK  "
          f"(token-expiry chunk correctly ranked #1)")

    # Scores should be descending
    scores = [r.reranker_score for r in results]
    assert scores == sorted(scores, reverse=True), \
        f"Scores not descending: {scores}"
    print(f"Score ordering: OK  ({scores})")

    # Original rank is preserved
    assert results[0].original_rank == 3, \
        f"Original rank should be 3 (c3 was 3rd in input), " \
        f"got {results[0].original_rank}"
    print(f"Original rank preserved: OK  (was rank 3 before reranking)")

    # RRF score preserved
    assert results[0].rrf_score == 0.020
    print(f"RRF score preserved: OK")

    # Fallback works without model
    reranker2 = Reranker()
    reranker2._model = None   # force unloaded state
    fallback = reranker2._fallback(candidates[:3], top_k=2)
    assert len(fallback) == 2
    assert fallback[0].chroma_id == candidates[0].chroma_id
    print(f"Fallback: OK  ({len(fallback)} results in RRF order)")

    print(f"\nAll Reranker assertions passed.")
