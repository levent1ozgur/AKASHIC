"""
pipeline/retrieval/fusion.py
Reciprocal Rank Fusion (RRF) for merging dense, sparse,
and graph-expanded retrieval results into a single ranked list.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from storage.chroma import QueryResult as DenseResult
from storage.bm25 import BM25Result as SparseResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RRF constant (standard default — higher k = less aggressive rank boosting)
# ---------------------------------------------------------------------------

DEFAULT_RRF_K = 60


# ---------------------------------------------------------------------------
# Unified result
# ---------------------------------------------------------------------------

@dataclass
class FusedResult:
    """A single result after RRF merging across all retrieval methods."""
    chroma_id:      str
    text:           str
    metadata:       dict
    rrf_score:      float
    collection:     str             # "documents" | "vault"
    source_methods: list[str]       = field(default_factory=list)
    # Per-method ranks (for debugging / telemetry)
    dense_rank:     Optional[int]   = None
    sparse_rank:    Optional[int]   = None
    graph_rank:     Optional[int]   = None


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

class RecipRankFusion:
    """
    Merges multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score for a document d:
        score(d) = Σ  1 / (k + rank_i(d))
    where rank_i(d) is the 1-based rank of d in result list i,
    and k is a smoothing constant (default 60).

    Documents not present in a list get no contribution from that list.

    Usage:
        fusion = RecipRankFusion(k=60)
        results = fusion.fuse(
            dense_docs=[...],
            sparse_docs=[...],
            dense_vault=[...],
            sparse_vault=[...],
            graph_expanded=[...],
            top_k=20,
        )
    """

    def __init__(self, k: int = DEFAULT_RRF_K):
        self.k = k

    def fuse(
        self,
        dense_docs:     list[DenseResult]  = None,
        sparse_docs:    list[SparseResult] = None,
        dense_vault:    list[DenseResult]  = None,
        sparse_vault:   list[SparseResult] = None,
        graph_expanded: list[DenseResult]  = None,
        top_k:          int = 20,
    ) -> list[FusedResult]:
        """
        Merge result lists from all retrieval methods.

        Args:
            dense_docs:     semantic search results from documents collection
            sparse_docs:    BM25 results from documents index
            dense_vault:    semantic search results from vault collection
            sparse_vault:   BM25 results from vault index
            graph_expanded: vault notes added via graph expansion
            top_k:          number of results to return

        Returns:
            Deduplicated list of FusedResult sorted by RRF score descending.
        """
        dense_docs     = dense_docs     or []
        sparse_docs    = sparse_docs    or []
        dense_vault    = dense_vault    or []
        sparse_vault   = sparse_vault   or []
        graph_expanded = graph_expanded or []

        # Accumulator: chroma_id → accumulated RRF score + metadata
        scores:   dict[str, float] = {}
        texts:    dict[str, str]   = {}
        metas:    dict[str, dict]  = {}
        colls:    dict[str, str]   = {}
        methods:  dict[str, list[str]] = {}
        d_ranks:  dict[str, int]   = {}
        s_ranks:  dict[str, int]   = {}
        g_ranks:  dict[str, int]   = {}

        def _accumulate(
            results: list,
            method: str,
            rank_store: Optional[dict] = None,
        ) -> None:
            for rank, result in enumerate(results, start=1):
                cid = _get_chroma_id(result)
                if not cid:
                    continue

                rrf = 1.0 / (self.k + rank)
                scores[cid]  = scores.get(cid, 0.0) + rrf
                texts[cid]   = _get_text(result)
                metas[cid]   = _get_metadata(result)
                colls[cid]   = _get_collection(result)

                if cid not in methods:
                    methods[cid] = []
                if method not in methods[cid]:
                    methods[cid].append(method)

                if rank_store is not None and cid not in rank_store:
                    rank_store[cid] = rank

        _accumulate(dense_docs,     "dense_docs",     d_ranks)
        _accumulate(sparse_docs,    "sparse_docs",    s_ranks)
        _accumulate(dense_vault,    "dense_vault",    d_ranks)
        _accumulate(sparse_vault,   "sparse_vault",   s_ranks)
        _accumulate(graph_expanded, "graph_expanded", g_ranks)

        # Sort by RRF score descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused: list[FusedResult] = []
        for cid, score in ranked[:top_k]:
            fused.append(FusedResult(
                chroma_id=cid,
                text=texts[cid],
                metadata=metas.get(cid, {}),
                rrf_score=score,
                collection=colls.get(cid, ""),
                source_methods=methods.get(cid, []),
                dense_rank=d_ranks.get(cid),
                sparse_rank=s_ranks.get(cid),
                graph_rank=g_ranks.get(cid),
            ))

        logger.debug(
            "RRF fusion: %d dense_docs + %d sparse_docs + %d dense_vault "
            "+ %d sparse_vault + %d graph → %d unique → top %d",
            len(dense_docs), len(sparse_docs), len(dense_vault),
            len(sparse_vault), len(graph_expanded),
            len(scores), len(fused),
        )

        return fused


# ---------------------------------------------------------------------------
# Result field accessors (handle both DenseResult and SparseResult)
# ---------------------------------------------------------------------------

def _get_chroma_id(result) -> str:
    if hasattr(result, "chroma_id"):
        return result.chroma_id
    return ""


def _get_text(result) -> str:
    if hasattr(result, "text"):
        return result.text
    return ""


def _get_metadata(result) -> dict:
    if hasattr(result, "metadata"):
        return result.metadata
    return {}


def _get_collection(result) -> str:
    if hasattr(result, "collection"):
        return result.collection
    if hasattr(result, "index"):
        # BM25Result uses "index" not "collection"
        idx = result.index
        return "vault" if idx == "vault" else "documents"
    return ""


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.DEBUG)

    from storage.chroma import QueryResult as DR
    from storage.bm25   import BM25Result  as SR

    fusion = RecipRankFusion(k=60)

    def make_dense(cid, text, dist=0.1, coll="documents") -> DR:
        return DR(
            chunk_id=cid, chroma_id=cid, text=text,
            metadata={"source": "test.pdf", "chunk_id": cid},
            distance=dist, collection=coll,
        )

    def make_sparse(cid, text, score=1.0, idx="documents") -> SR:
        return SR(
            chroma_id=cid, text=text,
            metadata={"source": "test.pdf", "chunk_id": cid},
            score=score, index=idx,
        )

    # -----------------------------------------------------------------------
    # Test 1: Basic merge — doc appearing in both dense and sparse gets boosted
    # -----------------------------------------------------------------------
    dense  = [make_dense("a", "Authentication tokens"), make_dense("b", "Rate limits")]
    sparse = [make_sparse("a", "Authentication tokens"), make_sparse("c", "Error codes")]

    results = fusion.fuse(dense_docs=dense, sparse_docs=sparse, top_k=10)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    # "a" appears in both lists → highest RRF score
    assert results[0].chroma_id == "a", \
        f"Expected 'a' first (appears in both), got '{results[0].chroma_id}'"
    assert "dense_docs"  in results[0].source_methods
    assert "sparse_docs" in results[0].source_methods
    print(f"Test 1 — basic merge: OK")
    print(f"  Top result: '{results[0].chroma_id}' "
          f"score={results[0].rrf_score:.4f} "
          f"methods={results[0].source_methods}")

    # -----------------------------------------------------------------------
    # Test 2: Deduplication — same chroma_id from different lists = one result
    # -----------------------------------------------------------------------
    dense2  = [make_dense("x", "Chunking strategy", coll="vault")]
    vault2  = [make_dense("x", "Chunking strategy", coll="vault")]
    results2 = fusion.fuse(dense_docs=dense2, dense_vault=vault2, top_k=10)
    assert len(results2) == 1, f"Dedup failed — got {len(results2)} results"
    assert results2[0].chroma_id == "x"
    print(f"Test 2 — deduplication: OK  (1 result after dedup)")

    # -----------------------------------------------------------------------
    # Test 3: Cross-source ranking — vault + docs interleaved correctly
    # -----------------------------------------------------------------------
    doc_results   = [make_dense(f"d{i}", f"Doc chunk {i}") for i in range(5)]
    vault_results = [make_dense(f"v{i}", f"Vault note {i}", coll="vault")
                     for i in range(5)]
    results3 = fusion.fuse(
        dense_docs=doc_results,
        dense_vault=vault_results,
        top_k=20,
    )
    assert len(results3) == 10
    collections = {r.collection for r in results3}
    assert "documents" in collections
    assert "vault"     in collections
    print(f"Test 3 — cross-source: OK  ({len(results3)} results, "
          f"collections={collections})")

    # -----------------------------------------------------------------------
    # Test 4: Graph expansion results are included
    # -----------------------------------------------------------------------
    graph = [make_dense("g1", "Linked vault note", coll="vault")]
    results4 = fusion.fuse(
        dense_vault=vault_results[:2],
        graph_expanded=graph,
        top_k=10,
    )
    graph_ids = [r.chroma_id for r in results4]
    assert "g1" in graph_ids, f"Graph result not in fused output: {graph_ids}"
    g1 = next(r for r in results4 if r.chroma_id == "g1")
    assert "graph_expanded" in g1.source_methods
    print(f"Test 4 — graph expansion: OK  (g1 in results, "
          f"methods={g1.source_methods})")

    # -----------------------------------------------------------------------
    # Test 5: top_k is respected
    # -----------------------------------------------------------------------
    many = [make_dense(f"m{i}", f"Chunk {i}") for i in range(50)]
    results5 = fusion.fuse(dense_docs=many, top_k=10)
    assert len(results5) == 10, f"top_k not respected: {len(results5)}"
    print(f"Test 5 — top_k: OK  ({len(results5)} results)")

    # -----------------------------------------------------------------------
    # Test 6: Empty inputs return empty list
    # -----------------------------------------------------------------------
    results6 = fusion.fuse(top_k=10)
    assert results6 == []
    print(f"Test 6 — empty inputs: OK")

    # -----------------------------------------------------------------------
    # Test 7: RRF score formula verification
    # -----------------------------------------------------------------------
    # A doc at rank 1 in one list: score = 1/(60+1) = 0.01639...
    # A doc at rank 1 in two lists: score = 2/(60+1) = 0.03279...
    single = fusion.fuse(
        dense_docs=[make_dense("s1", "solo")],
        top_k=1,
    )
    double = fusion.fuse(
        dense_docs=[make_dense("d1", "dual")],
        sparse_docs=[make_sparse("d1", "dual")],
        top_k=1,
    )
    expected_single = 1.0 / (60 + 1)
    expected_double = 2.0 / (60 + 1)
    assert abs(single[0].rrf_score - expected_single) < 1e-9, \
        f"Single score: {single[0].rrf_score} vs {expected_single}"
    assert abs(double[0].rrf_score - expected_double) < 1e-9, \
        f"Double score: {double[0].rrf_score} vs {expected_double}"
    print(f"Test 7 — RRF formula: OK  "
          f"(single={single[0].rrf_score:.5f}, "
          f"double={double[0].rrf_score:.5f})")

    print("\nAll RecipRankFusion assertions passed.")
