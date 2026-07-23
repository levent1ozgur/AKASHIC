"""
pipeline/retrieval/router.py
Rule-based query router.
Decides whether to search documents, vault, or both — without an LLM call
for the common cases. LLM fallback only for genuinely ambiguous queries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Route enum
# ---------------------------------------------------------------------------

class Route(Enum):
    DOCUMENTS = "docs"
    VAULT     = "vault"
    BOTH      = "both"


# ---------------------------------------------------------------------------
# Router result
# ---------------------------------------------------------------------------

@dataclass
class RouterResult:
    route:            Route
    metadata_filter:  Optional[dict]    = None   # e.g. {"source": "report.pdf"}
    hyde_enabled:     bool              = False
    reasoning:        str               = ""
    used_llm:         bool              = False


# ---------------------------------------------------------------------------
# Rule patterns
# ---------------------------------------------------------------------------

# Vault-intent signals
_VAULT_PATTERNS = [
    r"\bin my notes?\b",
    r"\bin the vault\b",
    r"\bI wrote\b",
    r"\bdid I write\b",
    r"\bhave I written\b",
    r"\bI noted\b",
    r"\bmy research\b",
    r"\bmy obsidian\b",
    r"\bfrom my notes?\b",
    r"\bin my second brain\b",
    r"\bI captured\b",
    r"\bI recorded\b",
]

# Document-intent signals
_DOC_PATTERNS = [
    r"\bin the (pdf|document|report|file|paper|doc)\b",
    r"\bthe (pdf|document|report|paper|doc) says?\b",
    r"\baccording to the (pdf|document|report|paper)\b",
    r"\bin (chapter|section|appendix|page)\s+\d",
    r"\bthe uploaded\b",
    r"\bthe attached\b",
]

# Short query: likely keyword lookup, disable HyDE
_SHORT_QUERY_TOKENS = 4

# Compiled patterns
_VAULT_RE = [re.compile(p, re.IGNORECASE) for p in _VAULT_PATTERNS]
_DOC_RE   = [re.compile(p, re.IGNORECASE) for p in _DOC_PATTERNS]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Routes a query to the appropriate index(es).

    Rule evaluation order:
      1. Vault-intent phrases     → vault only
      2. Document-intent phrases  → documents only
      3. Named document reference → documents only, with metadata filter
      4. Short query (<4 tokens)  → both, HyDE disabled
      5. Default                  → both

    LLM fallback: not implemented in this version — rule coverage is
    sufficient for the expected query patterns. Add if telemetry shows
    frequent misroutes.

    Usage:
        router = QueryRouter(known_sources=["report.pdf", "api_docs.pdf"])
        result = router.route("What did I write about chunking strategy?")
        # → Route.VAULT
    """

    def __init__(
        self,
        known_sources: Optional[list[str]] = None,
        hyde_min_tokens: int = 6,
    ):
        """
        Args:
            known_sources:   list of known document filenames in the index.
                             Used for Rule 3 (named document detection).
            hyde_min_tokens: minimum query length to enable HyDE.
        """
        self.known_sources   = known_sources or []
        self.hyde_min_tokens = hyde_min_tokens

        # Build case-insensitive source name patterns for Rule 3
        self._source_patterns = [
            (re.compile(re.escape(s), re.IGNORECASE), s)
            for s in self.known_sources
        ]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def route(self, query: str) -> RouterResult:
        """
        Route a query to the appropriate index(es).
        Returns a RouterResult with routing decision and metadata.
        """
        query = query.strip()
        if not query:
            return RouterResult(
                route=Route.BOTH,
                reasoning="Empty query — defaulting to both.",
            )

        token_count = len(query.split())

        # Rule 1: Vault-intent phrases
        for pattern in _VAULT_RE:
            if pattern.search(query):
                return RouterResult(
                    route=Route.VAULT,
                    hyde_enabled=self._hyde_ok(token_count),
                    reasoning=f"Vault-intent phrase matched: '{pattern.pattern}'",
                )

        # Rule 2: Document-intent phrases
        for pattern in _DOC_RE:
            if pattern.search(query):
                return RouterResult(
                    route=Route.DOCUMENTS,
                    hyde_enabled=self._hyde_ok(token_count),
                    reasoning=f"Document-intent phrase matched: '{pattern.pattern}'",
                )

        # Rule 3: Named document reference
        for src_pattern, source_name in self._source_patterns:
            if src_pattern.search(query):
                return RouterResult(
                    route=Route.DOCUMENTS,
                    metadata_filter={"source": source_name},
                    hyde_enabled=self._hyde_ok(token_count),
                    reasoning=f"Named document reference: '{source_name}'",
                )

        # Rule 4: Short query — both indexes, HyDE off
        if token_count < _SHORT_QUERY_TOKENS:
            return RouterResult(
                route=Route.BOTH,
                hyde_enabled=False,
                reasoning=f"Short query ({token_count} tokens) — "
                          "both indexes, HyDE disabled.",
            )

        # Rule 5: Default — both indexes
        return RouterResult(
            route=Route.BOTH,
            hyde_enabled=self._hyde_ok(token_count),
            reasoning="Default routing — both indexes.",
        )

    def update_sources(self, sources: list[str]) -> None:
        """Update the known document sources list (called after ingestion)."""
        self.known_sources = sources
        self._source_patterns = [
            (re.compile(re.escape(s), re.IGNORECASE), s)
            for s in sources
        ]
        logger.debug("Router updated with %d known sources", len(sources))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hyde_ok(self, token_count: int) -> bool:
        return token_count >= self.hyde_min_tokens


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    router = QueryRouter(
        known_sources=["api_reference.pdf", "architecture_report.pdf"],
        hyde_min_tokens=6,
    )

    cases = [
        # (query, expected_route, expected_filter_key)
        ("What did I write about chunking strategy?",
         Route.VAULT, None),
        ("In my notes about RAG pipelines",
         Route.VAULT, None),
        ("What does the PDF say about authentication?",
         Route.DOCUMENTS, None),
        ("In the document, what are the rate limits?",
         Route.DOCUMENTS, None),
        ("Tell me about api_reference.pdf error codes",
         Route.DOCUMENTS, "api_reference.pdf"),
        ("architecture_report.pdf chapter 3 summary",
         Route.DOCUMENTS, "architecture_report.pdf"),
        ("auth",                   # short query
         Route.BOTH, None),
        ("rate limit",             # short query (2 tokens)
         Route.BOTH, None),
        ("What are the best practices for embedding models in RAG systems?",
         Route.BOTH, None),
        ("",                       # empty
         Route.BOTH, None),
    ]

    all_ok = True
    for query, expected_route, expected_filter in cases:
        result = router.route(query)
        ok = result.route == expected_route
        filter_ok = (
            expected_filter is None
            or (result.metadata_filter or {}).get("source") == expected_filter
        )
        status = "OK " if (ok and filter_ok) else "FAIL"
        if not (ok and filter_ok):
            all_ok = False
        print(
            f"[{status}] '{query[:55]:<55}' "
            f"→ {result.route.value:<10} "
            f"filter={result.metadata_filter} "
            f"hyde={result.hyde_enabled}"
        )
        if not ok:
            print(f"       Expected route={expected_route.value}, "
                  f"got={result.route.value}")
        if not filter_ok:
            print(f"       Expected filter source='{expected_filter}', "
                  f"got={result.metadata_filter}")

    # HyDE enabled/disabled based on token count
    short  = router.route("auth tokens")
    long_q = router.route("What are the best practices for designing "
                          "authentication token systems in APIs?")
    assert not short.hyde_enabled,  "Short query should have HyDE disabled"
    assert long_q.hyde_enabled,     "Long query should have HyDE enabled"
    print(f"\nHyDE gating: OK  "
          f"(short={short.hyde_enabled}, long={long_q.hyde_enabled})")

    # update_sources
    router.update_sources(["new_doc.pdf"])
    result = router.route("What does new_doc.pdf say?")
    assert result.route == Route.DOCUMENTS
    assert result.metadata_filter == {"source": "new_doc.pdf"}
    print(f"update_sources: OK")

    print(f"\n{'All' if all_ok else 'Some'} router assertions "
          f"{'passed' if all_ok else 'FAILED'}.")
