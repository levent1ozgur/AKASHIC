"""
pipeline/retrieval/prompt_builder.py
Assembles the final prompt from reranked chunks,
with inline citations for every retrieved context block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from pipeline.retrieval.reranker import RerankedResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Citation format constants
# ---------------------------------------------------------------------------

CITE_DOCS  = "DOCS"
CITE_VAULT = "VAULT"

# System prompt template
_SYSTEM_PROMPT = """\
You are Odysseus, a local AI assistant with access to two knowledge sources:
- [DOCS] Uploaded documents (PDFs, reports, manuals, notes)
- [VAULT] Personal knowledge base (Obsidian notes, research, decisions)

Answer only from the supplied context below.
After each claim, cite the source using [DOCS: filename §section] or \
[VAULT: note_path §section].
If the context does not contain sufficient information to answer, \
say so explicitly — do not hallucinate.
Keep your answer concise and well-structured."""


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class BuiltPrompt:
    """The assembled prompt ready to send to a local LLM."""
    system:         str
    user:           str                 # query + context block
    full_prompt:    str                 # system + user concatenated (for models that want one string)
    context_blocks: list[str]           # individual formatted context blocks
    citations:      list[str]           # citation strings in order
    total_tokens:   int                 # estimated token count
    chunk_count:    int
    not_found:      bool = False        # True when no chunks were available


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Builds a structured prompt from reranked retrieval results.

    Each chunk is formatted as:
        [DOCS: report.pdf p.12 §API > Authentication]
        <chunk text>

    The full context block is injected between the system prompt
    and the user query, following the standard RAG prompt pattern.

    Usage:
        builder = PromptBuilder(chars_per_token=4)
        prompt = builder.build(
            query="How do authentication tokens expire?",
            results=reranked_results,
        )
        print(prompt.full_prompt)
    """

    def __init__(
        self,
        chars_per_token:    int = 4,
        max_context_tokens: int = 6000,     # leave room for response
        system_prompt:      Optional[str] = None,
    ):
        self.chars_per_token    = chars_per_token
        self.max_context_tokens = max_context_tokens
        self.system_prompt      = system_prompt or _SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(
        self,
        query:   str,
        results: list[RerankedResult],
    ) -> BuiltPrompt:
        """
        Build the final prompt.

        Args:
            query:   the user's original query
            results: reranked results from Reranker.rerank()

        Returns:
            BuiltPrompt with all fields populated.
        """
        if not results:
            return self._not_found(query)

        context_blocks: list[str] = []
        citations:      list[str] = []
        total_chars = 0
        max_chars   = self.max_context_tokens * self.chars_per_token

        for result in results:
            citation    = _format_citation(result)
            block       = f"{citation}\n{result.text.strip()}"
            block_chars = len(block)

            # Respect context window — stop adding chunks if we'd overflow
            if total_chars + block_chars > max_chars and context_blocks:
                logger.debug(
                    "Context window limit reached after %d chunks "
                    "(%d chars). Dropping remaining %d chunks.",
                    len(context_blocks),
                    total_chars,
                    len(results) - len(context_blocks),
                )
                break

            context_blocks.append(block)
            citations.append(citation)
            total_chars += block_chars

        if not context_blocks:
            return self._not_found(query)

        context_section = "\n\n".join(context_blocks)

        user_message = (
            f"Context:\n\n"
            f"{context_section}\n\n"
            f"---\n\n"
            f"Question: {query}"
        )

        full_prompt = (
            f"System:\n{self.system_prompt}\n\n"
            f"{user_message}"
        )

        estimated_tokens = len(full_prompt) // self.chars_per_token

        logger.debug(
            "Prompt built: %d chunks, ~%d tokens",
            len(context_blocks), estimated_tokens,
        )

        return BuiltPrompt(
            system=self.system_prompt,
            user=user_message,
            full_prompt=full_prompt,
            context_blocks=context_blocks,
            citations=citations,
            total_tokens=estimated_tokens,
            chunk_count=len(context_blocks),
            not_found=False,
        )

    def _not_found(self, query: str) -> BuiltPrompt:
        """Return a graceful 'not found' prompt when no context is available."""
        message = (
            "I could not find relevant information in your documents or "
            "knowledge base for this query. "
            "Please check that the relevant document has been ingested, "
            "or try rephrasing your question."
        )
        user_message = f"Question: {query}\n\n{message}"
        full_prompt  = f"System:\n{self.system_prompt}\n\n{user_message}"

        return BuiltPrompt(
            system=self.system_prompt,
            user=user_message,
            full_prompt=full_prompt,
            context_blocks=[],
            citations=[],
            total_tokens=len(full_prompt) // self.chars_per_token,
            chunk_count=0,
            not_found=True,
        )


# ---------------------------------------------------------------------------
# Citation formatting
# ---------------------------------------------------------------------------

def _format_citation(result: RerankedResult) -> str:
    """
    Format a citation header for a retrieved chunk.

    Examples:
        [DOCS: report.pdf p.12 §API > Authentication]
        [VAULT: Research/arch.md §Overview]
        [DOCS: api.pdf §introduction]
    """
    meta        = result.metadata
    source      = meta.get("source", "unknown")
    heading     = meta.get("heading_path", "").strip()
    page        = meta.get("page_estimate", 0)
    source_type = result.collection   # "documents" or "vault"

    cite_type = CITE_VAULT if source_type == "vault" else CITE_DOCS

    parts = [f"{cite_type}: {source}"]

    if page and int(page) > 0:
        parts.append(f"p.{page}")

    section = heading if heading else "introduction"
    parts.append(f"§{section}")

    return "[" + " ".join(parts) + "]"


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.DEBUG)

    from pipeline.retrieval.reranker import RerankedResult

    def make_result(
        cid: str, text: str, source: str,
        collection: str = "documents",
        heading: str = "",
        page: int = 0,
        score: float = 5.0,
    ) -> RerankedResult:
        return RerankedResult(
            chroma_id=cid,
            text=text,
            metadata={
                "source": source,
                "heading_path": heading,
                "page_estimate": page,
                "source_type": collection,
            },
            reranker_score=score,
            collection=collection,
            source_methods=["dense_docs"],
            rrf_score=0.03,
            original_rank=1,
        )

    builder = PromptBuilder(chars_per_token=4, max_context_tokens=2000)

    # -----------------------------------------------------------------------
    # Test 1: Normal prompt with mixed doc + vault results
    # -----------------------------------------------------------------------
    results = [
        make_result(
            "c1",
            "Authentication tokens expire after 24 hours of issuance.",
            source="api_reference.pdf",
            collection="documents",
            heading="API > Authentication",
            page=12,
            score=9.0,
        ),
        make_result(
            "c2",
            "My research note: use short-lived tokens and refresh them proactively.",
            source="Research/auth_notes.md",
            collection="vault",
            heading="Authentication Strategy",
            score=3.5,
        ),
        make_result(
            "c3",
            "Bearer token format: Authorization: Bearer <token>",
            source="api_reference.pdf",
            collection="documents",
            heading="API > Authentication > Examples",
            page=13,
            score=2.1,
        ),
    ]

    prompt = builder.build(
        query="How do authentication tokens expire?",
        results=results,
    )

    assert not prompt.not_found
    assert prompt.chunk_count == 3
    assert prompt.total_tokens > 0
    assert len(prompt.citations) == 3
    assert len(prompt.context_blocks) == 3
    print(f"Test 1 — normal prompt: OK  "
          f"(chunks={prompt.chunk_count}, ~{prompt.total_tokens} tokens)")

    # Verify citation format
    assert "[DOCS: api_reference.pdf p.12 §API > Authentication]" \
           in prompt.citations[0], f"Citation 0: {prompt.citations[0]}"
    assert "[VAULT: Research/auth_notes.md §Authentication Strategy]" \
           in prompt.citations[1], f"Citation 1: {prompt.citations[1]}"
    assert "p.13" in prompt.citations[2], f"Citation 2: {prompt.citations[2]}"
    print(f"Citations: OK")
    for i, c in enumerate(prompt.citations):
        print(f"  [{i+1}] {c}")

    # Verify citations appear in context blocks
    for block, citation in zip(prompt.context_blocks, prompt.citations):
        assert block.startswith(citation), \
            f"Block doesn't start with citation.\nBlock: {block[:80]}"
    print(f"Context blocks: OK  (each starts with its citation)")

    # Verify full_prompt contains system + context + query
    assert "Odysseus" in prompt.system
    assert "Context:" in prompt.full_prompt
    assert "Question: How do authentication tokens expire?" in prompt.full_prompt
    print(f"Full prompt structure: OK")

    # -----------------------------------------------------------------------
    # Test 2: Not-found case (empty results)
    # -----------------------------------------------------------------------
    prompt2 = builder.build(query="What is the meaning of life?", results=[])
    assert prompt2.not_found
    assert prompt2.chunk_count == 0
    assert prompt2.context_blocks == []
    assert "could not find" in prompt2.full_prompt.lower()
    print(f"Test 2 — not found: OK  (graceful message returned)")

    # -----------------------------------------------------------------------
    # Test 3: Context window truncation
    # -----------------------------------------------------------------------
    tiny_builder = PromptBuilder(chars_per_token=4, max_context_tokens=50)
    many_results = [
        make_result(f"c{i}", f"Chunk {i}: " + "x" * 100, "doc.pdf", score=float(10-i))
        for i in range(10)
    ]
    prompt3 = tiny_builder.build(query="test", results=many_results)
    # Should have fewer chunks than we passed in
    assert prompt3.chunk_count < 10, \
        f"Expected truncation, got {prompt3.chunk_count} chunks"
    print(f"Test 3 — context truncation: OK  "
          f"(kept {prompt3.chunk_count}/10 chunks within token limit)")

    # -----------------------------------------------------------------------
    # Test 4: Citation with no heading and no page
    # -----------------------------------------------------------------------
    bare = make_result("c99", "Some content.", "notes.txt",
                       heading="", page=0)
    prompt4 = builder.build(query="test", results=[bare])
    assert "§introduction" in prompt4.citations[0], \
        f"Expected '§introduction' fallback: {prompt4.citations[0]}"
    print(f"Test 4 — bare citation fallback: OK  ({prompt4.citations[0]})")

    # -----------------------------------------------------------------------
    # Test 5: Vault citation format
    # -----------------------------------------------------------------------
    vault = make_result("v1", "Vault content.", "Projects/Odysseus.md",
                        collection="vault", heading="Architecture", page=0)
    prompt5 = builder.build(query="test", results=[vault])
    assert prompt5.citations[0].startswith("[VAULT:"), \
        f"Expected [VAULT:...]: {prompt5.citations[0]}"
    assert "p." not in prompt5.citations[0], \
        f"Vault citation should not have page number: {prompt5.citations[0]}"
    print(f"Test 5 — vault citation: OK  ({prompt5.citations[0]})")

    print(f"\n--- Sample full prompt (first 800 chars) ---")
    print(prompt.full_prompt[:800])
    print(f"...\n")
    print(f"All PromptBuilder assertions passed.")
