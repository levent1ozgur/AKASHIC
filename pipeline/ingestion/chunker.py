"""
pipeline/ingestion/chunker.py
Hybrid hierarchical chunker.
Implements the 7-rule chunking strategy:
  Rule 1: Entire document fits → single chunk
  Rule 2: Split at H1
  Rule 3: Still too large → split at H2
  Rule 4: Still too large → split at H3
  Rule 5: Never split protected regions (tables, code, lists, quotes)
  Rule 6: Still too large → semantic-text-splitter
  Rule 7: Overlap only between prose chunks
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from semantic_text_splitter import TextSplitter

from pipeline.ingestion.metadata import DocumentMetadata, MetadataExtractor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_TARGET_TOKENS   = 512
DEFAULT_OVERLAP_TOKENS  = 64
DEFAULT_CHARS_PER_TOKEN = 4   # rough approximation for token estimation

# Pipeline version constants — bump when the chunking or context logic changes
CHUNKING_VERSION          = "hierarchical/v3"
CONTEXT_ENRICHMENT_VERSION = "v2"          # section_context + heading_path enrichment


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id:       str
    doc_id:         str
    source:         str             # filename or vault note path
    source_type:    str             # "document" | "vault"
    chunk_index:    int
    text:           str
    heading_path:   str             # e.g. "API > Authentication"
    heading_level:  int             # depth of deepest heading (0 if none)
    page_estimate:  Optional[int]
    contains_table: bool
    contains_code:  bool
    contains_list:  bool
    is_protected:   bool            # True if chunk is an intact protected block
    token_count:    int
    tags:           list[str]       = field(default_factory=list)
    wikilinks:      list[str]       = field(default_factory=list)
    language:       str             = "en"
    created_at:     str             = ""
    section_context: str = ""          # e.g. "Aloe > Indications" (for flat docs)
    chunking_version: str           = CHUNKING_VERSION
    context_enrichment_version: str = CONTEXT_ENRICHMENT_VERSION


@dataclass
class _Section:
    """Internal: a heading-delimited section before final chunking."""
    heading_path:  str
    heading_level: int
    text:          str              # full text of this section
    start_char:    int              # position in original document


# ---------------------------------------------------------------------------
# Protected region detection
# ---------------------------------------------------------------------------

# Patterns that mark the START of a protected region
_FENCE_START    = re.compile(r"^```")
_TABLE_ROW      = re.compile(r"^\|")
_BLOCKQUOTE     = re.compile(r"^>")
_LIST_ITEM      = re.compile(r"^(\s*[-*+]|\s*\d+\.) ")

# Section header detection for flat documents (e.g. 'Activities (Aloe) --')
_SECTION_RE = re.compile(r"^([A-Z][A-Za-z\s,]+)\s*\(([^)]+)\)\s*[\u2014\u2013\-]")

def _extract_section_context(lines):
    """Scan ALL lines for section headers, return LAST one found."""
    last_ctx = None
    for line in lines:
        m = _SECTION_RE.match(line.rstrip())
        if m:
            section, herb = m.group(1).strip(), m.group(2).strip()
            if 3 < len(section) < 60:
                herb = herb.split(";")[0].strip()
                last_ctx = f"{herb} > {section}"
    return last_ctx

def _is_protected_start(line: str) -> str | None:
    """Return protection type if line starts a protected region, else None."""
    s = line.rstrip()
    if _FENCE_START.match(s):    return "code"
    if _TABLE_ROW.match(s):      return "table"
    if _BLOCKQUOTE.match(s):     return "quote"
    if _LIST_ITEM.match(s):      return "list"
    return None


def _extract_protected_blocks(text: str) -> list[tuple[int, int, str]]:
    """
    Find all protected regions in text.
    Returns list of (start_line, end_line, block_type) — line indexes.
    """
    lines    = text.splitlines()
    blocks   = []
    i        = 0
    in_fence = False
    fence_start = -1

    while i < len(lines):
        line = lines[i]

        # Code fence toggle
        if _FENCE_START.match(line.rstrip()):
            if not in_fence:
                in_fence    = True
                fence_start = i
            else:
                blocks.append((fence_start, i, "code"))
                in_fence = False
            i += 1
            continue

        if in_fence:
            i += 1
            continue

        ptype = None
        if _TABLE_ROW.match(line):   ptype = "table"
        elif _BLOCKQUOTE.match(line): ptype = "quote"
        elif _LIST_ITEM.match(line):  ptype = "list"

        if ptype:
            start = i
            # Consume contiguous lines of the same type
            while i < len(lines) and (
                _is_protected_line(lines[i], ptype)
                or lines[i].strip() == ""   # allow blank lines within lists
            ):
                # Don't let blank lines extend tables past their natural end
                if ptype == "table" and lines[i].strip() == "":
                    break
                i += 1
            # Trim trailing blank lines
            end = i - 1
            while end > start and lines[end].strip() == "":
                end -= 1
            blocks.append((start, end, ptype))
        else:
            i += 1

    return blocks


def _is_protected_line(line: str, ptype: str) -> bool:
    """Check if a line continues a protected region of the given type."""
    if ptype == "code":    return True   # handled by fence toggle
    if ptype == "table":   return bool(_TABLE_ROW.match(line)) or line.strip() == "---"
    if ptype == "quote":   return bool(_BLOCKQUOTE.match(line))
    if ptype == "list":    return bool(_LIST_ITEM.match(line)) or line.startswith("  ")
    return False


# ---------------------------------------------------------------------------
# Heading splitter
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _split_by_headings(text: str, level: int) -> list[_Section]:
    """
    Split text at headings of the given level (1, 2, or 3).
    Returns list of _Section objects, one per heading block.
    Content before the first heading becomes a section with empty heading_path.
    """
    pattern = re.compile(rf"^({'#' * level})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [_Section(heading_path="", heading_level=0,
                         text=text, start_char=0)]

    sections: list[_Section] = []

    # Content before first heading
    pre = text[:matches[0].start()].strip()
    if pre:
        sections.append(_Section(
            heading_path="", heading_level=0,
            text=pre, start_char=0,
        ))

    for idx, match in enumerate(matches):
        heading_text = match.group(2).strip()
        start        = match.start()
        end          = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        sections.append(_Section(
            heading_path=heading_text,
            heading_level=level,
            text=section_text,
            start_char=start,
        ))

    return sections


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str, chars_per_token: int = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Fast token estimate: character count / chars_per_token."""
    return max(1, len(text) // chars_per_token)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class HierarchicalChunker:
    """
    Hybrid hierarchical chunker following the 7-rule strategy.

    Usage:
        chunker = HierarchicalChunker(
            target_tokens=512,
            overlap_tokens=64,
        )
        chunks = chunker.chunk(
            text=markdown_text,
            doc_id="uuid-1",
            source="report.pdf",
            source_type="document",
            metadata=meta,          # from MetadataExtractor
        )
    """

    def __init__(
        self,
        target_tokens:  int = DEFAULT_TARGET_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
    ):
        self.target_tokens   = target_tokens
        self.overlap_tokens  = overlap_tokens
        self.chars_per_token = chars_per_token
        self.target_chars    = target_tokens  * chars_per_token
        self.overlap_chars   = overlap_tokens * chars_per_token
        self._splitter       = TextSplitter(capacity=self.target_chars)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def chunk(
        self,
        text:        str,
        doc_id:      str,
        source:      str,
        source_type: str,
        metadata:    Optional[DocumentMetadata] = None,
        page_estimate: Optional[int] = None,
    ) -> list[Chunk]:
        """
        Split text into chunks following the 7-rule hierarchy.
        Returns an ordered list of Chunk objects.
        """
        if not text or not text.strip():
            return []

        meta = metadata or MetadataExtractor().extract(text)

        # Rule 1: entire document fits → single chunk
        if _estimate_tokens(text, self.chars_per_token) <= self.target_tokens:
            return [self._make_chunk(
                text=text,
                doc_id=doc_id,
                source=source,
                source_type=source_type,
                chunk_index=0,
                heading_path=meta.title or "",
                heading_level=1 if meta.title else 0,
                meta=meta,
                page_estimate=page_estimate,
            )]

        # Rules 2-6: hierarchical splitting
        raw_chunks = self._split_hierarchically(text, meta)

        # Rule 7: add overlap between prose chunks
        raw_chunks = self._add_overlap(raw_chunks)

        # Build final Chunk objects
        chunks: list[Chunk] = []
        # Collect per-chunk section context from raw tuples
        last_section = ""
        for idx, (chunk_text, heading_path, heading_level, is_protected, _sctx) in \
                enumerate(raw_chunks):
            if not chunk_text.strip():
                continue
            # Compute section context for this chunk
            sctx = _extract_section_context(chunk_text.splitlines())
            if not heading_path and sctx:
                last_section = sctx
            elif not heading_path and last_section:
                sctx = last_section

            chunks.append(self._make_chunk(
                text=chunk_text,
                doc_id=doc_id,
                source=source,
                source_type=source_type,
                chunk_index=idx,
                heading_path=heading_path,
                heading_level=heading_level,
                meta=meta,
                page_estimate=page_estimate,
                is_protected=is_protected,
                section_context=sctx or "",
            ))

        logger.info(
            "Chunked '%s': %d chunks from %d chars (%d est. tokens)",
            source, len(chunks), len(text),
            _estimate_tokens(text, self.chars_per_token),
        )
        return chunks

    # ------------------------------------------------------------------
    # Hierarchical splitting (Rules 2-6)
    # ------------------------------------------------------------------

    def _split_hierarchically(
        self,
        text: str,
        meta: DocumentMetadata,
    ) -> list[tuple[str, str, int, bool]]:
        """
        Returns list of (text, heading_path, heading_level, is_protected, section_context).
        """
        # Rule 2: split at H1
        h1_sections = _split_by_headings(text, level=1)
        result: list[tuple[str, str, int, bool]] = []

        for section in h1_sections:
            result.extend(
                self._refine_section(section, parent_path="", level=2)
            )

        return result

    def _refine_section(
        self,
        section: _Section,
        parent_path: str,
        level: int,
    ) -> list[tuple[str, str, int, bool]]:
        """
        Recursively refine a section that may be too large.
        Tries H2, then H3, then semantic-text-splitter.
        """
        path = _join_path(parent_path, section.heading_path)

        if _estimate_tokens(section.text, self.chars_per_token) <= self.target_tokens:
            return [(section.text, path, section.heading_level, False, "")]

        # Rule 3/4: try splitting at next heading level (up to H3)
        if level <= 3:
            sub_sections = _split_by_headings(section.text, level=level)
            if len(sub_sections) > 1:
                result = []
                for sub in sub_sections:
                    result.extend(
                        self._refine_section(sub, path, level + 1)
                    )
                return result

        # Rule 5 + 6: no useful heading split available
        # Extract protected blocks first, then split prose remainder
        return self._split_with_protection(section.text, path, section.heading_level)

    def _split_with_protection(
        self,
        text:          str,
        heading_path:  str,
        heading_level: int,
    ) -> list[tuple[str, str, int, bool]]:
        """
        Rule 5: extract protected blocks intact.
        Rule 6: split remaining prose with semantic-text-splitter.
        """
        lines  = text.splitlines()
        blocks = _extract_protected_blocks(text)

        # Build a map: line_number → block index
        protected_lines: set[int] = set()
        for b_start, b_end, _ in blocks:
            for ln in range(b_start, b_end + 1):
                protected_lines.add(ln)

        result: list[tuple[str, str, int, bool]] = []
        prose_lines: list[str] = []

        def _flush_prose() -> None:
            prose = "\n".join(prose_lines).strip()
            prose_lines.clear()
            if not prose:
                return
            if _estimate_tokens(prose, self.chars_per_token) <= self.target_tokens:
                result.append((prose, heading_path, heading_level, False, ""))
            else:
                # Rule 6: semantic splitter for remaining prose
                for piece in self._splitter.chunks(prose):
                    if piece.strip():
                        result.append((piece, heading_path, heading_level, False, ""))

        i = 0
        while i < len(lines):
            if i in protected_lines:
                # Flush any accumulated prose first
                _flush_prose()

                # Find which block this line belongs to
                block_text_lines = []
                block_end = i
                for b_start, b_end, btype in blocks:
                    if b_start <= i <= b_end:
                        block_text_lines = lines[b_start: b_end + 1]
                        block_end = b_end
                        break

                block_text = "\n".join(block_text_lines)
                result.append((block_text, heading_path, heading_level, True, ""))
                i = block_end + 1
            else:
                prose_lines.append(lines[i])
                i += 1

        _flush_prose()
        return result

    # ------------------------------------------------------------------
    # Rule 7: overlap injection
    # ------------------------------------------------------------------

    def _add_overlap(
        self,
        chunks: list[tuple[str, str, int, bool]],
    ) -> list[tuple[str, str, int, bool]]:
        """
        Prepend the tail of the previous prose chunk to the current prose chunk.
        Protected blocks (tables, code, etc.) never receive or donate overlap.
        A chunk that contains an unbalanced code fence never donates overlap.
        """
        if self.overlap_chars <= 0 or len(chunks) <= 1:
            return chunks

        result = []
        for idx, (text, hpath, hlevel, is_protected, _sctx) in enumerate(chunks):
            if idx == 0 or is_protected:
                result.append((text, hpath, hlevel, is_protected, ""))
                continue

            prev_text, _, _, prev_protected, _ = chunks[idx - 1]
            if prev_protected:
                result.append((text, hpath, hlevel, is_protected, ""))
                continue

            # Never donate overlap from a chunk with unbalanced code fences
            if prev_text.count("```") % 2 != 0:
                result.append((text, hpath, hlevel, is_protected, ""))
                continue

            # Take the tail of the previous chunk as overlap prefix
            tail = prev_text[-self.overlap_chars:].strip()
            # Strip any partial fence markers from the tail
            tail = re.sub(r"```[^`]*$", "", tail).strip()
            if tail:
                text = tail + "\n\n" + text

            result.append((text, hpath, hlevel, is_protected, ""))

        return result

    # ------------------------------------------------------------------
    # Chunk factory
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        text:          str,
        doc_id:        str,
        source:        str,
        source_type:   str,
        chunk_index:   int,
        heading_path:  str,
        heading_level: int,
        meta:          DocumentMetadata,
        page_estimate: Optional[int] = None,
        is_protected:    bool = False,
        section_context: str  = "",
    ) -> Chunk:
        from datetime import datetime, timezone
        return Chunk(
            chunk_id       = str(uuid.uuid4()),
            doc_id         = doc_id,
            source         = source,
            source_type    = source_type,
            chunk_index    = chunk_index,
            text           = text,
            heading_path   = heading_path,
            heading_level  = heading_level,
            page_estimate  = page_estimate,
            contains_table = bool(re.search(r"^\|", text, re.MULTILINE)),
            contains_code  = bool(re.search(r"^```", text, re.MULTILINE)),
            contains_list  = bool(re.search(r"^(\s*[-*+]|\s*\d+\.) ", text, re.MULTILINE)),
            is_protected   = is_protected,
            section_context = section_context,
            token_count    = _estimate_tokens(text, self.chars_per_token),
            tags           = list(meta.tags),
            wikilinks      = [w.target for w in meta.wikilinks],
            language       = meta.language,
            created_at     = datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _join_path(parent: str, child: str) -> str:
    """Join heading path components, skipping empty parts."""
    parts = [p for p in [parent, child] if p]
    return " > ".join(parts)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    logging.basicConfig(level=logging.INFO)

    chunker = HierarchicalChunker(
        target_tokens=100,      # small for testing
        overlap_tokens=20,
    )

    # -----------------------------------------------------------------------
    # Test 1: Small document → Rule 1 (single chunk)
    # -----------------------------------------------------------------------
    small_doc = "# Hello\n\nThis is a short note.\n"
    chunks = chunker.chunk(
        text=small_doc,
        doc_id="doc-1",
        source="small.md",
        source_type="document",
    )
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0].chunk_index == 0
    print(f"Rule 1 (single chunk): OK  → {len(chunks)} chunk")

    # -----------------------------------------------------------------------
    # Test 2: Multi-section document → Rules 2-4
    # -----------------------------------------------------------------------
    multi_doc = """\
# Chapter 1

This is chapter one content. It has some text to read through.

## Section 1.1

This is section one point one. It contains important information about
the first topic that we are discussing in this document.

## Section 1.2

This is section one point two. It contains different information about
the second topic which is also quite important for understanding.

# Chapter 2

This is chapter two content. It has completely different information
that is unrelated to chapter one but still important overall.

## Section 2.1

Final section with concluding remarks about the entire document.
"""
    chunks = chunker.chunk(
        text=multi_doc,
        doc_id="doc-2",
        source="multi.md",
        source_type="document",
    )
    assert len(chunks) >= 2, f"Expected >=2 chunks, got {len(chunks)}"
    # Verify heading paths are set
    paths = [c.heading_path for c in chunks]
    assert any("Chapter 1" in p for p in paths), f"paths: {paths}"
    assert any("Chapter 2" in p for p in paths), f"paths: {paths}"
    print(f"Rules 2-4 (hierarchical): OK  → {len(chunks)} chunks")
    for c in chunks:
        print(f"  [{c.chunk_index}] path='{c.heading_path}' "
              f"tokens={c.token_count} protected={c.is_protected}")

    # -----------------------------------------------------------------------
    # Test 3: Protected blocks stay intact
    # -----------------------------------------------------------------------
    protected_doc = """\
# API Reference

## Authentication

Use bearer tokens. They expire after 24 hours of issuance.

```python
import requests
headers = {"Authorization": "Bearer <token>"}
response = requests.get("/api/data", headers=headers)
print(response.json())
```

## Rate Limits

| Tier  | RPM | TPD   |
|-------|-----|-------|
| Free  | 10  | 1000  |
| Pro   | 100 | 50000 |
| Ultra | 500 | 999999|

## Error Handling

Handle errors gracefully using try/except blocks in your code.
"""
    chunks = chunker.chunk(
        text=protected_doc,
        doc_id="doc-3",
        source="api.md",
        source_type="document",
    )
    code_chunks  = [c for c in chunks if c.contains_code]
    table_chunks = [c for c in chunks if c.contains_table]
    assert len(code_chunks)  >= 1, "Expected at least one code chunk"
    assert len(table_chunks) >= 1, "Expected at least one table chunk"
    # Verify code block wasn't split (no partial code fence)
    for cc in code_chunks:
        assert cc.text.count("```") % 2 == 0, \
            f"Code fence was split: {cc.text[:100]}"
    print(f"Rule 5 (protected blocks): OK  → "
          f"{len(code_chunks)} code, {len(table_chunks)} table chunks")

    # -----------------------------------------------------------------------
    # Test 4: Overlap is added between prose chunks
    # -----------------------------------------------------------------------
    # Use a doc that will produce multiple prose chunks
    prose_doc = ("This is sentence number {n}. " * 8 + "\n\n") * 6
    prose_doc = "# Long Doc\n\n" + prose_doc
    chunks = chunker.chunk(
        text=prose_doc,
        doc_id="doc-4",
        source="prose.md",
        source_type="document",
    )
    if len(chunks) >= 2:
        # Second chunk should contain tail of first chunk (overlap)
        first_tail = chunks[0].text[-chunker.overlap_chars:].strip()
        if first_tail:
            assert first_tail in chunks[1].text, \
                "Overlap not found in second chunk"
            print(f"Rule 7 (overlap): OK  → overlap found in chunk[1]")
        else:
            print(f"Rule 7 (overlap): OK  → no tail to overlap (short chunk)")
    else:
        print(f"Rule 7 (overlap): SKIP  → only {len(chunks)} chunks produced")

    # -----------------------------------------------------------------------
    # Test 5: Chunk metadata is populated
    # -----------------------------------------------------------------------
    obsidian_note = """\
---
type: research
tags: [rag, pipeline]
ai-first: true
---

# AKASHIC Notes

See [[ChromaDB]] and [[BM25S]] for retrieval components.

Quick note for future reference.
"""
    chunks = chunker.chunk(
        text=obsidian_note,
        doc_id="doc-5",
        source="Research/odysseus.md",
        source_type="vault",
    )
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.source_type == "vault"
    assert c.source == "Research/odysseus.md"
    assert "rag" in c.tags or "pipeline" in c.tags
    assert "ChromaDB" in c.wikilinks or "BM25S" in c.wikilinks
    assert c.language == "en"
    assert c.chunk_id  # uuid is set
    print(f"Chunk metadata: OK  "
          f"(tags={c.tags}, wikilinks={c.wikilinks[:2]})")

    # -----------------------------------------------------------------------
    # Test 6: Empty document
    # -----------------------------------------------------------------------
    chunks = chunker.chunk(
        text="",
        doc_id="doc-6",
        source="empty.md",
        source_type="document",
    )
    assert chunks == []
    print(f"Empty document: OK  → 0 chunks")

    print("\nAll HierarchicalChunker assertions passed.")
