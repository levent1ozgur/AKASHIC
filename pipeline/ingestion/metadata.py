"""
pipeline/ingestion/metadata.py
Markdown AST parser for metadata extraction.
Extracts heading trees, tables, code blocks, wikilinks,
frontmatter, and language from Markdown documents.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import yaml
from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HeadingNode:
    level:    int           # 1-6
    text:     str
    position: int           # character offset in source


@dataclass
class TableInfo:
    heading_path: str       # e.g. "API > Rate Limits"
    row_count:    int
    col_count:    int
    position:     int


@dataclass
class CodeBlockInfo:
    language:  str          # e.g. "python", "" if unspecified
    line_count: int
    position:  int


@dataclass
class WikiLink:
    target:   str           # link target, e.g. "Projects/MyProject"
    display:  str           # display text (same as target if not specified)
    position: int


@dataclass
class DocumentMetadata:
    # Structure
    heading_tree:   list[HeadingNode]   = field(default_factory=list)
    tables:         list[TableInfo]     = field(default_factory=list)
    code_blocks:    list[CodeBlockInfo] = field(default_factory=list)
    wikilinks:      list[WikiLink]      = field(default_factory=list)

    # Obsidian frontmatter fields
    frontmatter:    dict                = field(default_factory=dict)
    note_type:      Optional[str]       = None   # from frontmatter "type"
    tags:           list[str]           = field(default_factory=list)
    ai_first:       bool                = False
    note_date:      Optional[str]       = None
    confidence:     Optional[str]       = None

    # Content metrics
    language:       str                 = "en"
    word_count:     int                 = 0
    char_count:     int                 = 0
    has_tables:     bool                = False
    has_code:       bool                = False
    has_images:     bool                = False
    has_wikilinks:  bool                = False

    # Derived
    title:          Optional[str]       = None   # first H1 text, if any


# ---------------------------------------------------------------------------
# Wikilink regex  [[Target]] or [[Target|Display]]
# ---------------------------------------------------------------------------

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

# Frontmatter fence
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Image markdown
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class MetadataExtractor:
    """
    Extracts rich metadata from a Markdown document.

    Usage:
        extractor = MetadataExtractor()
        meta = extractor.extract(markdown_text)
        print(meta.heading_tree)
        print(meta.tags)
        print(meta.language)
    """

    def __init__(self):
        # Enable table parsing
        self._md = MarkdownIt("commonmark").enable("table")

    def extract(self, markdown: str) -> DocumentMetadata:
        """
        Parse markdown and return a DocumentMetadata instance.
        Never raises — errors are logged and gracefully skipped.
        """
        meta = DocumentMetadata()

        if not markdown or not markdown.strip():
            return meta

        meta.char_count = len(markdown)
        meta.word_count = len(markdown.split())

        # 1. Frontmatter (must come before AST parsing,
        #    strip it so it doesn't pollute heading/table extraction)
        clean_md, frontmatter = _extract_frontmatter(markdown)
        meta.frontmatter = frontmatter
        _apply_frontmatter(meta, frontmatter)

        # 2. Parse AST
        tokens = self._md.parse(clean_md)

        # 3. Extract from token stream
        _extract_headings(tokens, clean_md, meta)
        _extract_tables(tokens, meta)
        _extract_code_blocks(tokens, meta)

        # 4. Wikilinks (regex over raw text — AST doesn't know about them)
        _extract_wikilinks(clean_md, meta)

        # 5. Images
        meta.has_images = bool(_IMAGE_RE.search(clean_md))

        # 6. Language detection
        meta.language = _detect_language(clean_md)

        # 7. Derived fields
        meta.has_tables    = bool(meta.tables)
        meta.has_code      = bool(meta.code_blocks)
        meta.has_wikilinks = bool(meta.wikilinks)
        if meta.heading_tree:
            h1s = [h for h in meta.heading_tree if h.level == 1]
            meta.title = h1s[0].text if h1s else meta.heading_tree[0].text

        return meta

    def heading_path_at(
        self,
        meta: DocumentMetadata,
        position: int,
    ) -> str:
        """
        Return the heading breadcrumb active at a given character position.
        Example: "API Reference > Authentication > Examples"
        Used by the chunker to attach heading_path to each chunk.
        """
        active: dict[int, str] = {}   # level → heading text
        for heading in meta.heading_tree:
            if heading.position > position:
                break
            active[heading.level] = heading.text
            # Clear deeper levels when a shallower heading appears
            for deeper in range(heading.level + 1, 7):
                active.pop(deeper, None)

        if not active:
            return ""
        return " > ".join(active[lvl] for lvl in sorted(active))


# ---------------------------------------------------------------------------
# Internal extraction functions
# ---------------------------------------------------------------------------

def _extract_frontmatter(markdown: str) -> tuple[str, dict]:
    """
    Strip YAML frontmatter from Markdown and parse it.
    Returns (clean_markdown, frontmatter_dict).
    """
    match = _FRONTMATTER_RE.match(markdown)
    if not match:
        return markdown, {}

    yaml_str = match.group(1)
    clean    = markdown[match.end():]
    try:
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            return clean, {}
        return clean, data
    except yaml.YAMLError as e:
        logger.warning("Failed to parse frontmatter YAML: %s", e)
        return clean, {}


def _apply_frontmatter(meta: DocumentMetadata, fm: dict) -> None:
    """Copy well-known frontmatter fields onto the metadata object."""
    if not fm:
        return

    meta.note_type  = fm.get("type")
    meta.note_date  = str(fm.get("date", "")) or None
    meta.ai_first   = bool(fm.get("ai-first", False))
    meta.confidence = fm.get("confidence")

    raw_tags = fm.get("tags", [])
    if isinstance(raw_tags, list):
        meta.tags = [str(t) for t in raw_tags]
    elif isinstance(raw_tags, str):
        meta.tags = [t.strip() for t in raw_tags.split(",") if t.strip()]


def _extract_headings(
    tokens: list,
    source: str,
    meta: DocumentMetadata,
) -> None:
    """Walk token stream and collect heading nodes."""
    i = 0
    char_pos = 0
    lines    = source.splitlines(keepends=True)
    line_offsets = _build_line_offsets(lines)

    while i < len(tokens):
        tok = tokens[i]
        if tok.type == "heading_open":
            level    = int(tok.tag[1])          # h1 → 1, h2 → 2, …
            map_info = tok.map                  # [start_line, end_line]
            position = line_offsets[map_info[0]] if map_info else 0

            # Next token is inline with the heading text
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                text = tokens[i + 1].content.strip()
                meta.heading_tree.append(
                    HeadingNode(level=level, text=text, position=position)
                )
        i += 1


def _extract_tables(tokens: list, meta: DocumentMetadata) -> None:
    """Walk token stream and collect table info."""
    in_table   = False
    row_count  = 0
    col_count  = 0
    position   = 0

    for tok in tokens:
        if tok.type == "table_open":
            in_table  = True
            row_count = 0
            col_count = 0
            position  = tok.map[0] if tok.map else 0

        elif tok.type == "table_close" and in_table:
            meta.tables.append(TableInfo(
                heading_path="",    # filled in by chunker using position
                row_count=row_count,
                col_count=col_count,
                position=position,
            ))
            in_table = False

        elif in_table and tok.type == "tr_open":
            row_count += 1

        elif in_table and tok.type == "td_open" and row_count == 1:
            col_count += 1

        elif in_table and tok.type == "th_open":
            col_count += 1


def _extract_code_blocks(tokens: list, meta: DocumentMetadata) -> None:
    """Collect fenced code block info."""
    for tok in tokens:
        if tok.type in {"fence", "code_block"}:
            lang       = (tok.info or "").strip().split()[0] if tok.info else ""
            line_count = (tok.content or "").count("\n")
            position   = tok.map[0] if tok.map else 0
            meta.code_blocks.append(CodeBlockInfo(
                language=lang,
                line_count=line_count,
                position=position,
            ))


def _extract_wikilinks(markdown: str, meta: DocumentMetadata) -> None:
    """Extract Obsidian [[wikilinks]] from raw Markdown text."""
    for match in _WIKILINK_RE.finditer(markdown):
        target  = match.group(1).strip()
        display = (match.group(2) or target).strip()
        meta.wikilinks.append(WikiLink(
            target=target,
            display=display,
            position=match.start(),
        ))


def _detect_language(text: str) -> str:
    """
    Detect document language using langdetect.
    Falls back to 'en' on any error or if text is too short.
    """
    # Strip Markdown syntax for cleaner detection
    plain = re.sub(r"[#*`\[\]()>|_~]", " ", text)
    plain = re.sub(r"\s+", " ", plain).strip()

    if len(plain) < 50:
        return "en"

    try:
        from langdetect import detect
        return detect(plain)
    except Exception:
        return "en"


def _build_line_offsets(lines: list[str]) -> list[int]:
    """
    Build a list of character offsets for the start of each line.
    line_offsets[i] = character position where line i begins.
    """
    offsets = [0]
    cumulative = 0
    for line in lines:
        cumulative += len(line)
        offsets.append(cumulative)
    return offsets


# ---------------------------------------------------------------------------
# Convenience: heading path helpers
# ---------------------------------------------------------------------------

def build_heading_path(headings: list[HeadingNode], up_to_index: int) -> str:
    """
    Build heading breadcrumb from the heading tree up to (and including)
    a specific index. Used by the chunker.
    """
    if not headings or up_to_index < 0:
        return ""

    active: dict[int, str] = {}
    for h in headings[: up_to_index + 1]:
        active[h.level] = h.text
        for deeper in range(h.level + 1, 7):
            active.pop(deeper, None)

    return " > ".join(active[lvl] for lvl in sorted(active))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    extractor = MetadataExtractor()

    # -----------------------------------------------------------------------
    # Test 1: Full Obsidian-style note with frontmatter + wikilinks
    # -----------------------------------------------------------------------
    obsidian_note = """\
---
type: research
date: 2026-06-29
tags: [ai, rag, architecture]
ai-first: true
confidence: high
---

# RAG Pipeline Architecture

## For future Claude

This note documents the AKASHIC design decisions.

## Overview

The pipeline uses [[ChromaDB]] for dense retrieval and [[BM25S]] for sparse search.
See also [[Projects/AKASHIC]] for the full project context.

## Components

### Ingestion

The ingestion pipeline converts documents via [[MarkItDown]].

| Component    | Purpose         |
|-------------|-----------------|
| Validator   | Security checks |
| Converter   | To Markdown     |
| Chunker     | Split text      |

### Retrieval

```python
results = store.query(collection="documents", query_embedding=vec, top_k=10)
```

## Decisions

See [[ADR/001-chunking-strategy]] for the chunking rationale.
"""

    meta = extractor.extract(obsidian_note)

    # Frontmatter
    assert meta.note_type  == "research",          f"type: {meta.note_type}"
    assert meta.ai_first   is True,                f"ai-first: {meta.ai_first}"
    assert meta.confidence == "high",              f"confidence: {meta.confidence}"
    assert "ai" in meta.tags,                      f"tags: {meta.tags}"
    assert len(meta.tags)  == 3,                   f"tag count: {meta.tags}"
    print(f"Frontmatter: OK  (type={meta.note_type}, tags={meta.tags})")

    # Headings
    assert len(meta.heading_tree) >= 5,            f"headings: {meta.heading_tree}"
    assert meta.title == "RAG Pipeline Architecture"
    h_levels = [h.level for h in meta.heading_tree]
    assert 1 in h_levels and 2 in h_levels and 3 in h_levels
    print(f"Headings: OK  ({len(meta.heading_tree)} found, title='{meta.title}')")

    # Wikilinks
    wl_targets = [w.target for w in meta.wikilinks]
    assert "ChromaDB"                 in wl_targets, f"wikilinks: {wl_targets}"
    assert "BM25S"                    in wl_targets
    assert "Projects/AKASHIC"        in wl_targets
    assert "ADR/001-chunking-strategy" in wl_targets
    assert meta.has_wikilinks is True
    print(f"Wikilinks: OK  ({len(meta.wikilinks)} found: {wl_targets})")

    # Table
    assert len(meta.tables) == 1,  f"tables: {meta.tables}"
    assert meta.tables[0].col_count == 2
    assert meta.has_tables is True
    print(f"Tables: OK  ({meta.tables[0].row_count} rows, "
          f"{meta.tables[0].col_count} cols)")

    # Code block
    assert len(meta.code_blocks) == 1, f"code blocks: {meta.code_blocks}"
    assert meta.code_blocks[0].language == "python"
    assert meta.has_code is True
    print(f"Code blocks: OK  (language={meta.code_blocks[0].language})")

    # Language
    assert meta.language == "en", f"language: {meta.language}"
    print(f"Language detection: OK  ('{meta.language}')")

    # -----------------------------------------------------------------------
    # Test 2: heading_path_at
    # -----------------------------------------------------------------------
    # Find position of "Retrieval" heading
    retrieval_h = next(
        h for h in meta.heading_tree if h.text == "Retrieval"
    )
    path = extractor.heading_path_at(meta, retrieval_h.position + 1)
    assert "RAG Pipeline Architecture" in path
    assert "Retrieval" in path
    print(f"heading_path_at: OK  ('{path}')")

    # -----------------------------------------------------------------------
    # Test 3: Plain document without frontmatter
    # -----------------------------------------------------------------------
    plain_doc = """\
# API Reference

## Authentication

Use bearer tokens for all requests. Tokens expire after 24 hours.

## Rate Limits

| Tier    | RPM |
|---------|-----|
| Free    | 10  |
| Pro     | 100 |

## Error Codes

```json
{"error": "rate_limit_exceeded"}
```
"""
    meta2 = extractor.extract(plain_doc)
    assert meta2.frontmatter == {}
    assert meta2.note_type is None
    assert meta2.title == "API Reference"
    assert len(meta2.tables) == 1
    assert meta2.tables[0].col_count == 2
    assert len(meta2.code_blocks) == 1
    assert meta2.code_blocks[0].language == "json"
    assert not meta2.has_wikilinks
    print(f"Plain doc: OK  (title='{meta2.title}', "
          f"tables={len(meta2.tables)}, code={len(meta2.code_blocks)})")

    # -----------------------------------------------------------------------
    # Test 4: Empty document
    # -----------------------------------------------------------------------
    meta3 = extractor.extract("")
    assert meta3.word_count == 0
    assert meta3.heading_tree == []
    print(f"Empty doc: OK  (no crash, word_count={meta3.word_count})")

    # -----------------------------------------------------------------------
    # Test 5: build_heading_path helper
    # -----------------------------------------------------------------------
    headings = [
        HeadingNode(1, "Chapter 1", 0),
        HeadingNode(2, "Section 1.1", 100),
        HeadingNode(3, "Subsection", 200),
        HeadingNode(2, "Section 1.2", 300),
    ]
    path0 = build_heading_path(headings, 0)
    assert path0 == "Chapter 1", f"path0: {path0}"

    path1 = build_heading_path(headings, 2)
    assert path1 == "Chapter 1 > Section 1.1 > Subsection", f"path1: {path1}"

    path2 = build_heading_path(headings, 3)
    assert path2 == "Chapter 1 > Section 1.2", \
        f"path2: {path2}"  # H2 clears H3
    print(f"build_heading_path: OK")

    print("\nAll MetadataExtractor assertions passed.")
