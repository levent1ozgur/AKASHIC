"""
pipeline/vault/graph_builder.py
Builds a wikilink graph from the Obsidian vault.
Used during retrieval for graph expansion:
  given a retrieved note, find linked notes and add them to the candidate pool.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Wikilink regex — matches [[Target]] and [[Target|Display]]
# ---------------------------------------------------------------------------

_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    """A single vault note in the graph."""
    path:       str             # relative path from vault root, e.g. "Research/arch.md"
    title:      str             # note title (H1 or filename stem)
    links_to:   list[str]       = field(default_factory=list)  # outgoing wikilinks (resolved paths)
    linked_from: list[str]      = field(default_factory=list)  # backlinks (resolved paths)


@dataclass
class GraphExpansionResult:
    """Result of a graph expansion query."""
    source_path:    str             # the note we expanded from
    linked_paths:   list[str]       # resolved paths of linked notes
    depth:          int             # expansion depth used


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class VaultGraphBuilder:
    """
    Scans an Obsidian vault, extracts [[wikilinks]], and builds a
    bidirectional link graph for use in retrieval graph expansion.

    The graph is persisted as a JSON file so it can be loaded quickly
    on startup without re-scanning the whole vault.

    Usage:
        builder = VaultGraphBuilder(vault_path="~/Documents/SecondBrain")
        builder.build()                          # scan vault, build graph
        builder.save("data/vault_graph.json")    # persist

        # Later, on startup:
        builder.load("data/vault_graph.json")

        # During retrieval:
        result = builder.expand("Research/arch.md", depth=1)
        print(result.linked_paths)
    """

    def __init__(self, vault_path: str | Path):
        self.vault_path = Path(vault_path).expanduser().resolve()
        self._nodes: dict[str, GraphNode] = {}   # relative_path → GraphNode
        self._built = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        exclude_patterns: Optional[list[str]] = None,
    ) -> int:
        """
        Scan the vault and build the wikilink graph.
        Returns the number of notes indexed.

        exclude_patterns: list of glob patterns to skip,
          e.g. [".git/**", ".opencode/node_modules/**"]
        """
        exclude_patterns = exclude_patterns or [
            ".git/**",
            ".opencode/node_modules/**",
            ".opencode/scripts/**",
        ]

        if not self.vault_path.exists():
            logger.warning("Vault path does not exist: %s", self.vault_path)
            return 0

        # Phase 1: Index all .md files and extract outgoing links
        md_files = [
            p for p in self.vault_path.rglob("*.md")
            if not _is_excluded(p, self.vault_path, exclude_patterns)
        ]

        logger.info("Scanning %d markdown files in vault", len(md_files))

        for md_path in md_files:
            rel_path = str(md_path.relative_to(self.vault_path))
            title    = _extract_title(md_path)
            links    = _extract_raw_links(md_path)

            self._nodes[rel_path] = GraphNode(
                path=rel_path,
                title=title,
                links_to=links,        # raw link targets, not yet resolved
                linked_from=[],
            )

        # Phase 2: Resolve raw link targets to actual paths
        # Obsidian supports shortest-path linking: [[arch]] resolves to
        # the first file whose stem matches, regardless of folder depth.
        stem_index = _build_stem_index(self._nodes)

        for node in self._nodes.values():
            resolved = []
            for raw_link in node.links_to:
                target = _resolve_link(raw_link, stem_index, self._nodes)
                if target:
                    resolved.append(target)
            node.links_to = resolved

        # Phase 3: Build backlinks (reverse the outgoing link graph)
        for node in self._nodes.values():
            for target_path in node.links_to:
                if target_path in self._nodes:
                    self._nodes[target_path].linked_from.append(node.path)

        self._built = True
        logger.info(
            "Vault graph built: %d nodes, %d total links",
            len(self._nodes),
            sum(len(n.links_to) for n in self._nodes.values()),
        )
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the graph to a JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            rel: {
                "path":         node.path,
                "title":        node.title,
                "links_to":     node.links_to,
                "linked_from":  node.linked_from,
            }
            for rel, node in self._nodes.items()
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Graph saved to %s (%d nodes)", out_path, len(data))

    def load(self, path: str | Path) -> bool:
        """
        Load a previously saved graph from JSON.
        Returns True on success, False if file not found.
        """
        in_path = Path(path)
        if not in_path.exists():
            logger.warning("Graph file not found: %s", in_path)
            return False

        with open(in_path, encoding="utf-8") as f:
            data = json.load(f)

        self._nodes = {
            rel: GraphNode(
                path=entry["path"],
                title=entry["title"],
                links_to=entry["links_to"],
                linked_from=entry["linked_from"],
            )
            for rel, entry in data.items()
        }
        self._built = True
        logger.info("Graph loaded from %s (%d nodes)", in_path, len(self._nodes))
        return True

    # ------------------------------------------------------------------
    # Expansion (used during retrieval)
    # ------------------------------------------------------------------

    def expand(
        self,
        source_path: str,
        depth: int = 1,
        max_results: int = 10,
        include_backlinks: bool = True,
    ) -> GraphExpansionResult:
        """
        Return paths of notes linked to/from source_path.

        Args:
            source_path:       relative vault path of the source note
            depth:             hop depth (1 = direct links only)
            max_results:       cap on returned paths
            include_backlinks: also include notes that link TO source_path

        Returns:
            GraphExpansionResult with resolved linked_paths.
        """
        if not self._built:
            logger.warning("Graph not built — call build() or load() first.")
            return GraphExpansionResult(
                source_path=source_path, linked_paths=[], depth=depth
            )

        visited: set[str] = {source_path}
        frontier: set[str] = {source_path}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for path in frontier:
                node = self._nodes.get(path)
                if not node:
                    continue
                for linked in node.links_to:
                    if linked not in visited:
                        next_frontier.add(linked)
                        visited.add(linked)
                if include_backlinks:
                    for backlinked in node.linked_from:
                        if backlinked not in visited:
                            next_frontier.add(backlinked)
                            visited.add(backlinked)
            frontier = next_frontier

        # Remove the source itself
        linked_paths = [p for p in visited if p != source_path]
        linked_paths = linked_paths[:max_results]

        return GraphExpansionResult(
            source_path=source_path,
            linked_paths=linked_paths,
            depth=depth,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_node(self, path: str) -> Optional[GraphNode]:
        """Return a GraphNode by relative path, or None if not found."""
        return self._nodes.get(path)

    def node_count(self) -> int:
        return len(self._nodes)

    def link_count(self) -> int:
        return sum(len(n.links_to) for n in self._nodes.values())

    def update_node(self, path: str, md_path: Path) -> None:
        """
        Re-scan a single file and update its node in the graph.
        Called by the vault watcher when a file changes.
        Rebuilds outgoing links for this node and updates backlinks.
        """
        stem_index = _build_stem_index(self._nodes)

        # Remove old backlinks from this node's targets
        old_node = self._nodes.get(path)
        if old_node:
            for target in old_node.links_to:
                if target in self._nodes:
                    try:
                        self._nodes[target].linked_from.remove(path)
                    except ValueError:
                        pass

        # Re-extract links
        title    = _extract_title(md_path)
        raw_links = _extract_raw_links(md_path)
        resolved = [
            t for raw in raw_links
            if (t := _resolve_link(raw, stem_index, self._nodes))
        ]

        self._nodes[path] = GraphNode(
            path=path,
            title=title,
            links_to=resolved,
            linked_from=old_node.linked_from if old_node else [],
        )

        # Add new backlinks
        for target in resolved:
            if target in self._nodes and path not in self._nodes[target].linked_from:
                self._nodes[target].linked_from.append(path)

    def remove_node(self, path: str) -> None:
        """Remove a node and clean up all its backlinks."""
        node = self._nodes.pop(path, None)
        if not node:
            return
        for target in node.links_to:
            if target in self._nodes:
                try:
                    self._nodes[target].linked_from.remove(path)
                except ValueError:
                    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_title(md_path: Path) -> str:
    """Return the first H1 heading, or the file stem if none found."""
    try:
        for line in md_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
    except Exception:
        pass
    return md_path.stem


def _extract_raw_links(md_path: Path) -> list[str]:
    """Extract raw wikilink targets from a file."""
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
        return [m.group(1).strip() for m in _WIKILINK_RE.finditer(text)]
    except Exception as e:
        logger.warning("Failed to read '%s': %s", md_path, e)
        return []


def _build_stem_index(
    nodes: dict[str, GraphNode],
) -> dict[str, list[str]]:
    """
    Build a mapping from filename stem → list of full relative paths.
    Used for Obsidian-style shortest-path link resolution.
    e.g. "arch" → ["Research/arch.md", "Old/arch.md"]
    """
    index: dict[str, list[str]] = {}
    for path in nodes:
        stem = Path(path).stem.lower()
        index.setdefault(stem, []).append(path)
    return index


def _resolve_link(
    raw: str,
    stem_index: dict[str, list[str]],
    nodes: dict[str, GraphNode],
) -> Optional[str]:
    """
    Resolve a raw wikilink target to a relative vault path.

    Resolution order:
      1. Exact match (with .md extension)
      2. Exact match as-is (already has .md)
      3. Stem match (Obsidian shortest-path)
      4. Case-insensitive stem match
    """
    raw = raw.strip()

    # Normalise: add .md if missing
    if not raw.lower().endswith(".md"):
        candidate = raw + ".md"
    else:
        candidate = raw

    # 1. Exact match
    if candidate in nodes:
        return candidate

    # 2. Case-insensitive exact match
    lower_map = {k.lower(): k for k in nodes}
    if candidate.lower() in lower_map:
        return lower_map[candidate.lower()]

    # 3. Stem match (shortest-path resolution)
    stem = Path(raw).stem.lower()
    matches = stem_index.get(stem, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer match in same folder as the raw link suggests
        folder = str(Path(raw).parent)
        for m in matches:
            if folder in m:
                return m
        return matches[0]   # fallback: first match

    return None


def _is_excluded(
    path: Path,
    vault_root: Path,
    patterns: list[str],
) -> bool:
    """Return True if path matches any exclusion glob pattern."""
    rel = path.relative_to(vault_root)
    for pattern in patterns:
        if rel.match(pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "vault"

        # Create a mini vault with wikilinks
        (vault / "Research").mkdir(parents=True)
        (vault / "Projects").mkdir()
        (vault / "ADR").mkdir()

        (vault / "index.md").write_text(
            "# Index\n\nSee [[Research/arch]] and [[Projects/AKASHIC]].\n"
        )
        (vault / "Research" / "arch.md").write_text(
            "# Architecture\n\nUses [[ChromaDB]] and [[BM25S]].\n"
            "See [[ADR/001-chunking]].\n"
        )
        (vault / "Research" / "ChromaDB.md").write_text(
            "# ChromaDB\n\nVector database. See [[Research/arch]].\n"
        )
        (vault / "Research" / "BM25S.md").write_text(
            "# BM25S\n\nSparse search index.\n"
        )
        (vault / "Projects" / "AKASHIC.md").write_text(
            "# AKASHIC\n\nMain project. References [[Research/arch]].\n"
        )
        (vault / "ADR" / "001-chunking.md").write_text(
            "# ADR 001: Chunking Strategy\n\nSee [[Research/arch]] for context.\n"
        )

        builder = VaultGraphBuilder(vault_path=vault)
        count = builder.build()

        assert count == 6, f"Expected 6 nodes, got {count}"
        print(f"Build: OK  ({count} notes indexed)")

        # Check link counts
        total_links = builder.link_count()
        assert total_links > 0
        print(f"Total links: {total_links}")

        # Check arch.md node
        arch = builder.get_node("Research/arch.md")
        assert arch is not None
        assert "Research/ChromaDB.md" in arch.links_to
        assert "Research/BM25S.md"    in arch.links_to
        assert "ADR/001-chunking.md"  in arch.links_to
        print(f"arch.md links_to: {arch.links_to}")

        # Check backlinks
        chroma_node = builder.get_node("Research/ChromaDB.md")
        assert chroma_node is not None
        assert "Research/arch.md" in chroma_node.linked_from
        print(f"ChromaDB.md linked_from: {chroma_node.linked_from}")

        # Graph expansion from arch.md (depth=1)
        result = builder.expand("Research/arch.md", depth=1)
        assert len(result.linked_paths) > 0
        assert "Research/ChromaDB.md" in result.linked_paths
        assert "Research/BM25S.md"    in result.linked_paths
        print(f"Expansion from arch.md (depth=1): {result.linked_paths}")

        # Graph expansion depth=2 (should reach index.md via Projects/AKASHIC)
        result2 = builder.expand("Research/arch.md", depth=2)
        assert len(result2.linked_paths) >= len(result.linked_paths)
        print(f"Expansion from arch.md (depth=2): {result2.linked_paths}")

        # Persistence
        graph_path = Path(tmp) / "graph.json"
        builder.save(graph_path)
        assert graph_path.exists()

        builder2 = VaultGraphBuilder(vault_path=vault)
        ok = builder2.load(graph_path)
        assert ok
        assert builder2.node_count() == 6
        arch2 = builder2.get_node("Research/arch.md")
        assert arch2.links_to == arch.links_to
        print(f"Persistence: OK  (saved + reloaded {builder2.node_count()} nodes)")

        # Incremental update
        new_note = vault / "Research" / "new_note.md"
        new_note.write_text("# New Note\n\nLinks to [[Research/arch]].\n")
        builder2._nodes["Research/new_note.md"] = GraphNode(
            path="Research/new_note.md", title="New Note",
            links_to=[], linked_from=[],
        )
        builder2.update_node("Research/new_note.md", new_note)
        updated = builder2.get_node("Research/new_note.md")
        assert "Research/arch.md" in updated.links_to
        print(f"Incremental update: OK  "
              f"(new_note.md links_to={updated.links_to})")

        # Node removal
        builder2.remove_node("Research/new_note.md")
        assert builder2.get_node("Research/new_note.md") is None
        arch3 = builder2.get_node("Research/arch.md")
        assert "Research/new_note.md" not in arch3.linked_from
        print(f"Node removal: OK")

        print("\nAll VaultGraphBuilder assertions passed.")
