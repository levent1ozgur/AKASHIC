"""
pipeline/vault/watcher.py
inotify-backed vault watcher (Linux/Fedora).
Detects .md file changes and queues them for incremental re-indexing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.inotify import InotifyObserver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class VaultEventType(Enum):
    CREATED  = "created"
    MODIFIED = "modified"
    DELETED  = "deleted"
    MOVED    = "moved"


@dataclass
class VaultEvent:
    event_type:  VaultEventType
    path:        str        # relative path from vault root
    old_path:    Optional[str] = None   # only for MOVED events


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------

class _VaultEventHandler(FileSystemEventHandler):
    """
    Filters raw watchdog events to only .md files,
    applies debouncing, and pushes to the event queue.
    """

    def __init__(
        self,
        vault_root:       Path,
        event_queue:      queue.Queue,
        exclude_patterns: list[str],
        debounce_seconds: float,
    ):
        super().__init__()
        self.vault_root       = vault_root
        self.event_queue      = event_queue
        self.exclude_patterns = exclude_patterns
        self.debounce_seconds = debounce_seconds
        # path → last event time (for debouncing)
        self._last_event: dict[str, float] = {}
        self._lock = threading.Lock()

    def _rel(self, abs_path: str) -> str:
        """Convert absolute path to vault-relative path."""
        try:
            return str(Path(abs_path).relative_to(self.vault_root))
        except ValueError:
            return abs_path

    def _should_process(self, abs_path: str) -> bool:
        """Return True if this file should be processed."""
        path = Path(abs_path)

        # Only .md files
        if path.suffix.lower() != ".md":
            return False

        rel = self._rel(abs_path)

        # Exclude patterns
        rel_path = Path(rel)
        for pattern in self.exclude_patterns:
            if rel_path.match(pattern):
                return False

        # Debounce: skip if same file was just processed
        now = time.monotonic()
        with self._lock:
            last = self._last_event.get(rel, 0.0)
            if now - last < self.debounce_seconds:
                logger.debug("Debounced event for '%s'", rel)
                return False
            self._last_event[rel] = now

        return True

    def on_created(self, event) -> None:
        if event.is_directory or not self._should_process(event.src_path):
            return
        rel = self._rel(event.src_path)
        logger.debug("Vault event: CREATED '%s'", rel)
        self.event_queue.put(VaultEvent(VaultEventType.CREATED, rel))

    def on_modified(self, event) -> None:
        if event.is_directory or not self._should_process(event.src_path):
            return
        rel = self._rel(event.src_path)
        logger.debug("Vault event: MODIFIED '%s'", rel)
        self.event_queue.put(VaultEvent(VaultEventType.MODIFIED, rel))

    def on_deleted(self, event) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".md":
            return
        rel = self._rel(event.src_path)
        logger.debug("Vault event: DELETED '%s'", rel)
        self.event_queue.put(VaultEvent(VaultEventType.DELETED, rel))

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        src = Path(event.src_path)
        dst = Path(event.dest_path)
        if src.suffix.lower() != ".md" and dst.suffix.lower() != ".md":
            return
        rel_src = self._rel(event.src_path)
        rel_dst = self._rel(event.dest_path)
        logger.debug("Vault event: MOVED '%s' → '%s'", rel_src, rel_dst)
        self.event_queue.put(VaultEvent(
            VaultEventType.MOVED, rel_dst, old_path=rel_src
        ))


# ---------------------------------------------------------------------------
# Vault watcher
# ---------------------------------------------------------------------------

class VaultWatcher:
    """
    Watches an Obsidian vault for .md file changes using inotify.

    On startup:
      - Scans for files changed since last_indexed_at (startup sync)
      - Then watches continuously via inotify

    Each detected event is put on an internal queue.
    A consumer thread calls the provided callback for each event.

    Usage:
        def on_event(event: VaultEvent):
            if event.event_type in (CREATED, MODIFIED):
                re_index(event.path)
            elif event.event_type == DELETED:
                remove_from_index(event.path)

        watcher = VaultWatcher(
            vault_path="~/Documents/SecondBrain",
            on_event=on_event,
        )
        watcher.start()
        # ... runs in background threads ...
        watcher.stop()
    """

    def __init__(
        self,
        vault_path:         str | Path,
        on_event:           Callable[[VaultEvent], None],
        exclude_patterns:   Optional[list[str]] = None,
        debounce_seconds:   float = 2.0,
        startup_sync:       bool = True,
    ):
        self.vault_path       = Path(vault_path).expanduser().resolve()
        self.on_event         = on_event
        self.exclude_patterns = exclude_patterns or [
            ".git/**",
            ".opencode/node_modules/**",
            ".opencode/scripts/**",
            "AGENTS.md",
            "INSTALL.md",
        ]
        self.debounce_seconds = debounce_seconds
        self.startup_sync     = startup_sync

        self._queue:    queue.Queue = queue.Queue()
        self._observer: Optional[InotifyObserver] = None
        self._consumer: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, last_indexed_at: Optional[float] = None) -> None:
        """
        Start watching the vault.

        Args:
            last_indexed_at: Unix timestamp of last full index run.
                             Files modified after this time are queued
                             immediately on startup (startup sync).
        """
        if not self.vault_path.exists():
            raise FileNotFoundError(
                f"Vault path does not exist: {self.vault_path}"
            )

        logger.info("Starting vault watcher on '%s'", self.vault_path)

        # Startup sync: queue files changed since last index
        if self.startup_sync and last_indexed_at is not None:
            self._startup_sync(last_indexed_at)

        # Start inotify observer
        handler = _VaultEventHandler(
            vault_root=self.vault_path,
            event_queue=self._queue,
            exclude_patterns=self.exclude_patterns,
            debounce_seconds=self.debounce_seconds,
        )
        self._observer = InotifyObserver()
        self._observer.schedule(handler, str(self.vault_path), recursive=True)
        self._observer.start()

        # Start consumer thread
        self._stop_event.clear()
        self._consumer = threading.Thread(
            target=self._consume,
            name="vault-watcher-consumer",
            daemon=True,
        )
        self._consumer.start()

        logger.info("Vault watcher started.")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the watcher and consumer thread."""
        self._stop_event.set()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=timeout)
            self._observer = None

        if self._consumer:
            self._consumer.join(timeout=timeout)
            self._consumer = None

        logger.info("Vault watcher stopped.")

    def is_running(self) -> bool:
        return (
            self._observer is not None
            and self._observer.is_alive()
        )

    # ------------------------------------------------------------------
    # Startup sync
    # ------------------------------------------------------------------

    def _startup_sync(self, last_indexed_at: float) -> None:
        """
        Queue all .md files modified after last_indexed_at.
        This catches changes that happened while the pipeline was offline.
        """
        queued = 0
        for md_path in self.vault_path.rglob("*.md"):
            rel = Path(md_path).relative_to(self.vault_path)

            # Check exclusions
            excluded = any(rel.match(p) for p in self.exclude_patterns)
            if excluded:
                continue

            try:
                mtime = md_path.stat().st_mtime
                if mtime > last_indexed_at:
                    rel_str = str(rel)
                    self._queue.put(VaultEvent(VaultEventType.MODIFIED, rel_str))
                    queued += 1
            except OSError:
                pass

        if queued:
            logger.info(
                "Startup sync: queued %d files modified since last index", queued
            )

    # ------------------------------------------------------------------
    # Consumer
    # ------------------------------------------------------------------

    def _consume(self) -> None:
        """
        Background thread: drain the event queue and call on_event.
        Runs until stop() is called.
        """
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.5)
                try:
                    self.on_event(event)
                except Exception as e:
                    logger.error(
                        "Error handling vault event %s '%s': %s",
                        event.event_type.value, event.path, e,
                    )
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue

    # ------------------------------------------------------------------
    # Queue inspection (for status endpoint)
    # ------------------------------------------------------------------

    def queue_depth(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)

    with tempfile.TemporaryDirectory() as tmp:
        vault = Path(tmp) / "vault"
        vault.mkdir()

        # Write initial note
        (vault / "note1.md").write_text("# Note 1\n\nInitial content.\n")

        # Collect events
        received: list[VaultEvent] = []
        event_lock = threading.Lock()

        def on_event(ev: VaultEvent) -> None:
            with event_lock:
                received.append(ev)
            logger.info(
                "EVENT: %s '%s'", ev.event_type.value, ev.path
            )

        watcher = VaultWatcher(
            vault_path=vault,
            on_event=on_event,
            debounce_seconds=0.1,   # short debounce for testing
            startup_sync=False,
        )
        watcher.start()
        assert watcher.is_running()
        print("Watcher started: OK")

        # Give inotify time to register
        time.sleep(0.3)

        # Create a new file
        (vault / "note2.md").write_text("# Note 2\n\nNew content.\n")
        time.sleep(0.5)

        # Modify an existing file
        (vault / "note1.md").write_text("# Note 1\n\nUpdated content.\n")
        time.sleep(0.5)

        # Non-.md file should be ignored
        (vault / "image.png").write_bytes(b"\x89PNG")
        time.sleep(0.3)

        # Delete a file
        (vault / "note2.md").unlink()
        time.sleep(0.5)

        watcher.stop()
        print("Watcher stopped: OK")

        # Verify events
        event_types = [e.event_type for e in received]
        paths       = [e.path for e in received]

        assert VaultEventType.CREATED  in event_types or \
               VaultEventType.MODIFIED in event_types, \
            f"Expected CREATE/MODIFY event, got: {event_types}"

        assert VaultEventType.DELETED in event_types, \
            f"Expected DELETE event, got: {event_types}"

        assert not any("image.png" in p for p in paths), \
            f"Non-.md file was not filtered: {paths}"

        print(f"Events received: {len(received)}")
        for ev in received:
            print(f"  {ev.event_type.value:10s} '{ev.path}'")

        # Startup sync test
        received2: list[VaultEvent] = []

        def on_event2(ev: VaultEvent) -> None:
            received2.append(ev)

        # Write a note with a future mtime
        note3 = vault / "note3.md"
        note3.write_text("# Note 3\n\nContent.\n")

        past_time = time.time() - 10   # 10 seconds ago
        watcher2 = VaultWatcher(
            vault_path=vault,
            on_event=on_event2,
            startup_sync=True,
        )
        watcher2.start(last_indexed_at=past_time)
        time.sleep(0.5)
        watcher2.stop()

        assert len(received2) >= 1, \
            f"Startup sync should have queued note3.md: {received2}"
        print(f"Startup sync: OK  ({len(received2)} events queued)")

        print("\nAll VaultWatcher assertions passed.")
