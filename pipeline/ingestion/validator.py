"""
pipeline/ingestion/validator.py
Security validation for uploaded files.
Checks size, extension, and MIME type before any processing begins.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import magic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed file types
# Maps lowercase extension → set of valid MIME types
# ---------------------------------------------------------------------------

ALLOWED_TYPES: dict[str, set[str]] = {
    ".pdf":  {"application/pdf"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document",
        "application/zip",          # DOCX is a ZIP internally
    },
    ".pptx": {
        "application/vnd.openxmlformats-officedocument"
        ".presentationml.presentation",
        "application/zip",
    },
    ".xlsx": {
        "application/vnd.openxmlformats-officedocument"
        ".spreadsheetml.sheet",
        "application/zip",
    },
    ".html": {"text/html"},
    ".htm":  {"text/html"},
    ".epub": {
        "application/epub+zip",
        "application/zip",
    },
    ".md":   {"text/plain", "text/markdown", "text/x-markdown"},
    ".txt":  {"text/plain"},
}

# Maximum file size in bytes (default 100 MB)
DEFAULT_MAX_SIZE_BYTES = 100 * 1024 * 1024

# Forbidden filename patterns (path traversal, null bytes, etc.)
_UNSAFE_PATTERN = re.compile(r"[/\\<>:\"|?*\x00]|\.\.+")


# ---------------------------------------------------------------------------
# Result / error types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    valid:      bool
    filename:   str                 # sanitised filename
    extension:  str                 # lowercase, e.g. ".pdf"
    mime_type:  str                 # detected MIME
    size_bytes: int
    error:      Optional[str] = None


class ValidationError(Exception):
    """Raised when a file fails validation."""
    pass


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class FileValidator:
    """
    Validates uploaded files before ingestion.

    Checks (in order):
      1. Filename sanity (no path traversal, no null bytes)
      2. File size <= max_size_bytes
      3. Extension is in the allowlist
      4. MIME type matches the declared extension

    Usage:
        validator = FileValidator(max_size_bytes=50 * 1024 * 1024)
        result = validator.validate("/tmp/uploads/report.pdf")
        if not result.valid:
            raise ValidationError(result.error)
    """

    def __init__(
        self,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
        allowed_types: Optional[dict[str, set[str]]] = None,
    ):
        self.max_size_bytes = max_size_bytes
        self.allowed_types  = allowed_types or ALLOWED_TYPES

    def validate(self, file_path: str | Path) -> ValidationResult:
        """
        Validate a file at the given path.
        Returns a ValidationResult — check .valid before proceeding.
        Never raises; errors are captured in .error field.
        """
        path = Path(file_path)

        # 1. File must exist
        if not path.exists():
            return self._fail(str(path), "", "", 0, "File does not exist.")
        if not path.is_file():
            return self._fail(str(path), "", "", 0, "Path is not a file.")

        filename  = path.name
        size      = path.stat().st_size
        extension = path.suffix.lower()

        # 2. Filename sanity
        sane, reason = _sanitise_filename(filename)
        if not sane:
            return self._fail(filename, extension, "", size, reason)

        # 3. Size check
        if size == 0:
            return self._fail(filename, extension, "", size, "File is empty.")
        if size > self.max_size_bytes:
            limit_mb = self.max_size_bytes / (1024 * 1024)
            actual_mb = size / (1024 * 1024)
            return self._fail(
                filename, extension, "", size,
                f"File size {actual_mb:.1f} MB exceeds limit of {limit_mb:.0f} MB."
            )

        # 4. Extension allowlist
        if extension not in self.allowed_types:
            allowed = ", ".join(sorted(self.allowed_types))
            return self._fail(
                filename, extension, "", size,
                f"Extension '{extension}' is not allowed. "
                f"Allowed extensions: {allowed}"
            )

        # 5. MIME type detection
        try:
            mime = magic.from_file(str(path), mime=True)
        except Exception as e:
            return self._fail(
                filename, extension, "", size,
                f"MIME detection failed: {e}"
            )

        # 6. MIME must match extension
        allowed_mimes = self.allowed_types[extension]
        if not _mime_matches(mime, allowed_mimes):
            return self._fail(
                filename, extension, mime, size,
                f"MIME type '{mime}' does not match extension '{extension}'. "
                f"Expected one of: {sorted(allowed_mimes)}"
            )

        logger.info(
            "Validated '%s': %s, %.1f KB, MIME=%s",
            filename, extension, size / 1024, mime,
        )
        return ValidationResult(
            valid=True,
            filename=_safe_filename(filename),
            extension=extension,
            mime_type=mime,
            size_bytes=size,
        )

    def validate_bytes(
        self,
        data: bytes,
        filename: str,
    ) -> ValidationResult:
        """
        Validate raw bytes (e.g. from an HTTP upload) without writing to disk first.
        Writes to a temp file internally for MIME detection, then deletes it.
        """
        import tempfile, os

        extension = Path(filename).suffix.lower()
        size      = len(data)

        # Filename sanity
        sane, reason = _sanitise_filename(filename)
        if not sane:
            return self._fail(filename, extension, "", size, reason)

        # Size
        if size == 0:
            return self._fail(filename, extension, "", size, "File is empty.")
        if size > self.max_size_bytes:
            limit_mb   = self.max_size_bytes / (1024 * 1024)
            actual_mb  = size / (1024 * 1024)
            return self._fail(
                filename, extension, "", size,
                f"File size {actual_mb:.1f} MB exceeds limit of {limit_mb:.0f} MB."
            )

        # Extension
        if extension not in self.allowed_types:
            allowed = ", ".join(sorted(self.allowed_types))
            return self._fail(
                filename, extension, "", size,
                f"Extension '{extension}' is not allowed. Allowed: {allowed}"
            )

        # Write to temp file for MIME detection
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=extension)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(data)
            mime = magic.from_file(tmp_path, mime=True)
        except Exception as e:
            return self._fail(
                filename, extension, "", size,
                f"MIME detection failed: {e}"
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        allowed_mimes = self.allowed_types[extension]
        if not _mime_matches(mime, allowed_mimes):
            return self._fail(
                filename, extension, mime, size,
                f"MIME type '{mime}' does not match extension '{extension}'."
            )

        return ValidationResult(
            valid=True,
            filename=_safe_filename(filename),
            extension=extension,
            mime_type=mime,
            size_bytes=size,
        )

    @staticmethod
    def _fail(
        filename: str,
        extension: str,
        mime: str,
        size: int,
        error: str,
    ) -> ValidationResult:
        logger.warning("Validation failed for '%s': %s", filename, error)
        return ValidationResult(
            valid=False,
            filename=filename,
            extension=extension,
            mime_type=mime,
            size_bytes=size,
            error=error,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_filename(filename: str) -> tuple[bool, str]:
    """
    Check filename for unsafe characters.
    Returns (is_safe, reason_if_not).
    """
    if not filename:
        return False, "Filename is empty."
    if _UNSAFE_PATTERN.search(filename):
        return False, (
            f"Filename '{filename}' contains unsafe characters "
            f"(path separators, null bytes, or double dots)."
        )
    if len(filename) > 255:
        return False, "Filename exceeds 255 characters."
    return True, ""


def _safe_filename(filename: str) -> str:
    """
    Return a filesystem-safe version of the filename.
    Replaces spaces with underscores, strips leading dots.
    """
    name = filename.strip().lstrip(".")
    name = re.sub(r"\s+", "_", name)
    return name or "unnamed_file"


def _mime_matches(detected: str, allowed: set[str]) -> bool:
    """
    Check if detected MIME is in the allowed set.
    Handles common MIME aliases (e.g. text/plain for .md files).
    """
    if detected in allowed:
        return True
    # text/plain is acceptable for any text-based format
    if detected == "text/plain" and any(
        "text" in m for m in allowed
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, os

    logging.basicConfig(level=logging.INFO)
    validator = FileValidator(max_size_bytes=10 * 1024 * 1024)  # 10 MB for test

    with tempfile.TemporaryDirectory() as tmp:

        # --- Valid .txt file ---
        txt_path = os.path.join(tmp, "notes.txt")
        with open(txt_path, "w") as f:
            f.write("Hello, this is a plain text note.")
        result = validator.validate(txt_path)
        assert result.valid, f"Expected valid: {result.error}"
        assert result.extension == ".txt"
        print(f"Valid .txt: OK  (MIME={result.mime_type})")

        # --- Valid .md file ---
        md_path = os.path.join(tmp, "readme.md")
        with open(md_path, "w") as f:
            f.write("# Title\n\nSome markdown content.")
        result = validator.validate(md_path)
        assert result.valid, f"Expected valid: {result.error}"
        print(f"Valid .md:  OK  (MIME={result.mime_type})")

        # --- File too large ---
        big_path = os.path.join(tmp, "big.txt")
        with open(big_path, "wb") as f:
            f.write(b"x" * (11 * 1024 * 1024))  # 11 MB > 10 MB limit
        result = validator.validate(big_path)
        assert not result.valid
        assert "exceeds limit" in result.error
        print(f"Too large:  OK  (error='{result.error[:50]}...')")

        # --- Empty file ---
        empty_path = os.path.join(tmp, "empty.txt")
        open(empty_path, "w").close()
        result = validator.validate(empty_path)
        assert not result.valid
        assert "empty" in result.error.lower()
        print(f"Empty file: OK  (error='{result.error}')")

        # --- Disallowed extension ---
        exe_path = os.path.join(tmp, "virus.exe")
        with open(exe_path, "w") as f:
            f.write("not really an exe")
        result = validator.validate(exe_path)
        assert not result.valid
        assert "not allowed" in result.error
        print(f"Bad ext:    OK  (error='{result.error[:50]}...')")

        # --- Path traversal in filename ---
        sane, reason = _sanitise_filename("../../etc/passwd")
        assert not sane
        print(f"Path traversal: OK  (caught: '{reason}')")

        # --- Nonexistent file ---
        result = validator.validate("/tmp/does_not_exist_xyz.pdf")
        assert not result.valid
        print(f"Missing file: OK  (error='{result.error}')")

        # --- validate_bytes ---
        result = validator.validate_bytes(
            b"# Markdown via bytes\n\nContent here.",
            "upload.md",
        )
        assert result.valid
        print(f"validate_bytes .md: OK  (MIME={result.mime_type})")

        print("\nAll FileValidator assertions passed.")
