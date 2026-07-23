"""
pipeline/ingestion/converter.py
MarkItDown wrapper with OCR fallback for scanned PDFs.
Converts uploaded files to Markdown as the canonical intermediate format.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from markitdown import MarkItDown

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum average characters per page to consider a PDF text-based.
# Below this threshold we assume the PDF is scanned and trigger OCR.
MIN_CHARS_PER_PAGE = 100

# Conversion methods
METHOD_MARKITDOWN = "markitdown"
METHOD_OCR        = "ocr"
METHOD_PASSTHROUGH = "passthrough"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConversionResult:
    markdown:          str
    method:            str              # markitdown | ocr | passthrough
    ocr_used:          bool
    page_count:        int
    char_count:        int
    word_count:        int
    warnings:          list[str] = field(default_factory=list)
    source_path:       str = ""
    output_path:       Optional[str] = None


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class DocumentConverter:
    """
    Converts uploaded documents to Markdown.

    Conversion strategy per file type:
      .md, .txt     → passthrough (already text, minimal cleanup)
      .pdf          → MarkItDown first; if text density is low → OCR fallback
      .docx/.pptx/.xlsx/.html/.epub → MarkItDown

    OCR fallback (PDF only):
      Requires: tesseract, poppler-utils (pdftoppm via pdf2image)
      Rasterises each page → Tesseract OCR → reassemble as Markdown

    Usage:
        converter = DocumentConverter()
        result = converter.convert("uploads/report.pdf")
        print(result.markdown[:500])
    """

    def __init__(
        self,
        min_chars_per_page: int = MIN_CHARS_PER_PAGE,
        ocr_enabled: bool = True,
        ocr_language: str = "eng",
    ):
        self.min_chars_per_page = min_chars_per_page
        self.ocr_enabled        = ocr_enabled
        self.ocr_language       = ocr_language
        self._md = MarkItDown()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def convert(
        self,
        file_path: str | Path,
        output_dir: Optional[str | Path] = None,
    ) -> ConversionResult:
        """
        Convert a file to Markdown.

        Args:
            file_path:  path to the source file
            output_dir: if provided, write {stem}.md here and set output_path

        Returns:
            ConversionResult with markdown text and metadata.
        """
        path      = Path(file_path)
        extension = path.suffix.lower()
        warnings: list[str] = []

        logger.info("Converting '%s' (extension=%s)", path.name, extension)

        # --- Passthrough for plain text formats ---
        if extension in {".md", ".txt"}:
            markdown, method = self._passthrough(path)

        # --- PDF: try MarkItDown first, OCR fallback if needed ---
        elif extension == ".pdf":
            markdown, method, ocr_warnings = self._convert_pdf(path)
            warnings.extend(ocr_warnings)

        # --- Everything else: MarkItDown ---
        else:
            markdown, method = self._convert_markitdown(path, warnings)

        # Post-processing: normalise whitespace
        markdown = _normalise_markdown(markdown)

        # Metrics
        page_count = _estimate_page_count(markdown, extension)
        char_count = len(markdown)
        word_count = len(markdown.split())
        ocr_used   = method == METHOD_OCR

        # Optionally write to disk
        output_path: Optional[str] = None
        if output_dir is not None:
            output_path = self._write_output(
                markdown, path.stem, Path(output_dir)
            )

        logger.info(
            "Converted '%s': method=%s, pages=%d, chars=%d, words=%d%s",
            path.name, method, page_count, char_count, word_count,
            " [OCR]" if ocr_used else "",
        )

        return ConversionResult(
            markdown=markdown,
            method=method,
            ocr_used=ocr_used,
            page_count=page_count,
            char_count=char_count,
            word_count=word_count,
            warnings=warnings,
            source_path=str(path),
            output_path=output_path,
        )

    # ------------------------------------------------------------------
    # Conversion strategies
    # ------------------------------------------------------------------

    def _passthrough(self, path: Path) -> tuple[str, str]:
        """Read plain text / Markdown files directly."""
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Failed to read '%s': %s", path.name, e)
            text = ""
        return text, METHOD_PASSTHROUGH

    def _convert_markitdown(
        self,
        path: Path,
        warnings: list[str],
    ) -> tuple[str, str]:
        """Convert via MarkItDown. Returns (markdown, method)."""
        try:
            result   = self._md.convert(str(path))
            markdown = result.text_content or ""
            if not markdown.strip():
                warnings.append(
                    f"MarkItDown produced empty output for '{path.name}'. "
                    "The file may be corrupted or contain only images."
                )
            return markdown, METHOD_MARKITDOWN
        except Exception as e:
            warnings.append(f"MarkItDown failed for '{path.name}': {e}")
            logger.warning("MarkItDown error on '%s': %s", path.name, e)
            return "", METHOD_MARKITDOWN

    def _convert_pdf(
        self,
        path: Path,
    ) -> tuple[str, str, list[str]]:
        """
        Convert PDF to Markdown.
        1. Try MarkItDown
        2. Check text density
        3. If density is too low and OCR is enabled: run OCR fallback
        """
        warnings: list[str] = []

        # Step 1: MarkItDown pass
        markdown = ""
        try:
            result   = self._md.convert(str(path))
            markdown = result.text_content or ""
        except Exception as e:
            warnings.append(f"MarkItDown failed: {e}")
            logger.warning("MarkItDown PDF error on '%s': %s", path.name, e)

        # Step 2: Estimate page count from MarkItDown output
        page_markers = len(re.findall(r"(?m)^-{3,}$", markdown))
        estimated_pages = max(page_markers, 1)

        # Step 3: Density check
        chars_per_page = len(markdown.strip()) / estimated_pages
        logger.debug(
            "'%s': ~%.0f chars/page (threshold=%d)",
            path.name, chars_per_page, self.min_chars_per_page,
        )

        if (
            chars_per_page < self.min_chars_per_page
            and self.ocr_enabled
        ):
            logger.info(
                "Low text density in '%s' (%.0f chars/page). "
                "Falling back to OCR.",
                path.name, chars_per_page,
            )
            ocr_markdown, ocr_warnings = self._ocr_pdf(path)
            warnings.extend(ocr_warnings)
            if ocr_markdown.strip():
                return ocr_markdown, METHOD_OCR, warnings
            else:
                warnings.append(
                    "OCR produced no output. "
                    "Returning MarkItDown result (may be empty)."
                )

        return markdown, METHOD_MARKITDOWN, warnings

    def _ocr_pdf(self, path: Path) -> tuple[str, list[str]]:
        """
        OCR fallback for scanned PDFs.
        Rasterises pages with pdf2image then runs Tesseract.
        Returns (markdown_text, warnings).
        """
        warnings: list[str] = []
        pages_text: list[str] = []

        try:
            from pdf2image import convert_from_path
            import pytesseract
        except ImportError as e:
            warnings.append(
                f"OCR dependencies missing ({e}). "
                "Install: pip install pdf2image pytesseract && "
                "sudo dnf install tesseract poppler-utils"
            )
            return "", warnings

        try:
            images = convert_from_path(str(path), dpi=200)
        except Exception as e:
            warnings.append(f"pdf2image failed: {e}")
            return "", warnings

        logger.info(
            "OCR: processing %d pages from '%s'", len(images), path.name
        )

        for page_num, image in enumerate(images, start=1):
            try:
                text = pytesseract.image_to_string(
                    image,
                    lang=self.ocr_language,
                    config="--psm 3",   # fully automatic page segmentation
                )
                if text.strip():
                    pages_text.append(
                        f"\n\n<!-- page {page_num} -->\n\n{text.strip()}"
                    )
            except Exception as e:
                warnings.append(f"OCR failed on page {page_num}: {e}")
                logger.warning("OCR error on page %d: %s", page_num, e)

        if not pages_text:
            warnings.append("OCR produced no text from any page.")
            return "", warnings

        markdown = "\n".join(pages_text)
        logger.info(
            "OCR complete: %d/%d pages yielded text.",
            len(pages_text), len(images),
        )
        return markdown, warnings

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    @staticmethod
    def _write_output(
        markdown: str,
        stem: str,
        output_dir: Path,
    ) -> str:
        """Write Markdown to output_dir/{stem}.md. Returns the path."""
        output_dir.mkdir(parents=True, exist_ok=True)
        # Sanitise stem for filesystem
        safe_stem = re.sub(r"[^\w\-]", "_", stem)
        out_path  = output_dir / f"{safe_stem}.md"
        out_path.write_text(markdown, encoding="utf-8")
        logger.debug("Wrote Markdown to '%s'", out_path)
        return str(out_path)


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def _normalise_markdown(text: str) -> str:
    """
    Clean up common MarkItDown / OCR output artifacts:
    - Collapse 3+ consecutive blank lines to 2
    - Strip trailing whitespace from each line
    - Ensure file ends with single newline
    """
    if not text:
        return ""
    # Strip trailing spaces per line
    lines = [line.rstrip() for line in text.splitlines()]
    text  = "\n".join(lines)
    # Collapse excessive blank lines
    text  = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def _estimate_page_count(markdown: str, extension: str) -> int:
    """
    Estimate page count from Markdown output.
    PDFs: count page-break markers or horizontal rules.
    Others: rough estimate from word count (250 words/page).
    """
    if extension == ".pdf":
        markers = len(re.findall(r"(?m)^-{3,}$", markdown))
        return max(markers, 1)
    words = len(markdown.split())
    return max(1, words // 250)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    converter = DocumentConverter(ocr_enabled=True)

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "markdown"

        # --- Plain text passthrough ---
        txt_path = Path(tmp) / "notes.txt"
        txt_path.write_text(
            "# My Notes\n\nThis is a plain text file.\n\n"
            "It has multiple paragraphs.\n",
            encoding="utf-8",
        )
        result = converter.convert(txt_path, output_dir=out_dir)
        assert result.method == METHOD_PASSTHROUGH
        assert "My Notes" in result.markdown
        assert result.char_count > 0
        assert result.output_path is not None
        print(f"Passthrough .txt: OK  (chars={result.char_count}, "
              f"words={result.word_count})")

        # --- Markdown passthrough ---
        md_path = Path(tmp) / "readme.md"
        md_path.write_text(
            "# README\n\n## Installation\n\nRun `pip install`.\n\n"
            "## Usage\n\nSee docs.\n",
            encoding="utf-8",
        )
        result = converter.convert(md_path, output_dir=out_dir)
        assert result.method == METHOD_PASSTHROUGH
        assert "README" in result.markdown
        print(f"Passthrough .md:  OK  (chars={result.char_count})")

        # --- HTML via MarkItDown ---
        html_path = Path(tmp) / "page.html"
        html_path.write_text(
            "<html><body>"
            "<h1>Hello World</h1>"
            "<p>This is a paragraph with <strong>bold</strong> text.</p>"
            "<ul><li>Item 1</li><li>Item 2</li></ul>"
            "</body></html>",
            encoding="utf-8",
        )
        result = converter.convert(html_path, output_dir=out_dir)
        assert result.method == METHOD_MARKITDOWN
        assert result.char_count > 0
        print(f"HTML via MarkItDown: OK  (chars={result.char_count}, "
              f"warnings={result.warnings})")

        # --- Normalisation ---
        messy = "Line 1   \nLine 2\n\n\n\n\nLine 3\n"
        clean = _normalise_markdown(messy)
        assert "\n\n\n" not in clean
        assert not any(line.endswith(" ") for line in clean.splitlines())
        print(f"Normalisation: OK")

        # --- Output file written ---
        written = list(out_dir.glob("*.md"))
        assert len(written) >= 2, f"Expected >=2 output files, got {len(written)}"
        print(f"Output files written: {[f.name for f in written]}")

        # --- Empty file handling ---
        empty_path = Path(tmp) / "empty.md"
        empty_path.write_text("", encoding="utf-8")
        result = converter.convert(empty_path)
        assert result.markdown == "" or result.markdown == "\n"
        print(f"Empty file: OK  (no crash)")

        print("\nAll DocumentConverter assertions passed.")
        print("\nNote: PDF and DOCX conversion require real files.")
        print("Those formats will be tested during integration testing.")
