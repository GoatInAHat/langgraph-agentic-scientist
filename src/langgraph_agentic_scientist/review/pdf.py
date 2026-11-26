"""Extracting the manuscript text for the reviewer (upstream ``load_paper``).

Upstream tries three extractors in order and takes the first text of at least ``MIN_PAPER_CHARS``
(100) characters: ``pymupdf4llm`` (Markdown), ``pymupdf`` (plain text) and ``pypdf``. An extractor
that raises counts as an empty text. The libraries are the optional ``review`` extra of this
package; one that is not installed is skipped like a failing one, and the error says so when no
extractor produced enough text.
"""

from __future__ import annotations

import contextlib
import importlib
import io
from collections.abc import Callable, Sequence
from typing import Any

from ..core.errors import ScientistError

MIN_PAPER_CHARS = 100
"""Upstream's minimum length of an extracted text (``aireview.MIN_PAPER_CHARS``)."""

MAX_PDF_BYTES = 100 * 1024 * 1024
"""Upstream's limit on the size of a reviewed PDF (``final_ai_review._PAPER_PDF_MAX_BYTES``)."""

PdfExtractor = Callable[[bytes], str]
"""An extractor: the PDF bytes to text; it may raise."""


class PdfTextError(ScientistError, ValueError):
    """The PDF is not a PDF, is too large, or yields too little text."""


def validate_pdf(data: bytes, path: str) -> None:
    """Check what upstream checks before reviewing a PDF.

    Upstream requires a ``.pdf`` extension and a ``%PDF-`` header (``_validated_pdf_path``) and
    reads at most 100 MiB (``final_ai_review``).

    Raises:
        PdfTextError: When the path, the size or the header is wrong.
    """
    if not path.lower().endswith(".pdf"):
        raise PdfTextError("input must have a .pdf extension")
    if len(data) > MAX_PDF_BYTES:
        raise PdfTextError(f"final paper PDF is too large: {path}")
    if data[:5] != b"%PDF-":
        raise PdfTextError("input does not have a valid PDF header")


def _load(module: str) -> Any:
    """Import an optional library (typed as ``Any``: the extractors need no stubs)."""
    return importlib.import_module(module)


def extract_with_pymupdf4llm(data: bytes) -> str:
    """Return the Markdown text of the PDF (upstream's first extractor)."""
    pymupdf = _load("pymupdf")
    with contextlib.redirect_stdout(io.StringIO()):
        pymupdf4llm = _load("pymupdf4llm")  # it prints a banner on import, which upstream hides
    with pymupdf.open(stream=data, filetype="pdf") as document:
        return str(pymupdf4llm.to_markdown(document))


def extract_with_pymupdf(data: bytes) -> str:
    """Return the plain text of every page (upstream's second extractor)."""
    pymupdf = _load("pymupdf")
    with pymupdf.open(stream=data, filetype="pdf") as document:
        return "".join(page.get_text() for page in document)


def extract_with_pypdf(data: bytes) -> str:
    """Return the text of every page (upstream's third extractor)."""
    reader = _load("pypdf").PdfReader(io.BytesIO(data))
    return "".join(page.extract_text() or "" for page in reader.pages)


DEFAULT_EXTRACTORS: tuple[PdfExtractor, ...] = (
    extract_with_pymupdf4llm,
    extract_with_pymupdf,
    extract_with_pypdf,
)
"""Upstream's ordered fallback chain."""


def extract_pdf_text(
    data: bytes,
    *,
    min_chars: int = MIN_PAPER_CHARS,
    extractors: Sequence[PdfExtractor] = DEFAULT_EXTRACTORS,
) -> str:
    """Return the first extracted text of at least ``min_chars`` characters.

    Args:
        data: The PDF bytes.
        min_chars: The minimum length of an acceptable text (upstream 100).
        extractors: The extractors, tried in order; one that raises is skipped.

    Raises:
        PdfTextError: When no extractor produced enough text.
    """
    missing: list[str] = []
    for extract in extractors:
        try:
            text = extract(data)
        except ImportError as exc:
            missing.append(exc.name or type(exc).__name__)
            text = ""
        except Exception:
            text = ""
        if len(text) >= min_chars:
            return text
    message = f"could not extract at least {min_chars} characters from the PDF"
    if missing:
        message += (
            f" (not installed: {', '.join(sorted(set(missing)))}; "
            "install the 'review' extra of this package)"
        )
    raise PdfTextError(message)


__all__ = [
    "DEFAULT_EXTRACTORS",
    "MAX_PDF_BYTES",
    "MIN_PAPER_CHARS",
    "PdfExtractor",
    "PdfTextError",
    "extract_pdf_text",
    "extract_with_pymupdf",
    "extract_with_pymupdf4llm",
    "extract_with_pypdf",
    "validate_pdf",
]
