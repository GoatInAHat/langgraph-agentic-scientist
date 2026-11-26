"""PDF text extraction: upstream's ordered fallback chain and its checks."""

from __future__ import annotations

import pytest
from review_helpers import PAPER_TEXT, make_pdf

from langgraph_agentic_scientist.review.pdf import (
    DEFAULT_EXTRACTORS,
    MAX_PDF_BYTES,
    MIN_PAPER_CHARS,
    PdfTextError,
    extract_pdf_text,
    extract_with_pymupdf,
    extract_with_pymupdf4llm,
    extract_with_pypdf,
    validate_pdf,
)

LONG = "x" * 150


def test_the_chain_is_upstreams_three_extractors_in_order() -> None:
    assert (
        extract_with_pymupdf4llm,
        extract_with_pymupdf,
        extract_with_pypdf,
    ) == DEFAULT_EXTRACTORS
    assert MIN_PAPER_CHARS == 100


def test_the_first_extractor_with_enough_text_wins() -> None:
    calls: list[str] = []

    def first(data: bytes) -> str:
        calls.append("first")
        return LONG

    def second(data: bytes) -> str:
        calls.append("second")
        return "y" * 200

    assert extract_pdf_text(b"%PDF-", extractors=(first, second)) == LONG
    assert calls == ["first"]


def test_a_short_text_or_a_failing_extractor_falls_through_to_the_next() -> None:
    def short(data: bytes) -> str:
        return "x" * 99

    def broken(data: bytes) -> str:
        raise RuntimeError("cannot parse")

    def good(data: bytes) -> str:
        return "z" * 100

    assert extract_pdf_text(b"%PDF-", extractors=(short, broken, good)) == "z" * 100


def test_the_minimum_is_inclusive_and_a_setting() -> None:
    assert extract_pdf_text(b"", extractors=(lambda data: "a" * 100,)) == "a" * 100
    with pytest.raises(PdfTextError, match="at least 100 characters"):
        extract_pdf_text(b"", extractors=(lambda data: "a" * 99,))
    assert extract_pdf_text(b"", min_chars=10, extractors=(lambda data: "a" * 10,)) == "a" * 10


def test_a_missing_library_is_skipped_and_named_when_nothing_else_helps() -> None:
    def missing(data: bytes) -> str:
        raise ModuleNotFoundError("no module", name="pymupdf4llm")

    assert extract_pdf_text(b"", extractors=(missing, lambda data: LONG)) == LONG
    with pytest.raises(PdfTextError, match="not installed: pymupdf4llm"):
        extract_pdf_text(b"", extractors=(missing,))


def test_the_real_chain_reads_a_text_pdf() -> None:
    pytest.importorskip("pymupdf")
    text = extract_pdf_text(make_pdf())
    assert "relationship between circle packing and search" in " ".join(text.split())


@pytest.mark.parametrize(
    "extractor",
    [extract_with_pymupdf4llm, extract_with_pymupdf, extract_with_pypdf],
    ids=["pymupdf4llm", "pymupdf", "pypdf"],
)
def test_each_extractor_reads_a_text_pdf(extractor) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("pymupdf")
    pytest.importorskip("pymupdf4llm")
    pytest.importorskip("pypdf")
    text = extractor(make_pdf())
    assert "circle packing" in " ".join(text.split())


def test_a_pdf_without_text_is_refused() -> None:
    pytest.importorskip("pymupdf")
    import pymupdf

    document = pymupdf.open()
    document.new_page()
    blank = bytes(document.tobytes())
    with pytest.raises(PdfTextError, match="at least 100 characters"):
        extract_pdf_text(blank)


def test_the_checks_before_extraction() -> None:
    validate_pdf(b"%PDF-1.7\n...", "paper/paper.pdf")
    validate_pdf(b"%PDF-1.7\n...", "paper/PAPER.PDF")
    with pytest.raises(PdfTextError, match="extension"):
        validate_pdf(b"%PDF-1.7", "paper/paper.txt")
    with pytest.raises(PdfTextError, match="header"):
        validate_pdf(b"<html>", "paper/paper.pdf")
    with pytest.raises(PdfTextError, match="too large"):
        validate_pdf(b"%PDF-" + b"0" * MAX_PDF_BYTES, "paper/paper.pdf")


def test_the_fixture_text_is_long_enough_for_the_review() -> None:
    assert len(PAPER_TEXT) >= MIN_PAPER_CHARS
