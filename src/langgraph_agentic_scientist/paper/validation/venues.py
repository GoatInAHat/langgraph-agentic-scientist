"""Venue-specific validation: author block and framing language.

Upstream ``validate_venue_alignment`` (``paper_step.py``) and its helpers. A venue that requires
real author metadata (single-blind venues, and venues that opt into the ``alife_framing`` check)
must carry the canonical author block; the title, abstract, introduction and conclusion are
checked for venue-facing framing language (soft warnings). The venue catalog replaces upstream's
repository-relative venue files.
"""

# ruff: noqa: E501

from __future__ import annotations

import re
from typing import Any

from ...venues import VenueCatalog
from .files import PaperTree, read_text
from .tex import (
    _contains_any_term,
    _extract_tex_command_body,
    _extract_tex_section_body,
    _extract_tex_title,
    _normalize_tex_whitespace,
    _strip_tex_comments,
)

GENERIC_FORBIDDEN_AUTHOR_TERMS = (
    "Anonymous Authors",
    "Anonymous Author",
    "Automated Research System",
    "Institution",
    "contact@institution.edu",
)
VENUE_REAL_AUTHOR_MARKERS = (
    "single-blind",
    "single blind",
    "not anonymous",
    "include real author names",
    "include author names",
    "real author names and affiliations",
)
ALIFE_REQUIRED_FRAMING_TERMS = (
    "artificial life",
    "life-like",
    "lifelike",
    "complex adaptive",
    "adaptive computation",
    "self-organization",
    "self-organisation",
    "robustness",
    "plasticity",
    "biologically plausible",
    "dale",
)
ALIFE_GENERIC_DRIFT_TERMS = (
    "edge devices",
    "deployment",
    "resource-constrained deployment",
    "scientific computing",
    "socio-technical",
)

CANONICAL_AUTHOR_INPUT_RE = re.compile(r"\\input\s*\{(?:\./)?paper_authors\.tex\}")


def _matches_canonical_author_block(author_text: str, canonical_text: str) -> bool:
    normalized_author = _normalize_tex_whitespace(_strip_tex_comments(author_text))
    if CANONICAL_AUTHOR_INPUT_RE.fullmatch(normalized_author):
        return True
    normalized_canonical = _normalize_tex_whitespace(_strip_tex_comments(canonical_text))
    return normalized_author == normalized_canonical


def _infer_venue_profile(
    *, paper_format: str | None, venue_brief_text: str, catalog: VenueCatalog
) -> dict[str, Any]:
    """Infer validation expectations from paper format and venue brief."""
    venue_lower = venue_brief_text.lower()
    is_alife = (
        catalog.has_check(paper_format, "alife_framing")
        or "artificial life" in venue_lower
        or "living and lifelike complex adaptive systems" in venue_lower
        or "life-like" in venue_lower
        or "lifelike" in venue_lower
    )
    require_real_authors = is_alife or any(
        marker in venue_lower for marker in VENUE_REAL_AUTHOR_MARKERS
    )
    return {
        "checked": require_real_authors or is_alife,
        "label": "ALIFE" if is_alife else "venue",
        "require_real_authors": require_real_authors,
        "framing_terms": ALIFE_REQUIRED_FRAMING_TERMS if is_alife else (),
        "generic_drift_terms": ALIFE_GENERIC_DRIFT_TERMS if is_alife else (),
    }


def validate_venue_alignment(
    tex_text: str,
    *,
    venue_brief_text: str,
    paper_format: str | None,
    paper_dir: PaperTree | None = None,
    catalog: VenueCatalog | None = None,
) -> dict[str, Any]:
    """Validate venue-specific author and framing expectations."""
    catalog = catalog or VenueCatalog.default()
    profile = _infer_venue_profile(
        paper_format=paper_format, venue_brief_text=venue_brief_text, catalog=catalog
    )
    report: dict[str, Any] = {
        "checked": profile["checked"],
        "paper_format": paper_format,
        "hard_errors": [],
        "soft_warnings": [],
    }
    if not profile["checked"]:
        return report

    hard_errors: list[str] = []
    soft_warnings: list[str] = []
    label = str(profile["label"])

    if bool(profile["require_real_authors"]):
        author_text = _extract_tex_command_body(tex_text, "author")
        canonical_author_text: str | None = None
        if paper_dir is not None:
            canonical_author_source = read_text(paper_dir, "paper_authors.tex")
            if canonical_author_source is not None:
                canonical_author_text = _strip_tex_comments(canonical_author_source).strip()
        if not author_text:
            hard_errors.append(
                f"{label} paper is missing an author block; submissions are not anonymous"
            )
        for term in GENERIC_FORBIDDEN_AUTHOR_TERMS:
            if term.lower() in author_text.lower():
                hard_errors.append(
                    f"{label} author block uses forbidden anonymous/generic placeholder {term!r}"
                )
        if (
            canonical_author_text is not None
            and author_text
            and not _matches_canonical_author_block(author_text, canonical_author_text)
        ):
            hard_errors.append(
                f"{label} author block does not match canonical `./paper_authors.tex`"
            )

    title_text = _extract_tex_title(tex_text)
    abstract_match = re.search(
        r"\\begin\{abstract\}(?P<body>.*?)\\end\{abstract\}", tex_text, re.DOTALL
    )
    introduction_text = _extract_tex_section_body(tex_text, "Introduction")
    conclusion_text = _extract_tex_section_body(tex_text, "Conclusion")
    section_texts = {
        "title": title_text,
        "abstract": abstract_match.group("body").strip() if abstract_match is not None else "",
        "introduction": introduction_text,
        "conclusion": conclusion_text,
    }
    framing_terms = tuple(profile["framing_terms"])
    if framing_terms:
        for section_name, section_text in section_texts.items():
            if not section_text:
                continue
            if not _contains_any_term(section_text, framing_terms):
                soft_warnings.append(
                    f"{label} paper {section_name} lacks explicit {label}-facing framing language"
                )

        discussion_text = _extract_tex_section_body(tex_text, "Discussion")
        drift_scope = "\n".join(
            part for part in (discussion_text, section_texts["conclusion"]) if part
        )
        drift_terms = tuple(profile["generic_drift_terms"])
        if drift_scope and drift_terms and _contains_any_term(drift_scope, drift_terms):
            soft_warnings.append(
                f"{label} paper discussion/conclusion drifts into generic deployment or application framing without clear venue-specific relevance"
            )

    report["hard_errors"] = hard_errors
    report["soft_warnings"] = soft_warnings
    return report


def detect_paper_format(tex_text: str, catalog: VenueCatalog | None = None) -> str | None:
    """Infer the paper format from the venue whose LaTeX markers match."""
    venue = (catalog or VenueCatalog.default()).detect(tex_text)
    return venue.name if venue is not None else None
