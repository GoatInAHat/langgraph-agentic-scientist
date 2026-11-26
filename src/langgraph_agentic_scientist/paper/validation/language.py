"""Visible-language validation of standalone papers.

Upstream ``validate_visible_language`` (``paper_step.py``): a standalone paper must not leak the
pipeline's internal terms into its visible text. Pure function on manuscript text.
"""

# ruff: noqa: E501

from __future__ import annotations

import re
from typing import Any

from .tex import _visible_tex_text

STANDALONE_LANGUAGE_SOFT_WARNING_TERMS = (
    "task target",
    "aspirational target",
    "implementations",
)


def _standalone_language_patterns(forbidden_terms: list[str]) -> list[tuple[str, re.Pattern[str]]]:
    """Compile visible-language checks for standalone paper mode."""
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for term in forbidden_terms:
        clean = term.strip()
        if not clean:
            continue
        escaped = re.escape(clean)
        if re.fullmatch(r"[A-Za-z0-9_]+", clean):
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)
        else:
            pattern = re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", re.IGNORECASE)
        patterns.append((clean, pattern))
    return patterns


def validate_visible_language(
    tex_text: str,
    *,
    paper_profile: str,
    presentation_contract: dict[str, Any],
) -> dict[str, Any]:
    """Validate visible standalone-paper wording without changing default behavior."""
    report = {
        "checked": paper_profile == "standalone",
        "paper_profile": paper_profile,
        "hard_errors": [],
        "soft_warnings": [],
        "forbidden_terms": [],
    }
    if paper_profile != "standalone":
        return report

    forbidden_terms_raw = presentation_contract.get("forbidden_visible_terms")
    forbidden_terms = (
        [
            str(term).strip()
            for term in forbidden_terms_raw
            if isinstance(term, str) and term.strip()
        ]
        if isinstance(forbidden_terms_raw, list)
        else []
    )
    for extra_term in ("first eligible", "correctness threshold"):
        if extra_term not in forbidden_terms:
            forbidden_terms.append(extra_term)
    report["forbidden_terms"] = forbidden_terms

    visible_text = _visible_tex_text(tex_text)
    hard_errors: list[str] = []
    for term, pattern in _standalone_language_patterns(forbidden_terms):
        match = pattern.search(visible_text)
        if match is None:
            continue
        excerpt_start = max(0, match.start() - 40)
        excerpt_end = min(len(visible_text), match.end() + 40)
        excerpt = visible_text[excerpt_start:excerpt_end].strip()
        hard_errors.append(
            f"visible standalone paper text uses forbidden pipeline term {term!r}: ...{excerpt}..."
        )

    soft_warnings: list[str] = []
    for term in STANDALONE_LANGUAGE_SOFT_WARNING_TERMS:
        if re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])", visible_text, re.IGNORECASE
        ):
            soft_warnings.append(
                f"visible standalone paper text uses discouraged phrasing {term!r}; prefer reader-facing language"
            )

    report["hard_errors"] = hard_errors
    report["soft_warnings"] = soft_warnings
    return report
