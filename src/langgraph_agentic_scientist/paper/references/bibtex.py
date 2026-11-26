"""BibTeX helpers shared by the reference retrieval and merge steps."""

from __future__ import annotations

import re
from typing import Any

CITATION_KEY_RE = re.compile(r"@\w+\{([^,]+),")
"""Matches the citation key of a BibTeX entry (upstream ``CITATION_KEY_RE``)."""


def extract_bibtex_key(bibtex: str) -> str | None:
    """Return the citation key of a BibTeX entry, or ``None``."""
    match = CITATION_KEY_RE.search(bibtex)
    if not match:
        return None
    citation_key = match.group(1).strip()
    return citation_key or None


def load_bibtex_rows(text: str) -> list[dict[str, str]]:
    """Split a ``.bib`` file into rows ``{"citation_key", "bibtex"}`` (upstream rule).

    An entry starts at a line whose first non-blank character is ``@``; an entry without a
    citation key is dropped.
    """
    text = text.strip()
    if not text:
        return []
    entries: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("@") and current:
            entries.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append("\n".join(current).strip())
    rows: list[dict[str, str]] = []
    for entry in entries:
        key = extract_bibtex_key(entry)
        if key is None:
            continue
        rows.append({"citation_key": key, "bibtex": entry})
    return rows


def bibtex_file_text(entries: list[str]) -> str:
    """Return the text of a ``.bib`` file holding ``entries`` (upstream ``write_bibtex_file``)."""
    cleaned = [entry.strip() for entry in entries if isinstance(entry, str) and entry.strip()]
    if not cleaned:
        raise ValueError("No BibTeX entries available to write.")
    return "\n\n".join(cleaned) + "\n"


def merge_reference_rows(
    *sources: tuple[list[dict[str, str]], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge rows of several sources, the first occurrence of a citation key winning.

    Each source is ``(bibtex rows, provenance rows)``. Returns the merged references (provenance
    row plus ``citation_key`` and ``bibtex``) and the merged provenance rows (without
    ``bibtex``), in order.
    """
    merged_refs: list[dict[str, Any]] = []
    merged_prov: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for bib_rows, prov_rows in sources:
        prov_by_key: dict[str, dict[str, Any]] = {}
        for row in prov_rows:
            key = str(row.get("citation_key", "")).strip()
            if key:
                prov_by_key[key] = row
        for row in bib_rows:
            key = str(row.get("citation_key", "")).strip()
            bibtex = row.get("bibtex")
            if not key or key in seen_keys or not isinstance(bibtex, str) or not bibtex.strip():
                continue
            prov_row = dict(prov_by_key.get(key, {}))
            prov_row["citation_key"] = key
            prov_row["bibtex"] = bibtex.strip()
            merged_refs.append(prov_row)
            merged_prov.append({k: v for k, v in prov_row.items() if k != "bibtex"})
            seen_keys.add(key)
    return merged_refs, merged_prov


__all__ = [
    "CITATION_KEY_RE",
    "bibtex_file_text",
    "extract_bibtex_key",
    "load_bibtex_rows",
    "merge_reference_rows",
]
