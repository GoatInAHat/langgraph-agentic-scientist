"""Grounding validation: reference provenance and the claims-to-sources map.

Upstream ``validate_reference_provenance`` and ``validate_claims_grounding`` (``paper_step.py``),
carried over unchanged: pure functions on the parsed JSON files.
"""

from __future__ import annotations

from typing import Any


def validate_reference_provenance(
    *,
    references_payload: dict[str, Any],
    bib_keys: set[str],
) -> tuple[set[str], list[str]]:
    """Validate references_provenance.json against references.bib."""
    errors: list[str] = []
    refs = references_payload.get("references")
    if not isinstance(refs, list):
        return set(), ["references_provenance.json must contain a `references` list"]

    provenance_keys: set[str] = set()
    provenance_ids: set[str] = set()
    for idx, ref_row in enumerate(refs):
        if not isinstance(ref_row, dict):
            errors.append(f"references[{idx}] must be an object")
            continue

        source_id = ref_row.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            errors.append(f"references[{idx}].id missing")
        else:
            if source_id in provenance_ids:
                errors.append(f"references[{idx}].id duplicates existing id: {source_id!r}")
            provenance_ids.add(source_id)

        citation_key = ref_row.get("citation_key")
        if not isinstance(citation_key, str) or not citation_key.strip():
            errors.append(f"references[{idx}].citation_key missing")
            continue
        if citation_key in provenance_keys:
            errors.append(
                f"references[{idx}].citation_key duplicates existing key: {citation_key!r}"
            )
        provenance_keys.add(citation_key)

    missing_in_provenance = sorted(bib_keys - provenance_keys)
    if missing_in_provenance:
        errors.append(
            "references.bib keys missing from references_provenance.json: "
            + ", ".join(repr(key) for key in missing_in_provenance)
        )

    missing_in_bib = sorted(provenance_keys - bib_keys)
    if missing_in_bib:
        errors.append(
            "references_provenance.json keys missing from references.bib: "
            + ", ".join(repr(key) for key in missing_in_bib)
        )

    return provenance_ids, errors


def validate_claims_grounding(
    *,
    claims_payload: dict[str, Any],
    allowed_source_ids: set[str],
    valid_citation_keys: set[str],
) -> list[str]:
    """Validate claims grounding."""
    errors: list[str] = []
    claims = claims_payload.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["claims_grounding.json must contain non-empty `claims` array"]

    for idx, claim_row in enumerate(claims):
        if not isinstance(claim_row, dict):
            errors.append(f"claims[{idx}] must be an object")
            continue
        claim_text = claim_row.get("claim")
        if not isinstance(claim_text, str) or not claim_text.strip():
            errors.append(f"claims[{idx}].claim must be non-empty string")
        source_ids = claim_row.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            errors.append(f"claims[{idx}].source_ids must be non-empty list")
        else:
            for source_id in source_ids:
                if not isinstance(source_id, str) or source_id not in allowed_source_ids:
                    errors.append(f"claims[{idx}].source_ids contains unknown id: {source_id!r}")

        citation_keys = claim_row.get("citation_keys")
        if citation_keys is None:
            continue
        if not isinstance(citation_keys, list):
            errors.append(f"claims[{idx}].citation_keys must be a list when provided")
            continue
        for citation_key in citation_keys:
            if not isinstance(citation_key, str) or citation_key not in valid_citation_keys:
                errors.append(f"claims[{idx}].citation_keys contains unknown key: {citation_key!r}")
    return errors
