"""The rubric contract of a post-hoc analysis workspace (upstream ``rubric_contract.py``).

Every analysis is judged by six rubrics. The descriptions are upstream's text, including the
description of ``checkpoint_guardrail_respected``, which still speaks of an evaluator rerun that
the analysis-only pipeline forbids. The status normalisation is upstream's: a verdict may be a
boolean or a string beginning with ``pass`` or ``fail``, and a rubric list may be given as a list
or as one entry per rubric id.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from ..workspace.protocol import Workspace

logger = logging.getLogger(__name__)

REQUIRED_RUBRIC_IDS: list[str] = [
    "analysis_summary_present",
    "analysis_manifest_present",
    "interestingness_rubric_present",
    "required_outputs_present",
    "claim_grounded",
    "checkpoint_guardrail_respected",
]
"""The six rubric ids every analysis must satisfy."""

RUBRIC_DESCRIPTIONS: dict[str, str] = {
    "analysis_summary_present": "analysis_summary.md exists and explains the analysis outcome.",
    "analysis_manifest_present": "analysis_manifest.json exists and records outputs, evidence, and evaluation usage.",  # noqa: E501
    "interestingness_rubric_present": "interestingness_rubric.json exists with structured qualitative assessments.",  # noqa: E501
    "required_outputs_present": "All expected outputs for this analysis are present on disk.",
    "claim_grounded": "The stated claim is backed by concrete evidence paths and produced artifacts.",  # noqa: E501
    "checkpoint_guardrail_respected": "Any evaluator rerun preserved correctness and respected the bootstrap metric guardrail.",  # noqa: E501
}
"""Upstream's rubric descriptions, verbatim."""


def build_rubric_contract(analysis_id: str, family: str) -> dict[str, Any]:
    """Return ``rubric_contract.json`` for one analysis."""
    rubrics = [
        {
            "id": rubric_id,
            "description": RUBRIC_DESCRIPTIONS[rubric_id],
            "type": "structural" if rubric_id.endswith("present") else "semantic",
        }
        for rubric_id in REQUIRED_RUBRIC_IDS
    ]
    return {
        "contract_version": 1,
        "analysis_id": analysis_id,
        "analysis_family": family,
        "rubrics": rubrics,
    }


def build_target_rubrics(rubric_contract: dict[str, Any]) -> list[dict[str, str]]:
    """Return ``target_rubrics.json``: the contract's rubrics as ``{id, description}``."""
    return [
        {"id": rubric["id"], "description": rubric["description"]}
        for rubric in rubric_contract.get("rubrics", [])
    ]


async def load_rubric_status(workspace: Workspace, path: str) -> dict[str, Any] | None:
    """Read a ``rubric_status.json``; ``None`` when absent, unreadable or not an object."""
    try:
        parsed = json.loads((await workspace.read_bytes(path)).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_verdict(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        canonical = re.sub(r"[^a-z]+", "", value.casefold())
        if canonical.startswith("pass"):
            return True
        if canonical.startswith("fail"):
            return False
    return None


def _normalize_overall_pass(
    rubric_status: dict[str, Any],
) -> tuple[bool | None, str | None]:
    for key in ("overall_pass", "all_passed"):
        if key not in rubric_status:
            continue
        overall_pass = _normalize_verdict(rubric_status.get(key))
        if overall_pass is None:
            return None, f"rubric_status.{key} must be a pass/fail verdict"
        return overall_pass, None
    return None, "rubric_status missing key: overall_pass"


def _normalize_rubric_entry(entry: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    rubric_id = entry.get("id")
    if not isinstance(rubric_id, str):
        return None, f"rubric entry missing 'id': {entry!r}"

    for key in ("pass", "status"):
        if key not in entry:
            continue
        verdict = _normalize_verdict(entry.get(key))
        if verdict is None:
            return None, f"rubric entry has invalid verdict: {entry!r}"
        normalized = dict(entry)
        normalized["pass"] = verdict
        return normalized, None

    return None, f"rubric entry missing 'pass' or 'status': {entry!r}"


def _normalize_rubrics_list(
    rubric_status: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    rubrics = rubric_status.get("rubrics")
    if rubrics is not None:
        if not isinstance(rubrics, list):
            return None, "rubric_status.rubrics must be a list"
        normalized_rubrics: list[dict[str, Any]] = []
        for entry in rubrics:
            if not isinstance(entry, dict):
                return None, f"rubric entry is not a dict: {entry!r}"
            normalized_entry, error = _normalize_rubric_entry(entry)
            if error is not None:
                return None, error
            assert normalized_entry is not None
            normalized_rubrics.append(normalized_entry)
        return normalized_rubrics, None

    normalized: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for rubric_id in REQUIRED_RUBRIC_IDS:
        entry = rubric_status.get(rubric_id)
        if entry is None:
            missing_ids.append(rubric_id)
            continue
        if not isinstance(entry, dict):
            return None, f"rubric entry is not a dict: {entry!r}"
        normalized_entry, error = _normalize_rubric_entry({"id": rubric_id, **entry})
        if error is not None:
            return None, error
        assert normalized_entry is not None
        normalized.append(normalized_entry)

    if missing_ids:
        return None, f"rubric_status missing rubric ids: {sorted(missing_ids)}"

    return normalized, None


def normalize_posthoc_rubric_status(
    rubric_status: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Normalise a ``rubric_status.json``; return the normalised object or an error text."""
    if rubric_status is None:
        return None, "missing or invalid rubric_status.json"

    analysis_id = rubric_status.get("analysis_id")
    if not isinstance(analysis_id, str):
        return None, "rubric_status missing key: analysis_id"

    overall_pass, overall_error = _normalize_overall_pass(rubric_status)
    if overall_error is not None:
        return None, overall_error
    assert overall_pass is not None

    rubric_entries = rubric_status
    nested_entries = rubric_status.get("rubric_status")
    if isinstance(nested_entries, dict):
        rubric_entries = {**nested_entries, **rubric_status}

    rubrics, rubrics_error = _normalize_rubrics_list(rubric_entries)
    if rubrics_error is not None:
        return None, rubrics_error
    assert rubrics is not None

    normalized = dict(rubric_status)
    normalized["analysis_id"] = analysis_id
    normalized["overall_pass"] = overall_pass
    normalized["rubrics"] = rubrics
    return normalized, None


def validate_posthoc_rubric_status(
    rubric_status: dict[str, Any] | None,
    analysis_id: str,
) -> tuple[bool, str | None]:
    """Validate a ``rubric_status.json`` against the contract; return ``(ok, error)``."""
    normalized, error = normalize_posthoc_rubric_status(rubric_status)
    if error is not None:
        return False, error
    assert normalized is not None

    if normalized.get("analysis_id") != analysis_id:
        return False, (
            f"analysis_id mismatch: expected {analysis_id!r}, got {normalized.get('analysis_id')!r}"
        )

    rubrics = normalized.get("rubrics")
    assert rubrics is not None

    found_ids: set[str] = set()
    for entry in rubrics:
        if not isinstance(entry, dict):
            return False, f"rubric entry is not a dict: {entry!r}"
        rubric_id = entry.get("id")
        if not isinstance(rubric_id, str):
            return False, f"rubric entry missing 'id': {entry!r}"
        if "pass" not in entry:
            return False, f"rubric entry missing 'pass': {entry!r}"
        found_ids.add(rubric_id)

    missing = set(REQUIRED_RUBRIC_IDS) - found_ids
    if missing:
        return False, f"rubric_status missing rubric ids: {sorted(missing)}"

    all_pass = all(
        entry.get("pass") is True for entry in rubrics if entry.get("id") in REQUIRED_RUBRIC_IDS
    )
    declared = normalized.get("overall_pass")
    if declared is True and not all_pass:
        return False, "overall_pass is true but not all required rubrics pass"
    if declared is not True and all_pass:
        logger.warning("rubric_status.overall_pass=%r but all required rubrics pass", declared)

    return True, None


__all__ = [
    "REQUIRED_RUBRIC_IDS",
    "RUBRIC_DESCRIPTIONS",
    "build_rubric_contract",
    "build_target_rubrics",
    "load_rubric_status",
    "normalize_posthoc_rubric_status",
    "validate_posthoc_rubric_status",
]
