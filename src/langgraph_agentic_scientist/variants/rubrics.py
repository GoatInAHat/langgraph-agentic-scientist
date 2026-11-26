"""The rubric contract of the variants phase (upstream ``rubric_contract.py``).

Every variant directory must satisfy the same nine rubric families; a baseline adapted from an
external reference implementation also satisfies three more. This module builds
``rubric_contract.json`` and ``target_rubrics.json``, validates the ``rubric_status.json`` that
the verifier agent fills in, and builds the host-owned canonical status from the agent's bounded
judgments: the agent decides each rubric's ``pass`` and ``evidence``; the host owns the identity,
the metrics and the aggregate ``overall_pass``.

The rubric descriptions are package data (``workflow/variants/rubric_descriptions/<id>``), so a
caller can override them entry by entry.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.state import RubricStatus
from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .constants import (
    BASELINE_REFERENCE_ID,
    CONTRACT_VERSION,
    MAX_EVIDENCE_LENGTH,
    MAX_RECOMMENDATION_LENGTH,
    MAX_RUBRIC_STATUS_BYTES,
    MAX_RUBRICS,
    REFERENCE_RUBRIC_IDS,
    REQUIRED_RUBRIC_IDS,
)
from .workspace_ops import read_text_or_none

DESCRIPTION_KEY = "workflow/variants/rubric_descriptions/{id}"


def rubric_description(prompts: PromptRegistry, rubric_id: str) -> str:
    """Return the description of a rubric id from the registry."""
    return prompts.text(DESCRIPTION_KEY.format(id=rubric_id))


def build_rubric_contract(
    prompts: PromptRegistry,
    variant_id: str,
    variant_kind: str,
    base_program: str = "best",
) -> dict[str, Any]:
    """Build the ``rubric_contract.json`` of one variant.

    The contract is identical across variants except for identifying metadata. A baseline whose
    base program is an external reference (other than the reserved ``baseline_reference``, the
    original seed) also requires the three adaptation-audit rubrics.
    """
    rubrics: list[dict[str, str]] = []
    for rubric_id in REQUIRED_RUBRIC_IDS:
        rubrics.append(
            {
                "id": rubric_id,
                "description": rubric_description(prompts, rubric_id),
                "type": "structural"
                if "schema" in rubric_id or "files" in rubric_id
                else "semantic",
            }
        )
    if base_program == "reference" and variant_id != BASELINE_REFERENCE_ID:
        for rubric_id in REFERENCE_RUBRIC_IDS:
            rubrics.append(
                {
                    "id": rubric_id,
                    "description": rubric_description(prompts, rubric_id),
                    "type": "semantic",
                }
            )
    return {
        "contract_version": CONTRACT_VERSION,
        "variant_id": variant_id,
        "variant_kind": variant_kind,
        "rubrics": rubrics,
    }


def required_rubric_ids(
    prompts: PromptRegistry, variant_id: str, variant_kind: str, base_program: str
) -> list[str]:
    """Return the rubric ids a variant's contract requires, in contract order."""
    contract = build_rubric_contract(prompts, variant_id, variant_kind, base_program)
    return [rubric["id"] for rubric in contract["rubrics"]]


def build_target_rubrics(rubric_contract: Mapping[str, Any]) -> list[dict[str, str]]:
    """Extract the ``target_rubrics.json`` list from a rubric contract."""
    return [
        {"id": rubric["id"], "description": rubric["description"]}
        for rubric in rubric_contract["rubrics"]
    ]


def _validated_rubrics(
    rubrics: Any, required: set[str], max_evidence: int = MAX_EVIDENCE_LENGTH
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    if not isinstance(rubrics, list):
        return None, "rubric_status.rubrics must be a list"
    if len(rubrics) > MAX_RUBRICS:
        return None, "rubric_status.rubrics exceeds the count limit"
    by_id: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(rubrics):
        if not isinstance(entry, dict):
            return None, f"rubric entry at index {index} must be an object"
        rubric_id = entry.get("id")
        if not isinstance(rubric_id, str):
            return None, f"rubric entry at index {index} is missing a string id"
        if rubric_id in by_id:
            return None, f"rubric_status contains a duplicate rubric id at index {index}"
        passed = entry.get("pass")
        if not isinstance(passed, bool):
            return None, f"rubric entry pass at index {index} must be boolean"
        evidence = entry.get("evidence")
        if not isinstance(evidence, str) or not evidence.strip():
            return None, f"rubric entry evidence at index {index} must be non-empty"
        if len(evidence) > max_evidence:
            return None, f"rubric entry evidence at index {index} exceeds the length limit"
        by_id[rubric_id] = {"id": rubric_id, "pass": passed, "evidence": evidence.strip()}
    found = set(by_id)
    missing = required - found
    if missing:
        return None, f"rubric_status missing rubric ids: {sorted(missing)}"
    unexpected = found - required
    if unexpected:
        return None, f"rubric_status contains {len(unexpected)} unexpected rubric ids"
    return by_id, None


def validate_variant_rubric_status(
    rubric_status: Mapping[str, Any] | None,
    variant_id: str,
    required_rubric_ids: set[str] | None = None,
    *,
    max_evidence: int = MAX_EVIDENCE_LENGTH,
) -> tuple[bool, str | None]:
    """Validate that a ``rubric_status.json`` covers all required rubric families.

    Returns ``(passed, error_message)``; the message is ``None`` on success.
    """
    if rubric_status is None:
        return False, "missing or invalid rubric_status.json"
    for key in ("variant_id", "overall_pass", "rubrics"):
        if key not in rubric_status:
            return False, f"rubric_status missing key: {key}"
    if rubric_status.get("variant_id") != variant_id:
        return False, (
            f"variant_id mismatch: expected {variant_id!r}, got {rubric_status.get('variant_id')!r}"
        )
    required = (
        set(required_rubric_ids) if required_rubric_ids is not None else set(REQUIRED_RUBRIC_IDS)
    )
    rubrics_by_id, error = _validated_rubrics(rubric_status.get("rubrics"), required, max_evidence)
    if error is not None:
        return False, error
    assert rubrics_by_id is not None
    all_pass = all(rubrics_by_id[rubric_id]["pass"] for rubric_id in required)
    declared_pass = rubric_status.get("overall_pass")
    if not isinstance(declared_pass, bool):
        return False, "rubric_status.overall_pass must be boolean"
    if declared_pass != all_pass:
        return False, "rubric_status.overall_pass does not match required rubrics"
    return True, None


def canonicalize_variant_rubric_status(
    rubric_status: Mapping[str, Any] | None,
    *,
    variant_id: str,
    attempt_id: str,
    idea_id: str,
    required_rubric_ids: list[str],
    combined_score: Any,
    correct: bool,
    max_evidence: int = MAX_EVIDENCE_LENGTH,
) -> tuple[dict[str, Any] | None, str | None]:
    """Build the host-owned status from the agent's bounded rubric judgments.

    Returns ``(status, error)``; exactly one is ``None``.
    """
    if rubric_status is None:
        return None, "missing or invalid rubric_status.json"
    required = set(required_rubric_ids)
    rubrics_by_id, error = _validated_rubrics(rubric_status.get("rubrics"), required, max_evidence)
    if error is not None:
        return None, error
    recommendation = rubric_status.get("next_recommendation", "")
    if not isinstance(recommendation, str):
        return None, "rubric_status.next_recommendation must be a string"
    if len(recommendation) > MAX_RECOMMENDATION_LENGTH:
        return None, "rubric_status.next_recommendation exceeds the length limit"
    assert rubrics_by_id is not None
    canonical_rubrics = [rubrics_by_id[rubric_id] for rubric_id in required_rubric_ids]
    return (
        {
            "variant_id": variant_id,
            "attempt_id": attempt_id,
            "idea_id": idea_id,
            "overall_pass": correct and all(entry["pass"] for entry in canonical_rubrics),
            "rubrics": canonical_rubrics,
            "metrics": {"combined_score": combined_score, "correct": correct},
            "next_recommendation": recommendation,
        },
        None,
    )


async def load_rubric_status(workspace: Workspace, path: str) -> dict[str, Any] | None:
    """Load a ``rubric_status.json`` (at most 1 MiB), returning ``None`` on any failure."""
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file" or entry.size > MAX_RUBRIC_STATUS_BYTES:
        return None
    text = await read_text_or_none(workspace, path)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def rubric_status_record(status: Mapping[str, Any]) -> RubricStatus:
    """Convert a canonical status mapping into the core's :class:`RubricStatus`."""
    return RubricStatus.model_validate(dict(status))


def program_rubric_path(task_bundle: str, program_index: int) -> str:
    """Return the path of ``program_<n>/rubric_status.json`` of a task bundle."""
    return join(task_bundle, f"program_{program_index}", "rubric_status.json")


def variant_attempt_id(variant_id: str, program_index: int) -> str:
    """Return upstream's attempt id of a variant's program: ``<variant>_attempt_<nn>``."""
    return f"{variant_id}_attempt_{program_index:02d}"


__all__ = [
    "build_rubric_contract",
    "build_target_rubrics",
    "canonicalize_variant_rubric_status",
    "load_rubric_status",
    "program_rubric_path",
    "required_rubric_ids",
    "rubric_description",
    "rubric_status_record",
    "validate_variant_rubric_status",
    "variant_attempt_id",
]
