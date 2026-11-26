"""Evidence retention checks for a revised manuscript (upstream ``revision_evidence.py``).

A revision may only change the text. Before it is published, upstream checks that

* the revised paper's ``paper_data.json`` still contains every datum of the original's, ignoring
  volatile timestamps, absolute locations and the editorial annotations of post-hoc analyses
  (:func:`validate_semantic_evidence`; the containment is matched as a set, specific items
  first);
* the revision has a method-scope audit, a passing evidence audit (``verdict`` ``pass``, no
  findings) and a passing validation report that records that audit, and that the hashes the
  audit recorded still describe the revision's inputs (:func:`validate_passing_audits`).

Two parts of the last check are the paper package's: the validity of the scope audit and the
recomputation of the audit's input hashes. A caller supplies them as the optional functions
``scope_audit`` and ``audit_input_hashes`` (``revision.scope_audit`` and
``revision.audit_input_hashes`` in ``GraphOverrides.slots``); without them the check covers what
the recorded files say, and the retention report says which checks were not made.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from ..workspace import Workspace, is_file, join

MAX_JSON_BYTES = 32 * 1024 * 1024
"""Upstream's bound on the JSON files read for the checks (``_MAX_JSON_BYTES``)."""

_VOLATILE_KEYS = frozenset(
    {
        "completed_at",
        "created_at",
        "generated_at",
        "started_at",
        "timestamp",
        "updated_at",
        "verified_at",
    }
)
_LOCATION_KEYS = frozenset({"path", "source_csv"})
_ABSOLUTE_LOCATION = "<absolute-location>"
_POSTHOC_EDITORIAL_FIELDS = frozenset(
    {
        "key_finding_for_paper",
        "relevance_justification",
        "recommended_usage",
        "relevance_to_thesis",
    }
)
_LEGACY_EVIDENCE_LABELS = {
    "all_programs": "program inventory",
    "task_names": "task inventory",
    "cross_task_results": "task inventory",
    "grounding_sources": "grounding inventory",
    "ablation_studies": "ablation inventory",
    "posthoc_analyses": "posthoc inventory",
    "final_evaluation": "final evaluation inventory",
    "downstream_study": "downstream study context",
}
COMPLETION_MANIFEST_AUDIT_PATH = "evidence-source/latest_manifest.json"
"""The completion manifest, which the host writes after the final audit and the audit excludes."""

AuditHashes = Callable[[str, Any], Awaitable[Mapping[str, str]]]
"""``(directory, runtime) -> {audit input key: sha256}``: recomputes an audit's input hashes."""

ScopeAudit = Callable[[str, Any], Awaitable[bool]]
"""``(directory, runtime) -> valid``: validates ``method_scope_audit.json`` of a directory."""


class EvidenceError(ValueError):
    """A revision loses or changes evidence, or lacks a passing audit."""


# --- containment of the original evidence -------------------------------------------------------


def immutable_paper_evidence(paper_data: dict[str, Any]) -> dict[str, Any]:
    """Return ``paper_data`` without the editorial annotations of its post-hoc analyses."""
    analyses = paper_data.get("posthoc_analyses")
    if not isinstance(analyses, list):
        return paper_data
    return {
        **paper_data,
        "posthoc_analyses": [
            {key: value for key, value in entry.items() if key not in _POSTHOC_EDITORIAL_FIELDS}
            if isinstance(entry, dict)
            else entry
            for entry in analyses
        ],
    }


def normalize(value: object, *, key: str | None = None, location_container: bool = False) -> object:
    """Drop volatile keys and replace absolute locations (upstream ``_normalize``)."""
    if isinstance(value, dict):
        return {
            item_key: normalize(item_value, key=item_key, location_container=key == "artifacts")
            for item_key, item_value in value.items()
            if item_key not in _VOLATILE_KEYS
        }
    if isinstance(value, list):
        return [normalize(item, key=key, location_container=location_container) for item in value]
    is_location = (
        location_container or key in _LOCATION_KEYS or bool(key and key.endswith(("_dir", "_path")))
    )
    if isinstance(value, str) and is_location and os.path.isabs(value):
        return _ABSOLUTE_LOCATION
    return value


def _specificity(value: object) -> int:
    if isinstance(value, dict):
        return 1 + sum(_specificity(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_specificity(item) for item in value)
    return 1


def contains(candidate: object, required: object) -> bool:
    """Whether ``candidate`` contains ``required`` (upstream ``_contains``).

    Mappings must have every required key with a containing value; a list must hold a distinct
    containing item for every required item (a bipartite matching, most specific items first);
    anything else must be equal.
    """
    if isinstance(required, dict):
        return isinstance(candidate, dict) and all(
            key in candidate and contains(candidate[key], value) for key, value in required.items()
        )
    if isinstance(required, list):
        if not isinstance(candidate, list) or len(candidate) < len(required):
            return False
        ordered = sorted(required, key=_specificity, reverse=True)
        matched = [-1] * len(candidate)

        def assign(required_index: int, seen: set[int]) -> bool:
            item = ordered[required_index]
            for candidate_index, candidate_item in enumerate(candidate):
                if candidate_index in seen or not contains(candidate_item, item):
                    continue
                seen.add(candidate_index)
                previous = matched[candidate_index]
                if previous == -1 or assign(previous, seen):
                    matched[candidate_index] = required_index
                    return True
            return False

        return all(assign(index, set()) for index in range(len(ordered)))
    return candidate == required


def contains_normalized(candidate: object, required: object) -> bool:
    """:func:`contains` after :func:`normalize` of both sides."""
    return contains(normalize(candidate), normalize(required))


def validate_semantic_evidence(source: dict[str, Any], candidate: dict[str, Any]) -> None:
    """Refuse a revision that loses or changes the original's ``paper_data`` evidence.

    Raises:
        EvidenceError: Naming the inventory that was lost when one is identifiable.
    """
    original = immutable_paper_evidence(source)
    revised = immutable_paper_evidence(candidate)
    if contains_normalized(revised, original):
        return
    for field, label in _LEGACY_EVIDENCE_LABELS.items():
        if field in original and not contains_normalized(revised.get(field), original[field]):
            raise EvidenceError(f"revision loses or changes {label}")
    raise EvidenceError("revision loses or changes paper data evidence")


# --- audits -------------------------------------------------------------------------------------


def candidate_audit_hashes(hashes: Mapping[str, str]) -> dict[str, str]:
    """Exclude the completion manifest from the hashes compared for a candidate revision."""
    return {
        path: digest for path, digest in hashes.items() if path != COMPLETION_MANIFEST_AUDIT_PATH
    }


def published_audit_hashes(hashes: Mapping[str, str]) -> dict[str, str]:
    """Exclude what publication changes: the manifest, the reviews and the revision records."""
    return {
        path: digest
        for path, digest in hashes.items()
        if path != COMPLETION_MANIFEST_AUDIT_PATH
        and not path.startswith("evidence-source/ai_review.")
        and not path.startswith("evidence-source/revisions/")
    }


async def read_json_object(workspace: Workspace, path: str, description: str) -> dict[str, Any]:
    """Read a JSON object file of at most 32 MiB.

    Raises:
        EvidenceError: When the file is missing, too large, not JSON or not an object.
    """
    try:
        data = await workspace.read_bytes(path)
    except FileNotFoundError as exc:
        raise EvidenceError(f"missing {description}: {path}") from exc
    if len(data) > MAX_JSON_BYTES:
        raise EvidenceError(f"{description} is too large: {path}")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise EvidenceError(f"invalid {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise EvidenceError(f"{description} must contain a JSON object")
    return payload


def audit_record_passes(audit: Mapping[str, Any]) -> bool:
    """Whether an evidence audit record is a pass with no findings."""
    return (
        audit.get("verdict") == "pass"
        and audit.get("finding_count") == 0
        and audit.get("findings") == []
    )


def report_passes(report: Mapping[str, Any]) -> bool:
    """Whether a validation report is ok and records a passing evidence audit."""
    recorded = report.get("agentic_evidence_audit")
    return (
        report_ok(report)
        and isinstance(recorded, dict)
        and recorded.get("checked") is True
        and audit_record_passes(recorded)
    )


def report_ok(report: Mapping[str, Any]) -> bool:
    """Whether a validation report passed (upstream's ``ok``; ``passed`` is accepted too)."""
    return report.get("ok", report.get("passed")) is True


async def validate_passing_audits(
    workspace: Workspace,
    directory: str,
    *,
    published: bool,
    runtime: Any = None,
    audit_input_hashes: AuditHashes | None = None,
    scope_audit: ScopeAudit | None = None,
) -> list[str]:
    """Check that a revision has a method-scope audit and a current passing evidence audit.

    Args:
        workspace: The workspace.
        directory: The revision (or published paper) directory.
        published: Compare the hashes as publication leaves them (the reviews and the revision
            records are excluded) rather than as a candidate has them.
        runtime: Passed to the optional functions.
        audit_input_hashes: Recomputes the audit's input hashes for ``directory``; without it the
            recorded hashes of the audit and of the report are only compared with each other.
        scope_audit: Validates the method-scope audit; without it only its presence is checked.

    Returns:
        The names of the checks that were not made.

    Raises:
        EvidenceError: When an audit is missing, failing or stale.
    """
    skipped: list[str] = []
    for name in ("method_scope_audit.json", "method_scope_audit.md"):
        if not await is_file(workspace, join(directory, name)):
            raise EvidenceError(f"revision lacks a valid method-scope audit: {name}")
    if scope_audit is not None:
        try:
            valid = await scope_audit(directory, runtime)
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise EvidenceError("could not verify revision method-scope audit") from exc
        if valid is not True:
            raise EvidenceError("revision lacks a valid method-scope audit")
    else:
        skipped.append("method_scope_audit validity")

    audit = await read_json_object(
        workspace, join(directory, "paper_evidence_audit.json"), "agentic paper-evidence audit"
    )
    report = await read_json_object(
        workspace, join(directory, "validation_report.json"), "validation report"
    )
    select = published_audit_hashes if published else candidate_audit_hashes
    recorded_audit = audit.get("input_sha256")
    report_audit = report.get("agentic_evidence_audit")
    recorded_report = report_audit.get("input_sha256") if isinstance(report_audit, dict) else None
    if audit_input_hashes is not None:
        try:
            expected = select(await audit_input_hashes(directory, runtime))
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            raise EvidenceError("could not verify current agentic evidence audit") from exc
        hashes_match = bool(expected) and (
            isinstance(recorded_audit, dict)
            and select(recorded_audit) == expected
            and isinstance(recorded_report, dict)
            and select(recorded_report) == expected
        )
    else:
        skipped.append("audit input hashes")
        hashes_match = (
            isinstance(recorded_audit, dict)
            and bool(recorded_audit)
            and isinstance(recorded_report, dict)
            and select(recorded_audit) == select(recorded_report)
        )
    if not (audit_record_passes(audit) and report_passes(report) and hashes_match):
        raise EvidenceError("revision lacks a current passing agentic evidence audit")
    return skipped


async def validate_revision_evidence(
    workspace: Workspace,
    source_dir: str,
    candidate_dir: str,
    *,
    published: bool = False,
    runtime: Any = None,
    audit_input_hashes: AuditHashes | None = None,
    scope_audit: ScopeAudit | None = None,
) -> list[str]:
    """Reject a revision that loses staged evidence or lacks passing audits.

    ``source_dir`` is the original paper (or its snapshot) and ``candidate_dir`` the revision (or,
    with ``published=True``, the promoted paper). Returns the names of the checks not made (see
    :func:`validate_passing_audits`).

    Raises:
        EvidenceError: When the check fails.
    """
    source = await read_json_object(
        workspace, join(source_dir, "paper_data.json"), "source paper data"
    )
    candidate = await read_json_object(
        workspace, join(candidate_dir, "paper_data.json"), "revision paper data"
    )
    validate_semantic_evidence(source, candidate)
    return await validate_passing_audits(
        workspace,
        candidate_dir,
        published=published,
        runtime=runtime,
        audit_input_hashes=audit_input_hashes,
        scope_audit=scope_audit,
    )


__all__ = [
    "COMPLETION_MANIFEST_AUDIT_PATH",
    "MAX_JSON_BYTES",
    "AuditHashes",
    "EvidenceError",
    "ScopeAudit",
    "audit_record_passes",
    "candidate_audit_hashes",
    "contains",
    "contains_normalized",
    "immutable_paper_evidence",
    "normalize",
    "published_audit_hashes",
    "read_json_object",
    "report_ok",
    "report_passes",
    "validate_passing_audits",
    "validate_revision_evidence",
    "validate_semantic_evidence",
]
