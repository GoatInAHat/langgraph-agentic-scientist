"""Qualification: the report, the failure history, the calibration and the judge's context.

Upstream qualifies a drafted scaffold by running the full production search workload once
(``task_drafting/qualification.py``). When that fails, the repair is a *pending transaction*
(``task.sh`` ``repair_qualification_transactionally``, ``qualification_repair.py``,
``qualification_calibration.py``, ``qualification_context.py``):

1. the failed report's scaffold fingerprint and category are appended to the failure history
   (:func:`record_failure`);
2. a repair agent edits the provisional draft (its prompt is assembled by
   :func:`render_qualification_repair_prompt`);
3. a repaired tree whose fingerprint is already in the history is an oscillation: the repair is
   rolled back and drafting ends with ``qualification_repair_oscillation``
   (:func:`reject_repeated_candidate`);
4. a calibration record binds the original and the calibrated fingerprints, the immutable
   accepted design and the qualification failure (:func:`create_calibration`); the transaction
   stays pending;
5. the repaired scaffold is qualified again and judged again, and the judge sees the calibration
   (:func:`alignment_context`);
6. only when alignment and qualification accept the same scaffold is the calibration accepted
   (:func:`accept_calibration`) and the transaction committed; any rejection rolls it back.

The records here are the typed state of that protocol; the graph
(:mod:`~langgraph_agentic_scientist.drafting.graph_scaffold`) drives it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel

from ..core.interfaces import GateReport
from ..core.render import split_repair_policy
from ..prompts import PromptRegistry
from .history import contains_sensitive_output, fingerprint_or_none, redact_sensitive_text
from .models import (
    QualificationCalibration,
    QualificationFailure,
    QualificationReceipt,
    QualificationRecord,
)

_HEX = frozenset("0123456789abcdef")

OSCILLATION_CATEGORY = "qualification_repair_oscillation"
OSCILLATION_DIAGNOSTIC = "Qualification repair reproduced a previously failed scaffold fingerprint."
"""The typed report of a repeated candidate (upstream ``reject_repeated_candidate``)."""

POST_VALIDATION_CATEGORY = "qualification_post_validation_failure"
ALIGNMENT_FAILURE_CATEGORY = "qualification_alignment_failure"
PASSED_CATEGORY = "qualification_passed"
DEADLINE_CATEGORY = "qualification_deadline_exhausted"
TERMINAL_GPU_CATEGORIES = frozenset({"gpu_allocation_invalid", "gpu_allocation_quarantined"})
"""Infrastructure categories that stop qualification at once (upstream ``task.sh``)."""

_ALLOWED_EVIDENCE_FIELDS = {
    "deadline_owner",
    "contract_wall_clock_seconds",
    "evaluator_timeout_seconds",
    "candidate_rpc_timeout_seconds",
    "broker_teardown_reserve_seconds",
    "container_cleanup_reserve_seconds",
    "candidate_rpc_calls",
    "candidate_calls_before_first_reset",
    "candidate_reset_count",
    "candidate_peer_disconnects",
    "max_output_bytes",
    "observed_output_bytes",
}


class QualificationError(ValueError):
    """A qualification record, calibration or receipt is invalid or stale."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in _HEX for char in value)


def _require_sha256(value: str, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_fingerprint(value: str, name: str) -> str:
    if not value.startswith("sha256:") or not _is_sha256(value.removeprefix("sha256:")):
        raise ValueError(f"{name} must be a sha256: task fingerprint")
    return value


def sha256_text(text: str | bytes) -> str:
    """Return the bare hex SHA-256 of ``text``."""
    data = text.encode("utf-8") if isinstance(text, str) else text
    return hashlib.sha256(data).hexdigest()


def report_text(value: BaseModel | Mapping[str, Any]) -> str:
    """Return the JSON text of a record, as a file of upstream's would hold it (sorted keys)."""
    payload = (
        value.model_dump(mode="json", by_alias=True)
        if hasattr(value, "model_dump")
        else dict(value)
    )
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


# --- the report ------------------------------------------------------------------------------


def record_from_gate(
    report: GateReport,
    *,
    receipt: QualificationReceipt | None = None,
) -> QualificationRecord:
    """Turn the gate's ``qualification`` report into a :class:`QualificationRecord`.

    Args:
        report: The report of ``TaskGate.run_check("qualification")``.
        receipt: The receipt of a passing qualification (built by the graph, which binds the
            publication plan, the evaluators and the requirements to the baseline's result).
    """
    status = (
        "pass"
        if report.passed
        else "infrastructure_failure"
        if report.infrastructure
        else "authored_failure"
    )
    elapsed = report.details.get("elapsed_seconds", 0.0)
    evidence = report.details.get("failure_evidence")
    return QualificationRecord(
        status=status,
        category=report.category or (PASSED_CATEGORY if report.passed else "qualification_failed"),
        stage=report.stage or "production-search",
        diagnostic="" if report.passed else report.diagnostic,
        task_fingerprint=report.fingerprint or "",
        elapsed_seconds=float(elapsed) if isinstance(elapsed, int | float) else 0.0,
        receipt=receipt if report.passed else None,
        failure_evidence=_projected_evidence(evidence) if isinstance(evidence, dict) else {},
    )


def _projected_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evidence.items()
        if key in _ALLOWED_EVIDENCE_FIELDS
        and isinstance(value, str | int | float)
        and not isinstance(value, bool)
    }


def deadline_exhausted_record() -> QualificationRecord:
    """Return the report upstream writes when no time is left for qualification."""
    return QualificationRecord(
        status="infrastructure_failure",
        category=DEADLINE_CATEGORY,
        stage="production-search",
        diagnostic=(
            "The overall task-drafting deadline has no time remaining for full qualification."
        ),
    )


# --- the failure history and the oscillation check ---------------------------------------------


def record_failure(
    history: list[QualificationFailure],
    *,
    fingerprint: str,
    category: str,
    evidence: str | bytes,
) -> list[QualificationFailure]:
    """Return the entries to append for a failed scaffold fingerprint.

    The result is empty when the fingerprint is empty (upstream records nothing) or already in the
    history, so the history keeps each fingerprint once.

    Raises:
        ValueError: For an invalid fingerprint or category.
    """
    if not fingerprint:
        return []
    _require_fingerprint(fingerprint, "fingerprint")
    if not category or len(category.encode("utf-8")) > 256:
        raise ValueError("category must be a short non-empty string")
    if any(item.fingerprint == fingerprint for item in history):
        return []
    return [
        QualificationFailure(
            fingerprint=fingerprint, category=category, evidence_sha256=sha256_text(evidence)
        )
    ]


def reject_repeated_candidate(
    history: list[QualificationFailure], *, fingerprint: str
) -> QualificationRecord | None:
    """Return the typed oscillation report when a candidate repeats a known failed fingerprint.

    Returns ``None`` for a new fingerprint.
    """
    _require_fingerprint(fingerprint, "fingerprint")
    if not any(item.fingerprint == fingerprint for item in history):
        return None
    return QualificationRecord(
        status="authored_failure",
        category=OSCILLATION_CATEGORY,
        stage="qualification-repair",
        diagnostic=OSCILLATION_DIAGNOSTIC,
        task_fingerprint=fingerprint,
    )


# --- the calibration ---------------------------------------------------------------------------


def create_calibration(
    *,
    original_fingerprint: str,
    calibrated_fingerprint: str,
    accepted_design_sha256: str,
    qualification_failure_sha256: str,
) -> QualificationCalibration:
    """Create the immutable-design calibration record of a provisional repair.

    Raises:
        ValueError: When a fingerprint or digest is malformed.
    """
    return QualificationCalibration(
        status="provisional",
        original_design_immutable=True,
        original_fingerprint=_require_fingerprint(original_fingerprint, "original_fingerprint"),
        calibrated_fingerprint=_require_fingerprint(
            calibrated_fingerprint, "calibrated_fingerprint"
        ),
        evidence_sha256={
            "accepted_design": _require_sha256(accepted_design_sha256, "accepted_design_sha256"),
            "qualification_failure": _require_sha256(
                qualification_failure_sha256, "qualification_failure_sha256"
            ),
        },
    )


def accept_calibration(
    calibration: QualificationCalibration,
    *,
    alignment_report_sha256: str,
    qualification_report_sha256: str,
    qualification_receipt_sha256: str,
) -> QualificationCalibration:
    """Bind a provisional calibration to both successful downstream gates.

    Raises:
        ValueError: When the calibration is not a valid provisional record or a digest is
            malformed.
    """
    evidence = calibration.evidence_sha256
    if (
        calibration.calibration_schema_version != 1
        or calibration.status != "provisional"
        or calibration.original_design_immutable is not True
        or not calibration.original_fingerprint.startswith("sha256:")
        or not calibration.calibrated_fingerprint.startswith("sha256:")
        or not _is_sha256(calibration.original_fingerprint.removeprefix("sha256:"))
        or not _is_sha256(calibration.calibrated_fingerprint.removeprefix("sha256:"))
        or set(evidence) != {"accepted_design", "qualification_failure"}
        or not all(_is_sha256(value) for value in evidence.values())
    ):
        raise ValueError("provisional calibration amendment is invalid")
    accepted = dict(evidence)
    accepted.update(
        {
            "alignment_report": _require_sha256(alignment_report_sha256, "alignment_report_sha256"),
            "qualification_report": _require_sha256(
                qualification_report_sha256, "qualification_report_sha256"
            ),
            "qualification_receipt": _require_sha256(
                qualification_receipt_sha256, "qualification_receipt_sha256"
            ),
        }
    )
    return calibration.model_copy(update={"status": "accepted", "evidence_sha256": accepted})


# --- the judge's context (upstream qualification_context.py and task.sh) ---------------------


def qualification_success_context(
    record: QualificationRecord, *, current_fingerprint: str, current_plan_sha256: str
) -> dict[str, object]:
    """Reject stale evidence before a production-search success is reused (and shown to the judge).

    Args:
        record: The accepted qualification report.
        current_fingerprint: The tree fingerprint of the scaffold now.
        current_plan_sha256: The publication plan digest of the scaffold now.

    Raises:
        QualificationError: When the success does not match the current scaffold.
    """
    receipt = record.receipt
    if (
        record.status != "pass"
        or record.category != PASSED_CATEGORY
        or record.stage != "production-search"
        or record.task_fingerprint != current_fingerprint
        or receipt is None
    ):
        raise QualificationError("qualification success does not match current scaffold")
    if not math.isfinite(record.elapsed_seconds) or record.elapsed_seconds < 0:
        raise QualificationError("qualification success has invalid elapsed time")
    if receipt.publication_plan_sha256 != current_plan_sha256 or (
        receipt.task_fingerprint != current_fingerprint
    ):
        raise QualificationError("task qualification receipt has a different publication plan")
    return {
        "status": "pass",
        "stage": "production-search",
        "task_fingerprint": current_fingerprint,
        "elapsed_seconds": record.elapsed_seconds,
        "report_sha256": sha256_text(report_text(record)),
        "receipt_sha256": sha256_text(report_text(receipt)),
    }


def qualification_failure_projection(
    record: QualificationRecord, *, current_fingerprint: str
) -> dict[str, object]:
    """Project a historical failure for the judge: allow-listed fields and a digest only."""
    payload = record.model_dump(mode="json")
    projection: dict[str, object] = {
        key: payload[key]
        for key in ("status", "category", "stage", "elapsed_seconds", "task_fingerprint")
        if isinstance(payload.get(key), str | int | float)
        and not isinstance(payload.get(key), bool)
    }
    projection["failure_evidence"] = _projected_evidence(record.failure_evidence)
    projection["output_evidence"] = {}
    projection["diagnostic_sha256"] = sha256_text(report_text(record))
    projection["untrusted_diagnostic_omitted"] = True
    projection["scope"] = "historical"
    projection["matches_current_scaffold"] = record.task_fingerprint == current_fingerprint
    return projection


def alignment_context(
    *,
    isolated: bool,
    calibration: QualificationCalibration | None,
    current_qualification: Mapping[str, object] | None,
    qualification_failure: Mapping[str, object] | None,
) -> str:
    """Build the harness attestation ``TASK_ALIGNMENT_CONTEXT`` of the judge prompt.

    Args:
        isolated: Whether the evaluator runs the candidate isolated (upstream's containerised
            run); the attestation names the guarantees accordingly.
        calibration: The provisional calibration of a pending qualification repair, which the
            judge compares against the canonical task (``qualification_calibration``).
        current_qualification: The success context (:func:`qualification_success_context`), or a
            pending marker; included when ``isolated``.
        qualification_failure: The failure projection of the repair's source report; included
            when ``isolated``.

    Returns:
        The attestation as JSON (sorted keys), as upstream's prompt carries it.
    """
    context: dict[str, object] = {
        "mode": "agent_authored_scaffold",
        "harness_guarantees": {
            "contract_preflight": "passed",
            "search_and_final_smoke": (
                "passed with reduced workload in isolated role containers"
                if isolated
                else "passed in native test topology"
            ),
            "production_entry": (
                "passed separately through ordinary device discovery and role-local topology"
                if isolated
                else "not asserted by native test topology"
            ),
        },
    }
    if calibration is not None:
        payload = calibration.model_dump(mode="json", by_alias=True)
        text = json.dumps(payload, sort_keys=True)
        if len(text.encode("utf-8")) > 16 * 1024:
            context["qualification_calibration"] = {"truncated_sha256_only": True}
        elif not contains_sensitive_output(text):
            context["qualification_calibration"] = payload
    if isolated:
        if current_qualification is not None:
            context["current_qualification"] = dict(current_qualification)
        if qualification_failure is not None:
            context["qualification_failure"] = dict(qualification_failure)
    return json.dumps(context, sort_keys=True)


# --- the qualification repair prompt (upstream task.sh) -----------------------------------


def encoded_block(text: str, maximum_bytes: int) -> str:
    """Return the base64 of the last ``maximum_bytes`` bytes of the redacted ``text``.

    An empty text gives an empty block, as an absent file does upstream.
    """
    if not text:
        return ""
    content = redact_sensitive_text(text).encode("utf-8")
    if len(content) > maximum_bytes:
        content = content[-maximum_bytes:]
    return base64.b64encode(content).decode("ascii")


def render_qualification_repair_prompt(
    prompts: PromptRegistry,
    *,
    diagnostic: str,
    design: str,
    alignment_report: str,
    calibration: str,
    post_validation: str,
    alignment_history: str,
    repair_history: str,
) -> str:
    """Assemble the qualification repairer's prompt as upstream's ``task.sh`` does.

    The inline paragraph and the five labelled base64 blocks (qualification diagnostic up to 16
    KiB, the immutable accepted design 96 KiB, the prior alignment report 16 KiB, the prior
    calibration record 16 KiB, the latest rejected repair validation 16 KiB; the last bytes are
    kept) come from ``workflow/drafting/qualification_repair``. Under the heading ``# Shared
    scaffold repair policy`` follow the preamble of ``drafting/repair_task`` (the text before
    ``# Role``) and its text from ``# Repair policy``; then the retained alignment history and
    the rejected-repair history.

    Args:
        prompts: The registry (so that a caller's override of either text is used).
        diagnostic: The text of the failed qualification report.
        design: The accepted design JSON, or the canonical ``task.md`` text when there is none.
        alignment_report: The prior alignment report text, or ``""``.
        calibration: The prior calibration record text, or ``""``.
        post_validation: The latest rejected repair validation text, or ``""``.
        alignment_history: The rendered alignment history, or ``""``.
        repair_history: The rendered rejected-repair history, or ``""``.

    Raises:
        ValueError: When ``drafting/repair_task`` has no ``# Repair policy`` heading.
    """
    template = prompts.expand_fragments(prompts.text("drafting/repair_task"))
    boundary_policy, repair_policy = split_repair_policy(template)
    text = prompts.render(
        "workflow/drafting/qualification_repair",
        {
            "DIAGNOSTIC_BASE64": encoded_block(diagnostic, 16 * 1024),
            "DESIGN_BASE64": encoded_block(design, 96 * 1024),
            "ALIGNMENT_REPORT_BASE64": encoded_block(alignment_report, 16 * 1024),
            "CALIBRATION_RECORD_BASE64": encoded_block(calibration, 16 * 1024),
            "POST_VALIDATION_BASE64": encoded_block(post_validation, 16 * 1024),
            "BOUNDARY_POLICY": boundary_policy,
            "REPAIR_POLICY": repair_policy,
        },
        fragments=False,
    )
    return (text + "\n" + alignment_history + "\n" + repair_history).rstrip("\n")


def is_fingerprint(value: object) -> bool:
    """Return whether ``value`` is a ``sha256:<64 hex>`` fingerprint."""
    return fingerprint_or_none(value) is not None


__all__ = [
    "ALIGNMENT_FAILURE_CATEGORY",
    "DEADLINE_CATEGORY",
    "OSCILLATION_CATEGORY",
    "OSCILLATION_DIAGNOSTIC",
    "PASSED_CATEGORY",
    "POST_VALIDATION_CATEGORY",
    "TERMINAL_GPU_CATEGORIES",
    "QualificationError",
    "accept_calibration",
    "alignment_context",
    "create_calibration",
    "deadline_exhausted_record",
    "encoded_block",
    "is_fingerprint",
    "qualification_failure_projection",
    "qualification_success_context",
    "record_failure",
    "record_from_gate",
    "reject_repeated_candidate",
    "render_qualification_repair_prompt",
    "report_text",
    "sha256_text",
]
