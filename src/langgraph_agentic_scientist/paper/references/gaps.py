"""Draft reference gaps: validation of the gap analyst's report (post-draft enrichment).

Upstream ``paper_reference_enrichment.py`` (the gap-report part). The draft-gap analyst (role
R31, run by the paper loop) reports where the draft lacks citations; ``status`` is ``ready`` with
rewrite targets, ``no_gaps``, or ``fail``. The rules are carried over unchanged.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ...core.state import utc_now
from ...workspace.paths import join
from ...workspace.protocol import Workspace

VALID_GAP_PRIORITIES = frozenset({"high", "medium", "low"})
VALID_CITATION_INTENTS = frozenset({"background", "positioning", "comparator", "benchmark_context"})


def validate_draft_reference_gaps_payload(payload: Mapping[str, Any]) -> None:
    """Validate a draft reference gap payload.

    Raises:
        ValueError: With upstream's message for the first rule the payload breaks.
    """
    status = str(payload.get("status", "")).strip()
    if status not in {"ready", "no_gaps", "fail"}:
        raise ValueError(
            "draft_reference_gaps.json must set status to 'ready', 'no_gaps', or 'fail'"
        )

    summary = " ".join(str(payload.get("summary", "")).split()).strip()
    if not summary:
        raise ValueError("draft_reference_gaps.json missing non-empty summary")

    rewrite_targets = payload.get("rewrite_targets", [])
    if not isinstance(rewrite_targets, list):
        raise ValueError("draft_reference_gaps.json field must be a list: rewrite_targets")

    missing_literatures = payload.get("missing_literatures", [])
    if not isinstance(missing_literatures, list):
        raise ValueError("draft_reference_gaps.json field must be a list: missing_literatures")

    if status == "fail":
        failure_reason = str(payload.get("failure_reason", "")).strip()
        if not failure_reason:
            raise ValueError(
                "draft_reference_gaps.json status=fail requires non-empty failure_reason"
            )
        return

    if status == "no_gaps":
        if rewrite_targets:
            raise ValueError(
                "draft_reference_gaps.json status=no_gaps requires empty rewrite_targets"
            )
        return

    if not rewrite_targets:
        raise ValueError(
            "draft_reference_gaps.json status=ready requires non-empty rewrite_targets"
        )

    seen_target_ids: set[str] = set()
    for idx, row in enumerate(rewrite_targets):
        if not isinstance(row, Mapping):
            raise ValueError(f"draft_reference_gaps.json rewrite_targets[{idx}] must be an object")
        target_id = str(row.get("target_id", "")).strip()
        section = " ".join(str(row.get("section", "")).split()).strip()
        priority = str(row.get("priority", "")).strip()
        citation_intent = str(row.get("citation_intent", "")).strip()
        reason = " ".join(str(row.get("reason", "")).split()).strip()
        anchor_text = " ".join(str(row.get("anchor_text", "")).split()).strip()
        search_topics = row.get("search_topics", [])

        if not target_id:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] missing non-empty target_id"
            )
        if target_id in seen_target_ids:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] duplicates target_id: {target_id}"
            )
        seen_target_ids.add(target_id)
        if not section:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] missing non-empty section"
            )
        if priority not in VALID_GAP_PRIORITIES:
            allowed = ", ".join(sorted(VALID_GAP_PRIORITIES))
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}].priority must be one of: {allowed}"
            )
        if citation_intent not in VALID_CITATION_INTENTS:
            allowed = ", ".join(sorted(VALID_CITATION_INTENTS))
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}].citation_intent must be one of: {allowed}"
            )
        if not reason:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] missing non-empty reason"
            )
        if not anchor_text:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] missing non-empty anchor_text"
            )
        if not isinstance(search_topics, list) or not search_topics:
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] must contain non-empty search_topics"
            )
        clean_topics = [" ".join(str(topic).split()).strip() for topic in search_topics]
        if any(not topic for topic in clean_topics):
            raise ValueError(
                f"draft_reference_gaps.json rewrite_targets[{idx}] contains empty search topic"
            )


async def write_draft_reference_gaps_validation_feedback(
    workspace: Workspace, paper_dir: str
) -> dict[str, Any]:
    """Validate ``draft_reference_gaps.md`` and ``.json`` and persist the feedback."""
    gaps_md = join(paper_dir, "draft_reference_gaps.md")
    gaps_json = join(paper_dir, "draft_reference_gaps.json")
    feedback: dict[str, Any] = {
        "generated_at": utc_now(),
        "paper_dir": workspace.host_path(paper_dir),
        "ok": False,
        "hard_errors": [],
    }
    try:
        if not (entry := await workspace.stat(gaps_md)) or entry.kind != "file":
            raise ValueError(
                f"Missing required draft reference gaps memo: {workspace.host_path(gaps_md)}"
            )
        if not (entry := await workspace.stat(gaps_json)) or entry.kind != "file":
            raise ValueError(
                f"Missing required draft reference gaps JSON: {workspace.host_path(gaps_json)}"
            )
        if not (await workspace.read_bytes(gaps_md)).decode("utf-8", errors="replace").strip():
            raise ValueError(f"Draft reference gaps memo is empty: {workspace.host_path(gaps_md)}")
        payload = json.loads((await workspace.read_bytes(gaps_json)).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object in {workspace.host_path(gaps_json)}")
        validate_draft_reference_gaps_payload(payload)
        if str(payload.get("status", "")).strip() == "fail":
            raise ValueError(
                f"draft reference gap analysis failed: {str(payload.get('failure_reason', '')).strip()}"
            )
    except (ValueError, UnicodeDecodeError) as exc:
        feedback["hard_errors"] = [str(exc)]
    else:
        feedback["ok"] = True

    await workspace.write_bytes(
        join(paper_dir, "draft_reference_gaps_validation_feedback.json"),
        json.dumps(feedback, indent=2, sort_keys=True).encode("utf-8"),
    )
    lines = [
        "# Draft Reference Gaps Validation Feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- paper_dir: `{feedback['paper_dir']}`",
        f"- ok: `{feedback['ok']}`",
        "",
    ]
    hard_errors = feedback.get("hard_errors", [])
    if isinstance(hard_errors, list) and hard_errors:
        lines.append("## Hard errors")
        lines.extend(f"- {message}" for message in hard_errors)
        lines.append("")
    await workspace.write_bytes(
        join(paper_dir, "draft_reference_gaps_validation_feedback.md"),
        (("\n".join(lines).strip()) + "\n").encode("utf-8"),
    )
    return feedback


__all__ = [
    "VALID_CITATION_INTENTS",
    "VALID_GAP_PRIORITIES",
    "validate_draft_reference_gaps_payload",
    "write_draft_reference_gaps_validation_feedback",
]
