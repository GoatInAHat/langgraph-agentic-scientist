"""Review summaries for events and terminal output (upstream ``review_summary.py``).

A summary is the overall score and the decision of a review. It is diagnostic: a review whose
fields are missing or invalid has no summary, and that never fails a phase.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

ReviewDecision = Literal["Accept", "Reject"]


def parse_review_summary(
    payload: Mapping[str, object],
) -> tuple[int | float | None, ReviewDecision | None]:
    """Return the overall score (1 to 10, finite) and the decision of a review payload."""
    raw_score = payload.get("Overall")
    overall: int | float | None = None
    if (isinstance(raw_score, int) and not isinstance(raw_score, bool)) or (
        isinstance(raw_score, float) and math.isfinite(raw_score)
    ):
        overall = raw_score if 1 <= raw_score <= 10 else None
    raw_decision = payload.get("Decision")
    decision: ReviewDecision | None = None
    if raw_decision == "Accept":
        decision = "Accept"
    elif raw_decision == "Reject":
        decision = "Reject"
    return overall, decision


def revision_review_fields(
    before: Mapping[str, object] | None, after: Mapping[str, object] | None
) -> dict[str, int | float | str | None]:
    """Return the before and after summaries and the score delta (upstream's event fields)."""
    before_score, before_decision = parse_review_summary(before or {})
    after_score, after_decision = parse_review_summary(after or {})
    return {
        "before_overall_score": before_score,
        "after_overall_score": after_score,
        "before_decision": before_decision,
        "after_decision": after_decision,
        "overall_score_delta": (
            after_score - before_score
            if before_score is not None and after_score is not None
            else None
        ),
    }


def _score_text(score: int | float | None) -> str:
    return f"{score:g}/10" if score is not None else "unavailable"


def format_review_result(fields: Mapping[str, object]) -> str:
    """Return the one-line display of a review (``Review result · score 4/10 · REJECT``)."""
    score, decision = parse_review_summary(
        {"Overall": fields.get("overall_score"), "Decision": fields.get("decision")}
    )
    decision_text = decision.upper() if decision else "decision unavailable"
    return f"Review result · score {_score_text(score)} · {decision_text}"


def format_revision_review(fields: Mapping[str, object]) -> str:
    """Return the one-line display of a revision's before and after assessments."""
    before_score, before_decision = parse_review_summary(
        {"Overall": fields.get("before_overall_score"), "Decision": fields.get("before_decision")}
    )
    after_score, after_decision = parse_review_summary(
        {"Overall": fields.get("after_overall_score"), "Decision": fields.get("after_decision")}
    )
    delta = (
        f" ({after_score - before_score:+g})"
        if before_score is not None and after_score is not None
        else ""
    )
    before_text = before_decision.upper() if before_decision else "unavailable"
    after_text = after_decision.upper() if after_decision else "unavailable"
    return (
        f"Revision review · overall {_score_text(before_score)} → {_score_text(after_score)}"
        f"{delta} · decision {before_text} → {after_text}"
    )


__all__ = [
    "ReviewDecision",
    "format_review_result",
    "format_revision_review",
    "parse_review_summary",
    "revision_review_fields",
]
