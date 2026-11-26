"""Review summaries: the score, the decision and their display (upstream ``review_summary``)."""

from __future__ import annotations

import pytest

from langgraph_agentic_scientist.revision.summary import (
    format_review_result,
    format_revision_review,
    parse_review_summary,
    revision_review_fields,
)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"Overall": 4, "Decision": "Reject"}, (4, "Reject")),
        ({"Overall": 6.5, "Decision": "Accept"}, (6.5, "Accept")),
        ({"Overall": True, "Decision": "Accept"}, (None, "Accept")),
        ({"Overall": 0, "Decision": "Weak Accept"}, (None, None)),
        ({"Overall": 11}, (None, None)),
        ({"Overall": float("nan")}, (None, None)),
        ({"Overall": "5"}, (None, None)),
        ({}, (None, None)),
    ],
)
def test_the_summary_of_a_review(payload: dict[str, object], expected: tuple[object, ...]) -> None:
    assert parse_review_summary(payload) == expected


def test_the_revision_fields_need_both_scores_for_a_delta() -> None:
    fields = revision_review_fields(
        {"Overall": 4, "Decision": "Reject"}, {"Overall": 3, "Decision": "Reject"}
    )
    assert fields == {
        "before_overall_score": 4,
        "after_overall_score": 3,
        "before_decision": "Reject",
        "after_decision": "Reject",
        "overall_score_delta": -1,
    }
    partial = revision_review_fields({"Overall": 4}, None)
    assert partial["overall_score_delta"] is None
    assert partial["after_overall_score"] is None


def test_the_display_lines_are_upstreams() -> None:
    fields = revision_review_fields(
        {"Overall": 4, "Decision": "Reject"}, {"Overall": 3, "Decision": "Reject"}
    )
    assert format_revision_review(fields) == (
        "Revision review · overall 4/10 → 3/10 (-1) · decision REJECT → REJECT"
    )
    assert format_revision_review(revision_review_fields(None, None)) == (
        "Revision review · overall unavailable → unavailable · decision unavailable → unavailable"
    )
    assert format_review_result({"overall_score": 7, "decision": "Accept"}) == (
        "Review result · score 7/10 · ACCEPT"
    )
    assert format_review_result({}) == ("Review result · score unavailable · decision unavailable")
