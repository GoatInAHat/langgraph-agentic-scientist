"""The review as feedback for the revision writer (upstream ``revision.py`` and ``paper_edit.sh``).

Upstream hands the revision writer a bounded, normalised subset of the external review as
``revision_feedback.json`` and ``revision_feedback.md``. Two checks apply, in this order:

* ``revision.load_revision_feedback`` reads the review (at most 1 MiB) and requires ``Summary`` to
  be a string and ``Strengths``, ``Weaknesses``, ``Questions`` and ``Limitations`` to be lists of
  strings;
* ``paper_edit.sh``'s ``stage_external_review_feedback`` checks the file again, additionally
  refuses lists of more than 100 items, keeps the sixteen fields of a review that are present,
  and marks the data as untrusted: ``untrusted_reviewer_data`` is true, ``source_sha256`` is the
  digest of the review file, and the Markdown rendering starts with a warning line and lists the
  five text fields.

The second check and both renderings are the paper package's
(:func:`~langgraph_agentic_scientist.paper.workspace.normalize_external_review`); this module
runs the first and then the second, with upstream's messages.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import Field

from ..core.state import StateModel
from ..paper.workspace import (
    EXTERNAL_REVIEW_FIELDS,
    EXTERNAL_REVIEW_MAX_BYTES,
    PaperPreparationError,
    normalize_external_review,
    render_external_review_json,
    render_external_review_markdown,
)

MAX_REVIEW_BYTES = EXTERNAL_REVIEW_MAX_BYTES
"""Upstream's bound on the size of a review used as feedback (``_MAX_REVIEW_BYTES``)."""

MAX_LIST_ITEMS = 100
"""Upstream's bound on the items of each list field of a review used as feedback."""

LIST_FIELDS = ("Strengths", "Weaknesses", "Questions", "Limitations")
"""The list fields upstream checks."""

KEPT_FIELDS = EXTERNAL_REVIEW_FIELDS
"""The sixteen fields of a review that are kept in the feedback when present."""

MARKDOWN_FIELDS = ("Summary", "Strengths", "Weaknesses", "Questions", "Limitations")
"""The fields that the Markdown rendering lists."""


class FeedbackError(ValueError):
    """The review cannot be used as revision feedback."""


class RevisionFeedback(StateModel):
    """The normalised feedback of a review (the content of ``revision_feedback.json``).

    Attributes:
        source_sha256: SHA-256 of the review file the feedback was taken from.
        values: The kept fields of the review that are present, under upstream's keys.
    """

    untrusted_reviewer_data: bool = True
    source_sha256: str
    values: dict[str, Any] = Field(default_factory=dict)

    def as_object(self) -> dict[str, Any]:
        """Return the JSON object of ``revision_feedback.json``."""
        return {
            "untrusted_reviewer_data": self.untrusted_reviewer_data,
            "source_sha256": self.source_sha256,
            **self.values,
        }


def load_revision_feedback(raw: bytes, path: str) -> RevisionFeedback:
    """Validate a review file and return its normalised feedback.

    Args:
        raw: The bytes of the review JSON.
        path: The review file's path, for upstream's message about invalid JSON.

    Raises:
        FeedbackError: When the review is too large, is not a JSON object, or a field has the
            wrong type or too many items.
    """
    if len(raw) > MAX_REVIEW_BYTES:
        raise FeedbackError(f"External review must be at most {MAX_REVIEW_BYTES} bytes")
    try:
        review = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FeedbackError(f"invalid AI review JSON: {path}") from exc
    if not isinstance(review, dict):
        raise FeedbackError("AI review must contain a JSON object")
    if not isinstance(review.get("Summary", ""), str):
        raise FeedbackError("Summary must be a string")
    for field in LIST_FIELDS:
        value = review.get(field, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise FeedbackError(f"{field} must be a list of strings")
    try:
        normalized = normalize_external_review(raw)
    except PaperPreparationError as exc:
        raise FeedbackError(str(exc)) from exc
    return RevisionFeedback(
        source_sha256=normalized["source_sha256"],
        values={field: normalized[field] for field in KEPT_FIELDS if field in normalized},
    )


def render_revision_feedback_json(feedback: RevisionFeedback) -> str:
    """Return ``revision_feedback.json`` as upstream writes it (sorted keys, two-space indent)."""
    return render_external_review_json(feedback.as_object())


def render_revision_feedback_markdown(feedback: RevisionFeedback) -> str:
    """Return ``revision_feedback.md`` as upstream writes it."""
    return render_external_review_markdown(feedback.as_object())


__all__ = [
    "KEPT_FIELDS",
    "LIST_FIELDS",
    "MARKDOWN_FIELDS",
    "MAX_LIST_ITEMS",
    "MAX_REVIEW_BYTES",
    "FeedbackError",
    "RevisionFeedback",
    "load_revision_feedback",
    "render_revision_feedback_json",
    "render_revision_feedback_markdown",
]
