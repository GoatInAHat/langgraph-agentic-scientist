"""State and records of the AI review graph (upstream ``src/aireview.py``).

The reviewer and the area chair answer through structured output. Their answer is a
:class:`ReviewerAnswer`: a ``thought`` (upstream's ``THOUGHT:`` part, which upstream does not keep)
and the ``review`` itself, an :class:`~langgraph_agentic_scientist.core.results.AIReview`, whose
validators are upstream's ``_validate_review`` (every field required, the score ranges, booleans
refused as numbers, ``Decision`` in ``Accept``/``Reject``, ``Ethical Concerns`` a boolean, list
fields of strings) and whose field aliases are upstream's JSON keys.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, NotRequired, TypedDict

from pydantic import Field

from ..core.results import AIReview, ReviewResult
from ..core.state import (
    Event,
    Failure,
    RunConfigRecord,
    StateModel,
    append,
    merge_dicts,
)

SCORE_FIELDS: tuple[str, ...] = (
    "originality",
    "quality",
    "clarity",
    "significance",
    "soundness",
    "presentation",
    "contribution",
    "overall",
    "confidence",
)
"""The nine numeric scores of a review, in ``_SCORE_RANGES`` order of upstream."""

SCORE_KEYS: tuple[str, ...] = tuple(
    AIReview.model_fields[name].alias or name for name in SCORE_FIELDS
)
"""The same scores under upstream's JSON keys (``Originality`` ... ``Confidence``)."""

DEFAULT_ARTIFACT_STEM = "ai_review"
"""Upstream's default name of the review artifacts (``ai_review.json``, ``ai_review.log``)."""

DEFAULT_MANIFEST_KEY = "ai_review"
"""Upstream's default key of the review status in ``latest_manifest.json``."""


class InvalidReviewOutputError(ValueError):
    """A generated reviewer response did not satisfy the review contract (upstream's class)."""


class ReviewerAnswer(StateModel):
    """The structured answer of a reviewer or of the area chair.

    ``thought`` is the note-taking part of upstream's format (``THOUGHT:``). Upstream asks for it
    and ignores it, so it is optional here and is not kept in the result.
    """

    thought: str = Field(
        default="",
        description="First briefly discuss your intuitions and reasoning for the evaluation.",
    )
    review: AIReview


def review_payload(review: AIReview) -> dict[str, Any]:
    """Return a review as upstream's JSON object: aliases as keys, integral scores as integers.

    Upstream's reviewers answer with JSON numbers and the final scores are Python integers; the
    record type stores the scores as floats, so integral values are written as integers again.
    """
    data: dict[str, Any] = review.model_dump(by_alias=True, mode="json")
    for key in SCORE_KEYS:
        value = data[key]
        if isinstance(value, float) and value.is_integer():
            data[key] = int(value)
    return data


def mean_scores(reviews: Sequence[AIReview]) -> dict[str, int]:
    """Return the rounded mean of every score over ``reviews``, as upstream computes it.

    Upstream replaces each of the nine scores of the area chair's answer by
    ``int(round(np.mean(...)))``; Python's ``round`` rounds halves to the even integer, as NumPy's
    does, so a mean of 3.5 is 4 and a mean of 2.5 is 2.
    """
    if not reviews:
        raise ValueError("no reviews to average")
    return {
        key: round(statistics.fmean(float(getattr(review, field)) for review in reviews))
        for field, key in zip(SCORE_FIELDS, SCORE_KEYS, strict=True)
    }


def aggregate_review(meta: AIReview, reviews: Sequence[AIReview]) -> AIReview:
    """Return the area chair's review with its nine scores replaced by the reviewers' mean."""
    payload = review_payload(meta)
    payload.update(mean_scores(reviews))
    return AIReview.model_validate(payload)


class ReviewInput(TypedDict):
    """What the review graph is given.

    Attributes:
        pdf_path: Workspace path of the manuscript PDF.
        paper_dir: The directory the review artifacts are written to (default: the PDF's).
        artifact_stem: Name of the artifacts: ``<stem>.json`` and ``<stem>.log`` (default
            ``ai_review``; the revision uses ``ai_review.before_revision`` and
            ``ai_review.after_revision``).
        manifest_key: Key of the review status in ``latest_manifest.json`` of ``paper_dir``
            (default ``ai_review``); the file is updated only when it exists.
        paper_text: The manuscript text, when the caller has it; the PDF is then not read.
        run_config: The recorded run configuration, when a parent graph recorded it.
    """

    pdf_path: str
    paper_dir: NotRequired[str]
    artifact_stem: NotRequired[str]
    manifest_key: NotRequired[str]
    paper_text: NotRequired[str]
    run_config: NotRequired[RunConfigRecord | None]


class ReviewState(ReviewInput):
    """The state of the review graph."""

    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    reviews: Annotated[dict[str, AIReview], merge_dicts]
    """The reviewers' reviews by reviewer number (``"1"`` ... ``"5"``)."""
    meta: AIReview | None
    """The area chair's answer as returned."""
    aggregate: AIReview | None
    """The area chair's answer with the mean scores: what ``<stem>.json`` holds."""
    notes: Annotated[dict[str, list[str]], merge_dicts]
    """Log lines by source (``reviewer 1`` ... ``meta-review``), written to ``<stem>.log``."""
    failures: Annotated[dict[str, Failure], merge_dicts]
    """Failures by source (``extract``, ``reviewer 1`` ..., ``meta-review``)."""
    failure: Failure | None
    result: ReviewResult | None
    events: Annotated[list[Event], append]


class ReviewOutput(TypedDict):
    """What the review graph returns."""

    result: ReviewResult


class ReviewerTask(TypedDict):
    """The private input of one ``reviewer`` task (``Send``)."""

    index: int
    paper_text: str
    run_config: RunConfigRecord | None


def sorted_reviews(reviews: Mapping[str, AIReview]) -> list[AIReview]:
    """Return the reviews ordered by reviewer number."""
    return [reviews[key] for key in sorted(reviews, key=int)]


__all__ = [
    "DEFAULT_ARTIFACT_STEM",
    "DEFAULT_MANIFEST_KEY",
    "SCORE_FIELDS",
    "SCORE_KEYS",
    "InvalidReviewOutputError",
    "ReviewInput",
    "ReviewOutput",
    "ReviewState",
    "ReviewerAnswer",
    "ReviewerTask",
    "aggregate_review",
    "mean_scores",
    "review_payload",
    "sorted_reviews",
]
