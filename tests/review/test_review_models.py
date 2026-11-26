"""The structured review models enforce upstream's contract (``aireview._validate_review``)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError
from review_helpers import review_dict

from langgraph_agentic_scientist.core.results import AIReview
from langgraph_agentic_scientist.review.models import (
    SCORE_KEYS,
    ReviewerAnswer,
    aggregate_review,
    mean_scores,
    review_payload,
)


def _review(**overrides: Any) -> AIReview:
    return AIReview.model_validate(review_dict(**overrides))


def test_a_valid_review_round_trips_under_upstreams_keys() -> None:
    review = _review()
    payload = review_payload(review)
    assert list(payload) == list(review_dict())
    assert "Ethical Concerns" in payload
    assert payload["Ethical Concerns"] is False
    # integral scores are integers again, as in upstream's JSON
    assert payload["Overall"] == 5
    assert isinstance(payload["Overall"], int)
    assert json.dumps(payload) == json.dumps(review_dict())


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("Originality", 0),
        ("Originality", 5),
        ("Quality", 4.5),
        ("Clarity", -1),
        ("Significance", 5),
        ("Soundness", 0),
        ("Presentation", 5),
        ("Contribution", 5),
        ("Overall", 0),
        ("Overall", 11),
        ("Confidence", 0),
        ("Confidence", 6),
    ],
)
def test_scores_outside_upstreams_ranges_are_refused(key: str, value: float) -> None:
    with pytest.raises(ValidationError):
        AIReview.model_validate(review_dict(**{key: value}))


@pytest.mark.parametrize("key", list(SCORE_KEYS))
def test_a_boolean_is_not_a_score(key: str) -> None:
    with pytest.raises(ValidationError):
        AIReview.model_validate(review_dict(**{key: True}))


def test_the_score_bounds_are_inclusive() -> None:
    low = {key: 1 for key in SCORE_KEYS}
    high = {"Originality": 4, "Overall": 10, "Confidence": 5}
    assert AIReview.model_validate(review_dict(**low)).overall == 1
    assert AIReview.model_validate(review_dict(**high)).overall == 10


@pytest.mark.parametrize(
    "decision", ["Weak Accept", "Borderline Reject", "Strong Reject", "accept", ""]
)
def test_the_decision_is_accept_or_reject(decision: str) -> None:
    with pytest.raises(ValidationError):
        AIReview.model_validate(review_dict(Decision=decision))
    assert _review(Decision="Accept").decision == "Accept"


@pytest.mark.parametrize("value", ["no", 0, 1, None])
def test_ethical_concerns_must_be_a_boolean(value: Any) -> None:
    with pytest.raises(ValidationError):
        AIReview.model_validate(review_dict(**{"Ethical Concerns": value}))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda review: review.pop("Summary"),
        lambda review: review.pop("Decision"),
        lambda review: review.pop("Ethical Concerns"),
        lambda review: review.__setitem__("Summary", ["not text"]),
        lambda review: review.__setitem__("Strengths", "text"),
        lambda review: review.__setitem__("Weaknesses", ["a", 3]),
        lambda review: review.__setitem__("Questions", None),
        lambda review: review.__setitem__("Limitations", [["nested"]]),
    ],
)
def test_required_fields_and_text_types_are_enforced(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    review = review_dict()
    mutate(review)
    with pytest.raises(ValidationError):
        AIReview.model_validate(review)


def test_unknown_keys_are_refused() -> None:
    with pytest.raises(ValidationError):
        AIReview.model_validate({**review_dict(), "Extra": 1})


def test_the_structured_answer_nests_the_review_and_keeps_the_thought_optional() -> None:
    answer = ReviewerAnswer.model_validate({"thought": "t", "review": review_dict()})
    assert answer.thought == "t"
    assert ReviewerAnswer.model_validate({"review": review_dict()}).thought == ""
    schema = ReviewerAnswer.model_json_schema()
    assert list(schema["properties"]) == ["thought", "review"]


def test_the_mean_score_rounds_halves_to_the_even_integer() -> None:
    def overall(*values: int) -> list[AIReview]:
        return [_review(Overall=value) for value in values]

    assert mean_scores(overall(2, 3))["Overall"] == 2  # 2.5
    assert mean_scores(overall(3, 4))["Overall"] == 4  # 3.5
    assert mean_scores(overall(4, 5))["Overall"] == 4  # 4.5
    assert mean_scores(overall(5, 6))["Overall"] == 6  # 5.5
    assert mean_scores(overall(4, 5, 5, 4, 5))["Overall"] == 5  # 4.6
    assert mean_scores(overall(4, 4, 5, 4, 4))["Overall"] == 4  # 4.2


def test_the_nine_scores_are_replaced_and_everything_else_is_the_area_chairs() -> None:
    reviews = [_review(Originality=2, Quality=4, Overall=3, Confidence=2) for _ in range(2)]
    reviews.append(_review(Originality=3, Quality=4, Overall=4, Confidence=3))
    meta = _review(
        Summary="meta",
        Originality=1,
        Quality=1,
        Overall=10,
        Confidence=5,
        Decision="Accept",
        Strengths=["from the chair"],
    )
    aggregate = aggregate_review(meta, reviews)
    payload = review_payload(aggregate)
    assert payload["Originality"] == 2  # 2.33
    assert payload["Quality"] == 4
    assert payload["Overall"] == 3  # 3.33
    assert payload["Confidence"] == 2  # 2.33
    assert all(isinstance(payload[key], int) for key in SCORE_KEYS)
    assert payload["Summary"] == "meta"
    assert payload["Decision"] == "Accept"
    assert payload["Strengths"] == ["from the chair"]
    assert list(payload) == list(review_dict())


def test_no_reviews_cannot_be_averaged() -> None:
    with pytest.raises(ValueError, match="no reviews"):
        mean_scores([])
