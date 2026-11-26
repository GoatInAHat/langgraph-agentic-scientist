"""The reviewer's and the area chair's messages, assembled as upstream assembles them.

Every text is a prompt-registry entry under ``review/`` (derived from upstream's constants); the
functions here only concatenate and fill them in upstream's order:

* reviewer system message: ``review/reviewer_system``;
* reviewer user message (``_review_prompt``): the NeurIPS form, the answer format
  (``TEMPLATE_INSTRUCTIONS``), the few-shot paper and review, and the request that carries the
  manuscript text;
* area chair system message: ``review/area_chair_system`` with the reviewer count;
* area chair user message (``_meta_review_prompt``): the form, the answer format and one block per
  review;
* second attempt: the user message followed by ``review/retry_correction``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from ..core.results import AIReview
from ..prompts import PromptRegistry
from .models import review_payload

REVIEWER_SYSTEM = "review/reviewer_system"
AREA_CHAIR_SYSTEM = "review/area_chair_system"
FORM = "review/neurips_form"
TEMPLATE_INSTRUCTIONS = "review/template_instructions"
FEWSHOT_WRAPPER = "review/fewshot_wrapper"
FEWSHOT_PAPER = "review/fewshot_paper"
FEWSHOT_REVIEW = "review/fewshot_review"
REVIEW_REQUEST = "review/review_request"
META_REVIEW_REQUEST = "review/meta_review_request"
RETRY_CORRECTION = "review/retry_correction"


def reviewer_system_message(prompts: PromptRegistry) -> str:
    """Return the system message of a reviewer call."""
    return prompts.text(REVIEWER_SYSTEM)


def review_form(prompts: PromptRegistry) -> str:
    """Return the review form with the answer format (upstream's ``NEURIPS_FORM``)."""
    return prompts.text(FORM) + prompts.text(TEMPLATE_INSTRUCTIONS)


def fewshot_block(prompts: PromptRegistry) -> str:
    """Return the few-shot paper and review block (upstream's ``_fewshot_prompt()``)."""
    return prompts.render(
        FEWSHOT_WRAPPER,
        {"PAPER_TEXT": prompts.text(FEWSHOT_PAPER), "REVIEW": prompts.text(FEWSHOT_REVIEW)},
    )


def review_message(prompts: PromptRegistry, paper_text: str, *, fewshot: bool = True) -> str:
    """Return the reviewer's user message for ``paper_text`` (upstream's ``_review_prompt``).

    With ``fewshot=False`` the example paper and review are left out; upstream always includes
    them (about 60,000 characters), and the option exists for models with a small context.
    """
    return (
        review_form(prompts)
        + (fewshot_block(prompts) if fewshot else "")
        + prompts.render(REVIEW_REQUEST, {"PAPER_TEXT": paper_text})
    )


def area_chair_system_message(prompts: PromptRegistry, reviewer_count: int) -> str:
    """Return the system message of the area chair call."""
    return prompts.render(AREA_CHAIR_SYSTEM, {"REVIEWER_COUNT": str(reviewer_count)})


def area_chair_message(prompts: PromptRegistry, reviews: Sequence[AIReview]) -> str:
    """Return the area chair's user message (upstream's ``_meta_review_prompt``).

    Each review is given as the JSON object of upstream (``json.dumps`` of the review).
    """
    blocks = "".join(
        prompts.render(
            META_REVIEW_REQUEST,
            {
                "INDEX": str(index + 1),
                "COUNT": str(len(reviews)),
                "REVIEW_JSON": json.dumps(review_payload(review)),
            },
        )
        for index, review in enumerate(reviews)
    )
    return review_form(prompts) + blocks


def with_retry_correction(prompts: PromptRegistry, message: str) -> str:
    """Return the second attempt's message: the first followed by the retry correction."""
    return message + prompts.text(RETRY_CORRECTION)


__all__ = [
    "area_chair_message",
    "area_chair_system_message",
    "fewshot_block",
    "review_form",
    "review_message",
    "reviewer_system_message",
    "with_retry_correction",
]
