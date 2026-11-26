"""The reviewer's texts are upstream's, assembled in upstream's order."""

from __future__ import annotations

import ast
import base64
import bz2
import json
from pathlib import Path

import pytest
from review_helpers import PAPER_TEXT, review_dict

from langgraph_agentic_scientist import PromptRegistry
from langgraph_agentic_scientist.core.results import AIReview
from langgraph_agentic_scientist.review import texts

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "third_party" / "ai-scientist-v3" / "src" / "aireview.py"


@pytest.fixture(scope="module")
def upstream() -> dict[str, ast.expr]:
    """The top-level assignments of upstream's ``aireview.py`` (never imported: it needs openai)."""
    tree = ast.parse(UPSTREAM.read_text(encoding="utf-8"))
    return {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }


@pytest.fixture(scope="module")
def prompts() -> PromptRegistry:
    return PromptRegistry.default()


def _constant(node: ast.expr) -> str:
    return str(ast.literal_eval(node))


def test_the_system_messages_and_the_retry_correction_are_upstreams(
    upstream: dict[str, ast.expr], prompts: PromptRegistry
) -> None:
    assert prompts.text("review/reviewer_system") == _constant(upstream["REVIEWER_SYSTEM_PROMPT"])
    assert prompts.text("review/retry_correction") == _constant(upstream["REVIEW_RETRY_CORRECTION"])
    chair = _constant(upstream["META_REVIEWER_SYSTEM_PROMPT"]).format(reviewer_count=5)
    assert texts.area_chair_system_message(prompts, 5) == chair


def test_the_review_form_is_upstreams_and_only_the_answer_format_differs(
    upstream: dict[str, ast.expr], prompts: PromptRegistry
) -> None:
    form_node = upstream["NEURIPS_FORM"]
    assert isinstance(form_node, ast.BinOp)
    assert prompts.text("review/neurips_form") == _constant(form_node.left)
    instructions = _constant(upstream["TEMPLATE_INSTRUCTIONS"])
    ours = prompts.text("review/template_instructions")
    assert ours != instructions
    # the field list, the scales and the closing sentences are untouched
    assert (
        ours.split("the following fields in the order:\n")[1]
        == instructions.split("the following fields in the order:\n")[1]
    )
    assert "This JSON will be automatically parsed, so ensure the format is precise." in ours
    assert "THOUGHT:" not in ours
    assert "```" not in ours
    assert 'two fields, "thought" and "review"' in ours


def test_the_few_shot_files_are_the_decoded_upstream_constants(
    upstream: dict[str, ast.expr], prompts: PromptRegistry
) -> None:
    def decode(name: str) -> str:
        return bz2.decompress(base64.b85decode(_constant(upstream[name]))).decode("utf-8")

    paper = decode("_FEWSHOT_PAPER_B85")
    review = json.loads(decode("_FEWSHOT_REVIEW_B85"))["review"]
    assert prompts.text("review/fewshot_paper") == paper
    assert prompts.text("review/fewshot_review") == review
    assert len(paper) > 50_000  # the single in-context example is long


def test_the_reviewer_message_is_the_form_the_example_and_the_paper(
    upstream: dict[str, ast.expr], prompts: PromptRegistry
) -> None:
    form = prompts.text("review/neurips_form") + prompts.text("review/template_instructions")
    fewshot = (
        "\nBelow are some sample reviews, copied from previous machine learning conferences.\n"
        "Note that while each review is formatted differently according to each reviewer's style, "
        "the reviews are well-structured and therefore easy to navigate.\n\nPaper:\n\n```\n"
        + prompts.text("review/fewshot_paper")
        + "\n```\n\nReview:\n\n```\n"
        + prompts.text("review/fewshot_review")
        + "\n```\n"
    )
    request = f"\nHere is the paper you are asked to review:\n```\n{PAPER_TEXT}\n```"
    assert texts.review_message(prompts, PAPER_TEXT) == form + fewshot + request
    assert texts.review_message(prompts, PAPER_TEXT, fewshot=False) == form + request


def test_a_paper_that_contains_placeholders_is_not_expanded(prompts: PromptRegistry) -> None:
    paper = "The method {{REVIEW}} and {{PAPER_TEXT}} stay as written. " * 4
    message = texts.review_message(prompts, paper, fewshot=False)
    assert message.endswith(f"```\n{paper}\n```")


def test_the_area_chairs_message_has_one_block_per_review(prompts: PromptRegistry) -> None:
    raw = [review_dict(Overall=index + 1) for index in range(3)]
    reviews = [AIReview.model_validate(item) for item in raw]
    message = texts.area_chair_message(prompts, reviews)
    form = prompts.text("review/neurips_form") + prompts.text("review/template_instructions")
    assert message.startswith(form)
    blocks = message[len(form) :]
    expected = ""
    for index, item in enumerate(raw):
        expected += f"\nReview {index + 1}/3:\n```\n{json.dumps(item)}\n```\n"
    assert blocks == expected


def test_the_second_attempt_appends_the_correction_to_the_same_message(
    prompts: PromptRegistry,
) -> None:
    message = texts.review_message(prompts, PAPER_TEXT, fewshot=False)
    again = texts.with_retry_correction(prompts, message)
    assert again.startswith(message)
    assert again[len(message) :] == (
        "\n\nRetry correction: The previous response did not satisfy the required review JSON "
        "contract. Return one complete strict JSON review with every required field and the "
        "specified field types and score ranges."
    )


def test_every_review_text_is_in_the_registry_and_can_be_overridden(
    prompts: PromptRegistry,
) -> None:
    keys = {key for key in prompts.keys() if key.startswith("review/")}  # noqa: SIM118
    assert keys == {
        "review/area_chair_system",
        "review/fewshot_paper",
        "review/fewshot_review",
        "review/fewshot_wrapper",
        "review/meta_review_request",
        "review/neurips_form",
        "review/retry_correction",
        "review/review_request",
        "review/reviewer_system",
        "review/template_instructions",
    }
    changed = prompts.with_overrides({"review/reviewer_system": "You review."})
    assert texts.reviewer_system_message(changed) == "You review."
    assert changed.fingerprint() != prompts.fingerprint()
