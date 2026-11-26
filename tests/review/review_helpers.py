"""Helpers of the review tests: valid reviews, scripted answers and PDF fixtures."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

from langgraph_agentic_scientist.review.models import ReviewerAnswer
from langgraph_agentic_scientist.testing import structured_call

PAPER_TEXT = "A study of the relationship between circle packing and search. " * 6


def review_dict(**overrides: Any) -> dict[str, Any]:
    """A valid review under upstream's keys; keyword overrides use the same keys with ``_``."""
    review: dict[str, Any] = {
        "Summary": "The paper studies packing.",
        "Strengths": ["Clear method."],
        "Weaknesses": ["Small evaluation."],
        "Originality": 3,
        "Quality": 3,
        "Clarity": 3,
        "Significance": 2,
        "Questions": ["Why this benchmark?"],
        "Limitations": ["Single task."],
        "Ethical Concerns": False,
        "Soundness": 3,
        "Presentation": 3,
        "Contribution": 2,
        "Overall": 5,
        "Confidence": 4,
        "Decision": "Reject",
    }
    for key, value in overrides.items():
        review[key.replace("_", " ")] = value
    return review


def answer(review: dict[str, Any] | None = None, *, thought: str = "notes") -> AIMessage:
    """The scripted structured answer of one reviewer call."""
    return structured_call(ReviewerAnswer, {"thought": thought, "review": review or review_dict()})


def last_text(messages: list[BaseMessage]) -> str:
    """The text of the last (human) message of a call."""
    content = messages[-1].content
    return content if isinstance(content, str) else str(content)


def make_pdf(text: str = PAPER_TEXT) -> bytes:
    """A one-page PDF with ``text`` (needs PyMuPDF)."""
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=11)
    data = bytes(document.tobytes())
    document.close()
    return data
