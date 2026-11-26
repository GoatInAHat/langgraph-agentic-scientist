"""AI review: the automated reviewer of upstream ``src/aireview.py`` as a LangGraph graph.

Use :func:`build_review_graph` to review a manuscript PDF with five reviewers and an area chair,
alone or mounted in a larger graph. The records are in :mod:`.models`, the PDF text extraction
in :mod:`.pdf`, the message assembly in :mod:`.texts` and the publication of the artifacts in
:mod:`.publish`.
"""

from .graph import build_review_graph, reviewer_retry_policy
from .models import (
    ReviewerAnswer,
    ReviewInput,
    ReviewOutput,
    ReviewState,
    aggregate_review,
    mean_scores,
    review_payload,
)
from .pdf import PdfTextError, extract_pdf_text

__all__ = [
    "PdfTextError",
    "ReviewInput",
    "ReviewOutput",
    "ReviewState",
    "ReviewerAnswer",
    "aggregate_review",
    "build_review_graph",
    "extract_pdf_text",
    "mean_scores",
    "review_payload",
    "reviewer_retry_policy",
]
