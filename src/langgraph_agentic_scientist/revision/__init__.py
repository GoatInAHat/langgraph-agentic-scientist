"""Revision: revise a published manuscript against its AI review (upstream ``revision.py``).

Use :func:`build_revision_graph` to edit a published paper from its review, check that the
revision keeps the original's evidence and passes the audits, publish it with the original kept,
and assess it with a second review. The records are in :mod:`.models`, the review as feedback in
:mod:`.feedback`, the evidence checks in :mod:`.evidence` and the publication in
:mod:`.promotion`.
"""

from .feedback import (
    FeedbackError,
    RevisionFeedback,
    load_revision_feedback,
    render_revision_feedback_json,
    render_revision_feedback_markdown,
)
from .graph import build_revision_graph, fixed_edit_options
from .models import (
    PromotionRecord,
    RetentionReport,
    RevisionInput,
    RevisionOutput,
    RevisionPaperInput,
    RevisionState,
)

__all__ = [
    "FeedbackError",
    "PromotionRecord",
    "RetentionReport",
    "RevisionFeedback",
    "RevisionInput",
    "RevisionOutput",
    "RevisionPaperInput",
    "RevisionState",
    "build_revision_graph",
    "fixed_edit_options",
    "load_revision_feedback",
    "render_revision_feedback_json",
    "render_revision_feedback_markdown",
]
