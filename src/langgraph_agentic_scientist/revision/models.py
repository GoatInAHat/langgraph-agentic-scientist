"""State and records of the revision graph (upstream ``src/pipeline/revision.py``).

The revision graph revises a published manuscript against its AI review and publishes the result
only when the revised paper passes the retention and audit checks; the original is kept in
``revisions/revision_01/before/``. Its input is the published paper directory and, optionally, the
review and the results of the earlier phases that the paper graph needs.
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired, TypedDict

from pydantic import Field

from ..core.results import (
    DownstreamSelection,
    EditOptions,
    InvestigationResult,
    PaperData,
    PaperResult,
    PosthocResult,
    ReviewResult,
    RevisionResult,
    VariantsResult,
)
from ..core.state import (
    Event,
    Failure,
    RunConfigRecord,
    StateModel,
    TaskRef,
    append,
    merge_dicts,
)
from ..evaluation import FinalEvaluationResult
from ..workspace import SnapshotRef
from .feedback import RevisionFeedback

SLOT_PAPER = "revision.paper"
"""Slot key of the paper graph that edits the manuscript (default: the package's paper graph)."""

SLOT_REVIEW = "review"
"""Slot key of the review graph that assesses the revision (the key the pipeline's review uses)."""

SLOT_AUDIT_INPUT_HASHES = "revision.audit_input_hashes"
"""Optional slot function recomputing an evidence audit's input hashes for a directory."""

SLOT_SCOPE_AUDIT = "revision.scope_audit"
"""Optional slot function validating the method-scope audit of a directory."""


class RetentionReport(StateModel):
    """The outcome of the evidence checks of a revision.

    Attributes:
        ok: Whether the revision retained the original's evidence and has passing audits.
        skipped_checks: Checks that need a function the caller did not supply (see
            :mod:`~langgraph_agentic_scientist.revision.evidence`).
    """

    ok: bool = True
    skipped_checks: list[str] = Field(default_factory=list)


class PromotionRecord(StateModel):
    """What promotion did.

    Attributes:
        before_dir: The directory that holds the original paper.
        archived: Original-only audited inputs moved to ``before/ancillary/``.
        skipped_checks: Checks of the published paper that were not made.
    """

    before_dir: str
    archived: list[str] = Field(default_factory=list)
    skipped_checks: list[str] = Field(default_factory=list)


class RevisionInput(TypedDict):
    """What the revision graph is given.

    Attributes:
        paper_dir: The published paper directory (workspace path).
        review: The first review, when the caller has it; the graph reads the review file of the
            paper directory either way (``ai_review.before_revision.json``, or ``ai_review.json``
            of a paper reviewed without a revision in view).
        revision_root: Where revision attempts are kept (``attempt_01``, ...); default ``revision``
            next to ``paper_dir``.
        run_config: The recorded run configuration, when a parent graph recorded it.
        task, run_id, investigation, downstream, variants, final, posthoc, paper_data,
        artifact_paths: What the paper graph needs to ground the revision, as the paper graph is
            given them (the results of the earlier phases, or a prebuilt ``PaperData``).
        edit_options: Options of the paper edit; the revision fixes ``reference_mode`` to
            ``preserve``, ``figure_refresh_mode`` to ``auto`` and ``skip_postdraft_enrichment``
            to true, as upstream does, and keeps the rest.
    """

    paper_dir: str
    review: NotRequired[ReviewResult | None]
    revision_root: NotRequired[str]
    run_config: NotRequired[RunConfigRecord | None]
    task: NotRequired[TaskRef | None]
    run_id: NotRequired[str]
    investigation: NotRequired[InvestigationResult | None]
    downstream: NotRequired[DownstreamSelection | None]
    variants: NotRequired[VariantsResult | None]
    final: NotRequired[FinalEvaluationResult | None]
    posthoc: NotRequired[PosthocResult | None]
    paper_data: NotRequired[PaperData | None]
    artifact_paths: NotRequired[dict[str, str]]
    edit_options: NotRequired[EditOptions | None]


class RevisionPaperInput(TypedDict):
    """What the ``revision.paper`` slot receives: the paper graph's input in revision mode.

    ``mode`` is ``revision``; ``previous_paper_dir`` is the published paper; ``output_dir`` is
    where the revised paper is to be written (the attempt directory); ``external_review`` is the
    review that drives the revision, and ``edit_options`` the fixed options. The rest is passed
    through from :class:`RevisionInput`. The slot returns a
    :class:`~langgraph_agentic_scientist.core.results.PaperResult` (or a mapping holding one under
    ``result`` or ``paper``); its ``paper_dir`` is the directory of the revised paper.
    """

    mode: str
    previous_paper_dir: str
    output_dir: str
    manifest_dir: NotRequired[str]
    external_review: ReviewResult
    edit_options: EditOptions
    revision_feedback: NotRequired[RevisionFeedback]
    run_config: NotRequired[RunConfigRecord | None]
    task: NotRequired[TaskRef | None]
    run_id: NotRequired[str]
    investigation: NotRequired[InvestigationResult | None]
    downstream: NotRequired[DownstreamSelection | None]
    variants: NotRequired[VariantsResult | None]
    final: NotRequired[FinalEvaluationResult | None]
    posthoc: NotRequired[PosthocResult | None]
    paper_data: NotRequired[PaperData | None]
    artifact_paths: NotRequired[dict[str, str]]


class RevisionState(RevisionInput, total=False):
    """The state of the revision graph."""

    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    before_review_path: str
    feedback: RevisionFeedback | None
    replay: str
    """``none``; ``completed`` (a promoted revision that already has its second review); or
    ``rereview`` (a promoted revision whose second review is still to be made)."""
    attempt_dir: str
    reuse_attempt: bool
    paper: PaperResult | None
    revision_dir: str
    retention: RetentionReport | None
    snapshot: SnapshotRef | None
    promotion: PromotionRecord | None
    second_review: ReviewResult | None
    failure: Failure | None
    result: RevisionResult | None
    events: Annotated[list[Event], append]


class RevisionOutput(TypedDict):
    """What the revision graph returns."""

    result: RevisionResult


__all__ = [
    "SLOT_AUDIT_INPUT_HASHES",
    "SLOT_PAPER",
    "SLOT_REVIEW",
    "SLOT_SCOPE_AUDIT",
    "PromotionRecord",
    "RetentionReport",
    "RevisionInput",
    "RevisionOutput",
    "RevisionPaperInput",
    "RevisionState",
]
