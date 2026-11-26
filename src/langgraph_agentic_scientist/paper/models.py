"""Records of the paper writing loop: structured answers, loop state and the phase's interface.

The structured answers are the outputs of upstream's structured roles (strategist, post-hoc
ranker, draft-gap analyst, post-draft reference planner, evidence auditor, in-loop reviewer).
Upstream's agents wrote them as files that the host re-read; here each role returns one of these
models through LangChain structured output, and the host writes the file under upstream's name.
Where upstream's host validation is stricter than a schema can say (``validate_strategy_with_
feedback``, ``validate_posthoc_ranking``), the models stay permissive and the ported validator
produces upstream's error messages, which drive the repair rounds as upstream's do.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import ConfigDict, Field, StringConstraints, model_validator

from ..core.results import (
    DownstreamSelection,
    EditOptions,
    InvestigationResult,
    PaperData,
    PaperResult,
    PosthocResult,
    ReviewResult,
    ValidationReport,
    VariantsResult,
)
from ..core.state import Event, Failure, RunConfigRecord, StateModel, TaskRef, merge_dicts
from ..evaluation.protocol import FinalEvaluationResult
from ..workspace.protocol import SnapshotRef

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

WriterFamily = Literal[
    "paper",
    "paper_detailed_grouped",
    "paper_standalone",
    "paper_edit",
    "paper_edit_standalone",
    "paper_revision",
]
"""The writer prompt families (upstream ``PROMPT_FAMILY_CONFIGS`` plus the edit and revision
prompts of ``paper_edit.sh``)."""

PaperMode = Literal["fresh", "edit", "revision"]
"""``fresh`` is ``paper.sh``; ``edit`` and ``revision`` are ``paper_edit.sh`` (revision is its
``--external-review`` mode with the options of upstream ``revision.py``)."""


class FamilyConfig(StateModel):
    """One writer prompt family: its prompt key and its writer skill."""

    family: str
    prompt_key: str
    writer_skill: str


FAMILY_CONFIGS: dict[str, FamilyConfig] = {
    "paper": FamilyConfig(family="paper", prompt_key="paper/paper", writer_skill="write-paper"),
    "paper_detailed_grouped": FamilyConfig(
        family="paper_detailed_grouped",
        prompt_key="paper/paper_detailed_grouped",
        writer_skill="write-paper",
    ),
    "paper_standalone": FamilyConfig(
        family="paper_standalone",
        prompt_key="paper/paper_standalone",
        writer_skill="write-paper-standalone",
    ),
}
"""Upstream ``PROMPT_FAMILY_CONFIGS`` (``paper_prompt_family.py``): the families a strategy may
select."""

EDIT_PROMPTS: dict[str, str] = {
    "default": "paper/paper_edit",
    "standalone": "paper/paper_edit_standalone",
}
"""The edit-mode writer prompts by paper profile (``paper_edit.sh`` defaults)."""

REVISION_PROMPT_KEY = "paper/paper_revision"
"""The revision writer prompt (``paper_edit.sh --external-review``)."""


# --- structured answers -------------------------------------------------------------------


class PaperStrategy(StateModel):
    """``paper_strategy.json`` (the strategy object of ``paper_strategy.md``).

    The fields are upstream's; their types are left permissive so that the ported
    ``validate_strategy_with_feedback`` reports upstream's errors and drives the repair rounds.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    status: str = Field("", description="`ready` or `fail`.")
    selected_prompt_family: str = Field(
        "", description="`paper`, `paper_detailed_grouped` or `paper_standalone`."
    )
    recommended_title: str = ""
    central_question: str = ""
    central_thesis: str = ""
    narrative_rationale: str = ""
    prior_paper_story_status: str | None = None
    edit_framing_decision: str | None = None
    staged_input_impact: list[Any] | None = None
    method_groups: list[Any] = Field(default_factory=list)
    results_focus: list[Any] = Field(default_factory=list)
    figure_plan: list[Any] = Field(default_factory=list)
    figure_specs: list[Any] = Field(default_factory=list)
    supplementary_plan: dict[str, Any] = Field(default_factory=dict)
    forbidden_headings: list[Any] = Field(default_factory=list)
    forbidden_phrases: list[Any] = Field(default_factory=list)
    notes_for_writer: list[Any] = Field(default_factory=list)
    failure_reason: str = ""


class StrategyAnswer(StateModel):
    """The strategist's structured answer: the memo and the strategy object."""

    memo: str = Field(description="The readable markdown memo (`paper_strategy.md`).")
    strategy: PaperStrategy = Field(description="The strategy object (`paper_strategy.json`).")


class RankedAnalysis(StateModel):
    """One entry of ``posthoc_ranking.json``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    analysis_id: str = ""
    rank: int | None = None
    relevance_to_thesis: str = Field("", description="`high`, `medium` or `low`.")
    relevance_justification: str = ""
    recommended_usage: str = Field(
        "", description="`main_results`, `supporting`, `appendix` or `skip`."
    )
    key_finding_for_paper: str = ""


class PosthocRanking(StateModel):
    """``posthoc_ranking.json``, the post-hoc ranker's structured answer."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    ranked_analyses: list[RankedAnalysis] = Field(default_factory=list)
    ranking_rationale: str = ""


class RewriteTarget(StateModel):
    """One rewrite target of ``draft_reference_gaps.json``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    target_id: str = ""
    section: str = ""
    priority: str = ""
    citation_intent: str = ""
    reason: str = ""
    anchor_text: str = ""
    search_topics: list[str] = Field(default_factory=list)


class DraftReferenceGaps(StateModel):
    """``draft_reference_gaps.json``."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    status: str = Field("", description="`ready`, `no_gaps` or `fail`.")
    summary: str = ""
    missing_literatures: list[str] = Field(default_factory=list)
    rewrite_targets: list[RewriteTarget] = Field(default_factory=list)
    failure_reason: str = ""


class DraftGapsAnswer(StateModel):
    """The draft-gap analyst's structured answer: memo and gap object."""

    memo: str = Field(description="The readable markdown memo (`draft_reference_gaps.md`).")
    gaps: DraftReferenceGaps = Field(description="The gap object (`draft_reference_gaps.json`).")


class AuditFinding(StateModel):
    """One evidence/claim mismatch finding (upstream ``AuditFinding``)."""

    id: NonEmptyText
    section: NonEmptyText
    claim: NonEmptyText
    issue: NonEmptyText
    evidence: NonEmptyText
    fix: NonEmptyText


class AuditPayload(StateModel):
    """The evidence auditor's structured answer (upstream ``AuditPayload``).

    The validator keeps the rules of upstream's line contract: ``pass`` has no findings, ``fail``
    has at least one, at most 20 findings.
    """

    verdict: Literal["pass", "fail"]
    summary: NonEmptyText
    findings: list[AuditFinding] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _verdict_matches_findings(self) -> AuditPayload:
        if self.verdict == "pass" and self.findings:
            raise ValueError("verdict 'pass' cannot include findings")
        if self.verdict == "fail" and not self.findings:
            raise ValueError("verdict 'fail' must include at least one finding")
        return self


class ReviewIssue(StateModel):
    """One actionable review issue (upstream ``ReviewIssue``)."""

    id: NonEmptyText
    section: NonEmptyText
    issue: NonEmptyText
    evidence: list[NonEmptyText] = Field(default_factory=list)
    requested_change: NonEmptyText


class ReviewScore(StateModel):
    """Reviewer score (upstream ``ReviewScore``)."""

    value: int = Field(ge=1, le=10)
    label: NonEmptyText


class ReviewPayload(StateModel):
    """The in-loop reviewer's structured answer (upstream ``ReviewPayload`` with the checks of
    ``validate_review_payload``)."""

    status: Literal["pass", "revise"]
    summary: NonEmptyText
    strengths: list[NonEmptyText] = Field(default_factory=list)
    major_comments: list[ReviewIssue] = Field(default_factory=list)
    minor_comments: list[ReviewIssue] = Field(default_factory=list)
    questions: list[NonEmptyText] = Field(default_factory=list)
    score: ReviewScore

    @model_validator(mode="after")
    def _upstream_rules(self) -> ReviewPayload:
        if self.status == "pass" and self.major_comments:
            raise ValueError("review status 'pass' cannot include major_comments")
        if self.status == "revise" and not self.major_comments:
            raise ValueError("review status 'revise' must include at least one major_comment")
        for issue in self.major_comments:
            if not issue.evidence:
                raise ValueError(f"major comment {issue.id!r} must include evidence")
        seen: set[str] = set()
        for issue in [*self.major_comments, *self.minor_comments]:
            if issue.id in seen:
                raise ValueError(f"duplicate review issue id: {issue.id}")
            seen.add(issue.id)
        return self


# --- loop records -------------------------------------------------------------------------

AuditOutcome = Literal["pass", "findings", "operational_failure", "contract_exhausted"]
"""Upstream's audit return codes: 0 pass, 2 findings, 1 operational failure, 3 contract
exhausted."""


class AuditResult(StateModel):
    """The outcome of one evidence-audit gate (all its contract attempts)."""

    outcome: AuditOutcome
    payload: AuditPayload | None = None
    attempts: int = 0
    detail: str = ""


class InLoopReview(StateModel):
    """The outcome of one in-loop review gate (all its contract attempts)."""

    outcome: Literal["completed", "stage_failed", "contract_exhausted"]
    review: ReviewPayload | None = None
    attempts: int = 0
    detail: str = ""

    @property
    def major_comment_count(self) -> int:
        """Number of major comments of the accepted review."""
        return len(self.review.major_comments) if self.review else 0


class FigureQualityState(StateModel):
    """Upstream's ``figure_quality_repair_*`` variables, kept in state across rounds."""

    attempted: bool = False
    pending: bool = False
    warning_reported: bool = False
    snapshot: SnapshotRef | None = None


class ConceptFigureState(StateModel):
    """The concept-figure transaction of ``lib/concept_figure.sh``, kept in state.

    Attributes:
        available: Whether a renderer is configured and the deadline budget allows rendering.
        unavailable_reason: Why rendering is unavailable.
        mode: The Autoillustrate mode chosen for the remaining budget.
        preserve_fingerprint: Fingerprint of the concept state when the loop began (preserve
            mode keeps the figure when it is unchanged).
        attempted: Whether the current request has been rendered (or failed).
        request_fingerprint: Fingerprint of the rendered request.
        failure_active: Whether the last attempt failed.
        failure_reason: Why it failed.
        rendered_path: Workspace path of the rendered figure.
        rendered_sha256: Digest of the rendered figure, to re-assert it after a repair.
        repair_authorized: Whether an evidence-audit finding authorised repairing the figure.
    """

    available: bool = False
    unavailable_reason: str = ""
    mode: Literal["full", "medium", "light"] = "full"
    preserve_fingerprint: str | None = None
    attempted: bool = False
    request_fingerprint: str | None = None
    failure_active: bool = False
    failure_reason: str = ""
    rendered_path: str | None = None
    rendered_sha256: str | None = None
    repair_authorized: bool = False


Stage = Literal["draft", "manuscript_repair", "figure_quality_repair"]


# --- phase interface ------------------------------------------------------------------------


class PaperInput(TypedDict):
    """What the paper phase is given.

    The phase runs with the results of earlier phases (``investigation`` ... ``posthoc``, as the
    pipeline mounts it), or with a prebuilt ``paper_data`` and the workspace paths of the
    artefacts it grounds (``artifact_paths``: ``results``, ``global_memory``, ``aggregated``,
    ``extra_tasks``, ``results_readme``, ``contributions``), or in ``edit`` mode with
    ``previous_paper_dir``. ``paper_dir`` (or ``output_dir``, upstream's ``--output-dir``)
    overrides the default output directory ``<run root>/paper/final_<run_id>``;
    ``manifest_dir`` (upstream's ``--manifest-dir``) is where ``latest_manifest.json`` is
    written, by default the output directory's parent. The pipeline writes into ``paper`` with
    the manifest beside the paper, and the revision into its attempt directory, as upstream's
    pipeline and revision commands do.
    """

    task: TaskRef
    run_id: str
    mode: NotRequired[PaperMode]
    run_config: NotRequired[RunConfigRecord | None]
    investigation: NotRequired[InvestigationResult | None]
    downstream: NotRequired[DownstreamSelection | None]
    variants: NotRequired[VariantsResult | None]
    final: NotRequired[FinalEvaluationResult | None]
    posthoc: NotRequired[PosthocResult | None]
    paper_data: NotRequired[PaperData | None]
    artifact_paths: NotRequired[dict[str, str]]
    previous_paper_dir: NotRequired[str | None]
    external_review: NotRequired[ReviewResult | dict[str, Any] | None]
    edit_options: NotRequired[EditOptions | None]
    paper_dir: NotRequired[str]
    output_dir: NotRequired[str]
    manifest_dir: NotRequired[str]


class PaperState(TypedDict, total=False):
    """The state of the paper phase (inputs, preparation, and the loop's counters and flags).

    Every counter and flag of upstream's loop lives here, so a resume re-enters the loop at the
    row that was running.
    """

    # inputs
    task: TaskRef
    run_id: str
    mode: PaperMode
    run_config: RunConfigRecord | None
    investigation: InvestigationResult | None
    downstream: DownstreamSelection | None
    variants: VariantsResult | None
    final: FinalEvaluationResult | None
    posthoc: PosthocResult | None
    paper_data: PaperData | None
    artifact_paths: dict[str, str]
    previous_paper_dir: str | None
    external_review: ReviewResult | dict[str, Any] | None
    edit_options: EditOptions | None
    paper_dir: str
    output_dir: str
    manifest_dir: str
    # bookkeeping
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    events: Annotated[list[Event], operator.add]
    # preparation
    manifest_root: str
    profile: Literal["default", "standalone"]
    two_stage: bool
    paper_format: str
    reference_mode: Literal["refresh", "preserve"]
    figure_refresh_mode: Literal["auto", "preserve", "refresh"]
    skip_postdraft_enrichment: bool
    writer_family: WriterFamily
    writer_prompt_key: str
    writer_skill: str
    selected_skills: list[str]
    reference_baseline_dir: str | None
    references_done: bool
    strategy: PaperStrategy | None
    strategy_round: int
    strategy_done: bool
    ranking: PosthocRanking | None
    ranking_round: int
    posthoc_analyses: list[dict[str, Any]]
    concept: ConceptFigureState
    # main loop
    repair_round: int
    review_round: int
    stage: Stage
    stage_rc: Literal["ok", "failed"] | None
    stage_failure: Failure | None
    reverted: bool
    figure_quality: FigureQualityState
    previous_failure_fingerprint: str | None
    failure_repeated: bool
    validation: ValidationReport | None
    validation_passed: bool
    audit: AuditResult | None
    audit_round: int
    audit_contract_attempt: int
    in_loop_review: InLoopReview | None
    review_contract_attempt: int
    revisions: Annotated[list[str], operator.add]
    references: Any
    strategy_ok: bool
    ranking_ok: bool
    validation_fingerprint: str | None
    validation_soft_warnings: list[str]
    postdraft_status: Literal["done", "no_gaps", "failed"] | None
    postdraft_detail: str
    status: Literal["running", "complete", "incomplete", "failed"]
    incomplete_reasons: list[str]
    failure: Failure | None
    next: str
    paper: PaperResult | None


class PaperOutput(TypedDict):
    """What the paper phase returns: ``paper`` (a :class:`PaperResult`) and the final
    ``paper_data``."""

    paper: PaperResult | None
    paper_data: PaperData | None


__all__ = [
    "EDIT_PROMPTS",
    "FAMILY_CONFIGS",
    "REVISION_PROMPT_KEY",
    "AuditFinding",
    "AuditOutcome",
    "AuditPayload",
    "AuditResult",
    "ConceptFigureState",
    "DraftGapsAnswer",
    "DraftReferenceGaps",
    "EditOptions",
    "FamilyConfig",
    "FigureQualityState",
    "InLoopReview",
    "PaperInput",
    "PaperMode",
    "PaperOutput",
    "PaperState",
    "PaperStrategy",
    "PosthocRanking",
    "RankedAnalysis",
    "ReviewIssue",
    "ReviewPayload",
    "ReviewScore",
    "RewriteTarget",
    "Stage",
    "StrategyAnswer",
    "WriterFamily",
]
