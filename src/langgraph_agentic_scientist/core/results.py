"""The interface records between phases: what each phase returns and the next phase reads.

These models are defined in the core so that every phase package can import them without
importing another phase. A phase package may store richer records in its own state; what crosses
a phase boundary is one of these.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from .state import (
    BestManifest,
    EvaluationRecord,
    Failure,
    IdeaBundle,
    ProgramRef,
    RubricStatus,
    StateModel,
    TaskRef,
)

# --- task drafting ------------------------------------------------------------------------


class DraftingResult(StateModel):
    """What task drafting returns.

    Attributes:
        status: ``published`` when an accepted scaffold was published into the task directory.
        task: The task reference after publication (with its contract).
        scaffold_fingerprint: The fingerprint of the published scaffold.
        receipts: Upstream's receipts (design receipt, alignment report, qualification record),
            by name.
        failure: Why drafting failed.
    """

    status: Literal["published", "failed", "skipped"]
    task: TaskRef | None = None
    scaffold_fingerprint: str | None = None
    receipts: dict[str, Any] = Field(default_factory=dict)
    failure: Failure | None = None


# --- investigation ------------------------------------------------------------------------


class IdeaRecord(StateModel):
    """An accepted idea of a cycle (one line of ``ideas_history.jsonl``)."""

    idea_id: str
    idea_dir: str
    cycle: int
    worker: int = 0
    bundle: IdeaBundle
    model_index: int | None = None
    accepted_at: str | None = None


class AttemptRecord(StateModel):
    """The record of one attempt (one line of ``attempts_history.jsonl``)."""

    attempt_id: str
    idea_id: str
    cycle: int
    worker: int = 0
    attempt_dir: str
    status: str
    best_program: ProgramRef | None = None
    combined_score: float | None = None
    correct: bool | None = None
    overall_pass: bool | None = None
    timed_out: bool = False
    evaluation_ids: list[str] = Field(default_factory=list)
    rubric_status: RubricStatus | None = None
    failure: Failure | None = None


class InvestigationResult(StateModel):
    """What investigation returns (``completed`` with a winner, ``completed_no_winner``, ...)."""

    status: Literal["completed", "completed_no_winner", "failed"]
    best: BestManifest | None = None
    baseline: EvaluationRecord | None = None
    ideas: list[IdeaRecord] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    evaluations: list[EvaluationRecord] = Field(default_factory=list)
    workers: int = 1
    results_dir: str | None = None
    failure: Failure | None = None


class DownstreamSelection(StateModel):
    """The no-winner fallback program frozen for the downstream phases (upstream 1.4)."""

    status: Literal["selected", "skipped"]
    program: ProgramRef | None = None
    source: Literal["candidate", "baseline"] | None = None
    evaluation_id: str | None = None
    combined_score: float | None = None
    reason: str = ""
    selection_path: str | None = None


# --- ablations ----------------------------------------------------------------------------


VariantKind = Literal["control", "reference", "baseline", "ablation", "custom"]


class VariantSpec(StateModel):
    """One materialised variant of the ablation study."""

    variant_id: str
    kind: VariantKind
    title: str = ""
    description: str = ""
    rationale: str = ""
    variant_dir: str
    task_dir: str | None = None
    program: ProgramRef | None = None
    task_name: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class VariantResult(StateModel):
    """The outcome of running one variant."""

    variant_id: str
    status: Literal["passed", "failed", "skipped", "timeout"]
    passed: bool = False
    best_program_index: int | None = None
    evaluations: list[EvaluationRecord] = Field(default_factory=list)
    rubric_status: RubricStatus | None = None
    task_name: str | None = None
    failure: Failure | None = None


class VariantsResult(StateModel):
    """What the ablation phase returns."""

    status: Literal["completed", "degraded", "failed", "skipped"]
    output_dir: str | None = None
    specs: list[VariantSpec] = Field(default_factory=list)
    variants: list[VariantResult] = Field(default_factory=list)
    cross_task: list[VariantResult] = Field(default_factory=list)
    aggregates: dict[str, str] = Field(default_factory=dict)
    readme_path: str | None = None
    failure: Failure | None = None


# --- post-hoc analysis --------------------------------------------------------------------


class AnalysisResult(StateModel):
    """The outcome of one post-hoc analysis."""

    analysis_id: str
    family: str = ""
    title: str = ""
    status: Literal["passed", "failed", "timeout", "skipped"]
    attempts: int = 0
    analysis_dir: str | None = None
    outputs: list[str] = Field(default_factory=list)
    manifest: dict[str, Any] = Field(default_factory=dict)
    failure: Failure | None = None


class PosthocResult(StateModel):
    """What the post-hoc phase returns."""

    status: Literal["completed", "degraded", "failed", "skipped"]
    output_dir: str | None = None
    bootstrap: dict[str, Any] = Field(default_factory=dict)
    analyses: list[AnalysisResult] = Field(default_factory=list)
    failure: Failure | None = None


# --- paper --------------------------------------------------------------------------------


class GroundingSource(StateModel):
    """One grounding source of ``paper_data.json`` (``result.*``, ``final.metric.*``, ...).

    Upstream sources carry ``type`` and ``value`` and, depending on the source, provenance
    fields such as ``program_dir`` or ``json_pointer``; those are kept as extra fields.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    type: str
    value: Any = None


class PaperData(StateModel):
    """The top level of upstream's ``paper_data.json``, with typed grounding sources.

    Sections that upstream adds only in some runs (``evidence_sources``, cross-task results,
    study context, the sealed final evaluation, post-hoc analyses) are kept as extra fields under
    upstream's names.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    generated_at: str
    paper_profile: Literal["default", "standalone"] = "default"
    presentation_contract: dict[str, Any] | None = None
    results_dir: str = ""
    best_correct: dict[str, Any] | None = None
    best_any: dict[str, Any] | None = None
    totals: dict[str, int] = Field(default_factory=dict)
    top_programs: list[dict[str, Any]] = Field(default_factory=list)
    improvement_trace: list[dict[str, Any]] = Field(default_factory=list)
    all_programs: list[dict[str, Any]] = Field(default_factory=list)
    idea_titles: Any = Field(default_factory=dict)
    grounding_sources: dict[str, GroundingSource] = Field(default_factory=dict)
    ablation_studies: list[dict[str, Any]] = Field(default_factory=list)


class EditOptions(StateModel):
    """Options of ``paper_edit.sh``; revision fixes them as upstream ``revision.py`` does.

    Attributes:
        edit_programs: ``--edit-program`` / ``--new-program``.
        old_programs: ``--old-program`` / ``--baseline-program``.
        figure_refresh_mode: ``auto`` is resolved as ``paper_edit.sh`` resolves it.
        reference_mode: ``auto`` is resolved as ``paper_edit.sh`` resolves it.
        skip_postdraft_enrichment: Skip the post-draft reference enrichment.
        contract_task_dir: ``--contract-task-dir``.
        investigation_results_dir: ``--investigation-results-dir``.
        main_file: The edited paper's main file, relative to ``previous_paper_dir``. ``None`` is
            upstream's seeding: ``paper.tex`` when present. A named main file must exist; it is
            seeded as the workspace's ``paper.tex`` together with the local files it inputs,
            recursively (``\\input``, ``\\include``, local packages and classes), and kept
            under its own name in ``previous_paper/``.
        venue_file: ``--venue-file``, in any mode: ``None`` stages the brief of the paper format,
            a workspace path stages that brief as ``VENUE.md`` (rendered as upstream renders
            the format's), and ``""`` stages none and removes a seeded one.
    """

    edit_programs: list[str] = Field(default_factory=list)
    old_programs: list[str] = Field(default_factory=list)
    figure_refresh_mode: Literal["auto", "preserve", "refresh"] = "auto"
    reference_mode: Literal["auto", "preserve", "refresh"] = "auto"
    skip_postdraft_enrichment: bool = False
    contract_task_dir: str | None = None
    investigation_results_dir: str | None = None
    main_file: str | None = None
    venue_file: str | None = None

    @field_validator("main_file")
    @classmethod
    def _relative_tex_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".tex"
            or str(path) in ("", ".")
        ):
            raise ValueError("main_file must be a relative .tex path inside the edited paper")
        return path.as_posix()


class ValidationReport(StateModel):
    """The outcome of the paper validation gate (``validation_report.json``)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    passed: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    figure_warnings: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    page_count: int | None = None
    pdf_path: str | None = None
    fingerprint: str | None = None


class PaperResult(StateModel):
    """What the paper phase returns."""

    status: Literal["complete", "incomplete", "failed", "skipped"]
    paper_dir: str | None = None
    final_dir: str | None = None
    pdf_path: str | None = None
    tex_path: str | None = None
    manifest_path: str | None = None
    incomplete_reason: str | None = None
    validation: ValidationReport | None = None
    failure: Failure | None = None


# --- review and revision ------------------------------------------------------------------

_SCORE_RANGES: dict[str, tuple[int, int]] = {
    "originality": (1, 4),
    "quality": (1, 4),
    "clarity": (1, 4),
    "significance": (1, 4),
    "soundness": (1, 4),
    "presentation": (1, 4),
    "contribution": (1, 4),
    "overall": (1, 10),
    "confidence": (1, 5),
}


class AIReview(StateModel):
    """One review in upstream's NeurIPS form, with upstream's validation (``aireview.py``).

    Field aliases are upstream's JSON keys (``"Ethical Concerns"`` keeps its space), so
    ``model_dump(by_alias=True)`` writes upstream's ``review.json``. Scores must be numbers
    (booleans are refused) within upstream's ranges; ``Decision`` is ``Accept`` or ``Reject``.
    """

    summary: str = Field(alias="Summary")
    strengths: list[str] = Field(alias="Strengths")
    weaknesses: list[str] = Field(alias="Weaknesses")
    originality: float = Field(alias="Originality")
    quality: float = Field(alias="Quality")
    clarity: float = Field(alias="Clarity")
    significance: float = Field(alias="Significance")
    questions: list[str] = Field(alias="Questions")
    limitations: list[str] = Field(alias="Limitations")
    ethical_concerns: bool = Field(alias="Ethical Concerns", strict=True)
    soundness: float = Field(alias="Soundness")
    presentation: float = Field(alias="Presentation")
    contribution: float = Field(alias="Contribution")
    overall: float = Field(alias="Overall")
    confidence: float = Field(alias="Confidence")
    decision: Literal["Accept", "Reject"] = Field(alias="Decision")

    @field_validator(*_SCORE_RANGES, mode="before")
    @classmethod
    def _score_in_range(cls, value: Any, info: Any) -> Any:
        minimum, maximum = _SCORE_RANGES[info.field_name]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"field {info.field_name!r} must be numeric")
        if not math.isfinite(float(value)) or not minimum <= value <= maximum:
            raise ValueError(f"field {info.field_name!r} is outside {minimum}..{maximum}")
        return value


class ReviewResult(StateModel):
    """What the AI review phase returns."""

    status: Literal["completed", "failed", "skipped"]
    reviews: list[AIReview] = Field(default_factory=list)
    meta_review: AIReview | None = None
    aggregate: AIReview | None = None
    review_path: str | None = None
    failure: Failure | None = None


class RevisionResult(StateModel):
    """What the revision phase returns."""

    status: Literal["promoted", "degraded", "failed", "skipped"]
    paper: PaperResult | None = None
    second_review: ReviewResult | None = None
    before_dir: str | None = None
    failure: Failure | None = None


__all__ = [
    "AIReview",
    "AnalysisResult",
    "AttemptRecord",
    "DownstreamSelection",
    "DraftingResult",
    "EditOptions",
    "GroundingSource",
    "IdeaRecord",
    "InvestigationResult",
    "PaperData",
    "PaperResult",
    "PosthocResult",
    "ReviewResult",
    "RevisionResult",
    "ValidationReport",
    "VariantKind",
    "VariantResult",
    "VariantSpec",
    "VariantsResult",
]
