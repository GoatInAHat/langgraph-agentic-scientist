"""Records of the post-hoc phase (upstream ``posthoc_pipeline/models.py`` and ``config.py``).

Two kinds of model live here. The *answers* of the two structured roles (:class:`PosthocBundle`
of the analysis generator, :class:`SelectionAnswer` of the analysis selector) carry upstream's
field names and are lenient about extra keys, as upstream's parsers were. The *records* the phase
keeps in state (:class:`AnalysisCandidate`, :class:`AnalysisSpec`, :class:`AttemptInspection`,
:class:`AnalysisRunResult`, :class:`BootstrapReport`) mirror upstream's dataclasses and JSON files
(``posthoc_selection.json``, ``analysis_spec.json``, ``attempt_result.json``,
``analysis_result.json``, ``bootstrap_report.json``); the paths they hold are workspace paths.

The normalisation helpers are upstream's: list-like fields never explode a string into characters,
prose output descriptions become deterministic safe paths, and implementation plans emitted as
strings or character arrays become lists of steps.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import BeforeValidator, ConfigDict, Field

from langgraph_agentic_scientist.core.results import (
    AnalysisResult,
    DownstreamSelection,
    InvestigationResult,
    PosthocResult,
    VariantsResult,
)
from langgraph_agentic_scientist.core.state import (
    BestManifest,
    Event,
    Failure,
    ProgramRef,
    RunConfigRecord,
    StateModel,
    TaskRef,
    append,
    merge_dicts,
)
from langgraph_agentic_scientist.workspace.protocol import SnapshotRef

_ANALYSIS_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,127}$")
_OUTPUT_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._/-]+$")
_OUTPUT_SUFFIXES = frozenset({".csv", ".json", ".md", ".pdf", ".png", ".svg", ".tex", ".txt"})
_IMAGE_OUTPUT_TOKENS = (
    "boxplot",
    "chart",
    "curve",
    "dendrogram",
    "figure",
    "heatmap",
    "histogram",
    "plot",
    "scatter",
    "violin",
)


# --- normalisation (upstream models.py) --------------------------------------------------------


def is_valid_analysis_id(value: str) -> bool:
    """Return whether an analysis ID is safe as one directory component."""
    return bool(_ANALYSIS_ID_PATTERN.fullmatch(value))


def normalize_string_list(value: Any) -> list[str]:
    """Normalize list-like fields without exploding strings into characters."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else []


def normalize_expected_outputs(value: Any) -> list[str]:
    """Convert prose output descriptions into deterministic safe relative file paths."""
    normalized: list[str] = []
    for index, item in enumerate(normalize_string_list(value), start=1):
        path = PurePosixPath(item)
        if (
            not path.is_absolute()
            and ".." not in path.parts
            and path.as_posix() == item
            and len(item) <= 240
            and all(len(part) <= 120 for part in path.parts)
            and _OUTPUT_PATH_PATTERN.fullmatch(item)
            and path.suffix.lower() in _OUTPUT_SUFFIXES
        ):
            normalized.append(item)
            continue
        lower_item = item.lower()
        if any(token in lower_item for token in _IMAGE_OUTPUT_TOKENS):
            suffix = ".png"
        elif "table" in lower_item or "result" in lower_item:
            suffix = ".csv"
        else:
            suffix = ".md"
        slug = re.sub(r"[^a-z0-9]+", "_", lower_item).strip("_")[:96]
        normalized.append(f"outputs/{index:02d}_{slug or 'artifact'}{suffix}")
    return normalized


def normalize_implementation_plan(value: Any) -> list[str]:
    """Normalize implementation plans emitted as strings, lists, or character arrays."""
    if value is None:
        return []
    if isinstance(value, str):
        return _split_implementation_plan_text(value)
    if isinstance(value, list):
        raw_items = [str(item) for item in value if str(item)]
        if _looks_like_character_plan(raw_items):
            return _split_implementation_plan_text("".join(raw_items))
        return [item.strip() for item in raw_items if item.strip()]
    return _split_implementation_plan_text(str(value))


def _looks_like_character_plan(items: list[str]) -> bool:
    if len(items) < 12:
        return False
    single_char_items = sum(1 for item in items if len(item) == 1)
    return single_char_items / len(items) >= 0.85


def _split_implementation_plan_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    line_steps: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[.)]\s+", "", line)
        if line:
            line_steps.append(line)
    if len(line_steps) > 1:
        return line_steps

    compact = re.sub(r"\s+", " ", text)
    numbered_steps = [
        re.sub(r"^\d+[.)]\s+", "", part).strip() for part in re.split(r"\s+(?=\d+[.)]\s)", compact)
    ]
    numbered_steps = [step for step in numbered_steps if step]
    if len(numbered_steps) > 1:
        return numbered_steps

    return [compact]


# --- the answers of the structured roles --------------------------------------------------------

_StringList = Annotated[list[str], BeforeValidator(normalize_string_list)]
_PlanList = Annotated[list[str], BeforeValidator(normalize_implementation_plan)]


class ProposedAnalysis(StateModel):
    """One analysis proposed by the generator (an entry of ``posthoc_bundle.json``)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    title: str
    family: str
    hypothesis: str = ""
    relevance_reason: str = ""
    feasibility_reason: str = ""
    required_artifacts: _StringList = Field(default_factory=list)
    requires_retrain: bool = False
    implementation_plan: _PlanList = Field(default_factory=list)
    expected_claim: str = ""
    expected_outputs: list[str] = Field(default_factory=list)


class PosthocBundle(StateModel):
    """The generator's answer: upstream's ``posthoc_bundle.json``.

    Attributes:
        decision: ``propose_analyses`` or ``no_relevant_analyses``.
        summary: A short summary.
        analyses: Zero or more proposals.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    decision: Literal["propose_analyses", "no_relevant_analyses"]
    summary: str = ""
    analyses: list[ProposedAnalysis] = Field(default_factory=list)


class SelectionDecision(StateModel):
    """One decision of the selector (an entry of ``decisions`` in the prompt's schema)."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str
    accepted: bool = True
    reason: str = ""


class SelectionAnswer(StateModel):
    """The selector's answer: the schema block of ``posthoc_select.md``."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    decisions: list[SelectionDecision] = Field(default_factory=list)


# --- records ------------------------------------------------------------------------------------


class AnalysisCandidate(StateModel):
    """A proposed post-hoc analysis candidate (upstream ``AnalysisCandidate``).

    ``model_dump(by_alias=True)`` writes upstream's ``to_dict()`` (the id under the key ``id``).
    """

    analysis_id: str = Field(alias="id")
    title: str
    family: str
    hypothesis: str = ""
    relevance_reason: str = ""
    feasibility_reason: str = ""
    required_artifacts: list[str] = Field(default_factory=list)
    requires_retrain: bool = False
    implementation_plan: list[str] = Field(default_factory=list)
    expected_claim: str = ""
    expected_outputs: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return upstream's JSON form of the candidate."""
        return self.model_dump(mode="json", by_alias=True)

    @classmethod
    def from_proposal(cls, proposal: ProposedAnalysis) -> AnalysisCandidate | None:
        """Build a candidate from a proposal; ``None`` for an unsafe id (upstream skips it)."""
        if not is_valid_analysis_id(proposal.id):
            return None
        return cls(
            analysis_id=proposal.id,
            title=proposal.title,
            family=proposal.family,
            hypothesis=proposal.hypothesis,
            relevance_reason=proposal.relevance_reason,
            feasibility_reason=proposal.feasibility_reason,
            required_artifacts=normalize_string_list(proposal.required_artifacts),
            requires_retrain=bool(proposal.requires_retrain),
            implementation_plan=normalize_implementation_plan(proposal.implementation_plan),
            expected_claim=proposal.expected_claim,
            expected_outputs=normalize_expected_outputs(proposal.expected_outputs),
        )


class RejectedCandidate(StateModel):
    """A candidate the selection rejected, with the reason."""

    candidate: AnalysisCandidate
    reason: str


class AnalysisSelection(StateModel):
    """The result of selecting candidates (upstream ``AnalysisSelection``)."""

    accepted: list[AnalysisCandidate] = Field(default_factory=list)
    rejected: list[RejectedCandidate] = Field(default_factory=list)


class AnalysisSpec(StateModel):
    """Specification of a materialised analysis workspace (upstream ``AnalysisSpec``)."""

    analysis_id: str
    title: str
    family: str
    analysis_dir: str
    hypothesis: str = ""
    relevance_reason: str = ""
    feasibility_reason: str = ""
    required_artifacts: list[str] = Field(default_factory=list)
    requires_retrain: bool = False
    implementation_plan: list[str] = Field(default_factory=list)
    expected_claim: str = ""
    expected_outputs: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return upstream's ``analysis_spec.json`` content."""
        return self.model_dump(mode="json")


class AttemptInspection(StateModel):
    """Inspection result of one attempt directory (upstream ``AttemptInspection``)."""

    attempt_dir: str
    attempt_id: str
    overall_pass: bool
    rubric_error: str | None
    required_outputs_present: bool
    missing_outputs: list[str]
    manifest_valid: bool
    manifest_error: str | None
    guardrail_passed: bool | None = None
    checkpoint_validated: bool = False
    manifest: dict[str, Any] | None = None
    rubric_status: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return upstream's JSON form (also the ``ERROR_CONTEXT`` of the next attempt)."""
        return self.model_dump(mode="json")


class AnalysisRunResult(StateModel):
    """Result of executing one analysis workspace (upstream ``AnalysisRunResult``)."""

    analysis_id: str
    status: Literal["completed", "failed", "timeout"]
    attempt_count: int
    exit_code: int
    selected_attempt: str | None
    rubric_passed: bool
    required_outputs_present: bool
    guardrail_passed: bool | None
    output: str
    produced_files: dict[str, bool]
    attempts: list[AttemptInspection] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return upstream's ``analysis_result.json`` content."""
        return self.model_dump(mode="json")


class BootstrapReport(StateModel):
    """Upstream's ``bootstrap_report.json`` (paths are workspace paths)."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    stage: str = "bootstrap"
    status: Literal["completed", "degraded"]
    bootstrapped_at: str
    source_bundle_dir: str
    analysis_ready_bundle_dir: str
    source_program_dir: str
    promoted_runtime_artifacts: dict[str, Any] | None
    baseline_primary_metric: dict[str, Any] | None
    current_primary_metric: dict[str, Any] | None
    guardrail: dict[str, Any] | None
    checkpoint: dict[str, Any] | None
    bootstrap_agent: dict[str, Any] | None
    rerun: dict[str, Any] | None
    retry_attempts: list[dict[str, Any]] = Field(default_factory=list)
    bootstrap_ready: bool
    failure_reason: str | None
    retrain_if_checkpoint_missing: bool
    bootstrap_max_attempts: int
    checkpoint_mode: str


# --- graph schemas ------------------------------------------------------------------------------


class PosthocInput(TypedDict):
    """Input of the post-hoc graph.

    Attributes:
        task: The run-local task snapshot.
        run_id: The run id (evaluation ids derive from it).
        run_config: The parent's recorded configuration when mounted.
        output_dir: Workspace directory of the phase's files (default ``posthoc``).
        investigation: Investigation result (its best program is the default incumbent).
        downstream: The no-winner fallback selection, preferred over the best program.
        variants: Ablation result; its output directory is staged as evidence when completed.
        incumbent: An explicit incumbent program, preferred over both.
        incumbent_manifest: Score, correctness and provenance of an explicit incumbent.
        evidence_dir: An explicit completed ablations directory staged as evidence.
        artifact_only: The integrated pipeline's mode (``checkpoint_mode='optional'`` and no
            bootstrap retraining), as upstream's ``build_posthoc_cmd`` sets it.
    """

    task: TaskRef
    run_id: str
    run_config: NotRequired[RunConfigRecord | None]
    output_dir: NotRequired[str]
    investigation: NotRequired[InvestigationResult | None]
    downstream: NotRequired[DownstreamSelection | None]
    variants: NotRequired[VariantsResult | None]
    incumbent: NotRequired[ProgramRef | None]
    incumbent_manifest: NotRequired[BestManifest | None]
    evidence_dir: NotRequired[str | None]
    artifact_only: NotRequired[bool]


class PosthocState(TypedDict, total=False):
    """State of the post-hoc graph."""

    task: TaskRef
    run_id: str
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    output_dir: str
    investigation: InvestigationResult | None
    downstream: DownstreamSelection | None
    variants: VariantsResult | None
    incumbent: ProgramRef | None
    incumbent_manifest: BestManifest | None
    evidence_dir: str | None
    artifact_only: bool
    phase_error: Failure | None
    best: dict[str, Any] | None
    extraction: dict[str, Any] | None
    baseline_primary: dict[str, Any] | None
    bootstrap_attempt: int
    bootstrap_attempts: list[dict[str, Any]]
    bootstrap_scratch: dict[str, Any]
    bootstrap_error: str | None
    bootstrap_status: Literal["pending", "completed", "degraded"]
    bootstrap: BootstrapReport | None
    candidates: list[AnalysisCandidate]
    generation: dict[str, Any]
    llm_decisions: dict[str, dict[str, Any]]
    selection: AnalysisSelection | None
    specs: list[AnalysisSpec]
    analysis_runs: Annotated[list[AnalysisRunResult], append]
    publication: dict[str, Any] | None
    posthoc: PosthocResult | None
    events: Annotated[list[Event], append]


class PosthocOutput(TypedDict, total=False):
    """Output of the post-hoc graph: the phase result and the published listing."""

    posthoc: PosthocResult
    publication: dict[str, Any] | None
    specs: list[AnalysisSpec]
    events: list[Event]


class AnalysisInput(TypedDict):
    """Input of the per-analysis subgraph (one ``Send`` per selected analysis)."""

    spec: AnalysisSpec
    task: TaskRef
    run_id: str
    run_config: RunConfigRecord | None
    output_dir: str
    artifact_only: bool


class AnalysisState(TypedDict, total=False):
    """State of the per-analysis subgraph (its attempt loop)."""

    spec: AnalysisSpec
    task: TaskRef
    run_id: str
    run_config: RunConfigRecord | None
    output_dir: str
    artifact_only: bool
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    pristine_digest: str | None
    pristine: SnapshotRef | None
    attempt: int
    inspections: list[AttemptInspection]
    agent_outputs: list[str]
    last_exit_code: int
    timed_out: bool
    status: Literal["running", "completed", "failed", "timeout"]
    analysis_run: AnalysisRunResult | None
    analysis_runs: Annotated[list[AnalysisRunResult], append]
    events: Annotated[list[Event], append]


class AnalysisOutput(TypedDict, total=False):
    """Output of the per-analysis subgraph, appended to the phase's ``analysis_runs``."""

    analysis_runs: list[AnalysisRunResult]
    events: list[Event]


__all__ = [
    "AnalysisCandidate",
    "AnalysisInput",
    "AnalysisOutput",
    "AnalysisResult",
    "AnalysisRunResult",
    "AnalysisSelection",
    "AnalysisSpec",
    "AnalysisState",
    "AttemptInspection",
    "BootstrapReport",
    "PosthocBundle",
    "PosthocInput",
    "PosthocOutput",
    "PosthocState",
    "ProposedAnalysis",
    "RejectedCandidate",
    "SelectionAnswer",
    "SelectionDecision",
    "is_valid_analysis_id",
    "normalize_expected_outputs",
    "normalize_implementation_plan",
    "normalize_string_list",
]
