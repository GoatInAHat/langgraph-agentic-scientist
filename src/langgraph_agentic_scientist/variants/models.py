"""Records and structured answers of the variants phase (ablations and baselines).

The phase interface records (:class:`~langgraph_agentic_scientist.VariantSpec`,
:class:`~langgraph_agentic_scientist.VariantResult`,
:class:`~langgraph_agentic_scientist.VariantsResult`) live in the core. This module holds the
records the phase keeps in its own state, the structured answers of its roles (the variant
generator, the variant reviewer and the README summariser) and the helpers that translate between
the core's :class:`~langgraph_agentic_scientist.VariantSpec` and upstream's ``variant_spec.json``.

Upstream's ``VariantSpec`` has the fields ``base_program``, ``hypothesis``, ``relevance_reason``,
``feasibility_reason``, ``target_components``, ``implementation_plan``, ``expected_effect`` and
``description_body``. The core's spec carries ``description`` (upstream's ``description_body``),
``rationale`` (``relevance_reason``) and a ``details`` mapping for the rest; :func:`spec_to_dict`
returns upstream's layout, which is what ``variant_spec.json`` and the prompts show.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.results import VariantResult, VariantSpec
from ..core.state import Failure, ProgramRef, StateModel

# --- candidates and selection ----------------------------------------------------------------


class VariantCandidate(StateModel):
    """A proposed baseline or ablation (upstream ``VariantCandidate``).

    Attributes:
        variant_id: Snake-case identifier; becomes the variant's directory name.
        title: Human-readable title.
        kind: ``baseline`` or ``ablation`` (custom baselines are ``baseline``).
        base_program: ``best`` or ``reference`` (a folder baseline starts from reference code).
        hypothesis: What the comparison is expected to show.
        relevance_reason: Why the comparison is scientifically meaningful.
        feasibility_reason: Why a coding agent can implement it in isolation.
        target_components: Components or functions to change.
        implementation_plan: Steps of the change.
        expected_effect: Expected impact on the evaluation metrics.
        description_body: Markdown body of a custom baseline (empty for generated candidates).
        reference_dir: Workspace path of a folder baseline's reference implementation.
    """

    variant_id: str
    title: str
    kind: str
    base_program: str = "best"
    hypothesis: str = ""
    relevance_reason: str = ""
    feasibility_reason: str = ""
    target_components: list[str] = Field(default_factory=list)
    implementation_plan: list[str] = Field(default_factory=list)
    expected_effect: str = ""
    description_body: str = ""
    reference_dir: str | None = None


class RejectedCandidate(StateModel):
    """A candidate the selection stage rejected, with the reason."""

    candidate: VariantCandidate
    reason: str


class VariantSelection(StateModel):
    """The result of the selection stage: accepted candidates, rejections and the controls."""

    accepted: list[VariantCandidate] = Field(default_factory=list)
    rejected: list[RejectedCandidate] = Field(default_factory=list)
    controls: list[str] = Field(default_factory=lambda: ["control_best"])


# --- structured answers of the roles ---------------------------------------------------------


class _Answer(BaseModel):
    """Base of the structured answers: unknown keys are ignored, as upstream's ``dict.get`` does."""

    model_config = ConfigDict(extra="ignore")


class VariantProposal(_Answer):
    """One candidate of the generator's bundle."""

    variant_id: str = Field(description="snake_case_identifier")
    title: str = Field(description="Human-readable title")
    kind: Literal["baseline", "ablation"] = Field(description='"baseline" or "ablation"')
    base_program: str = Field(default="best", description='"best"')
    hypothesis: str = Field(default="", description="What we expect to learn from this comparison")
    relevance_reason: str = Field(
        default="", description="Why this comparison is scientifically meaningful"
    )
    feasibility_reason: str = Field(
        default="", description="Why a coding agent can implement this in isolation"
    )
    target_components: list[str] = Field(
        default_factory=list, description="specific_component_or_function_to_change"
    )
    implementation_plan: list[str] = Field(
        default_factory=list, description="step 1 description, step 2 description, ..."
    )
    expected_effect: str = Field(
        default="", description="Expected impact on the evaluation metrics"
    )


class VariantBundleAnswer(_Answer):
    """The variant bundle (``variant_bundle.json``): baselines and ablations to run, or none."""

    decision: Literal["propose_variants", "no_relevant_variants"] = Field(
        default="no_relevant_variants",
        description='"propose_variants" or "no_relevant_variants"',
    )
    summary: str = Field(
        default="", description="Brief explanation of what variants you propose and why"
    )
    candidates: list[VariantProposal] = Field(default_factory=list)


class VariantDecision(_Answer):
    """The reviewer's decision on one candidate."""

    variant_id: str = Field(description="the_variant_id")
    accepted: bool = Field(default=True, description="true or false")
    reason: str = Field(
        default="", description="Clear explanation of why this variant is accepted or rejected"
    )


class VariantDecisionsAnswer(_Answer):
    """The reviewer's decisions on the proposed candidates (``variant_selection.json``)."""

    decisions: list[VariantDecision] = Field(default_factory=list)


class VariantSummariesAnswer(_Answer):
    """The README summariser's answer (``variant_summaries.json``)."""

    task_summary: str = Field(
        default="",
        description="One sentence describing what this task optimizes and how it is evaluated.",
    )
    variant_summaries: dict[str, str] = Field(
        default_factory=dict,
        description="Variant id to a 2-3 sentence summary of what the variant is and tests.",
    )


# --- bundles, retrofit and runs --------------------------------------------------------------


class BundleRef(StateModel):
    """The extracted bundle of stage 0.

    Attributes:
        output_dir: The phase's output directory in the workspace.
        extracted: Path of ``_extracted_bundle``.
        has_reference: Whether ``reference_program/`` exists in the bundle.
        entrypoint: Entrypoint of the best program (``program_0``), relative to the program.
        manifest: The extraction manifest as written to ``extraction_manifest.json``.
    """

    output_dir: str
    extracted: str
    has_reference: bool = True
    entrypoint: str = "src/program.py"
    manifest: dict[str, Any] = Field(default_factory=dict)


class RetrofitReport(StateModel):
    """The report of a retrofit (stage 1, or phase 0 of a cross-task adaptation).

    Attributes:
        stage: Always ``retrofit``.
        bundle_dir: The bundle that was retrofitted.
        output_bundle_dir: The retrofitted copy.
        exit_code: 0 on success (upstream's report field).
        mode: ``brokered_preserve`` when a contracted candidate was copied unchanged.
        error: Why the retrofit failed.
        program_tree_sha256: Digest of the preserved program of a brokered copy.
        sealed_evaluator_sha256: SHA-256 of the sealed ``evaluate.py``.
        sealed_source_tree_sha256: Evaluator tree digest of the source bundle.
        preservation: Brokered-preservation metadata (``None`` for a legacy executable).
        agent_model: The model that played the retrofit agent.
    """

    stage: Literal["retrofit"] = "retrofit"
    bundle_dir: str
    output_bundle_dir: str
    exit_code: int = 1
    mode: str | None = None
    error: str | None = None
    program_tree_sha256: str | None = None
    sealed_evaluator_sha256: str | None = None
    sealed_source_tree_sha256: str | None = None
    preservation: dict[str, str] | None = None
    agent_model: str | None = None


class VariantRun(StateModel):
    """The outcome of running one variant, with what upstream's ``VariantRunResult`` carries.

    Attributes:
        result: The core's :class:`~langgraph_agentic_scientist.VariantResult`.
        status: Upstream's status: ``completed`` (the rubrics passed), ``failed`` or ``timeout``.
        eval_count: Evaluations that completed.
        exit_code: The last exit code (124 for a timeout, an evaluator's return code otherwise).
        output: Bounded notes of the run (upstream's concatenated agent and error output).
        analysis_files: Analysis filename to whether it exists in some ``program_<n>``.
        spec: The spec that was run.
        task_bundle: Workspace path of the variant's ``task_bundle``.
        infrastructure_failure: An evaluator infrastructure failure, which stops the phase.
    """

    result: VariantResult
    status: Literal["completed", "failed", "timeout"]
    eval_count: int = 0
    exit_code: int = 1
    output: str = ""
    analysis_files: dict[str, bool] = Field(default_factory=dict)
    spec: VariantSpec
    task_bundle: str
    infrastructure_failure: Failure | None = None

    @property
    def rubric_passed(self) -> bool:
        """Whether a program of the variant passed the rubric contract (upstream's field)."""
        return self.result.passed


# --- the core's VariantSpec in upstream's layout ---------------------------------------------

_DETAIL_KEYS = (
    "base_program",
    "hypothesis",
    "feasibility_reason",
    "target_components",
    "implementation_plan",
    "expected_effect",
)


def make_spec(
    *,
    variant_id: str,
    title: str,
    kind: str,
    base_program: str,
    variant_dir: str,
    hypothesis: str = "",
    relevance_reason: str = "",
    feasibility_reason: str = "",
    target_components: Sequence[str] = (),
    implementation_plan: Sequence[str] = (),
    expected_effect: str = "",
    description_body: str = "",
    task_dir: str | None = None,
    program: ProgramRef | None = None,
    task_name: str | None = None,
) -> VariantSpec:
    """Build a core :class:`~langgraph_agentic_scientist.VariantSpec` from upstream's fields."""
    return VariantSpec(
        variant_id=variant_id,
        kind=kind,
        title=title,
        description=description_body,
        rationale=relevance_reason,
        variant_dir=variant_dir,
        task_dir=task_dir,
        program=program,
        task_name=task_name,
        details={
            "base_program": base_program,
            "hypothesis": hypothesis,
            "feasibility_reason": feasibility_reason,
            "target_components": list(target_components),
            "implementation_plan": list(implementation_plan),
            "expected_effect": expected_effect,
        },
    )


def spec_detail(spec: VariantSpec, key: str) -> Any:
    """Return one of upstream's spec fields that the core spec keeps in ``details``."""
    if key not in _DETAIL_KEYS:
        raise KeyError(key)
    default: Any = [] if key in ("target_components", "implementation_plan") else ""
    if key == "base_program":
        default = "best"
    return spec.details.get(key, default)


def spec_to_dict(spec: VariantSpec) -> dict[str, Any]:
    """Return the spec in upstream's ``VariantSpec.to_dict`` layout (``variant_spec.json``)."""
    return {
        "variant_id": spec.variant_id,
        "title": spec.title,
        "kind": spec.kind,
        "base_program": spec_detail(spec, "base_program"),
        "variant_dir": spec.variant_dir,
        "hypothesis": spec_detail(spec, "hypothesis"),
        "relevance_reason": spec.rationale,
        "feasibility_reason": spec_detail(spec, "feasibility_reason"),
        "target_components": spec_detail(spec, "target_components"),
        "implementation_plan": spec_detail(spec, "implementation_plan"),
        "expected_effect": spec_detail(spec, "expected_effect"),
        "description_body": spec.description,
    }


def spec_from_dict(data: Mapping[str, Any], *, variant_dir: str | None = None) -> VariantSpec:
    """Read a ``variant_spec.json`` mapping (upstream layout) into the core's spec."""
    return make_spec(
        variant_id=str(data["variant_id"]),
        title=str(data.get("title", "")),
        kind=str(data.get("kind", "")),
        base_program=str(data.get("base_program", "")),
        variant_dir=variant_dir if variant_dir is not None else str(data.get("variant_dir", "")),
        hypothesis=str(data.get("hypothesis", "")),
        relevance_reason=str(data.get("relevance_reason", "")),
        feasibility_reason=str(data.get("feasibility_reason", "")),
        target_components=[str(item) for item in data.get("target_components", [])],
        implementation_plan=[str(item) for item in data.get("implementation_plan", [])],
        expected_effect=str(data.get("expected_effect", "")),
        description_body=str(data.get("description_body", "")),
    )


def candidate_from_proposal(proposal: VariantProposal) -> VariantCandidate:
    """Convert a generator's proposal into a candidate (upstream ``_parse_variant_bundle``)."""
    return VariantCandidate(
        variant_id=proposal.variant_id,
        title=proposal.title,
        kind=proposal.kind,
        base_program=proposal.base_program,
        hypothesis=proposal.hypothesis,
        relevance_reason=proposal.relevance_reason,
        feasibility_reason=proposal.feasibility_reason,
        target_components=list(proposal.target_components),
        implementation_plan=list(proposal.implementation_plan),
        expected_effect=proposal.expected_effect,
    )


__all__ = [
    "BundleRef",
    "RejectedCandidate",
    "RetrofitReport",
    "VariantBundleAnswer",
    "VariantCandidate",
    "VariantDecision",
    "VariantDecisionsAnswer",
    "VariantProposal",
    "VariantRun",
    "VariantSelection",
    "VariantSummariesAnswer",
    "candidate_from_proposal",
    "make_spec",
    "spec_detail",
    "spec_from_dict",
    "spec_to_dict",
]
