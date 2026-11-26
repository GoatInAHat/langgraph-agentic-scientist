"""Records of the task design loop: the design, its review, the repair patch and the delta audit.

Upstream keeps these as JSON files that its agents write (``src/task_drafting/design.py`` loads and
validates them). In the port the roles answer through structured output, so each file is a
pydantic model: the model is the schema the role answers with, the contract functions of
:mod:`~langgraph_agentic_scientist.drafting.design` apply upstream's semantic rules to it, and the
host writes the JSON file under the name upstream uses.

JSON keys that clash with Python are aliased: ``schema_version`` (a class variable of
:class:`~langgraph_agentic_scientist.core.state.StateModel`) and ``class`` (a keyword). Every model
dumps with ``by_alias=True`` to upstream's JSON.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import Field, StrictInt

from ..core.state import StateModel

DESIGN_DIMENSIONS: tuple[str, ...] = (
    "goal_and_method_fidelity",
    "upstream_grounding",
    "search_proxy_fidelity",
    "final_evaluation_fidelity",
    "search_final_isolation",
    "metrics_and_rubrics",
    "candidate_editability",
    "compute_realism",
    "execution_feasibility",
    "reproducibility",
    "publication_scope",
)
"""The eleven dimensions of a design review, in upstream's order."""

MAX_DESIGN_ARTIFACT_BYTES = 1024 * 1024
"""Upstream's size bound on a design, a review or a patch (1 MiB)."""

MAX_DESIGN_PATCH_OPERATIONS = 64
"""Upstream's bound on the operations of one patch."""

IMMUTABLE_DESIGN_PATHS: tuple[str, ...] = ("/candidate/interface",)
"""JSON pointers that no patch and no repair authority may touch."""


def canonical_json(payload: Any) -> str:
    """Return upstream's canonical JSON text (sorted keys, no spaces) that fingerprints hash."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Capacity(StateModel):
    """The investigation capacity a design must fit (upstream ``MAX_IDEA_CYCLES``,
    ``MAX_PROGRAM_EVALS_PER_ATTEMPT``).

    The campaign evaluation ceiling is ``max_idea_cycles * max_program_evals_per_attempt``.
    """

    max_idea_cycles: int = Field(20, ge=1)
    max_program_evals_per_attempt: int = Field(4, ge=1)


# --- TaskDesign v1 ------------------------------------------------------------------------------


class DesignCanonicalTask(StateModel):
    """``canonical_task``: the path (always ``task.md``) and the SHA-256 of the task text."""

    path: str
    sha256: str


class DesignMethod(StateModel):
    """``method``: the method class and the bounded scientific claim."""

    class_name: str = Field(alias="class")
    claim: str


class DesignUpstream(StateModel):
    """One named external repository or asset that ``task.md`` requires."""

    repository: str
    revision: str
    license: str
    entrypoints: list[str]


class DesignCandidate(StateModel):
    """``candidate``: the interface (``python_rpc_v1:handle``), editable paths and state rules."""

    interface: str
    editable_paths: list[str]
    state_semantics: str


class DesignBudget(StateModel):
    """The budget of one plane."""

    unit: str
    training_steps: StrictInt
    wall_clock_seconds: StrictInt
    hardware: str
    precision: str


class DesignPlane(StateModel):
    """One evaluation plane (``search`` or ``final``) of the design."""

    resources: list[str]
    seeds: list[StrictInt]
    budget: DesignBudget
    baselines: list[str]
    metrics: list[str]
    correctness: list[str]
    smoke: str


class DesignPlanes(StateModel):
    """``planes``: exactly the search and the final plane."""

    search: DesignPlane
    final: DesignPlane


class DesignFeedback(StateModel):
    """``feedback``: what search may see and what stays sealed until the final evaluation."""

    search_visible: list[str]
    final_sealed: list[str]


class DesignPublication(StateModel):
    """``publication``: evidence to produce before a claim may be published."""

    required_evidence: list[str]


class TaskDesign(StateModel):
    """TaskDesign v1: the plan-only, pre-scaffold design of a task.

    This is the structured answer of the task designer and of the structural repairer. The
    semantic rules of upstream's loader (hash of ``task.md``, editable paths inside
    ``program_0``, disjoint seeds, unique values and so on) are applied by
    :func:`~langgraph_agentic_scientist.drafting.design.validate_design`.
    """

    design_schema_version: StrictInt = Field(1, alias="schema_version")
    canonical_task: DesignCanonicalTask
    method: DesignMethod
    upstream: list[DesignUpstream]
    candidate: DesignCandidate
    planes: DesignPlanes
    feedback: DesignFeedback
    publication: DesignPublication

    def payload(self) -> dict[str, Any]:
        """Return the design as upstream's JSON object."""
        return self.model_dump(mode="json", by_alias=True)

    def fingerprint(self) -> str:
        """Return the SHA-256 of the canonical JSON of the design (upstream's fingerprint)."""
        return hashlib.sha256(canonical_json(self.payload()).encode()).hexdigest()


# --- DesignReview v3 ----------------------------------------------------------------------------


class DesignDimension(StateModel):
    """One row of a design review: the verdict on one dimension with its evidence and scope.

    Attributes:
        id: One of :data:`DESIGN_DIMENSIONS`.
        verdict: ``pass`` or ``fail``.
        evidence: Concrete evidence strings (non-empty).
        design_paths: JSON pointers in the design that locate the evidence (non-empty).
        task_evidence: Lines or headings of ``task.md`` (non-empty).
        repair_paths: The narrowest JSON pointers a repair may replace; empty for a pass.
        remediation: The actionable remediation of a failed row; empty for a pass.
    """

    id: str
    verdict: Literal["pass", "fail"]
    evidence: list[str]
    design_paths: list[str]
    task_evidence: list[str]
    repair_paths: list[str]
    remediation: str


class DesignReview(StateModel):
    """A schema-3 design review: the judge's structured answer.

    ``verdict`` must be ``fail`` exactly when some dimension fails, and ``design_fingerprint``
    must equal the fingerprint of the reviewed design; both are checked by
    :func:`~langgraph_agentic_scientist.drafting.design.validate_review`.
    """

    review_schema_version: StrictInt = Field(3, alias="schema_version")
    verdict: Literal["pass", "fail"]
    summary: str
    design_fingerprint: str
    dimensions: list[DesignDimension]

    def payload(self) -> dict[str, Any]:
        """Return the review as upstream's JSON object."""
        return self.model_dump(mode="json", by_alias=True)

    @property
    def failed_dimensions(self) -> tuple[str, ...]:
        """The failed dimension ids, in the canonical order of :data:`DESIGN_DIMENSIONS`."""
        failed = {row.id for row in self.dimensions if row.verdict == "fail"}
        return tuple(identifier for identifier in DESIGN_DIMENSIONS if identifier in failed)


# --- DesignPatch --------------------------------------------------------------------------------


class DesignPatchOperation(StateModel):
    """One operation of a patch. Only ``replace`` is allowed; the contract rejects any other."""

    op: str
    path: str
    value: Any


class DesignPatch(StateModel):
    """A replace-only JSON patch on a design: the structured answer of the patch repairer."""

    patch_schema_version: StrictInt = Field(1, alias="schema_version")
    parent_fingerprint: str
    operations: list[DesignPatchOperation]

    def payload(self) -> dict[str, Any]:
        """Return the patch as upstream's JSON object."""
        return self.model_dump(mode="json", by_alias=True)


# --- DeltaAudit v2 ------------------------------------------------------------------------------


class DeltaRegression(StateModel):
    """One regression that a repair caused, bound to the paths it changed.

    ``repair_paths`` is required by schema 2 and absent in the legacy schema 1.
    """

    dimension: str
    design_paths: list[str]
    repair_paths: list[str] | None = None
    evidence: list[str]
    remediation: str


class DeltaAudit(StateModel):
    """A delta audit: the structured answer of the design delta auditor."""

    audit_schema_version: StrictInt = Field(2, alias="schema_version")
    verdict: Literal["pass", "fail"]
    summary: str
    parent_fingerprint: str
    candidate_fingerprint: str
    fixed_dimensions: list[str]
    regressions: list[DeltaRegression]

    def payload(self) -> dict[str, Any]:
        """Return the audit as upstream's JSON object (``repair_paths`` omitted when absent)."""
        data = self.model_dump(mode="json", by_alias=True)
        for row in data["regressions"]:
            if row.get("repair_paths") is None:
                row.pop("repair_paths", None)
        return data

    @property
    def regressed_dimensions(self) -> tuple[str, ...]:
        """The regressed dimension ids, in the order the auditor listed them."""
        return tuple(row.dimension for row in self.regressions)


# --- progress, memory, ledger -------------------------------------------------------------------


class DesignProgress(StateModel):
    """How a repair candidate compares with its parent (upstream's progress file).

    ``fixed``, ``still_failed`` and ``regressed`` come from comparing the two reviews. After a
    delta audit, ``fixed`` and ``regressed`` hold the auditor's view and the consensus view moves
    to ``consensus_fixed`` and ``consensus_regressed``.
    """

    fixed: list[str] = Field(default_factory=list)
    still_failed: list[str] = Field(default_factory=list)
    regressed: list[str] = Field(default_factory=list)
    improved: bool = False
    diagnostic: str | None = None
    candidate_findings: list[dict[str, Any]] = Field(default_factory=list)
    consensus_fixed: list[str] | None = None
    consensus_regressed: list[str] | None = None
    delta_regressions: list[dict[str, Any]] = Field(default_factory=list)
    audit_failure: str | None = None

    def payload(self) -> dict[str, Any]:
        """Return the progress as upstream's JSON object (only the keys upstream writes)."""
        data: dict[str, Any] = {
            "fixed": list(self.fixed),
            "still_failed": list(self.still_failed),
            "regressed": list(self.regressed),
            "improved": self.improved,
        }
        if self.diagnostic is not None:
            data["diagnostic"] = self.diagnostic
        data["candidate_findings"] = list(self.candidate_findings)
        if self.consensus_fixed is not None:
            data["consensus_fixed"] = list(self.consensus_fixed)
        if self.consensus_regressed is not None:
            data["consensus_regressed"] = list(self.consensus_regressed)
        if self.consensus_fixed is not None or self.delta_regressions:
            data["delta_regressions"] = list(self.delta_regressions)
        if self.audit_failure is not None:
            data["audit_failure"] = self.audit_failure
        return data


class RepairMemory(StateModel):
    """Rejected regressions kept as negative constraints for later repairs (at most 32).

    Upstream writes it as ``{"schema_version": 1, "constraints": [...]}``.
    """

    constraints: list[dict[str, Any]] = Field(default_factory=list)

    def payload(self) -> dict[str, Any]:
        """Return the memory as upstream's JSON object."""
        return {"schema_version": 1, "constraints": list(self.constraints)}


class DesignLedgerEntry(StateModel):
    """One line of upstream's candidate ledger (``candidate_ledger.jsonl``).

    ``event`` is one of ``reviewed``, ``full_review``, ``delta_review``, ``promoted``,
    ``frontier_advanced``, ``frontier_continued``, ``rejected``, ``repair_failed`` and
    ``latent_findings_rebound``.
    """

    event: str
    payload: dict[str, Any] = Field(default_factory=dict)

    def line(self) -> str:
        """Return the JSON line upstream appends to the ledger."""
        return json.dumps({**self.payload, "event": self.event}, ensure_ascii=False, sort_keys=True)


__all__ = [
    "DESIGN_DIMENSIONS",
    "IMMUTABLE_DESIGN_PATHS",
    "MAX_DESIGN_ARTIFACT_BYTES",
    "MAX_DESIGN_PATCH_OPERATIONS",
    "Capacity",
    "DeltaAudit",
    "DeltaRegression",
    "DesignBudget",
    "DesignCandidate",
    "DesignCanonicalTask",
    "DesignDimension",
    "DesignFeedback",
    "DesignLedgerEntry",
    "DesignMethod",
    "DesignPatch",
    "DesignPatchOperation",
    "DesignPlane",
    "DesignPlanes",
    "DesignProgress",
    "DesignPublication",
    "DesignReview",
    "DesignUpstream",
    "RepairMemory",
    "TaskDesign",
    "canonical_json",
]
