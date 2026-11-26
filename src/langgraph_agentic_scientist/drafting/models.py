"""Records of the scaffold drafting graph: alignment, transactions, qualification and history.

Every record is a :class:`~langgraph_agentic_scientist.core.state.StateModel`, so that it can be
stored in graph state, checkpointed, and (under a strict msgpack serializer) allow-listed through
:func:`~langgraph_agentic_scientist.serde.serde_allowlist`. Field names that upstream writes into
JSON files and that clash with the class-level ``schema_version`` of ``StateModel`` carry an alias
(``report_schema_version`` is read and written as ``schema_version``).

The alignment report is the structured answer of the alignment judge (upstream's
``task_alignment.json``, schema 1). Upstream's history records (``alignment_<n>.json``,
``qualification_<n>_transaction.json``, ``qualification_repair_history.json`` and the calibration
record) are kept as records in state instead of files, with the same fields and bounds.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from ..core.state import StateModel
from ..workspace.protocol import SnapshotRef

REQUIRED_ALIGNMENT_DIMENSIONS: tuple[str, ...] = (
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
"""The eleven dimensions of an alignment report (upstream ``REQUIRED_ALIGNMENT_DIMENSIONS``)."""

FINGERPRINTED_FILES: tuple[str, ...] = (
    "task.md",
    "capture.py",
    "task_contract.json",
    "evaluate.py",
    "evaluate_final.py",
    "program_0/program_spec.toml",
)
"""The scaffold files whose digests an alignment report must quote (upstream rule)."""

StrictLine = Annotated[int, Field(strict=True)]


# --- alignment -----------------------------------------------------------------------------


class AlignmentEvidence(StateModel):
    """One evidence item of a dimension: an existing scaffold file, a 1-based line, a detail."""

    path: str
    line: StrictLine
    detail: str


class AlignmentDimension(StateModel):
    """One dimension of the alignment report.

    ``verdict`` is ``pass`` or ``fail``; a failed dimension needs a remediation, a passed one an
    empty remediation (the host checks both, with upstream's messages).
    """

    id: str
    verdict: str
    evidence: list[AlignmentEvidence]
    remediation: str


class AlignmentReport(StateModel):
    """The alignment judge's report (upstream ``task_alignment.json``, schema 1).

    Attributes:
        report_schema_version: Written as ``schema_version``; must be ``1``.
        verdict: ``pass`` only if every dimension passes, otherwise ``fail``.
        summary: Concise decision rationale.
        fingerprints: Exactly the expected fingerprint object the judge was given.
        dimensions: One object for every required dimension.
    """

    report_schema_version: int = Field(alias="schema_version")
    verdict: str
    summary: str
    fingerprints: dict[str, str]
    dimensions: list[AlignmentDimension]


class AlignmentFinding(StateModel):
    """A failed dimension retained in the alignment history (bounded, redacted)."""

    id: str
    verdict: Literal["fail"] = "fail"
    remediation: str
    evidence: list[dict[str, Any]]
    truncated: bool = False


class AlignmentHistoryEntry(StateModel):
    """One validated alignment report in the retained history (upstream ``alignment_<n>.json``).

    The history is evidence for later repair prompts and judge prompts, never acceptance
    authority.
    """

    report_sha256: str
    reviewed_task_fingerprint: str
    declared_scaffold_fingerprint: str
    historical_verdict: Literal["pass", "fail"]
    findings: list[AlignmentFinding]
    report_name: str = ""


# --- gates and transactions ----------------------------------------------------------------


class RepairTransaction(StateModel):
    """A snapshot of the scaffold taken before a repair that may have to be undone.

    Attributes:
        label: The repair's label (``smoke_repair_0``, ``alignment_0``, ``qualification_1``, ...).
        kind: ``scoped`` (smoke repair: files and contract fields outside the mutable paths must
            not change) or ``tree`` (alignment and qualification repair: the whole authored
            tree is restored).
        snapshot: The workspace snapshot of the staged task directory.
        base_fingerprint: The tree fingerprint at the snapshot.
        files: For a scoped transaction, the digest of every file before the repair.
        mutable_paths: For a scoped transaction, the files a repair may change.
        contract_policy: How the contract may change: ``immutable``, ``synchronized-digests`` or
            ``synchronized-dependency-removals``.
        contract_invariants: The parsed contract before the repair (scoped transactions).
        retained_dependencies: For the dependency-removal policy, the dependency list the first
            validated repair derived from ``requirements.txt`` (upstream's journal).
        sources_before: For a tree transaction, the bounded source files at the snapshot (at most
            4 MiB in all), from which the diff of a rejected repair is computed; the workspace
            protocol does not let a snapshot be read, so the sources are kept here.
        sources_omitted: How many files were left out of ``sources_before``.
    """

    label: str
    kind: Literal["scoped", "tree"]
    snapshot: SnapshotRef
    base_fingerprint: str | None = None
    files: dict[str, str] = Field(default_factory=dict)
    sources_before: dict[str, str] = Field(default_factory=dict)
    sources_omitted: int = 0
    mutable_paths: list[str] = Field(default_factory=list)
    contract_policy: Literal[
        "immutable", "synchronized-digests", "synchronized-dependency-removals"
    ] = "synchronized-digests"
    contract_invariants: dict[str, Any] | None = None
    retained_dependencies: list[dict[str, str]] | None = None


class DependencyTransaction(StateModel):
    """The provisional dependency-removal state of a smoke repair chain.

    Upstream keeps one transaction across the smoke repairs of a chain until every smoke check
    passes; ``dependency_repair_active`` says whether removing an unused direct pin is allowed.
    """

    transaction: RepairTransaction
    dependency_repair_active: bool = False


class QualificationReceipt(StateModel):
    """What a passing qualification binds (upstream ``TaskQualificationReceipt``).

    The upstream receipt binds a container runtime (image id, install cache key); the port binds
    the scaffold, the published file plan, the evaluators, the requirements and the baseline's
    result instead.
    """

    task_fingerprint: str
    publication_plan_sha256: str
    evaluator_tree_sha256: str
    requirements_sha256: str
    baseline_combined_score: float
    baseline_correct: bool | None = None
    baseline_feasible: bool = True
    accelerator_model: str = "none"
    evaluator_gpu_count: int = 0
    candidate_gpu_count: int = 0
    evaluation_id: str = ""


class QualificationRecord(StateModel):
    """The report of a full production-search qualification (upstream attempt report).

    Attributes:
        status: ``pass``, ``authored_failure`` or ``infrastructure_failure``.
        category: ``qualification_passed`` on success, otherwise the failure category.
        stage: The stage that failed (``production-search`` by default).
        diagnostic: Bounded, sanitised diagnostic (repair prompts only).
        task_fingerprint: The scaffold fingerprint the qualification ran against.
        elapsed_seconds: Wall clock of the qualification.
        receipt: The receipt of a passing qualification.
        failure_evidence: Allow-listed numeric evidence of a failure.
    """

    status: Literal["pass", "authored_failure", "infrastructure_failure"]
    category: str
    stage: str = "production-search"
    diagnostic: str = ""
    task_fingerprint: str = ""
    elapsed_seconds: float = 0.0
    receipt: QualificationReceipt | None = None
    failure_evidence: dict[str, Any] = Field(default_factory=dict)


class QualificationFailure(StateModel):
    """A failed scaffold fingerprint in the qualification repair history.

    A repair that reproduces a fingerprint of this history is an oscillation.
    """

    fingerprint: str
    category: str
    evidence_sha256: str


class QualificationCalibration(StateModel):
    """The calibration record of a provisional qualification repair (upstream amendment).

    It binds the original and calibrated scaffold fingerprints, the immutable accepted design
    and the qualification failure; it becomes ``accepted`` only when alignment and qualification
    both accept the calibrated scaffold.
    """

    calibration_schema_version: int = Field(1, alias="schema_version")
    status: Literal["provisional", "accepted"] = "provisional"
    original_design_immutable: bool = True
    original_fingerprint: str
    calibrated_fingerprint: str
    evidence_sha256: dict[str, str]


class PendingQualificationRepair(StateModel):
    """A qualification repair that is neither committed nor rolled back.

    Attributes:
        transaction: The tree transaction taken before the first repair of the chain.
        base_fingerprint: The scaffold fingerprint at that snapshot.
        calibrated_fingerprint: The fingerprint of the repaired scaffold.
        calibration: The calibration record, once created.
        source_report: The qualification failure the repair answers.
        index: The repair's index in the chain.
    """

    transaction: RepairTransaction
    base_fingerprint: str
    calibrated_fingerprint: str | None = None
    calibration: QualificationCalibration | None = None
    source_report: QualificationRecord | None = None
    index: int = 0


class RejectedRepair(StateModel):
    """Bounded, source-only record of a rejected repair (upstream repair-context record).

    Attributes:
        base_fingerprint: The scaffold the repair started from.
        rejected_fingerprint: The scaffold the repair produced, when known.
        reports: Up to three reports (text bounded to 16 KiB, with their SHA-256).
        diff: A unified source diff of the repair, bounded to 64 KiB.
        truncated: Whether the diff was cut.
        omitted_files: Files left out of the diff (unreadable, sensitive or oversized).
    """

    base_fingerprint: str
    rejected_fingerprint: str | None = None
    reports: list[dict[str, Any]] = Field(default_factory=list)
    diff: str = ""
    truncated: bool = False
    omitted_files: int = 0


__all__ = [
    "FINGERPRINTED_FILES",
    "REQUIRED_ALIGNMENT_DIMENSIONS",
    "AlignmentDimension",
    "AlignmentEvidence",
    "AlignmentFinding",
    "AlignmentHistoryEntry",
    "AlignmentReport",
    "DependencyTransaction",
    "PendingQualificationRepair",
    "QualificationCalibration",
    "QualificationFailure",
    "QualificationReceipt",
    "QualificationRecord",
    "RejectedRepair",
    "RepairTransaction",
]
