"""Evaluator protocols: everything the candidate must not touch.

Candidates are evaluated through an :class:`Evaluator` (search, variant, bootstrap, smoke,
production-entry and qualification planes) and the sealed final evaluation through a
:class:`FinalEvaluator`. Neither lives in the candidate's workspace.

Contract of both protocols (tested by the package's own tests):

1. **Idempotence and re-attachment.** A call with an ``evaluation_id`` the evaluator has seen
   returns the recorded result and never evaluates again. For an id whose evaluation started and
   has not been recorded, the evaluator returns its result when it can still obtain it (it
   re-attaches to an evaluation that is still running, or reads one that finished while the
   caller was down) and never starts a second evaluation. Only when the result can no longer be
   obtained does it return ``status="interrupted"`` (search planes) or raise
   :class:`~langgraph_agentic_scientist.core.errors.FinalEvaluationAlreadyStarted` (sealed).
   The rule that an interrupted call consumed evaluation budget belongs to the agent's
   ``evaluate_program`` tool, not to this protocol.
2. **The evaluator owns its outputs.** It never reads anything the candidate wrote outside the
   program tree, and the candidate never sees evaluator files.
3. **Host failures are** ``infrastructure_error``, distinct from candidate failures.

Because ``evaluation_id`` is derived deterministically by the graph
(:mod:`~langgraph_agentic_scientist.evaluation.ids`), an evaluation of hours survives a crash and
a resume of the graph: the resumed node asks again under the same id and gets the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from ..core.state import EvaluationPlane, ProgramRef, StateModel, TaskRef

EvaluationStatus = Literal["completed", "failed", "timeout", "interrupted", "infrastructure_error"]


class EvidenceEpisode(StateModel):
    """One episode of ``evaluation_evidence.json`` (upstream schema 1)."""

    seed: int
    score: float
    correct: bool
    duration_seconds: float
    solution: Any = None
    candidate_diagnostics: dict[str, float] = Field(default_factory=dict)
    candidate_timeseries: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationEvidence(StateModel):
    """A validated ``evaluation_evidence.json`` (schema 1, upstream limits kept).

    Candidate diagnostics are observations authored by the sealed evaluator, never independent
    validity certificates.
    """

    evidence_schema_version: Literal[1] = Field(default=1, alias="schema_version")
    episodes: list[EvidenceEpisode]


class EvidenceMetadata(StateModel):
    """Upstream ``evidence_metadata``: availability, digest and episode count of the evidence."""

    status: Literal["available", "unavailable"]
    path: str | None = None
    sha256: str | None = None
    episode_count: int | None = None
    candidate_diagnostics_trust: str | None = None


class ComputeUsage(StateModel):
    """Host-measured compute use (upstream ``evaluation_usage.compute_usage``)."""

    wall_clock_limit_seconds: float
    elapsed_seconds: float
    max_rpc_calls: int
    rpc_calls: int | None
    outcome: Literal["completed", "timeout", "rpc_limit", "failed"]
    objective_counts: Literal["advisory"] = "advisory"


class EvaluationRequest(StateModel):
    """A request to evaluate one candidate program.

    Attributes:
        evaluation_id: Deterministic id, for example ``"<run>/w0/c003/program_1"``.
        plane: The evaluation plane.
        program: The program to evaluate, in the run's workspace; the evaluator copies or mounts
            it.
        task: The task the program is evaluated on.
        evaluator: Which task evaluator to run: ``search`` (``evaluate.py``) or ``final``
            (``evaluate_final.py``; the drafting smoke check runs both).
        num_replications: Replications passed to the evaluator (``--num_replications``).
        timeout_seconds: Wall-clock bound of the evaluation; ``None`` lets the evaluator decide.
        environment: Extra environment variables, for example ``AGENTIC_TASK_DRAFT_SMOKE=1``.
    """

    evaluation_id: str
    plane: EvaluationPlane = "search"
    program: ProgramRef
    task: TaskRef
    evaluator: Literal["search", "final"] = "search"
    num_replications: int = Field(default=1, ge=1)
    timeout_seconds: float | None = None
    environment: dict[str, str] = Field(default_factory=dict)


class EvaluationResult(StateModel):
    """The result of an :class:`EvaluationRequest`.

    Attributes:
        status: ``completed`` when the evaluator ran to the end and its summaries are valid.
        metrics: ``metrics.json`` of a completed evaluation (JSON values; ``combined_score`` is
            a finite number).
        correct: ``correct.json`` of a completed evaluation.
        feasible: ``qualification.json`` of a qualification run.
        evidence: The validated ``evaluation_evidence.json`` when the evaluator produced one.
        failure_category: For example ``evaluator_output_contract_invalid`` or
            ``output_limit_exceeded``.
        summary: Bounded diagnostic shown to agents.
        output_dir: Where the evaluator kept its own outputs, for inspection.
    """

    evaluation_id: str
    status: EvaluationStatus
    metrics: dict[str, Any] = Field(default_factory=dict)
    correct: bool | None = None
    feasible: bool | None = None
    evidence: EvaluationEvidence | None = None
    failure_category: str | None = None
    summary: str = ""
    output_dir: str | None = None
    duration_seconds: float | None = None
    compute_usage: ComputeUsage | None = None
    returncode: int | None = None


class SealedInputs(StateModel):
    """An opaque reference to the inputs drawn for the sealed final evaluation.

    Produced by the ``final_evaluation.draw_inputs`` slot after the candidate is frozen, stored
    in the :class:`FreezeRecord`, and passed to the :class:`FinalEvaluator`.
    """

    ref: str
    digest: str | None = None
    drawn_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FreezeRecord(StateModel):
    """The frozen incumbent of a sealed final evaluation and the digests that bind it."""

    program: ProgramRef
    program_tree_sha256: str
    task_tree_sha256: str
    evaluator_sha256: str
    frozen_at: str
    sealed_inputs: SealedInputs | None = None


class FinalEvaluationRequest(StateModel):
    """A request to run the sealed final evaluation of a frozen candidate."""

    evaluation_id: str
    frozen: FreezeRecord
    task: TaskRef
    sealed_inputs: SealedInputs | None = None
    timeout_seconds: float | None = None


class FinalEvaluationResult(StateModel):
    """The verified result of a sealed final evaluation; mirrors final manifest schema 3."""

    status: Literal["completed"] = "completed"
    correct: bool
    outcome: Literal["qualified", "not_qualified"]
    metrics: dict[str, float]
    declared_final_metrics: list[str]
    task_tree_sha256: str
    evaluator_sha256: str
    program_tree_sha256: str
    results_tree_sha256: str
    started_at: str
    completed_at: str
    evaluation_evidence: EvidenceMetadata | None = None
    compute_usage: ComputeUsage | None = None
    manifest_schema_version: int = 3
    output_dir: str | None = None


@runtime_checkable
class Evaluator(Protocol):
    """Evaluates candidate programs; see the module docstring for the contract."""

    @property
    def isolated(self) -> bool:
        """True when the candidate runs isolated from evaluator files and from the host."""
        ...

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate ``request.program``, or return the recorded result for a known id."""
        ...


@runtime_checkable
class FinalEvaluator(Protocol):
    """Runs the sealed final evaluation once per id; see the module docstring for the contract."""

    @property
    def isolated(self) -> bool:
        """True when the candidate runs isolated from evaluator files and from the host."""
        ...

    async def evaluate_sealed(self, request: FinalEvaluationRequest) -> FinalEvaluationResult:
        """Run the sealed evaluation, or return the recorded result for a known id.

        Raises:
            FinalEvaluationAlreadyStarted: When the id started and its result cannot be obtained.
        """
        ...


def result_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    """Return the numeric entries of a metrics mapping as floats (booleans excluded)."""
    return {
        str(key): float(value)
        for key, value in values.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


__all__ = [
    "ComputeUsage",
    "EvaluationEvidence",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationStatus",
    "Evaluator",
    "EvidenceEpisode",
    "EvidenceMetadata",
    "FinalEvaluationRequest",
    "FinalEvaluationResult",
    "FinalEvaluator",
    "FreezeRecord",
    "SealedInputs",
    "result_metrics",
]
