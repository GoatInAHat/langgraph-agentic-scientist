"""State records and graph schemas of the investigation phase.

Every pydantic model here derives from
:class:`~langgraph_agentic_scientist.core.state.StateModel`, so it is accepted by the strict
msgpack serializer once :func:`~langgraph_agentic_scientist.serde_allowlist` is given to the
checkpointer. The ``TypedDict`` classes are the input, state and output schemas of the graphs;
a replacement subgraph given through ``GraphOverrides.slots`` must accept the slot's input schema
and return its output schema.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import Field

from ..agents.memory import MemoryPreState
from ..core.results import (
    AttemptRecord,
    DownstreamSelection,
    IdeaRecord,
    InvestigationResult,
)
from ..core.state import (
    BestManifest,
    EvaluationRecord,
    Failure,
    IdeaBundle,
    ProgramRef,
    RunConfigRecord,
    StateModel,
    TaskRef,
    merge_dicts,
)
from ..evaluation.protocol import EvaluationResult


def collect_results(
    left: list[WorkerResult] | None, right: WorkerResult | list[WorkerResult] | None
) -> list[WorkerResult]:
    """Reducer of the workers' results: each worker returns one result, in parallel."""
    merged = list(left or [])
    if right is None:
        return merged
    merged.extend(right if isinstance(right, list) else [right])
    return merged


class OuterState(StateModel):
    """The campaign counters of one worker (upstream ``outer_state.json``).

    Attributes:
        current_cycle: The last cycle whose idea was accepted (set back by one when a cycle's
            ideation is exhausted, so that a resume retries the cycle).
        failure_streak: Consecutive cycles whose ideation failed.
        last_idea_id: The id of the last accepted idea.
        prune_chapter: The next chapter number of the pruning archive.
        chapter_start_cycle: The first cycle of the current chapter; the histories the agents
            see start there (pruning hides earlier ideas).
        agent_cycle_assignments: The model-pool index chosen for each cycle (keys are cycle
            numbers as strings), persisted so that a cycle keeps its model on resume.
    """

    current_cycle: int = 0
    failure_streak: int = 0
    last_idea_id: str | None = None
    prune_chapter: int = 0
    chapter_start_cycle: int = 1
    agent_cycle_assignments: dict[str, int] = Field(default_factory=dict)


class IdeaFailure(StateModel):
    """One failed idea-generation attempt (a line of ``idea_generation_failures.jsonl``)."""

    cycle: int
    generation_attempt: int
    reason: str
    detail: str = ""
    timed_out: bool = False


class CycleScratch(StateModel):
    """What a worker carries through one cycle.

    Attributes:
        index: The worker-local cycle number (1-based).
        global_cycle: The campaign-wide cycle number (``cycle_offset + index``).
        model_index: The model-pool index of the cycle, when the registry has a pool.
        idea: The accepted idea, once there is one.
        attempt_id: The attempt's id (``<idea_id>_attempt_01``), once there is an idea.
        memory_pre: The memory files' sizes after the pre-cycle pull.
        conclusions_sha256: Digest of the worker's own conclusions after the pull.
        memory_pushed: Whether the worker's delta of the cycle was pushed.
    """

    index: int
    global_cycle: int = 0
    model_index: int | None = None
    idea: IdeaRecord | None = None
    attempt_id: str | None = None
    memory_pre: MemoryPreState | None = None
    conclusions_sha256: str = ""
    memory_pushed: bool = False


class IdeationResult(StateModel):
    """What the ideation graph returns for one cycle.

    Attributes:
        status: ``accepted`` with a validated bundle, or ``exhausted`` when every generation
            attempt failed.
        bundle: The validated bundle (harness rubric rules applied).
        idea_workspace: Where the host wrote ``idea_bundle.json`` and ``target_rubrics.json``.
        failures: Every failed generation attempt of the cycle.
        attempts_used: How many generation attempts ran.
        failure: Why the cycle ended without an idea.
    """

    status: Literal["accepted", "exhausted"]
    bundle: IdeaBundle | None = None
    idea_workspace: str | None = None
    failures: list[IdeaFailure] = Field(default_factory=list)
    attempts_used: int = 0
    failure: Failure | None = None


class AttemptResult(StateModel):
    """What the attempt graph returns.

    Attributes:
        record: The attempt record (one line of ``attempts_history.jsonl``).
        evaluations: The host records of the attempt's candidate evaluations.
        terminal_failure: A host-owned terminal failure that ends the campaign
            (``evaluator_output_contract_invalid`` or ``agent_log_persistence_failed``).
        experiment_dir: The attempt's workspace directory.
        best_candidate: The best correct evaluated program of the attempt, with its score.
        best_score: ``combined_score`` of ``best_candidate``.
    """

    record: AttemptRecord
    evaluations: list[EvaluationRecord] = Field(default_factory=list)
    terminal_failure: Failure | None = None
    experiment_dir: str
    best_candidate: ProgramRef | None = None
    best_score: float | None = None


class WorkerPlan(StateModel):
    """One worker of a round.

    Attributes:
        worker_index: ``0`` for a single worker, ``1..N`` for parallel workers.
        results_dir: The worker's results root in the workspace.
        cycles: The cycles this worker runs in the round.
        cycle_offset: Cycles run by this campaign before the round (global numbering).
        seed: The program the worker starts from.
        round: The round number (0-based).
        share_memory: Whether the worker pulls and pushes shared memory through the store.
    """

    worker_index: int
    results_dir: str
    cycles: int
    cycle_offset: int = 0
    seed: ProgramRef
    round: int = 0
    share_memory: bool = False


class WorkerResult(StateModel):
    """What one worker returns to the investigation graph."""

    plan: WorkerPlan
    status: Literal["finished", "failed"]
    best: BestManifest | None = None
    baseline: EvaluationRecord | None = None
    ideas: list[IdeaRecord] = Field(default_factory=list)
    attempts: list[AttemptRecord] = Field(default_factory=list)
    evaluations: list[EvaluationRecord] = Field(default_factory=list)
    idea_failures: list[IdeaFailure] = Field(default_factory=list)
    outer: OuterState = Field(default_factory=OuterState)
    failure: Failure | None = None


# --- ideation graph ------------------------------------------------------------------------------


class IdeationInput(TypedDict):
    """Input of the ideation graph (and of the ``investigation.ideation`` slot)."""

    task: TaskRef
    run_id: str
    results_dir: str
    worker_index: int
    cycle: CycleScratch
    run_config: NotRequired[RunConfigRecord | None]


class IdeationOutput(TypedDict):
    """Output of the ideation graph."""

    ideation: IdeationResult


class IdeationState(TypedDict, total=False):
    """State of the ideation graph."""

    task: TaskRef
    run_id: str
    results_dir: str
    worker_index: int
    cycle: CycleScratch
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    generation_attempt: int
    validation_feedback: str
    staging_dir: str
    prompt_values: dict[str, str]
    session_failed: bool
    session_detail: str
    timed_out: bool
    draft: dict[str, Any] | None
    failures: Annotated[list[IdeaFailure], operator.add]
    ideation: IdeationResult


# --- attempt graph --------------------------------------------------------------------------------


class AttemptInput(TypedDict):
    """Input of the attempt graph (and of the ``investigation.attempt`` slot)."""

    task: TaskRef
    run_id: str
    results_dir: str
    worker_index: int
    cycle: CycleScratch
    seed_program: ProgramRef
    run_config: NotRequired[RunConfigRecord | None]


class AttemptOutput(TypedDict):
    """Output of the attempt graph."""

    attempt_result: AttemptResult


class AttemptState(TypedDict, total=False):
    """State of the attempt graph."""

    task: TaskRef
    run_id: str
    results_dir: str
    worker_index: int
    cycle: CycleScratch
    seed_program: ProgramRef
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    harness: bool
    experiment_dir: str
    attempt_id: str
    idea_id: str
    shared_pre: dict[str, int]
    prompt_values: dict[str, str]
    timed_out: bool
    session_failure: Failure | None
    check_error: str | None
    evaluation: EvaluationResult | None
    session_evaluations: list[EvaluationResult]
    terminal_failure: Failure | None
    attempt_result: AttemptResult


# --- worker graph ---------------------------------------------------------------------------------


class WorkerInput(TypedDict):
    """Input of the campaign worker graph (and of the ``investigation.worker`` slot).

    Only ``task`` and ``run_id`` are required: a worker run alone starts at cycle 1 of
    ``cycles`` (default ``max_idea_cycles``) in ``<run>/results`` from the task's seed program.
    """

    task: TaskRef
    run_id: str
    run_config: NotRequired[RunConfigRecord | None]
    worker_index: NotRequired[int]
    results_dir: NotRequired[str]
    cycles: NotRequired[int]
    cycle_offset: NotRequired[int]
    seed: NotRequired[ProgramRef]
    seed_best: NotRequired[BestManifest | None]
    round: NotRequired[int]
    share_memory: NotRequired[bool]
    prune_confirmed: NotRequired[bool]


class WorkerOutput(TypedDict):
    """Output of the campaign worker graph."""

    worker_result: WorkerResult


class WorkerState(TypedDict, total=False):
    """State of the campaign worker graph."""

    task: TaskRef
    run_id: str
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    worker_index: int
    results_dir: str
    cycles: int
    cycle_offset: int
    seed: ProgramRef
    seed_best: BestManifest | None
    round: int
    share_memory: bool
    prune_confirmed: bool
    outer: OuterState
    best: BestManifest | None
    baseline: EvaluationRecord | None
    cycle: CycleScratch | None
    seed_program: ProgramRef
    ideation: IdeationResult | None
    attempt_result: AttemptResult | None
    ideas_history: Annotated[list[IdeaRecord], operator.add]
    attempts_history: Annotated[list[AttemptRecord], operator.add]
    idea_failures: Annotated[list[IdeaFailure], operator.add]
    evaluations: Annotated[list[EvaluationRecord], operator.add]
    status: Literal["running", "finished", "failed"]
    failure: Failure | None
    worker_result: WorkerResult


# --- investigation graph --------------------------------------------------


class InvestigationInput(TypedDict):
    """Input of the investigation graph."""

    task: TaskRef
    run_id: str
    run_config: NotRequired[RunConfigRecord | None]


class InvestigationOutput(TypedDict):
    """Output of the investigation graph."""

    investigation: InvestigationResult


class WorkerSend(TypedDict):
    """The payload of the ``Send`` to a worker: the worker graph's input."""

    task: TaskRef
    run_id: str
    run_config: RunConfigRecord | None
    worker_index: int
    results_dir: str
    cycles: int
    cycle_offset: int
    seed: ProgramRef
    seed_best: BestManifest | None
    round: int
    share_memory: bool
    prune_confirmed: bool


class InvestigationState(TypedDict, total=False):
    """State of the investigation graph."""

    task: TaskRef
    run_id: str
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    results_dir: str
    round: int
    rounds: int
    prune_confirmed: bool
    cycles_done: int
    worker_plans: list[WorkerPlan]
    worker_result: Annotated[list[WorkerResult], collect_results]
    seeds: dict[str, ProgramRef]
    seed_bests: dict[str, BestManifest]
    stopped: bool
    investigation: InvestigationResult


# --- downstream selection --------------------------------------------------


class DownstreamInput(TypedDict):
    """Input of the downstream selection graph."""

    task: TaskRef
    run_id: str
    investigation: InvestigationResult
    run_config: NotRequired[RunConfigRecord | None]


class DownstreamOutput(TypedDict):
    """Output of the downstream selection graph."""

    downstream: DownstreamSelection


class DownstreamState(TypedDict, total=False):
    """State of the downstream selection graph."""

    task: TaskRef
    run_id: str
    investigation: InvestigationResult
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    proposal: DownstreamSelection | None
    downstream: DownstreamSelection


__all__ = [
    "AttemptInput",
    "AttemptOutput",
    "AttemptResult",
    "AttemptState",
    "CycleScratch",
    "DownstreamInput",
    "DownstreamOutput",
    "DownstreamState",
    "IdeaFailure",
    "IdeationInput",
    "IdeationOutput",
    "IdeationResult",
    "IdeationState",
    "InvestigationInput",
    "InvestigationOutput",
    "InvestigationState",
    "OuterState",
    "WorkerInput",
    "WorkerOutput",
    "WorkerPlan",
    "WorkerResult",
    "WorkerSend",
    "WorkerState",
    "collect_results",
]
