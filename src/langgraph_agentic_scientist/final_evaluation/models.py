"""State schemas of the sealed final evaluation graph."""

from __future__ import annotations

from typing import Annotated, Any, Literal, NotRequired, TypedDict

from ..core.results import DownstreamSelection, InvestigationResult
from ..core.state import (
    BestManifest,
    Event,
    Failure,
    ProgramRef,
    RunConfigRecord,
    TaskRef,
    append,
    merge_dicts,
)
from ..evaluation.protocol import FinalEvaluationResult, FreezeRecord, SealedInputs

FinalStatus = Literal["pending", "completed", "failed"]
"""``completed`` when the verified result is in state; ``failed`` ends the phase with no rerun."""


class FinalEvaluationInput(TypedDict):
    """Input of the final evaluation graph.

    Attributes:
        task: The run-local task snapshot.
        run_id: The run id; the evaluation id is ``<run_id>/final``.
        run_config: The parent's recorded configuration when mounted.
        investigation: The investigation result; its best program is the default incumbent.
        downstream: The no-winner fallback selection, which wins over the best program.
        incumbent: An explicit incumbent, which wins over both.
        incumbent_manifest: Score and provenance of an explicit incumbent.
    """

    task: TaskRef
    run_id: str
    run_config: NotRequired[RunConfigRecord | None]
    investigation: NotRequired[InvestigationResult | None]
    downstream: NotRequired[DownstreamSelection | None]
    incumbent: NotRequired[ProgramRef | None]
    incumbent_manifest: NotRequired[BestManifest | None]


class FinalEvaluationState(TypedDict, total=False):
    """State of the final evaluation graph."""

    task: TaskRef
    run_id: str
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    investigation: InvestigationResult | None
    downstream: DownstreamSelection | None
    incumbent: ProgramRef | None
    incumbent_manifest: BestManifest | None
    resolved: ProgramRef | None
    frozen: FreezeRecord | None
    sealed_inputs: SealedInputs | None
    final: FinalEvaluationResult | None
    final_status: FinalStatus
    failure: Failure | None
    events: Annotated[list[Event], append]


class FinalEvaluationOutput(TypedDict, total=False):
    """Output of the final evaluation graph.

    ``final`` is the verified result (``None`` when the phase failed), ``frozen`` the freeze
    record with the sealed inputs, ``final_status`` the phase status and ``failure`` why it
    failed. A failed phase is not rerun: the sealed evaluation runs at most once.
    """

    final: FinalEvaluationResult | None
    frozen: FreezeRecord | None
    final_status: FinalStatus
    failure: Failure | None
    events: list[Event]


class DrawInputsInput(TypedDict):
    """Input of the ``final_evaluation.draw_inputs`` slot: what is frozen, nothing more."""

    run_id: str
    task: TaskRef
    frozen: FreezeRecord


class DrawInputsOutput(TypedDict):
    """Output of the slot: the opaque reference to the inputs drawn after the freeze.

    ``None`` means the task's own ``resources/final/**``, as upstream uses them.
    """

    sealed_inputs: SealedInputs | None


__all__ = [
    "DrawInputsInput",
    "DrawInputsOutput",
    "FinalEvaluationInput",
    "FinalEvaluationOutput",
    "FinalEvaluationState",
    "FinalStatus",
]
