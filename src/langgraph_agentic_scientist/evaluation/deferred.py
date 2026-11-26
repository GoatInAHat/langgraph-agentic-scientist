"""Evaluators that hand each evaluation to the caller through ``interrupt``.

An orchestrator that runs evaluations in its own executor (for hours, across process restarts)
uses :class:`DeferredEvaluator`: the evaluating node calls
``interrupt({"kind": "evaluation_request", "request": request.model_dump(mode="json")})``, the run
stops with that payload, and the orchestrator resumes it with
``Command(resume=result)`` (or ``Command(resume={interrupt_id: result, ...})`` when a fan-out has
several pending evaluations). The resume value is validated as
:class:`~langgraph_agentic_scientist.evaluation.EvaluationResult`.

A node that evaluates through a deferred evaluator restarts from its beginning when it is
resumed, so it must do nothing with an effect before the call (design rule D6); the package's
nodes follow that rule.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from .protocol import (
    EvaluationRequest,
    EvaluationResult,
    FinalEvaluationRequest,
    FinalEvaluationResult,
)

EVALUATION_REQUEST_KIND = "evaluation_request"
FINAL_EVALUATION_REQUEST_KIND = "final_evaluation_request"


def _coerce(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


class DeferredEvaluator:
    """An :class:`~langgraph_agentic_scientist.evaluation.Evaluator` that defers to the caller.

    Args:
        isolated: What the evaluator reports as ``isolated``: whether the caller's executor runs
            candidates isolated from evaluator files and from the host.
    """

    def __init__(self, *, isolated: bool = True) -> None:
        self._isolated = isolated

    @property
    def isolated(self) -> bool:
        """Whether the caller's executor isolates candidates (as declared at construction)."""
        return self._isolated

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Interrupt the run with the request; return the result the caller resumes with."""
        answer = interrupt(
            {
                "kind": EVALUATION_REQUEST_KIND,
                "request": request.model_dump(mode="json", by_alias=True),
            }
        )
        return EvaluationResult.model_validate(_coerce(answer))


class DeferredFinalEvaluator:
    """A :class:`~langgraph_agentic_scientist.evaluation.FinalEvaluator` that defers to the caller.

    The interrupt payload is ``{"kind": "final_evaluation_request", "request": ...}`` and the
    resume value is validated as
    :class:`~langgraph_agentic_scientist.evaluation.FinalEvaluationResult`.
    """

    def __init__(self, *, isolated: bool = True) -> None:
        self._isolated = isolated

    @property
    def isolated(self) -> bool:
        """Whether the caller's executor isolates candidates (as declared at construction)."""
        return self._isolated

    async def evaluate_sealed(self, request: FinalEvaluationRequest) -> FinalEvaluationResult:
        """Interrupt the run with the request; return the result the caller resumes with."""
        answer = interrupt(
            {
                "kind": FINAL_EVALUATION_REQUEST_KIND,
                "request": request.model_dump(mode="json", by_alias=True),
            }
        )
        return FinalEvaluationResult.model_validate(_coerce(answer))


__all__ = [
    "EVALUATION_REQUEST_KIND",
    "FINAL_EVALUATION_REQUEST_KIND",
    "DeferredEvaluator",
    "DeferredFinalEvaluator",
]
