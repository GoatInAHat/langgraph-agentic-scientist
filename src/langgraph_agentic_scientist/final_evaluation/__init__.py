"""The sealed final evaluation: freeze, draw inputs, evaluate once, verify."""

from .graph import DRAW_INPUTS_SLOT, build_final_evaluation_graph
from .manifest import (
    Incumbent,
    evaluator_digest,
    freeze_incumbent,
    resolve_incumbent,
    task_digest,
    verify_final_result,
)
from .models import (
    DrawInputsInput,
    DrawInputsOutput,
    FinalEvaluationInput,
    FinalEvaluationOutput,
    FinalEvaluationState,
)

__all__ = [
    "DRAW_INPUTS_SLOT",
    "DrawInputsInput",
    "DrawInputsOutput",
    "FinalEvaluationInput",
    "FinalEvaluationOutput",
    "FinalEvaluationState",
    "Incumbent",
    "build_final_evaluation_graph",
    "evaluator_digest",
    "freeze_incumbent",
    "resolve_incumbent",
    "task_digest",
    "verify_final_result",
]
