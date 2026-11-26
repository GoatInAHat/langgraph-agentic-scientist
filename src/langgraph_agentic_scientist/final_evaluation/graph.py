"""The sealed final evaluation as a graph (upstream ``final_evaluation/runner.py``, host rules).

The nodes, in order (each step carries hook nodes ``final_evaluation.<step>.before/.after``):

``record_run_config`` -> ``resolve_incumbent`` -> ``freeze`` -> ``draw_inputs`` ->
``attach_inputs`` -> ``evaluate_sealed`` -> ``verify_manifest``

* ``resolve_incumbent`` checks that a final evaluator is configured and, for a contracted task,
  that it isolates the candidate (upstream refuses to evaluate a contracted task without
  container isolation; ``config.allow_unisolated_contracted`` lifts this and is recorded as an
  event), then resolves the frozen incumbent: a downstream selection first, else the best
  program.
* ``freeze`` digests the program, the task scaffold and the final evaluator. The
  :class:`~langgraph_agentic_scientist.evaluation.FreezeRecord` is in state when the node
  completes, so a resumed run never freezes again.
* ``draw_inputs`` is the replaceable slot ``final_evaluation.draw_inputs``: it receives the freeze
  record and returns :class:`~langgraph_agentic_scientist.evaluation.SealedInputs` (or ``None``,
  the default: the task's own ``resources/final/**``). It is a node of its own, so a completed
  draw is checkpointed and never repeated, also when ``evaluate_sealed`` is interrupted and
  resumed; an implementation that can itself be interrupted while drawing keys its draw by the
  evaluation id.
* ``evaluate_sealed`` calls ``ctx.final_evaluator.evaluate_sealed`` under the id ``<run>/final``.
  Nothing with an effect precedes the call in the node, so a node restarted after an interrupt
  asks the evaluator again under the same id. The once-only rule is the evaluator's: for an id that
  started and cannot be obtained it raises
  :class:`~langgraph_agentic_scientist.core.errors.FinalEvaluationAlreadyStarted`, which ends the
  phase ``failed`` with no rerun.
* ``verify_manifest`` applies the manifest rules of upstream's ``verify_final_manifest`` to the
  result (:func:`~langgraph_agentic_scientist.final_evaluation.manifest.verify_final_result`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..context import GraphOverrides, ScientistContext
from ..core.errors import (
    FinalEvaluationAlreadyStarted,
    FinalEvaluationError,
    FinalEvaluationInfrastructureError,
)
from ..core.runconfig import method_config, record_run_config
from ..core.state import Event, Failure, FailureScope
from ..core.steps import StepGraph
from ..evaluation.ids import final_id
from ..evaluation.protocol import FinalEvaluationRequest, SealedInputs
from .manifest import freeze_incumbent, resolve_incumbent, verify_final_result
from .models import (
    DrawInputsInput,
    DrawInputsOutput,
    FinalEvaluationInput,
    FinalEvaluationOutput,
    FinalEvaluationState,
)

PHASE = "final_evaluation"
DRAW_INPUTS_SLOT = "final_evaluation.draw_inputs"
ALREADY_STARTED_MESSAGE = "final evaluation already started; refusing rerun"


def _failed(category: str, detail: str, scope: FailureScope = "method") -> dict[str, Any]:
    """Return the state update that ends the phase failed."""
    return {
        "final_status": "failed",
        "failure": Failure(category=category, scope=scope, detail=detail),
        "events": [
            Event(
                kind="phase_end",
                phase=PHASE,
                data={"status": "failed", "category": category, "detail": detail},
            )
        ],
    }


async def _resolve_incumbent(
    state: FinalEvaluationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    ctx = runtime.context
    task = state["task"]
    evaluator = ctx.final_evaluator
    if evaluator is None:
        return _failed(
            "final_evaluator_missing", "the context has no final_evaluator", scope="task"
        )
    events: list[Event] = []
    if task.contract is not None and not evaluator.isolated:
        if not method_config(state).allow_unisolated_contracted:
            return _failed(
                "final_evaluation_requires_isolation",
                "contracted final evaluation requires container isolation",
                scope="task",
            )
        events.append(
            Event(
                kind="unisolated_contracted_final_evaluation",
                phase=PHASE,
                data={"allow_unisolated_contracted": True},
            )
        )
    try:
        incumbent = await resolve_incumbent(
            ctx.workspace,
            incumbent=state.get("incumbent"),
            incumbent_manifest=state.get("incumbent_manifest"),
            downstream=state.get("downstream"),
            investigation=state.get("investigation"),
        )
    except FinalEvaluationError as exc:
        return _failed("frozen_incumbent_unavailable", str(exc))
    events.append(
        Event(
            kind="final_incumbent_resolved",
            phase=PHASE,
            data={"program": incumbent.program.path, "source": incumbent.source},
        )
    )
    return {"resolved": incumbent.program, "events": events}


async def _freeze(
    state: FinalEvaluationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    program = state.get("resolved")
    if program is None:
        return _failed("frozen_incumbent_unavailable", "no incumbent was resolved")
    try:
        frozen = await freeze_incumbent(runtime.context.workspace, state["task"], program)
    except (FinalEvaluationError, OSError) as exc:
        return _failed("freeze_failed", str(exc), scope="task")
    return {
        "frozen": frozen,
        "events": [
            Event(
                kind="final_incumbent_frozen",
                phase=PHASE,
                data={
                    "program": frozen.program.path,
                    "program_tree_sha256": frozen.program_tree_sha256,
                },
            )
        ],
    }


async def _default_draw_inputs(
    value: DrawInputsInput, runtime: Runtime[ScientistContext]
) -> DrawInputsOutput:
    """The default draw: no sealed inputs, so the task's own ``resources/final/**`` are used."""
    return {"sealed_inputs": None}


def _draw_input(state: Mapping[str, Any]) -> DrawInputsInput:
    return {"run_id": state["run_id"], "task": state["task"], "frozen": state["frozen"]}


def _draw_output(result: Any) -> Mapping[str, Any]:
    if result is None or isinstance(result, SealedInputs):
        return {"sealed_inputs": result}
    if isinstance(result, Mapping):
        return {"sealed_inputs": result.get("sealed_inputs")}
    raise TypeError("the draw_inputs slot must return SealedInputs, None or {'sealed_inputs': ...}")


async def _attach_inputs(
    state: FinalEvaluationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    frozen = state["frozen"]
    assert frozen is not None
    sealed = state.get("sealed_inputs")
    update: dict[str, Any] = {"frozen": frozen.model_copy(update={"sealed_inputs": sealed})}
    if sealed is not None:
        update["events"] = [
            Event(
                kind="final_inputs_drawn",
                phase=PHASE,
                data={"ref": sealed.ref, "digest": sealed.digest, "drawn_at": sealed.drawn_at},
            )
        ]
    return update


async def _evaluate_sealed(
    state: FinalEvaluationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    frozen = state["frozen"]
    evaluator = runtime.context.final_evaluator
    assert frozen is not None and evaluator is not None
    request = FinalEvaluationRequest(
        evaluation_id=final_id(state["run_id"]),
        frozen=frozen,
        task=state["task"],
        sealed_inputs=frozen.sealed_inputs,
        timeout_seconds=method_config(state).final_evaluation.timeout_seconds,
    )
    try:
        result = await evaluator.evaluate_sealed(request)
    except FinalEvaluationAlreadyStarted as exc:
        return _failed("final_evaluation_already_started", f"{ALREADY_STARTED_MESSAGE}: {exc}")
    except FinalEvaluationInfrastructureError as exc:
        return _failed(
            "final_evaluation_infrastructure_error",
            f"final evaluation infrastructure failure: {exc}",
            scope="infrastructure",
        )
    except FinalEvaluationError as exc:
        return _failed("final_evaluation_failed", f"final evaluation failed: {exc}")
    return {"final": result}


async def _verify_manifest(
    state: FinalEvaluationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    result = state.get("final")
    frozen = state["frozen"]
    assert result is not None and frozen is not None
    try:
        verified = await verify_final_result(
            runtime.context.workspace, task=state["task"], frozen=frozen, result=result
        )
    except (FinalEvaluationError, OSError) as exc:
        update = _failed("final_manifest_invalid", str(exc))
        update["final"] = None
        return update
    return {
        "final": verified,
        "final_status": "completed",
        "events": [
            Event(
                kind="phase_end",
                phase=PHASE,
                data={"status": "completed", "outcome": verified.outcome},
            )
        ],
    }


def _route(state: FinalEvaluationState) -> str:
    return "stop" if state.get("final_status") == "failed" else "continue"


def build_final_evaluation_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "final_evaluation",
) -> CompiledStateGraph[
    FinalEvaluationState, ScientistContext, FinalEvaluationInput, FinalEvaluationOutput
]:
    """Build the sealed final evaluation graph (usable alone or mounted as a subgraph).

    The graph freezes an incumbent, draws the sealed inputs, runs the sealed evaluation once
    through ``ctx.final_evaluator`` and verifies the result. Replace the draw of the sealed
    inputs with ``overrides.slots["final_evaluation.draw_inputs"]`` (a compiled graph or an async
    callable ``(DrawInputsInput, runtime) -> DrawInputsOutput``).

    Args:
        checkpointer: The checkpointer when the graph runs at top level (``None`` inherits the
            parent's when mounted).
        store: The store when the graph runs at top level.
        overrides: Replacement slots.
        name: The compiled graph's name.

    Returns:
        A compiled graph whose input is :class:`FinalEvaluationInput` and whose output is
        :class:`FinalEvaluationOutput`. Run it with ``ainvoke(..., context=ScientistContext(...))``.
    """
    sg = StepGraph(
        PHASE,
        FinalEvaluationState,
        input_schema=FinalEvaluationInput,
        output_schema=FinalEvaluationOutput,
        overrides=overrides,
    )
    sg.add_node("record_run_config", record_run_config)
    sg.step("resolve_incumbent", _resolve_incumbent)
    sg.step("freeze", _freeze)
    sg.slot(
        "draw_inputs",
        _default_draw_inputs,
        key=DRAW_INPUTS_SLOT,
        input_mapper=_draw_input,
        output_mapper=_draw_output,
    )
    sg.add_node("attach_inputs", _attach_inputs)
    sg.step("evaluate_sealed", _evaluate_sealed)
    sg.step("verify_manifest", _verify_manifest)

    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "resolve_incumbent")
    sg.add_conditional_edges("resolve_incumbent", _route, {"continue": "freeze", "stop": END})
    sg.add_conditional_edges("freeze", _route, {"continue": "draw_inputs", "stop": END})
    sg.add_edge("draw_inputs", "attach_inputs")
    sg.add_edge("attach_inputs", "evaluate_sealed")
    sg.add_conditional_edges(
        "evaluate_sealed", _route, {"continue": "verify_manifest", "stop": END}
    )
    sg.add_edge("verify_manifest", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = ["DRAW_INPUTS_SLOT", "build_final_evaluation_graph"]
