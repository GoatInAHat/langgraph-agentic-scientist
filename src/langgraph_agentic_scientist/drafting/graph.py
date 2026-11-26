"""The task drafting graph: the design loop, then the scaffold loop (upstream phase A).

:func:`build_task_drafting_graph` composes the two graphs of the package:

``design`` (the replaceable slot ``drafting.design``)
    the task design loop of :func:`~langgraph_agentic_scientist.drafting.build_task_design_graph`
    (upstream A1) turns ``task.md`` into an accepted design before any scaffold exists. A design
    that fails ends drafting;
``scaffold`` (the replaceable slot ``drafting.scaffold``)
    :func:`~langgraph_agentic_scientist.drafting.build_task_scaffold_graph` (A2-A7) drafts,
    validates, aligns, qualifies and publishes the scaffold.

With ``DraftingConfig.workflow == "legacy"`` the design loop is skipped and the scaffold graph
shares one repair pool (``legacy_validation_max_repairs``), as upstream's ``--workflow legacy``
does.

A caller can replace either slot through
:class:`~langgraph_agentic_scientist.GraphOverrides` (``slots={"drafting.design": ...}``). A
replacement of the design slot receives the keys ``task``, ``capacity``, ``design_dir`` and
``run_config`` and must return ``accepted_design`` and ``accepted_review`` (JSON objects), or
``design_failure``; a replacement of the scaffold slot is a compiled graph with the scaffold graph's
input and output schemas.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.config import get_config
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..context import GraphOverrides, ScientistContext
from ..core.results import DraftingResult
from ..core.runconfig import method_config, record_run_config
from ..core.state import Failure, RunConfigRecord, TaskRef, merge_dicts
from ..core.steps import StepGraph
from .design_models import Capacity
from .graph_design import build_task_design_graph
from .graph_scaffold import PHASE, build_task_scaffold_graph
from .scaffold import draft_paths

STEP_DESIGN = "design"
STEP_SCAFFOLD = "scaffold"
DESIGN_SLOT = f"{PHASE}.{STEP_DESIGN}"
SCAFFOLD_SLOT = f"{PHASE}.{STEP_SCAFFOLD}"


class DraftingInput(TypedDict):
    """Input of the drafting graph.

    Attributes:
        task: The run-local task snapshot; its ``task.md`` has a ``Goal`` and a ``Rubrics`` section.
        capacity: The investigation capacity a design must fit (the configured investigation
            limits when absent).
        run_config: The recorded configuration of a parent run, when mounted.
    """

    task: TaskRef
    capacity: NotRequired[Capacity]
    run_config: NotRequired[RunConfigRecord | None]


class DraftingOutput(TypedDict):
    """Output of the drafting graph: the result (status, published task, receipts or failure)."""

    drafting: DraftingResult


class DraftingState(TypedDict, total=False):
    """State of the drafting graph: the shared keys of the design and scaffold steps."""

    task: TaskRef
    capacity: Capacity
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    accepted_design: dict[str, Any] | None
    accepted_review: dict[str, Any] | None
    design_failure: Failure | None
    drafting: DraftingResult


def _payload(value: Any) -> dict[str, Any] | None:
    """Return a design or review as a JSON object."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    payload = getattr(value, "payload", None)
    if callable(payload):
        result = payload()
        return dict(result) if isinstance(result, Mapping) else None
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dict(dump(mode="json", by_alias=True))
    return None


def _design_input(state: Mapping[str, Any]) -> dict[str, Any]:
    task: TaskRef = state["task"]
    value: dict[str, Any] = {
        "task": task,
        "design_dir": draft_paths(task.root).design_dir,
        "run_config": state.get("run_config"),
    }
    if state.get("capacity") is not None:
        value["capacity"] = state["capacity"]
    return value


def design_runner(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> Callable[[Mapping[str, Any], Runtime[ScientistContext]], Awaitable[dict[str, Any]]]:
    """Return the default of the design slot: run ``graph`` (the task design graph) in the node.

    The graph is built once by :func:`build_task_drafting_graph`, with the drafting graph's
    overrides, and invoked with the node's config, so that it runs as a subgraph of the node
    (checkpointed under its namespace). The design graph writes the accepted design to
    ``<drafting dir>/_task_design/task_design.json``, where the drafter's prompt says it is; its
    result is mapped to ``accepted_design``, ``accepted_review`` and ``design_failure``.
    """

    async def run_design_graph(
        value: Mapping[str, Any], runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        design_input: Any = dict(value)
        return dict(await graph.ainvoke(design_input, get_config()))

    return run_design_graph


def _design_output(result: Mapping[str, Any]) -> dict[str, Any]:
    if "design_failure" in result or "accepted_design" in result:
        return {
            key: result[key]
            for key in ("accepted_design", "accepted_review", "design_failure")
            if key in result
        }
    if result.get("status") == "failed" or result.get("design") is None:
        failure = result.get("failure")
        return {
            "design_failure": failure
            if isinstance(failure, Failure)
            else Failure(category="design_failed", scope="task", detail="no accepted design")
        }
    return {
        "accepted_design": _payload(result.get("design")),
        "accepted_review": _payload(result.get("accepted_review")),
        "design_failure": None,
    }


def _route_start(state: DraftingState) -> str:
    return "scaffold" if method_config(state).drafting.workflow == "legacy" else "design"


def _route_design(state: DraftingState) -> str:
    return "design_failed" if state.get("design_failure") is not None else "scaffold"


async def design_failed(state: DraftingState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """End drafting because the task design loop failed."""
    failure = state.get("design_failure") or Failure(category="design_failed", scope="task")
    return {"drafting": DraftingResult(status="failed", failure=failure)}


def build_task_drafting_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "task_drafting",
) -> CompiledStateGraph[DraftingState, ScientistContext, DraftingInput, DraftingOutput]:
    """Build the task drafting graph (upstream phase A).

    The graph runs the design loop (unless the workflow is ``legacy``) and then the scaffold loop,
    and returns a :class:`~langgraph_agentic_scientist.core.results.DraftingResult`: ``published``
    with the task reference of the published scaffold (its contract loaded), its fingerprint and
    its receipts, or ``failed`` with the failure. It can run alone or be mounted as the
    ``task_drafting`` phase of the pipeline.

    Args:
        checkpointer: The caller's checkpointer; ``None`` inherits the parent's when mounted.
        store: The caller's store; only a top-level caller passes one.
        overrides: Replacement graphs: the slots ``drafting.design`` and ``drafting.scaffold``;
            the same overrides reach the scaffold graph's own steps.
        name: The compiled graph's name.

    Returns:
        A compiled graph with input :class:`DraftingInput`, state :class:`DraftingState` and
        output :class:`DraftingOutput`. A run takes many supersteps; give it a high
        ``recursion_limit``. The drafting limits are read from the recorded configuration
        (``ScientistConfig.drafting``).
    """
    g = StepGraph(
        PHASE,
        DraftingState,
        input_schema=DraftingInput,
        output_schema=DraftingOutput,
        overrides=overrides,
    )
    g.add_node("record_run_config", record_run_config)
    g.slot(
        STEP_DESIGN,
        design_runner(build_task_design_graph(overrides=overrides)),
        input_mapper=_design_input,
        output_mapper=_design_output,
    )
    g.slot(STEP_SCAFFOLD, build_task_scaffold_graph(overrides=overrides))
    g.add_node("design_failed", design_failed)

    g.add_edge(START, "record_run_config")
    g.add_conditional_edges(
        "record_run_config",
        _route_start,
        {"design": STEP_DESIGN, "scaffold": STEP_SCAFFOLD},
    )
    g.add_conditional_edges(
        STEP_DESIGN,
        _route_design,
        {"design_failed": "design_failed", "scaffold": STEP_SCAFFOLD},
    )
    g.add_edge(STEP_SCAFFOLD, END)
    g.add_edge("design_failed", END)
    return g.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "DESIGN_SLOT",
    "SCAFFOLD_SLOT",
    "DraftingInput",
    "DraftingOutput",
    "DraftingState",
    "build_task_drafting_graph",
    "design_runner",
]
