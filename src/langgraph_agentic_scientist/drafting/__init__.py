"""Task drafting: the design loop, the scaffold loop and the gates (upstream phase A).

``build_task_drafting_graph`` runs the design loop and then the scaffold loop;
``build_task_design_graph`` (A1) and ``build_task_scaffold_graph`` (A2-A7) are usable alone.
``TaskGate`` is the protocol of the drafting gates and ``LocalTaskGate`` its default
implementation over an ``Evaluator``.

The records of the scaffold loop are in :mod:`~langgraph_agentic_scientist.drafting.models`, the
gate and the contract lint in :mod:`~langgraph_agentic_scientist.drafting.gate` and
:mod:`~langgraph_agentic_scientist.drafting.contract_lint`, the host-side handling of the
scaffold in :mod:`~langgraph_agentic_scientist.drafting.scaffold`, and the qualification
protocol in :mod:`~langgraph_agentic_scientist.drafting.qualification`. The design loop's records
and pure functions are in :mod:`~langgraph_agentic_scientist.drafting.design_models` and
:mod:`~langgraph_agentic_scientist.drafting.design`.
"""

from ..core.interfaces import GateCheck, GateReport, TaskGate
from .gate import LocalTaskGate, RepairSessionGate
from .graph import DraftingInput, DraftingOutput, DraftingState, build_task_drafting_graph
from .graph_design import build_task_design_graph
from .graph_scaffold import (
    SCAFFOLD_STEPS,
    ScaffoldInput,
    ScaffoldOutput,
    ScaffoldState,
    build_task_scaffold_graph,
)

__all__ = [
    "SCAFFOLD_STEPS",
    "DraftingInput",
    "DraftingOutput",
    "DraftingState",
    "GateCheck",
    "GateReport",
    "LocalTaskGate",
    "RepairSessionGate",
    "ScaffoldInput",
    "ScaffoldOutput",
    "ScaffoldState",
    "TaskGate",
    "build_task_design_graph",
    "build_task_drafting_graph",
    "build_task_scaffold_graph",
]
