"""Stand-ins for the phase graphs: small compiled graphs with the phases' input and output keys.

The pipeline mounts a phase through ``GraphOverrides.phases``. A stub records every call in a
ledger, can pause once with ``interrupt`` (to test resume), can raise once, and returns the result
the test chose. Results are the package's own record types, so the stubs also run under strict
msgpack checkpoints.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from langgraph_agentic_scientist import (
    DownstreamSelection,
    DraftingResult,
    GraphOverrides,
    InvestigationResult,
    PaperResult,
    PosthocResult,
    ReviewResult,
    RevisionResult,
    ScientistContext,
    VariantsResult,
)
from langgraph_agentic_scientist.core.results import AnalysisResult
from langgraph_agentic_scientist.core.state import Failure, ProgramRef, TaskRef, merge_dicts
from langgraph_agentic_scientist.evaluation.protocol import FinalEvaluationResult

Result = Any


CONTRACT_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "core" / "contract_task"
"""A task with a contract that requires a final evaluation."""


class StubInput(TypedDict, total=False):
    """Every key the pipeline may hand a phase."""

    task: TaskRef
    run_id: str
    run_config: Any
    capacity: dict[str, int]
    investigation: Any
    downstream: Any
    variants: Any
    final: Any
    posthoc: Any
    paper: Any
    review: Any
    mode: str
    previous_paper_dir: str
    edit_options: Any
    pdf_path: str
    paper_dir: str
    artifact_stem: str
    manifest_key: str


class StubState(StubInput, total=False):
    result: Any
    drafting: Any
    revision: Any
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    log: Annotated[list[str], operator.add]


class StubOutput(TypedDict, total=False):
    result: Any
    drafting: Any
    investigation: Any
    downstream: Any
    variants: Any
    final: Any
    posthoc: Any
    paper: Any
    review: Any
    revision: Any
    hook_outputs: dict[str, Any]


@dataclass
class Ledger:
    """What the stubs saw: calls per phase, the input of each call, and one-shot switches."""

    calls: Counter[str] = field(default_factory=Counter)
    inputs: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)
    pause: set[str] = field(default_factory=set)
    crash: set[str] = field(default_factory=set)

    def last_input(self, phase: str) -> dict[str, Any]:
        return self.inputs[phase][-1]


def final_result(outcome: str = "qualified") -> FinalEvaluationResult:
    """A verified final evaluation result of the given outcome."""
    digest = "sha256:" + "0" * 64
    return FinalEvaluationResult(
        correct=outcome == "qualified",
        outcome=outcome,  # type: ignore[arg-type]
        metrics={"final_score": 1.0},
        declared_final_metrics=["final_score"],
        task_tree_sha256=digest,
        evaluator_sha256=digest,
        program_tree_sha256=digest,
        results_tree_sha256=digest,
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T00:00:01+00:00",
    )


def default_results() -> dict[str, Result | Callable[[Mapping[str, Any]], Result]]:
    """The successful result of every phase, keyed by phase name."""
    return {
        "task_drafting": lambda given: DraftingResult(status="published", task=given.get("task")),
        "investigation": InvestigationResult(status="completed", results_dir="investigation"),
        "downstream_selection": DownstreamSelection(
            status="selected",
            program=ProgramRef(path="downstream/program"),
            source="candidate",
            reason="best valid candidate",
        ),
        "ablations": VariantsResult(status="completed", output_dir="ablations"),
        "final_evaluation": final_result("qualified"),
        "posthoc": PosthocResult(
            status="completed",
            output_dir="posthoc",
            analyses=[AnalysisResult(analysis_id="a1", status="passed", attempts=1)],
        ),
        "paper": PaperResult(
            status="complete",
            paper_dir="paper",
            pdf_path="paper/paper.pdf",
            tex_path="paper/paper.tex",
        ),
        "review": ReviewResult(status="completed", review_path="paper/ai_review.json"),
        "revision": RevisionResult(status="promoted", before_dir="paper/revisions/revision_01"),
    }


def failed_investigation(category: str = "campaign_failed", scope: str = "method") -> Result:
    """A failed investigation result."""
    return InvestigationResult(
        status="failed",
        failure=Failure(category=category, scope=scope),  # type: ignore[arg-type]
    )


def stub_phase(
    name: str,
    ledger: Ledger,
    result: Result | Callable[[Mapping[str, Any]], Result],
    *,
    narrow: bool = False,
    state_key_output: str | None = None,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build a stub phase graph.

    The graph has two nodes. ``work`` counts the call and may crash once (``ledger.crash``);
    ``ask`` may interrupt once (``ledger.pause``), so that a test can see whether ``work`` runs
    again on resume. ``narrow=True`` gives the graph an input schema of ``task`` only.
    ``state_key_output`` returns the result under that key instead of ``result``.
    """

    class NarrowInput(TypedDict):
        task: TaskRef

    async def work(state: StubState) -> dict[str, Any]:
        ledger.calls[name] += 1
        ledger.order.append(name)
        ledger.inputs.setdefault(name, []).append(dict(state))
        if name in ledger.crash:
            ledger.crash.discard(name)
            raise RuntimeError(f"{name} crashed")
        value = result(state) if callable(result) else result
        return {state_key_output or "result": value, "log": [name]}

    async def ask(state: StubState) -> dict[str, Any]:
        if name in ledger.pause:
            ledger.pause.discard(name)
            interrupt(f"{name}: continue?")
        return {}

    builder: StateGraph[Any, Any, Any, Any] = StateGraph(
        StubState,
        context_schema=ScientistContext,
        input_schema=NarrowInput if narrow else StubInput,
        output_schema=StubOutput,
    )
    builder.add_node("work", work)
    builder.add_node("ask", ask)
    builder.add_edge(START, "work")
    builder.add_edge("work", "ask")
    builder.add_edge("ask", END)
    return builder.compile(name=name)


def stub_overrides(
    ledger: Ledger,
    results: Mapping[str, Result | Callable[[Mapping[str, Any]], Result]] | None = None,
    *,
    slots: Mapping[str, Any] | None = None,
) -> GraphOverrides:
    """Return overrides that replace every phase with a stub (``results`` replaces defaults)."""
    chosen = {**default_results(), **(results or {})}
    phases = {name: stub_phase(name, ledger, value) for name, value in chosen.items()}
    return GraphOverrides(phases=phases, slots=dict(slots or {}))
