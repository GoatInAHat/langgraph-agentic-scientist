"""The post-hoc analysis phase as a graph (upstream ``posthoc_pipeline/pipeline.py``).

The steps, in order (each carries hook nodes ``posthoc.<step>.before/.after``):

``record_run_config`` -> ``extract`` -> ``bootstrap_prepare`` -> (``bootstrap_reset`` ->
``bootstrap_agent`` -> ``bootstrap_evaluate`` -> ``bootstrap_record``)* -> ``bootstrap_finish`` ->
``restore_evidence`` -> ``generate`` -> ``select`` -> ``apply_constraints`` -> ``materialize`` ->
``Send("run_analysis")`` per analysis -> ``publish``

* The stages and their rules are upstream's: see the modules of this package (``extract``,
  ``bootstrap``, ``generate``, ``select``, ``materialize``, ``execute``, ``publish``).
* The phase is best effort, as upstream's: a bootstrap that cannot produce a checkpoint ends
  ``degraded`` and the run goes on; selection fails open; the phase result is ``completed`` only
  when at least one analysis is valid, ``degraded`` when none is, and ``failed`` when the inputs
  could not be extracted (the pipeline treats both as a degraded optional phase).
* Analyses run in parallel through ``Send``, each in its own subgraph with its own attempt loop;
  upstream ran them one after another. Bound concurrency with the run config's
  ``max_concurrency`` (use 1 for analyses that use a single GPU).
* A step that waits on nothing but the filesystem does no effect twice: every step rebuilds its
  outputs from scratch, so a node restarted after a crash is harmless; the evaluator call of the
  bootstrap is keyed by ``<run>/posthoc/bootstrap_<n>`` and is the only effectful call of its
  node.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Send

from ..agents.runner import agent_node_retry_policy
from ..context import GraphOverrides, ScientistContext
from ..core.errors import FinalEvaluationError
from ..core.runconfig import record_run_config
from ..core.state import Event, Failure
from ..core.steps import StepGraph
from ..final_evaluation.manifest import resolve_incumbent
from .bootstrap import (
    bootstrap_agent,
    bootstrap_evaluate,
    bootstrap_finish,
    bootstrap_prepare,
    bootstrap_record,
    bootstrap_reset,
    route_after_agent,
    route_after_prepare,
    route_after_record,
)
from .execute import build_analysis_graph
from .extract import (
    BundleValidationError,
    ExtractionError,
    extract_bundle,
    restore_verified_evidence,
)
from .generate import generate_candidates
from .layout import Layout
from .materialize import materialize
from .models import AnalysisInput, PosthocInput, PosthocOutput, PosthocState
from .publish import publish
from .select import apply_constraints, select_analyses
from .settings import posthoc_config

PHASE = "posthoc"
ANALYSIS_SLOT = "posthoc.analysis"


async def extract(state: PosthocState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Resolve the incumbent, build the extracted bundle and stage the evidence."""
    ctx = runtime.context
    workspace = ctx.workspace
    layout = Layout.of(state.get("output_dir"))
    task = state["task"]
    try:
        incumbent = await resolve_incumbent(
            workspace,
            incumbent=state.get("incumbent"),
            incumbent_manifest=state.get("incumbent_manifest"),
            downstream=state.get("downstream"),
            investigation=state.get("investigation"),
        )
    except FinalEvaluationError as exc:
        return _failed("incumbent_unavailable", str(exc))
    variants = state.get("variants")
    evidence_dir = state.get("evidence_dir")
    if evidence_dir is None and variants is not None and variants.status == "completed":
        evidence_dir = variants.output_dir
    investigation = state.get("investigation")
    manifest = {
        "entrypoint_path": incumbent.program.entrypoint,
        "combined_score": incumbent.combined_score,
        "correct": incumbent.correct,
        "source_attempt_id": incumbent.source_attempt_id,
        "parallel_worker_index": incumbent.extra.get("parallel_worker_index"),
    }
    try:
        extraction = await extract_bundle(
            workspace,
            task=task,
            best_program_dir=incumbent.program.path,
            manifest=manifest,
            results_dir=investigation.results_dir if investigation is not None else None,
            reference_dir=task.seed_program.path if task.seed_program is not None else None,
            output_dir=layout.output,
            downstream_selection=incumbent.extra.get("downstream_selection"),
            evidence_dir=evidence_dir,
            config=posthoc_config(state),
        )
    except (ExtractionError, BundleValidationError, ValueError) as exc:
        return _failed("extraction_failed", str(exc))
    return {
        "extraction": extraction,
        "best": {"program": incumbent.program.model_dump(mode="json"), "source": incumbent.source},
        "events": [
            Event(
                kind="posthoc_extracted",
                phase=PHASE,
                data={"program": incumbent.program.path, "evidence": evidence_dir},
            )
        ],
    }


def _failed(category: str, detail: str) -> dict[str, Any]:
    return {
        "phase_error": Failure(category=category, scope="task", detail=detail),
        "events": [Event(kind="posthoc_error", phase=PHASE, data={"category": category})],
    }


async def restore_evidence(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Put the verified evidence back into the analysis-ready bundle after the bootstrap."""
    layout = Layout.of(state.get("output_dir"))
    try:
        await restore_verified_evidence(
            runtime.context.workspace, layout.bundle, layout.ready, posthoc_config(state)
        )
    except BundleValidationError as exc:
        return _failed("evidence_invalid", str(exc))
    return {}


def _route_error(state: PosthocState) -> str:
    return "stop" if state.get("phase_error") is not None else "go"


def _fan_out(state: Any) -> list[Send] | str:
    # ``state`` is annotated ``Any``: StepGraph wraps a router that has no key map and resolves the
    # wrapper's annotations in its own module, where ``PosthocState`` is not defined.
    specs = state.get("specs") or []
    if not specs:
        return "publish"
    return [
        Send(
            "run_analysis",
            AnalysisInput(
                spec=spec,
                task=state["task"],
                run_id=state["run_id"],
                run_config=state.get("run_config"),
                output_dir=state.get("output_dir") or Layout.of(None).output,
                artifact_only=bool(state.get("artifact_only")),
            ),
        )
        for spec in specs
    ]


def build_posthoc_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "posthoc",
) -> CompiledStateGraph[PosthocState, ScientistContext, PosthocInput, PosthocOutput]:
    """Build the post-hoc analysis graph (usable alone or mounted as a subgraph).

    The graph needs ``ctx.evaluator`` only for the bootstrap's single evaluator rerun. Replace the
    per-analysis subgraph with ``overrides.slots["posthoc.analysis"]`` (a compiled graph that
    accepts :class:`~langgraph_agentic_scientist.posthoc.models.AnalysisInput`).

    Args:
        checkpointer: The checkpointer when the graph runs at top level.
        store: The store when the graph runs at top level.
        overrides: Replacement slots.
        name: The compiled graph's name.

    Returns:
        A compiled graph whose input is :class:`PosthocInput` and whose output is
        :class:`PosthocOutput` (``posthoc`` is the phase result).
    """
    overrides = overrides or GraphOverrides()
    sg = StepGraph(
        PHASE,
        PosthocState,
        input_schema=PosthocInput,
        output_schema=PosthocOutput,
        overrides=overrides,
    )
    analysis = overrides.slots.get(ANALYSIS_SLOT) or build_analysis_graph(overrides=overrides)
    retry = agent_node_retry_policy()

    sg.add_node("record_run_config", record_run_config)
    sg.step("extract", extract)
    sg.step("bootstrap_prepare", bootstrap_prepare)
    sg.step("bootstrap_reset", bootstrap_reset)
    sg.step("bootstrap_agent", bootstrap_agent, retry_policy=retry)
    sg.step("bootstrap_evaluate", bootstrap_evaluate)
    sg.step("bootstrap_record", bootstrap_record)
    sg.step("bootstrap_finish", bootstrap_finish)
    sg.step("restore_evidence", restore_evidence)
    sg.step("generate", generate_candidates, retry_policy=retry, hook_writable=("candidates",))
    sg.step("select", select_analyses, retry_policy=retry)
    sg.step("apply_constraints", apply_constraints, hook_writable=("selection",))
    sg.step("materialize", materialize)
    sg.add_node("run_analysis", analysis)
    sg.step("publish", publish)

    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "extract")
    sg.add_conditional_edges(
        "extract", _route_error, {"go": "bootstrap_prepare", "stop": "publish"}
    )
    sg.add_conditional_edges(
        "bootstrap_prepare",
        route_after_prepare,
        {"attempt": "bootstrap_reset", "finish": "bootstrap_finish", "stop": "publish"},
    )
    sg.add_edge("bootstrap_reset", "bootstrap_agent")
    sg.add_conditional_edges(
        "bootstrap_agent",
        route_after_agent,
        {"evaluate": "bootstrap_evaluate", "record": "bootstrap_record"},
    )
    sg.add_edge("bootstrap_evaluate", "bootstrap_record")
    sg.add_conditional_edges(
        "bootstrap_record",
        route_after_record,
        {"attempt": "bootstrap_reset", "finish": "bootstrap_finish"},
    )
    sg.add_edge("bootstrap_finish", "restore_evidence")
    sg.add_conditional_edges(
        "restore_evidence", _route_error, {"go": "generate", "stop": "publish"}
    )
    sg.add_conditional_edges("generate", _route_error, {"go": "select", "stop": "publish"})
    sg.add_edge("select", "apply_constraints")
    sg.add_edge("apply_constraints", "materialize")
    sg.add_conditional_edges("materialize", _fan_out, ["run_analysis", "publish"])
    sg.add_edge("run_analysis", "publish")
    sg.add_edge("publish", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = ["ANALYSIS_SLOT", "build_posthoc_graph"]
