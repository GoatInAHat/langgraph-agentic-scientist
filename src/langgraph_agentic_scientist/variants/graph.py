"""The variants graph: ablations and baselines against a completed investigation (upstream's
variants pipeline, stages 0-8).

``build_variants_graph`` returns a compiled graph that can run alone or be mounted as the
``ablations`` phase of the pipeline. The nodes are the stages of upstream's pipeline:

=============================  ====================================================================
``prepare``                    checks the context, seals the evaluator tree (digest)
``extract``                    stage 0: the best-program bundle
``retrofit_prepare/agent/finish``  stage 1: the retrofit agent adds analysis outputs (sealed)
``generate``                   stage 2: the variant generator proposes baselines and ablations
``select``                     stage 3: the variant reviewer filters (fail-open); hard caps
``custom_baselines``           custom baselines are injected into the selection
``materialize``                stage 4: one workspace per variant
``run_control`` -> ``after_control`` -> ``run_variant``  stage 5: the control first, then the rest
``join_variants``              stage 5: summary, required controls, at least one passing variant
``cross_task``                 stage 6: adaptation to each extra task (a subgraph)
``aggregate``                  stage 7: combined analysis CSVs
``readme``                     stage 8: the README summariser, deterministic README with fallbacks
=============================  ====================================================================

Every node that runs a role, an evaluation or an effect has hook nodes (``variants.<step>.before``
and ``.after``). The hook-writable keys are ``candidates`` after ``generate``, ``selection`` after
``select`` and ``custom_baselines``. The variant runner and the cross-task subgraph are replaceable
through ``GraphOverrides.slots`` under ``variants.run_variant`` and ``variants.cross_task``.

A failed stage ends the phase with ``status="failed"`` and the failure; no later stage runs, as in
upstream, where each of these returns a non-zero exit code.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from ..agents.runner import agent_node_retry_policy, run_structured_role
from ..context import GraphOverrides, ScientistContext
from ..core.errors import ProgramSpecError
from ..core.render import truncate_utf8
from ..core.results import (
    DownstreamSelection,
    InvestigationResult,
    VariantSpec,
    VariantsResult,
)
from ..core.runconfig import method_config, record_run_config
from ..core.state import (
    Event,
    Failure,
    ProgramRef,
    RunConfigRecord,
    TaskRef,
    append,
    merge_dicts,
)
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.paths import join
from .aggregate import aggregate_variant_csvs
from .constants import (
    CONTROL_BEST_ID,
    GENERATION_DIR,
    PIPELINE_SUMMARY_FILENAME,
    README_WORK_DIR,
    RETROFITTED_BUNDLE_DIR,
)
from .cross_task import build_cross_task_graph
from .custom import inject_custom_baselines, load_custom_baselines
from .digests import managed_evaluator_tree_digest
from .execute import build_run_variant_graph, run_variant_input
from .extract import (
    BundleValidationError,
    ExtractionError,
    extract_bundle,
    resolve_best_program,
)
from .generate import build_generation_prompt, screen_candidates
from .materialize import materialize_variant_workspaces
from .models import (
    BundleRef,
    RetrofitReport,
    VariantBundleAnswer,
    VariantCandidate,
    VariantDecisionsAnswer,
    VariantRun,
    VariantSelection,
    VariantSummariesAnswer,
    spec_to_dict,
)
from .readme import (
    assemble_readme,
    build_readme_prompt,
    build_status_map,
    discover_tasks,
    fallback_summary,
    fallback_task_summary,
    load_flat_variant_specs,
    read_task_md,
)
from .retrofit import (
    finish_retrofit,
    retrofit_bundle_complete,
    run_retrofit_session,
    seal_and_copy,
)
from .roles import model_label
from .select import (
    apply_hard_caps,
    build_selection_prompt,
    decisions_from_answer,
    fail_open_decisions,
    selection_json,
)
from .workspace_ops import make_dir, read_json_object, write_json

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "variants"
FAILURE_DETAIL_BYTES = 2048


class VariantsInput(TypedDict):
    """What the variants graph is given.

    Attributes:
        task: The run-local task snapshot (``task.md``, the evaluators, the seed ``program_0``).
        run_id: The run id; evaluation ids start with it.
        run_config: The recorded method configuration (a mounted phase receives its parent's).
        best: The program to build the study around, when the caller names one.
        investigation: The investigation's result (its best-program manifest names the program).
        downstream: The frozen no-winner selection; wins over ``investigation``.
        output_dir: Where the phase writes (default ``variants``).
    """

    task: TaskRef
    run_id: str
    run_config: NotRequired[RunConfigRecord | None]
    best: NotRequired[ProgramRef | None]
    investigation: NotRequired[InvestigationResult | None]
    downstream: NotRequired[DownstreamSelection | None]
    output_dir: NotRequired[str]


class VariantsOutput(TypedDict):
    """What the variants graph returns."""

    variants: VariantsResult
    events: list[Event]


class VariantsState(VariantsInput, total=False):
    """State of the variants graph."""

    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    evaluator_digest: str | None
    bundle: BundleRef | None
    retrofit: RetrofitReport | None
    retrofit_session_ok: bool
    retrofit_session_error: str | None
    candidates: list[VariantCandidate]
    generation_decision: str | None
    selection: VariantSelection | None
    specs: list[VariantSpec]
    variant_runs: Annotated[list[VariantRun], append]
    cross_task_runs: list[VariantRun]
    cross_failure: Failure | None
    aggregates: dict[str, str]
    readme_path: str | None
    route: str
    failure: Failure | None
    events: Annotated[list[Event], append]
    variants: VariantsResult | None


# --- helpers -------------------------------------------------------------------------------------


def _out(state: Mapping[str, Any]) -> str:
    return str(state.get("output_dir") or DEFAULT_OUTPUT_DIR)


def _failure(category: str, detail: str, *, scope: str = "method") -> Failure:
    return Failure(
        category=category,
        scope=scope,
        detail=truncate_utf8(detail, FAILURE_DETAIL_BYTES),
    )


def _event(step: str, **data: Any) -> list[Event]:
    return [Event(kind="stage_end", phase="variants", step=step, data=data)]


def _ok(state: Mapping[str, Any]) -> str:
    return "finish" if state.get("failure") is not None else "ok"


def _variants_digest(specs: list[VariantSpec]) -> str:
    payload = json.dumps(
        [spec_to_dict(spec) for spec in specs], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


# --- stage nodes ---------------------------------------------------------------------------------


async def prepare(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Check the context and seal the evaluator tree (upstream's preflight of stage 5)."""
    ctx = runtime.context
    cfg = method_config(state)
    if ctx.evaluator is None:
        return {
            "failure": _failure(
                "no_evaluator", "the ablations phase needs ScientistContext.evaluator", scope="task"
            )
        }
    if (
        state["task"].contract is not None
        and not ctx.evaluator.isolated
        and not cfg.allow_unisolated_contracted
    ):
        return {
            "failure": _failure(
                "contracted_task_requires_isolation",
                "a contracted task is evaluated only by an isolated evaluator; "
                "set allow_unisolated_contracted for local experiments",
                scope="infrastructure",
            )
        }
    try:
        digest = await managed_evaluator_tree_digest(ctx.workspace, state["task"].root)
    except (OSError, ValueError) as exc:
        return {
            "failure": _failure(
                "evaluator_preflight_failed",
                f"Managed runtime evaluator preflight failed for {state['task'].root}: {exc}",
                scope="task",
            )
        }
    return {"evaluator_digest": digest, "output_dir": _out(state), "route": "ok"}


async def extract(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Stage 0: extract the best-program bundle."""
    ctx = runtime.context
    cfg = method_config(state).variants
    try:
        best, meta = resolve_best_program(
            best=state.get("best"),
            downstream=state.get("downstream"),
            investigation=state.get("investigation"),
        )
        bundle = await extract_bundle(
            ctx.workspace,
            task=state["task"],
            output_dir=_out(state),
            best=best,
            meta=meta,
            downstream=state.get("downstream"),
            baseline_program_override=cfg.baseline_program_override,
        )
    except (ExtractionError, BundleValidationError, ProgramSpecError, ValueError, OSError) as exc:
        return {"failure": _failure("bundle_extraction_failed", str(exc))}
    return {"bundle": bundle, "events": _event("extract", stage=0)}


async def retrofit_prepare(
    state: VariantsState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Stage 1, step 1: seal the evaluator and copy the bundle (brokered candidates finish here)."""
    ctx = runtime.context
    bundle = state["bundle"]
    assert bundle is not None
    report = await seal_and_copy(
        ctx.workspace,
        bundle_dir=bundle.extracted,
        output_bundle_dir=join(_out(state), RETROFITTED_BUNDLE_DIR),
        task_source_dir=state["task"].root,
        agent_model=model_label(ctx, Role.RETROFIT_AGENT),
    )
    if report.error is not None:
        return {"retrofit": report, "failure": _failure("retrofit_failed", report.error)}
    return {"retrofit": report, "route": "done" if report.mode == "brokered_preserve" else "agent"}


async def retrofit_agent(
    state: VariantsState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Stage 1, step 2: the retrofit agent (R16) instruments the copied program."""
    cfg = method_config(state).variants
    report = state["retrofit"]
    assert report is not None
    return await run_retrofit_session(runtime, report, timeout=cfg.retrofit_timeout_seconds)


async def retrofit_finish(
    state: VariantsState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Stage 1, step 3: restore the sealed evaluator, verify and publish the completion marker."""
    ws = runtime.context.workspace
    bundle = state["bundle"]
    report = state["retrofit"]
    assert bundle is not None and report is not None
    finished = await finish_retrofit(
        ws,
        report,
        session_ok=bool(state.get("retrofit_session_ok")),
        session_error=state.get("retrofit_session_error"),
    )
    complete = finished.exit_code == 0 and await retrofit_bundle_complete(
        ws,
        finished.output_bundle_dir,
        bundle.extracted,
        task_source_dir=state["task"].root,
    )
    if not complete:
        detail = finished.error or "the retrofitted bundle is incomplete or untrusted"
        return {
            "retrofit": finished,
            "failure": _failure(
                "retrofit_failed", f"Retrofit failed; stopping before variant generation: {detail}"
            ),
        }
    return {"retrofit": finished, "events": _event("retrofit", stage=1, mode=finished.mode)}


async def generate(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Stage 2: the variant generator (R17) proposes baselines and ablations."""
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).variants
    bundle = state["bundle"]
    assert bundle is not None
    generation_dir = join(_out(state), GENERATION_DIR)
    await make_dir(ws, generation_dir)
    prompt = await build_generation_prompt(
        ws, ctx.prompts, bundle.extracted, max_chars=cfg.diff_max_chars
    )
    outcome = await run_structured_role(
        role=Role.VARIANT_GENERATOR,
        schema=VariantBundleAnswer,
        runtime=runtime,
        root=generation_dir,
        prompt=prompt,
        timeout=cfg.generation_timeout_seconds,
    )
    answer = outcome.structured
    if not isinstance(answer, VariantBundleAnswer):
        logger.warning("variant generation produced no bundle: %s", outcome.failure)
        return {
            "candidates": [],
            "generation_decision": None,
            "events": _event("generate", stage=2, candidates=0),
        }
    screened = screen_candidates(answer)
    await write_json(
        ws, join(generation_dir, "variant_bundle.json"), answer.model_dump(mode="json")
    )
    await write_json(
        ws,
        join(generation_dir, "change_summary.json"),
        {
            "decision": answer.decision,
            "summary": answer.summary,
            "num_candidates": len(screened.candidates),
            "exit_code": 0 if outcome.ok else 1,
        },
    )
    return {
        "candidates": screened.candidates,
        "generation_decision": answer.decision,
        "events": _event("generate", stage=2, candidates=len(screened.candidates)),
    }


async def select(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Stage 3: the variant reviewer (R18) filters the proposals; hard caps; fail-open."""
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).variants
    bundle = state["bundle"]
    assert bundle is not None
    candidates = list(state.get("candidates") or [])
    generation_dir = join(_out(state), GENERATION_DIR)
    if not candidates:
        selection = VariantSelection(accepted=[], rejected=[], controls=[CONTROL_BEST_ID])
    else:
        prompt = await build_selection_prompt(
            ws, ctx.prompts, candidates, bundle.extracted, max_chars=cfg.diff_max_chars
        )
        outcome = await run_structured_role(
            role=Role.VARIANT_REVIEWER,
            schema=VariantDecisionsAnswer,
            runtime=runtime,
            root=generation_dir,
            prompt=prompt,
            timeout=cfg.selection_timeout_seconds,
        )
        answer = outcome.structured
        if not outcome.ok or not isinstance(answer, VariantDecisionsAnswer):
            logger.warning("selection agent failed; accepting all candidates")
            decisions = fail_open_decisions(candidates, "accepted (agent failed)")
        else:
            decisions = decisions_from_answer(candidates, answer)
        selection = apply_hard_caps(
            candidates,
            decisions,
            max_ablations=cfg.max_ablations,
            max_extra_baselines=cfg.max_extra_baselines,
        )
    await write_json(ws, join(generation_dir, "variant_selection.json"), selection_json(selection))
    return {
        "selection": selection,
        "events": _event(
            "select",
            stage=3,
            accepted=len(selection.accepted),
            rejected=len(selection.rejected),
        ),
    }


async def custom_baselines(
    state: VariantsState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Inject the custom baselines into the selection (between stages 3 and 4)."""
    cfg = method_config(state).variants
    selection = state.get("selection")
    if not cfg.custom_baselines:
        return {}
    try:
        custom = await load_custom_baselines(runtime.context.workspace, cfg.custom_baselines)
    except (FileNotFoundError, ValueError) as exc:
        return {"failure": _failure("custom_baseline_invalid", str(exc), scope="task")}
    merged = inject_custom_baselines(selection, custom, custom_only=cfg.custom_only)
    return {"selection": merged}


async def materialize(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Stage 4: one workspace per variant."""
    ctx = runtime.context
    cfg = method_config(state).variants
    bundle = state["bundle"]
    assert bundle is not None
    cross_requested = bool(cfg.extra_tasks)
    try:
        specs = await materialize_variant_workspaces(
            ctx.workspace,
            ctx.prompts,
            bundle_dir=bundle.extracted,
            retrofitted_bundle_dir=join(_out(state), RETROFITTED_BUNDLE_DIR),
            variants_run_dir=_out(state),
            selection=state.get("selection"),
            agent="langgraph-deepagents",
            agent_model=model_label(ctx, Role.VARIANT_EDITOR),
            max_evals_per_variant=cfg.max_evals_per_variant,
            custom_only=cfg.custom_only,
            required_controls=[CONTROL_BEST_ID] if cross_requested else [],
        )
    except (ValueError, ProgramSpecError, OSError) as exc:
        return {"failure": _failure("materialize_failed", str(exc))}
    controls = [spec for spec in specs if spec.variant_id == CONTROL_BEST_ID]
    if cross_requested and not (
        len(controls) == 1
        and controls[0].kind == "control"
        and controls[0].variant_dir == join(_out(state), "variants", CONTROL_BEST_ID)
    ):
        return {
            "failure": _failure(
                "materialize_failed", "Cross-task stage requires a canonical control_best"
            )
        }
    return {"specs": specs, "events": _event("materialize", stage=4, variants=len(specs))}


def _run_input(state: Mapping[str, Any], spec: VariantSpec) -> Any:
    cfg = method_config(state).variants
    return run_variant_input(
        spec=spec,
        task=state["task"],
        run_id=state["run_id"],
        run_config=state.get("run_config"),
        evaluator_digest=state.get("evaluator_digest"),
        agent_timeout=cfg.variant_timeout_seconds,
        id_parts=["variants"],
    )


def route_materialized(state: Mapping[str, Any]) -> Any:
    """The control runs first; without a control (custom-only) every variant runs at once."""
    if state.get("failure") is not None:
        return "finish"
    specs = list(state.get("specs") or [])
    controls = [spec for spec in specs if spec.kind == "control"]
    if controls:
        return [Send("run_control", _run_input(state, spec)) for spec in controls]
    if specs:
        return [Send("run_variant", _run_input(state, spec)) for spec in specs]
    return "join_variants"


async def after_control(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """A failed control aborts the remaining variants (they are not run)."""
    controls = {spec.variant_id for spec in state["specs"] if spec.kind == "control"}
    failed = [
        run.spec.variant_id
        for run in state.get("variant_runs", [])
        if run.spec.variant_id in controls and not run.rubric_passed
    ]
    if failed:
        logger.error("Control %s failed verification; aborting remaining variants", failed)
        return {"route": "abort"}
    return {"route": "fan_out"}


def route_after_control(state: Mapping[str, Any]) -> Any:
    """Fan out the non-control variants, or go straight to the join."""
    if state.get("route") == "abort":
        return "join_variants"
    rest = [spec for spec in state.get("specs") or [] if spec.kind != "control"]
    if not rest:
        return "join_variants"
    return [Send("run_variant", _run_input(state, spec)) for spec in rest]


async def join_variants(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Stage 5 summary: ``pipeline_summary.json``, required controls, one passing variant."""
    ws = runtime.context.workspace
    specs = state["specs"]
    runs = list(state.get("variant_runs", []))
    by_id = {run.spec.variant_id: run for run in runs}
    entries: list[dict[str, Any]] = []
    for spec in specs:
        run = by_id.get(spec.variant_id)
        entry: dict[str, Any] = {
            "variant_id": spec.variant_id,
            "kind": spec.kind,
            "title": spec.title,
        }
        if run is not None:
            entry.update(
                {
                    "status": run.status,
                    "rubric_passed": run.rubric_passed,
                    "eval_count": run.eval_count,
                    "exit_code": run.exit_code,
                    "analysis_files": run.analysis_files,
                }
            )
        else:
            entry["status"] = "not_executed"
        entries.append(entry)
    summary = {
        "schema_version": 1,
        "variant_specs_sha256": _variants_digest(specs),
        "total_variants": len(specs),
        "total_completed": sum(1 for r in runs if r.status == "completed"),
        "total_failed": sum(1 for r in runs if r.status == "failed"),
        "total_timeout": sum(1 for r in runs if r.status == "timeout"),
        "rubric_passed": sum(1 for r in runs if r.rubric_passed),
        "variants": entries,
    }
    await write_json(ws, join(_out(state), PIPELINE_SUMMARY_FILENAME), summary)
    passed = sum(1 for r in runs if r.rubric_passed)
    logger.info(
        "Variants pipeline complete: %d passed, %d failed out of %d total",
        passed,
        len(runs) - passed,
        len(runs),
    )
    infrastructure = next(
        (r.infrastructure_failure for r in runs if r.infrastructure_failure), None
    )
    control_ids = {spec.variant_id for spec in specs if spec.kind == "control"}
    failed_controls = [
        r.spec.variant_id for r in runs if r.spec.variant_id in control_ids and not r.rubric_passed
    ]
    event = _event("execute", stage=5, passed=passed, failed=len(runs) - passed)
    if infrastructure is not None:
        return {"failure": infrastructure, "events": event}
    if failed_controls:
        return {
            "failure": _failure(
                "required_control_failed",
                "Required controls failed verification: "
                f"{', '.join(failed_controls)}; stopping before paper stages",
            ),
            "events": event,
        }
    if passed == 0:
        return {
            "failure": _failure(
                "no_valid_variant",
                "No variant produced a valid evaluation; stopping before paper stages",
            ),
            "events": event,
        }
    return {"events": event}


def route_after_join(state: Mapping[str, Any]) -> str:
    """Cross-task adaptation only when extra tasks are configured."""
    if state.get("failure") is not None:
        return "finish"
    cfg = method_config(state).variants
    return "cross_task" if cfg.extra_tasks else "aggregate"


async def cross_result(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Turn the cross-task graph's failure into the phase's failure."""
    failure = state.get("cross_failure")
    if failure is not None:
        return {"failure": failure}
    return {"events": _event("cross_task", stage=6, runs=len(state.get("cross_task_runs", [])))}


async def aggregate(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Stage 7: aggregate the analysis CSVs."""
    ctx = runtime.context
    original = state["task"].root.rstrip("/").rsplit("/", 1)[-1] or "original"
    results = await aggregate_variant_csvs(
        ctx.workspace,
        ctx.prompts,
        _out(state),
        original_task_name=original,
        max_evidence=method_config(state).variants.rubric_evidence_max_chars,
    )
    aggregates = {
        f"{task}/{filename}": path
        for task, files in results.items()
        for filename, path in files.items()
        if path is not None
    }
    return {"aggregates": aggregates, "events": _event("aggregate", stage=7, tasks=len(results))}


async def readme(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Stage 8: the README summariser (R22), then the deterministic README with fallbacks."""
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).variants
    out = _out(state)
    specs = list(state["specs"])
    task_md = await read_task_md(ws, state["task"].root)
    task_info = await discover_tasks(ws, out, state["task"].root, None)
    flat: dict[str, list[dict[str, Any]]] = {}
    for info in task_info:
        if info.get("role") == "flat":
            flat_specs = await load_flat_variant_specs(ws, join(out, "extra_tasks", info["name"]))
            if flat_specs:
                flat[info["name"]] = flat_specs
    work = join(out, README_WORK_DIR)
    await make_dir(ws, work)
    prompt = build_readme_prompt(ctx.prompts, specs, task_md, flat)
    outcome = await run_structured_role(
        role=Role.README_SUMMARISER,
        schema=VariantSummariesAnswer,
        runtime=runtime,
        root=work,
        prompt=prompt,
        timeout=cfg.readme_timeout_seconds,
    )
    answer = outcome.structured if isinstance(outcome.structured, VariantSummariesAnswer) else None
    task_summary = (answer.task_summary if answer else "") or fallback_task_summary(
        ctx.prompts, task_md
    )
    summaries = {
        spec.variant_id: (answer.variant_summaries.get(spec.variant_id, "") if answer else "")
        or fallback_summary(ctx.prompts, spec)
        for spec in specs
    }
    status_map = build_status_map(await read_json_object(ws, join(out, PIPELINE_SUMMARY_FILENAME)))
    text = assemble_readme(ctx.prompts, task_summary, specs, summaries, status_map, task_info, flat)
    path = join(out, "README.md")
    await ws.write_bytes(path, text.encode("utf-8"))
    return {"readme_path": path, "events": _event("readme", stage=8)}


async def finish(state: VariantsState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Assemble the phase result."""
    failure = state.get("failure")
    runs = list(state.get("variant_runs", []))
    result = VariantsResult(
        status="failed" if failure is not None else "completed",
        output_dir=_out(state),
        specs=list(state.get("specs") or []),
        variants=[run.result for run in runs],
        cross_task=[run.result for run in state.get("cross_task_runs", [])],
        aggregates=dict(state.get("aggregates") or {}),
        readme_path=state.get("readme_path"),
        failure=failure,
    )
    return {"variants": result}


def _gate(state: Mapping[str, Any]) -> str:
    return _ok(state)


def _graph_slot(
    overrides: GraphOverrides, key: str
) -> CompiledStateGraph[Any, Any, Any, Any] | None:
    """Return the compiled graph overriding slot ``key`` (a callable is not accepted here)."""
    slot = overrides.slots.get(key)
    if slot is None:
        return None
    if not isinstance(slot, CompiledStateGraph):
        raise TypeError(f"slot {key} must be a compiled graph")
    return slot


def build_variants_graph(
    *,
    checkpointer: Any = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "variants",
) -> CompiledStateGraph[Any, ScientistContext, VariantsInput, VariantsOutput]:
    """Return the compiled variants graph (see the module docstring).

    Args:
        checkpointer: The checkpointer, only for the top-level caller; a mounted graph inherits.
        store: The store, only for the top-level caller.
        overrides: Replacement slots: ``variants.run_variant`` (the graph that runs one variant)
            and ``variants.cross_task`` (the cross-task subgraph).
        name: The graph's name.
    """
    overrides = overrides or GraphOverrides()
    run_graph = _graph_slot(overrides, "variants.run_variant") or build_run_variant_graph(
        overrides=overrides
    )
    cross_graph = _graph_slot(overrides, "variants.cross_task") or build_cross_task_graph(
        overrides=overrides, run_variant_graph=run_graph
    )
    graph = StepGraph(
        "variants",
        VariantsState,
        input_schema=VariantsInput,
        output_schema=VariantsOutput,
        overrides=overrides,
    )
    retry = agent_node_retry_policy()
    graph.add_node("record_run_config", record_run_config)
    graph.step("prepare", prepare)
    graph.step("extract", extract)
    graph.step("retrofit_prepare", retrofit_prepare)
    graph.step("retrofit_agent", retrofit_agent, retry_policy=retry)
    graph.step("retrofit_finish", retrofit_finish)
    graph.step("generate", generate, retry_policy=retry, hook_writable=("candidates",))
    graph.step("select", select, retry_policy=retry, hook_writable=("selection",))
    graph.step("custom_baselines", custom_baselines, hook_writable=("selection",))
    graph.step("materialize", materialize)
    graph.add_node("run_control", run_graph)
    graph.step("after_control", after_control)
    graph.add_node("run_variant", run_graph)
    graph.step("join_variants", join_variants)
    graph.step("cross_task", cross_graph)
    graph.add_node("cross_result", cross_result)
    graph.step("aggregate", aggregate)
    graph.step("readme", readme, retry_policy=retry)
    graph.step("finish", finish)

    graph.graph.add_edge(START, "record_run_config")
    graph.graph.add_edge("record_run_config", graph.entry("prepare"))
    graph.add_conditional_edges("prepare", _gate, {"ok": "extract", "finish": "finish"})
    graph.add_conditional_edges("extract", _gate, {"ok": "retrofit_prepare", "finish": "finish"})
    graph.add_conditional_edges(
        "retrofit_prepare",
        lambda s: (
            "finish"
            if s.get("failure") is not None
            else ("agent" if s.get("route") == "agent" else "done")
        ),
        {"agent": "retrofit_agent", "done": "retrofit_finish", "finish": "finish"},
    )
    graph.add_edge("retrofit_agent", "retrofit_finish")

    def after_retrofit(state: Mapping[str, Any]) -> str:
        if state.get("failure") is not None:
            return "finish"
        return "custom" if method_config(state).variants.custom_only else "generate"

    graph.add_conditional_edges(
        "retrofit_finish",
        after_retrofit,
        {"generate": "generate", "custom": "custom_baselines", "finish": "finish"},
    )
    graph.add_edge("generate", "select")
    graph.add_edge("select", "custom_baselines")
    graph.add_conditional_edges(
        "custom_baselines", _gate, {"ok": "materialize", "finish": "finish"}
    )
    graph.add_conditional_edges(
        "materialize", route_materialized, ["run_control", "run_variant", "join_variants", "finish"]
    )
    graph.graph.add_edge("run_control", graph.entry("after_control"))
    graph.add_conditional_edges(
        "after_control", route_after_control, ["run_variant", "join_variants"]
    )
    graph.graph.add_edge("run_variant", graph.entry("join_variants"))
    graph.add_conditional_edges(
        "join_variants",
        route_after_join,
        {"cross_task": "cross_task", "aggregate": "aggregate", "finish": "finish"},
    )
    graph.graph.add_edge(graph.exit("cross_task"), "cross_result")
    graph.graph.add_conditional_edges(
        "cross_result",
        lambda s: "finish" if s.get("failure") is not None else "ok",
        {"ok": graph.entry("aggregate"), "finish": graph.entry("finish")},
    )
    graph.add_edge("aggregate", "readme")
    graph.add_edge("readme", "finish")
    graph.graph.add_edge(graph.exit("finish"), END)
    return graph.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "VariantsInput",
    "VariantsOutput",
    "VariantsState",
    "build_variants_graph",
]
