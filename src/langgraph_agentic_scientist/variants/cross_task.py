"""Stage 6: cross-task generalisation (upstream ``cross_task.py``).

For each extra task (a task directory of the workspace with a pipeline-compatible evaluator) the
phase runs three steps:

0. **retrofit** the extra task's own program and evaluator-facing outputs with the retrofit agent
   (R16), given the original task's retrofitted evaluator as read-only context (the section of
   ``workflow/variants/reference_evaluator``); the extra task's evaluator stays sealed;
1. **adapt the control**: ``control_best`` is ported to the extra task in isolation, producing the
   *adapted control* (the best evaluated program of that run);
2. **adapt every other variant**: its ``program_0`` is the adapted control and
   ``original_variant_program/`` is the variant's best program on the original task; the agent
   applies the variant's algorithmic changes on top of the adapted control.

Both adaptations are ordinary variant runs (:func:`~.execute.build_run_variant_graph`) with a
cross-task ``task.md`` (the control adaptation is a ``baseline``-kind run, because it must create
``program_1``); the control adapts first, then the other variants run in parallel through
``Send``. Results live under ``extra_tasks/<task>/``. Any failed step fails the stage, as in
upstream, where a ``RuntimeError`` ends the pipeline.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from ..agents.runner import agent_node_retry_policy
from ..context import GraphOverrides, ScientistContext
from ..core.results import VariantSpec
from ..core.runconfig import method_config
from ..core.state import Failure, RunConfigRecord, StateModel, TaskRef, merge_dicts
from ..core.steps import StepGraph
from ..evaluation.program_clone import clone_program, evaluation_results_complete
from ..models import Role
from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace, is_dir, is_file
from .constants import CONTROL_BEST_ID, EXTRA_TASKS_SUBDIR, RETROFITTED_BUNDLE_DIR, VARIANTS_SUBDIR
from .digests import managed_evaluator_tree_digest
from .execute import RunVariantInput, build_run_variant_graph, run_variant_input
from .materialize import build_run_manifest, copy_task_files, piece
from .models import RetrofitReport, VariantRun, make_spec, spec_detail, spec_to_dict
from .retrofit import (
    finish_retrofit,
    retrofit_bundle_complete,
    run_retrofit_session,
    seal_and_copy,
)
from .roles import model_label
from .rubrics import build_rubric_contract, build_target_rubrics
from .tasks import task_ref_for
from .workspace_ops import (
    make_dir,
    program_dirs,
    read_text_or_none,
    write_json,
)

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
FAILURE_DETAIL_BYTES = 2048


async def find_best_program_dir(workspace: Workspace, task_bundle: str) -> str | None:
    """Return the highest-numbered evaluated ``program_<n>`` of a task bundle, or ``None``.

    A program counts when it holds ``metrics.json`` and its evaluation results are complete.
    """
    best: str | None = None
    best_index = -1
    for name in await program_dirs(workspace, task_bundle):
        program = join(task_bundle, name)
        index = int(name.split("_", 1)[1])
        if (
            index > best_index
            and await is_file(workspace, join(program, "metrics.json"))
            and await evaluation_results_complete(workspace, program)
        ):
            best, best_index = program, index
    return best


async def read_retrofitted_evaluator(workspace: Workspace, variants_run_dir: str) -> str:
    """Read the retrofitted ``evaluate.py`` of the original task (the extracted one as fallback)."""
    for bundle in (RETROFITTED_BUNDLE_DIR, "_extracted_bundle"):
        text = await read_text_or_none(workspace, join(variants_run_dir, bundle, "evaluate.py"))
        if text is not None:
            return text
    return ""


async def find_original_task_md(workspace: Workspace, variants_run_dir: str) -> str:
    """Retrieve the original ``task.md`` text from the bundles or a variant's workspace."""
    for bundle in (RETROFITTED_BUNDLE_DIR, "_extracted_bundle"):
        text = await read_text_or_none(workspace, join(variants_run_dir, bundle, "task.md"))
        if text is not None:
            return text
    variants_dir = join(variants_run_dir, VARIANTS_SUBDIR)
    if await is_dir(workspace, variants_dir):
        from .workspace_ops import list_child_dirs

        for name in await list_child_dirs(workspace, variants_dir):
            text = await read_text_or_none(
                workspace, join(variants_dir, name, "task_bundle", "original_task.md")
            )
            if text is not None:
                return text
    return "(Original task description not available.)"


def reference_evaluator_section(prompts: PromptRegistry, raw: str) -> str:
    """Return the reference-evaluator section for the retrofit prompt (empty without one)."""
    if not raw:
        return ""
    return prompts.render(
        "workflow/variants/reference_evaluator",
        {"REFERENCE_EVALUATOR_RAW": raw},
        fragments=False,
    )


# --- workspaces ----------------------------------------------------------------------------------


async def setup_task_bundle(
    workspace: Workspace,
    *,
    task_bundle: str,
    retrofitted_extra: str,
    program_src: str,
    original_task_md_text: str,
) -> None:
    """Set up a cross-task task bundle.

    Copies the task files of the **retrofitted** extra task (so ``evaluate.py`` is
    pipeline-compatible), clones the seed program as ``program_0`` (the clone's copy rules strip
    results and logs), saves the original task description as ``source_task.md`` and keeps the
    extra task's own ``task.md`` as ``original_task.md``.
    """
    await copy_task_files(workspace, retrofitted_extra, task_bundle)
    program_dst = join(task_bundle, "program_0")
    await workspace.remove(program_dst)
    await clone_program(workspace, src_dir=program_src, dst_dir=program_dst)
    await workspace.write_bytes(
        join(task_bundle, "source_task.md"), original_task_md_text.encode("utf-8")
    )
    copied = join(task_bundle, "task.md")
    original = join(task_bundle, "original_task.md")
    if await is_file(workspace, copied):
        await workspace.write_bytes(original, await workspace.read_bytes(copied))
        await workspace.remove(copied)
    elif await is_file(workspace, join(retrofitted_extra, "task.md")):
        await workspace.write_bytes(
            original, await workspace.read_bytes(join(retrofitted_extra, "task.md"))
        )


async def write_control_cross_task_md(
    workspace: Workspace,
    prompts: PromptRegistry,
    task_bundle: str,
    task_name: str,
    *,
    max_evals: int = 2,
) -> None:
    """Write ``task.md`` for the ``control_best`` cross-task adaptation."""

    def part(name: str, **values: str) -> str:
        return piece(prompts, "cross_task_control", name, **values)

    attempts = str(max_evals)
    sections = [
        part("title"),
        part("variant_id", VARIANT_ID=CONTROL_BEST_ID),
        part("target_task", TASK_NAME=task_name),
        part("max_attempts", MAX_EVALS=attempts),
        part("context", TASK_NAME=task_name),
        part("what_to_do", MAX_EVALS=attempts),
        part("task_references"),
    ]
    await workspace.write_bytes(join(task_bundle, "task.md"), "".join(sections).encode("utf-8"))


async def write_variant_cross_task_md(
    workspace: Workspace,
    prompts: PromptRegistry,
    task_bundle: str,
    orig_spec: VariantSpec,
    task_name: str,
    *,
    has_adapted_control: bool,
    max_evals: int = 2,
) -> None:
    """Write ``task.md`` for a non-control variant's cross-task adaptation."""

    def part(name: str, **values: str) -> str:
        return piece(prompts, "cross_task_variant", name, **values)

    attempts = str(max_evals)
    sections = [
        part("title", TITLE=orig_spec.title),
        part("variant_id", VARIANT_ID=orig_spec.variant_id),
        part("original_kind", KIND=orig_spec.kind),
        part("target_task", TASK_NAME=task_name),
        part("max_attempts", MAX_EVALS=attempts),
    ]
    if has_adapted_control:
        sections.append(part("context", TASK_NAME=task_name))
        sections.append(part("what_to_do", MAX_EVALS=attempts))
    else:
        sections.append(part("fallback_context", TASK_NAME=task_name))
        sections.append(part("fallback_what_to_do", MAX_EVALS=attempts))
    hypothesis = str(spec_detail(orig_spec, "hypothesis"))
    if hypothesis:
        sections.append(part("hypothesis", HYPOTHESIS=hypothesis))
        expected_effect = str(spec_detail(orig_spec, "expected_effect"))
        if expected_effect:
            sections.append(part("expected_effect", EXPECTED_EFFECT=expected_effect))
        if orig_spec.description:
            sections.append(part("description", DESCRIPTION_BODY=orig_spec.description))
    sections.append(part("task_references"))
    await workspace.write_bytes(join(task_bundle, "task.md"), "".join(sections).encode("utf-8"))


async def finalize_variant(
    workspace: Workspace,
    prompts: PromptRegistry,
    *,
    variant_dir: str,
    task_bundle: str,
    orig_spec: VariantSpec,
    task_name: str,
    agent: str,
    agent_model: str | None,
) -> VariantSpec:
    """Write the metadata files of a cross-task variant and return its spec.

    The control is adapted, not just evaluated, so its cross-task kind is ``baseline``.
    """
    await make_dir(workspace, variant_dir)
    cross_kind = "baseline" if orig_spec.kind == "control" else orig_spec.kind
    base_program = str(spec_detail(orig_spec, "base_program"))
    spec = make_spec(
        variant_id=orig_spec.variant_id,
        title=f"Cross-task: {orig_spec.title} -> {task_name}",
        kind=cross_kind,
        base_program=base_program,
        variant_dir=variant_dir,
        hypothesis=str(spec_detail(orig_spec, "hypothesis")),
        relevance_reason=orig_spec.rationale,
        feasibility_reason=str(spec_detail(orig_spec, "feasibility_reason")),
        target_components=spec_detail(orig_spec, "target_components"),
        implementation_plan=spec_detail(orig_spec, "implementation_plan"),
        expected_effect=str(spec_detail(orig_spec, "expected_effect")),
        description_body=orig_spec.description,
        task_dir=task_bundle,
        task_name=task_name,
    )
    await write_json(workspace, join(variant_dir, "variant_spec.json"), spec_to_dict(spec))
    contract = build_rubric_contract(
        prompts, orig_spec.variant_id, cross_kind, base_program=base_program
    )
    await write_json(workspace, join(variant_dir, "rubric_contract.json"), contract)
    await write_json(
        workspace, join(variant_dir, "target_rubrics.json"), build_target_rubrics(contract)
    )
    await write_json(
        workspace,
        join(variant_dir, "run_manifest.json"),
        build_run_manifest(spec, agent=agent, agent_model=agent_model),
    )
    await workspace.write_bytes(
        join(task_bundle, "rubric_contract.json"),
        await workspace.read_bytes(join(variant_dir, "rubric_contract.json")),
    )
    return spec


# --- the graph -----------------------------------------------------------------------------------


class ExtraTask(StateModel):
    """One extra task of the stage: its name, directory, task reference and sealed digest."""

    name: str
    path: str
    task: TaskRef
    evaluator_digest: str


class CrossTaskInput(TypedDict):
    """What the cross-task graph is given: the primary task, its materialised variants, the run."""

    task: TaskRef
    run_id: str
    run_config: RunConfigRecord | None
    output_dir: str
    specs: list[VariantSpec]


class CrossTaskOutput(TypedDict):
    """What the cross-task graph returns."""

    cross_task_runs: list[VariantRun]
    cross_failure: Failure | None


class CrossTaskState(CrossTaskInput, total=False):
    """State of the cross-task graph."""

    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    extra_tasks: list[ExtraTask]
    index: int
    original_task_md: str
    reference_evaluator: str
    retrofit: RetrofitReport | None
    retrofit_session_ok: bool
    retrofit_session_error: str | None
    route: str
    control_spec: VariantSpec | None
    cross_control: VariantSpec | None
    adapted_control: str | None
    cross_specs: list[VariantSpec]
    variant_runs: Annotated[list[VariantRun], operator.add]
    cross_task_runs: list[VariantRun]
    cross_failure: Failure | None


def _current(state: Mapping[str, Any]) -> ExtraTask:
    return state["extra_tasks"][state["index"]]  # type: ignore[no-any-return]


def _cross_dir(state: Mapping[str, Any]) -> str:
    return join(state["output_dir"], EXTRA_TASKS_SUBDIR, _current(state).name)


def _fail(category: str, detail: str) -> dict[str, Any]:
    return {
        "cross_failure": Failure(
            category=category, scope="method", detail=detail[:FAILURE_DETAIL_BYTES]
        )
    }


async def cross_prepare(
    state: CrossTaskState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Validate the extra tasks, seal their evaluators and read the original task's context."""
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).variants
    specs = state["specs"]
    if specs and not any(spec.variant_id == CONTROL_BEST_ID for spec in specs):
        return _fail("cross_task_failed", "cross-task adaptation requires control_best")
    extras: list[ExtraTask] = []
    names: set[str] = set()
    for raw in cfg.extra_tasks:
        path = raw.rstrip("/")
        name = path.rsplit("/", 1)[-1]
        if not NAME_RE.match(name) or name in names:
            return _fail(
                "cross_task_failed",
                f"extra task name {name!r} is not a unique safe name (letters, digits, ._:-)",
            )
        names.add(name)
        try:
            task = await task_ref_for(ws, path)
            digest = await managed_evaluator_tree_digest(ws, path)
        except (ValueError, OSError) as exc:
            return _fail(
                "cross_task_failed", f"cross-task {name!r} evaluator preflight failed: {exc}"
            )
        extras.append(ExtraTask(name=name, path=task.root, task=task, evaluator_digest=digest))
    if ctx.evaluator is None and extras:
        return _fail("cross_task_failed", "no evaluator is configured for this run")
    original = await find_original_task_md(ws, state["output_dir"])
    reference = reference_evaluator_section(
        ctx.prompts, await read_retrofitted_evaluator(ws, state["output_dir"])
    )
    return {
        "extra_tasks": extras,
        "index": 0,
        "original_task_md": original,
        "reference_evaluator": reference,
        "route": "finish" if not extras else "run",
    }


async def start_task(state: CrossTaskState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Begin one extra task: its evaluator must be unchanged since the preflight."""
    ws = runtime.context.workspace
    current = _current(state)
    if await managed_evaluator_tree_digest(ws, current.path) != current.evaluator_digest:
        return _fail(
            "cross_task_failed", f"cross-task {current.name!r} evaluator changed before adaptation"
        )
    await make_dir(ws, _cross_dir(state))
    return {"adapted_control": None, "cross_control": None, "cross_specs": []}


async def cross_retrofit_prepare(
    state: CrossTaskState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Seal and copy the extra task for retrofitting (phase 0), or reuse a complete retrofit."""
    ctx = runtime.context
    ws = ctx.workspace
    current = _current(state)
    retrofitted = join(_cross_dir(state), RETROFITTED_BUNDLE_DIR)
    if await retrofit_bundle_complete(ws, retrofitted, current.path, task_source_dir=current.path):
        return {"route": "adapt", "retrofit": None}
    report = await seal_and_copy(
        ws,
        bundle_dir=current.path,
        output_bundle_dir=retrofitted,
        task_source_dir=current.path,
        agent_model=model_label(ctx, Role.RETROFIT_AGENT),
    )
    if report.error is not None:
        return {
            "retrofit": report,
            **_fail(
                "cross_task_failed", f"cross-task {current.name!r} retrofit failed: {report.error}"
            ),
        }
    return {"retrofit": report, "route": "adapt" if report.mode == "brokered_preserve" else "agent"}


async def cross_retrofit_agent(
    state: CrossTaskState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the retrofit agent on the extra task's copy."""
    cfg = method_config(state).variants
    report = state["retrofit"]
    assert report is not None
    return await run_retrofit_session(
        runtime,
        report,
        reference_evaluator_source=state.get("reference_evaluator", ""),
        timeout=cfg.retrofit_timeout_seconds,
    )


async def cross_retrofit_finish(
    state: CrossTaskState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Restore the sealed evaluator and require a complete retrofit."""
    ws = runtime.context.workspace
    current = _current(state)
    report = state["retrofit"]
    assert report is not None
    finished = await finish_retrofit(
        ws,
        report,
        session_ok=bool(state.get("retrofit_session_ok")),
        session_error=state.get("retrofit_session_error"),
    )
    complete = finished.exit_code == 0 and await retrofit_bundle_complete(
        ws, finished.output_bundle_dir, current.path, task_source_dir=current.path
    )
    if not complete:
        return {
            "retrofit": finished,
            **_fail("cross_task_failed", f"cross-task {current.name!r} retrofit failed"),
        }
    return {"retrofit": finished}


async def prepare_control(
    state: CrossTaskState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Phase 1: prepare the workspace that adapts ``control_best`` to the extra task."""
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).variants
    current = _current(state)
    control = next((s for s in state["specs"] if s.variant_id == CONTROL_BEST_ID), None)
    if control is None:
        return {"control_spec": None, "cross_control": None}
    orig_bundle = join(control.variant_dir, "task_bundle")
    best = await find_best_program_dir(ws, orig_bundle)
    if best is None:
        return {
            "control_spec": control,
            **_fail(
                "cross_task_failed",
                f"cross-task {current.name!r} control_best has no evaluated program",
            ),
        }
    cross_dir = _cross_dir(state)
    variant_dir = join(cross_dir, VARIANTS_SUBDIR, CONTROL_BEST_ID)
    task_bundle = join(variant_dir, "task_bundle")
    await setup_task_bundle(
        ws,
        task_bundle=task_bundle,
        retrofitted_extra=join(cross_dir, RETROFITTED_BUNDLE_DIR),
        program_src=best,
        original_task_md_text=state["original_task_md"],
    )
    await write_control_cross_task_md(
        ws, ctx.prompts, task_bundle, current.name, max_evals=cfg.max_evals_per_variant
    )
    spec = await finalize_variant(
        ws,
        ctx.prompts,
        variant_dir=variant_dir,
        task_bundle=task_bundle,
        orig_spec=control,
        task_name=current.name,
        agent="langgraph-deepagents",
        agent_model=model_label(ctx, Role.VARIANT_EDITOR),
    )
    return {"control_spec": control, "cross_control": spec}


def _adapt_input(state: Mapping[str, Any], spec: VariantSpec) -> RunVariantInput:
    cfg = method_config(state).variants
    current = _current(state)
    return run_variant_input(
        spec=spec,
        task=current.task,
        run_id=state["run_id"],
        run_config=state.get("run_config"),
        evaluator_digest=current.evaluator_digest,
        agent_timeout=cfg.cross_task_timeout_seconds,
        id_parts=["extra", current.name],
        task_name=current.name,
    )


def route_control(state: Mapping[str, Any]) -> list[Send] | str:
    """Send the control adaptation to its run, or skip to the join."""
    if state.get("cross_failure") is not None:
        return "finish"
    spec = state.get("cross_control")
    if spec is None:
        return "after_control"
    return [Send("run_control", _adapt_input(state, spec))]


async def after_control(
    state: CrossTaskState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Require a verified adapted control and find its best program."""
    ws = runtime.context.workspace
    cfg = method_config(state).variants
    current = _current(state)
    spec = state.get("cross_control")
    if spec is None:
        return {}
    runs = [
        run for run in state.get("variant_runs", []) if run.spec.variant_dir == spec.variant_dir
    ]
    infrastructure = next((run.infrastructure_failure for run in runs if run.infrastructure_failure), None)
    if infrastructure is not None:
        return {"cross_failure": infrastructure}
    if len(runs) != 1 or not runs[0].rubric_passed:
        return _fail(
            "cross_task_failed",
            f"cross-task {current.name!r} adapted control failed verification",
        )
    adapted = await find_best_program_dir(ws, join(spec.variant_dir, "task_bundle"))
    if adapted is None:
        return _fail(
            "cross_task_failed",
            f"cross-task {current.name!r} adapted control has no evaluated program",
        )
    return {"adapted_control": adapted}


async def prepare_variants(
    state: CrossTaskState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Phase 2: one workspace per other variant, seeded with the adapted control."""
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).variants
    current = _current(state)
    others = [s for s in state["specs"] if s.variant_id != CONTROL_BEST_ID]
    adapted = state.get("adapted_control")
    if others and adapted is None:
        return _fail(
            "cross_task_failed", f"cross-task {current.name!r} has no verified adapted control"
        )
    cross_dir = _cross_dir(state)
    cross_specs: list[VariantSpec] = []
    for orig in others:
        orig_bundle = join(orig.variant_dir, "task_bundle")
        best = await find_best_program_dir(ws, orig_bundle)
        if best is None:
            continue
        variant_dir = join(cross_dir, VARIANTS_SUBDIR, orig.variant_id)
        task_bundle = join(variant_dir, "task_bundle")
        assert adapted is not None
        await setup_task_bundle(
            ws,
            task_bundle=task_bundle,
            retrofitted_extra=join(cross_dir, RETROFITTED_BUNDLE_DIR),
            program_src=adapted,
            original_task_md_text=state["original_task_md"],
        )
        original_variant = join(task_bundle, "original_variant_program")
        await ws.remove(original_variant)
        await clone_program(ws, src_dir=best, dst_dir=original_variant)
        await write_variant_cross_task_md(
            ws,
            ctx.prompts,
            task_bundle,
            orig,
            current.name,
            has_adapted_control=True,
            max_evals=cfg.max_evals_per_variant,
        )
        cross_specs.append(
            await finalize_variant(
                ws,
                ctx.prompts,
                variant_dir=variant_dir,
                task_bundle=task_bundle,
                orig_spec=orig,
                task_name=current.name,
                agent="langgraph-deepagents",
                agent_model=model_label(ctx, Role.VARIANT_EDITOR),
            )
        )
    return {"cross_specs": cross_specs}


def route_variants(state: Mapping[str, Any]) -> list[Send] | str:
    """Fan out the other variants of the current extra task, or go to the join."""
    if state.get("cross_failure") is not None:
        return "finish"
    specs = state.get("cross_specs") or []
    if not specs:
        return "join_task"
    return [Send("run_adapt_variant", _adapt_input(state, spec)) for spec in specs]


async def join_task(state: CrossTaskState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """End one extra task: fail on an evaluator infrastructure failure, else go to the next."""
    infrastructure = next(
        (
            run.infrastructure_failure
            for run in state.get("variant_runs", [])
            if run.infrastructure_failure is not None
        ),
        None,
    )
    if infrastructure is not None:
        return {"cross_failure": infrastructure, "route": "finish"}
    index = state["index"] + 1
    return {"index": index, "route": "next" if index < len(state["extra_tasks"]) else "finish"}


async def cross_finish(state: CrossTaskState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Collect the cross-task runs and the failure of the stage."""
    return {
        "cross_task_runs": list(state.get("variant_runs", [])),
        "cross_failure": state.get("cross_failure"),
    }


def _failed(state: Mapping[str, Any]) -> str:
    return "finish" if state.get("cross_failure") is not None else "ok"


def build_cross_task_graph(
    *,
    checkpointer: Any = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "cross_task",
    run_variant_graph: CompiledStateGraph[Any, Any, Any, Any] | None = None,
) -> CompiledStateGraph[Any, ScientistContext, CrossTaskInput, CrossTaskOutput]:
    """Return the compiled cross-task graph (see the module docstring).

    Args:
        checkpointer: Only for running the graph alone.
        store: Only for running the graph alone.
        overrides: Replacement slots; ``variants.run_variant`` replaces the graph that runs one
            variant.
        name: The graph's name.
        run_variant_graph: The compiled graph that runs one variant, when the caller has built
            it already (the variants graph shares its own); by default the ``variants.run_variant``
            slot of ``overrides`` or :func:`~langgraph_agentic_scientist.variants.execute.
            build_run_variant_graph`.
    """
    if run_variant_graph is None:
        slot = (overrides or GraphOverrides()).slots.get("variants.run_variant")
        if slot is not None and not isinstance(slot, CompiledStateGraph):
            raise TypeError("slot variants.run_variant must be a compiled graph")
        run_variant_graph = slot or build_run_variant_graph(overrides=overrides)
    graph = StepGraph(
        "cross_task",
        CrossTaskState,
        input_schema=CrossTaskInput,
        output_schema=CrossTaskOutput,
        overrides=overrides,
    )
    retry = agent_node_retry_policy()
    graph.step("prepare", cross_prepare)
    graph.step("start_task", start_task)
    graph.step("retrofit_prepare", cross_retrofit_prepare)
    graph.step("retrofit_agent", cross_retrofit_agent, retry_policy=retry)
    graph.step("retrofit_finish", cross_retrofit_finish)
    graph.step("prepare_control", prepare_control)
    graph.add_node("run_control", run_variant_graph)
    graph.step("after_control", after_control)
    graph.step("prepare_variants", prepare_variants)
    graph.add_node("run_adapt_variant", run_variant_graph)
    graph.step("join_task", join_task)
    graph.step("finish", cross_finish)

    graph.graph.add_edge(START, graph.entry("prepare"))
    graph.add_conditional_edges(
        "prepare",
        lambda s: (
            "finish" if s.get("cross_failure") is not None or s.get("route") == "finish" else "ok"
        ),
        {"ok": "start_task", "finish": "finish"},
    )
    graph.add_conditional_edges(
        "start_task", _failed, {"ok": "retrofit_prepare", "finish": "finish"}
    )
    graph.add_conditional_edges(
        "retrofit_prepare",
        lambda s: (
            "finish"
            if s.get("cross_failure") is not None
            else ("agent" if s.get("route") == "agent" else "adapt")
        ),
        {"agent": "retrofit_agent", "adapt": "prepare_control", "finish": "finish"},
    )
    graph.add_edge("retrofit_agent", "retrofit_finish")
    graph.add_conditional_edges(
        "retrofit_finish", _failed, {"ok": "prepare_control", "finish": "finish"}
    )
    graph.add_conditional_edges(
        "prepare_control", route_control, ["run_control", "after_control", "finish"]
    )
    graph.graph.add_edge("run_control", graph.entry("after_control"))
    graph.add_conditional_edges(
        "after_control", _failed, {"ok": "prepare_variants", "finish": "finish"}
    )
    graph.add_conditional_edges(
        "prepare_variants", route_variants, ["run_adapt_variant", "join_task", "finish"]
    )
    graph.graph.add_edge("run_adapt_variant", graph.entry("join_task"))
    graph.add_conditional_edges(
        "join_task", lambda s: s.get("route", "finish"), {"next": "start_task", "finish": "finish"}
    )
    graph.graph.add_edge(graph.exit("finish"), END)
    return graph.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "CrossTaskInput",
    "CrossTaskOutput",
    "ExtraTask",
    "build_cross_task_graph",
    "finalize_variant",
    "find_best_program_dir",
    "find_original_task_md",
    "read_retrofitted_evaluator",
    "reference_evaluator_section",
    "setup_task_bundle",
    "write_control_cross_task_md",
    "write_variant_cross_task_md",
]
