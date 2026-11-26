"""Stage 1: prepare an analysis-ready bundle with checkpoint guardrails (upstream ``bootstrap.py``).

The stage is a loop of graph steps with its counter in state:

``bootstrap_prepare`` copies the extracted bundle (without evidence) to the analysis-ready bundle,
promotes the runtime artifacts of the original best program into its ``program_0`` and validates
what is there. A usable checkpoint (or ``checkpoint_mode='optional'``) that passes the primary
metric guardrail finishes the stage. Otherwise, when the checkpoint is required and
``retrain_if_checkpoint_missing`` is true, up to ``bootstrap_max_attempts`` attempts run, each:
``bootstrap_reset`` (a fresh copy), ``bootstrap_agent`` (role R23 edits ``program_0`` so that it
saves a checkpoint), ``bootstrap_evaluate`` (the evaluator reruns once, through ``ctx.evaluator``
under the id ``<run>/posthoc/bootstrap_<n>``, plane ``bootstrap``) and ``bootstrap_record``
(checkpoint and guardrail validation). A total failure resets the bundle and ends ``degraded``;
the run goes on, as upstream's does. ``bootstrap_finish`` writes ``bootstrap_report.json``.

Adaptation: upstream's rerun wrote the evaluator's results into ``program_0``. The evaluator of
the port keeps its outputs in its own area, so the host promotes them: ``metrics.json`` and
``correct.json`` from the validated result, and every other file of the evaluator's
``output_dir`` when the workspace can read it (a saved checkpoint, ``_results.json``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.runtime import Runtime

from ..agents.runner import run_workspace_role
from ..context import ScientistContext
from ..core.state import Event, Failure, ProgramRef, TaskRef, utc_now
from ..evaluation.protocol import EvaluationRequest
from ..models import Role
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .common import (
    compare_metric_guardrail,
    copy_files,
    copy_selected_runtime_artifacts,
    find_checkpoint_candidates,
    infer_primary_metric,
    json_file_text,
    read_json,
    read_text_or_empty,
    resolve_program_entrypoint,
    validate_checkpoint_loadable,
    write_json,
)
from .layout import Layout
from .models import BootstrapReport, PosthocState
from .settings import posthoc_config

PHASE = "posthoc"
EMPTY_CHECKPOINT_REPORT: dict[str, Any] = {
    "exists": False,
    "paths": [],
    "selected_path": None,
    "load_error": None,
}


class BootstrapError(RuntimeError):
    """Bootstrap cannot produce a valid checkpoint-ready bundle (upstream ``BootstrapError``)."""


# --- host logic ---------------------------------------------------------------------------------


async def reset_bootstrap_bundle(
    workspace: Workspace, bundle_dir: str, ready_dir: str, source_program_dir: str
) -> dict[str, Any]:
    """Copy the extracted bundle to the analysis-ready bundle and promote runtime artifacts.

    The evidence is left out (the bootstrap agent must not see or change it; it is restored
    after the stage), and the original best program's metrics, correctness, results, analysis
    directories and checkpoints are copied into the new ``program_0``.
    """
    await workspace.remove(ready_dir)
    await workspace.copy_tree(bundle_dir, ready_dir)
    await workspace.remove(join(ready_dir, "evidence"))
    await workspace.remove(join(ready_dir, "evidence_manifest.json"))
    promoted: dict[str, Any] = {"files": [], "directories": [], "checkpoints": []}
    entry = await workspace.stat(source_program_dir)
    if entry is not None and entry.kind == "dir":
        promoted = await copy_selected_runtime_artifacts(
            workspace, source_program_dir, join(ready_dir, "program_0")
        )
    return promoted


async def load_baseline_metrics(
    workspace: Workspace, source_program_dir: str, extraction: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the baseline metrics of the extracted winner (upstream ``_load_baseline_metrics``).

    Raises:
        BootstrapError: When neither ``metrics.json`` nor the extraction manifest has a score.
    """
    entry = await workspace.stat(source_program_dir)
    metrics = (
        await read_json(workspace, join(source_program_dir, "metrics.json"))
        if entry is not None and entry.kind == "dir"
        else None
    )
    if metrics:
        return metrics
    combined_score = extraction.get("combined_score")
    if isinstance(combined_score, int | float) and not isinstance(combined_score, bool):
        return {"combined_score": float(combined_score)}
    raise BootstrapError("Unable to determine baseline metrics for bootstrap guardrail")


async def checkpoint_report(workspace: Workspace, program_dir: str) -> dict[str, Any]:
    """Report the checkpoint candidates of a program and whether the largest one loads."""
    candidates = await find_checkpoint_candidates(workspace, program_dir)
    if not candidates:
        return dict(EMPTY_CHECKPOINT_REPORT)
    selected, _size = max(candidates, key=lambda item: item[1])
    loadable, load_error = await validate_checkpoint_loadable(
        workspace, join(program_dir, selected)
    )
    return {
        "exists": True,
        "paths": [path for path, _ in candidates],
        "selected_path": selected,
        "load_error": None if loadable else load_error,
    }


async def validate_bootstrap_bundle(
    workspace: Workspace,
    target_program_dir: str,
    *,
    baseline_primary: dict[str, Any],
    primary_metric_key: str | None,
    primary_metric_tolerance: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Validate correctness, metrics and the primary metric guardrail of ``program_0``.

    Returns ``(current primary metric, guardrail, error text)``.
    """
    current_correct = await read_json(workspace, join(target_program_dir, "correct.json"))
    if not current_correct or current_correct.get("correct") is not True:
        return None, None, "analysis-ready bundle failed correctness validation"

    current_metrics = await read_json(workspace, join(target_program_dir, "metrics.json"))
    current_results = await read_json(workspace, join(target_program_dir, "_results.json"))
    if current_metrics is None:
        return None, None, "analysis-ready bundle is missing metrics.json after bootstrap"

    try:
        current_primary = infer_primary_metric(
            metrics=current_metrics, results=current_results, requested_key=primary_metric_key
        )
    except RuntimeError as exc:
        return None, None, str(exc)
    guardrail = compare_metric_guardrail(
        baseline=baseline_primary, candidate=current_primary, tolerance=primary_metric_tolerance
    )
    if not guardrail["passed"]:
        reason = guardrail["reason"] or "primary metric regressed beyond tolerance"
        return current_primary, guardrail, f"bootstrap guardrail failed: {reason}"
    return current_primary, guardrail, None


async def build_bootstrap_prompt(
    workspace: Workspace, ctx: ScientistContext, ready_dir: str, task: TaskRef
) -> str:
    """Render ``posthoc_bootstrap.md`` from the files of the analysis-ready bundle."""
    program_source = ""
    entrypoint = await resolve_program_entrypoint(workspace, join(ready_dir, "program_0"))
    entry = await workspace.stat(entrypoint)
    if entry is not None and entry.kind == "file":
        program_source = (await workspace.read_bytes(entrypoint)).decode("utf-8")
    evaluator = task.contract.evaluators.search if task.contract is not None else "evaluate.py"
    values = {
        "TASK_MD": await read_text_or_empty(workspace, join(ready_dir, "task.md")),
        "PROGRAM_SOURCE": program_source,
        "EVALUATOR_SOURCE": await read_text_or_empty(workspace, join(ready_dir, evaluator)),
        "RESULTS_JSON": await read_text_or_empty(
            workspace, join(ready_dir, "program_0", "_results.json")
        ),
    }
    return ctx.prompts.render("posthoc/posthoc_bootstrap", values, fragments=False)


async def promote_evaluation(
    workspace: Workspace,
    program_dir: str,
    result_output_dir: str | None,
    metrics: Any,
    correct: Any,
) -> None:
    """Make the evaluator's outputs the contents of ``program_dir``, as upstream's rerun did."""
    if result_output_dir:
        entry = await workspace.stat(result_output_dir)
        if entry is not None and entry.kind == "dir":
            await copy_files(
                workspace,
                result_output_dir,
                program_dir,
                await workspace.list_tree(result_output_dir),
            )
    await write_json(workspace, join(program_dir, "metrics.json"), metrics)
    await write_json(workspace, join(program_dir, "correct.json"), {"correct": correct})


def _scratch(state: Mapping[str, Any]) -> dict[str, Any]:
    return dict(state.get("bootstrap_scratch") or {})


def _phase_error(category: str, detail: str) -> dict[str, Any]:
    return {
        "phase_error": Failure(category=category, scope="task", detail=detail),
        "events": [Event(kind="bootstrap_error", phase=PHASE, data={"detail": detail})],
    }


# --- graph steps --------------------------------------------------------------------------------


async def bootstrap_prepare(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Compute the baseline, copy the bundle, and validate what is already there."""
    workspace = runtime.context.workspace
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    extraction = state["extraction"]
    assert extraction is not None
    source_program_dir = extraction["source_best_program_dir"]
    try:
        baseline_metrics = await load_baseline_metrics(workspace, source_program_dir, extraction)
        entry = await workspace.stat(source_program_dir)
        baseline_results = (
            await read_json(workspace, join(source_program_dir, "_results.json"))
            if entry is not None and entry.kind == "dir"
            else None
        )
        baseline_primary = infer_primary_metric(
            metrics=baseline_metrics,
            results=baseline_results,
            requested_key=config.primary_metric_key,
        )
    except (BootstrapError, RuntimeError) as exc:
        return _phase_error("bootstrap_baseline_unavailable", str(exc))

    promoted = await reset_bootstrap_bundle(
        workspace, layout.bundle, layout.ready, source_program_dir
    )
    target = join(layout.ready, "program_0")
    report = await checkpoint_report(workspace, target)
    scratch: dict[str, Any] = {
        "promoted": promoted,
        "checkpoint": report,
        "current_primary": None,
        "guardrail": None,
        "agent": None,
        "rerun": None,
        "reason": None,
    }
    status = "pending"
    usable = report["exists"] and report["load_error"] is None
    if usable or config.checkpoint_mode == "optional":
        current, guardrail, error = await validate_bootstrap_bundle(
            workspace,
            target,
            baseline_primary=baseline_primary,
            primary_metric_key=config.primary_metric_key,
            primary_metric_tolerance=config.primary_metric_tolerance,
        )
        if error is None:
            status = "completed"
            scratch.update(current_primary=current, guardrail=guardrail)
        else:
            scratch["reason"] = error
    else:
        scratch["reason"] = report["load_error"]
    return {
        "baseline_primary": baseline_primary,
        "bootstrap_attempt": 0,
        "bootstrap_attempts": [],
        "bootstrap_scratch": scratch,
        "bootstrap_status": status,
        "bootstrap_error": None,
    }


def route_after_prepare(state: PosthocState) -> str:
    """Finish, run the attempt loop, or degrade (upstream's conditions)."""
    if state.get("phase_error") is not None:
        return "stop"
    if state.get("bootstrap_status") == "completed":
        return "finish"
    config = posthoc_config(state)
    if config.checkpoint_mode == "required" and config.retrain_if_checkpoint_missing:
        return "attempt"
    return "finish"


async def bootstrap_reset(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Start attempt ``n``: a fresh copy of the bundle with the runtime artifacts promoted."""
    layout = Layout.of(state.get("output_dir"))
    extraction = state["extraction"]
    assert extraction is not None
    promoted = await reset_bootstrap_bundle(
        runtime.context.workspace,
        layout.bundle,
        layout.ready,
        extraction["source_best_program_dir"],
    )
    scratch = _scratch(state)
    scratch.update(attempt_promoted=promoted, attempt_agent=None, attempt_rerun=None)
    return {
        "bootstrap_attempt": int(state.get("bootstrap_attempt") or 0) + 1,
        "bootstrap_scratch": scratch,
        "bootstrap_error": None,
    }


async def bootstrap_agent(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the bootstrap agent (role R23) on the analysis-ready bundle."""
    ctx = runtime.context
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    prompt = await build_bootstrap_prompt(ctx.workspace, ctx, layout.ready, state["task"])
    outcome = await run_workspace_role(
        role=Role.BOOTSTRAP_AGENT,
        runtime=runtime,
        root=layout.ready,
        prompt=prompt,
        timeout=config.bootstrap_timeout_seconds,
    )
    exit_code = 124 if outcome.timed_out else (1 if outcome.failure is not None else 0)
    report: dict[str, Any] = {
        "exit_code": exit_code,
        "timed_out": outcome.timed_out,
        "final_text": outcome.final_text,
        "failure": outcome.failure.model_dump(mode="json") if outcome.failure else None,
    }
    scratch = _scratch(state)
    scratch["attempt_agent"] = report
    scratch["agent"] = report
    update: dict[str, Any] = {"bootstrap_scratch": scratch}
    if exit_code != 0:
        update["bootstrap_error"] = f"Bootstrap agent exited with code {exit_code}"
    return update


def route_after_agent(state: PosthocState) -> str:
    """Evaluate when the agent finished; record the failure otherwise."""
    return "record" if state.get("bootstrap_error") else "evaluate"


async def bootstrap_evaluate(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Rerun the official evaluator once on ``program_0`` (nothing precedes the call)."""
    ctx = runtime.context
    layout = Layout.of(state.get("output_dir"))
    attempt = int(state.get("bootstrap_attempt") or 1)
    program_dir = join(layout.ready, "program_0")
    evaluator = ctx.evaluator
    if evaluator is None:
        return {"bootstrap_error": "Bootstrap evaluator rerun failed: no evaluator is configured"}
    entrypoint = await resolve_entrypoint_relative(ctx.workspace, program_dir)
    request = EvaluationRequest(
        evaluation_id=f"{state['run_id']}/posthoc/bootstrap_{attempt:02d}",
        plane="bootstrap",
        program=ProgramRef(path=program_dir, entrypoint=entrypoint),
        task=state["task"],
    )
    result = await evaluator.evaluate(request)
    scratch = _scratch(state)
    scratch["attempt_rerun"] = {
        "evaluation_id": result.evaluation_id,
        "status": result.status,
        "returncode": result.returncode,
        "summary": result.summary,
        "output_dir": result.output_dir,
    }
    update: dict[str, Any] = {"bootstrap_scratch": scratch}
    if result.status != "completed":
        code = result.returncode if result.returncode is not None else 1
        update["bootstrap_error"] = f"Bootstrap evaluator rerun failed with exit code {code}"
        return update
    await promote_evaluation(
        ctx.workspace, program_dir, result.output_dir, result.metrics, result.correct
    )
    return update


async def resolve_entrypoint_relative(workspace: Workspace, program_dir: str) -> str:
    """Return the entrypoint of a program relative to its directory."""
    from ..evaluation.program_spec import resolve_entrypoint

    return await resolve_entrypoint(workspace, program_dir)


async def bootstrap_record(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Validate the attempt's checkpoint and guardrail and record the attempt report."""
    workspace = runtime.context.workspace
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    attempt = int(state.get("bootstrap_attempt") or 1)
    scratch = _scratch(state)
    program_dir = join(layout.ready, "program_0")
    error = state.get("bootstrap_error")
    report: dict[str, Any] = dict(EMPTY_CHECKPOINT_REPORT)
    current: dict[str, Any] | None = None
    guardrail: dict[str, Any] | None = None
    if error is None:
        report = await checkpoint_report(workspace, program_dir)
        baseline = state["baseline_primary"]
        assert baseline is not None
        current, guardrail, error = await validate_bootstrap_bundle(
            workspace,
            program_dir,
            baseline_primary=baseline,
            primary_metric_key=config.primary_metric_key,
            primary_metric_tolerance=config.primary_metric_tolerance,
        )
        if report["load_error"] is not None and error is None:
            error = report["load_error"]
        if not report["exists"] and error is None:
            error = "bootstrap completed but no checkpoint was found"
    attempt_report = {
        "attempt_index": attempt,
        "promoted_runtime_artifacts": scratch.get("attempt_promoted"),
        "bootstrap_agent": scratch.get("attempt_agent"),
        "rerun": scratch.get("attempt_rerun"),
        "checkpoint": report,
        "current_primary_metric": current,
        "guardrail": guardrail,
        "status": "completed" if error is None else "failed",
        "error": error,
    }
    attempts = [*(state.get("bootstrap_attempts") or []), attempt_report]
    update: dict[str, Any] = {"bootstrap_attempts": attempts, "bootstrap_error": error}
    if error is None:
        scratch.update(
            promoted=scratch.get("attempt_promoted"),
            checkpoint=report,
            current_primary=current,
            guardrail=guardrail,
            agent=scratch.get("attempt_agent"),
            rerun=scratch.get("attempt_rerun"),
            reason=None,
        )
        update["bootstrap_status"] = "completed"
    else:
        scratch["reason"] = error
    update["bootstrap_scratch"] = scratch
    return update


def route_after_record(state: PosthocState) -> str:
    """Finish on success; another attempt while the budget lasts; else finish degraded."""
    if state.get("bootstrap_status") == "completed":
        return "finish"
    if int(state.get("bootstrap_attempt") or 0) < posthoc_config(state).bootstrap_max_attempts:
        return "attempt"
    return "finish"


async def bootstrap_finish(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Reset a failed bootstrap, then write ``bootstrap_report.json``."""
    workspace = runtime.context.workspace
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    extraction = state["extraction"]
    baseline = state["baseline_primary"]
    assert extraction is not None and baseline is not None
    scratch = _scratch(state)
    status = "completed" if state.get("bootstrap_status") == "completed" else "degraded"
    promoted = scratch.get("promoted")
    report = scratch.get("checkpoint")
    current = scratch.get("current_primary")
    guardrail = scratch.get("guardrail")
    if status != "completed":
        source_program_dir = extraction["source_best_program_dir"]
        promoted = await reset_bootstrap_bundle(
            workspace, layout.bundle, layout.ready, source_program_dir
        )
        target = join(layout.ready, "program_0")
        report = await checkpoint_report(workspace, target)
        current, guardrail, _ = await validate_bootstrap_bundle(
            workspace,
            target,
            baseline_primary=baseline,
            primary_metric_key=config.primary_metric_key,
            primary_metric_tolerance=config.primary_metric_tolerance,
        )
    record = BootstrapReport(
        status=status,
        bootstrapped_at=utc_now(),
        source_bundle_dir=layout.bundle,
        analysis_ready_bundle_dir=layout.ready,
        source_program_dir=extraction["source_best_program_dir"],
        promoted_runtime_artifacts=promoted,
        baseline_primary_metric=baseline,
        current_primary_metric=current,
        guardrail=guardrail,
        checkpoint=report,
        bootstrap_agent=scratch.get("agent"),
        rerun=scratch.get("rerun"),
        retry_attempts=list(state.get("bootstrap_attempts") or []),
        bootstrap_ready=status == "completed",
        failure_reason=None if status == "completed" else scratch.get("reason"),
        retrain_if_checkpoint_missing=config.retrain_if_checkpoint_missing,
        bootstrap_max_attempts=config.bootstrap_max_attempts,
        checkpoint_mode=config.checkpoint_mode,
    )
    await workspace.write_bytes(
        layout.bootstrap_report, json_file_text(record.model_dump(mode="json")).encode("utf-8")
    )
    return {
        "bootstrap": record,
        "events": [
            Event(
                kind="bootstrap_report",
                phase=PHASE,
                data={"status": status, "ready": record.bootstrap_ready},
            )
        ],
    }


__all__ = [
    "BootstrapError",
    "bootstrap_agent",
    "bootstrap_evaluate",
    "bootstrap_finish",
    "bootstrap_prepare",
    "bootstrap_record",
    "bootstrap_reset",
    "build_bootstrap_prompt",
    "checkpoint_report",
    "load_baseline_metrics",
    "promote_evaluation",
    "reset_bootstrap_bundle",
    "route_after_agent",
    "route_after_prepare",
    "route_after_record",
    "validate_bootstrap_bundle",
]
