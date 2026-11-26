"""Stage 5: execute one post-hoc analysis (upstream ``execute.py`` and ``posthoc_agent.sh``).

One analysis is a subgraph (``build_analysis_graph``), run once per selected analysis through
``Send``. Its attempt loop is a graph loop with the counter in state, so every attempt is its own
agent session with its own checkpoint namespace:

``analysis_begin`` (digest and snapshot the pristine ``task_bundle``) -> ``attempt_prepare`` (a
fresh ``attempt_<n>/``: attempt 1 holds a copy of the bundle, later attempts a copy of the
previous attempt with the pristine bundle put back) -> ``attempt_execute`` (role R26) ->
``attempt_inspect`` (restore the pristine bundle when it changed, inspect the outputs, write
``attempt_result.json``) -> the next attempt, or ``analysis_finish`` (``analysis_result.json``).

Role R26 gets the ``run`` skill set (``edit-program``, ``evaluate-program``, ``verify-rubrics``,
``memory-log``), the ``clone_program`` tool and an ``evaluate_program`` tool whose budget is 0, as
upstream's ``--max-program-evals 0`` gives it: the analysis is analysis-only, so the tool refuses
every call. Upstream's rules for an attempt are kept: it passes only when the required root files
exist, ``rubric_status.json`` and ``analysis_manifest.json`` validate, every expected output
exists, the pristine bundle is unchanged, ``reran_evaluator`` is false and (unless the checkpoint
mode is ``optional``) a usable checkpoint is still available. A timed-out session ends the loop
with status ``timeout``.
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..agents.runner import agent_node_retry_policy, run_workspace_role
from ..agents.staging import skills_for_step
from ..agents.tools import (
    EvaluateProgramSettings,
    make_clone_program_tool,
    make_evaluate_program_tool,
)
from ..context import GraphOverrides, ScientistContext
from ..core.render import standard_values
from ..core.runconfig import record_run_config
from ..core.state import Event, utc_now
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.paths import join, safe_relpath
from ..workspace.protocol import Workspace
from .common import (
    find_checkpoint_candidates,
    list_program_dirs,
    read_json,
    read_text_or_empty,
    validate_checkpoint_loadable,
    validate_expected_outputs,
    write_json,
)
from .layout import Layout
from .models import (
    AnalysisInput,
    AnalysisOutput,
    AnalysisRunResult,
    AnalysisSpec,
    AnalysisState,
    AttemptInspection,
)
from .rubric_contract import (
    load_rubric_status,
    normalize_posthoc_rubric_status,
    validate_posthoc_rubric_status,
)
from .settings import posthoc_config

PHASE = "posthoc.analysis"
REQUIRED_ROOT_FILES = (
    "analysis_summary.md",
    "interestingness_rubric.json",
    "analysis_manifest.json",
    "rubric_status.json",
)
RETRAIN_ERROR = (
    "post-hoc analysis pipeline is analysis-only; use the dedicated ablation pipeline "
    "for retraining or evaluator-rerun analyses"
)
TIMEOUT_EXIT_CODE = 124


# --- inspection (upstream _inspect_attempt and its helpers) --------------------------------------


async def resolve_attempt_file(
    workspace: Workspace, attempt_dir: str, relpath: object
) -> str | None:
    """Return the workspace path of a regular file named relative to the attempt root, or ``None``.

    The path must be a non-empty relative string that stays inside the attempt root; the file must
    be a regular file, not a symlink.
    """
    if not isinstance(relpath, str) or not relpath or relpath.startswith("/"):
        return None
    try:
        relative = safe_relpath(relpath, allow_root=False)
    except ValueError:
        return None
    path = join(attempt_dir, relative)
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        return None
    return path


def validate_analysis_manifest(
    spec: AnalysisSpec, manifest: dict[str, Any] | None
) -> tuple[bool, str | None]:
    """Check the required keys of ``analysis_manifest.json`` (without the path checks)."""
    if manifest is None:
        return False, "missing or invalid analysis_manifest.json"
    for key in (
        "analysis_id",
        "attempt_id",
        "summary",
        "claim",
        "grounded",
        "reran_evaluator",
        "outputs",
        "evidence_paths",
        "checkpoint_paths",
    ):
        if key not in manifest:
            return False, f"analysis_manifest missing key: {key}"
    if manifest.get("analysis_id") != spec.analysis_id:
        return False, "analysis_manifest analysis_id mismatch"
    if manifest.get("reran_evaluator") is not False:
        return False, "analysis_manifest reran_evaluator must be false"
    if not isinstance(manifest.get("evidence_paths", []), list):
        return False, "analysis_manifest.evidence_paths must be a list"
    return True, None


async def validate_manifest_paths(
    workspace: Workspace, attempt_dir: str, manifest: dict[str, Any]
) -> str | None:
    """Return an error when a declared evidence path is missing or unsafe, else ``None``."""
    missing = [
        relpath
        for relpath in manifest.get("evidence_paths", [])
        if await resolve_attempt_file(workspace, attempt_dir, relpath) is None
    ]
    if missing:
        return f"analysis_manifest references missing evidence paths: {missing}"
    if manifest.get("grounded") is not True:
        return "analysis_manifest must declare grounded=true"
    return None


async def validate_attempt_runtime(
    workspace: Workspace,
    attempt_dir: str,
    manifest: dict[str, Any] | None,
    *,
    checkpoint_mode: str,
    expected_task_bundle_digest: str | None,
    source_task_bundle_mutated: bool,
) -> tuple[bool, bool | None, str | None]:
    """Return ``(checkpoint validated, guardrail passed, error)`` for an attempt."""
    manifest = manifest or {}
    if source_task_bundle_mutated:
        return False, False, "source task_bundle was modified during analysis execution"
    if expected_task_bundle_digest is not None:
        try:
            digest = await workspace.digest(join(attempt_dir, "task_bundle"))
        except (OSError, ValueError) as exc:
            return False, False, f"task_bundle inspection rejected: {exc}"
        if digest != expected_task_bundle_digest:
            return False, False, "task_bundle was modified during analysis execution"
    if manifest.get("reran_evaluator") is not False:
        return (
            False,
            False,
            "analysis_manifest reran_evaluator=true but this post-hoc pipeline is analysis-only",
        )

    checkpoint_paths = manifest.get("checkpoint_paths", [])
    candidates: list[str] = []
    if not isinstance(checkpoint_paths, list):
        return False, None, "analysis_manifest.checkpoint_paths must be a list"
    for relpath in checkpoint_paths:
        candidate = await resolve_attempt_file(workspace, attempt_dir, relpath)
        if candidate is None:
            return (
                False,
                None,
                "analysis_manifest.checkpoint_paths contains an unsafe or missing file",
            )
        candidates.append(candidate)

    if checkpoint_mode == "optional" and not checkpoint_paths:
        return False, None, None

    if not candidates:
        base_program = join(attempt_dir, "task_bundle", "program_0")
        candidates = [
            join(base_program, path)
            for path, _ in await find_checkpoint_candidates(workspace, base_program)
        ]
        if not candidates:
            programs = await list_program_dirs(workspace, join(attempt_dir, "task_bundle"))
            if programs:
                candidates = [
                    join(programs[-1], path)
                    for path, _ in await find_checkpoint_candidates(workspace, programs[-1])
                ]

    if not candidates:
        if checkpoint_mode == "optional":
            return False, None, None
        return False, None, "no checkpoint artifact found for analysis attempt"

    for checkpoint in candidates:
        loadable, load_error = await validate_checkpoint_loadable(workspace, checkpoint)
        if not loadable:
            return False, None, load_error

    return True, None, None


async def inspect_attempt(
    workspace: Workspace,
    spec: AnalysisSpec,
    attempt_dir: str,
    *,
    checkpoint_mode: str = "required",
    expected_task_bundle_digest: str | None = None,
    source_task_bundle_mutated: bool = False,
) -> AttemptInspection:
    """Inspect one attempt directory with upstream's rules."""
    attempt_id = attempt_dir.rsplit("/", 1)[-1]
    missing_root_files = [
        name for name in REQUIRED_ROOT_FILES if not await workspace.exists(join(attempt_dir, name))
    ]
    if missing_root_files:
        return AttemptInspection(
            attempt_dir=attempt_dir,
            attempt_id=attempt_id,
            overall_pass=False,
            rubric_error=f"missing attempt outputs: {missing_root_files}",
            required_outputs_present=False,
            missing_outputs=missing_root_files,
            manifest_valid=False,
            manifest_error="missing required attempt files",
        )

    raw_rubric_status = await load_rubric_status(workspace, join(attempt_dir, "rubric_status.json"))
    rubric_status, rubric_normalize_error = normalize_posthoc_rubric_status(raw_rubric_status)
    rubric_ok, rubric_error = validate_posthoc_rubric_status(raw_rubric_status, spec.analysis_id)
    if rubric_error is None and rubric_normalize_error is not None:
        rubric_error = rubric_normalize_error

    manifest = await read_json(workspace, join(attempt_dir, "analysis_manifest.json"))
    manifest_ok, manifest_error = validate_analysis_manifest(spec, manifest)
    if manifest_ok and manifest is not None:
        path_error = await validate_manifest_paths(workspace, attempt_dir, manifest)
        if path_error is not None:
            manifest_ok, manifest_error = False, path_error

    outputs_check = await validate_expected_outputs(workspace, attempt_dir, spec.expected_outputs)
    if manifest:
        manifest_outputs = manifest.get("outputs", {})
        if isinstance(manifest_outputs, dict):
            manifest_output_values: list[str] = []
            for value in manifest_outputs.values():
                if isinstance(value, list):
                    manifest_output_values.extend(str(item) for item in value)
            manifest_check = await validate_expected_outputs(
                workspace, attempt_dir, manifest_output_values
            )
            outputs_check["missing"].extend(
                item for item in manifest_check["missing"] if item not in outputs_check["missing"]
            )
            outputs_check["found"].extend(
                item for item in manifest_check["found"] if item not in outputs_check["found"]
            )
            outputs_check["all_present"] = not outputs_check["missing"]

    checkpoint_validated, guardrail_passed, guardrail_error = await validate_attempt_runtime(
        workspace,
        attempt_dir,
        manifest,
        checkpoint_mode=checkpoint_mode,
        expected_task_bundle_digest=expected_task_bundle_digest,
        source_task_bundle_mutated=source_task_bundle_mutated,
    )

    overall_pass = bool(
        rubric_ok
        and manifest_ok
        and outputs_check["all_present"]
        and (checkpoint_validated or (checkpoint_mode == "optional" and guardrail_error is None))
        and (guardrail_passed is not False)
        and bool(rubric_status and rubric_status.get("overall_pass") is True)
    )
    if not overall_pass and manifest_error is None and guardrail_error is not None:
        manifest_error = guardrail_error

    return AttemptInspection(
        attempt_dir=attempt_dir,
        attempt_id=attempt_id,
        overall_pass=overall_pass,
        rubric_error=rubric_error,
        required_outputs_present=outputs_check["all_present"],
        missing_outputs=list(outputs_check["missing"]),
        manifest_valid=manifest_ok,
        manifest_error=manifest_error,
        guardrail_passed=guardrail_passed,
        checkpoint_validated=checkpoint_validated,
        manifest=manifest,
        rubric_status=rubric_status,
    )


async def produced_files(
    workspace: Workspace, analysis_dir: str, attempts: list[AttemptInspection]
) -> dict[str, bool]:
    """Return which root files any attempt produced, and whether ``analysis_result.json`` exists."""
    produced = dict.fromkeys(REQUIRED_ROOT_FILES, False)
    for attempt in attempts:
        for name in REQUIRED_ROOT_FILES:
            produced[name] = produced[name] or await workspace.exists(
                join(attempt.attempt_dir, name)
            )
    produced["analysis_result.json"] = await workspace.exists(
        join(analysis_dir, "analysis_result.json")
    )
    return produced


async def update_run_manifest(
    workspace: Workspace,
    analysis_dir: str,
    status: str,
    *,
    attempt_count: int,
    selected_attempt: str | None,
) -> None:
    """Update ``run_manifest.json`` as upstream's ``_update_run_manifest`` does."""
    path = join(analysis_dir, "run_manifest.json")
    manifest = await read_json(workspace, path)
    if manifest is None:
        return
    now = utc_now()
    if status == "running":
        manifest["started_at"] = now
        manifest["completed_at"] = None
    if status in {"completed", "failed", "timeout"}:
        manifest["completed_at"] = now
        if manifest.get("status") != "running":
            manifest["started_at"] = None
    manifest["status"] = status
    manifest["attempt_count"] = attempt_count
    manifest["selected_attempt"] = selected_attempt
    await write_json(workspace, path, manifest)


# --- the attempt prompt and the agent's workspace -----------------------------------------------


async def build_attempt_prompt(
    workspace: Workspace,
    ctx: ScientistContext,
    *,
    spec: AnalysisSpec,
    analysis_dir: str,
    attempt_index: int,
    previous_attempt: AttemptInspection | None,
) -> str:
    """Render ``posthoc_agent.md`` for one attempt (``ERROR_CONTEXT`` from the previous one)."""
    values = {
        "ANALYSIS_SPEC": json.dumps(spec.to_dict(), indent=2),
        "RUBRIC_CONTRACT": await read_text_or_empty(
            workspace, join(analysis_dir, "rubric_contract.json")
        ),
        "ATTEMPT_NUMBER": str(attempt_index),
        "ERROR_CONTEXT": (
            json.dumps(previous_attempt.to_dict(), indent=2) if previous_attempt is not None else ""
        ),
    }
    return ctx.prompts.render("posthoc/posthoc_agent", values, fragments=False)


async def prepare_attempt_dir(workspace: Workspace, spec: AnalysisSpec, attempt_index: int) -> str:
    """Create ``attempt_<n>/`` (upstream ``_prepare_attempt_dir``) and stage the active files."""
    analysis_dir = spec.analysis_dir
    attempt_dir = join(analysis_dir, f"attempt_{attempt_index}")
    pristine = join(analysis_dir, "task_bundle")
    await workspace.remove(attempt_dir)
    if attempt_index == 1:
        await workspace.copy_tree(pristine, join(attempt_dir, "task_bundle"))
    else:
        await workspace.copy_tree(join(analysis_dir, f"attempt_{attempt_index - 1}"), attempt_dir)
        await workspace.remove(join(attempt_dir, "task_bundle"))
        await workspace.copy_tree(pristine, join(attempt_dir, "task_bundle"))
    for filename in ("analysis_spec.json", "rubric_contract.json", "target_rubrics.json"):
        await workspace.write_bytes(
            join(attempt_dir, filename), await workspace.read_bytes(join(analysis_dir, filename))
        )
    # posthoc_agent.sh: the rubrics and the spec are also staged as the active idea files.
    await workspace.write_bytes(
        join(attempt_dir, "active_rubrics.json"),
        await workspace.read_bytes(join(analysis_dir, "target_rubrics.json")),
    )
    await workspace.write_bytes(
        join(attempt_dir, "active_idea.json"),
        await workspace.read_bytes(join(analysis_dir, "analysis_spec.json")),
    )
    return attempt_dir


async def skill_values(
    workspace: Workspace, spec: AnalysisSpec, attempt_dir: str, attempt_id: str
) -> dict[str, str]:
    """Return the placeholder values of the staged skills (``posthoc_agent.sh``)."""
    task_md = await read_text_or_empty(workspace, join(attempt_dir, "task_bundle", "task.md"))
    values = standard_values(task_dir="task_bundle", results_dir=".", task_md=task_md or None)
    active_idea = (
        await read_text_or_empty(workspace, join(attempt_dir, "active_idea.json"))
    ).strip()
    active_rubrics = (
        await read_text_or_empty(workspace, join(attempt_dir, "active_rubrics.json"))
    ).strip()
    for label, text in (("idea", active_idea), ("rubrics", active_rubrics)):
        try:
            json.loads(text)
        except json.JSONDecodeError:
            if label == "idea":
                active_idea = "{}"
            else:
                active_rubrics = "[]"
    values.update(
        {
            "ACTIVE_IDEA_JSON": active_idea or "{}",
            "ACTIVE_RUBRICS_JSON": active_rubrics or "[]",
            "ACTIVE_IDEA_FILE": "./active_idea.json",
            "ACTIVE_RUBRICS_FILE": "./active_rubrics.json",
            "ATTEMPT_ID": attempt_id,
            "IDEA_ID": spec.analysis_id,
            "IDEA_DIR_NAME": spec.analysis_id,
            "ATTEMPT_DIR_NAME": "",
            "MAX_PROGRAM_EVALS_PER_ATTEMPT": "0",
            "GLOBAL_MEMORY_MD_FILE": "./memory.md",
            "GLOBAL_MEMORY_JSONL_FILE": "./memory.jsonl",
            "GLOBAL_MEMORY_RECENT_SUMMARY": "- none",
        }
    )
    return values


# --- graph steps --------------------------------------------------------------------------------


async def analysis_begin(
    state: AnalysisState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Digest and snapshot the pristine ``task_bundle`` and mark the analysis running."""
    workspace = runtime.context.workspace
    spec = state["spec"]
    if spec.requires_retrain:
        return {
            "status": "failed",
            "attempt": 0,
            "inspections": [],
            "agent_outputs": [RETRAIN_ERROR],
        }
    await update_run_manifest(
        workspace, spec.analysis_dir, "running", attempt_count=0, selected_attempt=None
    )
    bundle = join(spec.analysis_dir, "task_bundle")
    return {
        "pristine_digest": await workspace.digest(bundle),
        "pristine": await workspace.snapshot(bundle),
        "attempt": 0,
        "inspections": [],
        "agent_outputs": [],
        "last_exit_code": 0,
        "status": "running",
    }


def route_after_begin(state: AnalysisState) -> str:
    """An analysis that requires retraining ends without an attempt."""
    return "finish" if state.get("status") == "failed" else "attempt"


async def attempt_prepare(
    state: AnalysisState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Start attempt ``n``: its directory, the active files, the copy of the pristine bundle."""
    attempt = int(state.get("attempt") or 0) + 1
    await prepare_attempt_dir(runtime.context.workspace, state["spec"], attempt)
    return {"attempt": attempt}


async def attempt_execute(
    state: AnalysisState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the analysis executor (role R26) for the current attempt."""
    ctx = runtime.context
    workspace = ctx.workspace
    spec = state["spec"]
    attempt = int(state["attempt"])
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    attempt_dir = join(spec.analysis_dir, f"attempt_{attempt}")
    attempt_id = f"attempt_{attempt}"
    inspections = state.get("inspections") or []
    previous = inspections[-1] if inspections else None
    prompt = await build_attempt_prompt(
        workspace,
        ctx,
        spec=spec,
        analysis_dir=spec.analysis_dir,
        attempt_index=attempt,
        previous_attempt=previous,
    )
    values = await skill_values(workspace, spec, attempt_dir, attempt_id)
    tools = [
        make_clone_program_tool(workspace, attempt_dir, prompts=ctx.prompts),
        make_evaluate_program_tool(
            workspace,
            ctx.evaluator,
            attempt_dir,
            EvaluateProgramSettings(
                task=state["task"],
                id_prefix=f"{state['run_id']}/posthoc/{spec.analysis_id}/{attempt_id}",
                budget=0,
            ),
            prompts=ctx.prompts,
        ),
    ]
    env = {
        "MAX_PROGRAM_EVALS_PER_ATTEMPT": "0",
        "ATTEMPT_ID": attempt_id,
        "IDEA_ID": spec.analysis_id,
        "IDEA_DIR_NAME": spec.analysis_id,
        "ATTEMPT_DIR_NAME": "",
        "ANALYSIS_ID": spec.analysis_id,
        "ANALYSIS_FAMILY": spec.family,
        "POSTHOC_RUN_DIR": workspace.host_path(layout.output),
    }
    outcome = await run_workspace_role(
        role=Role.ANALYSIS_EXECUTOR,
        runtime=runtime,
        root=attempt_dir,
        prompt=prompt,
        skills=skills_for_step("run"),
        skill_values=values,
        tools=tools,
        timeout=config.analysis_timeout_seconds,
        env=env,
    )
    exit_code = TIMEOUT_EXIT_CODE if outcome.timed_out else (1 if outcome.failure else 0)
    text = outcome.final_text
    if outcome.failure is not None:
        text = f"{text}\n[posthoc] agent session failed: {outcome.failure.detail}".strip()
    return {
        "last_exit_code": exit_code,
        "timed_out": outcome.timed_out,
        "agent_outputs": [*(state.get("agent_outputs") or []), text],
    }


async def attempt_inspect(
    state: AnalysisState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Restore the pristine bundle when it changed, inspect the attempt and record the result."""
    workspace = runtime.context.workspace
    spec = state["spec"]
    attempt = int(state["attempt"])
    config = posthoc_config(state)
    bundle = join(spec.analysis_dir, "task_bundle")
    pristine_digest = state.get("pristine_digest")
    try:
        mutated = await workspace.digest(bundle) != pristine_digest
    except (OSError, ValueError):
        mutated = True
    if mutated:
        snapshot = state.get("pristine")
        assert snapshot is not None
        await workspace.restore(snapshot)
    attempt_dir = join(spec.analysis_dir, f"attempt_{attempt}")
    inspection = await inspect_attempt(
        workspace,
        spec,
        attempt_dir,
        checkpoint_mode=config.checkpoint_mode,
        expected_task_bundle_digest=pristine_digest,
        source_task_bundle_mutated=mutated,
    )
    exit_code = int(state.get("last_exit_code") or 0)
    await write_json(
        workspace,
        join(attempt_dir, "attempt_result.json"),
        {"exit_code": exit_code, "inspection": inspection.to_dict()},
    )
    update: dict[str, Any] = {"inspections": [*(state.get("inspections") or []), inspection]}
    if exit_code == TIMEOUT_EXIT_CODE:
        update["status"] = "timeout"
    elif inspection.overall_pass:
        update["status"] = "completed"
    return update


def route_after_inspect(state: AnalysisState) -> str:
    """Finish on a timeout, a pass or the last attempt; otherwise run another attempt."""
    if state.get("status") in {"timeout", "completed"}:
        return "finish"
    if int(state.get("attempt") or 0) >= posthoc_config(state).max_attempts_per_analysis:
        return "finish"
    return "attempt"


async def analysis_finish(
    state: AnalysisState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Write ``analysis_result.json`` and return the result."""
    workspace = runtime.context.workspace
    spec = state["spec"]
    inspections = list(state.get("inspections") or [])
    status = state.get("status") or "failed"
    if status == "running":
        status = "failed"
    selected_attempt = None
    if status == "completed" and inspections:
        selected_attempt = inspections[-1].attempt_id
    selected = next((item for item in inspections if item.attempt_id == selected_attempt), None)
    if spec.requires_retrain:
        result = AnalysisRunResult(
            analysis_id=spec.analysis_id,
            status="failed",
            attempt_count=0,
            exit_code=1,
            selected_attempt=None,
            rubric_passed=False,
            required_outputs_present=False,
            guardrail_passed=False,
            output=RETRAIN_ERROR,
            produced_files=await produced_files(workspace, spec.analysis_dir, []),
            attempts=[],
        )
        await write_json(
            workspace, join(spec.analysis_dir, "analysis_result.json"), result.to_dict()
        )
        await update_run_manifest(
            workspace, spec.analysis_dir, "failed", attempt_count=0, selected_attempt=None
        )
        return {"analysis_runs": [result], "analysis_run": result}
    result = AnalysisRunResult(
        analysis_id=spec.analysis_id,
        status=status,
        attempt_count=len(inspections),
        exit_code=int(state.get("last_exit_code") or 0),
        selected_attempt=selected_attempt,
        rubric_passed=bool(selected and selected.overall_pass),
        required_outputs_present=bool(selected and selected.required_outputs_present),
        guardrail_passed=selected.guardrail_passed if selected else None,
        output="\n".join(state.get("agent_outputs") or []),
        produced_files=await produced_files(workspace, spec.analysis_dir, inspections),
        attempts=inspections,
    )
    await write_json(workspace, join(spec.analysis_dir, "analysis_result.json"), result.to_dict())
    await update_run_manifest(
        workspace,
        spec.analysis_dir,
        status,
        attempt_count=len(inspections),
        selected_attempt=selected_attempt,
    )
    return {
        "analysis_runs": [result],
        "analysis_run": result,
        "events": [
            Event(
                kind="analysis_result",
                phase=PHASE,
                data={"analysis_id": spec.analysis_id, "status": status},
            )
        ],
    }


def build_analysis_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "posthoc_analysis",
) -> CompiledStateGraph[AnalysisState, ScientistContext, AnalysisInput, AnalysisOutput]:
    """Build the per-analysis subgraph (one attempt loop of up to ``max_attempts_per_analysis``).

    It is the target of the post-hoc graph's ``Send`` fan-out and can be run alone on an
    :class:`AnalysisInput`.
    """
    sg = StepGraph(
        PHASE,
        AnalysisState,
        input_schema=AnalysisInput,
        output_schema=AnalysisOutput,
        overrides=overrides,
    )
    sg.add_node("record_run_config", record_run_config)
    sg.step("analysis_begin", analysis_begin)
    sg.step("attempt_prepare", attempt_prepare)
    sg.step("attempt_execute", attempt_execute, retry_policy=agent_node_retry_policy())
    sg.step("attempt_inspect", attempt_inspect)
    sg.step("analysis_finish", analysis_finish)
    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "analysis_begin")
    sg.add_conditional_edges(
        "analysis_begin",
        route_after_begin,
        {"attempt": "attempt_prepare", "finish": "analysis_finish"},
    )
    sg.add_edge("attempt_prepare", "attempt_execute")
    sg.add_edge("attempt_execute", "attempt_inspect")
    sg.add_conditional_edges(
        "attempt_inspect",
        route_after_inspect,
        {"attempt": "attempt_prepare", "finish": "analysis_finish"},
    )
    sg.add_edge("analysis_finish", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "REQUIRED_ROOT_FILES",
    "build_analysis_graph",
    "build_attempt_prompt",
    "inspect_attempt",
    "prepare_attempt_dir",
    "resolve_attempt_file",
    "update_run_manifest",
    "validate_attempt_runtime",
]
