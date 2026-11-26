"""The attempt graph: one idea's implementation and evaluation, in both of upstream's regimes.

Upstream's ``run.sh`` runs one attempt per idea in an experiment directory ``ideas/<NN_slug>/``
seeded with the incumbent as ``program_0``. Two regimes exist:

* **Harness-evaluated** (forced for a task with a contract): the *candidate author* (R15,
  ``harness_edit.md``) clones ``program_0`` to ``program_1``, edits only ``program_1/src/**`` and
  ``program_spec.toml``, writes memory and stops. The host checks what was left, evaluates the
  candidate once through the context's evaluator (also after the author ran out of its wall
  clock), writes the rubric status through the context's rubric verifier and appends the
  ``evaluate`` memory event. The author is given **no package tools**: its prompt restricts it to
  plain file operations and forbids the clone and memory helpers.
* **Self-evaluated**: the *search agent* (R14, ``general.md``) iterates edit, evaluate and verify
  with the tools ``clone_program`` and ``evaluate_program`` (which count every call against
  ``max_program_evals_per_attempt``); the host afterwards runs upstream's post-checks.

The agent's root is the experiment directory. The memory files it is told to append to are a view
of the worker's files staged under ``_shared_memory/`` there; what it appended is merged back
after the session (see :mod:`~langgraph_agentic_scientist.investigation.memory_sync`), and the
post-check reads the records from the worker's files.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..agents.memory import has_record, parse_memory_records
from ..agents.runner import RoleOutcome, agent_node_retry_policy, run_workspace_role
from ..agents.staging import HARNESS_EDIT_SKILL_SOURCES, SKILLS_BY_STEP
from ..agents.tools import (
    EvaluateProgramSettings,
    make_clone_program_tool,
    make_evaluate_program_tool,
)
from ..context import GraphOverrides, ScientistContext
from ..core.errors import ConfigurationMismatch
from ..core.render import standard_values
from ..core.results import AttemptRecord
from ..core.runconfig import method_config, record_run_config
from ..core.state import EvaluationRecord, Failure, ProgramRef, RubricSpec, RubricStatus, TaskRef
from ..core.steps import StepGraph
from ..evaluation.ids import candidate_id
from ..evaluation.program_clone import clone_program, evaluation_results_complete
from ..evaluation.program_spec import resolve_entrypoint
from ..evaluation.protocol import EvaluationRequest, EvaluationResult
from ..models import Role
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .context import (
    GLOBAL_MEMORY_JSONL_FILE,
    SHARED_DIR,
    attempt_values,
    read_json_object,
    read_text_or,
    search_metric_feedback,
)
from .harness_eval import (
    AgentOutputError,
    HarnessRubricVerifier,
    ensure_agent_only_created_program_1,
    evaluation_feedback,
    exit_code_of,
    feedback_strings,
    memory_event,
    read_candidate_sources,
    status_json,
    write_evaluation_artifacts,
)
from .ideas import bundle_json, dump_json, rubric_json
from .memory_sync import append_memory_event_once, stage_shared_view, sync_shared_back
from .models import AttemptInput, AttemptOutput, AttemptResult, AttemptState
from .rubrics import extract_target_rubric_ids, validate_rubric_status

CONTRACT_FAILURE = "evaluator_output_contract_invalid"
"""Upstream's terminal task failure (exit code 86): the evaluator's output broke its contract."""
AGENT_LOG_FAILURE = "agent_log_persistence_failed"
"""Upstream's terminal infrastructure failure (exit code 87): an agent's log could not be kept."""
PROGRAM_DIR_RE = re.compile(r"^program_(\d+)$")
SEALED_TASK_FILES = ("evaluate_final.py", "task_contract.json", ".agentic_task_validation.json")


async def program_indices(workspace: Workspace, experiment_dir: str) -> list[int]:
    """Return the indices of the ``program_<n>`` directories of an experiment directory."""
    indices: set[int] = set()
    for entry in await workspace.list_tree(experiment_dir):
        parts = entry.path.split("/")
        match = PROGRAM_DIR_RE.match(parts[0])
        if match and (entry.kind == "dir" or len(parts) > 1):
            indices.add(int(match.group(1)))
    return sorted(indices)


async def _json_object(workspace: Workspace, path: str) -> dict[str, Any] | None:
    return await read_json_object(workspace, path)


async def find_best_program(
    workspace: Workspace, experiment_dir: str
) -> tuple[ProgramRef, float] | None:
    """Return the best evaluated, correct program of an experiment directory and its score.

    Upstream's ``find_best_program_in_experiment``: complete evaluation results, a numeric
    ``combined_score``, ``correct=true`` and an entrypoint that is a real file; the highest
    score wins, the lowest index on a tie.
    """
    best: tuple[ProgramRef, float] | None = None
    for index in await program_indices(workspace, experiment_dir):
        program = join(experiment_dir, f"program_{index}")
        if not await evaluation_results_complete(workspace, program):
            continue
        metrics = await _json_object(workspace, join(program, "metrics.json"))
        correct = await _json_object(workspace, join(program, "correct.json"))
        if not metrics or not correct:
            continue
        score = metrics.get("combined_score")
        if isinstance(score, bool) or not isinstance(score, int | float):
            continue
        if correct.get("correct") is not True:
            continue
        try:
            entrypoint = await resolve_entrypoint(workspace, program)
        except Exception:
            continue
        if best is None or float(score) > best[1]:
            best = (ProgramRef(path=program, entrypoint=entrypoint), float(score))
    return best


async def evaluation_records(
    workspace: Workspace,
    experiment_dir: str,
    *,
    attempt_id: str,
    idea_id: str,
) -> list[EvaluationRecord]:
    """Return the host records of the programs of an experiment directory that were evaluated."""
    records: list[EvaluationRecord] = []
    for index in await program_indices(workspace, experiment_dir):
        program = join(experiment_dir, f"program_{index}")
        if not await evaluation_results_complete(workspace, program):
            continue
        metrics = await _json_object(workspace, join(program, "metrics.json"))
        correct = await _json_object(workspace, join(program, "correct.json"))
        manifest = await _json_object(workspace, join(program, "evaluation_manifest.json"))
        if metrics is None or correct is None:
            continue
        try:
            entrypoint = await resolve_entrypoint(workspace, program)
        except Exception:
            entrypoint = "src/program.py"
        records.append(
            EvaluationRecord(
                evaluation_id=str((manifest or {}).get("evaluation_id") or program),
                program=ProgramRef(path=program, entrypoint=entrypoint),
                plane="search",
                status="completed",
                metrics=metrics,
                correct=correct.get("correct")
                if isinstance(correct.get("correct"), bool)
                else None,
                attempt_id=attempt_id,
                idea_id=idea_id,
            )
        )
    return records


async def post_check(
    workspace: Workspace,
    *,
    experiment_dir: str,
    attempt_id: str,
    target_ids: set[str],
    max_evals: int,
    memory_jsonl_path: str,
    idea_id: str,
) -> str | None:
    """Upstream's post-checks of an attempt (``run.sh:1432-1617``); the error, or ``None``.

    Every evaluated candidate (index above 0) has a valid rubric status, their number is at most
    the evaluation cap, the root ``rubric_status.json`` mirrors the latest, and the worker's
    ``memory.jsonl`` holds a record with the exact ``attempt_id`` and, for the latest evaluated
    program, the exact ``program_id`` on the same record. Memory records are read from the files.
    """
    evaluated: list[int] = []
    for index in await program_indices(workspace, experiment_dir):
        program = join(experiment_dir, f"program_{index}")
        if (
            await workspace.exists(join(program, "metrics.json"))
            and await workspace.exists(join(program, "correct.json"))
            and await evaluation_results_complete(workspace, program)
        ):
            evaluated.append(index)
    candidates = [index for index in evaluated if index > 0]
    if len(candidates) > max_evals:
        return f"evaluated candidate program count exceeds cap: {len(candidates)} > {max_evals}"
    latest: dict[str, Any] | None = None
    latest_index: int | None = None
    for index in candidates:
        label = f"./program_{index}/rubric_status.json"
        error, status = await _validated_status(
            workspace,
            join(experiment_dir, f"program_{index}", "rubric_status.json"),
            label,
            attempt_id=attempt_id,
            idea_id=idea_id,
            target_ids=target_ids,
        )
        if error:
            return error
        latest, latest_index = status, index
    error, root = await _validated_status(
        workspace,
        join(experiment_dir, "rubric_status.json"),
        "./rubric_status.json",
        attempt_id=attempt_id,
        idea_id=idea_id,
        target_ids=target_ids,
    )
    if error:
        return error
    assert root is not None
    if latest is not None:
        if root["overall_pass"] != latest["overall_pass"]:
            return (
                "root rubric_status.json overall_pass must match latest evaluated program artifact"
            )
        if root["rubrics"] != latest["rubrics"]:
            return "root rubric_status.json rubrics must match latest evaluated program artifact"
    if not await workspace.exists(memory_jsonl_path):
        return f"Missing required global memory artifact: {memory_jsonl_path}"
    records = parse_memory_records((await read_text_or(workspace, memory_jsonl_path)).splitlines())
    if not has_record(records, attempt_id=attempt_id):
        return f"memory.jsonl must contain at least one entry with current attempt_id={attempt_id}"
    if latest_index is not None and not has_record(
        records, attempt_id=attempt_id, program_id=f"program_{latest_index}"
    ):
        return (
            "memory.jsonl must contain entry for latest evaluated program_id="
            f"program_{latest_index}"
        )
    return None


async def _validated_status(
    workspace: Workspace,
    path: str,
    label: str,
    *,
    attempt_id: str,
    idea_id: str,
    target_ids: set[str],
) -> tuple[str | None, dict[str, Any] | None]:
    if not await workspace.exists(path):
        return f"Missing required artifact: {label}", None
    text = await read_text_or(workspace, path)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"Invalid JSON in {label}: {exc}", None
    if not isinstance(data, dict):
        return f"{label} must be a JSON object", None
    for key in ("attempt_id", "idea_id", "overall_pass", "rubrics"):
        if key not in data:
            return f"{label} missing key: {key}", None
    if data["attempt_id"] != attempt_id:
        return f"{label} attempt_id mismatch: expected {attempt_id}", None
    if data["idea_id"] != idea_id:
        return f"{label} idea_id mismatch: expected {idea_id}", None
    if not isinstance(data["overall_pass"], bool):
        return f"{label} overall_pass must be boolean", None
    if not isinstance(data["rubrics"], list):
        return f"{label} rubrics must be an array", None
    reported: set[str] = set()
    all_pass = True
    for idx, item in enumerate(data["rubrics"]):
        if not isinstance(item, dict):
            return f"{label} rubrics[{idx}] must be an object", None
        if not isinstance(item.get("id"), str) or not item["id"].strip():
            return f"{label} rubrics[{idx}].id must be non-empty string", None
        if not isinstance(item.get("pass"), bool):
            return f"{label} rubrics[{idx}].pass must be boolean", None
        reported.add(item["id"])
        if not item["pass"]:
            all_pass = False
    if target_ids and reported != target_ids:
        return (
            f"{label} rubric ids mismatch: expected {sorted(target_ids)} got {sorted(reported)}",
            None,
        )
    if data["overall_pass"] != all_pass:
        return f"{label} overall_pass must equal all(rubrics[*].pass)", None
    return None, data


# --- nodes -------------------------------------------------------------------------------------


def _session_failure(outcome: RoleOutcome, *, harness: bool) -> Failure | None:
    """Return the failure of a session as upstream's run exit code would have shown it."""
    if outcome.failure is not None:
        return outcome.failure
    if outcome.timed_out and not harness:
        return Failure(category="timeout", scope="method", detail="the agent timed out")
    return None


async def prepare_attempt(
    state: AttemptState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Prepare the experiment directory of the attempt (upstream ``run.sh`` up to the agent).

    Clones the incumbent as ``program_0`` through the trusted clone, stages ``task.md`` (and, for
    a self-evaluated attempt, ``evaluate.py``, ``capture.py`` and the task's helper modules as
    read-only views), ``active_idea.json`` and ``active_rubrics.json``, and the attempt's view of
    the memory files; sealed files are never staged. Safe to run again: it starts from the idea
    directory as the ideation left it.
    """
    ctx = runtime.context
    cfg = method_config(state).investigation
    ws = ctx.workspace
    task: TaskRef = state["task"]
    cycle = state["cycle"]
    assert cycle.idea is not None and cycle.attempt_id is not None
    experiment = cycle.idea.idea_dir
    harness = task.harness_evaluated
    attempt_id = cycle.attempt_id
    idea_id = cycle.idea.idea_id
    bundle = cycle.idea.bundle
    for index in await program_indices(ws, experiment):
        await ws.remove(join(experiment, f"program_{index}"))
    for leftover in ("rubric_status.json", ".agents", SHARED_DIR):
        await ws.remove(join(experiment, leftover))
    await clone_program(
        ws, src_dir=state["seed_program"].path, dst_dir=join(experiment, "program_0")
    )
    task_md = join(task.root, "task.md")
    if await ws.exists(task_md):
        await ws.write_bytes(join(experiment, "task.md"), await ws.read_bytes(task_md))
    if not harness:
        for name in ("capture.py", "evaluate.py"):
            source = join(task.root, name)
            if await ws.exists(source):
                await ws.write_bytes(join(experiment, name), await ws.read_bytes(source))
        for entry in await ws.list_tree(task.root):
            if "/" in entry.path or entry.kind != "file" or not entry.path.endswith(".py"):
                continue
            if entry.path in {"capture.py", "evaluate.py", "__init__.py", *SEALED_TASK_FILES}:
                continue
            await ws.write_bytes(
                join(experiment, entry.path), await ws.read_bytes(join(task.root, entry.path))
            )
    payload = bundle_json(bundle)
    idea_json = dump_json(payload)
    rubrics_json = dump_json(payload["target_rubrics"])
    await ws.write_bytes(join(experiment, "active_idea.json"), idea_json.encode("utf-8"))
    await ws.write_bytes(join(experiment, "active_rubrics.json"), rubrics_json.encode("utf-8"))
    sizes = await stage_shared_view(ws, state["results_dir"], join(experiment, SHARED_DIR))
    values = await attempt_values(
        ws,
        ctx.prompts,
        task=task,
        experiment_dir=experiment,
        shared_dir=join(experiment, SHARED_DIR),
        attempt_id=attempt_id,
        idea_id=idea_id,
        idea_dir_name=experiment.rsplit("/", 1)[-1],
        max_program_evals=cfg.max_program_evals_per_attempt,
        worker_index=state["worker_index"],
        cycle_index=cycle.index,
        active_idea=idea_json.strip() or "{}",
        active_rubrics=rubrics_json.strip() or "[]",
        include_public_contract=harness,
        public_contract_max_bytes=cfg.public_contract_max_bytes,
    )
    task_text = await read_text_or(ws, task_md, "") or None
    values.update(
        standard_values(task_dir=".", results_dir=".", task_md=task_text, task_md_path="./task.md")
    )
    values["TASK_MD_PATH"] = "./task.md"
    return {
        "harness": harness,
        "experiment_dir": experiment,
        "attempt_id": attempt_id,
        "idea_id": idea_id,
        "shared_pre": sizes,
        "prompt_values": values,
        "timed_out": False,
        "session_failure": None,
        "check_error": None,
        "session_evaluations": [],
        "terminal_failure": None,
    }


def _env(state: AttemptState) -> dict[str, str]:
    return {
        "ATTEMPT_ID": state["attempt_id"],
        "IDEA_ID": state["idea_id"],
        "CYCLE_INDEX": str(state["cycle"].index),
        "WORKER_INDEX": str(state["worker_index"]),
        "MAX_PROGRAM_EVALS_PER_ATTEMPT": str(
            method_config(state).investigation.max_program_evals_per_attempt
        ),
    }


async def author_candidate(
    state: AttemptState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the candidate author (R15): plain file operations only, no package tools."""
    cfg = method_config(state).investigation
    values = state["prompt_values"]
    outcome = await run_workspace_role(
        role=Role.CANDIDATE_AUTHOR,
        runtime=runtime,
        root=state["experiment_dir"],
        prompt_key="investigation/harness_edit",
        values=values,
        skills=SKILLS_BY_STEP["run_harness_eval"],
        skill_values=values,
        skill_sources=HARNESS_EDIT_SKILL_SOURCES,
        tools=(),
        timeout=cfg.run_timeout_seconds,
        env=_env(state),
        pool_index=state["cycle"].model_index,
    )
    return {
        "timed_out": outcome.timed_out,
        "session_failure": _session_failure(outcome, harness=True),
    }


async def search(state: AttemptState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run the search agent (R14) with the trusted clone and the evaluation tool."""
    ctx = runtime.context
    cfg = method_config(state).investigation
    experiment = state["experiment_dir"]
    values = state["prompt_values"]
    cycle = state["cycle"]
    prefix = f"{state['run_id']}/w{state['worker_index']}/c{cycle.global_cycle:03d}"
    tools = [
        make_clone_program_tool(ctx.workspace, experiment, prompts=ctx.prompts),
        make_evaluate_program_tool(
            ctx.workspace,
            ctx.evaluator,
            experiment,
            EvaluateProgramSettings(
                task=state["task"],
                id_prefix=prefix,
                budget=cfg.max_program_evals_per_attempt,
                plane="search",
                timeout_seconds=cfg.evaluation_timeout_seconds,
            ),
            prompts=ctx.prompts,
        ),
    ]
    outcome = await run_workspace_role(
        role=Role.SEARCH_AGENT,
        runtime=runtime,
        root=experiment,
        prompt_key="investigation/general",
        values=values,
        skills=SKILLS_BY_STEP["run"],
        skill_values=values,
        tools=tools,
        timeout=cfg.run_timeout_seconds,
        env=_env(state),
        pool_index=cycle.model_index,
    )
    return {
        "timed_out": outcome.timed_out,
        "session_failure": _session_failure(outcome, harness=False),
        "session_evaluations": list(outcome.evaluations),
    }


def _terminal(results: list[EvaluationResult]) -> Failure | None:
    for result in results:
        if result.failure_category == CONTRACT_FAILURE:
            return Failure(category=CONTRACT_FAILURE, scope="task", detail=result.summary[:2048])
    return None


async def collect_memory(state: AttemptState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Merge the agent's additions to its view of the memory files into the worker's files."""
    await sync_shared_back(
        runtime.context.workspace,
        state["results_dir"],
        join(state["experiment_dir"], SHARED_DIR),
        state["shared_pre"],
    )
    return {"terminal_failure": _terminal(list(state.get("session_evaluations") or []))}


async def check_outputs(state: AttemptState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Merge memory back, then check what the candidate author left (upstream's checks).

    A session that failed, or left more than ``program_1``, or evaluator artifacts, or a source
    tree the harness cannot inspect within its bounds, ends the attempt without an evaluation. A
    session that ran out of its wall clock is checked like any other and then evaluated.
    """
    ws = runtime.context.workspace
    experiment = state["experiment_dir"]
    await sync_shared_back(
        ws, state["results_dir"], join(experiment, SHARED_DIR), state["shared_pre"]
    )
    failure = state.get("session_failure")
    if failure is not None:
        return {"check_error": f"{failure.category}: {failure.detail}"}
    try:
        await ensure_agent_only_created_program_1(ws, experiment)
        await read_candidate_sources(ws, join(experiment, "program_1"))
    except AgentOutputError as exc:
        return {"check_error": str(exc)}
    return {"check_error": None}


async def evaluate_candidate(
    state: AttemptState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Evaluate ``program_1`` once through the context's evaluator (the call is the last act)."""
    ctx = runtime.context
    cfg = method_config(state).investigation
    if ctx.evaluator is None:
        raise ConfigurationMismatch(["evaluator: a harness-evaluated attempt needs an evaluator"])
    if (
        state["task"].contract is not None
        and not ctx.evaluator.isolated
        and not method_config(state).allow_unisolated_contracted
    ):
        raise ConfigurationMismatch(
            [
                "evaluator.isolated: a contracted task must be evaluated by an evaluator that "
                "isolates the candidate (allow_unisolated_contracted lifts this)"
            ]
        )
    program = join(state["experiment_dir"], "program_1")
    entrypoint = await resolve_entrypoint(ctx.workspace, program)
    request = EvaluationRequest(
        evaluation_id=candidate_id(
            state["run_id"], state["worker_index"], state["cycle"].global_cycle, "program_1"
        ),
        plane="search",
        program=ProgramRef(
            path=program, entrypoint=entrypoint, digest=await ctx.workspace.digest(program)
        ),
        task=state["task"],
        num_replications=1,
        timeout_seconds=float(cfg.harness_evaluation_timeout_seconds),
    )
    result = await ctx.evaluator.evaluate(request)
    return {"evaluation": result, "terminal_failure": _terminal([result])}


def _memory_metrics(task: TaskRef, evaluation: EvaluationResult, max_bytes: int) -> dict[str, Any]:
    return {
        **search_metric_feedback(task.contract, evaluation.metrics, max_bytes=max_bytes),
        "combined_score": evaluation.metrics.get("combined_score"),
        "correct": bool(evaluation.correct),
    }


async def verify_rubrics(state: AttemptState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Publish the evaluation's summaries and write the rubric status (harness-owned)."""
    ctx = runtime.context
    cfg = method_config(state).investigation
    ws = ctx.workspace
    experiment = state["experiment_dir"]
    program = join(experiment, "program_1")
    evaluation = state["evaluation"]
    assert evaluation is not None
    await write_evaluation_artifacts(
        ws,
        program,
        evaluation,
        attempt_id=state["attempt_id"],
        idea_id=state["idea_id"],
    )
    verifier = ctx.rubric_verifier or HarnessRubricVerifier(ctx.prompts)
    rubrics: list[RubricSpec] = list(state["cycle"].idea.bundle.target_rubrics)  # type: ignore[union-attr]
    try:
        status = await verifier.verify(
            rubrics=rubrics,
            evaluation=evaluation,
            program_dir=program,
            workspace=ws,
            attempt_id=state["attempt_id"],
            idea_id=state["idea_id"],
        )
    except AgentOutputError as exc:
        return {"check_error": str(exc)}
    metrics = _memory_metrics(state["task"], evaluation, cfg.harness_feedback_max_bytes)
    if not status.metrics:
        status = status.model_copy(update={"metrics": metrics})
    text = dump_json(status_json(status)) + "\n"
    await ws.write_bytes(join(program, "rubric_status.json"), text.encode("utf-8"))
    await ws.write_bytes(join(experiment, "rubric_status.json"), text.encode("utf-8"))
    return {}


async def record_memory(state: AttemptState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Append the host's ``evaluate`` memory event to the worker's memory files, once."""
    ctx = runtime.context
    cfg = method_config(state).investigation
    evaluation = state["evaluation"]
    assert evaluation is not None
    strings = feedback_strings(ctx.prompts)
    metrics = _memory_metrics(state["task"], evaluation, cfg.harness_feedback_max_bytes)
    feedback = evaluation_feedback(metrics, exit_code_of(evaluation), strings)
    event, markdown = memory_event(
        attempt_id=state["attempt_id"],
        idea_id=state["idea_id"],
        metrics=metrics,
        correct=bool(evaluation.correct),
        feedback=feedback,
        strings=strings,
    )
    await append_memory_event_once(
        ctx.workspace,
        state["results_dir"],
        event,
        markdown=markdown,
        marker=f"- attempt_id: {state['attempt_id']}\n- program_id: program_1\n",
    )
    return {}


async def finish_attempt(state: AttemptState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Validate the attempt's artifacts and build its :class:`AttemptResult`.

    Reads the rubric status from the files (upstream's validation), runs the post-checks
    (``run.sh``) when the run itself succeeded, and finds the best evaluated program.
    """
    ctx = runtime.context
    cfg = method_config(state).investigation
    ws = ctx.workspace
    experiment = state["experiment_dir"]
    attempt_id = state["attempt_id"]
    idea_id = state["idea_id"]
    cycle = state["cycle"]
    assert cycle.idea is not None
    target_ids = extract_target_rubric_ids(
        [rubric_json(rubric) for rubric in cycle.idea.bundle.target_rubrics]
    )
    failure = state.get("session_failure")
    check_error = state.get("check_error")
    terminal = state.get("terminal_failure")
    run_failed = failure is not None or check_error is not None
    if not run_failed:
        run_failed_error = await post_check(
            ws,
            experiment_dir=experiment,
            attempt_id=attempt_id,
            target_ids=target_ids,
            max_evals=cfg.max_program_evals_per_attempt,
            memory_jsonl_path=join(state["results_dir"], GLOBAL_MEMORY_JSONL_FILE),
            idea_id=idea_id,
        )
        if run_failed_error:
            check_error = run_failed_error
            run_failed = True
    status_data = await read_json_object(ws, join(experiment, "rubric_status.json"))
    validation = validate_rubric_status(
        status_data, target_ids, expected_attempt_id=attempt_id, expected_idea_id=idea_id
    )
    best = await find_best_program(ws, experiment)
    records: list[EvaluationRecord] = []
    if state.get("harness"):
        evaluation = state.get("evaluation")
        if evaluation is not None:
            records.append(
                EvaluationRecord(
                    evaluation_id=evaluation.evaluation_id,
                    program=ProgramRef(path=join(experiment, "program_1")),
                    plane="search",
                    status=evaluation.status,
                    metrics=dict(evaluation.metrics),
                    correct=evaluation.correct,
                    evidence_available=evaluation.evidence is not None,
                    failure_category=evaluation.failure_category,
                    summary=evaluation.summary[:2048],
                    attempt_id=attempt_id,
                    idea_id=idea_id,
                )
            )
    else:
        records = await evaluation_records(ws, experiment, attempt_id=attempt_id, idea_id=idea_id)
    status_metrics = status_data.get("metrics") if isinstance(status_data, dict) else None
    metrics_score = (
        status_metrics.get("combined_score") if isinstance(status_metrics, dict) else None
    )
    metrics_correct = status_metrics.get("correct") if isinstance(status_metrics, dict) else None
    score = (
        float(metrics_score)
        if isinstance(metrics_score, int | float) and not isinstance(metrics_score, bool)
        else (best[1] if best else None)
    )
    correct = metrics_correct if isinstance(metrics_correct, bool) else (True if best else None)
    passed = (not run_failed) and validation.valid and validation.overall_pass
    rubric_status = (
        RubricStatus.model_validate(validation.status)
        if validation.valid and validation.status is not None
        else None
    )
    final_failure = terminal or failure
    if final_failure is None and check_error:
        final_failure = Failure(category="attempt_failed", scope="method", detail=check_error)
    record = AttemptRecord(
        attempt_id=attempt_id,
        idea_id=idea_id,
        cycle=cycle.index,
        worker=state["worker_index"],
        attempt_dir=experiment,
        status="pass" if passed else "fail",
        best_program=best[0] if best else None,
        combined_score=score,
        correct=correct,
        overall_pass=validation.overall_pass if validation.valid else False,
        timed_out=bool(state.get("timed_out")),
        evaluation_ids=[record.evaluation_id for record in records],
        rubric_status=rubric_status,
        failure=final_failure,
    )
    return {
        "attempt_result": AttemptResult(
            record=record,
            evaluations=records,
            terminal_failure=terminal,
            experiment_dir=experiment,
            best_candidate=best[0] if best else None,
            best_score=best[1] if best else None,
        )
    }


def _after_prepare(state: AttemptState) -> str:
    return "author_candidate" if state.get("harness") else "search"


def _after_check(state: AttemptState) -> str:
    return "finish" if state.get("check_error") else "evaluate_candidate"


def _after_evaluation(state: AttemptState) -> str:
    return "finish" if state.get("terminal_failure") is not None else "verify_rubrics"


def _after_verify(state: AttemptState) -> str:
    return "finish" if state.get("check_error") else "record_memory"


def build_attempt_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "attempt",
) -> CompiledStateGraph[Any, ScientistContext, AttemptInput, AttemptOutput]:
    """Build the graph of one attempt, in the regime the task's ``harness_evaluated`` selects.

    Steps (hook keys ``attempt.<step>``), harness-evaluated: ``prepare``, ``author_candidate``,
    ``check_outputs``, ``evaluate_candidate``, ``verify_rubrics``, ``record_memory``, ``finish``;
    self-evaluated: ``prepare``, ``search``, ``collect_memory``, ``finish``. The worker mounts the
    graph as the slot ``investigation.attempt``; a replacement must accept :class:`AttemptInput`
    and return :class:`AttemptOutput`. Replace the rubric status rules with
    ``ScientistContext.rubric_verifier`` and the evaluation with ``ScientistContext.evaluator``.

    Args:
        checkpointer: The checkpointer when run at the top level; ``None`` inherits the parent's.
        store: The store when run at the top level.
        overrides: Replacement slots and hooks' graph overrides.
        name: The compiled graph's name.
    """
    sg = StepGraph(
        "attempt",
        AttemptState,
        input_schema=AttemptInput,
        output_schema=AttemptOutput,
        overrides=overrides,
    )
    retry = agent_node_retry_policy()
    sg.add_node("record_run_config", record_run_config)
    sg.step("prepare", prepare_attempt)
    sg.step("author_candidate", author_candidate, retry_policy=retry)
    sg.step("search", search, retry_policy=retry)
    sg.step("collect_memory", collect_memory)
    sg.step("check_outputs", check_outputs)
    sg.step("evaluate_candidate", evaluate_candidate)
    sg.step("verify_rubrics", verify_rubrics)
    sg.step("record_memory", record_memory)
    sg.step("finish", finish_attempt)
    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "prepare")
    sg.add_conditional_edges(
        "prepare",
        _after_prepare,
        {"author_candidate": "author_candidate", "search": "search"},
    )
    sg.add_edge("author_candidate", "check_outputs")
    sg.add_conditional_edges(
        "check_outputs",
        _after_check,
        {"finish": "finish", "evaluate_candidate": "evaluate_candidate"},
    )
    sg.add_conditional_edges(
        "evaluate_candidate",
        _after_evaluation,
        {"finish": "finish", "verify_rubrics": "verify_rubrics"},
    )
    sg.add_conditional_edges(
        "verify_rubrics", _after_verify, {"finish": "finish", "record_memory": "record_memory"}
    )
    sg.add_edge("record_memory", "finish")
    sg.add_edge("search", "collect_memory")
    sg.add_edge("collect_memory", "finish")
    sg.add_edge("finish", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "AGENT_LOG_FAILURE",
    "CONTRACT_FAILURE",
    "build_attempt_graph",
    "find_best_program",
    "post_check",
    "program_indices",
]
