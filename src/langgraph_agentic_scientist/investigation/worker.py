"""The campaign worker graph: one worker's idea and attempt cycles (upstream ``campaign_loop``).

A worker runs ``cycles`` cycles. Each cycle: choose the cycle's model (persisted), prune when due,
pull the shared memory (parallel workers), generate an idea (the ``ideation`` slot), accept it,
run its attempt (the ``attempt`` slot), record the attempt (a strictly better correct program
becomes the incumbent) and push the memory the worker appended. A cycle whose ideation is
exhausted ends the campaign (upstream's ``campaign_failed``, with the cycle counter set back so
that a resume retries it).

The only question upstream puts to a person is asked before the loop, once per campaign, when
pruning is enabled and the workspace is not isolated (:func:`confirm_prune`); the answer is kept
in state and never asked again.
"""

from __future__ import annotations

import json
import math
import posixpath
import random
from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..context import GraphOverrides, ScientistContext
from ..core.errors import ConfigurationMismatch
from ..core.results import AttemptRecord, IdeaRecord
from ..core.runconfig import method_config, record_run_config
from ..core.state import (
    Answer,
    BestManifest,
    EvaluationRecord,
    Failure,
    ProgramRef,
    Question,
    TaskRef,
    utc_now,
)
from ..core.steps import StepGraph, ask_person
from ..evaluation.ids import evaluation_id
from ..evaluation.program_clone import clone_program
from ..evaluation.program_spec import resolve_entrypoint
from ..evaluation.protocol import EvaluationRequest
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .attempt import AGENT_LOG_FAILURE, CONTRACT_FAILURE, build_attempt_graph
from .context import (
    ATTEMPTS_HISTORY_FILE,
    BEST_MANIFEST_FILE,
    CONCLUSIONS_FILE,
    CONCLUSIONS_HISTORY_FILE,
    GLOBAL_MEMORY_JSONL_FILE,
    GLOBAL_MEMORY_MD_FILE,
    IDEA_GENERATION_FAILURES_FILE,
    IDEAS_HISTORY_FILE,
    MEMORY_HEADER,
    OTHER_WORKERS_CONCLUSIONS_FILE,
    load_blocks,
    read_text_or,
)
from .ideas import bundle_json, dump_json, idea_dir_name, rubric_json
from .ideation import build_ideation_graph
from .memory_sync import ensure_memory_files, pull_shared_memory, push_shared_memory
from .models import (
    CycleScratch,
    OuterState,
    WorkerInput,
    WorkerOutput,
    WorkerPlan,
    WorkerResult,
    WorkerState,
)

BASELINE_SOURCE = "bootstrap"
"""``source_attempt_id`` of the seed program: a manifest that still has it has no winner."""
SEED_DIR = ".baseline-seed"


def default_results_dir(task: TaskRef) -> str:
    """Return ``<run>/results``, the results root next to the task snapshot."""
    parent = posixpath.dirname(task.root.rstrip("/"))
    return f"{parent}/results" if parent else "results"


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _jsonl(records: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records).encode("utf-8")


def idea_history_line(record: IdeaRecord) -> dict[str, Any]:
    """Return a line of ``ideas_history.jsonl`` (upstream's fields)."""
    return {
        "timestamp": record.accepted_at,
        "cycle_index": record.cycle,
        "idea_id": record.idea_id,
        "idea_dir_name": record.idea_dir.rsplit("/", 1)[-1],
        "title": record.bundle.title,
        "target_rubrics": [rubric_json(rubric) for rubric in record.bundle.target_rubrics],
        "model_index": record.model_index,
    }


def attempt_history_line(record: AttemptRecord) -> dict[str, Any]:
    """Return a line of ``attempts_history.jsonl`` (upstream's fields)."""
    line: dict[str, Any] = {
        "cycle_index": record.cycle,
        "idea_id": record.idea_id,
        "idea_dir_name": record.attempt_dir.rsplit("/", 1)[-1],
        "attempt_id": record.attempt_id,
        "attempt_number": 1,
        "run_exit_code": 0 if record.failure is None else 1,
        "attempt_status": record.status,
        "experiment_dir": record.attempt_dir,
        "rubric_status_valid": record.rubric_status is not None,
        "rubric_overall_pass": bool(record.overall_pass),
    }
    if record.failure is not None:
        line["failure_category"] = record.failure.category
    if record.rubric_status is not None and record.rubric_status.metrics:
        line["metrics"] = record.rubric_status.metrics
    return line


async def write_views(
    ws: Workspace, root: str, state: Mapping[str, Any], outer: OuterState, best: BestManifest | None
) -> None:
    """Write the host records the prompts name into the results root (upstream's file layout).

    The histories, the best manifest and ``outer_state.json`` are written from state (design
    decision D4); the histories show only the current chapter, as pruning hides earlier ideas.
    """
    start = outer.chapter_start_cycle
    ideas = [r for r in state.get("ideas_history") or [] if r.cycle >= start]
    attempts = [r for r in state.get("attempts_history") or [] if r.cycle >= start]
    failures = [r for r in state.get("idea_failures") or [] if r.cycle >= start]
    await ws.write_bytes(
        join(root, IDEAS_HISTORY_FILE), _jsonl([idea_history_line(r) for r in ideas])
    )
    await ws.write_bytes(
        join(root, ATTEMPTS_HISTORY_FILE), _jsonl([attempt_history_line(r) for r in attempts])
    )
    await ws.write_bytes(
        join(root, IDEA_GENERATION_FAILURES_FILE),
        _jsonl([r.model_dump(mode="json") for r in failures]),
    )
    if best is not None:
        await ws.write_bytes(
            join(root, BEST_MANIFEST_FILE),
            (json.dumps(best.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    await ws.write_bytes(
        join(root, "outer_state.json"),
        (json.dumps(outer.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )


async def archive_for_prune(
    ws: Workspace, blocks: Mapping[str, Any], root: str, chapter: int
) -> None:
    """Archive the memory files into ``_pruned/chapter_NNN`` and reset them (upstream's prune).

    Each file is archived only if its archive copy is missing, and the resets come after every
    archive, so the step gives the same result when it runs again after a crash. The best
    program is kept by the caller; histories are windowed by the chapter's first cycle.
    """
    archive = join(root, "_pruned", f"chapter_{chapter:03d}")
    names = (
        GLOBAL_MEMORY_MD_FILE,
        GLOBAL_MEMORY_JSONL_FILE,
        CONCLUSIONS_FILE,
        CONCLUSIONS_HISTORY_FILE,
    )
    for name in names:
        target = join(archive, name)
        if await ws.exists(join(root, name)) and not await ws.exists(target):
            await ws.write_bytes(target, await ws.read_bytes(join(root, name)))
    new_chapter = chapter + 1
    memory = (
        f"# Global Memory Log (Chapter {new_chapter})\n\n"
        f"_Pruned at cycle boundary. Previous context archived in "
        f"_pruned/chapter_{chapter:03d}/_\n\n"
    )
    entry = {
        "timestamp": utc_now(),
        "step": "prune",
        "scope": "global",
        "summary": (
            f"Idea pruning: reset context for chapter {new_chapter}. Previous context archived."
        ),
        "chapter": new_chapter,
    }
    await ws.write_bytes(join(root, GLOBAL_MEMORY_MD_FILE), memory.encode("utf-8"))
    await ws.write_bytes(
        join(root, GLOBAL_MEMORY_JSONL_FILE),
        (json.dumps(entry, sort_keys=True) + "\n").encode("utf-8"),
    )
    await ws.write_bytes(
        join(root, CONCLUSIONS_FILE),
        (str(blocks["conclusions"]["own_fallback"]) + "\n").encode("utf-8"),
    )
    await ws.remove(join(root, CONCLUSIONS_HISTORY_FILE))


# --- nodes -------------------------------------------------------------------------------------


async def init_campaign(state: WorkerState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Prepare the campaign: seed, memory files, results layout and the harness incumbent.

    Idempotent. Freezes the seed program under the results root, ensures the memory files,
    stages ``task/task.md`` for the ideator and, for a harness-evaluated task without an
    incumbent, evaluates the baseline through the evaluator under the id
    ``<run>/w<k>/r<round>/baseline`` (the evaluation is this node's last act). A baseline that
    cannot be validated ends the campaign with ``missing_validated_harness_incumbent``.
    """
    ctx = runtime.context
    ws = ctx.workspace
    config = method_config(state)
    cfg = config.investigation
    task = state["task"]
    worker = int(state.get("worker_index") or 0)
    root = state.get("results_dir") or default_results_dir(task)
    cycles = int(state.get("cycles") or cfg.max_idea_cycles)
    rnd = int(state.get("round") or 0)
    source = state.get("seed") or task.seed_program or ProgramRef(path=join(task.root, "program_0"))
    frozen_path = join(root, SEED_DIR, "program_0")
    if not await ws.exists(join(frozen_path, "program_spec.toml")):
        await ws.remove(frozen_path)
        await clone_program(ws, src_dir=source.path, dst_dir=frozen_path)
    entrypoint = await resolve_entrypoint(ws, frozen_path)
    frozen = ProgramRef(path=frozen_path, entrypoint=entrypoint)
    await ensure_memory_files(ws, load_blocks(ctx.prompts), root)
    task_md = join(task.root, "task.md")
    if await ws.exists(task_md):
        await ws.write_bytes(join(root, "task", "task.md"), await ws.read_bytes(task_md))
    outer = OuterState()
    update: dict[str, Any] = {
        "worker_index": worker,
        "results_dir": root,
        "cycles": cycles,
        "cycle_offset": int(state.get("cycle_offset") or 0),
        "seed": frozen,
        "round": rnd,
        "share_memory": bool(state.get("share_memory")),
        "prune_confirmed": bool(state.get("prune_confirmed")),
        "outer": outer,
        "status": "running",
        "failure": None,
        "cycle": None,
    }
    seeded = state.get("seed_best")
    best = BestManifest(
        program_dir=frozen_path,
        entrypoint_path=entrypoint,
        combined_score=seeded.combined_score if seeded else None,
        correct=seeded.correct if seeded else None,
        source_attempt_id=seeded.source_attempt_id if seeded else BASELINE_SOURCE,
    )
    update["best"] = best
    await write_views(ws, root, {}, outer, best)
    if not task.harness_evaluated or seeded is not None:
        return update
    if ctx.evaluator is None:
        raise ConfigurationMismatch(["evaluator: a harness-evaluated campaign needs an evaluator"])
    if (
        task.contract is not None
        and not ctx.evaluator.isolated
        and not config.allow_unisolated_contracted
    ):
        raise ConfigurationMismatch(
            [
                "evaluator.isolated: a contracted task must be evaluated by an evaluator that "
                "isolates the candidate (allow_unisolated_contracted lifts this)"
            ]
        )
    request = EvaluationRequest(
        evaluation_id=evaluation_id(state["run_id"], f"w{worker}", f"r{rnd}", "baseline"),
        plane="search",
        program=frozen.model_copy(update={"digest": await ws.digest(frozen_path)}),
        task=task,
        timeout_seconds=float(cfg.harness_evaluation_timeout_seconds),
    )
    result = await ctx.evaluator.evaluate(request)
    record = EvaluationRecord(
        evaluation_id=result.evaluation_id,
        program=frozen,
        plane="search",
        status=result.status,
        metrics=dict(result.metrics),
        correct=result.correct,
        evidence_available=result.evidence is not None,
        failure_category=result.failure_category,
        summary=result.summary[:2048],
        attempt_id="bootstrap",
    )
    score = _score(result.metrics.get("combined_score"))
    update["baseline"] = record
    update["evaluations"] = [record]
    if result.status != "completed" or score is None or result.correct is None:
        update["status"] = "failed"
        update["failure"] = Failure(
            category="missing_validated_harness_incumbent",
            scope="task",
            detail=(
                "a harness-evaluated campaign needs a validated, scored baseline; the baseline "
                f"evaluation ended {result.status}"
            ),
        )
        return update
    update["best"] = best.model_copy(
        update={"combined_score": score, "correct": bool(result.correct)}
    )
    return update


def prune_needed(state: Mapping[str, Any], runtime: Runtime[ScientistContext]) -> bool:
    """Whether the prune warning must still be put to a person (upstream's condition)."""
    cfg = method_config(state).investigation
    return (
        cfg.prune_every_n_cycles > 0
        and not runtime.context.workspace.isolated
        and not state.get("prune_confirmed")
    )


async def confirm_prune(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Ask once whether to prune without isolation (upstream ``confirm_prune_warning``).

    The question's text is upstream's banner; the answers are ``proceed`` and ``abort``, and the
    context's policy may answer in the person's place (:class:`AutoApprovePolicy` does what
    upstream does without a terminal). ``ask_person`` is the first and only effect of the node.
    """
    if not prune_needed(state, runtime):
        return {}
    banner = runtime.context.prompts.text("workflow/investigation/prune_warning")
    answer: Answer = await ask_person(
        Question(kind="prune_confirmation", text=banner, options=["proceed", "abort"]), runtime
    )
    if answer.choice == "abort":
        return {
            "prune_confirmed": False,
            "status": "failed",
            "failure": Failure(
                category="prune_aborted", scope="method", detail="the prune warning was declined"
            ),
        }
    return {"prune_confirmed": True}


async def cycle_start(state: WorkerState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Begin a cycle: model choice, pruning when due, the shared-memory pull, the history views.

    Everything here is idempotent: the model is chosen by a seeded draw and persisted, the
    prune archives each file once, the pull replaces files wholesale and the views are rewritten
    from state.
    """
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).investigation
    outer: OuterState = state["outer"]
    root = state["results_dir"]
    index = outer.current_cycle + 1
    global_cycle = int(state.get("cycle_offset") or 0) + index
    assignments = dict(outer.agent_cycle_assignments)
    model_index: int | None = None
    pool = ctx.models.pool
    if pool:
        if str(index) not in assignments:
            draw = random.Random(f"{state['run_id']}:{state['worker_index']}:{index}")
            assignments[str(index)] = draw.randrange(len(pool))
        model_index = assignments[str(index)]
    n = cfg.prune_every_n_cycles
    if n > 0 and index > 1 and (index - 1) % n == 0 and outer.chapter_start_cycle < index:
        chapter = (index - 1) // n
        await archive_for_prune(ws, load_blocks(ctx.prompts), root, chapter)
        outer = outer.model_copy(
            update={"prune_chapter": chapter + 1, "chapter_start_cycle": index}
        )
    outer = outer.model_copy(update={"agent_cycle_assignments": assignments})
    pre = None
    conclusions_sha = ""
    if state.get("share_memory"):
        pre, conclusions_sha = await pull_shared_memory(
            runtime.store,
            ws,
            load_blocks(ctx.prompts),
            run_id=state["run_id"],
            round_index=int(state.get("round") or 0),
            root=root,
            worker_index=state["worker_index"],
        )
    await write_views(ws, root, state, outer, state.get("best"))
    return {
        "outer": outer,
        "cycle": CycleScratch(
            index=index,
            global_cycle=global_cycle,
            model_index=model_index,
            memory_pre=pre,
            conclusions_sha256=conclusions_sha,
        ),
        "ideation": None,
        "attempt_result": None,
    }


async def accept_idea(state: WorkerState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Accept the validated idea: its directory, files and history record; the incumbent seeds."""
    ws = runtime.context.workspace
    ideation = state["ideation"]
    assert ideation is not None and ideation.bundle is not None
    outer: OuterState = state["outer"]
    cycle: CycleScratch = state["cycle"]  # type: ignore[assignment]
    bundle = ideation.bundle
    name = idea_dir_name(cycle_index=cycle.index, title=bundle.title, idea_id=bundle.idea_id)
    directory = join(state["results_dir"], "ideas", name)
    payload = bundle_json(bundle)
    await ws.write_bytes(join(directory, "idea_bundle.json"), dump_json(payload).encode("utf-8"))
    await ws.write_bytes(
        join(directory, "target_rubrics.json"),
        dump_json(payload["target_rubrics"]).encode("utf-8"),
    )
    record = IdeaRecord(
        idea_id=bundle.idea_id,
        idea_dir=directory,
        cycle=cycle.index,
        worker=state["worker_index"],
        bundle=bundle,
        model_index=cycle.model_index,
        accepted_at=utc_now(),
    )
    best = state["best"]
    assert best is not None
    return {
        "outer": outer.model_copy(
            update={
                "current_cycle": cycle.index,
                "last_idea_id": bundle.idea_id,
                "failure_streak": 0,
            }
        ),
        "cycle": cycle.model_copy(
            update={"idea": record, "attempt_id": f"{bundle.idea_id}_attempt_01"}
        ),
        "seed_program": ProgramRef(path=best.program_dir, entrypoint=best.entrypoint_path),
        "ideas_history": [record],
        "idea_failures": list(ideation.failures),
    }


async def campaign_failed(state: WorkerState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """End the campaign after an exhausted ideation (upstream ``campaign_failed``, exit 2).

    The cycle counter stays at the previous cycle, so a resume retries this cycle, and what the
    worker appended in the cycle is still pushed.
    """
    ctx = runtime.context
    outer: OuterState = state["outer"]
    cycle: CycleScratch = state["cycle"]  # type: ignore[assignment]
    ideation = state.get("ideation")
    if state.get("share_memory") and cycle.memory_pre is not None:
        await push_shared_memory(
            runtime.store,
            ctx.workspace,
            run_id=state["run_id"],
            round_index=int(state.get("round") or 0),
            root=state["results_dir"],
            worker_index=state["worker_index"],
            global_cycle=cycle.global_cycle,
            pre=cycle.memory_pre,
            conclusions_sha256=cycle.conclusions_sha256,
        )
    return {
        "outer": outer.model_copy(
            update={
                "current_cycle": cycle.index - 1,
                "failure_streak": outer.failure_streak + 1,
            }
        ),
        "idea_failures": list(ideation.failures) if ideation else [],
        "status": "failed",
        "failure": (ideation.failure if ideation and ideation.failure else None)
        or Failure(category="campaign_failed", detail="idea_generation_attempts_exhausted"),
    }


async def record_attempt(state: WorkerState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Record the attempt: terminal failures stop the campaign, a strict improvement is kept.

    The incumbent changes only when the attempt's best evaluated program is correct and its score
    is strictly higher than the correct incumbent's (or there was none).
    """
    result = state["attempt_result"]
    assert result is not None
    outer: OuterState = state["outer"]
    best: BestManifest | None = state.get("best")
    update: dict[str, Any] = {
        "attempts_history": [result.record],
        "evaluations": list(result.evaluations),
    }
    if result.terminal_failure is not None:
        update.update(status="failed", failure=result.terminal_failure)
        return update
    before = best.combined_score if best and best.correct is True else None
    candidate = result.best_candidate
    if (
        candidate is not None
        and result.best_score is not None
        and (before is None or result.best_score > before)
    ):
        update["best"] = BestManifest(
            program_dir=candidate.path,
            entrypoint_path=candidate.entrypoint,
            combined_score=result.best_score,
            correct=True,
            source_attempt_id=result.record.attempt_id,
        )
    update["outer"] = outer.model_copy(update={"failure_streak": 0})
    return update


async def push_memory(state: WorkerState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Push what the worker appended during the cycle to the shared memory (idempotent keys)."""
    cycle: CycleScratch = state["cycle"]  # type: ignore[assignment]
    if state.get("share_memory") and cycle.memory_pre is not None:
        await push_shared_memory(
            runtime.store,
            runtime.context.workspace,
            run_id=state["run_id"],
            round_index=int(state.get("round") or 0),
            root=state["results_dir"],
            worker_index=state["worker_index"],
            global_cycle=cycle.global_cycle,
            pre=cycle.memory_pre,
            conclusions_sha256=cycle.conclusions_sha256,
        )
    return {"cycle": cycle.model_copy(update={"memory_pushed": True})}


async def finish_campaign(state: WorkerState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Write the final records and return the worker's result."""
    ws = runtime.context.workspace
    outer: OuterState = state["outer"]
    best = state.get("best")
    status = "failed" if state.get("status") == "failed" else "finished"
    await write_views(ws, state["results_dir"], state, outer, best)
    result = WorkerResult(
        plan=WorkerPlan(
            worker_index=state["worker_index"],
            results_dir=state["results_dir"],
            cycles=state["cycles"],
            cycle_offset=int(state.get("cycle_offset") or 0),
            seed=state["seed"],
            round=int(state.get("round") or 0),
            share_memory=bool(state.get("share_memory")),
        ),
        status=status,
        best=best,
        baseline=state.get("baseline"),
        ideas=list(state.get("ideas_history") or []),
        attempts=list(state.get("attempts_history") or []),
        evaluations=list(state.get("evaluations") or []),
        idea_failures=list(state.get("idea_failures") or []),
        outer=outer,
        failure=state.get("failure"),
    )
    return {"worker_result": result, "status": status}


# --- routing -----------------------------------------------------------------------------------


def _next_cycle(state: WorkerState) -> str:
    if state.get("status") == "failed":
        return "finish"
    outer = state["outer"]
    return "finish" if outer.current_cycle + 1 > state["cycles"] else "cycle_start"


def _after_init(state: WorkerState) -> str:
    return "finish" if state.get("status") == "failed" else "confirm_prune"


def _after_ideation(state: WorkerState) -> str:
    ideation = state.get("ideation")
    return (
        "accept_idea"
        if ideation is not None and ideation.status == "accepted"
        else "campaign_failed"
    )


def _after_record(state: WorkerState) -> str:
    return "finish" if state.get("status") == "failed" else "push_memory"


def build_campaign_worker_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "worker",
) -> CompiledStateGraph[Any, ScientistContext, WorkerInput, WorkerOutput]:
    """Build the graph of one campaign worker (upstream ``campaign_loop.run_cycles_and_finalize``).

    Steps (hook keys ``worker.<step>``): ``init_campaign``, ``confirm_prune`` (only when pruning
    is enabled and the workspace is not isolated), ``cycle_start``, ``ideation`` (slot
    ``investigation.ideation``), ``accept_idea``, ``attempt`` (slot ``investigation.attempt``),
    ``record_attempt``, ``push_memory``, ``campaign_failed`` and ``finish``. The graph runs alone
    from ``{"task": ..., "run_id": ...}`` or is mounted by the investigation graph as the slot
    ``investigation.worker``; a replacement must accept :class:`WorkerInput` and return
    :class:`WorkerOutput`. It needs ``store`` (through the top-level compile) only when run with
    ``share_memory``.

    Args:
        checkpointer: The checkpointer when run at the top level; ``None`` inherits the parent's.
        store: The store when run at the top level.
        overrides: Replacement slots (``investigation.ideation``, ``investigation.attempt``).
        name: The compiled graph's name.
    """
    sg = StepGraph(
        "worker",
        WorkerState,
        input_schema=WorkerInput,
        output_schema=WorkerOutput,
        overrides=overrides,
    )
    sg.add_node("record_run_config", record_run_config)
    sg.step("init_campaign", init_campaign)
    sg.step("confirm_prune", confirm_prune)
    sg.step("cycle_start", cycle_start)
    sg.slot(
        "ideation",
        build_ideation_graph(overrides=overrides),
        key="investigation.ideation",
    )
    sg.step("accept_idea", accept_idea)
    sg.slot(
        "attempt",
        build_attempt_graph(overrides=overrides),
        key="investigation.attempt",
    )
    sg.step("record_attempt", record_attempt)
    sg.step("push_memory", push_memory)
    sg.step("campaign_failed", campaign_failed)
    sg.step("finish", finish_campaign)
    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "init_campaign")
    sg.add_conditional_edges(
        "init_campaign",
        _after_init,
        {"finish": "finish", "confirm_prune": "confirm_prune"},
    )
    sg.add_conditional_edges(
        "confirm_prune", _next_cycle, {"finish": "finish", "cycle_start": "cycle_start"}
    )
    sg.add_edge("cycle_start", "ideation")
    sg.add_conditional_edges(
        "ideation",
        _after_ideation,
        {"accept_idea": "accept_idea", "campaign_failed": "campaign_failed"},
    )
    sg.add_edge("accept_idea", "attempt")
    sg.add_edge("attempt", "record_attempt")
    sg.add_conditional_edges(
        "record_attempt", _after_record, {"finish": "finish", "push_memory": "push_memory"}
    )
    sg.add_conditional_edges(
        "push_memory", _next_cycle, {"finish": "finish", "cycle_start": "cycle_start"}
    )
    sg.add_edge("campaign_failed", "finish")
    sg.add_edge("finish", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "AGENT_LOG_FAILURE",
    "BASELINE_SOURCE",
    "CONTRACT_FAILURE",
    "MEMORY_HEADER",
    "OTHER_WORKERS_CONCLUSIONS_FILE",
    "build_campaign_worker_graph",
    "default_results_dir",
    "read_text_or",
]
