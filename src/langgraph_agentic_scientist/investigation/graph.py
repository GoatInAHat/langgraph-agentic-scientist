"""The investigation graph: the campaign over one or more workers, in rounds.

Upstream's investigation is a campaign of idea and attempt cycles; with ``workers > 1`` it is a set
of isolated workers that share memory at cycle boundaries, and with pruning enabled it runs in
barrier rounds in which each worker restarts from its best program
(``investigation/parallel_helpers.py``). This graph plans the round (:func:`plan_round`), fans the
workers out with ``Send`` (each worker is the compiled worker graph, mounted as a node), joins the
round, and either prunes into the next round or finalises the result. Before the first round, a
harness-evaluated campaign establishes its incumbent once (:func:`establish_incumbent`) and every
worker reads it.

Upstream's only question to a person, the prune warning, is asked here once for the whole
campaign and the answer is passed to every worker.
"""

from __future__ import annotations

import json
import math
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer, Send

from ..context import GraphOverrides, ScientistContext
from ..core.errors import ConfigurationMismatch
from ..core.results import InvestigationResult
from ..core.runconfig import method_config, record_run_config
from ..core.state import Answer, BestManifest, EvaluationRecord, Failure, ProgramRef, Question
from ..core.steps import StepGraph, ask_person
from ..drafting.scaffold import evaluator_tree_sha256, load_qualification_receipt
from ..evaluation.ids import campaign_baseline_id
from ..evaluation.program_spec import resolve_entrypoint
from ..workspace.paths import join
from .attempt import AGENT_LOG_FAILURE, CONTRACT_FAILURE
from .memory_sync import StoreRequired
from .models import (
    InvestigationInput,
    InvestigationOutput,
    InvestigationState,
    WorkerPlan,
    WorkerResult,
    WorkerSend,
)
from .worker import (
    BASELINE_SOURCE,
    build_campaign_worker_graph,
    default_results_dir,
    evaluate_baseline,
    prune_needed,
)

HOST_FAILURES = frozenset({CONTRACT_FAILURE, AGENT_LOG_FAILURE})
"""Terminal failures that stop the campaign whatever ``parallel_fail_fast`` says."""


def split_cycle_budget(total_cycles: int, workers: int) -> list[int]:
    """Split ``total_cycles`` across ``workers`` as evenly as possible (upstream's rule)."""
    base = total_cycles // workers
    remainder = total_cycles % workers
    return [base + (1 if idx < remainder else 0) for idx in range(workers)]


def plan_rounds(max_cycles: int, workers: int, prune_every: int) -> tuple[int, int]:
    """Return ``(active workers, number of rounds)`` as upstream's campaign decides them.

    Workers are capped by the cycle budget. One worker, or no pruning (or a period that is not
    below the budget), is one round; otherwise a round has ``prune_every`` cycles.
    """
    active = min(workers, max_cycles)
    if active == 1 or prune_every == 0 or prune_every >= max_cycles:
        return active, 1
    return active, math.ceil(max_cycles / prune_every)


async def confirm_prune(
    state: InvestigationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Ask once, for the whole campaign, whether to prune without isolation.

    Upstream asks (``confirm_prune_warning``) before the cycle loop, once per campaign, when
    pruning is enabled and the agent is not containerised, and continues without a terminal. The
    context's policy may answer in the person's place; ``abort`` ends the campaign.
    """
    if not prune_needed(state, runtime):
        return {}
    banner = runtime.context.prompts.text("workflow/investigation/prune_warning")
    answer: Answer = await ask_person(
        Question(kind="prune_confirmation", text=banner, options=["proceed", "abort"]), runtime
    )
    if answer.choice == "abort":
        return {
            "stopped": True,
            "investigation": InvestigationResult(
                status="failed",
                workers=0,
                failure=Failure(
                    category="prune_aborted",
                    scope="method",
                    detail="the prune warning was declined",
                ),
            ),
        }
    return {"prune_confirmed": True}


def _task_seed(task: Any) -> ProgramRef:
    return task.seed_program or ProgramRef(path=join(task.root, "program_0"))


async def _receipt_incumbent(
    runtime: Runtime[ScientistContext], task: Any, seed: ProgramRef
) -> tuple[BestManifest, EvaluationRecord] | None:
    """The baseline measured by the task's qualification, when it may stand as the incumbent.

    It may when the seed is the task's ``program_0``, the task publishes a qualification receipt
    with a finite baseline score and a correctness verdict, and the receipt's evaluator digest is
    that of the task's evaluators now.
    """
    ws = runtime.context.workspace
    if task.contract is None or seed.path != join(task.root, "program_0"):
        return None
    receipt = await load_qualification_receipt(ws, task.root)
    if receipt is None or receipt.baseline_correct is None:
        return None
    score = receipt.baseline_combined_score
    if not math.isfinite(score):
        return None
    try:
        digest = await evaluator_tree_sha256(ws, task.root, task.contract)
    except (OSError, FileNotFoundError):
        return None
    if digest != receipt.evaluator_tree_sha256:
        return None
    entrypoint = await resolve_entrypoint(ws, seed.path)
    incumbent = BestManifest(
        program_dir=seed.path,
        entrypoint_path=entrypoint,
        combined_score=score,
        correct=receipt.baseline_correct,
        source_attempt_id=BASELINE_SOURCE,
    )
    record = EvaluationRecord(
        evaluation_id=receipt.evaluation_id or "qualification",
        program=ProgramRef(path=seed.path, entrypoint=entrypoint),
        plane="search",
        status="completed",
        metrics={"combined_score": score},
        correct=receipt.baseline_correct,
        summary="baseline measured by the task's qualification (receipt)",
        attempt_id="bootstrap",
    )
    return incumbent, record


async def establish_incumbent(
    state: InvestigationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Establish the incumbent of a harness-evaluated campaign once, before any worker starts.

    Upstream reads the baseline's score from the task runtime's validated record, produced once,
    and every worker reconciles against it. The port takes, in order: the ``incumbent`` given in
    the input; the baseline score of the task's qualification receipt (when the seed is the
    task's ``program_0`` and the evaluators are those the receipt binds); or one evaluation of the
    baseline under the campaign-level id ``<run>/baseline``. Every worker of every round whose
    seed is the task's seed receives it as ``seed_best`` and evaluates nothing. A baseline that
    cannot be validated ends the campaign with ``missing_validated_harness_incumbent``.
    """
    task = state["task"]
    if not task.harness_evaluated:
        return {}
    given = state.get("incumbent")
    if given is not None:
        if given.correct is None or _finite(given.combined_score) is None:
            raise ConfigurationMismatch(
                ["incumbent: a given incumbent needs a finite combined_score and a verdict"]
            )
        return {"incumbent": given}
    seed = _task_seed(task)
    from_receipt = await _receipt_incumbent(runtime, task, seed)
    if from_receipt is not None:
        incumbent, record = from_receipt
        return {"incumbent": incumbent, "baseline": record, "baseline_evaluated": False}
    ws = runtime.context.workspace
    entrypoint = await resolve_entrypoint(ws, seed.path)
    program = ProgramRef(path=seed.path, entrypoint=entrypoint)
    record, score, failure = await evaluate_baseline(
        runtime,
        method_config(state),
        task,
        program.model_copy(update={"digest": await ws.digest(seed.path)}),
        campaign_baseline_id(state["run_id"]),
    )
    if failure is not None or score is None:
        return {
            "baseline": record,
            "baseline_evaluated": True,
            "stopped": True,
            "investigation": InvestigationResult(
                status="failed",
                workers=0,
                baseline=record,
                evaluations=[record],
                failure=failure,
            ),
        }
    incumbent = BestManifest(
        program_dir=seed.path,
        entrypoint_path=entrypoint,
        combined_score=score,
        correct=bool(record.correct),
        source_attempt_id=BASELINE_SOURCE,
    )
    return {"incumbent": incumbent, "baseline": record, "baseline_evaluated": True}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _round_dir(root: str, rounds: int, rnd: int) -> str:
    return join(root, f"round_{rnd:03d}") if rounds > 1 else root


async def plan_round(
    state: InvestigationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Plan the round: the workers, their cycle budgets, seeds and results roots.

    Raises:
        StoreRequired: When workers share memory and the run was given no store.
    """
    cfg = method_config(state).investigation
    task = state["task"]
    root = state.get("results_dir") or default_results_dir(task)
    active, rounds = plan_rounds(cfg.max_idea_cycles, cfg.workers, cfg.prune_every_n_cycles)
    rnd = int(state.get("round") or 0)
    done = int(state.get("cycles_done") or 0)
    this_total = (
        cfg.max_idea_cycles
        if rounds == 1
        else min(cfg.prune_every_n_cycles, cfg.max_idea_cycles - done)
    )
    share = active > 1 and not cfg.isolate_worker_memory
    if share and runtime.store is None:
        raise StoreRequired()
    seeds = dict(state.get("seeds") or {})
    seed_default = _task_seed(task)
    plans: list[WorkerPlan] = []
    offset = done
    for position, cycles in enumerate(split_cycle_budget(this_total, active)):
        worker = position + 1 if active > 1 else 0
        if cycles <= 0:
            continue
        directory = (
            join(_round_dir(root, rounds, rnd), f"worker_{worker:02d}")
            if active > 1
            else _round_dir(root, rounds, rnd)
        )
        plans.append(
            WorkerPlan(
                worker_index=worker,
                results_dir=directory,
                cycles=cycles,
                cycle_offset=offset,
                seed=seeds.get(str(worker), seed_default),
                round=rnd,
                share_memory=share,
            )
        )
        offset += cycles
    return {"results_dir": root, "round": rnd, "rounds": rounds, "worker_plans": plans}


def fan_out(state: Any) -> list[Send]:
    """Send one worker graph run per planned worker (a single worker is one ``Send``).

    A worker seeded from its own best program of an earlier round gets that program's manifest
    as ``seed_best``; a worker seeded from the task's seed gets the campaign's incumbent.
    """
    bests = state.get("seed_bests") or {}
    incumbent = state.get("incumbent")
    seed_path = _task_seed(state["task"]).path
    sends: list[Send] = []
    for plan in state["worker_plans"]:
        seed_best = bests.get(str(plan.worker_index))
        if seed_best is None and plan.seed.path == seed_path:
            seed_best = incumbent
        payload: WorkerSend = {
            "task": state["task"],
            "run_id": state["run_id"],
            "run_config": state.get("run_config"),
            "worker_index": plan.worker_index,
            "results_dir": plan.results_dir,
            "cycles": plan.cycles,
            "cycle_offset": plan.cycle_offset,
            "seed": plan.seed,
            "seed_best": seed_best,
            "round": plan.round,
            "share_memory": plan.share_memory,
            "prune_confirmed": bool(state.get("prune_confirmed")),
        }
        sends.append(Send("worker", payload))
    return sends


def _score(manifest: BestManifest | None) -> float | None:
    if manifest is None or manifest.correct is not True:
        return None
    score = manifest.combined_score
    if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
        return None
    return float(score)


async def join_round(
    state: InvestigationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Join the round: stop on a host-owned failure, update each worker's seed for the next round.

    ``parallel_fail_fast`` stops the campaign after a round with a failed worker (the workers of
    a round run concurrently, so there is no "first" failure to stop on earlier).
    """
    cfg = method_config(state).investigation
    rnd = int(state.get("round") or 0)
    results = [r for r in state.get("worker_result") or [] if r.plan.round == rnd]
    stopped = any(r.failure is not None and r.failure.category in HOST_FAILURES for r in results)
    if cfg.parallel_fail_fast and any(r.status == "failed" for r in results):
        stopped = True
    seeds = dict(state.get("seeds") or {})
    bests = dict(state.get("seed_bests") or {})
    for result in results:
        key = str(result.plan.worker_index)
        new = _score(result.best)
        old = _score(bests.get(key))
        if result.best is not None and new is not None and (old is None or new > old):
            seeds[key] = ProgramRef(
                path=result.best.program_dir, entrypoint=result.best.entrypoint_path
            )
            bests[key] = result.best
    cycles = sum(plan.cycles for plan in state.get("worker_plans") or [])
    return {
        "stopped": stopped,
        "seeds": seeds,
        "seed_bests": bests,
        "cycles_done": int(state.get("cycles_done") or 0) + cycles,
    }


def _after_join(state: InvestigationState) -> str:
    if state.get("stopped") or int(state.get("round") or 0) + 1 >= int(state.get("rounds") or 1):
        return "finalize"
    return "prune_round"


async def prune_round(
    state: InvestigationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Advance to the next round: every worker restarts from its best program.

    Each round has its own results directories, so the context of earlier rounds stays out of
    the next round's workspaces (upstream archives it under ``_pruned/``); the next round's
    workers are seeded from the seeds :func:`join_round` recorded.
    """
    return {"round": int(state.get("round") or 0) + 1}


def _winner(results: list[WorkerResult]) -> BestManifest | None:
    best: BestManifest | None = None
    for result in results:
        if _score(result.best) is None:
            continue
        if best is None or _score(result.best) > (_score(best) or -math.inf):  # type: ignore[operator]
            best = result.best
    if best is not None:
        return best
    return next((r.best for r in results if r.best is not None), None)


async def finalize(state: InvestigationState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Merge the workers' results into the :class:`InvestigationResult`.

    The global best is the highest-scoring correct program over every worker and round; the
    status is ``completed`` when it came from an attempt, ``completed_no_winner`` when the
    retained baseline is still the best, and ``failed`` when a worker failed.
    """
    if (
        state.get("investigation") is not None
        and state.get("stopped")
        and not state.get("worker_result")
    ):
        return {}
    task = state["task"]
    root = state.get("results_dir") or default_results_dir(task)
    results = list(state.get("worker_result") or [])
    failed = next((r.failure for r in results if r.status == "failed" and r.failure), None)
    best = _winner(results)
    status: Any
    if failed is not None:
        status = "failed"
    elif best is not None and best.source_attempt_id != BASELINE_SOURCE:
        status = "completed"
    else:
        status = "completed_no_winner"
    if best is not None:
        await runtime.context.workspace.write_bytes(
            join(root, "best_program_manifest.json"),
            (json.dumps(best.model_dump(mode="json"), indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            ),
        )
    plans_workers = {r.plan.worker_index for r in results}
    baseline = state.get("baseline")
    campaign_evaluations = (
        [baseline] if baseline is not None and state.get("baseline_evaluated") else []
    )
    result = InvestigationResult(
        status=status,
        best=best,
        baseline=baseline or next((r.baseline for r in results if r.baseline is not None), None),
        ideas=[idea for r in results for idea in r.ideas],
        attempts=[attempt for r in results for attempt in r.attempts],
        evaluations=campaign_evaluations + [ev for r in results for ev in r.evaluations],
        workers=len(plans_workers) or 1,
        results_dir=root,
        failure=failed,
    )
    return {"investigation": result}


def build_investigation_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "investigation",
) -> CompiledStateGraph[Any, ScientistContext, InvestigationInput, InvestigationOutput]:
    """Build the investigation graph (upstream's campaign, parallel workers and pruning rounds).

    Steps (hook keys ``investigation.<step>``): ``confirm_prune``, ``establish_incumbent``,
    ``plan_round``, ``worker``
    (slot ``investigation.worker``, a ``Send`` per worker), ``join_round``, ``prune_round`` and
    ``finalize``. ``store`` is needed (at the top-level compile) when ``workers > 1`` and workers
    share memory. The result is :class:`~langgraph_agentic_scientist.core.results.
    InvestigationResult`.

    Args:
        checkpointer: The checkpointer when run at the top level; ``None`` inherits the parent's.
        store: The store when run at the top level.
        overrides: Replacement slots (``investigation.worker``, ``investigation.ideation``,
            ``investigation.attempt``).
        name: The compiled graph's name.
    """
    sg = StepGraph(
        "investigation",
        InvestigationState,
        input_schema=InvestigationInput,
        output_schema=InvestigationOutput,
        overrides=overrides,
    )
    sg.add_node("record_run_config", record_run_config)
    sg.step("confirm_prune", confirm_prune)
    sg.step("establish_incumbent", establish_incumbent)
    sg.step("plan_round", plan_round)
    sg.slot(
        "worker",
        build_campaign_worker_graph(overrides=overrides),
        key="investigation.worker",
        hooks=False,
    )
    sg.step("join_round", join_round)
    sg.step("prune_round", prune_round)
    sg.step("finalize", finalize)
    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "confirm_prune")
    sg.add_conditional_edges(
        "confirm_prune",
        lambda state: "finalize" if state.get("stopped") else "establish_incumbent",
        {"finalize": "finalize", "establish_incumbent": "establish_incumbent"},
    )
    sg.add_conditional_edges(
        "establish_incumbent",
        lambda state: "finalize" if state.get("stopped") else "plan_round",
        {"finalize": "finalize", "plan_round": "plan_round"},
    )
    sg.add_conditional_edges("plan_round", fan_out)
    sg.add_edge("worker", "join_round")
    sg.add_conditional_edges(
        "join_round", _after_join, {"finalize": "finalize", "prune_round": "prune_round"}
    )
    sg.add_edge("prune_round", "plan_round")
    sg.add_edge("finalize", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "HOST_FAILURES",
    "build_investigation_graph",
    "plan_rounds",
    "split_cycle_budget",
]
