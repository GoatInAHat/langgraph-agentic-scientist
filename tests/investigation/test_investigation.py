"""The investigation graph: workers through Send, shared memory in the store, rounds, outcomes."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langgraph_agentic_scientist import AutoApprovePolicy
from langgraph_agentic_scientist.core.errors import ConfigurationMismatch
from langgraph_agentic_scientist.investigation import build_investigation_graph
from langgraph_agentic_scientist.investigation.graph import plan_rounds, split_cycle_budget
from langgraph_agentic_scientist.investigation.memory_sync import memory_namespace
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import FakeEvaluator, tool_call

from .helpers import (
    HARNESS_IDEA,
    author_policy,
    author_steps,
    config,
    idea_step,
    ideator_policy,
    memory_line,
    run_worker,
    scripted,
    task_with_harness,
    thread,
    winning_evaluator,
)


def _models(many: int = 12) -> Any:
    return scripted(
        {
            Role.IDEATOR: [ideator_policy()] * many,
            Role.CANDIDATE_AUTHOR: [author_policy] * (many * 5),
        }
    )


async def _run(ctx: Any, task: Any, *, store: Any = None, name: str = "inv") -> Any:
    app = build_investigation_graph(checkpointer=InMemorySaver(), store=store)
    out = await app.ainvoke({"task": task, "run_id": "run"}, thread(name), context=ctx)
    return out["investigation"]


def test_cycle_budget_and_round_planning() -> None:
    assert split_cycle_budget(7, 3) == [3, 2, 2]
    assert split_cycle_budget(2, 2) == [1, 1]
    assert plan_rounds(20, 1, 5) == (1, 1)
    assert plan_rounds(20, 4, 0) == (4, 1)
    assert plan_rounds(20, 4, 20) == (4, 1)
    assert plan_rounds(20, 4, 5) == (4, 4)
    assert plan_rounds(3, 8, 1) == (3, 3)


async def test_a_single_worker_is_one_send_and_a_winner_completes(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    ctx = make_context(
        models=_models(), config=config(max_idea_cycles=2), evaluator=winning_evaluator()
    )
    result = await _run(ctx, task)
    assert result.status == "completed" and result.workers == 1
    assert result.best is not None and result.best.source_attempt_id.endswith("_attempt_01")
    assert result.best.combined_score == 6.0
    assert result.results_dir == "run/results"
    assert len(result.attempts) == 2 and len(result.ideas) == 2
    assert result.baseline is not None
    assert await workspace.exists("run/results/best_program_manifest.json")


async def test_no_winner_when_the_baseline_stays_best(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    ctx = make_context(
        models=_models(), config=config(max_idea_cycles=1), evaluator=FakeEvaluator()
    )
    result = await _run(ctx, task)
    assert result.status == "completed_no_winner"
    assert result.best is not None and result.best.source_attempt_id == "bootstrap"


async def test_two_workers_run_through_send_and_push_to_the_store(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    store = InMemoryStore()
    ctx = make_context(
        models=_models(),
        config=config(max_idea_cycles=2, workers=2),
        evaluator=winning_evaluator(),
        policy=AutoApprovePolicy(),
    )
    result = await _run(ctx, task, store=store)
    assert result.status == "completed" and result.workers == 2
    assert len(result.attempts) == 2
    assert {a.worker for a in result.attempts} == {1, 2}
    items = await store.asearch(memory_namespace("run", 0), limit=50)
    keys = sorted(item.key for item in items)
    assert keys == ["delta/0001/01", "delta/0002/02"]
    for item in items:
        worker = item.value["worker"]
        lines = [json.loads(x) for x in item.value["jsonl_lines"]]
        attempt = next(a for a in result.attempts if a.worker == worker)
        # an agent's write_file entry and the host's evaluate event both reached the store
        assert {(r["step"], r["attempt_id"]) for r in lines if r.get("attempt_id")} == {
            ("edit", attempt.attempt_id),
            ("evaluate", attempt.attempt_id),
        }
    assert await workspace.exists("run/results/worker_01/ideas")
    assert await workspace.exists("run/results/worker_02/ideas")


async def test_workers_that_share_memory_need_a_store(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    ctx = make_context(
        models=_models(),
        config=config(max_idea_cycles=2, workers=2),
        evaluator=winning_evaluator(),
        policy=AutoApprovePolicy(),
    )
    with pytest.raises(ConfigurationMismatch, match="store"):
        await _run(ctx, task)


async def test_isolated_worker_memory_needs_no_store_and_shares_nothing(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    ctx = make_context(
        models=_models(),
        config=config(max_idea_cycles=2, workers=2, isolate_worker_memory=True),
        evaluator=winning_evaluator(),
        policy=AutoApprovePolicy(),
    )
    store = InMemoryStore()
    result = await _run(ctx, task, store=store)
    assert result.status == "completed"
    assert await store.asearch(memory_namespace("run", 0), limit=10) == []


async def test_pruning_rounds_restart_each_worker_from_its_best_program(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    evaluator = winning_evaluator()
    ctx = make_context(
        models=_models(),
        config=config(max_idea_cycles=4, workers=2, prune_every_n_cycles=2),
        evaluator=evaluator,
        policy=AutoApprovePolicy(),
    )
    result = await _run(ctx, task, store=InMemoryStore())
    assert result.status == "completed" and len(result.attempts) == 4
    ids = [call.evaluation_id for call in evaluator.calls]
    assert sum(i.endswith("r0/baseline") for i in ids) == 2
    assert not any(i.endswith("r1/baseline") for i in ids), (
        "round 1 starts from each worker's best, scored"
    )
    assert await workspace.exists("run/results/round_000/worker_01/ideas")
    assert await workspace.exists("run/results/round_001/worker_02/ideas")
    seed = (
        await workspace.read_bytes(
            "run/results/round_001/worker_01/.baseline-seed/program_0/src/program.py"
        )
    ).decode()
    assert seed == "SCORE = 6.0\n"


async def test_push_memory_run_again_after_a_crash_leaves_one_item_per_cycle_and_worker(
    make_context: Any, workspace: Any
) -> None:
    class FlakyStore(InMemoryStore):
        """Stores the first write and then raises, as if the caller crashed during the push."""

        crashed = False

        async def aput(
            self, namespace: Any, key: str, value: Any, *args: Any, **kwargs: Any
        ) -> None:
            await super().aput(namespace, key, value, *args, **kwargs)
            if not self.crashed:
                self.crashed = True
                raise ConnectionError("caller lost during push")

    task = await task_with_harness(workspace, harness=True)
    store = FlakyStore()
    models = scripted(
        {
            Role.IDEATOR: [idea_step(HARNESS_IDEA)],
            Role.CANDIDATE_AUTHOR: author_steps("idea_a_attempt_01"),
        }
    )
    ctx = make_context(models=models, config=config(), evaluator=FakeEvaluator())
    from langgraph_agentic_scientist.investigation import build_campaign_worker_graph

    app = build_campaign_worker_graph(checkpointer=InMemorySaver(), store=store)
    cfg = thread("flaky")
    inputs = {
        "task": task,
        "run_id": "run",
        "worker_index": 1,
        "results_dir": "run/results/worker_01",
        "share_memory": True,
        "cycles": 1,
    }
    with pytest.raises(ConnectionError):
        await app.ainvoke(inputs, cfg, context=ctx)
    out = await app.ainvoke(None, cfg, context=ctx)
    assert out["worker_result"].status == "finished"
    items = await store.asearch(memory_namespace("run", 0), limit=50)
    assert [item.key for item in items] == ["delta/0001/01"], (
        "one item per (cycle, worker), not duplicated"
    )


async def test_two_workers_share_memory_including_entries_written_with_write_file(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    store = InMemoryStore()
    line = memory_line("idea_x_attempt_01", "program_1", summary="found that SCORE=6 works")
    first = scripted(
        {
            Role.IDEATOR: [
                tool_call("write_file", {"file_path": "/memory.jsonl", "content": line}),
                idea_step(HARNESS_IDEA, idea_id="idea_x"),
            ],
            Role.CANDIDATE_AUTHOR: author_steps("idea_x_attempt_01", memory=False),
        }
    )
    ctx1 = make_context(models=first, config=config(), evaluator=FakeEvaluator())
    await run_worker(
        ctx1,
        task,
        name="w1",
        store=store,
        worker_index=1,
        cycle_offset=0,
        results_dir="run/results/worker_01",
        share_memory=True,
        cycles=1,
    )
    second = scripted(
        {
            Role.IDEATOR: [idea_step(HARNESS_IDEA, idea_id="idea_y")],
            Role.CANDIDATE_AUTHOR: author_steps("idea_y_attempt_01", memory=False),
        }
    )
    ctx2 = make_context(models=second, config=config(), evaluator=FakeEvaluator())
    await run_worker(
        ctx2,
        task,
        name="w2",
        store=store,
        worker_index=2,
        cycle_offset=1,
        results_dir="run/results/worker_02",
        share_memory=True,
        cycles=1,
    )
    prompt = str(second.scripted(Role.IDEATOR).calls[0][1].content)
    assert "summary=found that SCORE=6 works" in prompt, (
        "worker 1's write_file entry reached worker 2"
    )
    # worker 2's own memory file now holds worker 1's entries first, then its own
    memory = (
        (await workspace.read_bytes("run/results/worker_02/memory.jsonl")).decode().splitlines()
    )
    assert json.loads(memory[0])["attempt_id"] == "idea_x_attempt_01"
    keys = sorted(i.key for i in await store.asearch(memory_namespace("run", 0), limit=50))
    assert keys == ["delta/0001/01", "delta/0002/02"]
