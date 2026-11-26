"""The campaign worker: incumbents, terminal failures, the prune question, resume."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langgraph_agentic_scientist import GraphOverrides, Hooks
from langgraph_agentic_scientist.core.state import Answer, Question
from langgraph_agentic_scientist.evaluation.protocol import EvaluationResult
from langgraph_agentic_scientist.investigation import build_campaign_worker_graph
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import FakeEvaluator, completed_result

from .helpers import (
    HARNESS_IDEA,
    author_steps,
    config,
    idea_step,
    run_worker,
    scripted,
    task_with_harness,
    thread,
)


def _two_cycles() -> Any:
    return scripted(
        {
            Role.IDEATOR: [
                idea_step(HARNESS_IDEA),
                idea_step(HARNESS_IDEA, idea_id="idea_b", title="Another try"),
            ],
            Role.CANDIDATE_AUTHOR: [
                *author_steps("idea_a_attempt_01", score="6.0"),
                *author_steps("idea_b_attempt_01", score="5.0", old="SCORE = 6.0"),
            ],
        }
    )


def _scores(table: dict[str, float]) -> FakeEvaluator:
    return FakeEvaluator(
        default=lambda r: completed_result(
            r, combined_score=next((v for k, v in table.items() if k in r.evaluation_id), 1.0)
        )
    )


async def test_only_a_strict_improvement_replaces_the_incumbent(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    evaluator = _scores({"c001": 3.0, "c002": 2.0})
    ctx = make_context(models=_two_cycles(), config=config(max_idea_cycles=2), evaluator=evaluator)
    result = await run_worker(ctx, task)
    assert result.status == "finished"
    assert [a.attempt_id for a in result.attempts] == ["idea_a_attempt_01", "idea_b_attempt_01"]
    assert result.best is not None
    assert (result.best.source_attempt_id, result.best.combined_score) == ("idea_a_attempt_01", 3.0)
    assert result.best.program_dir.endswith("01_raise_the_score/program_1")
    # the second cycle was seeded from the incumbent, and its gate was the incumbent's score
    gate = json.loads(
        (
            await workspace.read_bytes("run/results/ideas/02_another_try/active_rubrics.json")
        ).decode()
    )[0]
    assert gate["id"] == "incumbent_score_improvement" and gate["threshold"] == 3.0
    best_file = json.loads(
        (await workspace.read_bytes("run/results/best_program_manifest.json")).decode()
    )
    assert best_file["combined_score"] == 3.0
    ideas = (await workspace.read_bytes("run/results/ideas_history.jsonl")).decode().splitlines()
    assert [json.loads(x)["idea_id"] for x in ideas] == ["idea_a", "idea_b"]


async def test_a_non_improving_campaign_keeps_the_baseline(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    ctx = make_context(
        models=_two_cycles(), config=config(max_idea_cycles=2), evaluator=FakeEvaluator()
    )
    result = await run_worker(ctx, task)
    assert result.best is not None and result.best.source_attempt_id == "bootstrap"
    assert result.baseline is not None and result.baseline.evaluation_id == "run/w0/r0/baseline"


async def test_a_contract_failure_of_the_evaluator_stops_the_campaign(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)

    def broken(request: Any) -> EvaluationResult:
        return EvaluationResult(
            evaluation_id=request.evaluation_id,
            status="failed",
            failure_category="evaluator_output_contract_invalid",
            summary="bad outputs",
        )

    models = _two_cycles()
    evaluator = FakeEvaluator(results={"run/w0/c001/program_1": broken})
    ctx = make_context(models=models, config=config(max_idea_cycles=2), evaluator=evaluator)
    result = await run_worker(ctx, task)
    assert result.status == "failed"
    assert (
        result.failure is not None
        and result.failure.category == "evaluator_output_contract_invalid"
    )
    assert models.scripted(Role.IDEATOR).call_count == 1, "no second cycle"


async def test_exhausted_ideation_ends_the_campaign_and_sets_the_cycle_back(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    models = scripted(
        {Role.IDEATOR: [idea_step(HARNESS_IDEA, title=""), idea_step(HARNESS_IDEA, title="")]}
    )
    ctx = make_context(models=models, config=config(max_idea_cycles=3), evaluator=FakeEvaluator())
    result = await run_worker(ctx, task)
    assert result.status == "failed" and result.failure is not None
    assert result.failure.category == "campaign_failed"
    assert result.outer.current_cycle == 0 and result.outer.failure_streak == 1
    assert len(result.idea_failures) == 2


async def test_a_baseline_that_cannot_be_validated_ends_the_campaign(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    evaluator = FakeEvaluator(
        default=lambda r: EvaluationResult(
            evaluation_id=r.evaluation_id, status="failed", summary="no"
        )
    )
    models = scripted({})
    ctx = make_context(models=models, config=config(), evaluator=evaluator)
    result = await run_worker(ctx, task)
    assert result.failure is not None
    assert result.failure.category == "missing_validated_harness_incumbent"


class CountingPolicy:
    """Answers the prune question and counts how often it is asked."""

    def __init__(self, choice: str = "proceed") -> None:
        self.choice = choice
        self.questions: list[Question] = []

    def decide(self, question: Question) -> Answer | None:
        self.questions.append(question)
        return Answer(choice=self.choice)


def _three_cycles() -> Any:
    ideas = [idea_step(HARNESS_IDEA, idea_id=f"idea_{n}", title=f"Try {n}") for n in "abc"]
    authors: list[Any] = []
    for n in "abc":
        authors += author_steps(f"idea_{n}_attempt_01", old="SCORE = 1.0", memory=False)
    return scripted({Role.IDEATOR: ideas, Role.CANDIDATE_AUTHOR: authors})


async def test_the_prune_question_is_asked_once_before_the_loop_not_before_each_prune(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    policy = CountingPolicy()
    ctx = make_context(
        models=_three_cycles(),
        config=config(max_idea_cycles=3, prune_every_n_cycles=1),
        evaluator=FakeEvaluator(),
        policy=policy,
    )
    result = await run_worker(ctx, task)
    assert result.status == "finished"
    assert len(policy.questions) == 1
    question = policy.questions[0]
    assert question.kind == "prune_confirmation" and "Pruning may be ineffective" in question.text
    # pruning happened at the boundaries: chapters archived, histories show the current chapter only
    assert await workspace.exists("run/results/_pruned/chapter_001/memory.jsonl")
    assert await workspace.exists("run/results/_pruned/chapter_002/memory.jsonl")
    assert result.outer.prune_chapter == 3
    ideas = (await workspace.read_bytes("run/results/ideas_history.jsonl")).decode().splitlines()
    assert len(ideas) == 1, "the history file shows only the current chapter (cycle 3)"
    assert len(result.ideas) == 3, "the result keeps every idea"


async def test_the_prune_question_goes_to_the_person_once_and_survives_a_resume(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    crashed: list[bool] = []

    async def crash_once(state: Any, runtime: Any) -> None:
        if not crashed:
            crashed.append(True)
            raise RuntimeError("crash between accepting the idea and the attempt")

    models = _three_cycles()
    hooks = Hooks(before={"worker.attempt": [crash_once]})
    ctx = make_context(
        models=models,
        config=config(max_idea_cycles=3, prune_every_n_cycles=1),
        evaluator=FakeEvaluator(),
        hooks=hooks,
    )
    app = build_campaign_worker_graph(checkpointer=InMemorySaver())
    cfg = thread("prune")
    out = await app.ainvoke({"task": task, "run_id": "run"}, cfg, context=ctx)
    assert out["__interrupt__"][0].value["kind"] == "prune_confirmation"
    with pytest.raises(RuntimeError):
        await app.ainvoke(Command(resume="proceed"), cfg, context=ctx)
    assert models.scripted(Role.IDEATOR).call_count == 1
    out = await app.ainvoke(None, cfg, context=ctx)
    assert "__interrupt__" not in out, "the answer is in state; the question is not asked again"
    assert out["worker_result"].status == "finished"
    assert models.scripted(Role.IDEATOR).call_count == 3, "the first idea was not generated twice"


async def test_abort_ends_the_campaign_before_any_cycle(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    models = _three_cycles()
    ctx = make_context(
        models=models,
        config=config(max_idea_cycles=3, prune_every_n_cycles=1),
        evaluator=FakeEvaluator(),
        policy=CountingPolicy("abort"),
    )
    result = await run_worker(ctx, task)
    assert result.status == "failed" and result.failure is not None
    assert result.failure.category == "prune_aborted"
    assert models.scripted(Role.IDEATOR).call_count == 0


async def test_an_isolated_workspace_is_not_asked(make_context: Any, workspace: Any) -> None:
    from langgraph_agentic_scientist import LocalWorkspace

    class Isolated(LocalWorkspace):
        @property
        def isolated(self) -> bool:
            return True

    isolated = Isolated(workspace.root)
    task = await task_with_harness(isolated, harness=True)
    policy = CountingPolicy()
    ctx = make_context(
        models=_three_cycles(),
        config=config(max_idea_cycles=2, prune_every_n_cycles=1),
        evaluator=FakeEvaluator(),
        policy=policy,
        workspace=isolated,
    )
    result = await run_worker(ctx, task)
    assert result.status == "finished" and policy.questions == []


async def test_a_resumed_worker_repeats_no_model_call_and_no_evaluation(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    models = _two_cycles()
    evaluator = FakeEvaluator(fail_once=["run/w0/c001/program_1"], reattach=True)
    ctx = make_context(models=models, config=config(max_idea_cycles=2), evaluator=evaluator)
    app = build_campaign_worker_graph(checkpointer=InMemorySaver())
    cfg = thread("resume")
    with pytest.raises(ConnectionError):
        await app.ainvoke({"task": task, "run_id": "run"}, cfg, context=ctx)
    calls = (
        models.scripted(Role.IDEATOR).call_count,
        models.scripted(Role.CANDIDATE_AUTHOR).call_count,
    )
    assert calls == (1, 4)
    out = await app.ainvoke(None, cfg, context=ctx)
    result = out["worker_result"]
    assert result.status == "finished" and len(result.attempts) == 2
    assert evaluator.runs["run/w0/c001/program_1"] == 1, (
        "the interrupted evaluation was not run twice"
    )
    assert all(count == 1 for count in evaluator.runs.values())
    # nothing before the crash was asked of a model again: the first idea and author ran once
    assert models.scripted(Role.IDEATOR).call_count == 2
    assert models.scripted(Role.CANDIDATE_AUTHOR).call_count == 8


async def test_hooks_fire_around_the_steps_of_the_worker(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    seen: list[str] = []

    async def mark(state: Any, runtime: Any) -> None:
        seen.append("accept_idea.after")

    ctx = make_context(
        models=_two_cycles(),
        config=config(),
        evaluator=FakeEvaluator(),
        hooks=Hooks(after={"worker.accept_idea": [mark]}),
    )
    await run_worker(ctx, task)
    assert seen == ["accept_idea.after"]


async def test_the_ideation_and_attempt_slots_are_replaceable(
    make_context: Any, workspace: Any
) -> None:
    from langgraph_agentic_scientist.core.state import IdeaBundle, RubricSpec
    from langgraph_agentic_scientist.investigation.models import IdeationResult

    task = await task_with_harness(workspace, harness=False)

    async def fixed_ideation(value: Any, runtime: Any) -> dict[str, Any]:
        bundle = IdeaBundle(
            idea_id="fixed",
            title="Fixed",
            hypothesis="h",
            implementation_plan=["p"],
            target_rubrics=[RubricSpec(id="r", description="d")],
        )
        return {"ideation": IdeationResult(status="accepted", bundle=bundle, attempts_used=1)}

    async def no_attempt(value: Any, runtime: Any) -> dict[str, Any]:
        from langgraph_agentic_scientist.core.results import AttemptRecord
        from langgraph_agentic_scientist.investigation.models import AttemptResult

        record = AttemptRecord(
            attempt_id=value["cycle"].attempt_id,
            idea_id="fixed",
            cycle=1,
            attempt_dir="x",
            status="fail",
        )
        return {"attempt_result": AttemptResult(record=record, experiment_dir="x")}

    overrides = GraphOverrides(
        slots={"investigation.ideation": fixed_ideation, "investigation.attempt": no_attempt}
    )
    models = scripted({})
    ctx = make_context(models=models, config=config(), evaluator=FakeEvaluator())
    from langgraph_agentic_scientist.investigation import build_campaign_worker_graph as build

    app = build(checkpointer=InMemorySaver(), overrides=overrides)
    out = await app.ainvoke({"task": task, "run_id": "run"}, thread("slots"), context=ctx)
    result = out["worker_result"]
    assert [i.idea_id for i in result.ideas] == ["fixed"]
    assert result.attempts[0].attempt_id == "fixed_attempt_01"


async def test_the_worker_checkpoints_and_resumes_under_the_strict_serializer(
    make_context: Any, workspace: Any
) -> None:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from langgraph_agentic_scientist import serde_allowlist

    task = await task_with_harness(workspace, harness=True)
    models = _two_cycles()
    evaluator = FakeEvaluator(fail_once=["run/w0/c001/program_1"], reattach=True)
    ctx = make_context(models=models, config=config(max_idea_cycles=2), evaluator=evaluator)
    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist()))
    app = build_campaign_worker_graph(checkpointer=saver)
    cfg = thread("strict")
    with pytest.raises(ConnectionError):
        await app.ainvoke({"task": task, "run_id": "run"}, cfg, context=ctx)
    out = await app.ainvoke(None, cfg, context=ctx)
    assert out["worker_result"].status == "finished"
    assert len(out["worker_result"].attempts) == 2
