"""The harness-evaluated attempt: the author, the host's checks, evaluation and rubric status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import FakeEvaluator, completed_result, final, tool_call

from .helpers import (
    FIXTURES,
    HARNESS_IDEA,
    author_steps,
    config,
    idea_step,
    ordered_fragments_present,
    run_worker,
    scripted,
    task_with_harness,
)

UPSTREAM = Path(__file__).resolve().parents[2] / "third_party/ai-scientist-v3/src/prompts"
ATTEMPT = "idea_a_attempt_01"
EXPERIMENT = "run/results/ideas/01_raise_the_score"


def _harness_models(author: list[Any]) -> Any:
    return scripted({Role.IDEATOR: [idea_step(HARNESS_IDEA)], Role.CANDIDATE_AUTHOR: author})


async def _status(workspace: Any, path: str) -> dict[str, Any]:
    return json.loads((await workspace.read_bytes(path)).decode())


async def test_prompt_is_the_upstream_text_and_the_author_has_no_package_tools(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    models = _harness_models(author_steps(ATTEMPT))
    ctx = make_context(models=models, config=config(), evaluator=FakeEvaluator())
    result = await run_worker(ctx, task)
    assert result.status == "finished", result.failure
    registry_text = ctx.prompts.text("investigation/harness_edit")
    assert registry_text.encode() == (UPSTREAM / "harness_edit.md").read_bytes()
    author = models.scripted(Role.CANDIDATE_AUTHOR)
    message = author.calls[0][1]
    assert isinstance(message, HumanMessage)
    assert ordered_fragments_present(registry_text, str(message.content))
    assert "Attempt ID: `idea_a_attempt_01`" in str(message.content)
    assert "## Public Search Contract" not in str(message.content)
    names = set(author.bound_tools[0])
    assert not names & {"clone_program", "evaluate_program", "task_gate", "report_blocker", "nl"}
    assert {"execute", "write_file", "edit_file"} <= names
    skill = (
        await workspace.read_bytes(f"{EXPERIMENT}/.agents/skills/edit-program/SKILL.md")
    ).decode()
    # the stage's version of the skill: a plain copy, no clone tool the role does not have
    assert "plain filesystem operations" in skill and "clone_program" not in skill
    assert "name: edit-program" in skill
    assert not await workspace.exists(f"{EXPERIMENT}/.agents/skills/evaluate-program/SKILL.md")


async def test_evaluation_rubric_status_and_memory(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    evaluator = FakeEvaluator(
        results={"run/w0/c001/program_1": lambda r: completed_result(r, combined_score=6.0)}
    )
    ctx = make_context(
        models=_harness_models(author_steps(ATTEMPT)), config=config(), evaluator=evaluator
    )
    result = await run_worker(ctx, task)
    record = result.attempts[0]
    # baseline 1.0 is the incumbent: the gate is combined_score > 1.0, and 6.0 passes it
    assert record.status == "pass" and record.overall_pass and record.failure is None
    assert [e.evaluation_id for e in evaluator.calls] == [
        "run/w0/r0/baseline",
        "run/w0/c001/program_1",
    ]
    assert evaluator.calls[1].timeout_seconds == 7200
    status = await _status(workspace, f"{EXPERIMENT}/rubric_status.json")
    assert status == await _status(workspace, f"{EXPERIMENT}/program_1/rubric_status.json")
    assert [r["id"] for r in status["rubrics"]] == ["incumbent_score_improvement", "sets_score"]
    assert all(r["pass"] and r["assessment"] == "passed" for r in status["rubrics"])
    assert status["metrics"] == {"combined_score": 6.0, "correct": True}
    # the entry the author wrote with write_file and the host's evaluate event are both there
    records = [
        json.loads(x)
        for x in (await workspace.read_bytes("run/results/memory.jsonl")).decode().splitlines()
    ]
    assert [(r["step"], r.get("program_id")) for r in records if r.get("attempt_id")] == [
        ("edit", "program_1"),
        ("evaluate", "program_1"),
    ]
    assert result.best is not None and result.best.source_attempt_id == ATTEMPT
    assert result.best.combined_score == 6.0
    assert (await workspace.read_bytes(f"{EXPERIMENT}/program_1/metrics.json")).decode().strip()


async def test_failed_rubric_and_unacceptable_candidate(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    models = _harness_models(author_steps(ATTEMPT, score="5.0"))
    ctx = make_context(models=models, config=config(), evaluator=FakeEvaluator())
    result = await run_worker(ctx, task)
    status = await _status(workspace, f"{EXPERIMENT}/rubric_status.json")
    by_id = {r["id"]: r for r in status["rubrics"]}
    assert not by_id["sets_score"]["pass"]
    assert "missing source pattern 'SCORE = 6.0'" in by_id["sets_score"]["evidence"]
    assert result.attempts[0].status == "fail"


async def test_incorrect_evaluation_fails_every_rubric_as_not_assessed(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    evaluator = FakeEvaluator(
        results={
            "run/w0/c001/program_1": lambda r: completed_result(
                r, combined_score=9.0, correct=False
            )
        }
    )
    ctx = make_context(
        models=_harness_models(author_steps(ATTEMPT)), config=config(), evaluator=evaluator
    )
    result = await run_worker(ctx, task)
    status = await _status(workspace, f"{EXPERIMENT}/rubric_status.json")
    assert [r["assessment"] for r in status["rubrics"]] == ["not_assessed"] * 2
    assert (
        "criterion not assessed because scientific acceptance failed"
        in status["rubrics"][0]["evidence"]
    )
    assert result.best is not None and result.best.source_attempt_id == "bootstrap"


async def test_failed_evaluation_is_not_assessed_and_not_complete(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    from langgraph_agentic_scientist.evaluation.protocol import EvaluationResult

    def failing(request: Any) -> Any:
        return EvaluationResult(
            evaluation_id=request.evaluation_id, status="timeout", summary="took too long"
        )

    evaluator = FakeEvaluator(results={"run/w0/c001/program_1": failing})
    ctx = make_context(
        models=_harness_models(author_steps(ATTEMPT)), config=config(), evaluator=evaluator
    )
    result = await run_worker(ctx, task)
    status = await _status(workspace, f"{EXPERIMENT}/rubric_status.json")
    assert "criterion not assessed because evaluation failed" in status["rubrics"][0]["evidence"]
    assert "exit=124" in status["rubrics"][0]["evidence"]
    memory = (await workspace.read_bytes("run/results/memory.jsonl")).decode()
    assert "evaluation failed: timeout" in memory
    assert result.attempts[0].best_program is None and result.attempts[0].status == "fail"


async def test_only_program_1_may_be_created(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    steps = [tool_call("execute", {"command": "cp -r program_0 program_2"}), *author_steps(ATTEMPT)]
    evaluator = FakeEvaluator()
    ctx = make_context(models=_harness_models(steps), config=config(), evaluator=evaluator)
    result = await run_worker(ctx, task)
    assert [c.evaluation_id for c in evaluator.calls] == ["run/w0/r0/baseline"], (
        "no candidate evaluation"
    )
    record = result.attempts[0]
    assert record.status == "fail" and record.failure is not None
    assert "forbids extra candidate directories: program_2" in record.failure.detail


async def test_evaluator_artifacts_written_by_the_agent_fail_the_attempt(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    steps = [
        *author_steps(ATTEMPT, memory=False)[:-1],
        tool_call("write_file", {"file_path": "/program_1/metrics.json", "content": "{}"}),
        final(),
    ]
    evaluator = FakeEvaluator()
    ctx = make_context(models=_harness_models(steps), config=config(), evaluator=evaluator)
    result = await run_worker(ctx, task)
    assert len(evaluator.calls) == 1
    assert "agent-created evaluator artifact is forbidden" in str(result.attempts[0].failure.detail)  # type: ignore[union-attr]


async def test_a_missing_memory_record_fails_the_post_check(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    ctx = make_context(
        models=_harness_models(author_steps(ATTEMPT, memory=False)),
        config=config(),
        evaluator=FakeEvaluator(),
    )
    result = await run_worker(ctx, task)
    # the host's own evaluate event carries the attempt id: the post-check passes without the agent
    assert result.attempts[0].failure is None


async def test_a_session_that_runs_out_of_its_wall_clock_is_still_evaluated(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    steps = [
        *author_steps(ATTEMPT, memory=False)[:-1],
        tool_call("execute", {"command": "sleep 3"}),
        final(),
    ]
    evaluator = FakeEvaluator(
        results={"run/w0/c001/program_1": lambda r: completed_result(r, combined_score=6.0)}
    )
    ctx = make_context(
        models=_harness_models(steps), config=config(run_timeout_seconds=1), evaluator=evaluator
    )
    result = await run_worker(ctx, task)
    record = result.attempts[0]
    assert record.timed_out is True
    assert [c.evaluation_id for c in evaluator.calls][-1] == "run/w0/c001/program_1"
    assert record.overall_pass is True


async def test_a_contracted_task_refuses_an_unisolated_evaluator(
    make_context: Any, workspace: Any
) -> None:
    import pytest

    from langgraph_agentic_scientist.core.contract import parse_task_contract_bytes
    from langgraph_agentic_scientist.core.errors import ConfigurationMismatch

    contract = parse_task_contract_bytes(
        (FIXTURES / "core/contract_task/task_contract.json").read_bytes()
    )
    task = (await task_with_harness(workspace, harness=True)).model_copy(
        update={"contract": contract}
    )
    ctx = make_context(
        models=scripted({}), config=config(), evaluator=FakeEvaluator(isolated=False)
    )
    with pytest.raises(ConfigurationMismatch):
        await run_worker(ctx, task)


class _StopBeforeAttempt(Exception):
    """Raised by the capturing slot once it has the attempt's input."""


async def test_the_attempt_graph_runs_alone_with_its_own_checkpointer(
    make_context: Any, workspace: Any
) -> None:
    from langgraph.checkpoint.memory import InMemorySaver

    from langgraph_agentic_scientist import GraphOverrides
    from langgraph_agentic_scientist.investigation import (
        build_attempt_graph,
        build_campaign_worker_graph,
    )
    from langgraph_agentic_scientist.investigation.models import AttemptInput

    task = await task_with_harness(workspace, harness=True)
    captured: dict[str, Any] = {}

    async def capture(state: Any, runtime: Any) -> Any:
        captured.update({key: state[key] for key in AttemptInput.__annotations__ if key in state})
        raise _StopBeforeAttempt

    # run a worker up to its first attempt, to get a real attempt input and workspace
    models = _harness_models(author_steps(ATTEMPT))
    ctx = make_context(models=models, config=config(), evaluator=FakeEvaluator())
    worker = build_campaign_worker_graph(
        checkpointer=InMemorySaver(),
        overrides=GraphOverrides(slots={"investigation.attempt": capture}),
    )
    with pytest.raises(_StopBeforeAttempt):
        await worker.ainvoke(
            {"task": task, "run_id": "run"}, {"configurable": {"thread_id": "w"}}, context=ctx
        )
    assert set(AttemptInput.__required_keys__) <= set(captured)

    attempt = build_attempt_graph(checkpointer=InMemorySaver())
    out = await attempt.ainvoke(captured, {"configurable": {"thread_id": "a"}}, context=ctx)

    result = out["attempt_result"]
    assert result.record.attempt_id == ATTEMPT and result.evaluations
    assert await workspace.exists(f"{EXPERIMENT}/program_1/src/program.py")
    state = await attempt.aget_state({"configurable": {"thread_id": "a"}})
    assert state.next == ()  # the run completed under the caller's checkpointer
