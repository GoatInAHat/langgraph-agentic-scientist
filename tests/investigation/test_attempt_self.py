"""The self-evaluated attempt: the search agent with the clone and evaluation tools."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage

from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import FakeEvaluator, final, tool_call

from .helpers import (
    PLAIN_IDEA,
    config,
    idea_step,
    memory_line,
    run_worker,
    scripted,
    task_with_harness,
)

ATTEMPT = "idea_a_attempt_01"
EXPERIMENT = "run/results/ideas/01_raise_the_score"


def _status(passing: bool = False) -> str:
    return json.dumps(
        {
            "attempt_id": ATTEMPT,
            "idea_id": "idea_a",
            "overall_pass": passing,
            "rubrics": [
                {"id": "score_above_five", "pass": passing, "evidence": "x"},
                {"id": "sets_score", "pass": True, "evidence": "y"},
            ],
            "metrics": {"combined_score": 1.0, "correct": True},
            "next_recommendation": "try again",
        }
    )


def _iteration(index: int, previous: str) -> list[Any]:
    program = f"program_{index}"
    return [
        tool_call("clone_program", {"from_dir": previous, "to_dir": program}),
        tool_call("evaluate_program", {"program_dir": program}),
        tool_call(
            "write_file", {"file_path": f"/{program}/rubric_status.json", "content": _status()}
        ),
        tool_call("execute", {"command": f"cp {program}/rubric_status.json rubric_status.json"}),
    ]


def _memory(count: int) -> Any:
    content = "".join(memory_line(ATTEMPT, f"program_{i}") for i in range(1, count + 1))
    return tool_call(
        "write_file", {"file_path": "/_shared_memory/memory.jsonl", "content": content}
    )


async def test_the_evaluation_budget_stops_the_search_at_four(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    steps: list[Any] = []
    previous = "program_0"
    for index in range(1, 5):
        steps += _iteration(index, previous)
        previous = f"program_{index}"
    steps += [
        tool_call("clone_program", {"from_dir": "program_4", "to_dir": "program_5"}),
        tool_call("evaluate_program", {"program_dir": "program_5"}),
        _memory(4),
        final("cap_reached"),
    ]
    models = scripted({Role.IDEATOR: [idea_step(PLAIN_IDEA)], Role.SEARCH_AGENT: steps})
    evaluator = FakeEvaluator()
    ctx = make_context(models=models, config=config(), evaluator=evaluator)
    result = await run_worker(ctx, task)
    assert [c.evaluation_id for c in evaluator.calls] == [
        f"run/w0/c001/eval_0{i}" for i in range(1, 5)
    ]
    record = result.attempts[0]
    assert record.failure is None, record.failure
    assert record.status == "fail" and record.overall_pass is False
    assert len(result.evaluations) == 4
    assert record.best_program is not None and record.best_program.path.endswith("program_1")
    assert not await workspace.exists(f"{EXPERIMENT}/program_5/metrics.json")
    # the search agent saw the tool-named prompt and was given the two tools
    search = models.scripted(Role.SEARCH_AGENT)
    prompt = str(search.calls[0][1].content)
    assert isinstance(search.calls[0][1], HumanMessage)
    assert 'evaluate_program(program_dir="./program_<idx>")' in prompt
    assert "src.investigation" not in prompt and "TASK_PYTHON" not in prompt
    assert {"clone_program", "evaluate_program"} <= set(search.bound_tools[0])
    # evaluate.py and capture.py are staged as read-only views for a self-evaluated attempt
    assert await workspace.exists(f"{EXPERIMENT}/evaluate.py")


async def test_a_passing_attempt_is_recorded_and_the_incumbent_improves(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    status = _status(passing=True)
    steps = [
        tool_call("clone_program", {"from_dir": "program_0", "to_dir": "program_1"}),
        tool_call(
            "edit_file",
            {
                "file_path": "/program_1/src/program.py",
                "old_string": "SCORE = 1.0",
                "new_string": "SCORE = 6.0",
            },
        ),
        tool_call("evaluate_program", {"program_dir": "program_1"}),
        tool_call("write_file", {"file_path": "/program_1/rubric_status.json", "content": status}),
        tool_call("execute", {"command": "cp program_1/rubric_status.json rubric_status.json"}),
        _memory(1),
        final("rubric_pass"),
    ]
    from langgraph_agentic_scientist.testing import completed_result

    evaluator = FakeEvaluator(default=lambda r: completed_result(r, combined_score=6.0))
    ctx = make_context(
        models=scripted({Role.IDEATOR: [idea_step(PLAIN_IDEA)], Role.SEARCH_AGENT: steps}),
        config=config(),
        evaluator=evaluator,
    )
    result = await run_worker(ctx, task)
    assert result.attempts[0].status == "pass"
    assert result.best is not None
    assert (result.best.source_attempt_id, result.best.combined_score) == (ATTEMPT, 6.0)
    assert result.status == "finished"


async def test_post_checks_read_the_memory_records_from_the_files(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    steps = [*_iteration(1, "program_0"), final("done without memory")]
    ctx = make_context(
        models=scripted({Role.IDEATOR: [idea_step(PLAIN_IDEA)], Role.SEARCH_AGENT: steps}),
        config=config(),
        evaluator=FakeEvaluator(),
    )
    result = await run_worker(ctx, task)
    failure = result.attempts[0].failure
    assert failure is not None
    assert "memory.jsonl must contain at least one entry with current attempt_id" in failure.detail


async def test_memory_for_the_wrong_program_fails_the_post_check(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    steps = [*_iteration(1, "program_0"), _memory(0), final()]
    steps[-2] = tool_call(
        "write_file",
        {"file_path": "/_shared_memory/memory.jsonl", "content": memory_line(ATTEMPT, "program_9")},
    )
    ctx = make_context(
        models=scripted({Role.IDEATOR: [idea_step(PLAIN_IDEA)], Role.SEARCH_AGENT: steps}),
        config=config(),
        evaluator=FakeEvaluator(),
    )
    result = await run_worker(ctx, task)
    assert "latest evaluated program_id=program_1" in (result.attempts[0].failure.detail)  # type: ignore[union-attr]


async def test_a_root_mirror_that_differs_fails_the_post_check(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    steps = _iteration(1, "program_0")[:3]
    steps += [
        tool_call(
            "write_file",
            {
                "file_path": "/rubric_status.json",
                "content": _status(passing=False)
                .replace("try again", "x")
                .replace('"pass": true', '"pass": false'),
            },
        ),
        _memory(1),
        final(),
    ]
    ctx = make_context(
        models=scripted({Role.IDEATOR: [idea_step(PLAIN_IDEA)], Role.SEARCH_AGENT: steps}),
        config=config(),
        evaluator=FakeEvaluator(),
    )
    result = await run_worker(ctx, task)
    failure = result.attempts[0].failure
    assert failure is not None and "root rubric_status.json" in failure.detail
