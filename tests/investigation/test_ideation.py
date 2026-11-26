"""The ideation graph: valid bundle, retry with feedback, harness rules, exhaustion."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_agentic_scientist.core.state import BestManifest
from langgraph_agentic_scientist.investigation import build_ideation_graph
from langgraph_agentic_scientist.investigation.models import CycleScratch
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import final, tool_call

from .helpers import (
    HARNESS_IDEA,
    PLAIN_IDEA,
    config,
    idea_step,
    scripted,
    task_with_harness,
    thread,
)


async def _run(ctx: Any, task: Any, name: str = "ideation") -> Any:
    app = build_ideation_graph(checkpointer=InMemorySaver())
    await ctx.workspace.write_bytes("run/results/memory.md", b"# Global Memory Log\n\n")
    out = await app.ainvoke(
        {
            "task": task,
            "run_id": "run",
            "results_dir": "run/results",
            "worker_index": 0,
            "cycle": CycleScratch(index=1, global_cycle=1),
        },
        thread(name),
        context=ctx,
    )
    return out["ideation"]


def _prompt(models: Any, call: int = 0) -> str:
    message = models.scripted(Role.IDEATOR).calls[call][1]
    assert isinstance(message, HumanMessage)
    return str(message.content)


async def test_valid_bundle_is_accepted_and_files_are_written(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    models = scripted({Role.IDEATOR: [idea_step()]})
    result = await _run(make_context(models=models, config=config()), task)
    assert result.status == "accepted" and result.attempts_used == 1
    assert result.bundle.idea_id == "idea_a"
    staging = result.idea_workspace
    assert staging.startswith("run/results/_idea_generation/cycle_0001_attempt_01")
    bundle = json.loads((await workspace.read_bytes(f"{staging}/idea_bundle.json")).decode())
    rubrics = json.loads((await workspace.read_bytes(f"{staging}/target_rubrics.json")).decode())
    assert rubrics == bundle["target_rubrics"]
    # the host appended the auto-healed idea-step memory entry, once
    memory = (await workspace.read_bytes("run/results/memory.jsonl")).decode().splitlines()
    assert [json.loads(line)["step"] for line in memory] == ["idea"]
    prompt = _prompt(models)
    assert "Make `SCORE` in the program as large as possible" in prompt
    assert "- idea_bundle_file" not in prompt and "{{" not in prompt
    # the ideator's skills are staged under its root
    assert await workspace.exists("run/results/.agents/skills/create-idea/SKILL.md")
    assert await workspace.exists("run/results/.agents/skills/memory-log/scripts/append_memory.py")


async def test_invalid_then_valid_carries_the_feedback_in_the_second_prompt(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    models = scripted({Role.IDEATOR: [idea_step(title=" "), idea_step()]})
    result = await _run(make_context(models=models, config=config()), task)
    assert result.status == "accepted" and result.attempts_used == 2
    assert [f.reason for f in result.failures] == ["invalid_idea_bundle"]
    first, second = _prompt(models, 0), _prompt(models, 1)
    assert "Previous idea artifact validation failed" not in first
    assert "Previous idea artifact validation failed" in second
    assert "title must be non-empty string" in second
    assert result.idea_workspace.endswith("attempt_02")


async def test_harness_mode_replaces_metric_rubrics_with_the_incumbent_gate(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    best = BestManifest(program_dir="x", combined_score=4.0, correct=True)
    await workspace.write_bytes(
        "run/results/best_program_manifest.json", json.dumps(best.model_dump(mode="json")).encode()
    )
    models = scripted({Role.IDEATOR: [idea_step(HARNESS_IDEA)]})
    result = await _run(make_context(models=models, config=config()), task)
    assert [r.id for r in result.bundle.target_rubrics] == [
        "incumbent_score_improvement",
        "sets_score",
    ]
    assert result.bundle.target_rubrics[0].description == "combined_score > 4.0"
    assert "Harness mode is enabled." in _prompt(models)


async def test_harness_mode_rejects_a_missing_verification_and_retries(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    models = scripted({Role.IDEATOR: [idea_step(PLAIN_IDEA), idea_step(HARNESS_IDEA)]})
    result = await _run(make_context(models=models, config=config()), task)
    assert result.status == "accepted"
    assert [f.reason for f in result.failures] == ["missing_harness_implementation_verification"]


async def test_exhaustion_ends_the_cycle_with_every_failure(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    models = scripted({Role.IDEATOR: [idea_step(title=""), idea_step(hypothesis="")]})
    result = await _run(make_context(models=models, config=config()), task)
    assert result.status == "exhausted" and result.bundle is None
    assert result.failure is not None and result.failure.category == "campaign_failed"
    assert [f.generation_attempt for f in result.failures] == [1, 2]


async def test_a_session_that_answers_nothing_is_a_failed_attempt(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    models = scripted({Role.IDEATOR: [final("no structured answer"), idea_step()]})
    result = await _run(make_context(models=models, config=config()), task)
    assert result.status == "accepted"
    assert [f.reason for f in result.failures] == ["idea_generation_exit_nonzero"]


async def test_the_ideator_may_write_memory_with_its_file_tools(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=False)
    line = (
        json.dumps(
            {
                "step": "idea",
                "scope": "idea",
                "summary": "s",
                "evidence": "e",
                "next_action": "n",
                "idea_id": "idea_a",
            }
        )
        + "\n"
    )
    models = scripted(
        {
            Role.IDEATOR: [
                tool_call("write_file", {"file_path": "/memory.jsonl", "content": line}),
                idea_step(),
            ]
        }
    )
    result = await _run(make_context(models=models, config=config()), task)
    assert result.status == "accepted"
    memory = (await workspace.read_bytes("run/results/memory.jsonl")).decode()
    assert memory == line, "the agent's entry was kept and the fallback entry was not duplicated"
