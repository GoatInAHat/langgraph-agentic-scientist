"""Builders shared by the investigation tests: ideas, agent scripts, configurations, runs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from langgraph_agentic_scientist import ScientistConfig, ScientistContext
from langgraph_agentic_scientist.config import AgentsConfig, InvestigationConfig, ModelRetryConfig
from langgraph_agentic_scientist.core.state import TaskRef
from langgraph_agentic_scientist.investigation.ideas import IdeaDraft
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import (
    FakeEvaluator,
    ScriptedModels,
    final,
    structured_call,
    tool_call,
    write_minimal_task,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

PLAIN_IDEA: dict[str, Any] = {
    "idea_id": "idea_a",
    "title": "Raise the score",
    "hypothesis": "A larger SCORE is better while it stays at most 10.",
    "implementation_plan": ["Set SCORE to 6.0"],
    "target_rubrics": [
        {"id": "score_above_five", "description": "combined_score > 5", "type": "metric"},
        {"id": "sets_score", "description": "The program sets SCORE.", "type": "implementation"},
    ],
    "notes_for_agent": "Edit only SCORE.",
}

HARNESS_IDEA: dict[str, Any] = {
    **PLAIN_IDEA,
    "target_rubrics": [
        {"id": "agent_metric", "description": "combined_score > 100", "type": "metric"},
        {
            "id": "sets_score",
            "description": "The program sets SCORE to 6.",
            "type": "implementation",
            "verification": {"type": "source_contains", "patterns": ["SCORE = 6.0"]},
        },
    ],
}


def idea_step(idea: Mapping[str, Any] | None = None, **updates: Any) -> Any:
    """A script step: the ideator's structured answer."""
    return structured_call(IdeaDraft, {**(idea or PLAIN_IDEA), **updates})


def memory_line(attempt_id: str, program_id: str | None, **extra: Any) -> str:
    """One JSONL memory record, as an agent would write it."""
    record = {
        "timestamp": "2026-01-01T00:00:00Z",
        "step": "edit",
        "scope": "attempt",
        "summary": "s",
        "evidence": "e",
        "next_action": "n",
        "attempt_id": attempt_id,
        **extra,
    }
    if program_id:
        record["program_id"] = program_id
    return json.dumps(record) + "\n"


def author_steps(
    attempt_id: str, *, score: str = "6.0", old: str = "SCORE = 1.0", memory: bool = True
) -> list[Any]:
    """The candidate author's script: clone by plain commands, edit, write memory, stop."""
    steps: list[Any] = [
        tool_call("execute", {"command": "cp -r program_0 program_1"}),
        tool_call(
            "edit_file",
            {
                "file_path": "/program_1/src/program.py",
                "old_string": old,
                "new_string": f"SCORE = {score}",
            },
        ),
    ]
    if memory:
        steps.append(
            tool_call(
                "write_file",
                {
                    "file_path": "/_shared_memory/memory.jsonl",
                    "content": memory_line(attempt_id, "program_1"),
                },
            )
        )
    steps.append(final("program_1 is ready"))
    return steps


def config(**investigation: Any) -> ScientistConfig:
    """A configuration whose retries do not sleep; ``investigation`` updates that section."""
    defaults = {"max_idea_cycles": 1, "max_idea_generation_attempts": 2}
    return ScientistConfig(
        agents=AgentsConfig(
            model_retry=ModelRetryConfig(
                max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0
            )
        ),
        investigation=InvestigationConfig(**{**defaults, **investigation}),
    )


async def task_with_harness(workspace: Any, *, harness: bool) -> TaskRef:
    """The minimal task under ``run/task``; harness-evaluated when asked."""
    task = await write_minimal_task(workspace, "run/task")
    return task.model_copy(update={"harness_evaluated": harness})


def scripted(scripts: Mapping[Role | str, Sequence[Any]], **kwargs: Any) -> ScriptedModels:
    """Scripted models with a window big enough for the tests."""
    return ScriptedModels.from_scripts(scripts, context_window=200_000, **kwargs)


def evaluator_scoring(score_of: Callable[[str], float]) -> FakeEvaluator:
    """A fake evaluator whose score depends on the evaluation id."""
    from langgraph_agentic_scientist.testing import completed_result

    return FakeEvaluator(
        default=lambda request: completed_result(
            request, combined_score=score_of(request.evaluation_id)
        )
    )


def thread(name: str) -> dict[str, Any]:
    """A run config with a thread id."""
    return {"configurable": {"thread_id": name}, "recursion_limit": 200}


def saver_and_store() -> tuple[InMemorySaver, InMemoryStore]:
    """A fresh in-memory checkpointer and store."""
    return InMemorySaver(), InMemoryStore()


__all__ = [
    "HARNESS_IDEA",
    "PLAIN_IDEA",
    "ScientistContext",
    "author_steps",
    "config",
    "evaluator_scoring",
    "final",
    "idea_step",
    "memory_line",
    "saver_and_store",
    "scripted",
    "task_with_harness",
    "thread",
]


async def run_worker(
    ctx: Any, task: TaskRef, *, name: str = "w", store: Any = None, **inputs: Any
) -> Any:
    """Run the worker graph alone to the end and return its :class:`WorkerResult`."""
    from langgraph_agentic_scientist.investigation import build_campaign_worker_graph

    app = build_campaign_worker_graph(checkpointer=InMemorySaver(), store=store)
    out = await app.ainvoke({"task": task, "run_id": "run", **inputs}, thread(name), context=ctx)
    return out["worker_result"]


def ordered_fragments_present(template: str, rendered: str) -> bool:
    """Whether every literal fragment of a template occurs in the rendered text, in order."""
    import re

    position = 0
    for fragment in re.split(r"\{\{[A-Za-z0-9_]+\}\}", template):
        if not fragment:
            continue
        found = rendered.find(fragment, position)
        if found < 0:
            return False
        position = found + len(fragment)
    return True


def _first_human(messages: Sequence[Any]) -> str:
    from langchain_core.messages import HumanMessage

    return next(str(m.content) for m in messages if isinstance(m, HumanMessage))


def author_policy(messages: Sequence[Any]) -> Any:
    """A candidate author that decides its next step from its messages (safe under concurrency).

    Clones by a plain command, edits ``SCORE``, creates the memory view with ``write_file`` (with
    the attempt id read from its own prompt) and stops.
    """
    import re

    from langchain_core.messages import ToolMessage

    done = sum(isinstance(m, ToolMessage) for m in messages)
    attempt = re.search(r"Attempt ID: `([^`]+)`", _first_human(messages))
    attempt_id = attempt.group(1) if attempt else "unknown"
    if done == 0:
        return tool_call("execute", {"command": "cp -r program_0 program_1"})
    if done == 1:
        return tool_call(
            "edit_file",
            {
                "file_path": "/program_1/src/program.py",
                "old_string": "SCORE = 1.0",
                "new_string": "SCORE = 6.0",
            },
        )
    if done == 2:
        return tool_call(
            "write_file",
            {
                "file_path": "/_shared_memory/memory.jsonl",
                "content": memory_line(attempt_id, "program_1"),
            },
        )
    return final("program_1 is ready")


def ideator_policy() -> Callable[[Sequence[Any]], Any]:
    """An ideator that returns a fresh harness idea on every call (safe under concurrency)."""
    import itertools

    counter = itertools.count(1)

    def policy(messages: Sequence[Any]) -> Any:
        number = next(counter)
        return idea_step(HARNESS_IDEA, idea_id=f"idea_{number}", title=f"Try number {number}")

    return policy


def winning_evaluator(score: float = 6.0) -> FakeEvaluator:
    """Baseline 1.0; every candidate ``score``."""
    from langgraph_agentic_scientist.testing import completed_result

    return FakeEvaluator(
        default=lambda r: completed_result(
            r, combined_score=1.0 if r.evaluation_id.endswith("baseline") else score
        )
    )
