"""Bounds of an unattended session: subagents, cancelled commands, command output."""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from testkit import fast_config
from typing_extensions import TypedDict

from langgraph_agentic_scientist import ScientistContext
from langgraph_agentic_scientist.agents.factory import DeepAgentFactory
from langgraph_agentic_scientist.agents.runner import RoleOutcome, run_workspace_role
from langgraph_agentic_scientist.config import ModelRetryConfig
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import ScriptedModels, final, tool_call

SUBTASK = "Inspect the directory and report."


class _State(TypedDict, total=False):
    done: bool


def _is_subagent(messages: list[BaseMessage]) -> bool:
    return any(isinstance(m, HumanMessage) and m.content == SUBTASK for m in messages)


def _main_or_sub(sub: AIMessage | BaseException) -> Any:
    def step(messages: list[BaseMessage]) -> AIMessage:
        if _is_subagent(messages):
            if isinstance(sub, BaseException):
                raise sub
            return sub
        if any(isinstance(m, ToolMessage) for m in messages):
            return final("main done")
        return tool_call("task", {"description": SUBTASK, "subagent_type": "general-purpose"})

    return step


async def _run(ctx: ScientistContext) -> RoleOutcome:
    holder: dict[str, RoleOutcome] = {}

    async def node(state: _State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        holder["outcome"] = await run_workspace_role(
            role=Role.TASK_DRAFTER, runtime=runtime, root="work", prompt="Delegate the task."
        )
        return {"done": True}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(_State, context_schema=ScientistContext)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    app = graph.compile(checkpointer=InMemorySaver())
    await app.ainvoke({}, {"configurable": {"thread_id": "bounds"}}, context=ctx)
    return holder["outcome"]


def _config(**agents: Any) -> Any:
    base = fast_config()
    return base.model_copy(update={"agents": base.agents.model_copy(update=agents)})


async def test_the_task_subagent_retries_a_transient_provider_failure(make_context: Any) -> None:
    models = ScriptedModels.from_scripts(
        {
            Role.TASK_DRAFTER: [
                _main_or_sub(final("unused")),
                ConnectionError("gateway dropped the connection"),
                final("sub done"),
                _main_or_sub(final("unused")),
            ]
        },
        context_window=200_000,
    )
    ctx = make_context(
        models=models,
        config=_config(model_retry=ModelRetryConfig(max_retries=2, initial_delay=0.0)),
    )
    outcome = await _run(ctx)
    assert outcome.ok, outcome.failure
    assert outcome.final_text == "main done"
    assert models.scripted(Role.TASK_DRAFTER).call_count == 4


async def test_the_task_subagent_is_bounded_by_the_model_call_limit(make_context: Any) -> None:
    looping = tool_call("ls", {"path": "/"})
    models = ScriptedModels.from_scripts(
        {Role.TASK_DRAFTER: [_main_or_sub(looping) for _ in range(40)]},
        context_window=200_000,
    )
    ctx = make_context(models=models, config=_config(model_call_limit=3))
    outcome = await _run(ctx)
    assert outcome.ok, outcome.failure
    assert outcome.final_text == "main done"
    assert models.scripted(Role.TASK_DRAFTER).call_count == 1 + 3 + 1


async def test_a_cancelled_command_kills_its_process_group(make_context: Any) -> None:
    ctx = make_context()
    backend = DeepAgentFactory().backend(ctx, "shell", read_only=False)
    command = "sleep 60 & echo $! > child.pid; wait"
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.aexecute(command, timeout=120), timeout=1.0)
    pid = int((await ctx.workspace.read_bytes("shell/child.pid")).decode().strip())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.1)
    os.kill(pid, 9)
    raise AssertionError("the command's child survived the cancellation")


async def test_command_output_is_bounded_by_the_configured_bytes(make_context: Any) -> None:
    ctx = make_context(config=_config(max_tool_output_bytes=64))
    backend = DeepAgentFactory().backend(ctx, "shell", read_only=False)
    result = await backend.aexecute("python3 -c \"print('x' * 4000)\"")
    assert result.truncated
    assert result.output.count("x") <= 64


def test_the_context_window_copy_keeps_the_callers_rate_limiter() -> None:
    from langchain_core.rate_limiters import InMemoryRateLimiter

    from langgraph_agentic_scientist.agents.factory import with_context_window
    from langgraph_agentic_scientist.testing import ScriptedChatModel

    limiter = InMemoryRateLimiter(requests_per_second=1)
    model = ScriptedChatModel(rate_limiter=limiter)
    assert with_context_window(model, 1000).rate_limiter is limiter
