"""The agent runner: sessions as subgraphs, resume mid-session, wall clock, strategies, windows."""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

import pytest
from deepagents.backends.protocol import SandboxBackendProtocol
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from langgraph_agentic_scientist import ScientistConfig, ScientistContext
from langgraph_agentic_scientist.agents.factory import (
    DeepAgentFactory,
    response_format_for,
    with_context_window,
)
from langgraph_agentic_scientist.agents.runner import run_structured_role, run_workspace_role
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import (
    ScriptedChatModel,
    ScriptedModels,
    final,
    json_answer,
    structured_call,
    tool_call,
)


class Verdict(BaseModel):
    """A small structured answer."""

    verdict: str
    score: int


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]
    note: str
    verdict: dict[str, Any] | None
    timed_out: bool


def _two_node_graph(
    counter: dict[str, int], *, slow: bool = False
) -> StateGraph[Any, Any, Any, Any]:
    @tool
    async def count_me(label: str) -> str:
        """Count a call (a tool with an effect that must not be repeated)."""
        counter[label] = counter.get(label, 0) + 1
        if slow:
            await asyncio.sleep(30)
        return f"counted {label}"

    async def edit(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        outcome = await run_workspace_role(
            role=Role.TASK_DRAFTER,
            runtime=runtime,
            root="work",
            prompt="Write the note and count once.",
            tools=[count_me],
            timeout=0.5 if slow else None,
        )
        note = None
        if await runtime.context.workspace.exists("work/note.txt"):
            note = (await runtime.context.workspace.read_bytes("work/note.txt")).decode()
        update: dict[str, Any] = {"log": ["edit"], "timed_out": outcome.timed_out}
        if note is not None:
            update["note"] = note
        if outcome.failure:
            update["log"] = ["edit-failed:" + outcome.failure.category]
        return update

    async def judge(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        outcome = await run_structured_role(
            role=Role.DESIGN_JUDGE,
            schema=Verdict,
            runtime=runtime,
            root="work",
            prompt="Judge the note.",
        )
        assert outcome.structured is not None, outcome.failure
        return {"log": ["judge"], "verdict": outcome.structured.model_dump()}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(State, context_schema=ScientistContext)
    graph.add_node("edit", edit)
    graph.add_node("judge", judge)
    graph.add_edge(START, "edit")
    graph.add_edge("edit", "judge")
    graph.add_edge("judge", END)
    return graph


async def test_smoke_resume_mid_agent_without_repeating_tool_calls(make_context: Any) -> None:
    counter: dict[str, int] = {}
    models = ScriptedModels.from_scripts(
        {
            Role.TASK_DRAFTER: [
                tool_call("count_me", {"label": "once"}),
                tool_call("write_file", {"file_path": "/note.txt", "content": "hello"}),
                ConnectionError("provider dropped the connection"),
                final("note written"),
            ],
            Role.DESIGN_JUDGE: [structured_call(Verdict, {"verdict": "fine", "score": 3})],
        },
        context_window=200_000,
    )
    ctx = make_context(models=models)
    app = _two_node_graph(counter).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "smoke"}}
    with pytest.raises(ConnectionError):
        await app.ainvoke({"log": []}, config, context=ctx)
    assert counter == {"once": 1}
    drafter = models.scripted(Role.TASK_DRAFTER)
    assert drafter.call_count == 3
    out = await app.ainvoke(None, config, context=ctx)
    assert counter == {"once": 1}, "a completed tool call ran again on resume"
    assert drafter.call_count == 4
    assert out["note"] == "hello"
    assert out["verdict"] == {"verdict": "fine", "score": 3}
    assert out["log"] == ["edit", "judge"]


async def test_wall_clock_is_caught_and_graph_continues(make_context: Any) -> None:
    counter: dict[str, int] = {}
    models = ScriptedModels.from_scripts(
        {
            Role.TASK_DRAFTER: [tool_call("count_me", {"label": "slow"}), final()],
            Role.DESIGN_JUDGE: [structured_call(Verdict, {"verdict": "late", "score": 1})],
        }
    )
    ctx = make_context(models=models)
    app = _two_node_graph(counter, slow=True).compile(checkpointer=InMemorySaver())
    out = await app.ainvoke({"log": []}, {"configurable": {"thread_id": "clock"}}, context=ctx)
    assert out["timed_out"] is True
    assert out["log"] == ["edit", "judge"]


def test_execute_timeouts_default_and_maximum(make_context: Any) -> None:
    ctx = make_context()
    backend = ctx.workspace.agent_backend("work", execute_timeout=600, execute_max_timeout=7200)
    shell = backend.default  # type: ignore[attr-defined]
    assert isinstance(shell, SandboxBackendProtocol)
    assert shell._default_timeout == 600
    refused = backend.execute("echo hi", timeout=7201)  # type: ignore[attr-defined]
    assert refused.exit_code == 1 and "exceeds maximum" in refused.output
    accepted = backend.execute("echo hi", timeout=5)  # type: ignore[attr-defined]
    assert accepted.exit_code == 0 and "hi" in accepted.output
    factory = DeepAgentFactory()
    middleware = factory.middleware(
        ScientistContext(models=ctx.models, workspace=ctx.workspace), backend
    )
    filesystem = middleware[0]
    assert filesystem._max_execute_timeout == 7200


def test_model_retry_middleware_has_upstream_schedule(make_context: Any) -> None:
    ctx = ScientistContext(models=make_context().models, workspace=make_context().workspace)
    middleware = DeepAgentFactory().middleware(ctx, ctx.workspace.agent_backend("w"))
    retry = next(item for item in middleware if isinstance(item, ModelRetryMiddleware))
    assert (retry.max_retries, retry.initial_delay, retry.backoff_factor, retry.max_delay) == (
        3,
        15.0,
        3.0,
        90.0,
    )
    delays = [min(retry.initial_delay * retry.backoff_factor**n, retry.max_delay) for n in range(3)]
    assert delays == [15.0, 45.0, 90.0]
    assert retry.on_failure == "error"


async def test_transient_failure_is_retried_inside_the_session(make_context: Any) -> None:
    from langgraph_agentic_scientist.config import AgentsConfig, ModelRetryConfig

    config = ScientistConfig(
        agents=AgentsConfig(
            model_retry=ModelRetryConfig(max_retries=2, initial_delay=0, max_delay=0)
        )
    )
    models = ScriptedModels.from_scripts(
        {
            Role.DESIGN_JUDGE: [
                ConnectionError("blip"),
                structured_call(Verdict, {"verdict": "ok", "score": 2}),
            ]
        }
    )
    ctx = make_context(models=models, config=config)

    class S(TypedDict, total=False):
        verdict: dict[str, Any]

    async def node(state: S, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        outcome = await run_structured_role(
            role=Role.DESIGN_JUDGE, schema=Verdict, runtime=runtime, prompt="judge"
        )
        assert outcome.structured is not None
        return {"verdict": outcome.structured.model_dump()}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(S, context_schema=ScientistContext)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    out = await graph.compile().ainvoke({}, context=ctx)
    assert out["verdict"] == {"verdict": "ok", "score": 2}


@pytest.mark.parametrize("method", ["function_calling", "json_schema"])
async def test_structured_strategy_follows_the_registry(make_context: Any, method: str) -> None:
    answer = (
        structured_call(Verdict, {"verdict": "yes", "score": 4})
        if method == "function_calling"
        else json_answer({"verdict": "yes", "score": 4})
    )
    models = ScriptedModels.from_scripts(
        {Role.DESIGN_JUDGE: [answer]}, structured_output_method=method
    )
    ctx = make_context(models=models)
    strategy = response_format_for(Verdict, models.method_for(Role.DESIGN_JUDGE))
    expected = ToolStrategy if method == "function_calling" else ProviderStrategy
    assert isinstance(strategy, expected)

    class S(TypedDict, total=False):
        verdict: dict[str, Any]

    async def node(state: S, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        outcome = await run_structured_role(
            role=Role.DESIGN_JUDGE, schema=Verdict, runtime=runtime, root="judge", prompt="judge"
        )
        assert outcome.structured is not None, outcome.failure
        return {"verdict": outcome.structured.model_dump()}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(S, context_schema=ScientistContext)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    out = await graph.compile().ainvoke({}, context=ctx)
    assert out["verdict"] == {"verdict": "yes", "score": 4}


def test_context_window_is_applied_to_a_copy() -> None:
    model = ScriptedChatModel(script=[])
    assert model.profile is None or "max_input_tokens" not in (model.profile or {})
    copy = with_context_window(model, 200_000)
    assert copy is not model
    assert copy.profile is not None and copy.profile["max_input_tokens"] == 200_000
    assert model.profile is None or "max_input_tokens" not in (model.profile or {})


def test_factory_warns_once_per_role_without_a_window(make_context: Any) -> None:
    models = ScriptedModels.from_scripts({Role.IDEATOR: []})
    ctx = make_context(models=models)
    factory = DeepAgentFactory()
    with pytest.warns(UserWarning, match="max_input_tokens"):
        factory.model_for(Role.IDEATOR, ctx)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        factory.model_for(Role.IDEATOR, ctx)


def test_model_call_limit_is_optional(make_context: Any) -> None:
    from langchain.agents.middleware import ModelCallLimitMiddleware

    from langgraph_agentic_scientist.agents.runner import agent_node_retry_policy
    from langgraph_agentic_scientist.config import AgentsConfig

    base = make_context()
    backend = base.workspace.agent_backend("w")
    factory = DeepAgentFactory()
    default = ScientistContext(models=base.models, workspace=base.workspace)
    assert not any(
        isinstance(m, ModelCallLimitMiddleware) for m in factory.middleware(default, backend)
    )
    limited = ScientistContext(
        models=base.models,
        workspace=base.workspace,
        config=ScientistConfig(agents=AgentsConfig(model_call_limit=40)),
    )
    (limit,) = [
        m for m in factory.middleware(limited, backend) if isinstance(m, ModelCallLimitMiddleware)
    ]
    assert limit.run_limit == 40 and limit.exit_behavior == "end"
    policy = agent_node_retry_policy()
    assert (policy.max_attempts, policy.initial_interval, policy.max_interval) == (4, 15.0, 90.0)
    assert policy.retry_on(ConnectionError()) and not policy.retry_on(ValueError())


async def test_skills_already_staged_are_not_staged_again(make_context: Any) -> None:
    models = ScriptedModels.from_scripts(
        {Role.TASK_DRAFTER: [final("nothing to do")]}, context_window=200_000
    )
    ctx = make_context(models=models)
    skill = "work/.agents/skills/memory-log/SKILL.md"
    await ctx.workspace.write_bytes(skill, b"changed by the caller after staging\n")

    async def session(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        await run_workspace_role(
            role=Role.TASK_DRAFTER,
            runtime=runtime,
            root="work",
            prompt="Do nothing.",
            skills=["memory-log"],
            stage=False,
        )
        return {"log": ["session"]}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(State, context_schema=ScientistContext)
    graph.add_node("session", session)
    graph.add_edge(START, "session")
    graph.add_edge("session", END)
    await graph.compile().ainvoke({"log": []}, context=ctx)
    assert (await ctx.workspace.read_bytes(skill)) == b"changed by the caller after staging\n"
