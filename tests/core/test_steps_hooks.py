"""StepGraph: hook nodes around steps, run once across interrupts; slots replaceable."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from langgraph_agentic_scientist import GraphOverrides, Hooks, ScientistContext
from langgraph_agentic_scientist.core.state import merge_dicts
from langgraph_agentic_scientist.core.steps import HookError, StepGraph


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    note: str
    answer: str


def _graph(*, pause: bool = False, overrides: GraphOverrides | None = None) -> Any:
    sg = StepGraph("demo", State, overrides=overrides)

    async def first(state: State) -> dict[str, Any]:
        return {"log": ["first"]}

    async def second(state: State) -> dict[str, Any]:
        answer = interrupt("go on?") if pause else "auto"
        return {"log": ["second"], "answer": answer}

    async def data_default(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        return {"log": ["data:default"]}

    sg.step("first", first)
    sg.step("second", second, hook_writable=("note",))
    sg.slot("data", data_default)
    sg.add_edge(START, "first")
    sg.add_edge("first", "second")
    sg.add_edge("second", "data")
    sg.add_edge("data", END)
    return sg.compile(checkpointer=InMemorySaver())


def test_hook_nodes_exist_at_the_documented_names() -> None:
    app = _graph()
    nodes = set(app.get_graph().nodes)
    for step in ("first", "second", "data"):
        assert {step, f"demo.{step}.before", f"demo.{step}.after"} <= nodes


async def test_hooks_run_once_across_an_interrupt(make_context: Any) -> None:
    calls: list[str] = []

    async def before_second(state: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
        calls.append("before")
        return {"note": "set by hook", "hook_outputs": {"seen": len(state.get("log", []))}}

    async def after_second(state: Mapping[str, Any], runtime: Any) -> None:
        calls.append("after")

    ctx = make_context(
        hooks=Hooks(before={"demo.second": [before_second]}, after={"demo.second": [after_second]})
    )
    app = _graph(pause=True)
    config = {"configurable": {"thread_id": "h"}}
    out = await app.ainvoke({"log": []}, config, context=ctx)
    assert "__interrupt__" in out
    assert calls == ["before"]
    out = await app.ainvoke(Command(resume="yes"), config, context=ctx)
    assert calls == ["before", "after"], "the completed before hook ran again"
    assert out["note"] == "set by hook"
    assert out["answer"] == "yes"
    assert out["hook_outputs"] == {"demo.second": {"seen": 1}}
    assert out["log"] == ["first", "second", "data:default"]


async def test_hooks_may_update_declared_keys_only(make_context: Any) -> None:
    async def bad(state: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
        return {"answer": "not allowed"}

    ctx = make_context(hooks=Hooks(before={"demo.second": [bad]}))
    with pytest.raises(HookError, match="may not update 'answer'"):
        await _graph().ainvoke({"log": []}, {"configurable": {"thread_id": "b"}}, context=ctx)


async def test_a_raising_hook_fails_the_step(make_context: Any) -> None:
    async def boom(state: Mapping[str, Any], runtime: Any) -> None:
        raise RuntimeError("hook failed")

    ctx = make_context(hooks=Hooks(after={"demo.first": [boom]}))
    with pytest.raises(RuntimeError, match="hook failed"):
        await _graph().ainvoke({"log": []}, {"configurable": {"thread_id": "r"}}, context=ctx)


async def test_without_hooks_the_hook_nodes_return_nothing(make_context: Any) -> None:
    updates = [
        chunk
        async for chunk in _graph().astream(
            {"log": []},
            {"configurable": {"thread_id": "n"}},
            context=make_context(),
            stream_mode="updates",
        )
    ]
    hook_updates = [
        value
        for chunk in updates
        for name, value in chunk.items()
        if name.endswith((".before", ".after"))
    ]
    assert hook_updates and all(not value for value in hook_updates)


async def test_slots_take_compiled_graphs_and_callables(make_context: Any) -> None:
    async def replacement(
        value: Mapping[str, Any], runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        return {"log": [f"data:callable:{len(value['log'])}"]}

    out = await _graph(overrides=GraphOverrides(slots={"demo.data": replacement})).ainvoke(
        {"log": []}, {"configurable": {"thread_id": "s1"}}, context=make_context()
    )
    assert out["log"][-1] == "data:callable:2"

    inner: StateGraph[Any, Any, Any, Any] = StateGraph(State)
    inner.add_node("work", lambda state: {"log": ["data:graph"]})
    inner.add_edge(START, "work")
    compiled = inner.compile()
    out = await _graph(overrides=GraphOverrides(slots={"demo.data": compiled})).ainvoke(
        {"log": []}, {"configurable": {"thread_id": "s2"}}, context=make_context()
    )
    assert out["log"][-1] == "data:graph"


async def test_conditional_edges_and_sends_route_to_entries(make_context: Any) -> None:
    sg = StepGraph("route", State)

    async def start(state: State) -> dict[str, Any]:
        return {"log": ["start"]}

    async def left(state: State) -> dict[str, Any]:
        return {"log": ["left"]}

    sg.step("start", start)
    sg.step("left", left)
    sg.add_edge(START, "start")
    sg.add_conditional_edges("start", lambda state: "left", ["left"])
    sg.add_edge("left", END)
    calls: list[str] = []

    async def hook(state: Mapping[str, Any], runtime: Any) -> None:
        calls.append("left.before")

    ctx = make_context(hooks=Hooks(before={"route.left": [hook]}))
    out = await sg.compile().ainvoke({"log": []}, context=ctx)
    assert out["log"] == ["start", "left"]
    assert calls == ["left.before"]


class _RouterState(TypedDict, total=False):
    """A state type local to this module (routers annotated with it once failed to compile)."""

    log: Annotated[list[str], operator.add]


def _route_with_a_local_annotation(state: _RouterState) -> list[Any]:
    from langgraph.types import Send

    return [Send("leaf", {"log": ["sent"]})]


async def test_a_router_annotated_with_a_module_local_type_compiles_and_routes(
    make_context: Any,
) -> None:
    sg = StepGraph("local", _RouterState)

    async def root(state: _RouterState) -> dict[str, Any]:
        return {"log": ["root"]}

    async def leaf(state: _RouterState) -> dict[str, Any]:
        return {"log": ["leaf"]}

    sg.step("root", root)
    sg.step("leaf", leaf, hooks=False)
    sg.add_edge(START, "root")
    sg.add_conditional_edges("root", _route_with_a_local_annotation)
    sg.add_edge("leaf", END)
    app = sg.compile()
    out = await app.ainvoke({"log": []}, context=make_context())
    assert out["log"][0] == "root" and out["log"][-1] == "leaf"
