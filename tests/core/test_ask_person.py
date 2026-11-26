"""ask_person: the policy answers, or the run interrupts; the node's effect happens once."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from langgraph_agentic_scientist import AutoApprovePolicy, ScientistContext
from langgraph_agentic_scientist.core.state import Question
from langgraph_agentic_scientist.core.steps import ask_person

QUESTION = Question(kind="prune_confirmation", text="Pruning may be ineffective. Continue?")


class State(TypedDict, total=False):
    choice: str


def _graph(effects: list[str]) -> Any:
    async def confirm(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        answer = await ask_person(QUESTION, runtime)
        effects.append(answer.choice)
        return {"choice": answer.choice}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(State, context_schema=ScientistContext)
    graph.add_node("confirm", confirm)
    graph.add_edge(START, "confirm")
    return graph.compile(checkpointer=InMemorySaver())


async def test_policy_answers_without_interrupt(make_context: Any) -> None:
    effects: list[str] = []
    out = await _graph(effects).ainvoke(
        {},
        {"configurable": {"thread_id": "auto"}},
        context=make_context(policy=AutoApprovePolicy()),
    )
    assert out == {"choice": "proceed"}
    assert effects == ["proceed"]


async def test_interrupt_policy_asks_the_person(make_context: Any) -> None:
    effects: list[str] = []
    app = _graph(effects)
    config = {"configurable": {"thread_id": "ask"}}
    out = await app.ainvoke({}, config, context=make_context())
    (pending,) = out["__interrupt__"]
    assert pending.value["kind"] == "prune_confirmation"
    assert pending.value["options"] == ["proceed", "abort"]
    assert effects == []
    out = await app.ainvoke(Command(resume="abort"), config, context=make_context())
    assert out == {"choice": "abort"}
    assert effects == ["abort"], "the effect after ask_person ran once"


async def test_an_answer_outside_the_options_is_refused(make_context: Any) -> None:
    app = _graph([])
    config = {"configurable": {"thread_id": "bad"}}
    await app.ainvoke({}, config, context=make_context())
    with pytest.raises(ValueError, match="not one of"):
        await app.ainvoke(Command(resume={"choice": "maybe"}), config, context=make_context())
