"""DeferredEvaluator: evaluations handed to the caller through interrupts, resumed with results."""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, Send

from langgraph_agentic_scientist import ScientistContext
from langgraph_agentic_scientist.core.state import ProgramRef, TaskRef
from langgraph_agentic_scientist.evaluation import (
    DeferredEvaluator,
    EvaluationRequest,
    EvaluationResult,
)


class State(TypedDict, total=False):
    programs: list[str]
    results: Annotated[list[dict[str, Any]], operator.add]


def _graph(completions: list[str]) -> Any:
    async def evaluate(state: dict[str, Any], runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        program = state["program"]
        evaluator = runtime.context.evaluator
        assert evaluator is not None
        result = await evaluator.evaluate(
            EvaluationRequest(
                evaluation_id=f"run/{program}",
                program=ProgramRef(path=program),
                task=TaskRef(root="task"),
            )
        )
        completions.append(program)
        return {"results": [result.model_dump(mode="json")]}

    def fan_out(state: State) -> list[Send]:
        return [Send("evaluate", {"program": program}) for program in state["programs"]]

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(State, context_schema=ScientistContext)
    graph.add_node("evaluate", evaluate)
    graph.add_conditional_edges(START, fan_out, ["evaluate"])
    return graph.compile(checkpointer=InMemorySaver())


def _result(request: dict[str, Any], score: float) -> dict[str, Any]:
    return EvaluationResult(
        evaluation_id=request["evaluation_id"],
        status="completed",
        metrics={"combined_score": score},
        correct=True,
    ).model_dump(mode="json")


async def test_single_evaluation_interrupts_and_resumes(make_context: Any) -> None:
    completions: list[str] = []
    app = _graph(completions)
    ctx = make_context(evaluator=DeferredEvaluator())
    config = {"configurable": {"thread_id": "one"}}
    out = await app.ainvoke({"programs": ["p1"]}, config, context=ctx)
    (pending,) = out["__interrupt__"]
    payload = pending.value
    json.dumps(payload)  # JSON-serialisable
    assert payload["kind"] == "evaluation_request"
    assert payload["request"]["evaluation_id"] == "run/p1"
    assert completions == []
    final = await app.ainvoke(Command(resume=_result(payload["request"], 2.0)), config, context=ctx)
    assert final["results"][0]["metrics"] == {"combined_score": 2.0}
    assert completions == ["p1"], "the evaluation node completed exactly once"


async def test_fan_out_resumes_by_interrupt_id(make_context: Any) -> None:
    completions: list[str] = []
    app = _graph(completions)
    ctx = make_context(evaluator=DeferredEvaluator())
    config = {"configurable": {"thread_id": "two"}}
    out = await app.ainvoke({"programs": ["p1", "p2"]}, config, context=ctx)
    pending = out["__interrupt__"]
    assert len(pending) == 2
    answers = {
        item.id: _result(
            item.value["request"],
            1.0 if item.value["request"]["evaluation_id"] == "run/p1" else 3.0,
        )
        for item in pending
    }
    final = await app.ainvoke(Command(resume=answers), config, context=ctx)
    scores = sorted(result["metrics"]["combined_score"] for result in final["results"])
    assert scores == [1.0, 3.0]
    assert sorted(completions) == ["p1", "p2"]
