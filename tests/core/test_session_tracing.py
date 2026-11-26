"""Agent sessions are marked in their run config, so tracing finds the root run of each session."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START
from langgraph.runtime import Runtime
from pydantic import BaseModel
from testkit import SessionRoots

from langgraph_agentic_scientist import ScientistContext
from langgraph_agentic_scientist.agents.runner import (
    run_structured_role,
    run_workspace_role,
    session_config,
)
from langgraph_agentic_scientist.core.steps import StepGraph
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import ScriptedModels, final, structured_call


class Verdict(BaseModel):
    """A small structured answer."""

    verdict: str


class State(TypedDict, total=False):
    log: Annotated[list[str], operator.add]


def _graph() -> Any:
    async def draft(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        await run_workspace_role(
            role=Role.TASK_DRAFTER, runtime=runtime, root="work", prompt="Write a note."
        )
        return {"log": ["draft"]}

    async def judge(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        await run_structured_role(
            role=Role.DESIGN_JUDGE, schema=Verdict, runtime=runtime, prompt="Judge it."
        )
        return {"log": ["judge"]}

    sg = StepGraph("demo", State)
    sg.step("draft", draft)
    sg.add_node("judge", judge)
    sg.add_edge(START, "draft")
    sg.add_edge("draft", "judge")
    sg.add_edge("judge", END)
    return sg.compile(checkpointer=InMemorySaver())


async def test_each_session_has_one_marked_root_run(
    make_context: Callable[..., ScientistContext],
) -> None:
    models = ScriptedModels.from_scripts(
        {
            Role.TASK_DRAFTER: [final("done")],
            Role.DESIGN_JUDGE: [structured_call(Verdict, {"verdict": "fine"})],
        },
        context_window=200_000,
    )
    handler = SessionRoots()
    await _graph().ainvoke(
        {"log": []},
        {"configurable": {"thread_id": "t"}, "callbacks": [handler]},
        context=make_context(models=models),
    )

    roots = [run for run in handler.runs if handler.is_root(run)]
    assert [run.name for run in roots] == ["agsci:task_drafter", "agsci:design_judge"]
    assert len(handler.runs) == 2
    drafter, judge = roots
    assert drafter.tags[-2:] == ["agsci", "task_drafter"]
    assert drafter.metadata["agsci_role"] == "task_drafter"
    assert drafter.metadata["agsci_phase"] == "demo"
    assert drafter.metadata["agsci_step"] == "draft"
    assert judge.tags[-2:] == ["agsci", "design_judge"]
    assert judge.metadata["agsci_role"] == "design_judge"
    assert judge.metadata["agsci_phase"] == "demo"
    assert judge.metadata["agsci_step"] == "judge"


def test_outside_a_step_the_phase_and_step_are_unknown() -> None:
    config = session_config(Role.IDEATOR)
    assert config["run_name"] == "agsci:ideator"
    assert config["tags"] == ["agsci", "ideator"]
    assert config["metadata"] == {
        "agsci_role": "ideator",
        "agsci_phase": None,
        "agsci_step": None,
    }
