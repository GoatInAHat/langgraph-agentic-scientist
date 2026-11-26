"""The scripted model drives Deep Agents and structured output by both strategies, offline."""

from __future__ import annotations

from pathlib import Path

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from pydantic import BaseModel

from langgraph_agentic_scientist.agents.factory import response_format_for, structured_output
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import (
    ScriptedChatModel,
    ScriptedModels,
    ScriptExhausted,
    final,
    json_answer,
    structured_call,
    tool_call,
)


class Plan(BaseModel):
    """A structured answer."""

    steps: list[str]


async def test_scripted_model_drives_a_deep_agent(tmp_path: Path) -> None:
    model = ScriptedChatModel(
        script=[
            tool_call("write_file", {"file_path": "/notes/a.txt", "content": "hello"}),
            final("done"),
        ]
    )
    agent = create_deep_agent(
        model=model, tools=[], backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    )
    result = await agent.ainvoke({"messages": [("user", "write a note")]})
    assert (tmp_path / "notes" / "a.txt").read_text() == "hello"
    assert result["messages"][-1].content == "done"
    assert model.call_count == 2
    assert "write_file" in model.bound_tools[0]


@pytest.mark.parametrize("method", ["function_calling", "json_schema"])
async def test_structured_output_by_both_strategies(tmp_path: Path, method: str) -> None:
    answer = (
        structured_call(Plan, {"steps": ["a"]})
        if method == "function_calling"
        else json_answer({"steps": ["a"]})
    )
    model = ScriptedChatModel(script=[answer])
    agent = create_deep_agent(
        model=model,
        tools=[],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        response_format=response_format_for(Plan, method),  # type: ignore[arg-type]
    )
    result = await agent.ainvoke({"messages": [("user", "plan")]})
    assert result["structured_response"] == Plan(steps=["a"])


async def test_plain_structured_call() -> None:
    model = ScriptedChatModel(script=[structured_call(Plan, {"steps": ["x", "y"]})])
    runnable = structured_output(model, Plan, "function_calling")
    assert await runnable.ainvoke("plan") == Plan(steps=["x", "y"])


def test_exhausted_script_names_the_role() -> None:
    models = ScriptedModels.from_scripts({Role.IDEATOR: []})
    with pytest.raises(ScriptExhausted, match="role ideator"):
        models.scripted(Role.IDEATOR).invoke("hello")
    assert models.scripted(Role.SEARCH_AGENT) is models.default


def test_copies_share_the_script() -> None:
    model = ScriptedChatModel(script=[final("one"), final("two")])
    copy = model.model_copy(update={"profile": {"max_input_tokens": 1000}})
    assert model.invoke("a").content == "one"
    assert copy.invoke("b").content == "two"
    assert model.call_count == 2
