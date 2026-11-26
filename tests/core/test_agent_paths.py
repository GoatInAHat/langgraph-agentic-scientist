"""An agent's file tools and its shell agree on paths, and the agent is told its directory."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from langgraph_agentic_scientist import ScientistContext
from langgraph_agentic_scientist.agents.runner import RoleOutcome, run_workspace_role
from langgraph_agentic_scientist.agents.tools import make_file_sha256_tool, make_nl_tool
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.prompts import PromptRegistry
from langgraph_agentic_scientist.testing import final, tool_call
from langgraph_agentic_scientist.workspace import LocalWorkspace
from langgraph_agentic_scientist.workspace.protocol import agent_path


class _State(TypedDict, total=False):
    done: bool


async def _run(ctx: ScientistContext, root: str) -> RoleOutcome:
    holder: dict[str, RoleOutcome] = {}

    async def node(state: _State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        holder["outcome"] = await run_workspace_role(
            role=Role.TASK_DRAFTER, runtime=runtime, root=root, prompt="Write the program."
        )
        return {"done": True}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(_State, context_schema=ScientistContext)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    app = graph.compile(checkpointer=InMemorySaver())
    await app.ainvoke({}, {"configurable": {"thread_id": "paths"}}, context=ctx)
    return holder["outcome"]


async def test_a_file_written_at_the_shells_path_is_what_the_shell_runs(
    make_context: Any, workspace: LocalWorkspace
) -> None:
    directory = agent_path(workspace, "ideas/one")
    seen: dict[str, str] = {}

    def write(messages: list[BaseMessage]) -> Any:
        system = next(m for m in messages if isinstance(m, SystemMessage))
        seen["system"] = str(system.content)
        return tool_call(
            "write_file",
            {"file_path": f"{directory}/program_1/src/program.py", "content": "print(41 + 1)\n"},
        )

    ctx = make_context(
        {
            Role.TASK_DRAFTER: [
                write,
                tool_call("execute", {"command": "python3 program_1/src/program.py"}),
                final("done"),
            ]
        }
    )
    outcome = await _run(ctx, "ideas/one")
    results = [m for m in outcome.messages if isinstance(m, ToolMessage)]
    assert "42" in str(results[-1].content)
    assert await workspace.read_bytes("ideas/one/program_1/src/program.py") == b"print(41 + 1)\n"
    assert f"`{directory}`" in seen["system"]
    assert directory == str(workspace.root / "ideas" / "one")


async def test_package_tools_take_the_paths_the_file_tools_use(workspace: LocalWorkspace) -> None:
    prompts = PromptRegistry.default()
    await workspace.write_bytes("role/notes.txt", b"one\ntwo\n")
    writable = make_nl_tool(workspace, "role", prompts=prompts)
    assert "two" in await writable.ainvoke({"path": "/notes.txt"})
    digest = make_file_sha256_tool(workspace, "role", prompts=prompts)
    assert "notes.txt" in await digest.ainvoke({"path": "notes.txt"})
    host = agent_path(workspace, "role", read_only=True)
    assert "one" in await writable.ainvoke({"path": f"{host}/notes.txt"})


async def test_a_host_path_elsewhere_in_the_workspace_is_refused(
    make_context: Any, workspace: LocalWorkspace
) -> None:
    outside = f"{workspace.root}/elsewhere/program.py"
    ctx = make_context(
        {
            Role.TASK_DRAFTER: [
                tool_call("write_file", {"file_path": outside, "content": "x\n"}),
                tool_call("write_file", {"file_path": "/inside.py", "content": "y\n"}),
                final("done"),
            ]
        }
    )
    outcome = await _run(ctx, "ideas/two")
    results = [str(m.content) for m in outcome.messages if isinstance(m, ToolMessage)]
    assert "outside the working directory" in results[0], (outcome.failure, outcome.messages)
    assert not await workspace.exists("elsewhere/program.py")
    assert not await workspace.exists("ideas/two/" + outside.lstrip("/"))
    assert await workspace.read_bytes("ideas/two/inside.py") == b"y\n"
