"""A read-only role changes no file through any tool; input hashes catch changes behind its back."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from langgraph_agentic_scientist import LocalWorkspace, ScientistContext
from langgraph_agentic_scientist.agents.runner import run_structured_role
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import (
    MemoryWorkspace,
    ScriptedModels,
    structured_call,
    tool_call,
)


class Audit(BaseModel):
    """The auditor's answer."""

    verdict: str


def _attempts() -> list[Any]:
    return [
        tool_call("write_file", {"file_path": "/new.txt", "content": "x"}),
        tool_call(
            "edit_file", {"file_path": "/paper.tex", "old_string": "claim", "new_string": "edited"}
        ),
        tool_call("delete", {"file_path": "/paper.tex"}),
        tool_call("execute", {"command": "touch x && echo pwned > paper.tex"}),
        tool_call("read_file", {"file_path": "/paper.tex"}),
        structured_call(Audit, {"verdict": "PASS"}),
    ]


def _graph(outcomes: list[Any]) -> Any:
    class S(TypedDict, total=False):
        verdict: str

    async def audit(state: S, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        outcome = await run_structured_role(
            role=Role.EVIDENCE_AUDITOR, schema=Audit, runtime=runtime, root="paper", prompt="audit"
        )
        outcomes.append(outcome)
        assert outcome.structured is not None, outcome.failure
        return {"verdict": outcome.structured.verdict}  # type: ignore[attr-defined]

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(S, context_schema=ScientistContext)
    graph.add_node("audit", audit)
    graph.add_edge(START, "audit")
    return graph.compile()


async def test_read_only_role_changes_no_file(workspace: LocalWorkspace, make_context: Any) -> None:
    await workspace.write_bytes("paper/paper.tex", b"claim\n")
    before = await workspace.digest("paper")
    outcomes: list[Any] = []
    models = ScriptedModels.from_scripts({Role.EVIDENCE_AUDITOR: _attempts()})
    out = await _graph(outcomes).ainvoke({}, context=make_context(models=models))
    assert out["verdict"] == "PASS"
    assert await workspace.digest("paper") == before
    assert not await workspace.exists("paper/x") and not await workspace.exists("paper/new.txt")
    tool_messages = {m.name: str(m.content) for m in outcomes[0].messages if m.type == "tool"}
    assert "permission denied" in tool_messages["write_file"]
    assert "permission denied" in tool_messages["edit_file"]
    assert "permission denied" in tool_messages["delete"]
    assert "Execution not available" in tool_messages["execute"]
    assert "claim" in tool_messages["read_file"]


async def test_read_only_role_on_a_memory_workspace(make_context: Any) -> None:
    memory = MemoryWorkspace({"paper/paper.tex": "claim\n"}, execute=lambda command, timeout: None)  # type: ignore[arg-type,return-value]
    snapshot = dict(memory.files)
    models = ScriptedModels.from_scripts({Role.EVIDENCE_AUDITOR: _attempts()})
    await _graph([]).ainvoke({}, context=make_context(models=models, workspace=memory))
    assert memory.files == snapshot


async def test_input_hash_check_detects_a_change_behind_the_auditors_back(
    workspace: LocalWorkspace,
) -> None:
    await workspace.write_bytes("paper/paper.tex", b"claim\n")
    await workspace.write_bytes("paper/claims_grounding.json", b"{}")
    before = {
        path: await workspace.digest(f"paper/{path}")
        for path in ("paper.tex", "claims_grounding.json")
    }
    await workspace.write_bytes("paper/claims_grounding.json", b'{"changed": true}')
    after = {path: await workspace.digest(f"paper/{path}") for path in before}
    assert [path for path in before if before[path] != after[path]] == ["claims_grounding.json"]
