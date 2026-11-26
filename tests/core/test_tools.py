"""Tools replacing upstream's trust-boundary helpers, the blocker report and read-only reads."""

from __future__ import annotations

import hashlib
import json
from typing import Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime

from langgraph_agentic_scientist import LocalWorkspace, ScientistContext
from langgraph_agentic_scientist.agents.runner import run_workspace_role
from langgraph_agentic_scientist.agents.tools import (
    EvaluateProgramSettings,
    make_clone_program_tool,
    make_evaluate_program_tool,
    make_file_sha256_tool,
    make_nl_tool,
    make_report_blocker_tool,
    make_task_gate_tool,
)
from langgraph_agentic_scientist.core.interfaces import GateReport
from langgraph_agentic_scientist.evaluation import EvaluationEvidence, EvaluationResult
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.prompts import PromptRegistry
from langgraph_agentic_scientist.testing import (
    FakeEvaluator,
    FakeTaskGate,
    ScriptedModels,
    final,
    minimal_task_ref,
    tool_call,
    write_program,
)

PROMPTS = PromptRegistry.default()


async def _seed(workspace: LocalWorkspace) -> None:
    await write_program(workspace, "attempt/program_0", "SCORE = 1.0\n")
    await workspace.write_bytes("attempt/program_0/__pycache__/x.pyc", b"x")
    await workspace.write_bytes("attempt/program_0/metrics.json", b"{}")


async def _complete_iteration(
    workspace: LocalWorkspace, index: int, *, overall_pass: bool = False
) -> None:
    program = f"attempt/program_{index}"
    await workspace.write_bytes(f"{program}/metrics.json", b'{"combined_score": 2.0}')
    await workspace.write_bytes(f"{program}/correct.json", b'{"correct": true}')
    manifest = {
        "producer": "isolated_evaluator",
        "schema_version": 1,
        "program_id": f"program_{index}",
        "output_dir": "../.evaluation_outputs",
        "returncode": 0,
        "published_summaries": ["metrics.json", "correct.json"],
    }
    await workspace.write_bytes(
        f"{program}/evaluation_manifest.json", json.dumps(manifest).encode()
    )
    rubric = {
        "attempt_id": "a1",
        "idea_id": "i1",
        "overall_pass": overall_pass,
        "rubrics": [{"id": "r", "pass": overall_pass, "evidence": "e"}],
    }
    data = json.dumps(rubric).encode()
    await workspace.write_bytes(f"{program}/rubric_status.json", data)
    await workspace.write_bytes("attempt/rubric_status.json", data)


async def test_clone_program_rules(workspace: LocalWorkspace) -> None:
    await _seed(workspace)
    clone = make_clone_program_tool(workspace, "attempt", prompts=PROMPTS)
    result = await clone.ainvoke({"from_dir": "./program_0", "to_dir": "./program_1"})
    assert result.startswith("cloned ./program_0 -> ./program_1"), result
    copied = {entry.path for entry in await workspace.list_tree("attempt/program_1")}
    assert "src/program.py" in copied and "program_spec.toml" in copied
    assert not any("__pycache__" in path or path == "metrics.json" for path in copied)
    refused = await clone.ainvoke({"from_dir": "program_1", "to_dir": "program_2"})
    assert "missing required artifact: metrics.json" in refused
    assert "adjacent" in await clone.ainvoke({"from_dir": "program_0", "to_dir": "program_3"})
    await _complete_iteration(workspace, 1)
    assert (await clone.ainvoke({"from_dir": "program_1", "to_dir": "program_2"})).startswith(
        "cloned"
    )
    await _complete_iteration(workspace, 2, overall_pass=True)
    assert "already passes all rubrics" in await clone.ainvoke(
        {"from_dir": "program_2", "to_dir": "program_3"}
    )
    await _complete_iteration(workspace, 2)
    await workspace.write_bytes("attempt/rubric_status.json", b'{"other": 1}')
    assert "root rubric_status.json does not match" in await clone.ainvoke(
        {"from_dir": "program_2", "to_dir": "program_3"}
    )
    assert "refused" in await clone.ainvoke(
        {"from_dir": "../escape/program_0", "to_dir": "program_9"}
    )


def _evaluate_session(
    budget: int, results: dict[str, EvaluationResult] | None = None
) -> tuple[Any, FakeEvaluator, list[Any]]:
    evaluator = FakeEvaluator(results or {})
    outcomes: list[Any] = []

    class S(TypedDict, total=False):
        done: bool

    async def node(state: S, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        tool = make_evaluate_program_tool(
            runtime.context.workspace,
            evaluator,
            "attempt",
            EvaluateProgramSettings(
                task=minimal_task_ref("task"), id_prefix="run/w0/c001", budget=budget
            ),
            prompts=runtime.context.prompts,
        )
        outcome = await run_workspace_role(
            role=Role.SEARCH_AGENT, runtime=runtime, root="attempt", prompt="evaluate", tools=[tool]
        )
        outcomes.append(outcome)
        return {"done": True}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(S, context_schema=ScientistContext)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    return graph.compile(checkpointer=InMemorySaver()), evaluator, outcomes


async def test_evaluate_program_budget_counts_every_call(
    workspace: LocalWorkspace, make_context: Any
) -> None:
    await _seed(workspace)
    await workspace.remove("attempt/program_0/metrics.json")
    await write_program(workspace, "attempt/program_1", "SCORE = 2.0\n")
    failed = EvaluationResult(evaluation_id="x", status="failed", summary="boom")
    interrupted = EvaluationResult(evaluation_id="x", status="interrupted", summary="lost")
    app, evaluator, outcomes = _evaluate_session(
        3, {"run/w0/c001/eval_01": failed, "run/w0/c001/eval_02": interrupted}
    )
    models = ScriptedModels.from_scripts(
        {
            Role.SEARCH_AGENT: [
                tool_call("evaluate_program", {"program_dir": "./program_1"}),
                tool_call("evaluate_program", {"program_dir": "./program_1"}),
                tool_call("evaluate_program", {"program_dir": "./program_0"}),
                tool_call("evaluate_program", {"program_dir": "./program_0"}),
                final(),
            ]
        }
    )
    await app.ainvoke(
        {}, {"configurable": {"thread_id": "budget"}}, context=make_context(models=models)
    )
    ids = [request.evaluation_id for request in evaluator.calls]
    assert ids == ["run/w0/c001/eval_01", "run/w0/c001/eval_02", "run/w0/c001/eval_03"]
    (outcome,) = outcomes
    assert [result.status for result in outcome.evaluations] == [
        "failed",
        "interrupted",
        "completed",
    ]
    tool_texts = [
        m.content for m in outcome.messages if getattr(m, "name", None) == "evaluate_program"
    ]
    assert "budget of this attempt is consumed" in tool_texts[-1]
    manifest = json.loads(await workspace.read_bytes("attempt/program_0/evaluation_manifest.json"))
    assert manifest["producer"] == "isolated_evaluator"
    assert manifest["evaluation_id"] == "run/w0/c001/eval_03"
    assert json.loads(await workspace.read_bytes("attempt/program_0/correct.json")) == {
        "correct": True
    }


async def test_evaluate_program_publishes_the_summaries_and_not_the_evidence(
    workspace: LocalWorkspace, make_context: Any
) -> None:
    await write_program(workspace, "attempt/program_1", "SCORE = 2.0\n")
    evidence = EvaluationEvidence.model_validate(
        {
            "schema_version": 1,
            "episodes": [{"seed": 0, "score": 2.0, "correct": True, "duration_seconds": 1.0}],
        }
    )
    completed = EvaluationResult(
        evaluation_id="run/w0/c001/eval_01",
        status="completed",
        metrics={"combined_score": 2.0},
        correct=True,
        evidence=evidence,
    )
    app, _, _ = _evaluate_session(1, {"run/w0/c001/eval_01": completed})
    models = ScriptedModels.from_scripts(
        {Role.SEARCH_AGENT: [tool_call("evaluate_program", {"program_dir": "program_1"}), final()]}
    )
    await app.ainvoke(
        {}, {"configurable": {"thread_id": "evidence"}}, context=make_context(models=models)
    )
    assert await workspace.exists("attempt/program_1/metrics.json")
    assert await workspace.exists("attempt/program_1/correct.json")
    assert not await workspace.exists("attempt/program_1/evaluation_evidence.json")


async def test_evaluate_program_budget_zero_refuses(
    workspace: LocalWorkspace, make_context: Any
) -> None:
    await write_program(workspace, "attempt/program_0", "SCORE = 1.0\n")
    app, evaluator, outcomes = _evaluate_session(0)
    models = ScriptedModels.from_scripts(
        {Role.SEARCH_AGENT: [tool_call("evaluate_program", {"program_dir": "program_0"}), final()]}
    )
    await app.ainvoke(
        {}, {"configurable": {"thread_id": "zero"}}, context=make_context(models=models)
    )
    assert evaluator.calls == []
    assert outcomes[0].evaluations == []


async def test_task_gate_is_bounded(workspace: LocalWorkspace) -> None:
    gate = FakeTaskGate(
        {
            "acceptance-smoke": [
                GateReport(check="acceptance-smoke", passed=False, diagnostic="smoke failed")
            ]
        }
    )
    tool = make_task_gate_tool(
        workspace,
        gate,
        minimal_task_ref(),
        id_prefix="run/draft/smoke_r1",
        max_checks=1,
        prompts=PROMPTS,
    )

    def call(call_id: str) -> dict[str, Any]:
        return {
            "type": "tool_call",
            "name": "task_gate",
            "id": call_id,
            "args": {"check": "acceptance-smoke"},
        }

    from langchain.tools import ToolRuntime
    from langchain_core.messages import AIMessage

    message = AIMessage(content="", tool_calls=[call("c1"), call("c2")])
    runtime1 = ToolRuntime(
        state={"messages": [message]},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="c1",
        store=None,
    )
    first = await tool.coroutine(check="acceptance-smoke", runtime=runtime1)  # type: ignore[attr-defined]
    assert "acceptance-smoke: failed" in first and "smoke failed" in first
    runtime2 = ToolRuntime(
        state={"messages": [message]},
        context=None,
        config={},
        stream_writer=lambda _: None,
        tool_call_id="c2",
        store=None,
    )
    second = await tool.coroutine(check="acceptance-smoke", runtime=runtime2)  # type: ignore[attr-defined]
    assert "request refused" in second
    assert gate.calls == [("acceptance-smoke", "run/draft/smoke_r1/gate_01")]
    assert "unknown check" in await tool.coroutine(check="rm -rf", runtime=runtime1)  # type: ignore[attr-defined]


async def test_report_blocker_ends_the_session(
    workspace: LocalWorkspace, make_context: Any
) -> None:
    class S(TypedDict, total=False):
        blocker: str

    async def node(state: S, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        outcome = await run_workspace_role(
            role=Role.TASK_DRAFTER,
            runtime=runtime,
            root="draft",
            prompt="draft",
            tools=[make_report_blocker_tool(prompts=runtime.context.prompts)],
        )
        return {"blocker": outcome.blocker}

    models = ScriptedModels.from_scripts(
        {Role.TASK_DRAFTER: [tool_call("report_blocker", {"text": "dataset unavailable"})]}
    )
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(S, context_schema=ScientistContext)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    out = await graph.compile().ainvoke({}, context=make_context(models=models))
    assert out["blocker"] == "dataset unavailable"
    assert models.scripted(Role.TASK_DRAFTER).remaining == 0


async def test_nl_and_file_sha256(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("judge/task.md", b"first\nsecond\n")
    nl = make_nl_tool(workspace, "judge", prompts=PROMPTS)
    assert await nl.ainvoke({"path": "task.md"}) == "     1\tfirst\n     2\tsecond"
    assert await nl.ainvoke({"path": "/task.md"}) == "     1\tfirst\n     2\tsecond"
    assert "No such file" in await nl.ainvoke(
        {"path": "missing.md"}
    ) or "nl: missing.md" in await nl.ainvoke({"path": "missing.md"})
    sha = make_file_sha256_tool(workspace, "judge", prompts=PROMPTS)
    expected = hashlib.sha256(b"first\nsecond\n").hexdigest()
    assert await sha.ainvoke({"path": "./task.md"}) == f"{expected}  ./task.md"
    assert "sha256sum: ../x" in await sha.ainvoke({"path": "../x"})


async def test_an_interrupted_evaluation_is_reattached_on_resume(
    workspace: LocalWorkspace, make_context: Any
) -> None:
    import pytest

    await write_program(workspace, "attempt/program_1", "SCORE = 2.0\n")
    app, evaluator, outcomes = _evaluate_session(2)
    evaluator.fail_once = {"run/w0/c001/eval_01"}
    evaluator.reattach = True
    models = ScriptedModels.from_scripts(
        {Role.SEARCH_AGENT: [tool_call("evaluate_program", {"program_dir": "program_1"}), final()]}
    )
    ctx = make_context(models=models)
    config = {"configurable": {"thread_id": "reattach"}}
    with pytest.raises(ConnectionError):
        await app.ainvoke({}, config, context=ctx)
    await app.ainvoke(None, config, context=ctx)
    assert [request.evaluation_id for request in evaluator.calls] == ["run/w0/c001/eval_01"] * 2
    assert evaluator.runs["run/w0/c001/eval_01"] == 1
    assert [result.status for result in outcomes[-1].evaluations] == ["completed"]
    assert models.scripted(Role.SEARCH_AGENT).call_count == 2
