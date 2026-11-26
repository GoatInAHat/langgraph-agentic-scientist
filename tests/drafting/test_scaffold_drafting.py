"""The drafting graph: the design loop, then the scaffold loop, with replaceable slots."""

from __future__ import annotations

import json
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from test_design_support import TASK_MD, make_design, review_payload
from test_design_support import judge as design_judge
from test_scaffold_support import (
    RECURSION,
    drafter_script,
    fast,
    judge_step,
    make_context,
    write_scaffold,
)

from langgraph_agentic_scientist import GraphOverrides, LocalWorkspace
from langgraph_agentic_scientist.core.results import DraftingResult
from langgraph_agentic_scientist.drafting import build_task_drafting_graph
from langgraph_agentic_scientist.drafting.design_models import DesignReview, TaskDesign
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import minimal_task_ref, structured_call

DESIGN = make_design()


async def invoke(graph: Any, context: Any, task: Any) -> dict[str, Any]:
    config = {"configurable": {"thread_id": "d"}, **RECURSION}
    return await graph.ainvoke({"task": task}, config, context=context)


async def test_the_real_design_loop_feeds_the_scaffold_loop(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await workspace.write_bytes("run/task/task.md", TASK_MD.encode())
    seen: dict[str, Any] = {}

    async def scaffold(state: Any, runtime: Any) -> dict[str, Any]:
        seen.update(state)
        return {"drafting": DraftingResult(status="skipped")}

    context, models = make_context(
        workspace,
        {
            Role.TASK_DESIGNER: [structured_call(TaskDesign, DESIGN)],
            Role.DESIGN_JUDGE: [design_judge()],
        },
    )
    graph = build_task_drafting_graph(
        checkpointer=InMemorySaver(),
        overrides=GraphOverrides(slots={"drafting.scaffold": scaffold}),
    )
    out = await invoke(graph, context, minimal_task_ref("run/task"))
    assert out["drafting"].status == "skipped"
    assert seen["accepted_design"] == DESIGN.payload()
    assert seen["accepted_review"]["verdict"] == "pass"
    written = json.loads(
        (tmp_path / "run/_task_drafting/_task_design/task_design.json").read_text()
    )
    assert written == DESIGN.payload(), "the design is where the drafter's prompt says it is"
    assert models.scripted(Role.TASK_DESIGNER).call_count == 1


async def test_a_failed_design_ends_drafting_before_any_scaffold(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await workspace.write_bytes("run/task/task.md", TASK_MD.encode())
    wrong = structured_call(DesignReview, review_payload("0" * 64))
    called: list[str] = []

    async def scaffold(state: Any, runtime: Any) -> dict[str, Any]:
        called.append("scaffold")
        return {}

    context, _ = make_context(
        workspace,
        {Role.TASK_DESIGNER: [structured_call(TaskDesign, DESIGN)], Role.DESIGN_JUDGE: [wrong] * 3},
    )
    graph = build_task_drafting_graph(
        checkpointer=InMemorySaver(),
        overrides=GraphOverrides(slots={"drafting.scaffold": scaffold}),
    )
    out = await invoke(graph, context, minimal_task_ref("run/task"))
    assert out["drafting"].status == "failed"
    assert out["drafting"].failure.category == "task_design_review_invalid"
    assert called == []


async def test_the_legacy_workflow_skips_the_design_loop(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    task = await write_scaffold(workspace, only_task_md=True)
    called: list[str] = []

    async def design(state: Any, runtime: Any) -> dict[str, Any]:
        called.append("design")
        return {"accepted_design": {"x": 1}}

    context, _ = make_context(
        workspace,
        {Role.TASK_DRAFTER: drafter_script(), Role.ALIGNMENT_JUDGE: [judge_step()]},
        config=fast(workflow="legacy"),
    )
    graph = build_task_drafting_graph(
        checkpointer=InMemorySaver(), overrides=GraphOverrides(slots={"drafting.design": design})
    )
    out = await invoke(graph, context, task)
    assert out["drafting"].status == "published", out["drafting"].failure
    assert called == []
    assert not (tmp_path / "_task_drafting/_task_design").exists()


async def test_an_accepted_design_reaches_the_judge_and_the_receipts(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    task = await write_scaffold(workspace, only_task_md=True)
    design = {"schema_version": 1, "method": {"class": "toy"}}

    async def accepted(state: Any, runtime: Any) -> dict[str, Any]:
        return {"accepted_design": design, "accepted_review": {"verdict": "pass"}}

    context, models = make_context(
        workspace, {Role.TASK_DRAFTER: drafter_script(), Role.ALIGNMENT_JUDGE: [judge_step()]}
    )
    graph = build_task_drafting_graph(
        checkpointer=InMemorySaver(), overrides=GraphOverrides(slots={"drafting.design": accepted})
    )
    out = await invoke(graph, context, task)
    drafting = out["drafting"]
    assert drafting.status == "published", drafting.failure
    assert drafting.receipts["design"] == design and drafting.receipts["design_review"] == {
        "verdict": "pass"
    }
    judge_prompt = next(
        message.content
        for message in models.scripted(Role.ALIGNMENT_JUDGE).calls[0]
        if message.type == "human"
    )
    assert "Accepted execution design: `./_accepted_task_design.json`" in judge_prompt
    view = tmp_path / "_task_drafting/.harness/judge_view_0/_accepted_task_design.json"
    assert json.loads(view.read_text()) == design
    draft_prompt = next(
        message.content
        for message in models.scripted(Role.TASK_DRAFTER).calls[0]
        if message.type == "human"
    )
    assert "Accepted task design: `./_task_design/task_design.json`" in draft_prompt
