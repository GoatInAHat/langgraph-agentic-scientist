"""The paper graph alone: prebuilt paper data, a replaced ``paper.data`` slot, the final-outcome
gate, and resume after a crash without repeating a completed step."""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from test_loop_support import (
    audit_pass,
    build,
    fast_config,
    models,
    no_gaps,
    paper_config,
    paper_data,
    reports,
    run,
    setup_task,
    write_paper,
)

from langgraph_agentic_scientist import Hooks
from langgraph_agentic_scientist.core.results import PaperData
from langgraph_agentic_scientist.models import Role


def _roles() -> Any:
    return models(
        {
            Role.PAPER_WRITER: write_paper(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )


async def test_prebuilt_paper_data_with_artifact_paths(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    await workspace.write_bytes("mine/results/README.md", b"# My campaign\nThree variants.\n")
    await workspace.write_bytes("mine/memory/memory.jsonl", b'{"step": "idea", "summary": "s"}\n')
    await workspace.write_bytes("mine/memory/memory.md", b"# memory\n")
    await workspace.write_bytes("mine/contributions.md", b"We contribute {{PAPER_FORMAT}} things.")
    scripted = _roles()
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(
        build(),
        ctx,
        task,
        paper_dir="out/paper_1",
        artifact_paths={
            "results": "mine/results",
            "global_memory": "mine/memory",
            "contributions": "mine/contributions.md",
        },
    )
    assert out["paper"].status == "complete"
    assert out["paper"].paper_dir == "out/paper_1"
    assert await workspace.exists("out/latest_manifest.json")
    readme = (await workspace.read_bytes("out/paper_1/results_readme.md")).decode()
    assert readme.startswith("# My campaign")
    contributions = (await workspace.read_bytes("out/paper_1/contributions.md")).decode()
    assert contributions == "We contribute iclr things."
    summary = (await workspace.read_bytes("out/paper_1/global_memory_summary.md")).decode()
    assert summary.startswith("# Global Memory Summary")
    assert "step=idea" in summary
    prompt = scripted.scripted(Role.PAPER_WRITER).calls[0]
    text = next(m.content for m in prompt if m.type == "human")
    assert "# My campaign" in text
    assert "A contributions scaffold is provided for this run." in text
    memory = (await workspace.read_bytes("mine/memory/memory.jsonl")).decode()
    assert '"step": "paper"' in memory


async def test_replaced_paper_data_slot_as_callable(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    calls: list[str] = []

    async def builder(state: Any, runtime: Any) -> dict[str, Any]:
        calls.append(state["paper_dir"])
        return {"paper_data": paper_data(idea_titles=["mine"])}

    ctx = make_context(
        models=_roles(), paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    graph = build(slots={"paper.data": builder})
    out = await run(graph, ctx, task, paper_data=None)
    assert out["paper"].status == "complete"
    assert calls == ["run/paper/final_r1"]
    written = json.loads(
        (await workspace.read_bytes("run/paper/final_r1/paper_data.json")).decode()
    )
    assert written["idea_titles"] == ["mine"]


class _SlotState(TypedDict, total=False):
    paper_dir: str
    paper_data: PaperData | None


async def test_replaced_paper_data_slot_as_graph(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)

    def produce(state: _SlotState) -> dict[str, Any]:
        return {"paper_data": paper_data(idea_titles=["graph"])}

    sub = StateGraph(_SlotState)
    sub.add_node("produce", produce)
    sub.add_edge(START, "produce")
    sub.add_edge("produce", END)
    ctx = make_context(
        models=_roles(), paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(slots={"paper.data": sub.compile()}), ctx, task, paper_data=None)
    assert out["paper"].status == "complete"
    assert out["paper_data"].idea_titles == ["graph"]


async def test_final_outcome_gate(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    data = paper_data(
        final_evaluation={"status": "completed", "correct": False, "outcome": "not_qualified"}
    )
    ctx = make_context(
        models=models({}), paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task, paper_data=data)
    assert out["paper"].status == "failed"
    assert out["paper"].failure.detail.startswith(
        "Final evaluation did not qualify for publication"
    )
    allowed = make_context(
        models=_roles(),
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(), final_outcome_policy="allow_not_qualified"),
    )
    out = await run(build(), allowed, task, paper_data=data, thread="allowed")
    assert out["paper"].status == "complete"


async def test_no_evaluated_programs_fails(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    data = paper_data()
    data.totals["evaluated_programs"] = 0
    ctx = make_context(
        models=models({}), paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task, paper_data=data)
    assert out["paper"].failure.detail == (
        "No evaluated programs found in results directory; cannot write paper."
    )


async def test_resume_after_a_crash_repeats_no_completed_step(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    crashes = {"left": 1}

    async def crash_once(state: Any, runtime: Any) -> None:
        if crashes["left"]:
            crashes["left"] -= 1
            raise RuntimeError("host crashed before validation")

    scripted = _roles()
    toolchain = reports(True)
    ctx = make_context(
        models=scripted,
        paper_toolchain=toolchain,
        config=fast_config(paper=paper_config()),
        hooks=Hooks(before={"paper.validate": [crash_once]}),
    )
    graph = build()
    config = {"configurable": {"thread_id": "resume"}}
    payload = {"task": task, "run_id": "r1", "paper_data": paper_data()}
    try:
        await graph.ainvoke(payload, config, context=ctx)
    except RuntimeError as exc:
        assert "host crashed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected the crash")
    assert scripted.scripted(Role.PAPER_WRITER).call_count == 2
    assert toolchain.validations == []
    out = await graph.ainvoke(None, config, context=ctx)
    assert out["paper"].status == "complete"
    assert scripted.scripted(Role.PAPER_WRITER).call_count == 2
    assert scripted.scripted(Role.DRAFT_GAP_ANALYST).call_count == 1
    assert len(toolchain.validations) == 1
