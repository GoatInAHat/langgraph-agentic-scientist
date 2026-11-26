"""Stage 6: cross-task generalisation (adapt the control, then every other variant)."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_agentic_scientist import ScientistContext
from langgraph_agentic_scientist.core.results import VariantsResult
from langgraph_agentic_scientist.core.state import ProgramRef, TaskRef
from langgraph_agentic_scientist.evaluation.protocol import EvaluationResult
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import (
    FakeEvaluator,
    ScriptedModels,
    final,
    write_minimal_task,
)
from langgraph_agentic_scientist.variants import build_variants_graph
from langgraph_agentic_scientist.variants.cross_task import (
    find_best_program_dir,
    reference_evaluator_section,
)
from langgraph_agentic_scientist.workspace import LocalWorkspace

from .helpers import bundle_answer, decisions_answer, proposal, study_scripts, summaries_answer


def _scripts() -> dict[Role, Any]:
    return study_scripts(
        generator=[bundle_answer(proposal("ablation_no_reheat"))],
        reviewer=[decisions_answer(("ablation_no_reheat", True, "ok"))],
        summariser=[summaries_answer("S.")],
        retrofit=[final("primary retrofit"), final("extra retrofit")],
    )


async def _run(ctx: ScientistContext, task: TaskRef, best: ProgramRef) -> VariantsResult:
    graph = build_variants_graph(checkpointer=InMemorySaver())
    out = await graph.ainvoke(
        {"task": task, "run_id": "r1", "best": best},
        {"configurable": {"thread_id": "t"}},
        context=ctx,
    )
    result = out["variants"]
    assert isinstance(result, VariantsResult)
    return result


def _read(workspace: LocalWorkspace, path: str) -> str:
    return (workspace.root / path).read_bytes().decode()


async def test_cross_task_adapts_the_control_first_then_the_variants(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    await write_minimal_task(workspace, "extra1", program="SCORE = 0.5\n")
    evaluator = FakeEvaluator()
    ctx = phase_context(_scripts(), evaluator=evaluator, extra_tasks=("extra1",))
    result = await _run(ctx, task, best)
    assert result.status == "completed", result.failure
    ids = [call.evaluation_id for call in evaluator.calls]
    # primary study first (control before the rest), then the extra task: control first
    assert ids[0] == "r1/variants/control_best/program_0"
    extra = [i for i in ids if "/extra/" in i]
    assert extra[0] == "r1/extra/extra1/control_best/program_1"  # the control adapts first
    assert set(extra[1:]) == {
        "r1/extra/extra1/baseline_reference/program_1",
        "r1/extra/extra1/ablation_no_reheat/program_1",
    }
    assert ids.index(extra[0]) > max(i for i, v in enumerate(ids) if "/extra/" not in v)
    assert {call.task.root for call in evaluator.calls if "/extra/" in call.evaluation_id} == {
        "extra1"
    }
    assert {call.task.root for call in evaluator.calls if "/extra/" not in call.evaluation_id} == {
        "task"
    }
    assert next(r.variant_id for r in result.cross_task) == "control_best"
    assert len(result.cross_task) == 3 and all(r.passed for r in result.cross_task)
    assert {r.task_name for r in result.cross_task} == {"extra1"}

    base = "variants/extra_tasks/extra1"
    assert (workspace.root / base / "_retrofitted_bundle/.retrofit_complete.json").is_file()
    control = f"{base}/variants/control_best/task_bundle"
    assert "# Cross-Task Adaptation: Control (best program)" in _read(
        workspace, f"{control}/task.md"
    )
    assert "Target task:** extra1" in _read(workspace, f"{control}/task.md")
    assert "SCORE = 3.0" in _read(
        workspace, f"{control}/program_0/src/program.py"
    )  # the best program
    assert _read(workspace, f"{control}/source_task.md").startswith("# Minimal task")
    assert _read(workspace, f"{control}/original_task.md").startswith("# Minimal task")
    ablation = f"{base}/variants/ablation_no_reheat/task_bundle"
    # the variant starts from the adapted control and carries its own best program for reference
    assert (workspace.root / ablation / "original_variant_program/program_spec.toml").is_file()
    text = _read(workspace, f"{ablation}/task.md")
    assert text.startswith("# Cross-Task Adaptation: Title of ablation_no_reheat\n")
    assert "**Original kind:** ablation" in text and "## Original Variant Information" in text
    spec = json.loads(_read(workspace, f"{base}/variants/control_best/variant_spec.json"))
    assert spec["kind"] == "baseline"  # the control adapts, it does not just evaluate
    assert spec["title"] == "Cross-task: Control (best program) -> extra1"
    # the retrofit agent of the extra task ran, with the original evaluator as reference context
    retrofit = ctx.models.scripted(Role.RETROFIT_AGENT)  # type: ignore[attr-defined]
    assert retrofit.call_count == 2
    prompt = retrofit.calls[1][-1].content
    assert "## Reference Evaluator (read-only context)" in prompt and "{{" not in prompt
    # aggregation and the README cover the extra task
    assert (
        result.aggregates["extra1/run_summary.csv"] == "variants/aggregated/extra1/run_summary.csv"
    )
    assert "### extra1 (cross-task)" in _read(workspace, "variants/README.md")
    assert "extra_tasks/{task}/variants/{id}/" in _read(workspace, "variants/README.md")


async def test_a_failing_adapted_control_fails_the_stage_and_ends_the_phase(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    await write_minimal_task(workspace, "extra1")
    failed = EvaluationResult(evaluation_id="x", status="failed", summary="cannot adapt")
    evaluator = FakeEvaluator(results={"r1/extra/extra1/control_best/program_1": failed})
    ctx = phase_context(_scripts(), evaluator=evaluator, extra_tasks=("extra1",))
    result = await _run(ctx, task, best)
    assert result.status == "failed" and result.failure is not None
    assert result.failure.category == "cross_task_failed"
    assert "adapted control failed verification" in result.failure.detail
    # no variant was adapted, and neither aggregation nor README ran
    assert not any(
        "ablation_no_reheat/program_1" in c.evaluation_id and "extra" in c.evaluation_id
        for c in evaluator.calls
    )
    assert result.readme_path is None and result.aggregates == {}


async def test_extra_tasks_must_be_valid(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    ctx = phase_context(_scripts(), extra_tasks=("missing_task",))
    result = await _run(ctx, task, best)
    assert result.status == "failed" and result.failure is not None
    assert "evaluator preflight failed" in result.failure.detail
    await write_minimal_task(workspace, "a/same")
    await write_minimal_task(workspace, "b/same")
    ctx = phase_context(_scripts(), extra_tasks=("a/same", "b/same"))
    graph = build_variants_graph(checkpointer=InMemorySaver())
    out = await graph.ainvoke(
        {"task": task, "run_id": "r2", "best": best, "output_dir": "o2"},
        {"configurable": {"thread_id": "t2"}},
        context=ctx,
    )
    assert out["variants"].failure is not None
    assert "not a unique safe name" in out["variants"].failure.detail
    assert isinstance(ctx.models, ScriptedModels)


async def test_best_program_dir_is_the_highest_complete_evaluated_program(
    workspace: LocalWorkspace,
) -> None:
    for index in (0, 1, 2):
        await workspace.write_bytes(f"b/program_{index}/program_spec.toml", b"x")
    await workspace.write_bytes("b/program_1/metrics.json", b"{}")
    await workspace.write_bytes("b/program_2/metrics.json", b"{}")
    await workspace.write_bytes(
        "b/program_2/.evaluation_pending", b"id"
    )  # an unfinished evaluation
    assert await find_best_program_dir(workspace, "b", upto=2) == "b/program_1"
    assert await find_best_program_dir(workspace, "empty", upto=2) is None


def test_reference_evaluator_section_is_empty_without_an_evaluator() -> None:
    from langgraph_agentic_scientist import PromptRegistry

    prompts = PromptRegistry.default()
    assert reference_evaluator_section(prompts, "") == ""
    text = reference_evaluator_section(prompts, "print(1)")
    assert text.startswith("## Reference Evaluator (read-only context)") and "print(1)" in text


class _StopBeforeCrossTask(Exception):
    """Raised by the capturing hook once it has the cross-task stage's input."""


async def test_the_cross_task_graph_runs_alone_with_its_own_checkpointer(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    import dataclasses

    from langgraph_agentic_scientist import Hooks
    from langgraph_agentic_scientist.variants import build_cross_task_graph
    from langgraph_agentic_scientist.variants.cross_task import CrossTaskInput

    task, best = study
    await write_minimal_task(workspace, "extra1", program="SCORE = 0.5\n")
    captured: dict[str, Any] = {}

    async def capture(state: Any, runtime: Any) -> None:
        captured.update({key: state[key] for key in CrossTaskInput.__annotations__ if key in state})
        raise _StopBeforeCrossTask

    evaluator = FakeEvaluator()
    ctx = phase_context(_scripts(), evaluator=evaluator, extra_tasks=("extra1",))
    hooked = dataclasses.replace(ctx, hooks=Hooks(before={"variants.cross_task": [capture]}))
    with pytest.raises(_StopBeforeCrossTask):
        await _run(hooked, task, best)
    assert set(CrossTaskInput.__annotations__) <= set(captured)

    graph = build_cross_task_graph(checkpointer=InMemorySaver())
    out = await graph.ainvoke(captured, {"configurable": {"thread_id": "alone"}}, context=ctx)

    assert out["cross_failure"] is None
    assert out["cross_task_runs"][0].result.variant_id == "control_best"
    assert all(run.result.passed for run in out["cross_task_runs"])
    assert (await graph.aget_state({"configurable": {"thread_id": "alone"}})).next == ()
