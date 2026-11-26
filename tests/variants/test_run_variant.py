"""One variant: edit tries, host evaluation, verification, canonicalisation, resume."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_agentic_scientist import ScientistContext
from langgraph_agentic_scientist.core.results import VariantSpec
from langgraph_agentic_scientist.core.state import ProgramRef, TaskRef
from langgraph_agentic_scientist.evaluation.protocol import EvaluationRequest, EvaluationResult
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import (
    FakeEvaluator,
    ScriptedModels,
    completed_result,
    final,
    tool_call,
)
from langgraph_agentic_scientist.variants import build_run_variant_graph, run_variant_input
from langgraph_agentic_scientist.variants import execute as ex
from langgraph_agentic_scientist.variants.models import VariantCandidate, VariantRun
from langgraph_agentic_scientist.workspace import LocalWorkspace

from .helpers import (
    cloning_editor,
    idle_editor,
    idle_verifier,
    judging_verifier,
    make_variant_workspaces,
    study_scripts,
)


class _Crash(BaseException):
    """A crash of the process: not an error the graph can catch."""


CANDIDATE = VariantCandidate(
    variant_id="ablation_no_reheat",
    title="No reheat",
    kind="ablation",
    hypothesis="reheating helps",
    target_components=["reheating"],
)


async def _run(
    ctx: ScientistContext,
    workspace: LocalWorkspace,
    task: TaskRef,
    best: ProgramRef,
    *,
    control: bool = False,
    timeout: float | None = None,
    max_evals: int = 2,
    saver: InMemorySaver | None = None,
    thread: str = "t",
    resume: bool = False,
) -> tuple[VariantRun | None, VariantSpec]:
    specs, extra = await make_variant_workspaces(
        ctx, workspace, task, best, candidates=[CANDIDATE], max_evals=max_evals
    )
    spec = next(s for s in specs if s.kind == ("control" if control else "ablation"))
    graph = build_run_variant_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": thread}}
    payload = (
        None
        if resume
        else run_variant_input(
            spec=spec,
            task=task,
            run_id="r",
            run_config=extra["run_config"],
            evaluator_digest=extra["digest"],
            agent_timeout=timeout,
            id_parts=["variants"],
        )
    )
    out = await graph.ainvoke(payload, config, context=ctx)
    runs = out["variant_runs"]
    return (runs[0] if runs else None), spec


def _models(ctx: ScientistContext) -> ScriptedModels:
    assert isinstance(ctx.models, ScriptedModels)
    return ctx.models


async def test_the_control_evaluates_program_0_and_no_editor_runs(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    evaluator = FakeEvaluator()
    ctx = phase_context(study_scripts(), evaluator=evaluator)
    run, spec = await _run(ctx, workspace, task, best, control=True)
    assert run is not None and run.status == "completed" and run.rubric_passed
    assert _models(ctx).scripted(Role.VARIANT_EDITOR).call_count == 0
    assert [call.evaluation_id for call in evaluator.calls] == ["r/variants/control_best/program_0"]
    assert run.result.best_program_index == 0 and run.eval_count == 1
    assert run.analysis_files == {
        "run_summary.csv": True,
        "metric_values.csv": True,
        "timeseries.csv": True,
        "rubric_status.json": True,
    }
    manifest = json.loads((workspace.root / spec.variant_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["eval_count"] == 1


async def test_edit_sessions_are_tried_at_most_three_times(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    evaluator = FakeEvaluator()
    ctx = phase_context(study_scripts(editor=idle_editor()), evaluator=evaluator)
    run, _ = await _run(ctx, workspace, task, best)
    assert run is not None and run.status == "failed" and not run.rubric_passed
    assert _models(ctx).scripted(Role.VARIANT_EDITOR).call_count == 3  # one session per try
    assert evaluator.calls == []
    assert "was not created after 3 tries" in run.output
    # each try is told which attempt it is
    prompts = [c[1].content for c in _models(ctx).scripted(Role.VARIANT_EDITOR).calls]
    assert [f"This is edit attempt {n}." in p for n, p in enumerate(prompts, start=1)] == [True] * 3


async def test_host_computes_overall_pass_and_writes_the_canonical_status_and_mirror(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    ctx = phase_context(study_scripts())
    run, spec = await _run(ctx, workspace, task, best)
    assert run is not None and run.status == "completed"
    bundle = workspace.root / spec.variant_dir / "task_bundle"
    status = json.loads((bundle / "program_1/rubric_status.json").read_text())
    assert status["overall_pass"] is True  # the verifier left it false: the host owns it
    assert (
        status["variant_id"] == "ablation_no_reheat" and status["idea_id"] == "ablation_no_reheat"
    )
    assert status["attempt_id"] == "ablation_no_reheat_attempt_01"
    assert status["metrics"] == {"combined_score": 1.0, "correct": True}
    assert (bundle / "rubric_status.json").read_text() == (
        bundle / "program_1/rubric_status.json"
    ).read_text()
    summary = (bundle / "program_1/analysis/run_summary.csv").read_bytes().decode()
    assert summary.splitlines()[1].split(",")[7] == "True"  # overall_pass qualified
    (record,) = run.result.evaluations
    assert (
        record.plane == "variant"
        and record.evaluation_id == "r/variants/ablation_no_reheat/program_1"
    )
    assert record.attempt_id == "ablation_no_reheat_attempt_01"
    assert run.result.rubric_status is not None and run.result.rubric_status.overall_pass is True
    # skills of the two roles are staged in the bundle; only these two roles get skills
    skills = sorted(p.name for p in (bundle / ".agents/skills").iterdir())
    assert skills == ["edit-program", "memory-log", "verify-rubrics"]


async def test_an_invalid_status_is_retried_then_replaced_by_an_all_fail_status(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    ctx = phase_context(study_scripts(verifier=idle_verifier()), max_evals_per_variant=2)
    run, spec = await _run(ctx, workspace, task, best)
    assert run is not None and run.status == "failed" and run.eval_count == 2
    verifier = _models(ctx).scripted(Role.VARIANT_VERIFIER)
    assert verifier.call_count == 6  # three tries for each of the two programs
    prompts = [c[1].content for c in verifier.calls]
    assert "Previous host validation error: (none; first verification attempt)" in prompts[0]
    assert (
        "Previous host validation error: rubric entry evidence at index 0 must be non-empty"
        in prompts[1]
    )
    status = json.loads(
        (workspace.root / spec.variant_dir / "task_bundle/program_2/rubric_status.json").read_text()
    )
    assert status["overall_pass"] is False
    assert all(r["evidence"].startswith("Verification incomplete:") for r in status["rubrics"])
    assert status["next_recommendation"] == "Repair rubric verification before another evaluation."


async def test_failed_rubrics_lead_to_a_repair_program(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    ctx = phase_context(study_scripts(verifier=judging_verifier(passing={2})))
    run, spec = await _run(ctx, workspace, task, best)
    assert run is not None and run.status == "completed" and run.eval_count == 2
    assert run.result.best_program_index == 2
    editor = _models(ctx).scripted(Role.VARIANT_EDITOR)
    second_prompt = next(c[1].content for c in editor.calls if "./program_2/" in c[1].content)
    assert '"failed_rubric_ids": ["analysis_files_present"' in second_prompt
    assert '"overall_pass": false' in second_prompt
    # the first program's status was valid and failing; the second passed
    first = json.loads(
        (workspace.root / spec.variant_dir / "task_bundle/program_1/rubric_status.json").read_text()
    )
    assert first["overall_pass"] is False


async def test_an_evaluation_that_did_not_complete_ends_the_variant(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    failed = EvaluationResult(
        evaluation_id="x", status="failed", failure_category="evaluator_failed", summary="boom"
    )
    evaluator = FakeEvaluator(default=failed)
    ctx = phase_context(study_scripts(), evaluator=evaluator)
    run, spec = await _run(ctx, workspace, task, best)
    assert run is not None and run.status == "failed" and run.eval_count == 0
    assert "evaluator_failed" in run.output and "boom" in run.output
    assert _models(ctx).scripted(Role.VARIANT_VERIFIER).call_count == 0
    assert len(evaluator.calls) == 1  # no second program after a failed evaluation
    assert not (workspace.root / spec.variant_dir / "task_bundle/program_1/metrics.json").exists()


async def test_an_evaluation_timeout_is_a_timeout(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    late = EvaluationResult(evaluation_id="x", status="timeout", summary="took too long")
    ctx = phase_context(study_scripts(), evaluator=FakeEvaluator(default=late))
    run, _ = await _run(ctx, workspace, task, best)
    assert run is not None and run.status == "timeout" and run.exit_code == 124
    assert run.result.status == "timeout"


async def test_an_infrastructure_error_is_reported_apart_from_candidate_failures(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    broken = EvaluationResult(evaluation_id="x", status="infrastructure_error", summary="no docker")
    ctx = phase_context(study_scripts(), evaluator=FakeEvaluator(default=broken))
    run, _ = await _run(ctx, workspace, task, best)
    assert run is not None and run.infrastructure_failure is not None
    assert run.infrastructure_failure.scope == "infrastructure"


async def test_the_wall_clock_of_an_agent_session_ends_the_variant_as_a_timeout(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study

    def slow(messages: list[Any]) -> Any:
        time.sleep(0.4)
        return final("too late")

    ctx = phase_context(study_scripts(editor=[slow] * 5), evaluator=FakeEvaluator())
    run, _ = await _run(ctx, workspace, task, best, timeout=0.05)
    assert run is not None and run.status == "timeout" and run.exit_code == 124
    assert _models(ctx).scripted(Role.VARIANT_EDITOR).call_count == 1  # no retry after a timeout
    assert ctx.evaluator is not None and not ctx.evaluator.calls  # type: ignore[attr-defined]


async def test_a_candidate_that_holds_evaluator_artifacts_is_refused(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    steps = cloning_editor()
    clone = steps[0]

    def cheating(messages: list[Any]) -> Any:
        result = clone(messages)
        planted = any(getattr(m, "name", "") == "write_file" for m in messages)
        if not result.tool_calls and not planted:  # the clone is made: plant an artifact
            return tool_call(
                "write_file", {"file_path": "/program_1/metrics.json", "content": "{}"}
            )
        return result

    evaluator = FakeEvaluator()
    ctx = phase_context(study_scripts(editor=[cheating] * 10), evaluator=evaluator)
    run, _ = await _run(ctx, workspace, task, best)
    assert run is not None and run.status == "failed"
    assert "candidate contains evaluator artifact" in run.output
    assert evaluator.calls == []  # nothing was evaluated


async def test_a_change_of_the_sealed_evaluator_during_evaluation_is_detected(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study

    def tamper(request: EvaluationRequest) -> EvaluationResult:
        (workspace.root / "task" / "evaluate.py").write_text("print('rigged')\n")
        return completed_result(request)

    ctx = phase_context(study_scripts(), evaluator=FakeEvaluator(default=tamper))  # type: ignore[arg-type]
    run, spec = await _run(ctx, workspace, task, best)
    assert run is not None and run.status == "failed"
    assert "sealed evaluator source tree changed" in run.output
    # nothing of the evaluation was left published
    assert not (workspace.root / spec.variant_dir / "task_bundle/program_1/metrics.json").exists()
    assert not (workspace.root / spec.variant_dir / "task_bundle/program_1/analysis").exists()
    assert _models(ctx).scripted(Role.VARIANT_VERIFIER).call_count == 0


async def test_resume_after_a_crash_in_the_evaluation_does_not_evaluate_or_edit_twice(
    workspace: LocalWorkspace, study: tuple[TaskRef, ProgramRef], phase_context: Any
) -> None:
    task, best = study
    evaluator = FakeEvaluator(fail_once=["r/variants/ablation_no_reheat/program_1"], reattach=True)
    ctx = phase_context(study_scripts(), evaluator=evaluator)
    saver = InMemorySaver()
    with pytest.raises(ConnectionError):
        await _run(ctx, workspace, task, best, saver=saver)
    editor = _models(ctx).scripted(Role.VARIANT_EDITOR)
    edits_before = editor.call_count
    # resume with the same thread: the evaluator re-attaches, the edit is not repeated
    graph = build_run_variant_graph(checkpointer=saver)
    out = await graph.ainvoke(None, {"configurable": {"thread_id": "t"}}, context=ctx)
    (run,) = out["variant_runs"]
    assert run.status == "completed"
    assert editor.call_count == edits_before
    assert evaluator.runs["r/variants/ablation_no_reheat/program_1"] == 1


async def test_publication_is_redone_after_a_crash_between_the_evaluator_and_the_files(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, best = study
    evaluator = FakeEvaluator()
    ctx = phase_context(study_scripts(), evaluator=evaluator)
    original: Callable[..., Any] = ex.publish_evaluation
    crashed = {"done": False}

    async def crash_once(*args: Any, **kwargs: Any) -> None:
        await original(*args, **kwargs)
        if not crashed["done"]:
            crashed["done"] = True
            await args[0].write_bytes(
                args[1] + "/.evaluation_pending", kwargs["request_id"].encode()
            )  # as if the manifest had not been written
            raise _Crash("process died after publishing")

    monkeypatch.setattr(ex, "publish_evaluation", crash_once)
    saver = InMemorySaver()
    with pytest.raises(_Crash):
        await _run(ctx, workspace, task, best, saver=saver)
    graph = build_run_variant_graph(checkpointer=saver)
    out = await graph.ainvoke(None, {"configurable": {"thread_id": "t"}}, context=ctx)
    (run,) = out["variant_runs"]
    assert run.status == "completed" and run.eval_count == 1
    assert evaluator.runs["r/variants/ablation_no_reheat/program_1"] == 1
