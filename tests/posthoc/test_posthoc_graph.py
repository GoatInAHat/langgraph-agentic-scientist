"""The post-hoc graph alone: the whole flow, the bootstrap, the attempt loop, the publication."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from posthoc_support import (
    config,
    evaluator_with_score,
    executor_script,
    finished_run,
    generator,
    last_human_text,
    proposal,
    selector,
    shell,
    valid_outputs,
    write_all,
)

from langgraph_agentic_scientist.evaluation import DeferredEvaluator
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.posthoc import build_posthoc_graph
from langgraph_agentic_scientist.testing import (
    ScriptedModels,
    final,
    tool_call,
)

ANALYSIS = "weights"
ANALYSIS_DIR = f"posthoc/analyses/{ANALYSIS}"


def _models(**scripts: Any) -> ScriptedModels:
    return ScriptedModels.from_scripts(
        {Role(role): steps for role, steps in scripts.items()}, context_window=200_000
    )


def _input(task: Any, investigation: Any, **extra: Any) -> dict[str, Any]:
    return {"task": task, "run_id": "r1", "investigation": investigation, **extra}


def _thread(name: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": name}}


async def _read(workspace: Any, path: str) -> Any:
    return json.loads((await workspace.read_bytes(path)).decode())


# --- the whole flow -----------------------------------------------------------------------------


async def test_a_valid_analysis_is_published(make_context: Any, workspace: Any) -> None:
    task, investigation = await finished_run(workspace)
    models = _models(
        analysis_generator=generator(
            proposal(ANALYSIS),
            proposal("ablate_it", family="ablation_study", title="An ablation"),
            proposal("retrain_it", family="other", title="Retrain", requires_retrain=True),
        ),
        analysis_selector=selector((ANALYSIS, True, "good"), ("ablate_it", True, "ok")),
        analysis_executor=executor_script(ANALYSIS, valid_outputs(ANALYSIS, 1)),
    )
    evaluator = evaluator_with_score()
    ctx = make_context(models=models, evaluator=evaluator, config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("flow"), context=ctx)

    result = out["posthoc"]
    assert result.status == "completed", result.failure
    assert [analysis.analysis_id for analysis in result.analyses] == [ANALYSIS]
    analysis = result.analyses[0]
    assert analysis.status == "passed" and analysis.attempts == 1
    assert analysis.analysis_dir == f"{ANALYSIS_DIR}/attempt_1"
    assert analysis.manifest["claim"] == "weights are non-negative"
    assert analysis.outputs == ["weights.csv"]

    assert evaluator.calls == [], "a usable checkpoint needs no bootstrap rerun"
    report = await _read(workspace, "posthoc/bootstrap_report.json")
    assert report["bootstrap_ready"] is True and report["status"] == "completed"
    assert report["checkpoint"]["selected_path"] == "best_model.pkl"
    assert report["baseline_primary_metric"]["key"] == "combined_score"

    assert await workspace.exists("posthoc/generation/posthoc_bundle.json")
    selection = await _read(workspace, "posthoc/generation/posthoc_selection.json")
    assert [item["id"] for item in selection["accepted"]] == [ANALYSIS]
    reasons = {item["id"]: item["reason"] for item in selection["rejected"]}
    assert "excludes ablations" in reasons["ablate_it"]
    assert "analysis-only" in reasons["retrain_it"] or "not relevant" in reasons["retrain_it"]

    ranking = await _read(workspace, "posthoc/ranking.json")
    assert ranking["delegated_to_paper_agent"] is True and ranking["selected_analysis_id"] is None
    (entry,) = ranking["valid"]
    assert entry["selected_attempt"] == "attempt_1"
    assert entry["analysis_manifest_path"] == f"{ANALYSIS_DIR}/attempt_1/analysis_manifest.json"
    latest = await _read(workspace, "posthoc/latest_manifest.json")
    assert latest["valid_analyses"][0]["analysis_id"] == ANALYSIS
    summary = await _read(workspace, "posthoc/pipeline_summary.json")
    assert summary["status"] == "completed" and summary["total_valid"] == 1
    assert not await workspace.exists("latest_manifest.json"), "the root is left alone"
    stored = await _read(workspace, f"{ANALYSIS_DIR}/analysis_result.json")
    assert stored["status"] == "completed" and stored["selected_attempt"] == "attempt_1"
    run_manifest = await _read(workspace, f"{ANALYSIS_DIR}/run_manifest.json")
    assert run_manifest["status"] == "completed" and run_manifest["attempt_count"] == 1


async def test_the_analysis_workspace_is_built_as_upstream_builds_it(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(ANALYSIS, valid_outputs(ANALYSIS, 1)),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    await app.ainvoke(_input(task, investigation), _thread("layout"), context=ctx)

    bundle = f"{ANALYSIS_DIR}/task_bundle"
    assert not await workspace.exists(f"{bundle}/evaluate.py"), "the evaluator is removed"
    assert not await workspace.exists(f"{bundle}/extraction_manifest.json")
    assert await workspace.exists(f"{bundle}/original_task.md")
    task_md = (await workspace.read_bytes(f"{bundle}/task.md")).decode()
    assert task_md.startswith(
        "# Post-Hoc Analysis: Analysis weights\n\n- **Analysis ID:** `weights`"
    )
    assert "## Required Artifacts\n- `best_model.pkl`\n" in task_md
    assert "1. load the checkpoint\n2. plot the weights\n" in task_md
    assert task_md.endswith(
        "Refer to `original_task.md` for the original search-time task and evaluator constraints.\n"
    )
    assert await workspace.exists(f"{bundle}/program_0/best_model.pkl")
    assert await workspace.exists(f"{bundle}/reference_program/program_spec.toml")
    contract = await _read(workspace, f"{ANALYSIS_DIR}/rubric_contract.json")
    assert [item["id"] for item in contract["rubrics"]][-1] == "checkpoint_guardrail_respected"
    assert (await _read(workspace, f"{ANALYSIS_DIR}/target_rubrics.json"))[0].keys() == {
        "id",
        "description",
    }
    # the active files the posthoc agent script stages, and the staged skills
    attempt = f"{ANALYSIS_DIR}/attempt_1"
    assert await workspace.exists(f"{attempt}/active_idea.json")
    assert await workspace.exists(f"{attempt}/active_rubrics.json")
    assert await workspace.exists(f"{attempt}/.agents/skills/edit-program/SKILL.md")
    skill = await workspace.read_bytes(f"{attempt}/.agents/skills/evaluate-program/SKILL.md")
    assert b"{{RESULTS_DIR}}" not in skill and b"0" in skill


async def test_the_executor_prompt_is_upstreams_and_has_the_run_skills_and_tools(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(ANALYSIS, valid_outputs(ANALYSIS, 1)),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    await app.ainvoke(_input(task, investigation), _thread("prompt"), context=ctx)
    executor = models.scripted(Role.ANALYSIS_EXECUTOR)
    prompt = last_human_text(executor.calls[0])
    assert prompt.startswith("Execute one post-hoc analysis attempt.")
    assert "Attempt number: 1" in prompt
    assert '"analysis_id": "weights"' in prompt
    assert "Previous error context:\n```json\n\n```" in prompt
    tools = set(executor.bound_tools[0])
    assert {"clone_program", "evaluate_program", "write_file", "execute"} <= tools
    skills = {
        entry.path.split("/")[0]
        for entry in await workspace.list_tree(f"{ANALYSIS_DIR}/attempt_1/.agents/skills")
    }
    assert skills == {"edit-program", "evaluate-program", "verify-rubrics", "memory-log"}
    for role in (Role.ANALYSIS_GENERATOR, Role.ANALYSIS_SELECTOR):
        assert "skills" not in " ".join(models.scripted(role).bound_tools[0])


# --- the attempt loop ---------------------------------------------------------------------------


async def test_a_failed_attempt_is_retried_with_its_error_as_context(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    incomplete = {"analysis_summary.md": "only a summary\n"}
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(ANALYSIS, incomplete, valid_outputs(ANALYSIS, 2)),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("retry"), context=ctx)
    assert out["posthoc"].status == "completed"
    assert out["posthoc"].analyses[0].attempts == 2
    executor = models.scripted(Role.ANALYSIS_EXECUTOR)
    prompts = [last_human_text(call) for call in executor.calls]
    second = next(prompt for prompt in prompts if "Attempt number: 2" in prompt)
    assert "missing attempt outputs" in second and '"overall_pass": false' in second
    stored = await _read(workspace, f"{ANALYSIS_DIR}/attempt_1/attempt_result.json")
    assert stored["inspection"]["required_outputs_present"] is False
    assert (await _read(workspace, f"{ANALYSIS_DIR}/analysis_result.json"))["attempt_count"] == 2
    # the repair happens in the next attempt directory, which starts from the previous one
    assert await workspace.exists(f"{ANALYSIS_DIR}/attempt_2/analysis_summary.md")


async def test_the_attempt_budget_is_a_bound_and_the_analysis_is_rejected(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    nothing: dict[str, str] = {"note.txt": "nothing useful\n"}
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(ANALYSIS, nothing, nothing),
    )
    ctx = make_context(
        models=models,
        evaluator=evaluator_with_score(),
        config=config(max_attempts_per_analysis=2),
    )
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("budget"), context=ctx)
    result = out["posthoc"]
    assert result.status == "degraded"
    assert result.failure is not None and result.failure.category == "no_valid_analyses"
    assert result.analyses[0].status == "failed" and result.analyses[0].attempts == 2
    ranking = await _read(workspace, "posthoc/ranking.json")
    assert ranking["valid"] == []
    assert ranking["rejected"][0]["reason"] == "no_successful_attempt"
    assert not await workspace.exists("posthoc/latest_manifest.json")
    summary = await _read(workspace, "posthoc/pipeline_summary.json")
    assert summary["status"] == "no_valid_analyses"


async def test_a_tampered_task_bundle_is_restored_and_the_attempt_fails(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    tamper_source = [
        shell("echo changed > ../task_bundle/tamper.txt"),
        write_all(valid_outputs(ANALYSIS, 1)),
    ]
    tamper_attempt = [shell("echo changed > task_bundle/tamper.txt")]
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(
            ANALYSIS, tamper_source, tamper_attempt, valid_outputs(ANALYSIS, 3)
        ),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("tamper"), context=ctx)
    result = out["posthoc"]
    assert result.status == "completed" and result.analyses[0].attempts == 3
    first = await _read(workspace, f"{ANALYSIS_DIR}/attempt_1/attempt_result.json")
    assert first["inspection"]["manifest_error"] == (
        "source task_bundle was modified during analysis execution"
    )
    second = await _read(workspace, f"{ANALYSIS_DIR}/attempt_2/attempt_result.json")
    assert (
        second["inspection"]["manifest_error"]
        == "task_bundle was modified during analysis execution"
    )
    assert not await workspace.exists(f"{ANALYSIS_DIR}/task_bundle/tamper.txt"), "pristine restored"
    assert not await workspace.exists(f"{ANALYSIS_DIR}/attempt_3/task_bundle/tamper.txt")


async def test_a_rerun_evaluator_claim_and_missing_evidence_fail_the_attempt(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    reran = valid_outputs(ANALYSIS, 1, reran_evaluator=True)
    ungrounded = valid_outputs(ANALYSIS, 2, evidence_paths=["task_bundle/no_such_file.txt"])
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(ANALYSIS, reran, ungrounded, valid_outputs(ANALYSIS, 3)),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    await app.ainvoke(_input(task, investigation), _thread("claims"), context=ctx)
    one = await _read(workspace, f"{ANALYSIS_DIR}/attempt_1/attempt_result.json")
    assert one["inspection"]["manifest_error"] == "analysis_manifest reran_evaluator must be false"
    two = await _read(workspace, f"{ANALYSIS_DIR}/attempt_2/attempt_result.json")
    assert "missing evidence paths" in two["inspection"]["manifest_error"]


async def test_a_timed_out_session_ends_the_analysis_as_timeout(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=[TimeoutError("the session outlived its wall clock")],
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("timeout"), context=ctx)
    result = out["posthoc"]
    assert result.analyses[0].status == "timeout" and result.analyses[0].attempts == 1
    ranking = await _read(workspace, "posthoc/ranking.json")
    assert ranking["rejected"][0]["reason"] == "timeout"
    assert (await _read(workspace, f"{ANALYSIS_DIR}/analysis_result.json"))["exit_code"] == 124


async def test_the_executor_has_no_evaluation_budget(make_context: Any, workspace: Any) -> None:
    task, investigation = await finished_run(workspace)
    evaluator = evaluator_with_score()
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(
            ANALYSIS,
            [tool_call("evaluate_program", {"program_dir": "task_bundle/program_0"})],
        ),
    )
    ctx = make_context(
        models=models, evaluator=evaluator, config=config(max_attempts_per_analysis=1)
    )
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    await app.ainvoke(_input(task, investigation), _thread("budget0"), context=ctx)
    assert evaluator.calls == [], "the budget of 0 refuses before any evaluation"
    seen = [
        message.content
        for call in models.scripted(Role.ANALYSIS_EXECUTOR).calls
        for message in call
        if getattr(message, "type", "") == "tool"
    ]
    assert any("budget of this attempt is consumed" in str(text) for text in seen)


# --- the bootstrap ------------------------------------------------------------------------------


async def test_a_missing_checkpoint_degrades_the_bootstrap_after_three_attempts(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace, checkpoint=False)
    models = _models(
        bootstrap_agent=[final("no change"), final("no change"), final("no change")],
        analysis_generator=generator(),
    )
    evaluator = evaluator_with_score()
    ctx = make_context(models=models, evaluator=evaluator, config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("degraded"), context=ctx)
    result = out["posthoc"]
    assert result.status == "degraded"
    assert result.failure is not None and result.failure.category == "bootstrap_not_ready"
    assert [call.evaluation_id for call in evaluator.calls] == [
        "r1/posthoc/bootstrap_01",
        "r1/posthoc/bootstrap_02",
        "r1/posthoc/bootstrap_03",
    ]
    assert all(call.plane == "bootstrap" for call in evaluator.calls)
    report = await _read(workspace, "posthoc/bootstrap_report.json")
    assert report["status"] == "degraded" and report["bootstrap_ready"] is False
    assert report["failure_reason"] == "bootstrap completed but no checkpoint was found"
    assert [item["attempt_index"] for item in report["retry_attempts"]] == [1, 2, 3]
    assert report["bootstrap_max_attempts"] == 3 and report["checkpoint_mode"] == "required"
    assert report["checkpoint"]["exists"] is False
    # the run goes on: the generator was asked, with the report in its prompt
    prompt = last_human_text(models.scripted(Role.ANALYSIS_GENERATOR).calls[0])
    assert "bootstrap_ready" in prompt and "degraded" in prompt


async def test_the_bootstrap_agent_can_create_the_checkpoint(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace, checkpoint=False)
    from posthoc_support import CHECKPOINT

    models = _models(
        bootstrap_agent=[
            tool_call(
                "write_file", {"file_path": "/program_0/best_weights.pkl", "content": CHECKPOINT}
            ),
            final("checkpoint saving added"),
        ],
        analysis_generator=generator(),
    )
    evaluator = evaluator_with_score(2.0)
    ctx = make_context(models=models, evaluator=evaluator, config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("created"), context=ctx)
    assert out["posthoc"].bootstrap["bootstrap_ready"] is True
    (request,) = evaluator.calls
    assert request.evaluation_id == "r1/posthoc/bootstrap_01"
    assert request.program.path == "posthoc/_analysis_ready_bundle/program_0"
    report = await _read(workspace, "posthoc/bootstrap_report.json")
    assert report["checkpoint"]["selected_path"] == "best_weights.pkl"
    assert report["current_primary_metric"]["value"] == 2.0
    prompt = last_human_text(models.scripted(Role.BOOTSTRAP_AGENT).calls[0])
    assert prompt.startswith("You are preparing the extracted best program for post-hoc analysis.")
    assert "SCORE = 2.0" in prompt, "the program source is in the prompt"
    assert "argparse" in prompt, "the evaluator source is in the prompt"


async def test_a_regressed_primary_metric_fails_the_guardrail(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace, checkpoint=False)
    from posthoc_support import CHECKPOINT

    save = [
        tool_call("write_file", {"file_path": "/program_0/model.pkl", "content": CHECKPOINT}),
        final("done"),
    ]
    models = _models(
        bootstrap_agent=save * 3,
        analysis_generator=generator(),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(1.0), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    await app.ainvoke(_input(task, investigation), _thread("guardrail"), context=ctx)
    report = await _read(workspace, "posthoc/bootstrap_report.json")
    assert report["bootstrap_ready"] is False
    assert report["failure_reason"].startswith("bootstrap guardrail failed")
    assert "regressed beyond tolerance" in report["failure_reason"]
    attempt = report["retry_attempts"][0]
    assert attempt["guardrail"]["passed"] is False and attempt["status"] == "failed"
    assert attempt["current_primary_metric"]["value"] == 1.0


async def test_the_integrated_mode_needs_no_checkpoint_and_no_retraining(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace, checkpoint=False)
    models = _models(analysis_generator=generator())
    evaluator = evaluator_with_score()
    ctx = make_context(models=models, evaluator=evaluator, config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    await app.ainvoke(
        _input(task, investigation, artifact_only=True), _thread("artifact"), context=ctx
    )
    assert evaluator.calls == []
    report = await _read(workspace, "posthoc/bootstrap_report.json")
    assert report["bootstrap_ready"] is True and report["checkpoint_mode"] == "optional"
    assert report["retrain_if_checkpoint_missing"] is False


async def test_the_bootstrap_rerun_interrupts_once_and_the_agent_is_not_run_twice(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace, checkpoint=False)
    from posthoc_support import CHECKPOINT

    models = _models(
        bootstrap_agent=[
            tool_call("write_file", {"file_path": "/program_0/model.pkl", "content": CHECKPOINT}),
            final("done"),
        ],
        analysis_generator=generator(),
    )
    ctx = make_context(models=models, evaluator=DeferredEvaluator(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    thread = _thread("deferred")
    out = await app.ainvoke(_input(task, investigation), thread, context=ctx)
    (pending,) = out["__interrupt__"]
    request = pending.value["request"]
    assert request["evaluation_id"] == "r1/posthoc/bootstrap_01" and request["plane"] == "bootstrap"
    agent = models.scripted(Role.BOOTSTRAP_AGENT)
    calls_before = agent.call_count
    from langgraph_agentic_scientist.evaluation import EvaluationResult

    answer = EvaluationResult(
        evaluation_id=request["evaluation_id"],
        status="completed",
        metrics={"combined_score": 2.0},
        correct=True,
    ).model_dump(mode="json")
    final_state = await app.ainvoke(Command(resume=answer), thread, context=ctx)
    assert final_state["posthoc"].bootstrap["bootstrap_ready"] is True
    assert agent.call_count == calls_before, "the bootstrap agent ran again after the interrupt"


# --- failure to extract, no candidates, selector failure -----------------------------------------


async def test_a_missing_incumbent_fails_the_phase(make_context: Any, workspace: Any) -> None:
    task, _ = await finished_run(workspace)
    ctx = make_context(models=_models(), evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke({"task": task, "run_id": "r1"}, _thread("noinc"), context=ctx)
    assert out["posthoc"].status == "failed"
    assert out["posthoc"].failure.category == "incumbent_unavailable"
    assert not await workspace.exists("posthoc/ranking.json")


async def test_zero_candidates_are_allowed_and_the_selector_is_not_asked(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    models = _models(analysis_generator=generator())
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("zero"), context=ctx)
    assert out["posthoc"].status == "degraded"
    assert out["posthoc"].failure.category == "no_valid_analyses"
    selection = await _read(workspace, "posthoc/generation/posthoc_selection.json")
    assert selection["accepted"] == [] and selection["schema_version"] == 1
    assert models.scripted(Role.ANALYSIS_SELECTOR).call_count == 0
    summary = await _read(workspace, "posthoc/generation/change_summary.json")
    assert summary["decision"] == "no_relevant_analyses" and summary["num_candidates"] == 0


async def test_the_selector_fails_open(make_context: Any, workspace: Any) -> None:
    task, investigation = await finished_run(workspace)
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=[RuntimeError("the selector's server went away")],
        analysis_executor=executor_script(ANALYSIS, valid_outputs(ANALYSIS, 1)),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("failopen"), context=ctx)
    assert out["posthoc"].status == "completed"


async def test_hooks_around_the_steps_run_once_across_an_interrupt(
    make_context: Any, workspace: Any
) -> None:
    from langgraph_agentic_scientist import Hooks

    task, investigation = await finished_run(workspace, checkpoint=False)
    from posthoc_support import CHECKPOINT

    count = {"before_evaluate": 0}

    async def before(state: Any, runtime: Any) -> dict[str, Any]:
        count["before_evaluate"] += 1
        return {"hook_outputs": {"seen": count["before_evaluate"]}}

    models = _models(
        bootstrap_agent=[
            tool_call("write_file", {"file_path": "/program_0/model.pkl", "content": CHECKPOINT}),
            final("done"),
        ],
        analysis_generator=generator(),
    )
    ctx = make_context(
        models=models,
        evaluator=DeferredEvaluator(),
        config=config(),
        hooks=Hooks(before={"posthoc.bootstrap_evaluate": [before]}),
    )
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    thread = _thread("hooks")
    out = await app.ainvoke(_input(task, investigation), thread, context=ctx)
    (pending,) = out["__interrupt__"]
    from langgraph_agentic_scientist.evaluation import EvaluationResult

    answer = EvaluationResult(
        evaluation_id=pending.value["request"]["evaluation_id"],
        status="completed",
        metrics={"combined_score": 2.0},
        correct=True,
    ).model_dump(mode="json")
    await app.ainvoke(Command(resume=answer), thread, context=ctx)
    assert count["before_evaluate"] == 1, "a completed hook ran again after the interrupt"
    snapshot = await app.aget_state(thread)
    assert snapshot.values["hook_outputs"]["posthoc.bootstrap_evaluate"] == {"seen": 1}


# --- inputs of the phase ------------------------------------------------------------------------


async def test_evidence_of_completed_ablations_reaches_the_generator_and_every_analysis(
    make_context: Any, workspace: Any
) -> None:
    from posthoc_support import ablations as _ablations

    from langgraph_agentic_scientist.core.results import VariantsResult

    task, investigation = await finished_run(workspace)
    await _ablations(workspace, root="ablations")
    variants = VariantsResult(status="completed", output_dir="ablations")
    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(ANALYSIS, valid_outputs(ANALYSIS, 1)),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(
        _input(task, investigation, variants=variants), _thread("evidence"), context=ctx
    )
    assert out["posthoc"].status == "completed"
    prompt = last_human_text(models.scripted(Role.ANALYSIS_GENERATOR).calls[0])
    assert "evidence/pipeline_summary.json" in prompt, "the manifest names the staged paths"
    assert await workspace.exists("posthoc/generation/evidence/README.md")
    assert await workspace.exists(f"{ANALYSIS_DIR}/task_bundle/evidence/README.md")
    assert await workspace.exists(f"{ANALYSIS_DIR}/task_bundle/evidence_manifest.json")
    failed = await app.ainvoke(
        _input(
            task,
            investigation,
            variants=VariantsResult(status="completed", output_dir="nowhere"),
        ),
        _thread("noevidence"),
        context=ctx,
    )
    assert failed["posthoc"].status == "failed"
    assert failed["posthoc"].failure.category == "extraction_failed"


async def test_a_downstream_selection_is_the_incumbent_and_is_recorded(
    make_context: Any, workspace: Any
) -> None:
    from langgraph_agentic_scientist.core.results import DownstreamSelection
    from langgraph_agentic_scientist.core.state import ProgramRef
    from langgraph_agentic_scientist.testing import write_program

    task, investigation = await finished_run(workspace)
    await write_program(workspace, "downstream/program", "SCORE = 1.5\n")
    await workspace.write_bytes("downstream/program/metrics.json", b'{"combined_score": 1.5}')
    await workspace.write_bytes("downstream/program/correct.json", b'{"correct": true}')
    await workspace.write_bytes("downstream/program/m.pkl", b"x")
    downstream = DownstreamSelection(
        status="selected",
        program=ProgramRef(path="downstream/program"),
        source="candidate",
        combined_score=1.5,
    )
    ctx = make_context(
        models=_models(analysis_generator=generator()),
        evaluator=evaluator_with_score(),
        config=config(),
    )
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    await app.ainvoke(
        _input(task, investigation, downstream=downstream), _thread("downstream"), context=ctx
    )
    manifest = await _read(workspace, "posthoc/_extracted_bundle/extraction_manifest.json")
    assert manifest["source_best_program_dir"] == "downstream/program"
    assert manifest["downstream_selection"]["status"] == "selected"
    assert manifest["combined_score"] == 1.5


async def test_several_analyses_run_as_separate_subgraphs(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace)
    models = _models(
        analysis_generator=generator(proposal("first", family="a"), proposal("second", family="b")),
        analysis_selector=selector(("first", True, "x"), ("second", True, "y")),
        analysis_executor=[
            *executor_script("first", valid_outputs("first", 1)),
            *executor_script("second", valid_outputs("second", 1)),
        ],
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    config_run = {**_thread("many"), "max_concurrency": 1}
    out = await app.ainvoke(_input(task, investigation), config_run, context=ctx)
    result = out["posthoc"]
    assert result.status == "completed"
    assert [analysis.analysis_id for analysis in result.analyses] == ["first", "second"]
    ranking = await _read(workspace, "posthoc/ranking.json")
    assert [entry["analysis_id"] for entry in ranking["valid"]] == ["first", "second"]


async def test_the_bootstrap_works_with_the_local_command_evaluator(
    make_context: Any, workspace: Any
) -> None:
    from posthoc_support import CHECKPOINT

    from langgraph_agentic_scientist.evaluation import LocalCommandEvaluator

    task, investigation = await finished_run(workspace, checkpoint=False)
    models = _models(
        bootstrap_agent=[
            tool_call(
                "write_file", {"file_path": "/program_0/model_state.pkl", "content": CHECKPOINT}
            ),
            final("done"),
        ],
        analysis_generator=generator(),
    )
    ctx = make_context(models=models, evaluator=LocalCommandEvaluator(workspace), config=config())
    app = build_posthoc_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task, investigation), _thread("local"), context=ctx)
    assert out["posthoc"].bootstrap["bootstrap_ready"] is True, out["posthoc"].bootstrap
    ready = "posthoc/_analysis_ready_bundle/program_0"
    assert await _read(workspace, f"{ready}/metrics.json") == {"combined_score": 2.0}
    assert await _read(workspace, f"{ready}/correct.json") == {"correct": True}
    assert await workspace.exists(".evaluations/r1/posthoc/bootstrap_01/invocation.json")


class _StopBeforeAnalysis(Exception):
    """Raised by the capturing slot once it has an analysis input."""


async def test_the_analysis_graph_runs_alone_with_its_own_checkpointer(
    make_context: Any, workspace: Any
) -> None:
    from langgraph_agentic_scientist import GraphOverrides
    from langgraph_agentic_scientist.posthoc import ANALYSIS_SLOT, build_analysis_graph
    from langgraph_agentic_scientist.posthoc.models import AnalysisInput

    task, investigation = await finished_run(workspace)
    captured: dict[str, Any] = {}

    async def capture(state: Any, runtime: Any) -> Any:
        captured.update({key: state[key] for key in AnalysisInput.__annotations__ if key in state})
        raise _StopBeforeAnalysis

    models = _models(
        analysis_generator=generator(proposal(ANALYSIS)),
        analysis_selector=selector((ANALYSIS, True, "good")),
        analysis_executor=executor_script(ANALYSIS, valid_outputs(ANALYSIS, 1)),
    )
    ctx = make_context(models=models, evaluator=evaluator_with_score(), config=config())
    phase = build_posthoc_graph(
        checkpointer=InMemorySaver(), overrides=GraphOverrides(slots={ANALYSIS_SLOT: capture})
    )
    with pytest.raises(_StopBeforeAnalysis):
        await phase.ainvoke(_input(task, investigation), _thread("capture"), context=ctx)
    assert set(AnalysisInput.__annotations__) <= set(captured)

    graph = build_analysis_graph(checkpointer=InMemorySaver())
    out = await graph.ainvoke(captured, _thread("alone"), context=ctx)

    (run,) = out["analysis_runs"]
    assert run.analysis_id == ANALYSIS and run.status == "completed" and run.attempt_count == 1
    assert (await graph.aget_state(_thread("alone"))).next == ()
