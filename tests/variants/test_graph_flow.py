"""The variants graph end to end with scripted roles and a fake evaluator."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_agentic_scientist import GraphOverrides, Hooks, ScientistContext
from langgraph_agentic_scientist.core.results import (
    DownstreamSelection,
    InvestigationResult,
    VariantsResult,
)
from langgraph_agentic_scientist.core.state import BestManifest, ProgramRef, TaskRef
from langgraph_agentic_scientist.evaluation.protocol import EvaluationResult
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import FakeEvaluator, ScriptedModels
from langgraph_agentic_scientist.variants import build_run_variant_graph, build_variants_graph
from langgraph_agentic_scientist.variants.tasks import task_ref_for
from langgraph_agentic_scientist.workspace import LocalWorkspace

from .helpers import (
    bundle_answer,
    decisions_answer,
    idle_editor,
    proposal,
    study_scripts,
    summaries_answer,
    variants_config,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "core"


def _config(thread: str = "t") -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


async def _run(
    ctx: ScientistContext,
    task: TaskRef,
    best: ProgramRef | None = None,
    *,
    thread: str = "t",
    saver: InMemorySaver | None = None,
    overrides: GraphOverrides | None = None,
    **extra: Any,
) -> VariantsResult:
    graph = build_variants_graph(checkpointer=saver or InMemorySaver(), overrides=overrides)
    payload: dict[str, Any] = {"task": task, "run_id": "r1", **extra}
    if best is not None:
        payload["best"] = best
    out = await graph.ainvoke(payload, _config(thread), context=ctx)
    result = out["variants"]
    assert isinstance(result, VariantsResult)
    return result


def _models(ctx: ScientistContext) -> ScriptedModels:
    assert isinstance(ctx.models, ScriptedModels)
    return ctx.models


def _read(workspace: LocalWorkspace, path: str) -> str:
    return (workspace.root / path).read_bytes().decode()


def _json(workspace: LocalWorkspace, path: str) -> Any:
    return json.loads(_read(workspace, path))


def _standard(**kw: Any) -> dict[Role, Any]:
    return study_scripts(
        generator=[
            bundle_answer(proposal("ablation_no_reheat"), proposal("baseline_simple", "baseline"))
        ],
        reviewer=[
            decisions_answer(("ablation_no_reheat", True, "fine"), ("baseline_simple", True, "ok"))
        ],
        summariser=[summaries_answer("Maximise SCORE.", control_best="The best program.")],
        **kw,
    )


async def test_full_flow_control_first_then_variants(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    evaluator = FakeEvaluator()
    ctx = phase_context(_standard(), evaluator=evaluator)
    result = await _run(ctx, task, best)
    assert result.status == "completed", result.failure
    ids = [spec.variant_id for spec in result.specs]
    assert ids == ["control_best", "baseline_reference", "ablation_no_reheat", "baseline_simple"]
    assert [run.status for run in result.variants].count("passed") == 4
    # the control was evaluated before any other variant, and everything on the variant plane
    assert evaluator.calls[0].evaluation_id == "r1/variants/control_best/program_0"
    assert {call.plane for call in evaluator.calls} == {"variant"}
    assert len(evaluator.calls) == 4
    readme = _read(workspace, "variants/README.md")
    assert readme.startswith("# Variants Experiment Overview\n\nMaximise SCORE.\n")
    assert "The best program." in readme  # the agent's summary of the control
    assert (
        "Hypothesis of ablation_no_reheat" in readme
    )  # fallback for a variant it did not summarise
    summary = _json(workspace, "variants/pipeline_summary.json")
    assert summary["rubric_passed"] == 4 and summary["total_variants"] == 4
    assert result.readme_path == "variants/README.md"
    assert result.aggregates["task/run_summary.csv"] == "variants/aggregated/task/run_summary.csv"
    # the selection record keeps the reviewer's decisions
    selection = _json(workspace, "variants/generation/variant_selection.json")
    assert [a["variant_id"] for a in selection["accepted"]] == [
        "ablation_no_reheat",
        "baseline_simple",
    ]
    assert (workspace.root / "variants/generation/variant_bundle.json").is_file()
    assert _json(workspace, "variants/generation/change_summary.json")["num_candidates"] == 2


async def test_only_the_edit_and_verify_roles_get_skills(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    ctx = phase_context(_standard())
    await _run(ctx, task, best)
    for directory in ("_retrofitted_bundle", "generation", "_readme_work", "_extracted_bundle"):
        assert not (workspace.root / "variants" / directory / ".agents").exists(), directory
    for variant in ("control_best", "ablation_no_reheat", "baseline_simple"):
        skills = (
            workspace.root / f"variants/variants/{variant}/task_bundle/.agents/skills"
        ).iterdir()
        assert {p.name for p in skills} <= {"edit-program", "memory-log", "verify-rubrics"}
    # the skill texts carry the variant's values (RESULTS_DIR is the bundle itself)
    text = _read(
        workspace,
        "variants/variants/ablation_no_reheat/task_bundle/.agents/skills/edit-program/SKILL.md",
    )
    assert "{{RESULTS_DIR}}" not in text and "program_<idx-1>" in text


async def test_generator_and_reviewer_prompts_carry_the_diff_and_the_candidates(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    ctx = phase_context(_standard())
    await _run(ctx, task, best)
    generator = _models(ctx).scripted(Role.VARIANT_GENERATOR)
    prompt = generator.calls[0][-1].content
    assert isinstance(prompt, str)
    assert "# Variant Generation: Propose Baselines and Ablations" in prompt
    assert "-SCORE = 1.0" in prompt and "+SCORE = 3.0" in prompt  # best vs the seed
    assert "{{" not in prompt
    assert "Return your output as your structured response" in prompt
    reviewer = _models(ctx).scripted(Role.VARIANT_REVIEWER)
    review_prompt = reviewer.calls[0][-1].content
    assert isinstance(review_prompt, str)
    assert '"variant_id": "ablation_no_reheat"' in review_prompt and "{{" not in review_prompt
    assert (
        "base_program"
        not in review_prompt.split("## Proposed Candidates")[1].split("## Your Task")[0]
    )


async def test_a_failing_reviewer_accepts_every_candidate(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    scripts = study_scripts(
        generator=[bundle_answer(proposal("ablation_a"), proposal("ablation_b"))],
        reviewer=[ValueError("the reviewer crashed")],
        summariser=[summaries_answer("S.")],
    )
    result = await _run(phase_context(scripts), task, best)
    assert [s.variant_id for s in result.specs][2:] == ["ablation_a", "ablation_b"]
    selection = _json(workspace, "variants/generation/variant_selection.json")
    assert selection["num_accepted"] == 2 and selection["num_rejected"] == 0


async def test_hard_caps_bound_what_the_reviewer_accepts(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    entries = [proposal(f"ablation_{i}") for i in range(5)] + [
        proposal(f"baseline_{i}", "baseline") for i in range(4)
    ]
    scripts = study_scripts(
        generator=[bundle_answer(*entries)],
        reviewer=[decisions_answer(*[(e["variant_id"], True, "ok") for e in entries])],
        summariser=[summaries_answer("S.")],
    )
    result = await _run(phase_context(scripts, max_ablations=2, max_extra_baselines=1), task, best)
    ids = [s.variant_id for s in result.specs]
    assert ids[2:] == ["ablation_0", "ablation_1", "baseline_0"]
    selection = _json(workspace, "variants/generation/variant_selection.json")
    assert selection["num_rejected"] == 6
    assert {r["reason"] for r in selection["rejected"]} == {
        "ablation cap reached (2)",
        "baseline cap reached (1)",
    }


async def test_zero_proposals_are_valid_and_the_reviewer_is_not_asked(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    scripts = study_scripts(
        generator=[bundle_answer(decision="no_relevant_variants")],
        summariser=[summaries_answer("S.")],
    )
    ctx = phase_context(scripts)
    result = await _run(ctx, task, best)
    assert result.status == "completed"
    assert [s.variant_id for s in result.specs] == ["control_best", "baseline_reference"]
    assert _models(ctx).scripted(Role.VARIANT_REVIEWER).call_count == 0


async def test_a_failed_control_aborts_the_remaining_variants(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    failed = EvaluationResult(evaluation_id="x", status="failed", summary="the control broke")
    evaluator = FakeEvaluator(results={"r1/variants/control_best/program_0": failed})
    ctx = phase_context(_standard(), evaluator=evaluator)
    result = await _run(ctx, task, best)
    assert result.status == "failed"
    assert result.failure is not None and result.failure.category == "required_control_failed"
    assert "control_best" in result.failure.detail
    assert [c.evaluation_id for c in evaluator.calls] == ["r1/variants/control_best/program_0"]
    summary = _json(workspace, "variants/pipeline_summary.json")
    statuses = {v["variant_id"]: v["status"] for v in summary["variants"]}
    assert statuses == {
        "control_best": "failed",
        "baseline_reference": "not_executed",
        "ablation_no_reheat": "not_executed",
        "baseline_simple": "not_executed",
    }
    # no later stage ran
    assert result.readme_path is None and result.aggregates == {}
    assert not (workspace.root / "variants/README.md").exists()


async def test_zero_passing_variants_fail_the_phase(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, _ = study
    await workspace.write_bytes(
        "custom/baseline_mine.md", b"---\nvariant_id: baseline_mine\ntitle: Mine\n---\nBody.\n"
    )
    failed = EvaluationResult(evaluation_id="x", status="failed", summary="no")
    ctx = phase_context(
        study_scripts(),
        evaluator=FakeEvaluator(default=failed),
        custom_only=True,
        custom_baselines=("custom/baseline_mine.md",),
    )
    result = await _run(ctx, task, ProgramRef(path="runs/best/program_1"))
    assert result.status == "failed"
    assert result.failure is not None and result.failure.category == "no_valid_variant"


async def test_custom_only_runs_the_custom_baselines_and_skips_generation_and_controls(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    await workspace.write_bytes(
        "custom/dfa.md",
        b"---\nvariant_id: dfa\ntitle: DFA\nhypothesis: random feedback\n---\n"
        b"## Description\nDo DFA.\n",
    )
    evaluator = FakeEvaluator()
    ctx = phase_context(
        study_scripts(summariser=[summaries_answer("S.", baseline_dfa="Direct feedback.")]),
        evaluator=evaluator,
        custom_only=True,
        custom_baselines=("custom/dfa.md",),
    )
    result = await _run(ctx, task, best)
    assert result.status == "completed", result.failure
    assert [s.variant_id for s in result.specs] == ["baseline_dfa"]
    models = _models(ctx)
    assert models.scripted(Role.VARIANT_GENERATOR).call_count == 0
    assert models.scripted(Role.VARIANT_REVIEWER).call_count == 0
    assert not (workspace.root / "variants/generation").exists()
    task_md = _read(workspace, "variants/variants/baseline_dfa/task_bundle/task.md")
    assert "## Detailed Description\n## Description\nDo DFA." in task_md
    assert [c.evaluation_id for c in evaluator.calls] == ["r1/variants/baseline_dfa/program_1"]


async def test_custom_baselines_join_the_generated_ones_and_replace_the_same_id(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    await workspace.write_bytes(
        "custom/simple.md",
        b"---\nvariant_id: baseline_simple\ntitle: Mine\nhypothesis: mine\n---\nCustom body.\n",
    )
    ctx = phase_context(_standard(), custom_baselines=("custom/simple.md",))
    result = await _run(ctx, task, best)
    simple = next(s for s in result.specs if s.variant_id == "baseline_simple")
    assert simple.title == "Mine" and simple.description == "Custom body."
    assert [s.variant_id for s in result.specs].count("baseline_simple") == 1


async def test_a_custom_baseline_with_a_reserved_id_is_refused(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    await workspace.write_bytes("custom/c.md", b"---\nvariant_id: baseline_reference\n---\nx\n")
    # 'baseline_reference' is already prefixed; it is reserved for the original seed
    ctx = phase_context(_standard(), custom_baselines=("custom/c.md",))
    result = await _run(ctx, task, best)
    assert result.status == "failed" and result.failure is not None
    assert result.failure.category == "materialize_failed"
    assert "reserved variant IDs" in result.failure.detail


async def test_readme_falls_back_when_the_summariser_fails(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    scripts = study_scripts(
        generator=[bundle_answer(proposal("ablation_no_reheat", hypothesis=""))],
        reviewer=[decisions_answer(("ablation_no_reheat", True, "ok"))],
        summariser=[RuntimeError("model down")],
    )
    result = await _run(phase_context(scripts), task, best)
    assert result.status == "completed"
    readme = _read(workspace, "variants/README.md")
    # the task summary is the first non-heading line of task.md; summaries are hypotheses
    assert (
        "\n\nMake `SCORE` in the program as large as possible while keeping it at most 10.\n"
        in readme
    )
    assert "Baseline control using the best program as-is." in readme
    assert "Compare the improved best program against the original seed." in readme
    assert "Title of ablation_no_reheat (ablation variant)." in readme


async def test_the_best_program_may_come_from_a_downstream_selection_or_an_investigation(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    scripts = study_scripts(
        generator=[bundle_answer(decision="no_relevant_variants")],
        summariser=[summaries_answer("S.")],
    )
    by_investigation = InvestigationResult(
        status="completed",
        best=BestManifest(
            program_dir=best.path, entrypoint_path=best.entrypoint, combined_score=3.0, correct=True
        ),
    )
    result = await _run(phase_context(scripts), task, investigation=by_investigation)
    assert result.status == "completed", result.failure
    manifest = _json(workspace, "variants/_extracted_bundle/extraction_manifest.json")
    assert manifest["source_best_program_dir"] == "runs/best/program_1"
    assert manifest["combined_score"] == 3.0 and manifest["downstream_selection"] is None
    # the downstream selection (the no-winner fallback) wins over the investigation's manifest
    fallback = DownstreamSelection(
        status="selected", program=best, source="candidate", combined_score=2.0
    )
    scripts = study_scripts(
        generator=[bundle_answer(decision="no_relevant_variants")],
        summariser=[summaries_answer("S.")],
    )
    result = await _run(
        phase_context(scripts),
        task,
        thread="t2",
        investigation=by_investigation,
        downstream=fallback,
        output_dir="out2",
    )
    manifest = _json(workspace, "out2/_extracted_bundle/extraction_manifest.json")
    assert manifest["combined_score"] == 2.0
    assert manifest["downstream_selection"]["status"] == "selected"


async def test_no_best_program_is_a_failure(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, _ = study
    result = await _run(phase_context(study_scripts()), task)
    assert result.status == "failed" and result.failure is not None
    assert result.failure.category == "bundle_extraction_failed"
    assert "No best program" in result.failure.detail


async def test_baseline_program_override_replaces_the_reference(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    from langgraph_agentic_scientist.testing import write_program

    await write_program(workspace, "other/program_5", "SCORE = 2.0\n")
    scripts = study_scripts(
        generator=[bundle_answer(decision="no_relevant_variants")],
        summariser=[summaries_answer("S.")],
    )
    ctx = phase_context(scripts, baseline_program_override="other/program_5")
    result = await _run(ctx, task, best)
    assert result.status == "completed", result.failure
    assert (
        _read(workspace, "variants/_extracted_bundle/reference_program/src/program.py")
        == "SCORE = 2.0\n"
    )
    prompt = _models(ctx).scripted(Role.VARIANT_GENERATOR).calls[0][-1].content
    assert isinstance(prompt, str) and "-SCORE = 2.0" in prompt and "+SCORE = 3.0" in prompt


async def test_the_phase_needs_an_evaluator_and_isolates_contracted_tasks(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    make_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    ctx = make_context(study_scripts(), config=variants_config())
    result = await _run(ctx, task, best)
    assert result.failure is not None and result.failure.category == "no_evaluator"
    # a contracted task is evaluated only by an isolated evaluator
    shutil.copytree(FIXTURES / "contract_task", workspace.root / "ctask")
    contracted = await task_ref_for(workspace, "ctask")
    assert contracted.contract is not None
    local = FakeEvaluator(isolated=False)
    ctx = make_context(study_scripts(), evaluator=local, config=variants_config())
    result = await _run(ctx, contracted, ProgramRef(path="ctask/program_0"), thread="t2")
    assert result.failure is not None
    assert result.failure.category == "contracted_task_requires_isolation"
    assert local.calls == []


async def test_a_brokered_candidate_needs_no_retrofit_agent(
    workspace: LocalWorkspace,
    phase_context: Callable[..., ScientistContext],
    make_context: Callable[..., ScientistContext],
) -> None:
    shutil.copytree(FIXTURES / "contract_task", workspace.root / "ctask")
    contracted = await task_ref_for(workspace, "ctask")
    scripts = study_scripts(
        generator=[bundle_answer(decision="no_relevant_variants")],
        summariser=[summaries_answer("S.")],
        retrofit=[],  # a call would exhaust the script and fail the test
    )
    ctx = make_context(
        scripts,
        evaluator=FakeEvaluator(),
        config=variants_config(),
    )
    from langgraph_agentic_scientist import ScientistConfig

    ctx = ScientistContext(
        models=ctx.models,
        workspace=workspace,
        evaluator=FakeEvaluator(isolated=False),
        config=ScientistConfig(allow_unisolated_contracted=True, agents=ctx.config.agents),
    )
    result = await _run(ctx, contracted, ProgramRef(path="ctask/program_0"))
    assert result.status == "completed", result.failure
    assert _models(ctx).scripted(Role.RETROFIT_AGENT).call_count == 0
    marker = _json(workspace, "variants/_retrofitted_bundle/.retrofit_complete.json")
    assert marker["schema_version"] == 2


async def test_hooks_run_around_steps_and_may_adjust_the_selection(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    make_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    seen: list[str] = []

    async def drop_the_baseline(state: Any, runtime: Any) -> dict[str, Any]:
        selection = state["selection"]
        seen.append("after select")
        kept = [c for c in selection.accepted if c.kind == "ablation"]
        return {
            "selection": selection.model_copy(update={"accepted": kept}),
            "hook_outputs": {"dropped": 1},
        }

    async def before_materialize(state: Any, runtime: Any) -> None:
        seen.append("before materialize")

    ctx = make_context(
        _standard(),
        evaluator=FakeEvaluator(),
        config=variants_config(),
        hooks=Hooks(
            before={"variants.materialize": [before_materialize]},
            after={"variants.select": [drop_the_baseline]},
        ),
    )
    graph = build_variants_graph(checkpointer=InMemorySaver())
    out = await graph.ainvoke({"task": task, "run_id": "r1", "best": best}, _config(), context=ctx)
    result = out["variants"]
    assert [s.variant_id for s in result.specs] == [
        "control_best",
        "baseline_reference",
        "ablation_no_reheat",
    ]
    assert seen == ["after select", "before materialize"]


async def test_the_variant_runner_is_replaceable(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    scripts = study_scripts(
        generator=[bundle_answer(decision="no_relevant_variants")],
        summariser=[summaries_answer("S.")],
    )
    ctx = phase_context(scripts)
    replacement = build_run_variant_graph(name="my_runner")
    result = await _run(
        ctx, task, best, overrides=GraphOverrides(slots={"variants.run_variant": replacement})
    )
    assert result.status == "completed"
    with pytest.raises(TypeError, match="must be a compiled graph"):
        build_variants_graph(overrides=GraphOverrides(slots={"variants.run_variant": idle_editor}))  # type: ignore[dict-item]


async def test_resume_after_a_crash_repeats_no_role_and_no_evaluation(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    task, best = study
    key = "r1/variants/ablation_no_reheat/program_1"
    evaluator = FakeEvaluator(fail_once=[key], reattach=True)
    ctx = phase_context(_standard(), evaluator=evaluator)
    saver = InMemorySaver()
    graph = build_variants_graph(checkpointer=saver)
    with pytest.raises(ConnectionError):
        await graph.ainvoke({"task": task, "run_id": "r1", "best": best}, _config(), context=ctx)
    models = _models(ctx)
    before = {
        role: models.scripted(role).call_count
        for role in (Role.RETROFIT_AGENT, Role.VARIANT_GENERATOR, Role.VARIANT_REVIEWER)
    }
    assert all(count == 1 for count in before.values())
    out = await graph.ainvoke(None, _config(), context=ctx)
    result = out["variants"]
    assert result.status == "completed", result.failure
    after = {role: models.scripted(role).call_count for role in before}
    assert after == before
    assert evaluator.runs[key] == 1
    assert all(runs == 1 for runs in evaluator.runs.values())


async def test_state_survives_a_strict_checkpoint_round_trip(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    phase_context: Callable[..., ScientistContext],
) -> None:
    """Every type the phase stores is on the allowlist: resume deserialises it under it."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from langgraph_agentic_scientist import serde_allowlist

    task, best = study
    key = "r1/variants/ablation_no_reheat/program_1"
    evaluator = FakeEvaluator(fail_once=[key], reattach=True)
    ctx = phase_context(_standard(), evaluator=evaluator)
    serde = JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist())
    saver = InMemorySaver(serde=serde)
    graph = build_variants_graph(checkpointer=saver)
    with pytest.raises(ConnectionError):
        await graph.ainvoke({"task": task, "run_id": "r1", "best": best}, _config(), context=ctx)
    out = await graph.ainvoke(None, _config(), context=ctx)
    assert out["variants"].status == "completed", out["variants"].failure


async def test_deferred_evaluations_pause_the_run_and_resume_by_interrupt_id(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    make_context: Callable[..., ScientistContext],
) -> None:
    """An orchestrator that evaluates in its own executor resumes the graph with the results."""
    from langgraph.types import Command

    from langgraph_agentic_scientist.evaluation import DeferredEvaluator

    task, best = study
    scripts = study_scripts(
        generator=[bundle_answer(proposal("ablation_no_reheat"))],
        reviewer=[decisions_answer(("ablation_no_reheat", True, "ok"))],
        summariser=[summaries_answer("S.")],
    )
    ctx = make_context(scripts, evaluator=DeferredEvaluator(), config=variants_config())
    graph = build_variants_graph(checkpointer=InMemorySaver())

    def answer(request: dict[str, Any], score: float = 2.0) -> dict[str, Any]:
        return EvaluationResult(
            evaluation_id=request["evaluation_id"],
            status="completed",
            metrics={"combined_score": score},
            correct=True,
        ).model_dump(mode="json")

    out = await graph.ainvoke({"task": task, "run_id": "r1", "best": best}, _config(), context=ctx)
    (first,) = out["__interrupt__"]
    assert first.value["kind"] == "evaluation_request"
    assert first.value["request"]["evaluation_id"] == "r1/variants/control_best/program_0"
    assert first.value["request"]["plane"] == "variant"
    # the control's evaluation is answered; the other two variants are then pending at once
    out = await graph.ainvoke(
        Command(resume=answer(first.value["request"])), _config(), context=ctx
    )
    pending = {i.value["request"]["evaluation_id"]: i for i in out["__interrupt__"]}
    assert set(pending) == {
        "r1/variants/baseline_reference/program_1",
        "r1/variants/ablation_no_reheat/program_1",
    }
    resume = {i.id: answer(i.value["request"], 3.0) for i in pending.values()}
    out = await graph.ainvoke(Command(resume=resume), _config(), context=ctx)
    result = out["variants"]
    assert result.status == "completed", result.failure
    assert [r.passed for r in result.variants].count(True) == 3
    editor = _models(ctx).scripted(Role.VARIANT_EDITOR)
    assert editor.call_count == 4  # two sessions of one call each; none repeated by the resumes


async def test_with_the_local_evaluator_the_programs_own_series_reach_the_analysis(
    workspace: LocalWorkspace,
    study: tuple[TaskRef, ProgramRef],
    make_context: Callable[..., ScientistContext],
) -> None:
    from langgraph_agentic_scientist.evaluation import LocalCommandEvaluator

    task, best = study
    evaluator_source = (workspace.root / "task/evaluate.py").read_text() + (
        "\nseries = os.path.join(args.results_dir, 'runs', 'rep_1', 'analysis')\n"
        "os.makedirs(series, exist_ok=True)\n"
        "with open(os.path.join(series, 'timeseries.csv'), 'w') as handle:\n"
        "    handle.write('evaluation_id,variant_id,series_name,step,value\\n')\n"
        "    handle.write(f'x,y,loss,0,{score}\\n')\n"
    )
    await workspace.write_bytes("task/evaluate.py", evaluator_source.encode())
    scripts = study_scripts(
        generator=[bundle_answer(decision="no_relevant_variants")],
        summariser=[summaries_answer("S.")],
    )
    ctx = make_context(
        scripts, evaluator=LocalCommandEvaluator(workspace), config=variants_config()
    )
    result = await _run(ctx, task, best)
    assert result.status == "completed", result.failure
    series = _read(
        workspace, "variants/variants/control_best/task_bundle/program_0/analysis/timeseries.csv"
    )
    assert "control_best_eval_1,control_best,runs/rep_1/loss,0,3.0" in series
    metrics = _json(workspace, "variants/variants/control_best/task_bundle/program_0/metrics.json")
    assert metrics == {"combined_score": 3.0}
    ref = _json(
        workspace, "variants/variants/baseline_reference/task_bundle/program_1/metrics.json"
    )
    assert ref == {"combined_score": 1.0}  # the seed
    # the evaluator kept its own outputs apart from the candidate's workspace
    assert (
        workspace.root / ".evaluations/r1/variants/control_best/program_0/result.json"
    ).is_file()
    assert not (workspace.root / "variants/variants/control_best/task_bundle/.evaluations").exists()
