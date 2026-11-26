"""The task design graph with scripted models: the state machine of upstream's phase A1."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from test_design_support import (
    CLAIM,
    TASK_MD,
    auditor,
    design_payload,
    judge,
    make_design,
    patcher,
    prompt_of,
    review_payload,
)
from testkit import fast_config

from langgraph_agentic_scientist import ScientistConfig, ScientistContext
from langgraph_agentic_scientist.config import DraftingConfig
from langgraph_agentic_scientist.core.state import RunConfigRecord, TaskRef
from langgraph_agentic_scientist.drafting.design_models import DesignReview, TaskDesign
from langgraph_agentic_scientist.drafting.graph_design import build_task_design_graph
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import ScriptedModels, minimal_task_ref, structured_call


class Parent(TypedDict, total=False):
    """A parent state that shares the design graph's keys."""

    task: TaskRef
    run_config: RunConfigRecord | None
    status: str


WALL = "/planes/search/budget/wall_clock_seconds"
DESIGN = make_design()


async def _setup(workspace: Any) -> Any:
    await workspace.write_bytes("run/task/task.md", TASK_MD.encode("utf-8"))
    return minimal_task_ref("run/task")


def _models(**scripts: Any) -> ScriptedModels:
    return ScriptedModels.from_scripts(
        {Role(key): value for key, value in scripts.items()}, context_window=200_000
    )


async def _run(
    make_context: Any,
    workspace: Any,
    models: ScriptedModels,
    *,
    config: ScientistConfig | None = None,
) -> tuple[dict[str, Any], Any]:
    task = await _setup(workspace)
    ctx = make_context(models=models, **({"config": config} if config else {}))
    app = build_task_design_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke({"task": task}, {"configurable": {"thread_id": "design"}}, context=ctx)
    return out, ctx


def _config(**drafting: Any) -> ScientistConfig:
    base = fast_config()
    return base.model_copy(update={"drafting": DraftingConfig(**drafting)})


async def test_accepted_on_the_first_review(make_context: Any, workspace: Any) -> None:
    seen: list[str] = []
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[judge(seen=seen)],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "accepted"
    assert out["design"].fingerprint() == DESIGN.fingerprint()
    assert out["accepted_review"].verdict == "pass"
    assert [entry.event for entry in out["ledger"]] == ["reviewed"]
    assert models.scripted(Role.DESIGN_JUDGE).call_count == 1
    assert out["design_dir"] == "run/_task_design"
    for name in ("task_design.json", "accepted_review.json", "design_receipt.json"):
        assert await workspace.exists(f"run/_task_design/{name}")
    assert out["receipt"]["verdict"] == "pass"
    assert f"`{DESIGN.fingerprint()}`" in seen[0]


def _wrong_fingerprint() -> Any:
    return structured_call(DesignReview, review_payload("0" * 64))


async def test_invalid_review_is_retried_with_the_retry_feedback(
    make_context: Any, workspace: Any
) -> None:
    seen: list[str] = []
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[_wrong_fingerprint(), _wrong_fingerprint(), judge(seen=seen)],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "accepted"
    # each attempt is its own session, and the retry prompt carries the section of every failure
    assert models.scripted(Role.DESIGN_JUDGE).call_count == 3
    calls = models.scripted(Role.DESIGN_JUDGE).calls
    first, second, third = (prompt_of(messages) for messages in calls)
    assert "# Retry feedback" not in first
    assert second.count("# Retry feedback") == 1
    assert third.count("# Retry feedback") == 2
    assert "task design review fingerprint does not match design" in third
    assert "Your previous review artifact failed its contract." in second


async def test_a_review_that_stays_invalid_ends_the_loop_after_three_attempts(
    make_context: Any, workspace: Any
) -> None:
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[_wrong_fingerprint() for _ in range(3)],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "failed"
    assert out["failure"].category == "task_design_review_invalid"
    assert models.scripted(Role.DESIGN_JUDGE).call_count == 3
    assert not await workspace.exists("run/_task_design/design_receipt.json")


async def test_a_session_without_a_structured_review_counts_as_an_invalid_attempt(
    make_context: Any, workspace: Any
) -> None:
    from langgraph_agentic_scientist.testing import final

    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[final("I could not review it."), judge()],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "accepted"
    second = prompt_of(models.scripted(Role.DESIGN_JUDGE).calls[1])
    assert "# Retry feedback" in second


def _bad_sha(payload: dict[str, Any]) -> None:
    payload["canonical_task"]["sha256"] = "0" * 64


async def test_structural_repair_does_not_spend_the_semantic_budget(
    make_context: Any, workspace: Any
) -> None:
    broken = make_design(_bad_sha)
    seen: list[str] = []

    def repair(messages: Any) -> Any:
        seen.append(prompt_of(messages))
        return structured_call(TaskDesign, DESIGN)

    models = _models(
        task_designer=[structured_call(TaskDesign, broken)],
        design_structure_repairer=[repair],
        design_judge=[judge()],
    )
    # a semantic budget of zero still admits the design: the structural repair spent none of it
    out, _ = await _run(make_context, workspace, models, config=_config(design_max_repairs=0))
    assert out["status"] == "accepted"
    assert models.scripted(Role.DESIGN_STRUCTURE_REPAIRER).call_count == 1
    assert "canonical task hash does not match task.md" in seen[0]
    assert '"source": "schema_validation"' in seen[0]
    assert out["design"].fingerprint() == DESIGN.fingerprint()


async def test_structural_repairs_are_bounded_at_three(make_context: Any, workspace: Any) -> None:
    broken = make_design(_bad_sha)
    models = _models(
        task_designer=[structured_call(TaskDesign, broken)],
        design_structure_repairer=[structured_call(TaskDesign, broken) for _ in range(3)],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "failed"
    assert out["failure"].category == "task_design_structure_invalid"
    assert "after 3 structural repair attempt(s)" in out["failure"].detail
    assert models.scripted(Role.DESIGN_STRUCTURE_REPAIRER).call_count == 3
    assert models.scripted(Role.DESIGN_JUDGE).call_count == 0


async def test_the_structural_bound_is_read_from_the_recorded_configuration(
    make_context: Any, workspace: Any
) -> None:
    broken = make_design(_bad_sha)
    models = _models(
        task_designer=[structured_call(TaskDesign, broken)],
        design_structure_repairer=[structured_call(TaskDesign, broken)],
    )
    out, _ = await _run(
        make_context, workspace, models, config=_config(design_max_structure_repairs=1)
    )
    assert out["status"] == "failed"
    assert models.scripted(Role.DESIGN_STRUCTURE_REPAIRER).call_count == 1


async def test_a_missing_design_fails_the_loop(make_context: Any, workspace: Any) -> None:
    from langgraph_agentic_scientist.testing import final

    models = _models(task_designer=[final("no design")])
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "failed"
    assert out["failure"].category == "task_design_generation_failed"


async def test_patch_format_retries_do_not_spend_the_semantic_budget(
    make_context: Any, workspace: Any
) -> None:
    seen: list[str] = []
    fixed_review = judge()
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[judge({"compute_realism": WALL}), fixed_review, fixed_review],
        design_patch_repairer=[
            patcher(WALL, 7200, op="remove", seen=seen),
            patcher(WALL, 7200, op="remove", seen=seen),
            patcher(WALL, 7200, seen=seen),
        ],
    )
    # one semantic repair is allowed, and three patch attempts happen inside it
    out, _ = await _run(make_context, workspace, models, config=_config(design_max_repairs=1))
    assert out["status"] == "accepted"
    assert models.scripted(Role.DESIGN_PATCH_REPAIRER).call_count == 3
    assert "# Format retry feedback" not in seen[0]
    assert "# Format retry feedback" in seen[1]
    assert "task design patch operations must use replace" in seen[2]
    assert seen[2].count("# Format retry feedback") == 1
    assert [entry.event for entry in out["ledger"]] == ["reviewed", "full_review", "promoted"]


async def test_patch_format_attempts_are_bounded_at_three(
    make_context: Any, workspace: Any
) -> None:
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[judge({"compute_realism": WALL})],
        design_patch_repairer=[patcher(WALL, 7200, op="remove") for _ in range(3)],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "failed"
    assert out["failure"].category == "task_design_patch_invalid"
    assert models.scripted(Role.DESIGN_PATCH_REPAIRER).call_count == 3
    assert [entry.event for entry in out["ledger"]] == ["reviewed", "repair_failed"]
    assert await workspace.exists("run/_task_design/candidate_ledger.jsonl")


async def test_a_patch_outside_the_authorised_paths_is_a_format_failure(
    make_context: Any, workspace: Any
) -> None:
    seen: list[str] = []
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[judge({"compute_realism": WALL}), judge(), judge()],
        design_patch_repairer=[
            patcher(CLAIM, "Another claim.", seen=seen),
            patcher(WALL, 7200, seen=seen),
        ],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "accepted"
    assert "outside failed repair paths: /method/claim" in seen[1]


async def test_patch_then_focused_review_then_full_review_then_promotion(
    make_context: Any, workspace: Any
) -> None:
    judge_prompts: list[str] = []
    patch_prompts: list[str] = []
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[
            judge({"compute_realism": WALL}, seen=judge_prompts),
            judge(seen=judge_prompts),
            judge(seen=judge_prompts),
        ],
        design_patch_repairer=[patcher(WALL, 7200, seen=patch_prompts)],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "accepted"
    assert [entry.event for entry in out["ledger"]] == ["reviewed", "full_review", "promoted"]
    initial, focused, full = judge_prompts
    assert "# Focused repair verification" not in initial
    assert "# Focused repair verification" in focused
    assert 'Target dimension IDs: `["compute_realism"]`' in focused
    assert "# Focused repair verification" not in full
    # the repaired design is the accepted one
    assert out["design"].payload()["planes"]["search"]["budget"]["wall_clock_seconds"] == 7200
    for name in ("parent_review.json", "parent_design.json", "changed_paths.json"):
        assert await workspace.exists(f"run/_task_design/review_1_a/{name}")
    assert await workspace.exists("run/_task_design/review_2_a/task_design.json")
    assert '"mode"' not in patch_prompts[0]
    assert "compute_realism" in patch_prompts[0]


async def test_every_prompt_names_the_files_of_its_session_directory(
    make_context: Any, workspace: Any
) -> None:
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[judge({"compute_realism": WALL}), judge(), judge()],
        design_patch_repairer=[patcher(WALL, 7200)],
    )
    await _run(make_context, workspace, models)
    designer_prompt = prompt_of(models.scripted(Role.TASK_DESIGNER).calls[0])
    assert "`./task.md`" in designer_prompt and "`./task_design.json`" in designer_prompt
    assert "`20` idea cycles" in designer_prompt and "`4`" in designer_prompt
    assert "{{" not in designer_prompt
    for call in models.scripted(Role.DESIGN_JUDGE).calls:
        assert "{{" not in prompt_of(call)
    assert "{{" not in prompt_of(models.scripted(Role.DESIGN_PATCH_REPAIRER).calls[0])
    assert await workspace.read_bytes("run/_task_design/generate/task.md") == TASK_MD.encode()


async def test_the_capacity_input_reaches_the_designer(make_context: Any, workspace: Any) -> None:
    from langgraph_agentic_scientist.drafting.design_models import Capacity

    task = await _setup(workspace)
    models = _models(task_designer=[structured_call(TaskDesign, DESIGN)], design_judge=[judge()])
    ctx = make_context(models=models)
    app = build_task_design_graph(checkpointer=InMemorySaver())
    capacity = Capacity(max_idea_cycles=7, max_program_evals_per_attempt=3)
    await app.ainvoke(
        {"task": task, "capacity": capacity},
        {"configurable": {"thread_id": "cap"}},
        context=ctx,
    )
    prompt = prompt_of(models.scripted(Role.TASK_DESIGNER).calls[0])
    assert "at most `7` idea cycles" in prompt and "`3` candidate evaluations" in prompt


REVISION = "/upstream/0/revision"


async def test_delta_audit_with_indirect_regressions_advances_the_frontier(
    make_context: Any, workspace: Any
) -> None:
    """A repair that fixes the failure while the review finds one it did not cause.

    The candidate fixes ``compute_realism``; the reviews fail ``reproducibility`` citing the
    changed budget but repairable only at ``/upstream/0/revision``. The auditor confirms that
    regression, whose repair path the patch did not touch, so the candidate becomes a
    quarantined frontier, its rejected regression goes to the repair memory, and the next repair
    is authorised at the unchanged path.
    """
    repro = {"reproducibility": REVISION}
    cited = {"reproducibility": WALL}
    patch_prompts: list[str] = []
    audit_prompts: list[str] = []
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[
            judge({"compute_realism": WALL}),
            judge(repro, design_paths=cited),
            judge(repro, design_paths=cited),
            judge(),
            judge(),
        ],
        design_patch_repairer=[
            patcher(WALL, 7200, seen=patch_prompts),
            patcher(REVISION, "fedcba9876543210", seen=patch_prompts),
        ],
        design_delta_auditor=[
            auditor(
                fixed=["compute_realism"],
                regressions=[
                    {
                        "dimension": "reproducibility",
                        "design_paths": [WALL],
                        "repair_paths": [REVISION],
                        "evidence": ["The budget change moved the pin."],
                        "remediation": "Repair reproducibility.",
                    }
                ],
                parent_fingerprint=DESIGN.fingerprint(),
                seen=audit_prompts,
            )
        ],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "accepted"
    assert [entry.event for entry in out["ledger"]] == [
        "reviewed",
        "full_review",
        "delta_review",
        "frontier_advanced",
        "full_review",
        "promoted",
    ]
    assert models.scripted(Role.DESIGN_DELTA_AUDITOR).call_count == 1
    assert "Expected candidate fingerprint" in audit_prompts[0]
    # the second repair carries the previous candidate's progress and the repair memory
    second = patch_prompts[1]
    assert '"repair_memory"' in second and '"previous_candidate"' in second
    assert '"constraints"' in second
    assert out["design"].payload()["upstream"][0]["revision"] == "fedcba9876543210"
    assert out["design"].payload()["planes"]["search"]["budget"]["wall_clock_seconds"] == 7200
    for name in (
        "parent_task_design.json",
        "task_design.json",
        "parent_review.json",
        "candidate_review.json",
    ):
        assert await workspace.exists(f"run/_task_design/delta_2/{name}")


async def test_delta_audits_are_bounded_at_three_attempts_and_reject_the_candidate(
    make_context: Any, workspace: Any
) -> None:
    repro = {"reproducibility": REVISION}
    cited = {"reproducibility": WALL}
    audit_prompts: list[str] = []
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[
            judge({"compute_realism": WALL}),
            judge(repro, design_paths=cited),
            judge(repro, design_paths=cited),
        ],
        design_patch_repairer=[patcher(WALL, 7200)],
        design_delta_auditor=[
            auditor(fixed=["compute_realism"], parent_fingerprint="0" * 64, seen=audit_prompts)
            for _ in range(3)
        ],
    )
    out, _ = await _run(make_context, workspace, models, config=_config(design_max_repairs=1))
    assert models.scripted(Role.DESIGN_DELTA_AUDITOR).call_count == 3
    assert "# Retry feedback" not in audit_prompts[0]
    # the retry prompt carries only the previous attempt's failure, as upstream builds it
    assert audit_prompts[2].count("# Retry feedback") == 1
    assert "Your previous delta artifact failed its contract." in audit_prompts[1]
    assert "parent fingerprint does not match" in audit_prompts[2]
    assert out["status"] == "failed"
    assert out["failure"].category == "task_design_not_accepted"
    assert [entry.event for entry in out["ledger"]] == ["reviewed", "full_review", "rejected"]
    rejected = out["ledger"][-1].payload
    assert rejected["audit_failure"] == "delta audit invocation or artifact validation failed"
    assert rejected["improved"] is False


async def test_semantic_repairs_are_bounded_at_eight_and_focus_follows_two_idle_rounds(
    make_context: Any, workspace: Any
) -> None:
    failing = {"compute_realism": WALL}
    patch_prompts: list[str] = []
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[judge(failing) for _ in range(9)],
        design_patch_repairer=[
            patcher(WALL, 7000 + index, seen=patch_prompts) for index in range(8)
        ],
    )
    out, _ = await _run(make_context, workspace, models)
    assert out["status"] == "failed"
    assert out["failure"].category == "task_design_not_accepted"
    assert "after 8 semantic repair attempt(s)" in out["failure"].detail
    assert models.scripted(Role.DESIGN_PATCH_REPAIRER).call_count == 8
    assert models.scripted(Role.DESIGN_JUDGE).call_count == 9
    assert [entry.event for entry in out["ledger"]] == ["reviewed"] + ["rejected"] * 8
    # no focus before two non-improving rounds, focus from the third repair on
    assert all('"mode": "focused"' not in prompt for prompt in patch_prompts[:2])
    assert all('"mode": "focused"' in prompt for prompt in patch_prompts[2:])
    # rejected candidates never replace the repair base
    assert all(f"`{DESIGN.fingerprint()}`" in prompt for prompt in patch_prompts)


async def test_the_focus_threshold_is_a_configured_limit(make_context: Any, workspace: Any) -> None:
    failing = {"compute_realism": WALL}
    patch_prompts: list[str] = []
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[judge(failing) for _ in range(4)],
        design_patch_repairer=[
            patcher(WALL, 7000 + index, seen=patch_prompts) for index in range(3)
        ],
    )
    out, _ = await _run(
        make_context,
        workspace,
        models,
        config=_config(design_max_repairs=3, design_non_improving_before_focus=1),
    )
    assert out["status"] == "failed"
    assert '"mode": "focused"' not in patch_prompts[0]
    assert all('"mode": "focused"' in prompt for prompt in patch_prompts[1:])


async def test_unchanged_full_review_failures_are_rebound_to_the_retained_parent(
    make_context: Any, workspace: Any
) -> None:
    patch_prompts: list[str] = []
    latent = {"publication_scope": "/publication/required_evidence"}
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[
            judge({"compute_realism": WALL}),
            judge(),
            judge(latent),
            judge(),
        ],
        design_patch_repairer=[
            patcher(WALL, 7200, seen=patch_prompts),
            patcher(WALL, 7300, seen=patch_prompts),
        ],
        design_delta_auditor=[
            auditor(
                fixed=["compute_realism"],
                regressions=[
                    {
                        "dimension": "reproducibility",
                        "design_paths": [WALL],
                        "repair_paths": [CLAIM],
                        "evidence": ["Unrelated drift."],
                        "remediation": "Repair reproducibility.",
                    }
                ],
                parent_fingerprint=DESIGN.fingerprint(),
            )
        ],
    )
    out, _ = await _run(make_context, workspace, models, config=_config(design_max_repairs=1))
    events = [entry.event for entry in out["ledger"]]
    assert events == [
        "reviewed",
        "full_review",
        "delta_review",
        "rejected",
        "latent_findings_rebound",
    ]
    rebound = out["ledger"][-1].payload
    assert rebound["design_fingerprint"] == DESIGN.fingerprint()
    failed = {row["id"] for row in rebound["dimensions"] if row["verdict"] == "fail"}
    assert failed == {"compute_realism", "publication_scope"}
    assert out["failure"].category == "task_design_not_accepted"


async def test_hooks_run_around_the_roles_and_a_completed_step_is_not_repeated_on_resume(
    make_context: Any, workspace: Any
) -> None:
    from langgraph_agentic_scientist import Hooks

    crashes = {"left": 1}
    calls: list[str] = []

    async def before_review(state: Any, runtime: Any) -> dict[str, Any]:
        calls.append("before_review")
        return {"hook_outputs": {"seen": True}}

    async def after_review(state: Any, runtime: Any) -> None:
        calls.append("after_review")
        if crashes["left"]:
            crashes["left"] -= 1
            raise RuntimeError("the process died after the judge answered")

    async def before_generate(state: Any, runtime: Any) -> None:
        calls.append("before_generate")

    hooks = Hooks(
        before={
            "drafting.design_review": [before_review],
            "drafting.design_generate": [before_generate],
        },
        after={"drafting.design_review": [after_review]},
    )
    models = _models(task_designer=[structured_call(TaskDesign, DESIGN)], design_judge=[judge()])
    task = await _setup(workspace)
    ctx = make_context(models=models, hooks=hooks)
    app = build_task_design_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "resume"}}
    with pytest.raises(RuntimeError, match="the process died"):
        await app.ainvoke({"task": task}, config, context=ctx)
    out = await app.ainvoke(None, config, context=ctx)
    assert out["status"] == "accepted"
    # no model call and no completed hook ran twice; only the hook that crashed ran again
    assert models.scripted(Role.TASK_DESIGNER).call_count == 1
    assert models.scripted(Role.DESIGN_JUDGE).call_count == 1
    assert calls == ["before_generate", "before_review", "after_review", "after_review"]


async def test_the_graph_has_hook_nodes_at_its_role_steps() -> None:
    app = build_task_design_graph()
    nodes = set(app.get_graph().nodes)
    for step in (
        "design_generate",
        "design_structure_repair",
        "design_review",
        "design_patch_repair",
        "design_delta_audit",
        "design_publish",
    ):
        assert f"drafting.{step}.before" in nodes and f"drafting.{step}.after" in nodes
    assert "record_run_config" in nodes


async def test_a_mounted_graph_inherits_the_checkpointer_and_the_recorded_configuration(
    make_context: Any, workspace: Any
) -> None:
    parent: StateGraph[Any, Any, Any, Any] = StateGraph(Parent, context_schema=ScientistContext)
    parent.add_node("design", build_task_design_graph())
    parent.add_edge(START, "design")
    parent.add_edge("design", END)
    models = _models(task_designer=[structured_call(TaskDesign, DESIGN)], design_judge=[judge()])
    task = await _setup(workspace)
    ctx = make_context(models=models)
    app = parent.compile(checkpointer=InMemorySaver())
    out = await app.ainvoke({"task": task}, {"configurable": {"thread_id": "mount"}}, context=ctx)
    assert out["status"] == "accepted"


def test_no_quorum_and_no_adjudicator_exist() -> None:
    from langgraph_agentic_scientist import PromptRegistry

    keys = PromptRegistry.default().keys()
    assert not [key for key in keys if "adjudicat" in key]
    assert not [role for role in Role if "adjudicat" in role.value]
    design_prompts = sorted(key for key in keys if key.startswith("drafting/") and "design" in key)
    assert design_prompts == [
        "drafting/audit_task_design_delta",
        "drafting/design_task",
        "drafting/repair_task_design",
        "drafting/repair_task_design_structure",
        "drafting/validate_task_design",
    ]


def test_the_default_design_prompts_are_the_default_registry_entries() -> None:
    assert DraftingConfig().design_max_repairs == 8
    assert DraftingConfig().design_max_structure_repairs == 3
    assert DraftingConfig().design_review_attempts == 3
    assert DraftingConfig().design_format_attempts == 3
    assert DraftingConfig().design_delta_attempts == 3
    assert DraftingConfig().design_non_improving_before_focus == 2
    assert design_payload()["candidate"]["interface"] == "python_rpc_v1:handle"


async def test_a_crash_after_a_retried_review_does_not_repeat_either_session(
    make_context: Any, workspace: Any
) -> None:
    """Each review attempt is its own node execution, so each has its own checkpoint."""
    from langgraph_agentic_scientist import Hooks

    reviews = {"seen": 0}

    async def crash_after_second_review(state: Any, runtime: Any) -> None:
        reviews["seen"] += 1
        if reviews["seen"] == 2:
            raise RuntimeError("the process died after the second review")

    hooks = Hooks(after={"drafting.design_review": [crash_after_second_review]})
    models = _models(
        task_designer=[structured_call(TaskDesign, DESIGN)],
        design_judge=[_wrong_fingerprint(), judge()],
    )
    task = await _setup(workspace)
    ctx = make_context(models=models, hooks=hooks)
    app = build_task_design_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "retry-resume"}}
    with pytest.raises(RuntimeError, match="second review"):
        await app.ainvoke({"task": task}, config, context=ctx)
    assert models.scripted(Role.DESIGN_JUDGE).call_count == 2
    out = await app.ainvoke(None, config, context=ctx)
    assert out["status"] == "accepted"
    assert models.scripted(Role.TASK_DESIGNER).call_count == 1
    assert models.scripted(Role.DESIGN_JUDGE).call_count == 2


def test_the_role_nodes_carry_the_outer_retry_line_for_transient_provider_errors() -> None:
    app = build_task_design_graph()
    nodes = app.builder.nodes
    for step in (
        "design_generate",
        "design_structure_repair",
        "design_review",
        "design_patch_repair",
        "design_delta_audit",
    ):
        policy = nodes[step].retry_policy
        assert policy.max_attempts == 4 and policy.initial_interval == 15.0
    assert not nodes["check_review"].retry_policy
