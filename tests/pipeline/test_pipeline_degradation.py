"""Upstream's degradation rules: what a failed, skipped or degraded phase does to the run."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from pipeline_stubs import CONTRACT_FIXTURE, Ledger, failed_investigation, final_result
from testkit import thread

from langgraph_agentic_scientist import (
    DraftingResult,
    InvestigationResult,
    LocalWorkspace,
    PaperResult,
    PipelineConfigurationError,
    PosthocResult,
    ReviewResult,
    ScientistConfig,
    VariantsResult,
)
from langgraph_agentic_scientist.config import ALL_PHASES
from langgraph_agentic_scientist.core.results import AnalysisResult
from langgraph_agentic_scientist.core.state import Failure
from langgraph_agentic_scientist.pipeline import prepare_task_snapshot

ALL = ScientistConfig(phases=ALL_PHASES, fail_fast=False)


def _statuses(out: Mapping[str, Any]) -> dict[str, str]:
    return {name: status.status for name, status in out["phase_status"].items()}


async def test_failed_ablations_degrade_the_run_and_the_paper_is_still_written(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = {"ablations": VariantsResult(status="failed", failure=Failure(category="x"))}
    out = await make_app(results).ainvoke(
        {"task_dir": circpack}, thread("var"), context=make_context(config=ALL)
    )
    statuses = _statuses(out)
    assert statuses["ablations"] == "degraded"
    assert statuses["posthoc"] == "degraded"  # upstream: selected ablations did not succeed
    assert out["phase_status"]["posthoc"].reason == "selected ablations did not succeed"
    assert statuses["paper"] == "completed" and out["paper"].status == "complete"
    assert out["status"] == "completed_degraded" and not out["failures"]
    assert "posthoc" not in ledger.calls
    assert "posthoc" not in ledger.last_input("paper")  # paper continues without post-hoc


async def test_posthoc_without_a_valid_analysis_is_degraded(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    empty = PosthocResult(
        status="completed",
        analyses=[AnalysisResult(analysis_id="a", status="failed", attempts=3)],
    )
    out = await make_app({"posthoc": empty}).ainvoke(
        {"task_dir": circpack}, thread("ph"), context=make_context(config=ALL)
    )
    assert out["phase_status"]["posthoc"].status == "degraded"
    assert out["phase_status"]["posthoc"].reason == "no_valid_analyses"
    assert out["status"] == "completed_degraded"
    assert "posthoc" not in ledger.last_input("paper")


async def test_a_failed_review_degrades_and_skips_the_revision(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = {"review": ReviewResult(status="failed", failure=Failure(category="reviewer"))}
    out = await make_app(results).ainvoke(
        {"task_dir": circpack}, thread("rev"), context=make_context(config=ALL)
    )
    statuses = _statuses(out)
    assert statuses["review"] == "degraded"
    assert statuses["revision"] == "skipped"
    assert out["phase_status"]["revision"].reason == "dependency review failed"
    assert out["paper"].status == "complete"  # a failed review never discards the paper
    assert out["status"] == "completed_degraded"
    assert "revision" not in ledger.calls


async def test_review_without_a_pdf_degrades_without_calling_the_phase(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = {"paper": PaperResult(status="complete", paper_dir="paper")}
    out = await make_app(results).ainvoke(
        {"task_dir": circpack}, thread("nopdf"), context=make_context(config=ALL)
    )
    assert _statuses(out)["review"] == "degraded" and "review" not in ledger.calls


async def test_an_incomplete_paper_fails_the_run_and_skips_review_and_revision(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = {
        "paper": PaperResult(status="incomplete", incomplete_reason="paper validation failed")
    }
    out = await make_app(results).ainvoke(
        {"task_dir": circpack}, thread("inc"), context=make_context(config=ALL)
    )
    assert _statuses(out)["paper"] == "failed"
    assert out["phase_status"]["paper"].reason == "paper validation failed"
    assert _statuses(out)["review"] == "skipped" and _statuses(out)["revision"] == "skipped"
    assert out["status"] == "failed" and out["failures"][0].category == "paper_failed"
    assert out["stop_reason"] is None  # fail_fast is off: the run reached its end


async def test_fail_fast_stops_at_the_first_failed_phase(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    ctx = make_context(config=ScientistConfig(phases=ALL_PHASES, fail_fast=True))
    out = await make_app({"investigation": failed_investigation()}).ainvoke(
        {"task_dir": circpack}, thread("ff"), context=ctx
    )
    assert out["status"] == "failed" and out["stop_reason"] == "campaign_failed"
    assert ledger.order == ["investigation"]
    assert out["phase_status"]["ablations"].reason == "run stopped: campaign_failed"
    assert _statuses(out)["paper"] == "skipped"


async def test_without_fail_fast_the_dependants_of_a_failed_phase_are_skipped(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    out = await make_app({"investigation": failed_investigation()}).ainvoke(
        {"task_dir": circpack}, thread("nff"), context=make_context(config=ALL)
    )
    statuses = _statuses(out)
    assert statuses["investigation"] == "failed"
    for phase in ("ablations", "final_evaluation", "paper", "review", "revision"):
        assert statuses[phase] == "skipped"
    for phase in ("ablations", "final_evaluation", "paper"):
        assert out["phase_status"][phase].reason == "dependency investigation failed"
    # upstream checks the ablations first: post-hoc analysis is degraded, not skipped
    assert statuses["posthoc"] == "degraded"
    assert out["stop_reason"] is None and ledger.order == ["investigation"]


@pytest.mark.parametrize(
    ("category", "scope"),
    [
        ("evaluator_output_contract_invalid", "task"),
        ("agent_log_persistence_failed", "infrastructure"),
    ],
)
async def test_the_terminal_host_failures_stop_whatever_fail_fast_says(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
    category: str,
    scope: str,
) -> None:
    ctx = make_context(config=ScientistConfig(fail_fast=False))
    out = await make_app({"investigation": failed_investigation(category, scope)}).ainvoke(
        {"task_dir": circpack}, thread("term"), context=ctx
    )
    assert out["stop_reason"] == category and out["status"] == "failed"
    assert out["failures"][0].scope == scope
    assert ledger.order == ["investigation"]


async def test_no_winner_with_best_available_freezes_a_fallback_for_the_downstream_phases(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = {"investigation": InvestigationResult(status="completed_no_winner")}
    out = await make_app(results).ainvoke(
        {"task_dir": circpack}, thread("nw1"), context=make_context(config=ALL)
    )
    assert ledger.order[:3] == ["investigation", "downstream_selection", "ablations"]
    assert out["downstream"].status == "selected" and out["outcome"] == "completed_no_winner"
    assert ledger.last_input("ablations")["downstream"].source == "candidate"
    assert ledger.last_input("paper")["downstream"] is not None
    assert out["status"] == "completed"


async def test_no_winner_with_skip_skips_the_downstream_phases(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = {"investigation": InvestigationResult(status="completed_no_winner")}
    ctx = make_context(
        config=ScientistConfig(phases=ALL_PHASES, no_winner_policy="skip", fail_fast=False)
    )
    out = await make_app(results).ainvoke({"task_dir": circpack}, thread("nw2"), context=ctx)
    assert ledger.order == ["investigation"]
    statuses = _statuses(out)
    assert statuses["downstream_selection"] == "skipped"
    for phase in ("ablations", "final_evaluation", "posthoc", "paper", "review", "revision"):
        assert statuses[phase] == "skipped"
        assert out["phase_status"][phase].reason == (
            "investigation completed without a qualifying winner"
        )
    assert out["outcome"] == "completed_no_winner" and out["status"] == "completed"


async def test_no_winner_without_a_selectable_program_skips_the_downstream_phases(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    from langgraph_agentic_scientist import DownstreamSelection

    results = {
        "investigation": InvestigationResult(status="completed_no_winner"),
        "downstream_selection": DownstreamSelection(status="skipped", reason="no valid score"),
    }
    out = await make_app(results).ainvoke(
        {"task_dir": circpack}, thread("nw3"), context=make_context(config=ALL)
    )
    assert ledger.order == ["investigation", "downstream_selection"]
    assert _statuses(out)["paper"] == "skipped"


async def test_a_final_evaluation_that_fails_degrades_when_the_task_does_not_require_it(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    out = await make_app({"final_evaluation": None}).ainvoke(
        {"task_dir": circpack}, thread("fin1"), context=make_context(config=ALL)
    )
    assert _statuses(out)["final_evaluation"] == "degraded"
    assert out["paper"].status == "complete" and "final" not in ledger.last_input("paper")
    assert out["status"] == "completed_degraded"


async def test_an_unqualified_final_evaluation_supports_a_paper_only_when_the_policy_allows(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = {"final_evaluation": final_result("not_qualified")}
    out = await make_app(results).ainvoke(
        {"task_dir": circpack}, thread("nq1"), context=make_context(config=ALL)
    )
    assert out["outcome"] == "not_qualified"
    assert "final" not in ledger.last_input("paper")  # qualified_only: not a paper input
    ledger.calls.clear()
    ledger.inputs.clear()
    allow = ScientistConfig(
        phases=ALL_PHASES, final_outcome_policy="allow_not_qualified", fail_fast=False
    )
    out = await make_app(results).ainvoke(
        {"task_dir": circpack}, thread("nq2"), context=make_context(config=allow)
    )
    assert ledger.last_input("paper")["final"].outcome == "not_qualified"
    assert out["outcome"] == "not_qualified" and out["status"] == "completed"


# --- a task whose contract requires a final evaluation ----------------------------------------


async def test_a_contracted_paper_without_the_final_phase_is_a_configuration_error(
    make_app: Callable[..., Any], make_context: Callable[..., Any], contracted: str
) -> None:
    ctx = make_context(config=ScientistConfig(phases=("investigation", "paper"), fail_fast=False))
    with pytest.raises(PipelineConfigurationError, match="contracted paper phase requires final"):
        await make_app().ainvoke({"task_dir": contracted}, thread("c1"), context=ctx)


async def test_a_contract_forces_harness_evaluated_attempts(
    make_app: Callable[..., Any], make_context: Callable[..., Any], contracted: str, ledger: Ledger
) -> None:
    out = await make_app().ainvoke(
        {"task_dir": contracted}, thread("c2"), context=make_context(config=ALL)
    )
    assert out["task"].harness_evaluated and out["task"].contract is not None
    assert out["status"] == "completed"  # the final evaluation ran and qualified
    assert ledger.last_input("paper")["final"].outcome == "qualified"


async def test_a_required_final_evaluation_that_fails_blocks_the_paper(
    make_app: Callable[..., Any], make_context: Callable[..., Any], contracted: str, ledger: Ledger
) -> None:
    out = await make_app({"final_evaluation": None}).ainvoke(
        {"task_dir": contracted}, thread("c3"), context=make_context(config=ALL)
    )
    statuses = _statuses(out)
    assert statuses["final_evaluation"] == "failed"
    assert statuses["paper"] == "skipped" and statuses["review"] == "skipped"
    assert out["phase_status"]["paper"].reason == "required final evaluation did not succeed"
    assert out["status"] == "failed"
    assert [item.category for item in out["failures"]].count("final_evaluation_failed") == 1
    assert "paper" not in ledger.calls


async def test_a_required_final_evaluation_that_did_not_qualify_needs_a_fallback_or_the_policy(
    make_app: Callable[..., Any], make_context: Callable[..., Any], contracted: str, ledger: Ledger
) -> None:
    results = {"final_evaluation": final_result("not_qualified")}
    out = await make_app(results).ainvoke(
        {"task_dir": contracted}, thread("c4"), context=make_context(config=ALL)
    )
    assert out["phase_status"]["paper"].reason == "required final evaluation did not qualify"
    assert out["status"] == "completed" and not out["failures"]  # not a failure: skipped
    assert out["outcome"] == "not_qualified" and "paper" not in ledger.calls
    allow = ScientistConfig(
        phases=ALL_PHASES, final_outcome_policy="allow_not_qualified", fail_fast=False
    )
    await make_app(results).ainvoke(
        {"task_dir": contracted}, thread("c5"), context=make_context(config=allow)
    )
    assert ledger.last_input("paper")["final"].outcome == "not_qualified"


async def test_a_required_final_evaluation_that_failed_stops_with_fail_fast(
    make_app: Callable[..., Any], make_context: Callable[..., Any], contracted: str, ledger: Ledger
) -> None:
    ctx = make_context(config=ScientistConfig(phases=ALL_PHASES, fail_fast=True))
    out = await make_app({"final_evaluation": None}).ainvoke(
        {"task_dir": contracted}, thread("c6"), context=ctx
    )
    assert out["stop_reason"] == "final_evaluation_failed"
    assert "posthoc" not in ledger.calls


async def test_a_required_final_evaluation_of_an_earlier_run_counts_only_under_the_policy(
    make_app: Callable[..., Any], make_context: Callable[..., Any], contracted: str, ledger: Ledger
) -> None:
    earlier = {
        "task_dir": contracted,
        "investigation": InvestigationResult(status="completed"),
        "paper": PaperResult(status="complete", paper_dir="paper", pdf_path="paper/paper.pdf"),
    }
    strict = make_context(config=ScientistConfig(phases=("paper", "review"), fail_fast=False))
    with pytest.raises(PipelineConfigurationError, match="requires final phase"):
        await make_app().ainvoke({**earlier, "final": final_result()}, thread("c7"), context=strict)
    allow = ScientistConfig(
        phases=("paper", "review"), final_outcome_policy="allow_not_qualified", fail_fast=False
    )
    out = await make_app().ainvoke(
        {**earlier, "final": final_result()}, thread("c8"), context=make_context(config=allow)
    )
    assert ledger.order == ["paper", "review"] and out["status"] == "completed"
    assert _statuses(out)["investigation"] == "skipped"
    ledger.calls.clear()
    out = await make_app().ainvoke(earlier, thread("c9"), context=make_context(config=allow))
    assert "paper" not in ledger.calls and out["status"] == "failed"  # no final evaluation at all


# --- drafting ---------------------------------------------------------------------------------


def _drafting_publishes_a_contract(workspace: LocalWorkspace) -> Callable[[Any], DraftingResult]:
    """A drafting stub that replaces the snapshot's scaffold with a contracted task."""

    def publish(given: Mapping[str, Any]) -> DraftingResult:
        task = Path(workspace.root) / "task"
        keep = (task / ".agentic_task_snapshot.json").read_bytes()
        shutil.rmtree(task)
        shutil.copytree(CONTRACT_FIXTURE, task, ignore=shutil.ignore_patterns("__pycache__"))
        (task / "capture.py").write_text("# capture\n")
        (task / ".agentic_task_snapshot.json").write_bytes(keep)
        return DraftingResult(status="published", task=given.get("task"))

    return publish


def _bare_task(workspace: LocalWorkspace) -> str:
    source = Path(workspace.root) / "inputs" / "bare"
    source.mkdir(parents=True)
    (source / "task.md").write_text("# Task\n\n## Goal\nDo a thing.\n\n## Rubrics\n- done\n")
    return "inputs/bare"


async def test_drafting_runs_only_when_the_scaffold_is_missing_and_the_task_is_resealed(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    workspace: LocalWorkspace,
    ledger: Ledger,
) -> None:
    bare = _bare_task(workspace)
    ctx = make_context(
        config=ScientistConfig(phases=("task_drafting", "investigation"), fail_fast=False)
    )
    out = await make_app({"task_drafting": _drafting_publishes_a_contract(workspace)}).ainvoke(
        {"task_dir": bare}, thread("d1"), context=ctx
    )
    assert ledger.order == ["task_drafting", "investigation"]
    assert ledger.last_input("task_drafting")["capacity"] == {
        "max_idea_cycles": 20,
        "max_program_evals_per_attempt": 4,
    }
    task = out["task"]  # refreshed from the published scaffold
    assert task.contract is not None and task.harness_evaluated and task.seed_program is not None
    assert ledger.last_input("investigation")["task"] == task
    await prepare_task_snapshot(workspace, bare)  # the resealed snapshot verifies


async def test_a_drafted_task_that_needs_a_final_evaluation_that_is_not_selected_stops_the_run(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    workspace: LocalWorkspace,
    ledger: Ledger,
) -> None:
    bare = _bare_task(workspace)
    ctx = make_context(
        config=ScientistConfig(phases=("task_drafting", "investigation", "paper"), fail_fast=False)
    )
    out = await make_app({"task_drafting": _drafting_publishes_a_contract(workspace)}).ainvoke(
        {"task_dir": bare}, thread("d2"), context=ctx
    )
    assert out["stop_reason"] == "drafted_dependencies_invalid" and out["status"] == "failed"
    assert ledger.order == ["task_drafting"]


async def test_a_failed_drafting_skips_everything_after_it(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    workspace: LocalWorkspace,
    ledger: Ledger,
) -> None:
    bare = _bare_task(workspace)
    failed = DraftingResult(status="failed", failure=Failure(category="design_failed"))
    ctx = make_context(
        config=ScientistConfig(phases=("task_drafting", "investigation"), fail_fast=False)
    )
    out = await make_app({"task_drafting": failed}).ainvoke(
        {"task_dir": bare}, thread("d3"), context=ctx
    )
    statuses = _statuses(out)
    assert statuses["task_drafting"] == "failed" and statuses["investigation"] == "skipped"
    assert out["phase_status"]["investigation"].reason == "dependency task_drafting failed"
    assert out["status"] == "failed" and ledger.order == ["task_drafting"]
