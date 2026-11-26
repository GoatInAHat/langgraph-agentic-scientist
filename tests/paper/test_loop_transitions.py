"""One test per row of both transition tables (``docs/adaptations/paper_loop.md``).

Rows ``F01``-``F26`` are the fresh loop of ``paper.sh`` (lines 3825-4184); rows ``E01``-``E25``
the edit loop of ``paper_edit.sh`` (lines 4252-4522). The decisions are pure functions of
:mod:`langgraph_agentic_scientist.paper.graph`; the rows whose effect is a host step (rendering,
validation, the audit, restoring references) are tested on that step.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_loop_support import (
    SCOPE_AUDIT,
    audit_pass,
    fast_config,
    models,
    paper_config,
    setup_task,
    strategy,
)

from langgraph_agentic_scientist.core.results import ValidationReport
from langgraph_agentic_scientist.core.runconfig import build_run_config
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper.graph import (
    _apply,
    _default_figures,
    _default_validation,
    decide_after_audit,
    decide_after_postdraft,
    decide_after_review,
    decide_after_validate,
    decide_after_write,
    decide_figure_quality,
    decide_loop,
    decide_postdraft,
    decide_review_check,
    decide_select_stage,
)
from langgraph_agentic_scientist.paper.models import ConceptFigureState, FigureQualityState
from langgraph_agentic_scientist.paper.workspace import restore_revision_references
from langgraph_agentic_scientist.testing import FakeFigureRenderer, FakePaperToolchain


class _Runtime:
    def __init__(self, context: Any) -> None:
        self.context = context


# --- shared rows: stage selection ------------------------------------------------------------


@pytest.mark.parametrize("mode", ["fresh", "edit"])
@pytest.mark.parametrize(
    ("row", "pending", "round_", "stage"),
    [
        ("01", True, 2, "figure_quality_repair"),
        ("02", False, 0, "draft"),
        ("03", False, 2, "manuscript_repair"),
    ],
)
def test_rows_01_03_select_stage(
    mode: str, row: str, pending: bool, round_: int, stage: str
) -> None:
    decision = decide_select_stage(fq_pending=pending, repair_round=round_)
    assert decision.next == "write"
    assert decision.updates["stage"] == stage
    assert decision.updates["figure_quality_pending"] is False
    assert decision.updates["reverted"] is False


# --- fresh table --------------------------------------------------------------------------------


def test_F04_write_ok() -> None:
    d = decide_after_write(mode="fresh", stage="draft", ok=True, repair_round=0, max_rounds=10)
    assert (d.next, d.updates["stage_rc"], d.actions) == ("postdraft_gate", "ok", ())


def test_F05_failed_figure_quality_stage_restores() -> None:
    d = decide_after_write(
        mode="fresh", stage="figure_quality_repair", ok=False, repair_round=10, max_rounds=10
    )
    assert d.next == "postdraft_gate" and d.actions == ("restore_fq",) and d.updates["reverted"]


def test_F06_failed_write_after_last_round_is_incomplete() -> None:
    d = decide_after_write(
        mode="fresh", stage="manuscript_repair", ok=False, repair_round=10, max_rounds=10
    )
    assert d.next == "outcome"
    assert d.incomplete == "exhausted repair rounds after 10 attempts; see validation_feedback.json"


def test_F07_failed_write_starts_the_next_round() -> None:
    d = decide_after_write(mode="fresh", stage="draft", ok=False, repair_round=0, max_rounds=10)
    assert d.next == "select_stage" and d.updates["repair_round"] == 1 and d.incomplete is None


def test_F08_postdraft_enrichment_once_and_failure_ends_the_phase() -> None:
    assert decide_postdraft(repair_round=0, enrichment_done=False, skip=False).next == "postdraft"
    assert decide_postdraft(repair_round=0, enrichment_done=True, skip=False).next == "figures"
    assert decide_postdraft(repair_round=1, enrichment_done=False, skip=False).next == "figures"
    failed = decide_after_postdraft(status="failed", detail="gap rounds exhausted")
    assert failed.next == "outcome" and failed.failed == "gap rounds exhausted"
    assert decide_after_postdraft(status="no_gaps", detail="").next == "figures"


async def test_F09_figures_render_unless_reverted(make_context: Any, workspace: Any) -> None:
    renderer = FakeFigureRenderer()
    ctx = make_context(figure_renderer=renderer)
    paper_dir = "run/paper/p"
    spec = {
        "figure_id": "overview",
        "figure_type": "diagram",
        "preferred_generator": "generate-concept-figure",
        "required": True,
        "purpose": "Show the method.",
        "preferred_span": "full_width",
    }
    await workspace.write_bytes(
        f"{paper_dir}/paper_strategy.json", json.dumps({"figure_specs": [spec]}).encode()
    )
    state = {"paper_dir": paper_dir, "concept": ConceptFigureState(available=True)}
    assert await _default_figures({**state, "reverted": True}, _Runtime(ctx)) == {}
    update = await _default_figures(state, _Runtime(ctx))
    assert update["concept"].attempted and not update["concept"].failure_active
    assert renderer.requests[0].instructions == "Show the method."
    assert renderer.requests[0].metadata["placement"] == "full-width"
    assert await workspace.exists(f"{paper_dir}/figures/overview.png")
    await workspace.write_bytes(f"{paper_dir}/figures/overview.png", b"placeholder")
    again = await _default_figures({**state, "concept": update["concept"]}, _Runtime(ctx))
    assert (
        await workspace.read_bytes(f"{paper_dir}/figures/overview.png")
    ) == FakeFigureRenderer.PNG
    assert len(renderer.requests) == 1
    assert any(event.kind == "concept_figure_reasserted" for event in again["events"])


async def test_F10_validation_writes_report_feedback_and_fingerprint(
    make_context: Any, workspace: Any
) -> None:
    toolchain = FakePaperToolchain([ValidationReport(passed=False, errors=["missing section"])])
    ctx = make_context(paper_toolchain=toolchain, config=fast_config(paper=paper_config()))
    paper_dir = "run/paper/p"
    await workspace.write_bytes(f"{paper_dir}/paper.tex", b"tex")
    state = {"paper_dir": paper_dir, "run_config": build_run_config(ctx), "repair_round": 2}
    update = await _default_validation(state, _Runtime(ctx))
    assert update["validation_passed"] is False
    assert update["validation_fingerprint"]
    feedback = json.loads(
        (await workspace.read_bytes(f"{paper_dir}/validation_feedback.json")).decode()
    )
    assert feedback["hard_errors"] == ["missing section"]
    assert await workspace.exists(f"{paper_dir}/validation_report.json")
    assert await workspace.exists(f"{paper_dir}/paper_diagnostics/validation/round_0002")


def test_F11_figure_quality_validation_regression_restores_and_revalidates() -> None:
    d = decide_after_validate(
        mode="fresh",
        stage="figure_quality_repair",
        passed=False,
        reverted=False,
        fingerprint="f",
        previous=None,
    )
    assert d.next == "validate" and d.actions == ("restore_fq",) and d.updates["reverted"]


def test_F12_fingerprint_repetition_and_clearing() -> None:
    repeated = decide_after_validate(
        mode="fresh",
        stage="manuscript_repair",
        passed=False,
        reverted=False,
        fingerprint="abc",
        previous="abc",
    )
    assert repeated.next == "audit" and repeated.updates["failure_repeated"] is True
    fresh = decide_after_validate(
        mode="fresh", stage="draft", passed=False, reverted=False, fingerprint="abc", previous=None
    )
    assert fresh.updates == {"previous_failure_fingerprint": "abc", "failure_repeated": False}
    cleared = decide_after_validate(
        mode="fresh", stage="draft", passed=True, reverted=False, fingerprint=None, previous="abc"
    )
    assert cleared.updates["previous_failure_fingerprint"] is None


async def test_F13_audit_runs_read_only_on_a_copy(make_context: Any, workspace: Any) -> None:
    from langgraph_agentic_scientist.paper.audit import build_audit_graph

    scripted = models({Role.EVIDENCE_AUDITOR: [audit_pass()]})
    ctx = make_context(models=scripted, config=fast_config(paper=paper_config()))
    task = await setup_task(workspace)
    paper_dir = "run/paper/p"
    await workspace.write_bytes(f"{paper_dir}/paper.tex", b"\\documentclass{article}")
    await workspace.write_bytes(f"{paper_dir}/paper_data.json", b"{}")
    out = await build_audit_graph().ainvoke(
        {"run_config": build_run_config(ctx), "task": task, "paper_dir": paper_dir},
        context=ctx,
    )
    assert out["audit"].outcome == "pass"
    assert not await workspace.exists(
        "run/paper/.paper_evidence_audit_workspaces/p-round1-attempt1"
    )
    backends = (
        [type(b).__name__ for b in ctx.workspace.backends]
        if hasattr(ctx.workspace, "backends")
        else []
    )
    assert all(name == "ReadOnlyBackend" for name in backends)


@pytest.mark.parametrize(
    ("row", "outcome", "actions", "effective"),
    [
        ("14", "findings", ("restore_fq",), "pass"),
        ("14b", "operational_failure", ("restore_fq",), "pass"),
        ("15", "pass", ("discard_fq",), None),
    ],
)
def test_F14_F15_figure_quality_stage_and_the_audit(
    row: str, outcome: str, actions: tuple[str, ...], effective: str | None
) -> None:
    d = decide_after_audit(
        mode="fresh",
        stage="figure_quality_repair",
        reverted=False,
        outcome=outcome,
        repair_round=3,
        max_rounds=10,
    )
    assert d.actions == actions and d.next == "review_check"
    assert d.updates.get("audit_effective") == effective


def test_F16_findings_after_the_last_round_are_incomplete() -> None:
    d = decide_after_audit(
        mode="fresh",
        stage="manuscript_repair",
        reverted=False,
        outcome="findings",
        repair_round=10,
        max_rounds=10,
    )
    assert d.next == "outcome" and d.incomplete.startswith(
        "paper evidence audit reported revision findings after exhausting repair rounds"
    )


def test_F17_findings_start_a_repair_round() -> None:
    d = decide_after_audit(
        mode="fresh",
        stage="draft",
        reverted=False,
        outcome="findings",
        repair_round=0,
        max_rounds=10,
    )
    assert d.next == "select_stage" and d.updates["repair_round"] == 1
    assert d.actions == ("authorize_concept", "save_audit_pdf")


def test_F18_operational_audit_failure_is_incomplete() -> None:
    d = decide_after_audit(
        mode="fresh",
        stage="draft",
        reverted=False,
        outcome="operational_failure",
        repair_round=0,
        max_rounds=10,
    )
    assert d.next == "outcome" and d.incomplete.startswith(
        "paper evidence audit failed operationally"
    )


def test_F19_contract_exhaustion_is_not_gating() -> None:
    d = decide_after_audit(
        mode="fresh",
        stage="draft",
        reverted=False,
        outcome="contract_exhausted",
        repair_round=0,
        max_rounds=10,
    )
    assert d.next == "review_check" and d.incomplete is None


def test_F20_review_gate() -> None:
    assert (
        decide_review_check(mode="fresh", review_round=0, review_rounds=1, passed=False).next
        == "review_gate"
    )
    assert (
        decide_review_check(mode="fresh", review_round=1, review_rounds=1, passed=True).next
        == "figure_quality_gate"
    )
    failed = decide_after_review(
        mode="fresh", outcome="stage_failed", review_round=0, repair_round=0, max_rounds=10
    )
    assert failed.next == "outcome" and failed.incomplete.startswith("paper review stage failed")
    exhausted = decide_after_review(
        mode="fresh", outcome="completed", review_round=0, repair_round=10, max_rounds=10
    )
    assert exhausted.next == "outcome" and exhausted.updates["review_round"] == 1
    assert exhausted.incomplete.startswith("exhausted repair rounds after 10 attempts")
    repair = decide_after_review(
        mode="fresh", outcome="contract_exhausted", review_round=0, repair_round=2, max_rounds=10
    )
    assert repair.next == "select_stage" and repair.updates == {
        "review_round": 1,
        "repair_round": 3,
    }
    assert repair.actions == ("save_review_pdf",)


class _NoSnapshotWorkspace:
    async def snapshot(self, path: str) -> Any:
        raise OSError("snapshots are not supported")


async def test_F21_figure_quality_repair_needs_a_snapshot(
    make_context: Any, workspace: Any
) -> None:
    d = decide_figure_quality(
        passed=True,
        actionable=True,
        attempted=False,
        warning_reported=False,
        repair_round=1,
        max_rounds=10,
    )
    assert d.next == "select_stage" and d.actions == ("snapshot_fq",)
    ctx = make_context(config=fast_config(paper=paper_config()))
    paper_dir = "run/paper/p"
    await workspace.write_bytes(f"{paper_dir}/paper.tex", b"tex")
    state = {"paper_dir": paper_dir, "repair_round": 1, "figure_quality": FigureQualityState()}
    ok = await _apply(d, state, _Runtime(ctx))
    assert ok["next"] == "select_stage" and ok["repair_round"] == 2
    assert ok["figure_quality"].pending and ok["figure_quality"].snapshot is not None
    failing = make_context(workspace=_NoSnapshotWorkspace())
    no = await _apply(d, state, _Runtime(failing))
    assert no["next"] == "loop_decision" and no["figure_quality"].warning_reported
    assert not no["figure_quality"].pending and "repair_round" not in no


def test_F22_remaining_warnings_are_reported_once() -> None:
    first = decide_figure_quality(
        passed=True,
        actionable=True,
        attempted=True,
        warning_reported=False,
        repair_round=2,
        max_rounds=10,
    )
    assert first.next == "loop_decision" and first.updates == {"fq_warning_reported": True}
    again = decide_figure_quality(
        passed=True,
        actionable=True,
        attempted=True,
        warning_reported=True,
        repair_round=2,
        max_rounds=10,
    )
    assert again.updates == {}
    budget = decide_figure_quality(
        passed=True,
        actionable=True,
        attempted=False,
        warning_reported=False,
        repair_round=10,
        max_rounds=10,
    )
    assert budget.next == "loop_decision" and budget.actions == ()
    quiet = decide_figure_quality(
        passed=False,
        actionable=True,
        attempted=False,
        warning_reported=False,
        repair_round=0,
        max_rounds=10,
    )
    assert quiet.next == "loop_decision"


def test_F23_validated_paper_is_finalized() -> None:
    assert (
        decide_loop(
            mode="fresh",
            passed=True,
            repeated=True,
            repair_round=10,
            max_rounds=10,
            audit_outcome=None,
        ).next
        == "finalize"
    )


def test_F24_repeated_failure_is_incomplete() -> None:
    d = decide_loop(
        mode="fresh", passed=False, repeated=True, repair_round=1, max_rounds=10, audit_outcome=None
    )
    assert d.next == "outcome" and d.incomplete.startswith(
        "paper validation errors and inputs were unchanged after repair"
    )


def test_F25_failed_validation_starts_a_round() -> None:
    d = decide_loop(
        mode="fresh",
        passed=False,
        repeated=False,
        repair_round=4,
        max_rounds=10,
        audit_outcome="pass",
    )
    assert d.next == "select_stage" and d.updates == {"repair_round": 5}


def test_F26_failed_validation_after_the_last_round_is_incomplete() -> None:
    d = decide_loop(
        mode="fresh",
        passed=False,
        repeated=False,
        repair_round=10,
        max_rounds=10,
        audit_outcome="findings",
    )
    assert d.next == "outcome"
    assert d.incomplete == (
        "paper validation failed; see validation_feedback.json; paper evidence audit reported "
        "revision findings; see paper_evidence_audit_feedback.json"
    )


# --- edit table -----------------------------------------------------------------------------------


async def test_E04_revision_restores_preserved_references(workspace: Any) -> None:
    await workspace.write_bytes("baseline/references.bib", b"@a{x}")
    await workspace.write_bytes("baseline/references_provenance.json", b"{}")
    await workspace.write_bytes("p/references.bib", b"@changed{y}")
    await restore_revision_references(workspace, "p", "baseline")
    assert await workspace.read_bytes("p/references.bib") == b"@a{x}"
    assert await workspace.read_bytes("p/references_provenance.json") == b"{}"


def test_E05_write_ok() -> None:
    assert (
        decide_after_write(mode="edit", stage="draft", ok=True, repair_round=0, max_rounds=3).next
        == "postdraft_gate"
    )


def test_E06_failed_figure_quality_stage_restores() -> None:
    d = decide_after_write(
        mode="edit", stage="figure_quality_repair", ok=False, repair_round=3, max_rounds=3
    )
    assert d.actions == ("restore_fq",) and d.next == "postdraft_gate"


def test_E07_failed_write_after_last_round_fails() -> None:
    d = decide_after_write(
        mode="edit", stage="manuscript_repair", ok=False, repair_round=3, max_rounds=3
    )
    assert d.next == "outcome" and d.failed.startswith("exhausted repair rounds after 3 attempts")
    assert d.incomplete is None


def test_E08_failed_write_starts_the_next_round() -> None:
    d = decide_after_write(mode="revision", stage="draft", ok=False, repair_round=0, max_rounds=3)
    assert d.next == "select_stage" and d.updates["repair_round"] == 1


def test_E09_postdraft_honours_skip() -> None:
    assert decide_postdraft(repair_round=0, enrichment_done=False, skip=True).next == "figures"
    assert decide_postdraft(repair_round=0, enrichment_done=False, skip=False).next == "postdraft"


async def test_E10_preserve_mode_keeps_an_unchanged_concept_state(
    make_context: Any, workspace: Any
) -> None:
    from langgraph_agentic_scientist.paper.figures import preserve_fingerprint

    renderer = FakeFigureRenderer()
    ctx = make_context(figure_renderer=renderer)
    paper_dir = "run/paper/p"
    await workspace.write_bytes(f"{paper_dir}/figures/a.png", b"png")
    fingerprint = await preserve_fingerprint(workspace, paper_dir)
    state = {
        "paper_dir": paper_dir,
        "mode": "edit",
        "figure_refresh_mode": "preserve",
        "concept": ConceptFigureState(available=True, preserve_fingerprint=fingerprint),
    }
    update = await _default_figures(state, _Runtime(ctx))
    assert [event.kind for event in update["events"]] == ["concept_figure_preserved"]
    assert renderer.requests == []


async def test_E11_edit_validation_asks_for_the_scope_audit(
    make_context: Any, workspace: Any
) -> None:
    toolchain = FakePaperToolchain()
    seen: dict[str, Any] = {}
    original = toolchain.validate

    async def spy(paper_dir: str, *, workspace: Any, options: dict[str, Any]) -> ValidationReport:
        seen.update(options)
        return await original(paper_dir, workspace=workspace, options=options)

    toolchain.validate = spy  # type: ignore[method-assign]
    ctx = make_context(paper_toolchain=toolchain, config=fast_config(paper=paper_config()))
    state = {"paper_dir": "run/paper/p", "mode": "edit", "run_config": build_run_config(ctx)}
    update = await _default_validation(state, _Runtime(ctx))
    assert seen["require_scope_audit"] is True and seen["mode"] == "edit"
    assert update["validation_passed"] is False
    feedback = json.loads(
        (await workspace.read_bytes("run/paper/p/validation_feedback.json")).decode()
    )
    assert any("missing required edit-scope artifact" in e for e in feedback["hard_errors"])
    assert feedback["auxiliary_checks"]["method_scope_audit"]["ok"] is False
    await workspace.write_bytes("run/paper/p/paper.tex", b"Dense packing.")
    await workspace.write_bytes(
        "run/paper/p/method_scope_audit.json", json.dumps(SCOPE_AUDIT).encode()
    )
    await workspace.write_bytes("run/paper/p/method_scope_audit.md", b"# audit")
    again = await _default_validation(state, _Runtime(ctx))
    assert again["validation_passed"] is True


def test_E12_figure_quality_regression_restores_and_revalidates() -> None:
    d = decide_after_validate(
        mode="edit",
        stage="figure_quality_repair",
        passed=False,
        reverted=False,
        fingerprint=None,
        previous=None,
    )
    assert d.next == "validate" and d.actions == ("restore_fq",)


def test_E13_validated_figure_quality_repair_discards_the_snapshot() -> None:
    d = decide_after_validate(
        mode="edit",
        stage="figure_quality_repair",
        passed=True,
        reverted=False,
        fingerprint=None,
        previous=None,
    )
    assert d.next == "review_check" and d.actions == ("discard_fq",)


def test_E14_fingerprint() -> None:
    d = decide_after_validate(
        mode="edit",
        stage="manuscript_repair",
        passed=False,
        reverted=False,
        fingerprint="z",
        previous="z",
    )
    assert d.updates["failure_repeated"] is True and d.next == "review_check"


def test_E15_review_gate_in_edit_mode_fails_hard() -> None:
    assert (
        decide_review_check(mode="edit", review_round=0, review_rounds=2, passed=True).next
        == "review_gate"
    )
    failed = decide_after_review(
        mode="edit", outcome="stage_failed", review_round=0, repair_round=0, max_rounds=3
    )
    assert failed.failed and failed.incomplete is None
    exhausted = decide_after_review(
        mode="edit", outcome="completed", review_round=0, repair_round=3, max_rounds=3
    )
    assert exhausted.failed.startswith("exhausted repair rounds after 3 attempts")
    repair = decide_after_review(
        mode="edit", outcome="completed", review_round=1, repair_round=1, max_rounds=3
    )
    assert repair.next == "select_stage" and repair.actions == ("save_review_pdf",)


def test_E16_audit_runs_only_on_a_validated_paper() -> None:
    assert (
        decide_review_check(mode="edit", review_round=1, review_rounds=1, passed=True).next
        == "audit"
    )
    assert (
        decide_review_check(mode="edit", review_round=1, review_rounds=1, passed=False).next
        == "figure_quality_gate"
    )


def test_E17_findings_after_the_last_round_fail() -> None:
    d = decide_after_audit(
        mode="edit", stage="draft", reverted=False, outcome="findings", repair_round=3, max_rounds=3
    )
    assert d.failed == "evidence audit findings remain after 3 repair rounds"


def test_E18_findings_start_a_round() -> None:
    d = decide_after_audit(
        mode="edit", stage="draft", reverted=False, outcome="findings", repair_round=1, max_rounds=3
    )
    assert d.next == "select_stage" and d.updates == {"repair_round": 2} and d.actions == ()


@pytest.mark.parametrize("outcome", ["operational_failure", "contract_exhausted"])
def test_E19_operational_audit_failure_fails(outcome: str) -> None:
    d = decide_after_audit(
        mode="edit", stage="draft", reverted=False, outcome=outcome, repair_round=0, max_rounds=3
    )
    assert d.failed == "required evidence audit failed operationally"


def test_E20_figure_quality_snapshot() -> None:
    d = decide_figure_quality(
        passed=True,
        actionable=True,
        attempted=False,
        warning_reported=False,
        repair_round=0,
        max_rounds=3,
    )
    assert d.actions == ("snapshot_fq",)


def test_E21_warnings_reported_once() -> None:
    d = decide_figure_quality(
        passed=True,
        actionable=True,
        attempted=True,
        warning_reported=False,
        repair_round=1,
        max_rounds=3,
    )
    assert d.updates == {"fq_warning_reported": True}


def test_E22_validated_paper_is_finalized() -> None:
    assert (
        decide_loop(
            mode="edit",
            passed=True,
            repeated=False,
            repair_round=0,
            max_rounds=3,
            audit_outcome="pass",
        ).next
        == "finalize"
    )


def test_E23_repeated_failure_fails() -> None:
    d = decide_loop(
        mode="edit", passed=False, repeated=True, repair_round=1, max_rounds=3, audit_outcome=None
    )
    assert d.failed.startswith("validation errors and inputs were unchanged after repair")


def test_E24_exhausted_rounds_fail() -> None:
    d = decide_loop(
        mode="edit", passed=False, repeated=False, repair_round=3, max_rounds=3, audit_outcome=None
    )
    assert d.failed.startswith("exhausted repair rounds after 3 attempts")


def test_E25_failed_validation_starts_a_round() -> None:
    d = decide_loop(
        mode="revision",
        passed=False,
        repeated=False,
        repair_round=1,
        max_rounds=3,
        audit_outcome=None,
    )
    assert d.next == "select_stage" and d.updates == {"repair_round": 2}


async def test_strategy_family_is_rendered_for_the_concept_request(make_context: Any) -> None:
    """The strategy helper of the fixtures validates (it drives the two-stage tests)."""
    from langgraph_agentic_scientist.paper.models import StrategyAnswer

    message = strategy()
    args = message.tool_calls[0]["args"]
    assert StrategyAnswer.model_validate(args).strategy.selected_prompt_family == "paper_standalone"
