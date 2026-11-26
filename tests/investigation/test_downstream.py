"""No-winner downstream selection: ranking, the baseline case, freezing, no reselection."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from langgraph_agentic_scientist.core.results import AttemptRecord, InvestigationResult
from langgraph_agentic_scientist.core.state import BestManifest, EvaluationRecord, ProgramRef
from langgraph_agentic_scientist.investigation import build_downstream_selection_graph
from langgraph_agentic_scientist.investigation.downstream import (
    StaleDownstreamSelection,
    select_fallback,
)
from langgraph_agentic_scientist.testing import write_program

from .helpers import config, scripted, task_with_harness, thread


def _record(
    attempt: str,
    program: str,
    score: float | None,
    *,
    correct: bool = True,
    status: str = "completed",
    plane: str = "search",
) -> EvaluationRecord:
    return EvaluationRecord(
        evaluation_id=f"{attempt}/{program}",
        program=ProgramRef(path=f"results/ideas/{attempt}/{program}"),
        plane=plane,  # type: ignore[arg-type]
        status=status,
        metrics={} if score is None else {"combined_score": score},
        correct=correct,
        attempt_id=attempt,
    )


def _attempt(attempt: str) -> AttemptRecord:
    return AttemptRecord(
        attempt_id=attempt, idea_id=attempt, cycle=1, attempt_dir="x", status="fail"
    )


def _result(
    records: list[EvaluationRecord],
    *,
    status: str = "completed_no_winner",
    best: BestManifest | None = None,
) -> InvestigationResult:
    return InvestigationResult(
        status=status,  # type: ignore[arg-type]
        best=best
        or BestManifest(
            program_dir="results/.baseline-seed/program_0",
            combined_score=1.0,
            correct=True,
            source_attempt_id="bootstrap",
        ),
        attempts=[_attempt("a1"), _attempt("a2"), _attempt("a3")],
        evaluations=records,
    )


def test_candidates_rank_by_score_then_attempt_order_then_program_index() -> None:
    records = [
        _record("a1", "program_1", 0.5),
        _record("a2", "program_2", 0.9),
        _record("a2", "program_1", 0.9),
        _record("a3", "program_1", 0.9),
    ]
    chosen = select_fallback(_result(records))
    assert chosen.status == "selected" and chosen.source == "candidate"
    assert chosen.program is not None and chosen.program.path.endswith("a2/program_1")
    assert chosen.combined_score == 0.9


def test_exclusions_seeded_copies_incorrect_unscored_and_failed_evaluations() -> None:
    records = [
        _record("a1", "program_0", 5.0),
        _record("a1", "program_1", 4.0, correct=False),
        _record("a2", "program_1", None),
        _record("a2", "program_2", 3.0, status="failed"),
        _record("a3", "program_1", float("inf")),
        _record("a3", "program_2", 2.0, plane="variant"),
    ]
    chosen = select_fallback(_result(records))
    assert chosen.source == "baseline", "no valid candidate: the retained baseline is used"
    assert chosen.combined_score == 1.0


def test_a_single_lower_scoring_candidate_beats_the_baseline() -> None:
    chosen = select_fallback(_result([_record("a1", "program_1", 0.2)]))
    assert chosen.source == "candidate" and chosen.combined_score == 0.2


def test_without_any_valid_program_the_selection_is_skipped() -> None:
    best = BestManifest(
        program_dir="x", combined_score=None, correct=None, source_attempt_id="bootstrap"
    )
    chosen = select_fallback(_result([], best=best))
    assert chosen.status == "skipped" and chosen.program is None
    assert select_fallback(_result([], status="completed")).status == "skipped"


async def _run(ctx: Any, task: Any, investigation: InvestigationResult, app: Any, name: str) -> Any:
    out = await app.ainvoke(
        {"task": task, "run_id": "run", "investigation": investigation}, thread(name), context=ctx
    )
    return out["downstream"]


async def test_the_chosen_program_is_frozen_and_a_resume_never_reselects(
    make_context: Any, workspace: Any
) -> None:
    task = await task_with_harness(workspace, harness=True)
    await write_program(workspace, "results/ideas/a1/program_1", "SCORE = 7.0\n")
    await workspace.write_bytes("results/ideas/a1/program_1/metrics.json", b"{}")
    await write_program(workspace, "results/ideas/a2/program_1", "SCORE = 8.0\n")
    first = _result([_record("a1", "program_1", 0.4)])
    ctx = make_context(models=scripted({}), config=config())
    app = build_downstream_selection_graph(checkpointer=InMemorySaver())
    selection = await _run(ctx, task, first, app, "d1")
    assert selection.status == "selected" and selection.source == "candidate"
    assert selection.program is not None and selection.program.path == "run/downstream/program"
    frozen = (await workspace.read_bytes("run/downstream/program/src/program.py")).decode()
    assert frozen == "SCORE = 7.0\n"
    assert not await workspace.exists("run/downstream/program/metrics.json"), (
        "evaluation results are not frozen"
    )
    manifest = json.loads((await workspace.read_bytes("run/.downstream_selection.json")).decode())
    assert manifest["program_tree_sha256"] == selection.program.digest
    assert manifest["investigation_outcome"] == "completed_no_winner"
    # a later run with a better candidate does not reselect: the frozen selection stands
    better = _result([_record("a1", "program_1", 0.4), _record("a2", "program_1", 0.9)])
    again = await _run(
        ctx, task, better, build_downstream_selection_graph(checkpointer=InMemorySaver()), "d2"
    )
    assert again.program is not None and again.program.digest == selection.program.digest
    assert again.evaluation_id == selection.evaluation_id == "a1/program_1"


async def test_a_changed_frozen_program_is_refused(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    await write_program(workspace, "results/ideas/a1/program_1", "SCORE = 7.0\n")
    ctx = make_context(models=scripted({}), config=config())
    result = _result([_record("a1", "program_1", 0.4)])
    await _run(
        ctx, task, result, build_downstream_selection_graph(checkpointer=InMemorySaver()), "d1"
    )
    await workspace.write_bytes("run/downstream/program/src/program.py", b"SCORE = 9.0\n")
    with pytest.raises(StaleDownstreamSelection):
        await _run(
            ctx, task, result, build_downstream_selection_graph(checkpointer=InMemorySaver()), "d2"
        )


async def test_a_skipped_selection_freezes_nothing(make_context: Any, workspace: Any) -> None:
    task = await task_with_harness(workspace, harness=True)
    best = BestManifest(program_dir="x", source_attempt_id="bootstrap")
    ctx = make_context(models=scripted({}), config=config())
    app = build_downstream_selection_graph(checkpointer=InMemorySaver())
    selection = await _run(ctx, task, _result([], best=best), app, "d")
    assert selection.status == "skipped"
    assert not await workspace.exists("run/.downstream_selection.json")
