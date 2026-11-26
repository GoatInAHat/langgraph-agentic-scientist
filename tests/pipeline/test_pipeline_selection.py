"""Phase selection: normalisation, and upstream's configuration errors raised before any phase."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pipeline_stubs import Ledger, final_result
from testkit import thread

from langgraph_agentic_scientist import (
    EditOptions,
    GraphOverrides,
    InvestigationResult,
    PaperResult,
    PipelineConfigurationError,
    ReviewResult,
    ScientistConfig,
)
from langgraph_agentic_scientist.config import ALL_PHASES, DEFAULT_PHASES
from langgraph_agentic_scientist.core.state import TaskRef
from langgraph_agentic_scientist.pipeline import normalize_phases, validate_selection
from langgraph_agentic_scientist.testing import FakeEvaluator, FakeFinalEvaluator

DONE = InvestigationResult(status="completed")


def test_phases_are_normalised_to_upstreams_order_once_each() -> None:
    assert normalize_phases(["paper", "investigation", "paper"]) == ["investigation", "paper"]
    assert normalize_phases(list(reversed(ALL_PHASES))) == list(ALL_PHASES)


def test_an_empty_or_unknown_selection_is_refused() -> None:
    with pytest.raises(PipelineConfigurationError, match="no phases"):
        normalize_phases([])
    with pytest.raises(PipelineConfigurationError, match="'tidy'"):
        normalize_phases(["tidy"])


async def test_the_default_selection_is_upstreams_six_phases(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    out = await make_app().ainvoke({"task_dir": circpack}, thread("def"), context=make_context())
    assert out["phases"] == list(DEFAULT_PHASES)
    assert "review" not in ledger.calls and out["phase_status"]["review"].reason == "not selected"


async def test_the_input_overrides_the_configured_selection(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    out = await make_app().ainvoke(
        {"task_dir": circpack, "phases": ["investigation"]}, thread("sel"), context=make_context()
    )
    assert out["phases"] == ["investigation"] and ledger.order == ["investigation"]


@pytest.mark.parametrize(
    ("phases", "given", "message"),
    [
        (["paper"], {}, "without investigation needs the investigation result"),
        (["posthoc"], {}, "without investigation needs the investigation result"),
        (
            ["paper"],
            {"investigation": InvestigationResult(status="failed")},
            "did not complete",
        ),
        (["review"], {}, "review without paper needs the paper result"),
        (["revision"], {}, "revision without paper needs the paper result"),
        (
            ["revision"],
            {"paper": PaperResult(status="complete")},
            "revision without review needs a completed review",
        ),
        (
            ["revision"],
            {
                "paper": PaperResult(status="complete"),
                "review": ReviewResult(status="failed"),
            },
            "needs a completed review",
        ),
        (
            ["investigation", "paper", "revision"],
            {},
            "paper and revision requires review in the same phase selection",
        ),
        (["investigation"], {"previous_paper_dir": "paper"}, "previous_paper_dir needs the paper"),
    ],
)
async def test_selections_upstream_would_refuse_are_refused_before_any_phase_runs(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
    phases: list[str],
    given: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(PipelineConfigurationError, match=message):
        await make_app().ainvoke(
            {"task_dir": circpack, "phases": phases, **given}, thread("bad"), context=make_context()
        )
    assert not ledger.calls


async def test_a_later_phase_may_run_against_the_results_of_an_earlier_run(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    out = await make_app().ainvoke(
        {
            "task_dir": circpack,
            "phases": ["paper", "review", "revision"],
            "investigation": DONE,
            "final": final_result(),
            "review": None,
        },
        thread("later"),
        context=make_context(),
    )
    assert ledger.order == ["paper", "review", "revision"] and out["status"] == "completed"
    assert ledger.last_input("paper")["investigation"] == DONE


async def test_paper_edit_mode_carries_the_previous_paper_and_the_edit_options(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    workspace: Any,
    ledger: Ledger,
) -> None:
    (workspace.root / "old_paper").mkdir()
    options = EditOptions(edit_programs=["program_3"], figure_refresh_mode="preserve")
    await make_app().ainvoke(
        {
            "task_dir": circpack,
            "phases": ["paper"],
            "investigation": DONE,
            "previous_paper_dir": "old_paper",
            "edit_options": options,
        },
        thread("edit"),
        context=make_context(),
    )
    paper = ledger.last_input("paper")
    assert paper["mode"] == "edit" and paper["previous_paper_dir"] == "old_paper"
    assert paper["edit_options"] == options


async def test_a_missing_previous_paper_is_refused(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str
) -> None:
    with pytest.raises(PipelineConfigurationError, match="previous_paper_dir does not exist"):
        await make_app().ainvoke(
            {
                "task_dir": circpack,
                "phases": ["paper"],
                "investigation": DONE,
                "previous_paper_dir": "nowhere",
            },
            thread("edit2"),
            context=make_context(),
        )


# --- what the context must hold ---------------------------------------------------------------


def _state(phases: list[str], task: TaskRef) -> dict[str, Any]:
    return {"phases": phases, "task": task, "investigation": DONE}


def test_the_phases_that_evaluate_need_the_contexts_evaluators(make_context: Any) -> None:
    task = TaskRef(root="task")
    context = make_context()
    with pytest.raises(PipelineConfigurationError) as info:
        validate_selection(
            _state(["investigation", "final_evaluation"], task),
            ScientistConfig(),
            drafting_pending=False,
            context=context,
            overrides=GraphOverrides(),
        )
    assert "no evaluator, which investigation need" in str(info.value)
    assert "no final_evaluator" in str(info.value)
    validate_selection(
        _state(["investigation", "final_evaluation"], task),
        ScientistConfig(),
        drafting_pending=False,
        context=make_context(evaluator=FakeEvaluator(), final_evaluator=FakeFinalEvaluator()),
        overrides=GraphOverrides(),
    )


def test_a_phase_the_caller_replaced_needs_no_evaluator(make_context: Any) -> None:
    from pipeline_stubs import stub_phase

    validate_selection(
        _state(["investigation"], TaskRef(root="task")),
        ScientistConfig(),
        drafting_pending=False,
        context=make_context(),
        overrides=GraphOverrides(phases={"investigation": stub_phase("i", Ledger(), DONE)}),
    )


def test_drafting_needs_an_evaluator_only_when_it_will_run(make_context: Any) -> None:
    task = TaskRef(root="task")
    kwargs: dict[str, Any] = {
        "config": ScientistConfig(),
        "context": make_context(),
        "overrides": GraphOverrides(),
    }
    validate_selection(
        {"phases": ["task_drafting"], "task": task}, drafting_pending=False, **kwargs
    )
    with pytest.raises(PipelineConfigurationError, match="task_drafting need"):
        validate_selection(
            {"phases": ["task_drafting"], "task": task}, drafting_pending=True, **kwargs
        )


async def test_a_contracted_task_refuses_an_evaluator_that_does_not_isolate_candidates(
    make_context: Any, workspace: Any, contracted: str
) -> None:
    from langgraph_agentic_scientist.pipeline import load_task_ref, prepare_task_snapshot

    await prepare_task_snapshot(workspace, contracted)
    task = await load_task_ref(workspace, ScientistConfig())
    leaky = make_context(
        evaluator=FakeEvaluator(isolated=False), final_evaluator=FakeFinalEvaluator(isolated=False)
    )
    selected = ["investigation", "final_evaluation"]
    with pytest.raises(PipelineConfigurationError, match="does not isolate candidates") as info:
        validate_selection(
            _state(selected, task),
            ScientistConfig(),
            drafting_pending=False,
            context=leaky,
            overrides=GraphOverrides(),
        )
    assert "of investigation" in str(info.value) and "of final_evaluation" in str(info.value)
    validate_selection(  # for local experiments and tests
        _state(selected, task),
        ScientistConfig(allow_unisolated_contracted=True),
        drafting_pending=False,
        context=leaky,
        overrides=GraphOverrides(),
    )
    validate_selection(
        _state(selected, task),
        ScientistConfig(),
        drafting_pending=False,
        context=make_context(evaluator=FakeEvaluator(), final_evaluator=FakeFinalEvaluator()),
        overrides=GraphOverrides(),
    )
