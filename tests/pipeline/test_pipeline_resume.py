"""Resuming a run: nothing that completed runs again, and a changed configuration is refused."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pipeline_stubs import Ledger
from testkit import thread

from langgraph_agentic_scientist import (
    ConfigurationMismatch,
    InvestigationConfig,
    ScientistConfig,
)
from langgraph_agentic_scientist.config import ALL_PHASES

PHASES = [
    "investigation",
    "ablations",
    "final_evaluation",
    "posthoc",
    "paper",
    "review",
    "revision",
]
ALL = ScientistConfig(phases=ALL_PHASES)


@pytest.mark.parametrize("phase", PHASES)
async def test_a_run_interrupted_inside_a_phase_resumes_without_repeating_any_work(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
    phase: str,
) -> None:
    app = make_app()
    ctx = make_context(config=ALL)
    config = thread(f"pause-{phase}")
    ledger.pause.add(phase)
    out = await app.ainvoke({"task_dir": circpack}, config, context=ctx)
    assert "__interrupt__" in out
    done = ledger.order[: ledger.order.index(phase) + 1]
    assert done[-1] == phase
    out = await app.ainvoke(Command(resume="go"), config, context=ctx)
    assert out["status"] == "completed", out["phase_status"]
    assert ledger.order == PHASES, "a phase ran twice, or in the wrong order"
    assert all(count == 1 for count in ledger.calls.values())
    assert [e.kind for e in out["events"]].count("pipeline_start") == 1


@pytest.mark.parametrize("phase", PHASES)
async def test_a_run_that_crashed_in_a_phase_resumes_from_its_checkpoint(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
    phase: str,
) -> None:
    app = make_app()
    ctx = make_context(config=ALL)
    config = thread(f"crash-{phase}")
    ledger.crash.add(phase)
    with pytest.raises(RuntimeError, match=f"{phase} crashed"):
        await app.ainvoke({"task_dir": circpack}, config, context=ctx)
    out = await app.ainvoke(None, config, context=ctx)
    assert out["status"] == "completed"
    expected = {name: 1 for name in PHASES} | {phase: 2}  # the crashed phase restarts, once
    assert dict(ledger.calls) == expected


async def test_the_snapshot_is_not_taken_again_on_resume(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
) -> None:
    app = make_app()
    ctx = make_context(config=ALL)
    config = thread("snap-resume")
    ledger.pause.add("investigation")
    await app.ainvoke({"task_dir": circpack, "run_id": "fixed"}, config, context=ctx)
    state = await app.aget_state(config)
    assert state.values["run_id"] == "fixed" and state.values["task"].root == "task"
    out = await app.ainvoke(Command(resume="go"), config, context=ctx)
    assert out["run_id"] == "fixed"


async def test_resuming_with_a_different_method_configuration_is_refused_before_the_next_phase(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
) -> None:
    app = make_app()
    config = thread("changed-config")
    ledger.pause.add("investigation")
    await app.ainvoke({"task_dir": circpack}, config, context=make_context(config=ALL))
    changed = ScientistConfig(
        phases=ALL_PHASES, investigation=InvestigationConfig(max_idea_cycles=5)
    )
    with pytest.raises(ConfigurationMismatch, match=r"investigation\.max_idea_cycles"):
        await app.ainvoke(Command(resume="go"), config, context=make_context(config=changed))
    assert ledger.calls["ablations"] == 0


async def test_a_runtime_only_setting_may_change_on_resume(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
) -> None:
    from langgraph_agentic_scientist.config import AgentsConfig

    app = make_app()
    config = thread("runtime-only")
    ledger.pause.add("paper")
    await app.ainvoke({"task_dir": circpack}, config, context=make_context(config=ALL))
    changed = ScientistConfig(phases=ALL_PHASES, agents=AgentsConfig(execute_timeout_seconds=30))
    out = await app.ainvoke(Command(resume="go"), config, context=make_context(config=changed))
    assert out["status"] == "completed"


async def test_a_prompt_override_changed_between_runs_is_refused(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
) -> None:
    from langgraph_agentic_scientist import PromptRegistry

    app = make_app()
    config = thread("prompt-change")
    ledger.pause.add("investigation")
    await app.ainvoke({"task_dir": circpack}, config, context=make_context(config=ALL))
    registry = PromptRegistry.default().with_overrides({"fragments/baseline_failures": "changed"})
    with pytest.raises(ConfigurationMismatch, match="prompt:fragments/baseline_failures"):
        await app.ainvoke(
            Command(resume="go"), config, context=make_context(config=ALL, prompts=registry)
        )


async def test_a_second_thread_in_the_same_workspace_reuses_the_snapshot(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
    workspace: Any,
) -> None:
    import shutil

    saver = InMemorySaver()
    ctx = make_context(config=ScientistConfig(phases=("investigation",)))
    first = await make_app(saver=saver).ainvoke({"task_dir": circpack}, thread("a"), context=ctx)
    shutil.rmtree(workspace.root / "inputs")  # the source is not needed once the snapshot exists
    second = await make_app(saver=saver).ainvoke({"task_dir": circpack}, thread("b"), context=ctx)
    assert first["task"] == second["task"]
