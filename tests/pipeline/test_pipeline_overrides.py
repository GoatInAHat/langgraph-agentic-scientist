"""Extension points: replaced phases, slots handed to phase factories, and hooks around steps."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from pipeline_stubs import Ledger, default_results, stub_phase
from testkit import thread

from langgraph_agentic_scientist import (
    GraphOverrides,
    Hooks,
    PaperResult,
    ScientistConfig,
    build_pipeline_graph,
)
from langgraph_agentic_scientist import pipeline as pipeline_module
from langgraph_agentic_scientist.config import ALL_PHASES
from langgraph_agentic_scientist.core.steps import HookError
from langgraph_agentic_scientist.testing import FakeEvaluator, FakeFinalEvaluator

ALL = ScientistConfig(phases=ALL_PHASES)


async def test_a_replaced_phase_is_used_and_the_overrides_reach_the_other_factories(
    make_context: Callable[..., Any],
    circpack: str,
    ledger: Ledger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def loader(spec: pipeline_module.PhaseSpec) -> Callable[..., Any]:
        def factory(**kwargs: Any) -> CompiledStateGraph[Any, Any, Any, Any]:
            seen[spec.name] = kwargs
            return stub_phase(spec.name, ledger, default_results()[spec.name])

        return factory

    monkeypatch.setattr(pipeline_module, "_load_factory", loader)

    async def custom_slot(value: Any, runtime: Any) -> dict[str, Any]:
        return {}

    custom_paper = stub_phase(
        "custom_paper",
        ledger,
        PaperResult(status="complete", paper_dir="elsewhere", pdf_path="elsewhere/p.pdf"),
        state_key_output="paper",
    )
    overrides = GraphOverrides(phases={"paper": custom_paper}, slots={"paper.audit": custom_slot})
    out = await build_pipeline_graph(checkpointer=InMemorySaver(), overrides=overrides).ainvoke(
        {"task_dir": circpack},
        thread("override"),
        context=make_context(
            config=ALL, evaluator=FakeEvaluator(), final_evaluator=FakeFinalEvaluator()
        ),
    )
    assert "paper" not in seen, "the factory of a replaced phase must not be called"
    assert set(seen) == {
        "task_drafting",
        "investigation",
        "downstream_selection",
        "ablations",
        "final_evaluation",
        "posthoc",
        "review",
        "revision",
    }
    assert all(kwargs["overrides"] is overrides for kwargs in seen.values())
    assert all(kwargs["name"] == name for name, kwargs in seen.items())
    assert out["paper"].paper_dir == "elsewhere"
    assert ledger.calls["custom_paper"] == 1
    assert ledger.last_input("review")["paper_dir"] == "elsewhere"


async def test_a_phase_graph_receives_only_the_keys_its_input_schema_declares(
    make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = default_results()
    phases = {name: stub_phase(name, ledger, value) for name, value in results.items()}
    phases["paper"] = stub_phase("paper", ledger, results["paper"], narrow=True)
    out = await build_pipeline_graph(
        checkpointer=InMemorySaver(), overrides=GraphOverrides(phases=phases)
    ).ainvoke({"task_dir": circpack}, thread("narrow"), context=make_context(config=ALL))
    assert out["status"] == "completed"
    assert set(ledger.last_input("paper")) - {"hook_outputs", "log"} == {"task"}


async def test_a_phase_may_return_its_result_under_the_pipelines_key(
    make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    results = default_results()
    phases = {
        name: stub_phase(name, ledger, value, state_key_output=spec.state_key)
        for name, value in results.items()
        for spec in pipeline_module.PHASE_SPECS
        if spec.name == name
    }
    out = await build_pipeline_graph(
        checkpointer=InMemorySaver(), overrides=GraphOverrides(phases=phases)
    ).ainvoke({"task_dir": circpack}, thread("keyed"), context=make_context(config=ALL))
    assert out["investigation"].status == "completed" and out["paper"].status == "complete"
    assert out["final"].outcome == "qualified" and out.get("downstream") is None


def _hook(log: list[str], name: str, outputs: Mapping[str, Any] | None = None) -> Any:
    async def hook(state: Mapping[str, Any], runtime: Any) -> dict[str, Any] | None:
        log.append(name)
        return {"hook_outputs": dict(outputs)} if outputs else None

    return hook


async def test_hooks_fire_at_the_documented_step_names_once_each_across_an_interrupt(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    log: list[str] = []
    steps = ["snapshot_task", "check_dependencies", "investigation", "paper", "finish"]
    hooks = Hooks(
        before={f"pipeline.{s}": [_hook(log, f"before:{s}")] for s in steps},
        after={
            **{f"pipeline.{s}": [_hook(log, f"after:{s}")] for s in steps},
            "pipeline.paper": [_hook(log, "after:paper", {"copied": True})],
        },
    )
    ctx = make_context(config=ALL, hooks=hooks)
    app = make_app()
    config = thread("hooks")
    ledger.pause.add("investigation")
    await app.ainvoke({"task_dir": circpack}, config, context=ctx)
    assert log == [
        "before:snapshot_task",
        "after:snapshot_task",
        "before:check_dependencies",
        "after:check_dependencies",
        "before:investigation",
    ]
    out = await app.ainvoke(Command(resume="go"), config, context=ctx)
    assert log == [
        "before:snapshot_task",
        "after:snapshot_task",
        "before:check_dependencies",
        "after:check_dependencies",
        "before:investigation",
        "after:investigation",
        "before:paper",
        "after:paper",
        "before:finish",
        "after:finish",
    ]
    assert out["hook_outputs"] == {"pipeline.paper": {"copied": True}}


async def test_a_hook_may_not_rewrite_the_state_of_a_pipeline_step(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str
) -> None:
    async def tamper(state: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
        return {"paper": "not allowed"}

    ctx = make_context(config=ALL, hooks=Hooks(after={"pipeline.paper": [tamper]}))
    with pytest.raises(HookError, match="may not update 'paper'"):
        await make_app().ainvoke({"task_dir": circpack}, thread("tamper"), context=ctx)


async def test_a_raising_hook_fails_the_step_it_surrounds(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    async def veto(state: Mapping[str, Any], runtime: Any) -> None:
        raise PermissionError("not today")

    ctx = make_context(config=ALL, hooks=Hooks(before={"pipeline.final_evaluation": [veto]}))
    with pytest.raises(PermissionError, match="not today"):
        await make_app().ainvoke({"task_dir": circpack}, thread("veto"), context=ctx)
    assert "final_evaluation" not in ledger.calls and ledger.calls["ablations"] == 1


async def test_the_graph_names_a_hook_pair_around_every_phase() -> None:
    app = build_pipeline_graph(overrides=GraphOverrides(phases=_all_stub_phases()))
    nodes = set(app.get_graph().nodes)
    for spec in pipeline_module.PHASE_SPECS:
        assert {
            spec.name,
            f"pipeline.{spec.name}.before",
            f"pipeline.{spec.name}.after",
            f"route_{spec.name}",
        } <= nodes
    assert {"pipeline.snapshot_task.before", "pipeline.finish.after"} <= nodes


def _all_stub_phases() -> dict[str, CompiledStateGraph[Any, Any, Any, Any]]:
    ledger = Ledger()
    return {name: stub_phase(name, ledger, value) for name, value in default_results().items()}
