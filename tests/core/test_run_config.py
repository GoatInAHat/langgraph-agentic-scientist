"""record_run_config: configuration and prompts recorded, verified on resume, schema versioned."""

from __future__ import annotations

from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from langgraph_agentic_scientist import (
    ConfigurationMismatch,
    IncompatibleCheckpoint,
    InvestigationConfig,
    PromptRegistry,
    ScientistConfig,
    ScientistContext,
)
from langgraph_agentic_scientist.config import AgentsConfig, runtime_only_paths
from langgraph_agentic_scientist.core.runconfig import method_config, record_run_config
from langgraph_agentic_scientist.core.state import RunConfigRecord


class State(TypedDict, total=False):
    run_config: RunConfigRecord | None
    cycles: int
    answer: str


def _graph() -> Any:
    async def read_limits(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        answer = interrupt("pause")
        return {"cycles": method_config(state).investigation.max_idea_cycles, "answer": answer}

    graph: StateGraph[Any, Any, Any, Any] = StateGraph(State, context_schema=ScientistContext)
    graph.add_node("record_run_config", record_run_config)
    graph.add_node("read_limits", read_limits)
    # A phase mounted later verifies the record with its own first node.
    graph.add_node("next_phase_record", record_run_config)
    graph.add_edge(START, "record_run_config")
    graph.add_edge("record_run_config", "read_limits")
    graph.add_edge("read_limits", "next_phase_record")
    return graph.compile(checkpointer=InMemorySaver())


async def _start(app: Any, ctx: ScientistContext, thread: str) -> dict[str, Any]:
    config = {"configurable": {"thread_id": thread}}
    await app.ainvoke({}, config, context=ctx)
    return config


async def test_records_configuration_and_prompt_fingerprints(make_context: Any) -> None:
    from langgraph.types import Command

    ctx = make_context(config=ScientistConfig(investigation=InvestigationConfig(max_idea_cycles=3)))
    app = _graph()
    config = await _start(app, ctx, "record")
    state = (await app.aget_state(config)).values
    record = state["run_config"]
    assert isinstance(record, RunConfigRecord)
    assert record.method_config["investigation"]["max_idea_cycles"] == 3
    assert "agents" not in record.method_config
    assert record.prompts_sha256 == ctx.prompts.fingerprint()
    # The recorded limit wins over a context that is re-created with the same configuration.
    out = await app.ainvoke(Command(resume="ok"), config, context=ctx)
    assert out["cycles"] == 3


async def test_a_changed_method_field_is_refused_on_resume(make_context: Any) -> None:
    from langgraph.types import Command

    app = _graph()
    config = await _start(app, make_context(), "method")
    changed = make_context(
        config=ScientistConfig(investigation=InvestigationConfig(max_idea_cycles=7))
    )
    with pytest.raises(ConfigurationMismatch, match=r"investigation\.max_idea_cycles"):
        await app.ainvoke(Command(resume="ok"), config, context=changed)
    out = await app.ainvoke(Command(resume="ok"), config, context=make_context())
    assert out["cycles"] == 20


async def test_a_changed_prompt_is_refused(make_context: Any) -> None:
    app = _graph()
    config = await _start(app, make_context(), "prompt")
    prompts = PromptRegistry.default().with_overrides({"harness/nl": "different"})
    with pytest.raises(ConfigurationMismatch, match="prompt:harness/nl"):
        await record_run_config(
            (await app.aget_state(config)).values, _runtime(make_context(prompts=prompts))
        )


async def test_a_changed_runtime_only_field_is_accepted(make_context: Any) -> None:
    app = _graph()
    config = await _start(app, make_context(), "runtime")
    runtime_change = make_context(
        config=ScientistConfig(agents=AgentsConfig(execute_timeout_seconds=30))
    )
    assert (
        await record_run_config((await app.aget_state(config)).values, _runtime(runtime_change))
        == {}
    )
    assert "agents" in runtime_only_paths() and "review.sampling" in runtime_only_paths()


async def test_an_older_schema_version_is_refused(make_context: Any) -> None:
    app = _graph()
    config = await _start(app, make_context(), "schema")
    state = (await app.aget_state(config)).values
    old = state["run_config"].model_copy(update={"state_schema_version": 0})
    with pytest.raises(IncompatibleCheckpoint):
        await record_run_config({"run_config": old}, _runtime(make_context()))


def _runtime(ctx: ScientistContext) -> Any:
    from langgraph.runtime import Runtime

    return Runtime(context=ctx)
