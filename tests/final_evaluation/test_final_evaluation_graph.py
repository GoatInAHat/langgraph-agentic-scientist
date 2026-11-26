"""The sealed final evaluation graph: once-only, freeze, drawn inputs, verification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langgraph_agentic_scientist import (
    FinalEvaluationAlreadyStarted,
    GraphOverrides,
    LocalWorkspace,
    ScientistConfig,
)
from langgraph_agentic_scientist.core.contract import load_task_contract
from langgraph_agentic_scientist.core.results import DownstreamSelection, InvestigationResult
from langgraph_agentic_scientist.core.state import BestManifest, ProgramRef, TaskRef
from langgraph_agentic_scientist.evaluation import (
    DeferredFinalEvaluator,
    EvidenceMetadata,
    FinalEvaluationRequest,
    FinalEvaluationResult,
    LocalFinalEvaluator,
    SealedInputs,
)
from langgraph_agentic_scientist.final_evaluation import (
    DRAW_INPUTS_SLOT,
    build_final_evaluation_graph,
    resolve_incumbent,
)
from langgraph_agentic_scientist.final_evaluation.manifest import verify_final_result
from langgraph_agentic_scientist.testing import (
    FakeFinalEvaluator,
    write_minimal_task,
    write_program,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "core"
CONFIG = ScientistConfig(allow_unisolated_contracted=True)


async def _contracted_task(tmp_path: Path, workspace: LocalWorkspace) -> TaskRef:
    shutil.copytree(FIXTURES / "contract_task", tmp_path / "ws" / "task", dirs_exist_ok=True)
    contract = await load_task_contract(workspace, "task")
    return TaskRef(root="task", contract=contract, harness_evaluated=True)


def _input(task: TaskRef, **extra: Any) -> dict[str, Any]:
    return {
        "task": task,
        "run_id": "r1",
        "incumbent": ProgramRef(path="task/program_0"),
        **extra,
    }


def _config(thread: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}}


def _result_for(request: FinalEvaluationRequest, **update: Any) -> FinalEvaluationResult:
    contract = request.task.contract
    assert contract is not None
    final = contract.compute.final
    base = FakeFinalEvaluator()._answer(request)
    usage = base.compute_usage
    assert usage is not None
    result = base.model_copy(
        update={
            "declared_final_metrics": [metric.name for metric in contract.metrics.final],
            "metrics": {metric.name: 0.5 for metric in contract.metrics.final},
            "compute_usage": usage.model_copy(
                update={
                    "max_rpc_calls": final.max_candidate_rpc_calls,
                    "wall_clock_limit_seconds": final.wall_clock_seconds,
                }
            ),
            "evaluation_evidence": EvidenceMetadata(status="unavailable"),
            **update,
        }
    )
    return result


# --- the happy path and the shape of the request ------------------------------------------------


async def test_freeze_evaluate_verify(make_context: Any, tmp_path: Path, workspace: Any) -> None:
    task = await _contracted_task(tmp_path, workspace)
    fake = FakeFinalEvaluator(_result_for_callable)
    ctx = make_context(final_evaluator=fake, config=CONFIG)
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task), _config("ok"), context=ctx)
    assert out["final_status"] == "completed"
    assert out["final"].outcome == "qualified"
    (request,) = fake.calls
    assert request.evaluation_id == "r1/final"
    assert request.frozen.program.path == "task/program_0"
    assert request.frozen.program_tree_sha256 == await workspace.digest("task/program_0")
    assert request.sealed_inputs is None, "the default draw is the task's own final resources"
    assert out["frozen"] == request.frozen


def _result_for_callable(request: FinalEvaluationRequest) -> FinalEvaluationResult:
    return _result_for(request)


async def test_contracted_task_refuses_an_evaluator_that_does_not_isolate(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    fake = FakeFinalEvaluator(_result_for_callable, isolated=False)
    ctx = make_context(final_evaluator=fake)
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task), _config("iso"), context=ctx)
    assert out["final_status"] == "failed"
    assert out["failure"].detail == "contracted final evaluation requires container isolation"
    assert fake.calls == []
    allowed = make_context(final_evaluator=fake, config=CONFIG)
    out = await app.ainvoke(_input(task), _config("iso2"), context=allowed)
    assert out["final_status"] == "completed"
    assert any(event.kind == "unisolated_contracted_final_evaluation" for event in out["events"])


async def test_missing_evaluator_and_missing_incumbent_fail_the_phase(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task), _config("none"), context=make_context())
    assert out["failure"].category == "final_evaluator_missing"
    ctx = make_context(final_evaluator=FakeFinalEvaluator(), config=CONFIG)
    bare = {"task": task, "run_id": "r1"}
    out = await app.ainvoke(bare, _config("noinc"), context=ctx)
    assert out["failure"].detail == "best-program manifest has no frozen incumbent"


# --- resolving the frozen incumbent -------------------------------------------------------------


async def test_downstream_selection_wins_over_the_best_program(workspace: Any) -> None:
    await write_program(workspace, "results/w00/program_1", "SCORE = 2.0\n")
    await write_program(workspace, "downstream/program", "SCORE = 1.5\n")
    investigation = InvestigationResult(
        status="completed",
        best=BestManifest(program_dir="results/w00/program_1", combined_score=2.0, correct=True),
        results_dir="results",
    )
    downstream = DownstreamSelection(
        status="selected", program=ProgramRef(path="downstream/program"), combined_score=1.5
    )
    chosen = await resolve_incumbent(workspace, downstream=downstream, investigation=investigation)
    assert chosen.source == "downstream" and chosen.program.path == "downstream/program"
    best = await resolve_incumbent(workspace, investigation=investigation)
    assert best.source == "best" and best.program.entrypoint == "src/program.py"
    skipped = DownstreamSelection(status="skipped")
    assert (
        await resolve_incumbent(workspace, downstream=skipped, investigation=investigation)
    ).source == "best"


async def test_incumbent_must_be_inside_the_results_and_have_a_safe_entrypoint(
    workspace: Any,
) -> None:
    from langgraph_agentic_scientist.core.errors import FinalEvaluationError

    await write_program(workspace, "elsewhere/program_1", "SCORE = 2.0\n")
    outside = InvestigationResult(
        status="completed",
        best=BestManifest(program_dir="elsewhere/program_1"),
        results_dir="results",
    )
    with pytest.raises(FinalEvaluationError, match="escapes investigation results"):
        await resolve_incumbent(workspace, investigation=outside)
    escaping = InvestigationResult(
        status="completed",
        best=BestManifest(program_dir="elsewhere/program_1", entrypoint_path="../x.py"),
    )
    with pytest.raises(FinalEvaluationError, match="invalid frozen incumbent entrypoint"):
        await resolve_incumbent(workspace, investigation=escaping)


# --- the sealed inputs are drawn once, after the freeze -----------------------------------------


async def test_draw_inputs_runs_once_and_its_inputs_reach_the_evaluator(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    draws: list[str] = []

    async def draw(value: dict[str, Any], runtime: Any) -> dict[str, Any]:
        draws.append(value["frozen"].program_tree_sha256)
        assert value["frozen"].sealed_inputs is None
        return {
            "sealed_inputs": SealedInputs(
                ref="seeds://drawn", digest="sha256:" + "a" * 64, drawn_at="2026-01-02T00:00:00Z"
            )
        }

    fake = FakeFinalEvaluator(_result_for_callable)
    ctx = make_context(final_evaluator=fake, config=CONFIG)
    app = build_final_evaluation_graph(
        checkpointer=InMemorySaver(),
        overrides=GraphOverrides(slots={DRAW_INPUTS_SLOT: draw}),
    )
    out = await app.ainvoke(_input(task), _config("draw"), context=ctx)
    assert out["final_status"] == "completed"
    (request,) = fake.calls
    assert request.sealed_inputs is not None and request.sealed_inputs.ref == "seeds://drawn"
    assert request.frozen.sealed_inputs == request.sealed_inputs
    assert out["frozen"].sealed_inputs == request.sealed_inputs
    assert draws == [request.frozen.program_tree_sha256]


async def test_draw_inputs_runs_once_across_an_interrupted_evaluate_sealed(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    draws: list[int] = []

    async def draw(value: dict[str, Any], runtime: Any) -> SealedInputs:
        draws.append(len(draws))
        return SealedInputs(ref=f"draw-{len(draws)}", drawn_at="2026-01-02T00:00:00Z")

    ctx = make_context(final_evaluator=DeferredFinalEvaluator(), config=CONFIG)
    app = build_final_evaluation_graph(
        checkpointer=InMemorySaver(),
        overrides=GraphOverrides(slots={DRAW_INPUTS_SLOT: draw}),
    )
    config = _config("interrupted")
    out = await app.ainvoke(_input(task), config, context=ctx)
    (pending,) = out["__interrupt__"]
    payload = pending.value
    json.dumps(payload)
    assert payload["kind"] == "final_evaluation_request"
    request = FinalEvaluationRequest.model_validate(payload["request"])
    assert request.sealed_inputs is not None and request.sealed_inputs.ref == "draw-1"
    assert draws == [0]
    result = _result_for(request).model_dump(mode="json")
    final = await app.ainvoke(Command(resume=result), config, context=ctx)
    assert final["final_status"] == "completed"
    assert draws == [0], "the sealed inputs were drawn again after the interrupt"
    assert final["frozen"].sealed_inputs.ref == "draw-1"


async def test_a_crash_during_the_evaluation_never_repeats_the_draw_or_the_evaluation(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    draws: list[int] = []

    async def draw(value: dict[str, Any], runtime: Any) -> SealedInputs:
        draws.append(1)
        return SealedInputs(ref="once", drawn_at="2026-01-02T00:00:00Z")

    fake = FakeFinalEvaluator(_result_for_callable, fail_once=True)
    ctx = make_context(final_evaluator=fake, config=CONFIG)
    app = build_final_evaluation_graph(
        checkpointer=InMemorySaver(),
        overrides=GraphOverrides(slots={DRAW_INPUTS_SLOT: draw}),
    )
    config = _config("crash")
    with pytest.raises(ConnectionError):
        await app.ainvoke(_input(task), config, context=ctx)
    out = await app.ainvoke(None, config, context=ctx)
    assert out["final_status"] == "failed"
    assert out["failure"].category == "final_evaluation_already_started"
    assert "already started; refusing rerun" in out["failure"].detail
    assert draws == [1]
    assert fake.runs == 1, "the sealed evaluation ran a second time"
    assert out["final"] is None if "final" in out else True


# --- once-only replay with the local evaluator --------------------------------------------------


async def test_a_second_run_replays_the_verified_manifest(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    ctx = make_context(final_evaluator=LocalFinalEvaluator(workspace), config=CONFIG)
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    first = await app.ainvoke(_input(task), _config("first"), context=ctx)
    assert first["final_status"] == "completed", first.get("failure")
    manifest_path = tmp_path / "ws" / ".final_evaluations" / "r1" / "final" / "final_manifest.json"
    before = manifest_path.read_bytes()
    second = await app.ainvoke(_input(task), _config("second"), context=ctx)
    assert second["final_status"] == "completed"
    assert second["final"].started_at == first["final"].started_at
    assert second["final"].results_tree_sha256 == first["final"].results_tree_sha256
    assert manifest_path.read_bytes() == before, "the sealed evaluator ran again"


async def test_an_invocation_without_a_manifest_is_refused(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    invocation = tmp_path / "ws" / ".final_evaluations" / "r1" / "final" / "invocation.json"
    invocation.parent.mkdir(parents=True)
    invocation.write_text("{}")
    ctx = make_context(final_evaluator=LocalFinalEvaluator(workspace), config=CONFIG)
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task), _config("refused"), context=ctx)
    assert out["final_status"] == "failed"
    assert out["failure"].category == "final_evaluation_already_started"
    with pytest.raises(FinalEvaluationAlreadyStarted):
        await LocalFinalEvaluator(workspace).evaluate_sealed(
            FinalEvaluationRequest(evaluation_id="r1/final", frozen=out["frozen"], task=task)
        )


async def test_a_not_qualified_outcome_is_a_completed_phase(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    (tmp_path / "ws" / "task" / "program_0" / "src" / "program.py").write_text(
        "def handle(request):\n    return 100.0\n"
    )
    ctx = make_context(final_evaluator=LocalFinalEvaluator(workspace), config=CONFIG)
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task), _config("nq"), context=ctx)
    assert out["final_status"] == "completed", out.get("failure")
    assert out["final"].outcome == "not_qualified" and out["final"].correct is False
    assert out["final"].manifest_schema_version == 3


# --- verification -------------------------------------------------------------------------------


async def test_a_program_changed_during_the_evaluation_is_detected(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    program = tmp_path / "ws" / "task" / "program_0" / "src" / "program.py"

    def tamper(request: FinalEvaluationRequest) -> FinalEvaluationResult:
        program.write_text(program.read_text() + "# changed while the evaluation ran\n")
        return _result_for(request)

    ctx = make_context(final_evaluator=FakeFinalEvaluator(tamper), config=CONFIG)
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task), _config("tamper"), context=ctx)
    assert out["final_status"] == "failed"
    assert out["failure"].category == "final_manifest_invalid"
    assert out["failure"].detail == "frozen incumbent changed during final evaluation"


@pytest.mark.parametrize(
    ("update", "message"),
    [
        ({"program_tree_sha256": "sha256:" + "0" * 64}, "final program digest mismatch"),
        ({"task_tree_sha256": "sha256:" + "0" * 64}, "final task digest mismatch"),
        ({"evaluator_sha256": "sha256:" + "0" * 64}, "final evaluator digest mismatch"),
        (
            {"declared_final_metrics": ["other"], "metrics": {"other": 1.0}},
            "differ from task contract",
        ),
        ({"manifest_schema_version": 4}, "unsupported schema version"),
        ({"manifest_schema_version": 1, "correct": False, "outcome": "not_qualified"}, "schema-v1"),
        ({"outcome": "not_qualified"}, "outcome does not match correct"),
        ({"metrics": {"final_score": float("nan")}}, "not finite"),
        ({"compute_usage": None}, "compute usage"),
    ],
)
async def test_manifest_rules_are_applied_to_the_returned_result(
    make_context: Any, tmp_path: Path, workspace: Any, update: dict[str, Any], message: str
) -> None:
    task = await _contracted_task(tmp_path, workspace)
    ctx = make_context(
        final_evaluator=FakeFinalEvaluator(lambda request: _result_for(request, **update)),
        config=CONFIG,
    )
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task), _config(message[:8]), context=ctx)
    assert out["final_status"] == "failed", out
    assert out["failure"].category == "final_manifest_invalid"
    assert message in out["failure"].detail
    assert out["final"] is None


async def test_compute_limits_must_match_the_contract(
    make_context: Any, tmp_path: Path, workspace: Any
) -> None:
    task = await _contracted_task(tmp_path, workspace)

    def too_generous(request: FinalEvaluationRequest) -> FinalEvaluationResult:
        base = _result_for(request)
        assert base.compute_usage is not None
        return base.model_copy(
            update={"compute_usage": base.compute_usage.model_copy(update={"max_rpc_calls": 7})}
        )

    ctx = make_context(final_evaluator=FakeFinalEvaluator(too_generous), config=CONFIG)
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(_input(task), _config("limits"), context=ctx)
    assert out["failure"].detail == "final compute usage limits differ from task contract"


async def test_schema_one_and_two_results_are_replayed(make_context: Any, workspace: Any) -> None:
    task = await write_minimal_task(workspace)
    await workspace.write_bytes("task/evaluate_final.py", b"# sealed\n")
    ctx = make_context(
        final_evaluator=FakeFinalEvaluator(
            lambda request: (
                FakeFinalEvaluator()
                ._answer(request)
                .model_copy(update={"manifest_schema_version": 1, "compute_usage": None})
            )
        )
    )
    app = build_final_evaluation_graph(checkpointer=InMemorySaver())
    out = await app.ainvoke(
        {"task": task, "run_id": "r2", "incumbent": ProgramRef(path="task/program_0")},
        _config("v1"),
        context=ctx,
    )
    assert out["final_status"] == "completed", out.get("failure")
    assert out["final"].manifest_schema_version == 1


async def test_verify_final_result_checks_the_results_digest_when_it_can_read_them(
    workspace: Any,
) -> None:
    task = await write_minimal_task(workspace)
    await workspace.write_bytes("task/evaluate_final.py", b"# sealed\n")
    from langgraph_agentic_scientist.core.errors import FinalEvaluationError
    from langgraph_agentic_scientist.final_evaluation import freeze_incumbent

    frozen = await freeze_incumbent(workspace, task, ProgramRef(path="task/program_0"))
    await workspace.write_bytes("kept/final/results/metrics.json", b'{"combined_score": 1}')
    request = FinalEvaluationRequest(evaluation_id="r/final", frozen=frozen, task=task)
    result = (
        FakeFinalEvaluator()
        ._answer(request)
        .model_copy(update={"output_dir": "kept/final", "manifest_schema_version": 2})
    )
    with pytest.raises(FinalEvaluationError, match="final results digest mismatch"):
        await verify_final_result(workspace, task=task, frozen=frozen, result=result)
    good = result.model_copy(
        update={"results_tree_sha256": await workspace.digest("kept/final/results")}
    )
    assert (await verify_final_result(workspace, task=task, frozen=frozen, result=good)).correct
