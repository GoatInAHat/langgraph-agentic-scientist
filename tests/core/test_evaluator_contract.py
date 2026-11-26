"""The evaluator contract: idempotence, re-attachment, once-only, and upstream's output rules."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.core.contract import load_task_contract
from langgraph_agentic_scientist.core.errors import (
    EvaluatorOutputContractError,
    FinalEvaluationAlreadyStarted,
    FinalEvaluationError,
)
from langgraph_agentic_scientist.core.state import ProgramRef, TaskRef
from langgraph_agentic_scientist.evaluation import (
    EvaluationRequest,
    FinalEvaluationRequest,
    FreezeRecord,
    LocalCommandEvaluator,
    LocalFinalEvaluator,
)
from langgraph_agentic_scientist.evaluation.evidence import (
    decode_evaluation_evidence,
    evidence_metadata,
    validate_compute_usage,
    validate_evaluator_outputs,
    validate_summaries,
)
from langgraph_agentic_scientist.evaluation.local import validate_final_manifest
from langgraph_agentic_scientist.testing import (
    FakeEvaluator,
    FakeFinalEvaluator,
    write_minimal_task,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "core"


def _request(
    task: TaskRef, evaluation_id: str, program: str = "task/program_0"
) -> EvaluationRequest:
    return EvaluationRequest(
        evaluation_id=evaluation_id,
        program=ProgramRef(path=program),
        task=task,
        timeout_seconds=60,
    )


# --- LocalCommandEvaluator ------------------------------------------------------------------


async def test_local_evaluator_runs_once_per_id(workspace: LocalWorkspace) -> None:
    task = await write_minimal_task(workspace)
    evaluator = LocalCommandEvaluator(workspace)
    first = await evaluator.evaluate(_request(task, "run/w0/baseline"))
    assert first.status == "completed", first.summary
    assert first.metrics["combined_score"] == 1.0 and first.correct is True
    stdout = await workspace.read_bytes(".evaluations/run/w0/baseline/stdout.log")
    assert b"combined_score=1.0" in stdout
    await workspace.remove(".evaluations/run/w0/baseline/stdout.log")
    second = await evaluator.evaluate(_request(task, "run/w0/baseline"))
    assert second == first
    assert not await workspace.exists(".evaluations/run/w0/baseline/stdout.log"), "it ran again"


async def test_local_evaluator_reports_interrupted_for_an_unrecorded_start(
    workspace: LocalWorkspace,
) -> None:
    task = await write_minimal_task(workspace)
    await workspace.write_bytes(".evaluations/run/e1/invocation.json", b"{}")
    result = await LocalCommandEvaluator(workspace).evaluate(_request(task, "run/e1"))
    assert result.status == "interrupted"


async def test_local_evaluator_failures_are_classified(workspace: LocalWorkspace) -> None:
    task = await write_minimal_task(workspace, program="raise SystemExit(3)\n")
    failed = await LocalCommandEvaluator(workspace).evaluate(_request(task, "run/fail"))
    assert failed.status == "failed" and failed.failure_category == "evaluator_failed"
    await workspace.write_bytes("task/program_0/program_spec.toml", b"schema_version = 2\n")
    spec = await LocalCommandEvaluator(workspace).evaluate(_request(task, "run/spec"))
    assert spec.status == "failed" and spec.failure_category == "invalid_program_spec"
    await workspace.remove("task/evaluate.py")
    missing = await LocalCommandEvaluator(workspace).evaluate(_request(task, "run/missing"))
    assert missing.status == "infrastructure_error"


async def test_contracted_outputs_are_validated(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "contract_task", tmp_path / "ws" / "task")
    workspace = LocalWorkspace(tmp_path / "ws")
    contract = await load_task_contract(workspace, "task")
    task = TaskRef(root="task", contract=contract, harness_evaluated=True)
    result = await LocalCommandEvaluator(workspace).evaluate(_request(task, "run/c1"))
    assert result.status == "completed", result.summary
    assert set(result.metrics) == {"combined_score", "accuracy"}
    assert result.evidence is not None and result.evidence.episodes[0].seed == 1
    narrowed = task.model_copy(
        update={
            "contract": contract.model_copy(
                update={
                    "metrics": contract.metrics.model_copy(
                        update={"search": contract.metrics.search[:1]}
                    )
                }
            )
        }
    )
    invalid = await LocalCommandEvaluator(workspace).evaluate(_request(narrowed, "run/c2"))
    assert invalid.status == "failed"
    assert invalid.failure_category == "evaluator_output_contract_invalid"


# --- FakeEvaluator ----------------------------------------------------------------------------


async def test_fake_evaluator_is_idempotent_and_reattaches(workspace: LocalWorkspace) -> None:
    task = await write_minimal_task(workspace)
    fake = FakeEvaluator(fail_once=["run/x"], reattach=True)
    with pytest.raises(ConnectionError):
        await fake.evaluate(_request(task, "run/x"))
    eventual = await fake.evaluate(_request(task, "run/x"))
    assert eventual.status == "completed"
    assert fake.runs["run/x"] == 1
    assert await fake.evaluate(_request(task, "run/x")) == eventual
    lossy = FakeEvaluator(fail_once=["run/y"], reattach=False)
    with pytest.raises(ConnectionError):
        await lossy.evaluate(_request(task, "run/y"))
    assert (await lossy.evaluate(_request(task, "run/y"))).status == "interrupted"
    assert lossy.runs["run/y"] == 1


# --- LocalFinalEvaluator ----------------------------------------------------------------------


async def _contracted(tmp_path: Path) -> tuple[LocalWorkspace, TaskRef, FreezeRecord]:
    shutil.copytree(FIXTURES / "contract_task", tmp_path / "ws" / "task")
    workspace = LocalWorkspace(tmp_path / "ws")
    contract = await load_task_contract(workspace, "task")
    task = TaskRef(root="task", contract=contract, harness_evaluated=True)
    from langgraph_agentic_scientist.core.contract import task_scaffold_fingerprint

    frozen = FreezeRecord(
        program=ProgramRef(path="task/program_0"),
        program_tree_sha256=await workspace.digest("task/program_0"),
        task_tree_sha256=await task_scaffold_fingerprint(workspace, "task"),
        evaluator_sha256=await workspace.digest("task/evaluate_final.py"),
        frozen_at="2026-01-01T00:00:00+00:00",
    )
    return workspace, task, frozen


async def test_local_final_evaluator_is_once_only(tmp_path: Path) -> None:
    workspace, task, frozen = await _contracted(tmp_path)
    evaluator = LocalFinalEvaluator(workspace)
    request = FinalEvaluationRequest(evaluation_id="run/final", frozen=frozen, task=task)
    result = await evaluator.evaluate_sealed(request)
    assert result.declared_final_metrics == ["final_score"]
    assert result.outcome == "qualified" and result.correct is True
    assert result.compute_usage is not None and result.compute_usage.max_rpc_calls == 100
    manifest = json.loads(
        await workspace.read_bytes(".final_evaluations/run/final/final_manifest.json")
    )
    assert manifest["schema_version"] == 3
    replay = await evaluator.evaluate_sealed(request)
    assert replay.results_tree_sha256 == result.results_tree_sha256
    await workspace.write_bytes(".final_evaluations/run/other/invocation.json", b"{}")
    with pytest.raises(FinalEvaluationAlreadyStarted):
        await evaluator.evaluate_sealed(request.model_copy(update={"evaluation_id": "run/other"}))


async def test_final_evaluation_refuses_a_changed_program(tmp_path: Path) -> None:
    workspace, task, frozen = await _contracted(tmp_path)
    await workspace.write_bytes("task/program_0/src/program.py", b"def handle(r):\n    return 9\n")
    with pytest.raises(FinalEvaluationError, match="program digest mismatch"):
        await LocalFinalEvaluator(workspace).evaluate_sealed(
            FinalEvaluationRequest(evaluation_id="run/final", frozen=frozen, task=task)
        )


async def test_fake_final_evaluator_refuses_a_restart() -> None:
    from langgraph_agentic_scientist.core.state import ProgramRef as Ref

    frozen = FreezeRecord(
        program=Ref(path="p"),
        program_tree_sha256="sha256:" + "0" * 64,
        task_tree_sha256="sha256:" + "1" * 64,
        evaluator_sha256="sha256:" + "2" * 64,
        frozen_at="now",
    )
    fake = FakeFinalEvaluator(fail_once=True)
    request = FinalEvaluationRequest(evaluation_id="r/final", frozen=frozen, task=TaskRef(root="t"))
    with pytest.raises(ConnectionError):
        await fake.evaluate_sealed(request)
    with pytest.raises(FinalEvaluationAlreadyStarted):
        await fake.evaluate_sealed(request)
    assert fake.runs == 1


def test_final_manifest_rules() -> None:
    digest = "sha256:" + "a" * 64
    manifest = {
        "schema_version": 2,
        "status": "completed",
        "started_at": "s",
        "completed_at": "c",
        "correct": False,
        "outcome": "not_qualified",
        "metrics": {"m": 1.5},
        "declared_final_metrics": ["m"],
        "task_tree_sha256": digest,
        "evaluator_sha256": digest,
        "program_tree_sha256": digest,
        "results_tree_sha256": digest,
    }
    assert validate_final_manifest(manifest).outcome == "not_qualified"
    with pytest.raises(FinalEvaluationError, match="outcome does not match"):
        validate_final_manifest({**manifest, "outcome": "qualified"})
    with pytest.raises(FinalEvaluationError, match="unexpected fields"):
        validate_final_manifest({**manifest, "extra": 1})
    with pytest.raises(FinalEvaluationError, match="not numeric"):
        validate_final_manifest({**manifest, "metrics": {"m": True}})


# --- evidence and output shapes ----------------------------------------------------------------


def _episode(seed: int, **extra: object) -> dict[str, object]:
    return {"seed": seed, "score": 1.0, "correct": True, "duration_seconds": 0.5, **extra}


def test_evidence_validation_limits() -> None:
    good = {"schema_version": 1, "episodes": [_episode(1), _episode(2)]}
    assert decode_evaluation_evidence(json.dumps(good).encode())["episodes"][1]["seed"] == 2
    cases = {
        "exceeds the 16 MiB limit": b" " * (16 * 1024 * 1024 + 1),
        "duplicate keys": b'{"schema_version": 1, "schema_version": 1, "episodes": []}',
        "duplicate seeds": json.dumps(
            {"schema_version": 1, "episodes": [_episode(1), _episode(1)]}
        ).encode(),
        "nesting depth": json.dumps(
            {"schema_version": 1, "episodes": [_episode(1, solution=[[[[[[[[[[[[[1]]]]]]]]]]]]])]}
        ).encode(),
        "signed 64-bit": json.dumps({"schema_version": 1, "episodes": [_episode(2**63)]}).encode(),
        "named finite numbers": json.dumps(
            {"schema_version": 1, "episodes": [_episode(1, candidate_diagnostics={"bad name!": 1})]}
        ).encode(),
        "series name is invalid": json.dumps(
            {
                "schema_version": 1,
                "episodes": [
                    _episode(1, candidate_timeseries=[{"series_name": "", "step": 0, "value": 1}])
                ],
            }
        ).encode(),
        "count limit or are empty": json.dumps({"schema_version": 1, "episodes": []}).encode(),
        "undeclared fields": json.dumps(
            {"schema_version": 1, "episodes": [_episode(1, extra=1)]}
        ).encode(),
        "schema_version must be 1": json.dumps(
            {"schema_version": 2, "episodes": [_episode(1)]}
        ).encode(),
    }
    for message, raw in cases.items():
        with pytest.raises(ValueError, match=message):
            decode_evaluation_evidence(raw)
    rows = [{"series_name": "loss", "step": index, "value": 1.0} for index in range(100_001)]
    with pytest.raises(ValueError, match="row limit"):
        decode_evaluation_evidence(
            json.dumps(
                {"schema_version": 1, "episodes": [_episode(1, candidate_timeseries=rows)]}
            ).encode()
        )
    metadata = evidence_metadata(json.dumps(good).encode())
    assert metadata.status == "available" and metadata.episode_count == 2
    assert evidence_metadata(None).status == "unavailable"


def test_metrics_and_correctness_shapes() -> None:
    metrics, correct = validate_summaries(
        b'{"combined_score": 2.5, "extra": "x"}', b'{"correct": false}'
    )
    assert metrics["combined_score"] == 2.5 and correct is False
    for raw, message in (
        (b'{"combined_score": true}', "must be numeric"),
        (b'{"combined_score": NaN}', "must be finite"),
    ):
        with pytest.raises(RuntimeError, match=message):
            validate_summaries(raw, b'{"correct": true}')
    with pytest.raises(RuntimeError, match="must be a boolean"):
        validate_summaries(b'{"combined_score": 1}', b'{"correct": 1}')
    validate_evaluator_outputs(
        b'{"combined_score": 1}', b'{"correct": true}', None, ["combined_score"]
    )
    for metrics_raw, correct_raw, message in (
        (b'{"combined_score": 1, "x": 2}', b'{"correct": true}', "exactly match"),
        (b'{"combined_score": 1}', b'{"correct": true, "y": 1}', "exactly one boolean"),
        (b'{"combined_score": 1, "combined_score": 2}', b'{"correct": true}', "duplicate keys"),
        (None, b'{"correct": true}', "missing"),
    ):
        with pytest.raises(EvaluatorOutputContractError, match=message):
            validate_evaluator_outputs(metrics_raw, correct_raw, None, ["combined_score"])


def test_compute_usage_rules() -> None:
    usage = {
        "wall_clock_limit_seconds": 10,
        "elapsed_seconds": 1,
        "max_rpc_calls": 5,
        "rpc_calls": 6,
        "outcome": "completed",
        "objective_counts": "advisory",
    }
    with pytest.raises(ValueError, match="exceeds RPC limit"):
        validate_compute_usage(usage)
    validate_compute_usage({**usage, "rpc_calls": None})
