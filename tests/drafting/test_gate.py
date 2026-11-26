"""``LocalTaskGate``: contract preflight, smoke, production entry and qualification."""

from __future__ import annotations

import json
from typing import Any

import pytest
from test_scaffold_support import fixture_result, make_evaluator, write_scaffold

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.config import DraftingConfig
from langgraph_agentic_scientist.core.interfaces import GateReport
from langgraph_agentic_scientist.drafting import LocalTaskGate
from langgraph_agentic_scientist.drafting.gate import (
    RepairSessionGate,
    acceptance_budget_seconds,
    sanitize_diagnostic,
    sanitize_preflight_diagnostic,
    typed_smoke_diagnostic,
)
from langgraph_agentic_scientist.drafting.scaffold import (
    DEPENDENCY_AUTHORIZED,
    draft_paths,
    open_scoped_transaction,
)
from langgraph_agentic_scientist.evaluation.protocol import EvaluationRequest, EvaluationResult


async def staged(tmp_path: Any, **kwargs: Any) -> tuple[LocalWorkspace, Any]:
    workspace = LocalWorkspace(tmp_path)
    task = await write_scaffold(workspace, **kwargs)
    return workspace, task


async def check(gate: LocalTaskGate, name: str, workspace: Any, task: Any) -> GateReport:
    return await gate.run_check(name, task=task, workspace=workspace, evaluation_id="e")  # type: ignore[arg-type]


def answering(**by_plane: Any) -> Any:
    """An evaluator handler: ``by_plane`` maps ``<plane>/<evaluator>`` to a replacement result."""

    def handler(request: EvaluationRequest) -> EvaluationResult:
        key = f"{request.plane}/{request.evaluator}"
        answer = by_plane.get(key)
        if answer is None:
            return fixture_result(request)
        return (
            answer(request)
            if callable(answer)
            else answer.model_copy(update={"evaluation_id": request.evaluation_id})
        )

    return handler


async def test_preflight_passes_a_valid_scaffold_and_reports_its_fingerprint(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    report = await check(LocalTaskGate(), "preflight", workspace, task)
    assert report.passed and report.fingerprint and report.fingerprint.startswith("sha256:")


async def test_preflight_diagnostic_is_the_sanitised_single_line_text(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    text = (await workspace.read_bytes("task/task_contract.json")).decode()
    await workspace.write_bytes(
        "task/task_contract.json",
        text.replace("python_rpc_v1:handle", "python_rpc_v1:bad").encode(),
    )
    report = await check(LocalTaskGate(), "preflight", workspace, task)
    assert not report.passed and not report.infrastructure
    assert report.category == "contract_invalid"
    assert "\n" not in report.diagnostic
    assert report.diagnostic.endswith("\\x0a")
    assert "Contract findings:\\x0a- candidate.interface must be python_rpc_v1:handle" in (
        report.diagnostic
    )


async def test_preflight_refuses_a_modified_canonical_task(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    await workspace.write_bytes("task/task.md", b"# Another task\n")
    report = await check(LocalTaskGate(), "preflight", workspace, task)
    assert report.category == "scaffold_unsafe"
    assert "Draft or repair agent modified canonical task.md" in report.diagnostic


async def test_acceptance_runs_contract_smokes_and_production_entry_in_order(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    evaluator = make_evaluator()
    report = await check(LocalTaskGate(evaluator), "acceptance-smoke", workspace, task)
    assert report.passed and report.check == "acceptance-smoke"
    assert [(call.plane, call.evaluator) for call in evaluator.calls] == [
        ("smoke", "search"),
        ("smoke", "final"),
        ("production_entry", "search"),
        ("production_entry", "final"),
    ]
    assert evaluator.calls[0].environment["AGENTIC_TASK_DRAFT_SMOKE"] == "1"
    assert evaluator.calls[2].environment["AGENTIC_TASK_PRODUCTION_ENTRY"] == "1"
    assert evaluator.calls[0].timeout_seconds == 1800.0
    assert evaluator.calls[2].timeout_seconds == 300.0


async def test_a_non_isolated_evaluator_skips_the_production_entry(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    evaluator = make_evaluator(isolated=False)
    gate = LocalTaskGate(evaluator)
    assert (await check(gate, "acceptance-smoke", workspace, task)).passed
    assert [call.plane for call in evaluator.calls] == ["smoke", "smoke"]
    report = await check(gate, "production-entry", workspace, task)
    assert report.infrastructure and report.category == "production_entry_requires_container"


@pytest.mark.parametrize(
    ("answer", "authored", "category"),
    [
        (EvaluationResult(evaluation_id="x", status="timeout"), True, "search_timeout"),
        (
            EvaluationResult(evaluation_id="x", status="failed", summary="boom"),
            True,
            "search_nonzero_exit",
        ),
        (
            EvaluationResult(
                evaluation_id="x", status="failed", failure_category="output_limit_exceeded"
            ),
            True,
            "search_excessive_output",
        ),
        (
            EvaluationResult(
                evaluation_id="x", status="completed", metrics={"combined_score": 1.0}, correct=True
            ),
            True,
            "search_invalid_results",
        ),
        (
            EvaluationResult(
                evaluation_id="x",
                status="completed",
                metrics={"combined_score": float("nan"), "accuracy": 1.0},
                correct=True,
            ),
            True,
            "search_invalid_results",
        ),
        (
            EvaluationResult(
                evaluation_id="x",
                status="completed",
                metrics={"combined_score": 1.0, "accuracy": 1.0},
            ),
            True,
            "search_invalid_results",
        ),
        (
            EvaluationResult(evaluation_id="x", status="infrastructure_error"),
            False,
            "search_process_runtime",
        ),
        (
            EvaluationResult(
                evaluation_id="x",
                status="infrastructure_error",
                failure_category="runtime_bootstrap",
            ),
            False,
            "runtime_bootstrap",
        ),
        (
            EvaluationResult(evaluation_id="x", status="interrupted"),
            False,
            "evaluation_interrupted",
        ),
        (
            EvaluationResult(
                evaluation_id="x", status="failed", failure_category="runtime_dependencies"
            ),
            True,
            "runtime_dependencies",
        ),
    ],
)
async def test_smoke_failures_are_typed_authored_or_infrastructure(
    tmp_path: Any, answer: EvaluationResult, authored: bool, category: str
) -> None:
    workspace, task = await staged(tmp_path)
    gate = LocalTaskGate(make_evaluator(answering(**{"smoke/search": answer})))
    report = await check(gate, "search-smoke", workspace, task)
    assert not report.passed
    assert report.infrastructure is (not authored)
    assert report.category == category


async def test_the_final_evaluators_diagnostic_is_never_shown(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    answer = EvaluationResult(evaluation_id="x", status="failed", summary="sealed final detail")
    gate = LocalTaskGate(make_evaluator(answering(**{"smoke/final": answer})))
    report = await check(gate, "final-smoke", workspace, task)
    assert report.category == "final_nonzero_exit" and report.diagnostic == ""


async def test_an_evaluator_that_modifies_the_scaffold_fails_the_check(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)

    def tamper(request: EvaluationRequest) -> EvaluationResult:
        (tmp_path / "task" / "resources" / "search" / "target.txt").write_text("changed\n")
        return fixture_result(request)

    gate = LocalTaskGate(make_evaluator(tamper))
    report = await check(gate, "search-smoke", workspace, task)
    assert report.category == "evaluator_modified_scaffold" and not report.infrastructure


async def test_a_reserved_harness_value_in_an_evaluator_is_an_authored_failure(
    tmp_path: Any,
) -> None:
    workspace, task = await staged(tmp_path)
    await workspace.write_bytes("task/evaluate.py", b'import os\nos.environ["AGENTIC_HARNESS_X"]\n')
    report = await check(LocalTaskGate(make_evaluator()), "contract", workspace, task)
    assert report.category == "harness_capability_invalid"
    assert report.details["component"] == "search_evaluator"


async def test_the_resource_limit_is_a_bound(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    gate = LocalTaskGate(make_evaluator(), config=DraftingConfig(max_runtime_bytes=4))
    report = await check(gate, "acceptance-smoke", workspace, task)
    assert report.category == "runtime_resources_exceed_limit"
    assert "observed=8 limit=4" in report.diagnostic


async def test_qualification_needs_a_completed_feasible_baseline(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    evaluator = make_evaluator()
    report = await check(LocalTaskGate(evaluator), "qualification", workspace, task)
    assert report.passed and report.category == "qualification_passed"
    assert report.details["baseline_combined_score"] == -1.0
    call = evaluator.calls[0]
    assert call.plane == "qualification" and call.timeout_seconds == 57600.0
    assert call.environment["AGENTIC_TASK_QUALIFICATION"] == "1"
    infeasible = EvaluationResult(
        evaluation_id="x",
        status="completed",
        metrics={"combined_score": 1.0, "accuracy": 1.0},
        correct=False,
        feasible=False,
    )
    gate = LocalTaskGate(make_evaluator(answering(**{"qualification/search": infeasible})))
    report = await check(gate, "qualification", workspace, task)
    assert report.category == "qualification_infeasible" and not report.infrastructure
    timeout = EvaluationResult(evaluation_id="x", status="timeout", summary="slow")
    gate = LocalTaskGate(make_evaluator(answering(**{"qualification/search": timeout})))
    assert (await check(gate, "qualification", workspace, task)).category == "qualification_timeout"
    infra = EvaluationResult(
        evaluation_id="x", status="infrastructure_error", failure_category="gpu_allocation_busy"
    )
    gate = LocalTaskGate(make_evaluator(answering(**{"qualification/search": infra})))
    report = await check(gate, "qualification", workspace, task)
    assert report.infrastructure and report.category == "gpu_allocation_busy"


async def test_without_an_evaluator_a_smoke_is_an_infrastructure_failure(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    report = await check(LocalTaskGate(), "search-smoke", workspace, task)
    assert report.infrastructure and report.category == "search_no_evaluator"
    assert (await check(LocalTaskGate(), "bogus", workspace, task)).category == "unknown_check"  # type: ignore[arg-type]


def test_acceptance_budget_is_upstreams() -> None:
    assert acceptance_budget_seconds(DraftingConfig()) == 3900
    assert acceptance_budget_seconds(DraftingConfig(evaluator_smoke_timeout_seconds=7200)) == 7170


def test_sanitised_diagnostics_hide_credentials_and_are_bounded() -> None:
    assert sanitize_diagnostic("line\nnext\ttab\x00") == "line\\x0anext\\x09tab\\x00"
    secret = "token sk-abcdefghijklmnopqrstuvwxyz0123"
    assert sanitize_diagnostic(secret).startswith("<redacted-sensitive-text:")
    assert sanitize_diagnostic("fetched https://user:pw@host/x").startswith(
        "<redacted-sensitive-text:"
    )
    long = sanitize_diagnostic("a" * 20000)
    assert len(long.encode()) <= 16 * 1024 and long.endswith("…")
    tail = sanitize_preflight_diagnostic("x" * 40000 + "END")
    assert len(tail) == 32768 and tail.endswith("END")


def test_typed_smoke_diagnostic_withholds_the_authored_text() -> None:
    report = GateReport(
        check="search-smoke",
        passed=False,
        category="search_nonzero_exit",
        stage="execution",
        diagnostic="secret evaluator detail",
        fingerprint="sha256:" + "a" * 64,
        details={"component": "search_evaluator"},
    )
    payload = json.loads(typed_smoke_diagnostic(report))
    assert payload["status"] == "authored_failure" and payload["exit_code"] == 3
    assert payload["gate"] == "search-smoke" and payload["component"] == "search_evaluator"
    assert "secret evaluator detail" not in payload["diagnostic"]
    assert payload["diagnostic"].startswith("search-smoke failed with search_nonzero_exit")
    infra = report.model_copy(update={"infrastructure": True})
    assert json.loads(typed_smoke_diagnostic(infra))["diagnostic"] == "secret evaluator detail"


async def test_a_dependency_failure_in_a_repair_authorises_removal_once(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    paths = draft_paths("task")
    await workspace.copy_tree("task", paths.task)
    transaction = await open_scoped_transaction(
        workspace, paths, label="smoke_repair_0", policy="synchronized-digests"
    )
    failing = EvaluationResult(
        evaluation_id="x",
        status="failed",
        failure_category="runtime_dependencies",
        summary="no module",
    )
    inner = LocalTaskGate(make_evaluator(answering(**{"smoke/search": failing})))
    staged_task = task.model_copy(update={"root": paths.task})
    gate = RepairSessionGate(inner, paths=paths, transaction=transaction, label="smoke_repair_0")
    report = await gate.run_check(
        "search-smoke", task=staged_task, workspace=workspace, evaluation_id="e"
    )
    assert report.category == "runtime_dependencies"
    assert report.diagnostic.startswith(DEPENDENCY_AUTHORIZED)
    assert await workspace.exists(paths.authorization("smoke_repair_0"))
    # a second failing check still carries the authorisation
    again = await gate.run_check(
        "search-smoke", task=staged_task, workspace=workspace, evaluation_id="e2"
    )
    assert again.diagnostic.startswith(DEPENDENCY_AUTHORIZED)


async def test_authorisation_is_refused_after_requirements_changed(tmp_path: Any) -> None:
    workspace, task = await staged(tmp_path)
    paths = draft_paths("task")
    await workspace.copy_tree("task", paths.task)
    transaction = await open_scoped_transaction(
        workspace, paths, label="smoke_repair_0", policy="synchronized-digests"
    )
    await workspace.write_bytes(f"{paths.task}/requirements.txt", b"python==3.12\n# edited\n")
    failing = EvaluationResult(
        evaluation_id="x", status="failed", failure_category="runtime_dependencies"
    )
    inner = LocalTaskGate(make_evaluator(answering(**{"smoke/search": failing})))
    gate = RepairSessionGate(inner, paths=paths, transaction=transaction, label="smoke_repair_0")
    report = await gate.run_check(
        "search-smoke",
        task=task.model_copy(update={"root": paths.task}),
        workspace=workspace,
        evaluation_id="e",
    )
    assert report.infrastructure and report.category == "repair_authorization_failed"
    assert not await workspace.exists(paths.authorization("smoke_repair_0"))
