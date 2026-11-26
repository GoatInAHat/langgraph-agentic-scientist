"""The contract lint and the mechanical contract synchronisation (upstream rules and messages)."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

import pytest
from test_scaffold_support import FIXTURE, write_scaffold

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.drafting.contract_lint import (
    ContractSyncError,
    contract_findings,
    declared_dependencies_match_requirements,
    isolated_rpc_transport_findings,
    preflight_findings,
    retained_declared_dependencies,
    synchronize_declared_dependency_removals,
    synchronize_declared_digests,
    unavailable_harness_environment_findings,
)

CONTRACT: dict[str, Any] = json.loads((FIXTURE / "task_contract.json").read_text())


def findings_of(mutate: Any, requirements: bytes | None = b"python==3.12\n") -> tuple[str, ...]:
    contract = copy.deepcopy(CONTRACT)
    mutate(contract)
    return contract_findings(
        json.dumps(contract).encode(),
        requirements=requirements,
        require_explicit_rpc_budgets=True,
    )


def test_the_fixture_contract_has_no_finding() -> None:
    assert findings_of(lambda contract: None) == ()


def test_every_independent_finding_is_reported_at_once() -> None:
    def mutate(contract: dict[str, Any]) -> None:
        contract["candidate"]["interface"] = "other:handle"
        contract["candidate"]["editable_paths"] = ["program_*/src"]
        contract["evaluators"]["search"] = "search.py"
        contract["resources"]["final_root"] = "elsewhere"
        contract["protocol"]["final_seeds"] = [1]
        contract["protocol"]["baselines"] = []
        contract["compute"]["search"]["accelerator"] = "tpu"
        del contract["compute"]["final"]["max_candidate_rpc_calls"]

    found = findings_of(mutate)
    assert "candidate.interface must be python_rpc_v1:handle" in found
    assert "candidate.editable_paths[0] must be a safe POSIX relative path" in found
    assert "evaluators.search must be evaluate.py" in found
    assert "resource roots must be resources/search and resources/final" in found
    assert "protocol search and final seeds must be disjoint" in found
    assert "protocol.baselines must be a non-empty array" in found
    assert "compute.search.accelerator must be 'cpu' or 'gpu'" in found
    assert "compute.final is missing required field(s): max_candidate_rpc_calls" in found


def test_reset_sequences_are_checked_against_seeds_and_the_rpc_budget() -> None:
    def mutate(contract: dict[str, Any]) -> None:
        contract["protocol"]["search_reset_sequence"] = [1, 1, 1]
        contract["compute"]["search"]["max_candidate_rpc_calls"] = 2
        contract["protocol"]["final_reset_sequence"] = [8]
        contract["protocol"]["final_smoke_reset_sequence"] = [9]

    found = findings_of(mutate)
    assert "protocol.search_reset_sequence exceeds compute.search.max_candidate_rpc_calls" in found
    assert "protocol.final_reset_sequence must cover exactly final_seeds" in found
    assert (
        "protocol.final_smoke_reset_sequence must contain only the first production reset seed"
        in found
    )


def test_requirements_must_pin_exactly_the_declared_dependencies() -> None:
    found = findings_of(lambda contract: None, requirements=b"python==3.12\nnumpy==1.26.0\n")
    assert any("requirements-only=['numpy==1.26.0']" in item for item in found)
    assert "requirements.txt must be a safe regular file" in findings_of(
        lambda contract: None, requirements=None
    )
    assert "requirements.txt must contain only pinned package==version entries" in findings_of(
        lambda contract: None, requirements=b"numpy>=1\n"
    )


def test_findings_are_bounded_and_not_json_is_left_to_the_parser() -> None:
    def mutate(contract: dict[str, Any]) -> None:
        contract["metrics"]["search"] = [{"x": index} for index in range(200)]
        contract["data"]["search_roles"] = ["Bad Role"] * 100

    found = findings_of(mutate)
    assert len(found) <= 65
    assert found[-1].endswith("additional bounded lint finding(s) omitted")
    assert contract_findings(b"not json") == ()


async def test_digests_of_declared_paths_are_refreshed_and_nothing_else(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    path = "task/resources/search/target.txt"
    await workspace.write_bytes(path, b"4.5\n")
    assert await synchronize_declared_digests(workspace, "task") is True
    contract = json.loads(await workspace.read_bytes("task/task_contract.json"))
    expected = hashlib.sha256(b"4.5\n").hexdigest()
    assert contract["provenance"]["artifacts"][0]["sha256"] == expected
    assert await synchronize_declared_digests(workspace, "task") is False
    assert await preflight_findings(workspace, "task") == ()


async def test_a_contract_that_needs_a_semantic_edit_is_refused(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    contract = copy.deepcopy(CONTRACT)
    contract["canonical_task"]["path"] = "other.md"
    await workspace.write_bytes("task/task_contract.json", json.dumps(contract).encode())
    with pytest.raises(ContractSyncError, match=r"canonical_task\.path must remain task\.md"):
        await synchronize_declared_digests(workspace, "task")
    duplicate = b'{"schema_version": 1, "schema_version": 1}'
    await workspace.write_bytes("task/task_contract.json", duplicate)
    with pytest.raises(ContractSyncError, match="duplicate JSON key"):
        await synchronize_declared_digests(workspace, "task")


async def test_a_repair_may_only_remove_a_declared_pin(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    contract = copy.deepcopy(CONTRACT)
    contract["provenance"]["dependencies"].append({"name": "numpy", "version": "1.26.0"})
    await workspace.write_bytes("task/task_contract.json", json.dumps(contract).encode())
    allowed = contract["provenance"]["dependencies"]
    await workspace.write_bytes("task/requirements.txt", b"python==3.12\n")
    assert await retained_declared_dependencies(
        workspace, "task", allowed_dependencies=allowed
    ) == [{"name": "python", "version": "3.12"}]
    assert not await declared_dependencies_match_requirements(workspace, "task")
    assert await synchronize_declared_dependency_removals(
        workspace, "task", allowed_dependencies=allowed
    )
    synchronized = json.loads(await workspace.read_bytes("task/task_contract.json"))
    assert synchronized["provenance"]["dependencies"] == [{"name": "python", "version": "3.12"}]
    assert await declared_dependencies_match_requirements(workspace, "task")
    await workspace.write_bytes("task/requirements.txt", b"python==3.12\nscipy==1.0.0\n")
    with pytest.raises(ContractSyncError, match="only remove pinned dependencies"):
        await retained_declared_dependencies(workspace, "task", allowed_dependencies=allowed)


async def test_the_retired_exchange_names_are_found_outside_docstrings(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    assert await isolated_rpc_transport_findings(workspace, "task") == ()
    await workspace.write_bytes(
        "task/evaluate.py",
        b'"""EXCHANGE_DIR in a docstring is fine."""\nimport os\nos.environ["EXCHANGE_DIR"]\n',
    )
    await workspace.write_bytes(
        "task/program_0/src/helper.py", b"x = 1\nAGENTIC_CANDIDATE_EXCHANGE = 2\n"
    )
    found = await isolated_rpc_transport_findings(workspace, "task")
    assert any(item.startswith("evaluate.py:3:") and "EXCHANGE_DIR" in item for item in found)
    assert any(item.startswith("program_0/src/helper.py:2:") for item in found)
    assert not any(item.startswith("evaluate.py:1:") for item in found)


async def test_evaluators_may_not_read_reserved_harness_values(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    source = (
        b'import os\nvalue = os.environ.get("AGENTIC_HARNESS_SECRET")\n'
        b"proxy.candidate_peak_accelerator_memory_gib\n"
    )
    await workspace.write_bytes("task/evaluate.py", source)
    findings = await unavailable_harness_environment_findings(
        workspace,
        "task",
        (("search_evaluator", "evaluate.py"), ("final_evaluator", "evaluate_final.py")),
    )
    assert [item.component for item in findings] == ["search_evaluator", "search_evaluator"]
    assert any("AGENTIC_HARNESS_SECRET" in item.diagnostic for item in findings)
    assert any("candidate_peak_accelerator_memory_gib" in item.diagnostic for item in findings)
