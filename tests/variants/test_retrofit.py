"""Stage 1: the retrofit seals the evaluator, preserves brokered candidates, and verifies itself."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from langgraph_agentic_scientist.testing import write_minimal_task
from langgraph_agentic_scientist.variants.retrofit import (
    finish_retrofit,
    retrofit_bundle_complete,
    seal_and_copy,
)
from langgraph_agentic_scientist.workspace import LocalWorkspace

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "core"


async def _legacy_bundle(workspace: LocalWorkspace) -> None:
    await write_minimal_task(workspace, "bundle")


async def test_the_agent_may_not_change_the_evaluator_and_the_marker_is_written_on_success(
    workspace: LocalWorkspace,
) -> None:
    await _legacy_bundle(workspace)
    report = await seal_and_copy(
        workspace, bundle_dir="bundle", output_bundle_dir="out", task_source_dir="bundle"
    )
    assert report.error is None and report.mode is None and report.exit_code == 1
    assert not (workspace.root / "out/.retrofit_complete.json").exists()
    # the agent instruments the program and tampers with the evaluator
    await workspace.write_bytes("out/program_0/src/program.py", b"SCORE = 1.0\n# instrumented\n")
    await workspace.write_bytes("out/evaluate.py", b"print('rigged')\n")
    finished = await finish_retrofit(workspace, report, session_ok=True)
    assert finished.error is None and finished.exit_code == 0
    assert (workspace.root / "out/evaluate.py").read_bytes() == (
        workspace.root / "bundle/evaluate.py"
    ).read_bytes()
    assert b"instrumented" in (workspace.root / "out/program_0/src/program.py").read_bytes()
    marker = json.loads((workspace.root / "out/.retrofit_complete.json").read_text())
    assert marker["schema_version"] == 1 and set(marker) == {
        "schema_version",
        "evaluator_sha256",
        "source_tree_sha256",
    }
    assert await retrofit_bundle_complete(workspace, "out", "bundle", task_source_dir="bundle")


async def test_a_failed_or_timed_out_session_restores_the_evaluator_and_writes_no_marker(
    workspace: LocalWorkspace,
) -> None:
    await _legacy_bundle(workspace)
    report = await seal_and_copy(
        workspace, bundle_dir="bundle", output_bundle_dir="out", task_source_dir=None
    )
    await workspace.write_bytes("out/evaluate.py", b"print('rigged')\n")
    finished = await finish_retrofit(
        workspace, report, session_ok=False, session_error="Agent exited with code 124"
    )
    assert finished.exit_code == 1 and finished.error == "Agent exited with code 124"
    assert (workspace.root / "out/evaluate.py").read_bytes() == (
        workspace.root / "bundle/evaluate.py"
    ).read_bytes()
    assert not (workspace.root / "out/.retrofit_complete.json").exists()
    assert not await retrofit_bundle_complete(workspace, "out", "bundle")


async def test_a_source_evaluator_that_changes_during_the_retrofit_is_detected(
    workspace: LocalWorkspace,
) -> None:
    await _legacy_bundle(workspace)
    report = await seal_and_copy(
        workspace, bundle_dir="bundle", output_bundle_dir="out", task_source_dir=None
    )
    await workspace.write_bytes("bundle/evaluate.py", b"print('changed under us')\n")
    finished = await finish_retrofit(workspace, report, session_ok=True)
    assert finished.error and "Sealed evaluator changed during retrofit" in finished.error
    assert not (workspace.root / "out/.retrofit_complete.json").exists()
    # an extra file in the source tree (not evaluate.py) is caught by the tree digest
    await write_minimal_task(workspace, "bundle")
    report = await seal_and_copy(
        workspace, bundle_dir="bundle", output_bundle_dir="out", task_source_dir=None
    )
    await workspace.write_bytes("bundle/helper.py", b"x = 1\n")
    finished = await finish_retrofit(workspace, report, session_ok=True)
    assert finished.error == "Sealed evaluator source changed during retrofit"


async def test_completion_marker_is_checked_against_the_bundle(workspace: LocalWorkspace) -> None:
    await _legacy_bundle(workspace)
    report = await seal_and_copy(
        workspace, bundle_dir="bundle", output_bundle_dir="out", task_source_dir=None
    )
    assert (await finish_retrofit(workspace, report, session_ok=True)).exit_code == 0
    assert await retrofit_bundle_complete(workspace, "out", "bundle")
    await workspace.write_bytes("out/evaluate.py", b"print('rigged')\n")
    assert not await retrofit_bundle_complete(workspace, "out", "bundle")
    await workspace.write_bytes(
        "out/evaluate.py", (workspace.root / "bundle/evaluate.py").read_bytes()
    )
    assert await retrofit_bundle_complete(workspace, "out", "bundle")
    await workspace.write_bytes("bundle/extra.txt", b"changed source\n")
    assert not await retrofit_bundle_complete(workspace, "out", "bundle")


async def test_the_task_runtime_is_never_copied(workspace: LocalWorkspace) -> None:
    await _legacy_bundle(workspace)
    await workspace.write_bytes("bundle/.venv/bin/python", b"#!")
    await seal_and_copy(
        workspace, bundle_dir="bundle", output_bundle_dir="out", task_source_dir=None
    )
    assert not (workspace.root / "out/.venv").exists()


async def test_a_brokered_candidate_is_copied_unchanged_and_needs_no_agent(
    tmp_path: Path,
) -> None:
    shutil.copytree(FIXTURES / "contract_task", tmp_path / "ws" / "bundle")
    workspace = LocalWorkspace(tmp_path / "ws")
    report = await seal_and_copy(
        workspace, bundle_dir="bundle", output_bundle_dir="out", task_source_dir="bundle"
    )
    assert report.error is None
    assert report.mode == "brokered_preserve" and report.exit_code == 0
    assert report.program_tree_sha256
    marker = json.loads((workspace.root / "out/.retrofit_complete.json").read_text())
    assert marker["schema_version"] == 2
    assert marker["copied_program_tree_sha256"] == report.program_tree_sha256
    assert marker["mode"] == "brokered_preserve" and marker["task_scaffold_sha256"]
    # finishing a brokered retrofit changes nothing, and the verification accepts it
    assert (await finish_retrofit(workspace, report, session_ok=True)) == report
    assert await retrofit_bundle_complete(workspace, "out", "bundle", task_source_dir="bundle")
    # a program that differs from the preserved one fails the verification
    await workspace.write_bytes("out/program_0/src/program.py", b"changed = True\n")
    assert not await retrofit_bundle_complete(workspace, "out", "bundle", task_source_dir="bundle")


async def test_an_unsupported_candidate_interface_stops_the_retrofit(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "contract_task", tmp_path / "ws" / "bundle")
    workspace = LocalWorkspace(tmp_path / "ws")
    # An invalid contract is refused before anything is copied.
    contract = json.loads((workspace.root / "bundle/task_contract.json").read_text())
    contract["candidate"]["interface"] = "other:handle"
    await workspace.write_bytes("bundle/task_contract.json", json.dumps(contract).encode())
    report = await seal_and_copy(
        workspace, bundle_dir="bundle", output_bundle_dir="out", task_source_dir="bundle"
    )
    assert report.error and report.error.startswith("Cannot seal evaluator")
    assert not (workspace.root / "out").exists()
