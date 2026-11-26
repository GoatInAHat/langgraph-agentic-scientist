"""Host-side scaffold handling: staging, fingerprints, inspection, canonical files, transactions,
the judge's view, report validation, publication and the alignment history."""

from __future__ import annotations

import copy
import json
import os
from typing import Any

import pytest
from test_scaffold_support import alignment_report, write_scaffold

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.drafting.history import (
    capture_alignment_report,
    render_alignment_history,
)
from langgraph_agentic_scientist.drafting.models import AlignmentEvidence
from langgraph_agentic_scientist.drafting.scaffold import (
    AlignmentReportError,
    PublicationError,
    RepairScopeError,
    ScaffoldError,
    alignment_fingerprints,
    bounded_repair_report,
    build_publication_plan,
    candidate_entrypoint,
    canonicalize_staged_scaffold,
    contract_policy_for_smoke_diagnostic,
    draft_paths,
    draft_values,
    inspect_scaffold,
    open_scoped_transaction,
    open_tree_transaction,
    prepare_judge_view,
    prompt_values,
    publish_scaffold,
    restore_transaction,
    stage_task_copy,
    tree_fingerprint,
    validate_alignment_report,
    validate_repair_scope,
    validate_staged_paths,
)


async def staged(tmp_path: Any) -> tuple[LocalWorkspace, Any, Any]:
    workspace = LocalWorkspace(tmp_path)
    task = await write_scaffold(workspace)
    paths = draft_paths("task")
    await stage_task_copy(workspace, "task", paths)
    return workspace, task, paths


def test_paths_and_prompt_values_follow_upstream_rendering() -> None:
    paths = draft_paths("run/task")
    assert paths.root == "run/_task_drafting" and paths.task == "run/_task_drafting/task"
    assert draft_paths("task").root == "_task_drafting"
    values = prompt_values(
        task_dir="./task", results_dir=".", task_md="## Goal\ng\n", extra={"X": "y"}
    )
    assert values["TASK_DIR"] == "task" and values["TASK_MD_PATH"] == "task/task.md"
    assert values["RESULTS_DIR"] == "." and values["TASK_GOAL"] == "g" and values["X"] == "y"
    dot = prompt_values(task_dir=".", task_md=None, extra=None)
    assert dot["TASK_MD_PATH"] == "task.md" and dot["TASK_GOAL"] == "(missing `task.md`)"
    extra = draft_values("program_0/src/program.py")
    assert extra["DRAFT_ENTRYPOINT_PATH"] == "./task/program_0/src/program.py"
    assert extra["TASK_DESIGN_PATH"] == "./_task_design/task_design.json"
    assert extra["WORKER_INDEX"] == "0" and extra["GLOBAL_MEMORY_MD_FILE"] == "./memory.md"


async def test_staging_honours_the_ignore_file_and_replaces_a_partial_copy(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    await workspace.write_bytes("task/.agentic_task_ignore", b"resources/\n/capture.py\n")
    paths = draft_paths("task")
    await workspace.write_bytes(f"{paths.task}/leftover.txt", b"partial copy")
    await stage_task_copy(workspace, "task", paths)
    assert not await workspace.exists(f"{paths.task}/leftover.txt")
    assert not await workspace.exists(f"{paths.task}/resources")
    assert not await workspace.exists(f"{paths.task}/capture.py")
    assert await workspace.exists(f"{paths.task}/evaluate.py")


async def test_the_candidate_entrypoint_is_declared_or_the_default(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    assert await candidate_entrypoint(workspace, "task") == "program_0/src/program.py"
    await workspace.remove("task/program_0/program_spec.toml")
    assert await candidate_entrypoint(workspace, "task") == "program_0/src/program.py"
    spec = (
        'schema_version = 1\ncode_root = "src"\n\n[entrypoint]\npath = "main.py"\n'
        'language = "python"\ntype = "script"\n\n[copy]\nexclude = []\ninclude = []\n\n'
        '[integration]\nkind = "generic"\nnotes = ""\n'
    )
    await workspace.write_bytes("task/program_0/program_spec.toml", spec.encode())
    assert await candidate_entrypoint(workspace, "task") == "program_0/src/main.py"


async def test_the_tree_fingerprint_changes_with_content_and_refuses_links(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    first = await tree_fingerprint(workspace, paths.task)
    assert first == await tree_fingerprint(workspace, paths.task)
    await workspace.write_bytes(f"{paths.task}/.venv/ignored", b"x")
    await workspace.write_bytes(f"{paths.task}/.agentic_runtime.json", b"{}")
    assert await tree_fingerprint(workspace, paths.task) == first, "runtime entries are omitted"
    await workspace.write_bytes(f"{paths.task}/evaluate.py", b"# changed\n")
    assert await tree_fingerprint(workspace, paths.task) != first
    os.symlink("evaluate.py", tmp_path / paths.task / "link.py")
    with pytest.raises(ScaffoldError, match=r"scaffold contains a symlink: link\.py"):
        await tree_fingerprint(workspace, paths.task)


async def test_alignment_fingerprints_cover_the_judged_files(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    fingerprints = await alignment_fingerprints(workspace, paths.task)
    assert set(fingerprints) == {
        "task.md",
        "capture.py",
        "task_contract.json",
        "evaluate.py",
        "evaluate_final.py",
        "program_0/program_spec.toml",
        "program_0/src/program.py",
        "declared_scaffold",
    }
    assert fingerprints["declared_scaffold"].startswith("sha256:")
    await workspace.remove(f"{paths.task}/capture.py")
    with pytest.raises(ScaffoldError, match=r"alignment: capture\.py"):
        await alignment_fingerprints(workspace, paths.task)


async def test_inspection_distinguishes_missing_from_unsafe(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    entry = "program_0/src/program.py"
    assert (await inspect_scaffold(workspace, paths, entry)).kind == "ok"
    await workspace.remove(f"{paths.task}/capture.py")
    await workspace.remove(f"{paths.task}/program_0")
    result = await inspect_scaffold(workspace, paths, entry)
    assert result.kind == "missing"
    assert result.diagnostic == (
        "Missing required drafted artifact: capture.py\n"
        "Missing required drafted directory: program_0"
    )
    await workspace.write_bytes(f"{paths.task}/capture.py", b"x")
    await workspace.write_bytes(f"{paths.task}/program_0/src/program.py", b"x")
    os.symlink("evaluate.py", tmp_path / paths.task / "capture_link.py")
    os.remove(tmp_path / paths.task / "capture.py")
    os.symlink("evaluate.py", tmp_path / paths.task / "capture.py")
    unsafe = await inspect_scaffold(workspace, paths, entry)
    assert unsafe.kind == "unsafe" and unsafe.diagnostic == "Unsafe drafted artifact: capture.py"


async def test_staged_paths_are_validated_with_upstreams_messages(tmp_path: Any) -> None:
    workspace, task, paths = await staged(tmp_path)
    sha = task.task_md_sha256
    await validate_staged_paths(workspace, paths.task, canonical_task_sha256=sha)
    await workspace.write_bytes(f"{paths.task}/task.md", b"changed")
    with pytest.raises(ScaffoldError, match=r"modified canonical task\.md"):
        await validate_staged_paths(workspace, paths.task, canonical_task_sha256=sha)
    await workspace.write_bytes(f"{paths.task}/task.md", (tmp_path / "task/task.md").read_bytes())
    await workspace.write_bytes(f"{paths.task}/.agentic_repair_prompt.md", b"x")
    with pytest.raises(ScaffoldError, match="reserved harness path"):
        await validate_staged_paths(workspace, paths.task, canonical_task_sha256=sha)


async def test_canonicalisation_restores_the_harness_owned_files(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    canonical_program = (tmp_path / "task/program_0/src/program.py").read_bytes()
    canonical_capture = (tmp_path / "task/capture.py").read_bytes()
    await workspace.write_bytes(f"{paths.task}/capture.py", b"tampered")
    await workspace.write_bytes(f"{paths.task}/program_0/program_spec.toml", b"tampered")
    await workspace.write_bytes(f"{paths.task}/program_0/src/program.py", b"tampered")
    await workspace.write_bytes(f"{paths.task}/program_0/src/__pycache__/x.pyc", b"\x00")
    await canonicalize_staged_scaffold(workspace, paths, "task")
    assert (tmp_path / paths.task / "capture.py").read_bytes() == canonical_capture
    assert (tmp_path / paths.task / "program_0/src/program.py").read_bytes() == canonical_program
    assert b"schema_version" in (tmp_path / paths.task / "program_0/program_spec.toml").read_bytes()
    assert not await workspace.exists(f"{paths.task}/program_0/src/__pycache__")


async def test_a_missing_canonical_entrypoint_is_code_three(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    await workspace.remove("task/program_0/src/program.py")
    await workspace.remove("task/capture.py")
    paths = draft_paths("task")
    await stage_task_copy(workspace, "task", paths)
    await workspace.write_bytes(f"{paths.task}/capture.py", b"x")
    with pytest.raises(ScaffoldError) as raised:
        await canonicalize_staged_scaffold(workspace, paths, "task")
    assert raised.value.exit_code == 3
    assert "Canonical candidate entrypoint does not exist: program_0/src/program.py" in str(
        raised.value
    )


async def test_a_scoped_repair_may_change_only_mutable_files(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    transaction = await open_scoped_transaction(
        workspace, paths, label="smoke_repair_0", policy="synchronized-digests"
    )
    assert transaction.mutable_paths == sorted(
        [
            "evaluate.py",
            "evaluate_final.py",
            "program_0/src/program.py",
            "requirements.txt",
            "task_contract.json",
        ]
    )
    await workspace.write_bytes(f"{paths.task}/evaluate.py", b"# fixed\n")
    changed, _ = await validate_repair_scope(workspace, paths, transaction)
    assert changed == ("evaluate.py",)
    await workspace.write_bytes(f"{paths.task}/capture.py", b"# not allowed\n")
    with pytest.raises(RepairScopeError, match=r"out-of-scope path\(s\): capture\.py"):
        await validate_repair_scope(workspace, paths, transaction)
    await restore_transaction(workspace, transaction)
    assert (tmp_path / paths.task / "capture.py").read_bytes() == (
        tmp_path / "task/capture.py"
    ).read_bytes()
    assert (tmp_path / paths.task / "evaluate.py").read_bytes() == (
        tmp_path / "task/evaluate.py"
    ).read_bytes()


async def test_a_scoped_repair_may_not_change_what_the_contract_means(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    contract = json.loads((tmp_path / paths.task / "task_contract.json").read_text())
    changed = copy.deepcopy(contract)
    changed["protocol"]["promotion_rule"] = "anything goes"
    immutable = await open_scoped_transaction(workspace, paths, label="a", policy="immutable")
    assert "task_contract.json" not in immutable.mutable_paths
    digests = await open_scoped_transaction(
        workspace, paths, label="b", policy="synchronized-digests"
    )
    await workspace.write_bytes(f"{paths.task}/task_contract.json", json.dumps(changed).encode())
    with pytest.raises(RepairScopeError, match="out-of-scope path"):
        await validate_repair_scope(workspace, paths, immutable)
    with pytest.raises(RepairScopeError, match="repair changed task-contract semantics"):
        await validate_repair_scope(workspace, paths, digests)
    await workspace.write_bytes(f"{paths.task}/task_contract.json", json.dumps(contract).encode())
    await workspace.write_bytes(f"{paths.task}/requirements.txt", b"python==3.12\nnumpy==1.0.0\n")
    with pytest.raises(RepairScopeError, match="require a runtime_dependencies repair diagnostic"):
        await validate_repair_scope(workspace, paths, digests)


async def test_dependency_removal_is_synchronised_and_journaled(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    task = await write_scaffold(workspace)
    contract = json.loads((tmp_path / "task/task_contract.json").read_text())
    contract["provenance"]["dependencies"].append({"name": "numpy", "version": "1.26.0"})
    await workspace.write_bytes("task/task_contract.json", json.dumps(contract).encode())
    await workspace.write_bytes("task/requirements.txt", b"python==3.12\nnumpy==1.26.0\n")
    paths = draft_paths("task")
    await stage_task_copy(workspace, "task", paths)
    assert task is not None
    diagnostic = json.dumps({"category": "runtime_dependencies"})
    policy = contract_policy_for_smoke_diagnostic(diagnostic)
    assert policy == "synchronized-dependency-removals"
    assert contract_policy_for_smoke_diagnostic("not json") == "synchronized-digests"
    transaction = await open_scoped_transaction(workspace, paths, label="r0", policy=policy)
    await workspace.write_bytes(f"{paths.task}/requirements.txt", b"python==3.12\n")
    _, updated = await validate_repair_scope(workspace, paths, transaction)
    assert updated.retained_dependencies == [{"name": "python", "version": "3.12"}]
    synchronized = json.loads((tmp_path / paths.task / "task_contract.json").read_text())
    assert synchronized["provenance"]["dependencies"] == [{"name": "python", "version": "3.12"}]
    await workspace.write_bytes(f"{paths.task}/requirements.txt", b"python==3.12\nscipy==1.0\n")
    with pytest.raises(RepairScopeError, match="only remove pinned dependencies"):
        await validate_repair_scope(workspace, paths, updated)


async def test_a_tree_transaction_restores_everything_and_keeps_the_sources(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    transaction = await open_tree_transaction(workspace, paths, label="alignment_0")
    assert transaction.base_fingerprint == await tree_fingerprint(workspace, paths.task)
    assert (
        "evaluate.py" in transaction.sources_before
        and "task_contract.json" in transaction.sources_before
    )
    assert not any(name.startswith("resources/") for name in transaction.sources_before)
    await workspace.write_bytes(f"{paths.task}/new.py", b"x")
    await workspace.write_bytes(f"{paths.task}/evaluate.py", b"changed")
    await restore_transaction(workspace, transaction)
    assert not await workspace.exists(f"{paths.task}/new.py")
    assert await tree_fingerprint(workspace, paths.task) == transaction.base_fingerprint


async def test_the_judge_view_omits_runtime_state_and_adds_the_design(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    await workspace.write_bytes(f"{paths.task}/.venv/bin/python", b"x")
    await workspace.write_bytes(f"{paths.task}/.agentic_runtime.json", b"{}")
    await workspace.write_bytes(f"{paths.task}/sub/.venv/keep", b"x")
    await workspace.write_bytes(paths.design_file, b'{"design": 1}')
    view = await prepare_judge_view(workspace, paths, 3, has_design=True)
    assert view == paths.judge_view(3)
    assert not await workspace.exists(f"{view}/.venv")
    assert not await workspace.exists(f"{view}/.agentic_runtime.json")
    assert await workspace.exists(f"{view}/sub/.venv/keep")
    assert (tmp_path / view / "_accepted_task_design.json").read_bytes() == b'{"design": 1}'
    again = await prepare_judge_view(workspace, paths, 3, has_design=False)
    assert not await workspace.exists(f"{again}/_accepted_task_design.json")


async def test_alignment_report_validation_has_upstreams_rules(tmp_path: Any) -> None:
    workspace, _, paths = await staged(tmp_path)
    fingerprints = await alignment_fingerprints(workspace, paths.task)
    view = paths.task
    good = alignment_report(fingerprints, failing=["compute_realism"])
    assert await validate_alignment_report(
        workspace, good, expected_fingerprints=fingerprints, evidence_root=view
    ) == ("compute_realism",)

    async def refused(report: Any, message: str) -> None:
        with pytest.raises(AlignmentReportError, match=message):
            await validate_alignment_report(
                workspace, report, expected_fingerprints=fingerprints, evidence_root=view
            )

    await refused(alignment_report({**fingerprints, "task.md": "0"}), "fingerprints do not match")
    await refused(good.model_copy(update={"verdict": "pass"}), "top-level verdict does not match")
    await refused(good.model_copy(update={"report_schema_version": 2}), "schema_version must be 1")
    await refused(good.model_copy(update={"summary": "  "}), "requires a summary")
    await refused(
        good.model_copy(update={"dimensions": good.dimensions[:-1]}), "exactly the required"
    )
    duplicate = good.model_copy(update={"dimensions": [good.dimensions[0], *good.dimensions[:-1]]})
    await refused(duplicate, "duplicate alignment dimension")
    for mutate, message in (
        (lambda d: d.model_copy(update={"remediation": ""}), "requires remediation"),
        (lambda d: d.model_copy(update={"evidence": []}), "requires non-empty evidence"),
        (
            lambda d: d.model_copy(
                update={"evidence": [AlignmentEvidence(path="../x", line=1, detail="d")]}
            ),
            "safe relative file",
        ),
        (
            lambda d: d.model_copy(
                update={"evidence": [AlignmentEvidence(path="nope.py", line=1, detail="d")]}
            ),
            "safe relative file",
        ),
        (
            lambda d: d.model_copy(
                update={"evidence": [AlignmentEvidence(path="task.md", line=999, detail="d")]}
            ),
            "exceeds file length",
        ),
        (
            lambda d: d.model_copy(
                update={"evidence": [AlignmentEvidence(path="task.md", line=1, detail=" ")]}
            ),
            "detail is required",
        ),
    ):
        index = next(i for i, row in enumerate(good.dimensions) if row.verdict == "fail")
        rows = list(good.dimensions)
        rows[index] = mutate(rows[index])
        await refused(good.model_copy(update={"dimensions": rows}), message)
    passed = alignment_report(fingerprints)
    rows = list(passed.dimensions)
    rows[0] = rows[0].model_copy(update={"remediation": "unneeded"})
    await refused(passed.model_copy(update={"dimensions": rows}), "must not include remediation")


def test_the_repair_report_keeps_failed_dimensions_within_bounds() -> None:
    fingerprints = {"declared_scaffold": "sha256:" + "a" * 64}
    report = alignment_report(fingerprints, failing=["compute_realism"])
    row = next(d for d in report.dimensions if d.verdict == "fail")
    row.evidence = [AlignmentEvidence(path="p" * 600, line=1, detail="d" * 900) for _ in range(10)]
    row.remediation = "r" * 3000
    text = bounded_repair_report(report)
    payload = json.loads(text)
    assert [d["id"] for d in payload["dimensions"]] == ["compute_realism"]
    kept = payload["dimensions"][0]
    assert len(kept["evidence"]) == 8 and kept["evidence_omitted"] == 2
    assert len(kept["evidence"][0]["path"].encode()) <= 512 + 3
    assert len(kept["evidence"][0]["detail"].encode()) <= 768 + 3
    assert len(kept["remediation"].encode()) <= 2048 + 3
    huge = report.model_copy(update={"summary": "s" * 10})
    for dimension in huge.dimensions:
        dimension.verdict = "fail"
        dimension.evidence = [AlignmentEvidence(path="p" * 512, line=1, detail="d" * 768)] * 8
        dimension.remediation = "r" * 2048
    with pytest.raises(ValueError, match="exceeds 96 KiB"):
        bounded_repair_report(huge)


def test_the_alignment_history_is_bounded_and_untrusted() -> None:
    fingerprints = {"declared_scaffold": "sha256:" + "d" * 64}
    report = alignment_report(fingerprints, failing=["compute_realism", "reproducibility"])
    report.dimensions[7].evidence = [
        AlignmentEvidence(path="a.py", line=n, detail="x" * 600) for n in range(1, 6)
    ]
    entry = capture_alignment_report(
        report, task_fingerprint="sha256:" + "e" * 64, name="alignment_0.json"
    )
    assert entry.historical_verdict == "fail" and len(entry.findings) == 2
    finding = next(f for f in entry.findings if f.id == "compute_realism")
    assert len(finding.evidence) == 3 and finding.truncated
    assert len(finding.evidence[0]["detail"].encode()) <= 512
    heading = "\n\n# Prior independent alignment findings (untrusted historical data)\nX\n"
    entries = [entry.model_copy(update={"report_name": f"alignment_{n}.json"}) for n in range(20)]
    text = render_alignment_history(
        entries, current_fingerprint="sha256:" + "f" * 64, heading=heading
    )
    assert text.startswith(heading) and len(text.encode()) <= 64 * 1024
    payload = json.loads(text[len(heading) :])
    assert payload["omitted_older_reports"] == 4 and len(payload["alignment_history"]) == 16
    assert payload["alignment_history"][0]["report"] == "alignment_4.json"
    assert render_alignment_history([], current_fingerprint="x", heading=heading) == ""
    with pytest.raises(ValueError, match="invalid reviewed task fingerprint"):
        capture_alignment_report(report, task_fingerprint="bad", name="alignment_0.json")
    broken = report.model_copy(update={"verdict": "pass"})
    with pytest.raises(ValueError, match="verdict disagrees with dimensions"):
        capture_alignment_report(broken, task_fingerprint="sha256:" + "e" * 64, name="a.json")


async def test_publication_is_idempotent_and_the_marker_is_last(tmp_path: Any) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    paths = draft_paths("run/task")
    await workspace.write_bytes("run/task/task.md", (tmp_path / "task/task.md").read_bytes())
    await stage_task_copy(workspace, "task", paths_for_root(paths))
    staged_dir = paths_for_root(paths).task
    plan = await build_publication_plan(
        workspace, staged_dir, "run/task", max_runtime_bytes=1 << 20
    )
    assert set(plan.files) == {
        "capture.py",
        "task.md",
        "evaluate.py",
        "evaluate_final.py",
        "task_contract.json",
        "program_0/program_spec.toml",
        "program_0/src/program.py",
        "requirements.txt",
    }
    assert plan.resources == "resources" and plan.sha256 == plan.sha256
    report = b'{"verdict": "pass"}\n'
    receipt = b'{"receipt": true}\n'
    await publish_scaffold(
        workspace,
        plan,
        staged_dir=staged_dir,
        published_dir="run/task",
        alignment_report=report,
        qualification_receipt=receipt,
    )
    await publish_scaffold(
        workspace,
        plan,
        staged_dir=staged_dir,
        published_dir="run/task",
        alignment_report=report,
        qualification_receipt=receipt,
    )
    assert (tmp_path / "run/task/.agentic_task_validation.json").read_bytes() == report
    ignore = (tmp_path / "run/task/.agentic_task_ignore").read_text().splitlines()
    assert "/evaluate_final.py" in ignore and "/resources/final/" in ignore
    manifest = json.loads((tmp_path / "run/task/.agentic_task_publication.json").read_text())
    assert manifest["files"]["capture.py"] and manifest["resources"]["bytes"] == 8
    with pytest.raises(PublicationError, match="different validation marker"):
        await publish_scaffold(
            workspace,
            plan,
            staged_dir=staged_dir,
            published_dir="run/task",
            alignment_report=b"other",
            qualification_receipt=receipt,
        )
    await workspace.write_bytes("run/task/evaluate.py", b"conflict")
    with pytest.raises(PublicationError, match=r"conflicting file: run/task/evaluate\.py"):
        await publish_scaffold(
            workspace,
            plan,
            staged_dir=staged_dir,
            published_dir="run/task",
            alignment_report=report,
            qualification_receipt=receipt,
        )
    with pytest.raises(PublicationError, match="stale qualification receipt"):
        await publish_scaffold(
            workspace,
            plan,
            staged_dir=staged_dir,
            published_dir="run/task",
            alignment_report=report,
            qualification_receipt=None,
        )


def paths_for_root(paths: Any) -> Any:
    return paths


async def test_publication_refuses_helpers_reserved_paths_and_oversized_resources(
    tmp_path: Any,
) -> None:
    workspace = LocalWorkspace(tmp_path)
    await write_scaffold(workspace)
    with pytest.raises(
        PublicationError, match="runtime_resources_exceed_limit: observed=8 limit=4"
    ):
        await build_publication_plan(workspace, "task", "task", max_runtime_bytes=4)
    contract = json.loads((tmp_path / "task/task_contract.json").read_text())
    contract["provenance"]["artifacts"][0]["role"] = "search_helper"
    await workspace.write_bytes("task/task_contract.json", json.dumps(contract).encode())
    with pytest.raises(
        PublicationError, match="generated task-local evaluator helpers are unsupported"
    ):
        await build_publication_plan(workspace, "task", "task", max_runtime_bytes=1 << 20)
