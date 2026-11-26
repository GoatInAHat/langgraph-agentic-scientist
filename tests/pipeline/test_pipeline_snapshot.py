"""The run-local task snapshot: a verified copy, made once, honouring the task's ignore file."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from testkit import thread

from langgraph_agentic_scientist import LocalWorkspace, ScientistConfig
from langgraph_agentic_scientist.pipeline import (
    TASK_SNAPSHOT_METADATA,
    TaskSnapshotError,
    load_task_ref,
    missing_scaffold_files,
    prepare_task_snapshot,
    reseal_task_snapshot,
)


def _task(workspace: LocalWorkspace, name: str = "src_task") -> Path:
    root = Path(workspace.root) / name
    (root / "program_0" / "src").mkdir(parents=True)
    (root / "task.md").write_text("# T\n\n## Goal\nGo.\n\n## Rubrics\n- r\n")
    (root / "evaluate.py").write_text("print(1)\n")
    (root / "capture.py").write_text("print(2)\n")
    (root / "program_0" / "src" / "program.py").write_text("X = 1\n")
    (root / "program_0" / "program_spec.toml").write_text(
        'schema_version = 1\ncode_root = "src"\n[entrypoint]\npath = "program.py"\n'
        'language = "python"\ntype = "script"\n[copy]\nexclude = []\ninclude = []\n'
        '[integration]\nkind = "generic"\nnotes = ""\n'
    )
    return root


async def test_the_copy_honours_the_ignore_file_and_records_a_digest(
    workspace: LocalWorkspace,
) -> None:
    source = _task(workspace)
    (source / ".agentic_task_ignore").write_text("# comment\n/secret/\n*.log\n")
    (source / "secret").mkdir()
    (source / "secret" / "key.txt").write_text("k")
    (source / "run.log").write_text("noise")
    await prepare_task_snapshot(workspace, "src_task")
    root = Path(workspace.root) / "task"
    assert (root / "evaluate.py").is_file() and not (root / "secret").exists()
    assert not (root / "run.log").exists()
    metadata = json.loads((root / TASK_SNAPSHOT_METADATA).read_text())
    assert metadata["version"] == 2 and metadata["task_name"] == "src_task"
    assert len(metadata["source_digest"]) == 64


async def test_an_existing_snapshot_is_verified_and_never_refreshed(
    workspace: LocalWorkspace,
) -> None:
    source = _task(workspace)
    await prepare_task_snapshot(workspace, "src_task")
    (source / "evaluate.py").write_text("print('changed source')\n")
    await prepare_task_snapshot(workspace, "src_task")  # reuse: the source is not read again
    assert (Path(workspace.root) / "task" / "evaluate.py").read_text() == "print(1)\n"
    (Path(workspace.root) / "task" / "evaluate.py").write_text("tampered\n")
    with pytest.raises(TaskSnapshotError, match="does not match its source digest"):
        await prepare_task_snapshot(workspace, "src_task")


async def test_an_interrupted_copy_without_metadata_is_made_again(
    workspace: LocalWorkspace,
) -> None:
    _task(workspace)
    partial = Path(workspace.root) / "task"
    partial.mkdir()
    (partial / "half.txt").write_text("partial")
    await prepare_task_snapshot(workspace, "src_task")
    assert not (partial / "half.txt").exists() and (partial / "evaluate.py").is_file()


async def test_resealing_records_what_drafting_published(workspace: LocalWorkspace) -> None:
    _task(workspace)
    await prepare_task_snapshot(workspace, "src_task")
    (Path(workspace.root) / "task" / "evaluate_final.py").write_text("print(3)\n")
    with pytest.raises(TaskSnapshotError):
        await prepare_task_snapshot(workspace, "src_task")
    await reseal_task_snapshot(workspace)
    await prepare_task_snapshot(workspace, "src_task")


@pytest.mark.parametrize(
    ("link", "target", "message"),
    [
        ("abs", "/etc/passwd", "absolute symlink"),
        ("up", "../../outside", "escaping symlink"),
    ],
)
async def test_unsafe_symlinks_in_the_source_are_refused(
    workspace: LocalWorkspace, link: str, target: str, message: str
) -> None:
    source = _task(workspace)
    os.symlink(target, source / link)
    with pytest.raises(TaskSnapshotError, match=message):
        await prepare_task_snapshot(workspace, "src_task")
    assert not (Path(workspace.root) / "task").exists()


async def test_a_safe_relative_symlink_is_carried(workspace: LocalWorkspace) -> None:
    source = _task(workspace)
    os.symlink("evaluate.py", source / "alias.py")
    await prepare_task_snapshot(workspace, "src_task")
    assert os.readlink(Path(workspace.root) / "task" / "alias.py") == "evaluate.py"


async def test_an_ignored_unsafe_symlink_does_not_matter(workspace: LocalWorkspace) -> None:
    source = _task(workspace)
    (source / ".agentic_task_ignore").write_text("/venv/\n")
    (source / "venv").mkdir()
    os.symlink("/usr/bin/python3", source / "venv" / "python")
    await prepare_task_snapshot(workspace, "src_task")


@pytest.mark.parametrize("source", ["task", "task/inner", ""])
async def test_a_source_overlapping_the_snapshot_is_refused(
    workspace: LocalWorkspace, source: str
) -> None:
    with pytest.raises((TaskSnapshotError, ValueError)):
        await prepare_task_snapshot(workspace, source)


async def test_a_missing_source_is_refused(workspace: LocalWorkspace) -> None:
    with pytest.raises(TaskSnapshotError, match="not a real directory"):
        await prepare_task_snapshot(workspace, "nowhere")


async def test_scaffold_detection_names_what_is_missing(workspace: LocalWorkspace) -> None:
    source = _task(workspace)
    assert await missing_scaffold_files(workspace, "src_task") == []
    (source / "capture.py").unlink()
    (source / "program_0" / "src" / "program.py").unlink()
    assert await missing_scaffold_files(workspace, "src_task") == [
        "capture.py",
        "program_0/<entrypoint>",
    ] or await missing_scaffold_files(workspace, "src_task") == ["capture.py"]


async def test_a_contract_adds_files_to_the_scaffold(
    workspace: LocalWorkspace, contracted: str
) -> None:
    assert await missing_scaffold_files(workspace, contracted) == []
    (Path(workspace.root) / contracted / "evaluate_final.py").unlink()
    assert await missing_scaffold_files(workspace, contracted) == ["evaluate_final.py"]


async def test_the_task_reference_carries_the_sections_the_contract_and_the_seed(
    workspace: LocalWorkspace, contracted: str
) -> None:
    await prepare_task_snapshot(workspace, contracted)
    task = await load_task_ref(workspace, ScientistConfig())
    assert task.root == "task" and task.goal and task.task_md_sha256
    assert task.contract is not None and task.harness_evaluated
    assert task.seed_program is not None and task.seed_program.digest.startswith("sha256:")  # type: ignore[union-attr]


async def test_the_run_id_must_be_a_single_path_component(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str
) -> None:
    from langgraph_agentic_scientist import PipelineConfigurationError

    with pytest.raises(PipelineConfigurationError, match="run_id"):
        await make_app().ainvoke(
            {"task_dir": circpack, "run_id": "../escape"}, thread("rid"), context=make_context()
        )
