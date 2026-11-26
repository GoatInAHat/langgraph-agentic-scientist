"""LocalWorkspace: safe paths, copies with ignore rules, digests, snapshots, bounded commands."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

from langgraph_agentic_scientist.core.errors import WorkspacePathError
from langgraph_agentic_scientist.workspace import (
    LocalWorkspace,
    parse_ignore_patterns,
    safe_relpath,
    task_copy_path_is_ignored,
)


def test_safe_relpath_normalises_and_refuses_traversal() -> None:
    assert safe_relpath("./a//b/") == "a/b"
    assert safe_relpath("/a/b") == "a/b"
    assert safe_relpath(".") == ""
    for bad in ("../x", "a/../../x", "~/x", "a\\b", "a\x00b"):
        with pytest.raises(WorkspacePathError):
            safe_relpath(bad)
    with pytest.raises(WorkspacePathError):
        safe_relpath("", allow_root=False)


async def test_traversal_and_symlink_escape_are_refused(
    workspace: LocalWorkspace, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    os.symlink(outside, workspace.root / "link")
    with pytest.raises(WorkspacePathError):
        await workspace.read_bytes("../outside/secret.txt")
    with pytest.raises(WorkspacePathError):
        await workspace.read_bytes("link/secret.txt")
    with pytest.raises(WorkspacePathError):
        await workspace.write_bytes("link/new.txt", b"x")
    entry = await workspace.stat("link")
    assert entry is not None and entry.kind == "symlink"


async def test_write_read_exclusive_and_stat(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("a/b.txt", b"one")
    assert await workspace.read_bytes("a/b.txt") == b"one"
    await workspace.write_bytes("a/b.txt", b"two")
    assert await workspace.read_bytes("a/b.txt") == b"two"
    with pytest.raises(FileExistsError):
        await workspace.write_bytes("a/b.txt", b"three", exclusive=True)
    entry = await workspace.stat("a")
    assert entry is not None and entry.kind == "dir"
    assert await workspace.stat("missing") is None
    with pytest.raises(FileNotFoundError):
        await workspace.read_bytes("missing")


def test_ignore_patterns_follow_upstream_rules() -> None:
    patterns = parse_ignore_patterns("# comment\n\n*.log\n/data/raw/\n/**/cache\nbuild/\n")
    assert patterns == ("*.log", "/data/raw/", "/**/cache", "build/")
    assert task_copy_path_is_ignored("x/run.log", is_dir=False, patterns=patterns)
    assert task_copy_path_is_ignored("data/raw", is_dir=True, patterns=patterns)
    assert not task_copy_path_is_ignored("data/raw", is_dir=False, patterns=patterns)
    assert not task_copy_path_is_ignored("other/data/raw", is_dir=True, patterns=patterns)
    assert task_copy_path_is_ignored("a/b/cache", is_dir=True, patterns=patterns)
    assert task_copy_path_is_ignored("build", is_dir=True, patterns=patterns)
    assert not task_copy_path_is_ignored("src/main.py", is_dir=False, patterns=patterns)


async def test_copy_tree_with_ignore_rules(workspace: LocalWorkspace) -> None:
    for path in ("task/task.md", "task/run.log", "task/data/raw/big.bin", "task/data/keep.csv"):
        await workspace.write_bytes(path, b"x")
    await workspace.copy_tree("task", "snapshot", ignore=("*.log", "/data/raw/"))
    listed = {entry.path for entry in await workspace.list_tree("snapshot")}
    assert listed == {"task.md", "data", "data/keep.csv"}
    with pytest.raises(FileExistsError):
        await workspace.copy_tree("task", "snapshot")


async def test_tree_digest_is_stable_and_upstream_compatible(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("p/src/program.py", b"print(1)\n")
    await workspace.write_bytes("p/program_spec.toml", b"x")
    first = await workspace.digest("p")
    assert first == await workspace.digest("p")
    hasher = hashlib.sha256()
    for relative, content in (
        ("program_spec.toml", b"x"),
        ("src", None),
        ("src/program.py", b"print(1)\n"),
    ):
        if content is None:
            hasher.update(b"dir\0" + relative.encode() + b"\0")
        else:
            hasher.update(
                b"file\0" + relative.encode() + b"\0" + str(len(content)).encode() + b"\0" + content
            )
    assert first == "sha256:" + hasher.hexdigest()
    await workspace.write_bytes("p/src/program.py", b"print(2)\n")
    assert await workspace.digest("p") != first
    assert (
        await workspace.digest("p/program_spec.toml")
        == "sha256:" + hashlib.sha256(b"x").hexdigest()
    )


async def test_snapshot_and_restore(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("paper/paper.tex", b"original")
    ref = await workspace.snapshot("paper")
    await workspace.write_bytes("paper/paper.tex", b"changed")
    await workspace.write_bytes("paper/extra.txt", b"new")
    await workspace.restore(ref)
    assert await workspace.read_bytes("paper/paper.tex") == b"original"
    assert not await workspace.exists("paper/extra.txt")
    missing = await workspace.snapshot("not_yet")
    await workspace.write_bytes("not_yet/file", b"x")
    await workspace.restore(missing)
    assert not await workspace.exists("not_yet")


async def test_run_output_is_bounded_and_timeout_kills(workspace: LocalWorkspace) -> None:
    result = await workspace.run(
        [sys.executable, "-c", "import sys; sys.stdout.write('x' * 200000)"],
        cwd="",
        max_output_bytes=1000,
    )
    assert result.returncode == 0 and result.truncated
    assert len(result.stdout.encode()) < 1200
    slow = await workspace.run(
        [sys.executable, "-c", "import time; time.sleep(30)"], cwd="", timeout=0.5
    )
    assert slow.timed_out and slow.returncode == 124


async def test_remove_and_list_tree(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("d/a.txt", b"a")
    entries = await workspace.list_tree("d")
    assert entries[0].sha256 == "sha256:" + hashlib.sha256(b"a").hexdigest()
    await workspace.remove("d")
    assert not await workspace.exists("d")
    await workspace.remove("d")  # missing is not an error
    with pytest.raises(WorkspacePathError):
        await workspace.remove("")
