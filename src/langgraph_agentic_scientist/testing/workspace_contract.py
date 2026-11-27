"""Contract tests of the :class:`~langgraph_agentic_scientist.workspace.Workspace` protocol.

A workspace on another substrate (a container, a remote sandbox, an object store) shows that it
keeps the protocol by running these tests over it. They assert only what the protocol states, and
the commands they run are ``sh -c`` scripts that need no host path, so they run unchanged inside a
container. The module imports pytest and is meant to be imported by test code only; the tests are
coroutines and need pytest-asyncio.

Define a fixture ``contract_workspace`` that returns a fresh, empty workspace, and subclass
:class:`WorkspaceContract`::

    from langgraph_agentic_scientist.testing.workspace_contract import WorkspaceContract

    class TestMyWorkspace(WorkspaceContract):
        @pytest.fixture
        def contract_workspace(self, tmp_path):
            return MyWorkspace(tmp_path)

or define the fixture in a ``conftest.py`` and import the ready test class into a test module with
``from langgraph_agentic_scientist.testing.workspace_contract import *``. A workspace whose ``run``
and agent ``execute`` run no real commands sets ``runs_commands = False`` on its subclass, which
skips the command tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import time
from typing import Any, ClassVar

import pytest

from ..core.errors import WorkspacePathError
from ..workspace.digest import tree_digest
from ..workspace.paths import join
from ..workspace.protocol import SnapshotRef, Workspace, agent_path
from ..workspace.readonly import ReadOnlyBackend

ESCAPING_PATHS = ("../outside", "inside/../../outside", "~/outside")
LONG_OUTPUT = 'i=0; while [ "$i" -lt 2000 ]; do printf "%050d" 0; i=$((i+1)); done'
OUTPUT_ALLOWANCE = 256


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


async def _write_tree(workspace: Workspace, base: str, files: dict[str, bytes]) -> None:
    for relative, data in files.items():
        await workspace.write_bytes(join(base, relative), data)


async def _files(workspace: Workspace, base: str) -> dict[str, bytes]:
    return {
        entry.path: await workspace.read_bytes(join(base, entry.path))
        for entry in await workspace.list_tree(base)
        if entry.kind == "file"
    }


def _in_agent_root(base: str, name: str) -> str:
    return f"{base.rstrip('/')}/{name}"


class WorkspaceContract:
    """The protocol's promises as tests over the fixture ``contract_workspace``."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.asyncio]
    runs_commands: ClassVar[bool] = True

    def _require_commands(self) -> None:
        if not self.runs_commands:
            pytest.skip("the workspace runs no commands")

    async def test_write_then_read(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        await workspace.write_bytes("files/nested/one.bin", b"\x00one")
        assert await workspace.read_bytes("files/nested/one.bin") == b"\x00one"
        await workspace.write_bytes("files/nested/one.bin", b"two")
        assert await workspace.read_bytes("files/nested/one.bin") == b"two"
        await workspace.write_bytes("./files/dotted.txt", b"dotted")
        assert await workspace.read_bytes("files/dotted.txt") == b"dotted"
        with pytest.raises(FileNotFoundError):
            await workspace.read_bytes("files/missing.txt")

    async def test_exclusive_write(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        await workspace.write_bytes("claim/once.txt", b"first", exclusive=True)
        with pytest.raises(FileExistsError):
            await workspace.write_bytes("claim/once.txt", b"second", exclusive=True)
        assert await workspace.read_bytes("claim/once.txt") == b"first"

    async def test_stat_and_exists(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        await workspace.write_bytes("tree/leaf.txt", b"abc")
        leaf = await workspace.stat("tree/leaf.txt")
        assert leaf is not None
        assert (leaf.path, leaf.kind, leaf.size) == ("tree/leaf.txt", "file", 3)
        directory = await workspace.stat("tree")
        assert directory is not None
        assert directory.kind == "dir"
        assert await workspace.stat("tree/missing") is None
        assert await workspace.exists("tree/leaf.txt")
        assert await workspace.exists("tree")
        assert await workspace.exists("")
        assert not await workspace.exists("tree/missing")

    async def test_list_tree(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        await _write_tree(workspace, "listed", {"b.txt": b"bee", "a/deep/c.bin": b"\x01\x02"})
        entries = await workspace.list_tree("listed")
        assert [(entry.path, entry.kind) for entry in entries] == [
            ("a", "dir"),
            ("a/deep", "dir"),
            ("a/deep/c.bin", "file"),
            ("b.txt", "file"),
        ]
        assert all(entry.sha256 is None for entry in entries)
        sizes = {entry.path: entry.size for entry in entries if entry.kind == "file"}
        assert sizes == {"a/deep/c.bin": 2, "b.txt": 3}
        digested = {
            entry.path: entry.sha256
            for entry in await workspace.list_tree("listed", with_digest=True)
            if entry.kind == "file"
        }
        assert digested == {"a/deep/c.bin": _sha256(b"\x01\x02"), "b.txt": _sha256(b"bee")}

    async def test_remove(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        await _write_tree(workspace, "gone", {"file.txt": b"x", "dir/a.txt": b"a", "dir/b/c": b"c"})
        await workspace.remove("gone/file.txt")
        assert not await workspace.exists("gone/file.txt")
        await workspace.remove("gone/dir")
        assert not await workspace.exists("gone/dir")
        assert not await workspace.exists("gone/dir/b/c")
        await workspace.remove("gone/never-there")

    async def test_digest(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        files = {"x.txt": b"x", "sub/y.txt": b"y"}
        await _write_tree(workspace, "digested", files)
        assert await workspace.digest("digested/x.txt") == _sha256(b"x")

        async def read(relative: str) -> bytes:
            return await workspace.read_bytes(join("digested", relative))

        expected = await tree_digest(await workspace.list_tree("digested"), read)
        assert await workspace.digest("digested") == expected
        await _write_tree(workspace, "twin", files)
        assert await workspace.digest("twin") == expected
        await workspace.write_bytes("twin/sub/y.txt", b"changed")
        assert await workspace.digest("twin") != expected

    async def test_copy_tree(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        files = {
            "keep.txt": b"keep",
            "nested/keep.py": b"print()",
            "run.log": b"log",
            "nested/deep.log": b"log",
            "cache/blob": b"cache",
        }
        await _write_tree(workspace, "source", files)
        await workspace.copy_tree("source", "copy")
        assert await _files(workspace, "copy") == files
        await workspace.copy_tree("source", "filtered", ignore=["*.log", "cache/"])
        assert await _files(workspace, "filtered") == {
            "keep.txt": b"keep",
            "nested/keep.py": b"print()",
        }
        assert await _files(workspace, "source") == files
        with pytest.raises(FileExistsError):
            await workspace.copy_tree("source", "copy")

    async def test_snapshot_and_restore_a_file(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        await workspace.write_bytes("snap/file.txt", b"before")
        ref = await workspace.snapshot("snap/file.txt")
        assert ref.path == "snap/file.txt"
        await workspace.write_bytes("snap/file.txt", b"after")
        await workspace.restore(ref)
        assert await workspace.read_bytes("snap/file.txt") == b"before"

    async def test_snapshot_and_restore_a_tree(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        files = {"a.txt": b"a", "sub/b.txt": b"b"}
        await _write_tree(workspace, "snaptree", files)
        ref = await workspace.snapshot("snaptree")
        await workspace.write_bytes("snaptree/a.txt", b"changed")
        await workspace.write_bytes("snaptree/new.txt", b"new")
        await workspace.remove("snaptree/sub")
        stored = SnapshotRef.model_validate(ref.model_dump(mode="json"))
        await workspace.restore(stored)
        assert await _files(workspace, "snaptree") == files

    async def test_snapshot_of_a_missing_path(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        ref = await workspace.snapshot("absent/item")
        await workspace.write_bytes("absent/item/created.txt", b"later")
        await workspace.restore(ref)
        assert not await workspace.exists("absent/item")

    @pytest.mark.parametrize("path", ESCAPING_PATHS)
    async def test_escaping_paths_are_refused(
        self, contract_workspace: Workspace, path: str
    ) -> None:
        workspace = contract_workspace
        await workspace.write_bytes("safe/file.txt", b"safe")
        operations: list[Any] = [
            lambda: workspace.read_bytes(path),
            lambda: workspace.write_bytes(path, b"escaped"),
            lambda: workspace.write_bytes(path, b"escaped", exclusive=True),
            lambda: workspace.stat(path),
            lambda: workspace.exists(path),
            lambda: workspace.list_tree(path),
            lambda: workspace.remove(path),
            lambda: workspace.digest(path),
            lambda: workspace.copy_tree("safe", path),
            lambda: workspace.copy_tree(path, "elsewhere"),
            lambda: workspace.snapshot(path),
        ]
        for operation in operations:
            with pytest.raises(WorkspacePathError):
                await operation()
        assert await workspace.read_bytes("safe/file.txt") == b"safe"

    async def test_writable_agent_backend_shares_the_files(
        self, contract_workspace: Workspace
    ) -> None:
        workspace = contract_workspace
        backend: Any = workspace.agent_backend("agent")
        assert not isinstance(backend, ReadOnlyBackend)
        assert callable(getattr(backend, "aexecute", None))
        base = agent_path(workspace, "agent")
        assert backend.write(_in_agent_root(base, "from_agent.txt"), "agent text").error is None
        assert await workspace.read_bytes("agent/from_agent.txt") == b"agent text"
        await workspace.write_bytes("agent/from_host.txt", b"host text")
        [download] = backend.download_files([_in_agent_root(base, "from_host.txt")])
        assert download.error is None
        assert download.content == b"host text"

    async def test_read_only_agent_backend(self, contract_workspace: Workspace) -> None:
        workspace = contract_workspace
        await workspace.write_bytes("ro/present.txt", b"present")
        backend: Any = workspace.agent_backend("ro", read_only=True)
        assert isinstance(backend, ReadOnlyBackend)
        base = agent_path(workspace, "ro", read_only=True)
        [download] = backend.download_files([_in_agent_root(base, "present.txt")])
        assert download.content == b"present"
        assert backend.write(_in_agent_root(base, "new.txt"), "no").error is not None
        assert backend.edit(_in_agent_root(base, "present.txt"), "present", "x").error is not None
        assert not await workspace.exists("ro/new.txt")
        assert await workspace.read_bytes("ro/present.txt") == b"present"

    async def test_run_passes_argv_without_a_shell(self, contract_workspace: Workspace) -> None:
        self._require_commands()
        workspace = contract_workspace
        await workspace.write_bytes("cmd/.keep", b"")
        argv = ["sh", "-c", 'printf "%s" "$1"; printf err >&2; exit 3', "sh", "a b;$HOME*"]
        result = await workspace.run(argv, cwd="cmd")
        assert result.argv == argv
        assert (result.stdout, result.stderr, result.returncode) == ("a b;$HOME*", "err", 3)
        assert not result.timed_out
        assert not result.truncated

    async def test_run_cwd_is_relative_to_the_root(self, contract_workspace: Workspace) -> None:
        self._require_commands()
        workspace = contract_workspace
        await workspace.write_bytes("cwd/inner/marker.txt", b"here")
        result = await workspace.run(
            ["sh", "-c", "cat marker.txt; printf made > made.txt"], cwd="cwd/inner"
        )
        assert (result.returncode, result.stdout) == (0, "here")
        assert await workspace.read_bytes("cwd/inner/made.txt") == b"made"

    async def test_run_environment(self, contract_workspace: Workspace) -> None:
        self._require_commands()
        workspace = contract_workspace
        await workspace.write_bytes("env/.keep", b"")
        script = 'printf "%s|%s" "$LAS_CONTRACT" "${LAS_CONTRACT_ABSENT-unset}"'
        given = await workspace.run(["sh", "-c", script], cwd="env", env={"LAS_CONTRACT": "v"})
        assert given.stdout == "v|unset"
        inherited = await workspace.run(["sh", "-c", "printf ok"], cwd="env")
        assert (inherited.returncode, inherited.stdout) == (0, "ok")

    async def test_run_timeout(self, contract_workspace: Workspace) -> None:
        self._require_commands()
        workspace = contract_workspace
        await workspace.write_bytes("slow/.keep", b"")
        started = time.monotonic()
        result = await workspace.run(["sh", "-c", "sleep 30"], cwd="slow", timeout=1)
        assert time.monotonic() - started < 20
        assert result.timed_out
        assert result.returncode == 124

    async def test_run_output_is_bounded(self, contract_workspace: Workspace) -> None:
        self._require_commands()
        workspace = contract_workspace
        await workspace.write_bytes("loud/.keep", b"")
        result = await workspace.run(["sh", "-c", LONG_OUTPUT], cwd="loud", max_output_bytes=1000)
        assert result.returncode == 0
        assert result.truncated
        assert 0 < len(result.stdout.encode()) <= 1000 + OUTPUT_ALLOWANCE

    async def test_agent_execute_runs_in_the_root(self, contract_workspace: Workspace) -> None:
        self._require_commands()
        workspace = contract_workspace
        backend: Any = workspace.agent_backend("exec", env={"LAS_CONTRACT": "v"})
        response = await backend.aexecute('printf made > made.txt; printf "%s" "$LAS_CONTRACT"')
        assert response.exit_code == 0
        assert "v" in response.output
        assert await workspace.read_bytes("exec/made.txt") == b"made"

    async def test_agent_execute_refuses_a_timeout_above_the_maximum(
        self, contract_workspace: Workspace
    ) -> None:
        self._require_commands()
        workspace = contract_workspace
        backend: Any = workspace.agent_backend("bounded", execute_max_timeout=5)
        response = await backend.aexecute("printf x > refused.txt", timeout=6)
        assert response.exit_code != 0
        assert not await workspace.exists("bounded/refused.txt")
        allowed = await backend.aexecute("printf x > allowed.txt", timeout=5)
        assert allowed.exit_code == 0
        assert await workspace.exists("bounded/allowed.txt")

    async def test_agent_execute_default_timeout_ends_the_command(
        self, contract_workspace: Workspace
    ) -> None:
        self._require_commands()
        workspace = contract_workspace
        backend: Any = workspace.agent_backend("timed", execute_timeout=1)
        started = time.monotonic()
        response = await backend.aexecute("(sleep 3; printf late > late.txt) & sleep 30")
        assert time.monotonic() - started < 20
        assert response.exit_code != 0
        await asyncio.sleep(4)
        assert not await workspace.exists("timed/late.txt")

    async def test_agent_execute_cancelled_ends_the_command(
        self, contract_workspace: Workspace
    ) -> None:
        self._require_commands()
        workspace = contract_workspace
        backend: Any = workspace.agent_backend("cancelled")
        task = asyncio.ensure_future(
            backend.aexecute("(sleep 3; printf late > late.txt) & sleep 30")
        )
        await asyncio.sleep(1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await asyncio.sleep(4)
        assert not await workspace.exists("cancelled/late.txt")

    async def test_agent_execute_output_is_bounded(self, contract_workspace: Workspace) -> None:
        self._require_commands()
        workspace = contract_workspace
        backend: Any = workspace.agent_backend("loud_agent", max_output_bytes=1000)
        response = await backend.aexecute(LONG_OUTPUT)
        assert response.exit_code == 0
        assert 0 < len(response.output.encode()) <= 1000 + OUTPUT_ALLOWANCE


class TestWorkspaceContract(WorkspaceContract):
    """The contract over the ``contract_workspace`` fixture of the importing suite."""


__all__ = ["TestWorkspaceContract", "WorkspaceContract"]
