"""A LocalWorkspace subclass supplies its own shell and keeps the package's composition."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol

from langgraph_agentic_scientist.workspace import AGENT_ARTIFACTS_DIR, LocalWorkspace


class RemoteShell(FilesystemBackend, SandboxBackendProtocol):
    """A shell that answers from a script, as a container's or a remote sandbox's would."""

    def __init__(self, root: Path, output: str) -> None:
        super().__init__(root_dir=root, virtual_mode=True)
        self.output = output
        self.commands: list[tuple[str, int | None]] = []

    @property
    def id(self) -> str:
        return "remote"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        self.commands.append((command, timeout))
        return ExecuteResponse(output=self.output, exit_code=0)


class RemoteWorkspace(LocalWorkspace):
    def __init__(self, root: Path, output: str = "ok") -> None:
        super().__init__(root, env={"FROM_WORKSPACE": "1"})
        self.output = output
        self.shells: list[tuple[RemoteShell, dict[str, Any]]] = []

    def agent_path(self, root: str, *, read_only: bool = False) -> str:
        return "/"

    def shell_backend(
        self,
        root_path: Path,
        *,
        timeout: int,
        max_output_bytes: int,
        env: Mapping[str, str],
    ) -> SandboxBackendProtocol:
        shell = RemoteShell(root_path, self.output)
        self.shells.append(
            (
                shell,
                {
                    "root_path": root_path,
                    "timeout": timeout,
                    "max_output_bytes": max_output_bytes,
                    "env": dict(env),
                },
            )
        )
        return shell


async def test_subclass_shell_gets_the_composition(tmp_path: Path) -> None:
    workspace = RemoteWorkspace(tmp_path / "ws")
    backend: Any = workspace.agent_backend(
        "agents/a",
        execute_timeout=30,
        execute_max_timeout=60,
        env={"FROM_SESSION": "2"},
        max_output_bytes=1000,
    )
    [(shell, arguments)] = workspace.shells
    assert arguments == {
        "root_path": workspace.root / "agents" / "a",
        "timeout": 30,
        "max_output_bytes": 1000,
        "env": {"FROM_WORKSPACE": "1", "FROM_SESSION": "2"},
    }
    assert backend.default is shell

    refused = await backend.aexecute("touch x", timeout=61)
    assert refused.exit_code == 1
    assert "exceeds maximum" in refused.output
    assert backend.execute("touch x", timeout=61).exit_code == 1
    assert shell.commands == []

    ran = await backend.aexecute("echo", timeout=60)
    assert (ran.output, ran.exit_code) == ("ok", 0)
    assert shell.commands == [("echo", 60)]

    assert backend.write("/notes.md", "kept").error is None
    assert (workspace.root / "agents" / "a" / "notes.md").read_text() == "kept"
    assert backend.write("/.agent_artifacts/history.md", "offloaded").error is None
    key = hashlib.sha256(b"agents/a").hexdigest()[:16]
    artifacts = workspace.root / AGENT_ARTIFACTS_DIR / key
    assert (artifacts / "history.md").read_text() == "offloaded"
    assert not (workspace.root / "agents" / "a" / ".agent_artifacts").exists()


async def test_output_of_a_shell_that_ignores_the_bound_is_bounded(tmp_path: Path) -> None:
    workspace = RemoteWorkspace(tmp_path / "ws", output="é" * 5000)
    backend: Any = workspace.agent_backend("a", max_output_bytes=1001)
    response = await backend.aexecute("cat big")
    assert response.truncated
    assert response.output.startswith("é" * 500)
    assert len(response.output.encode()) <= 1001 + 64
    assert response.output.endswith("Output truncated at 1001 bytes.")
    assert backend.execute("cat big").truncated


async def test_default_shell_output_is_not_cut_twice(tmp_path: Path) -> None:
    workspace = LocalWorkspace(tmp_path / "ws")
    backend: Any = workspace.agent_backend("a", max_output_bytes=100)
    response = await backend.aexecute("python3 -c \"print('a' * 1000)\"")
    assert response.truncated
    assert response.output == "a" * 100 + "\n\n... Output truncated at 100 bytes."


def test_read_only_backend_does_not_use_the_shell(tmp_path: Path) -> None:
    workspace = RemoteWorkspace(tmp_path / "ws")
    workspace.agent_backend("a", read_only=True)
    assert workspace.shells == []
