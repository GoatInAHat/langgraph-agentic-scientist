"""The workspace protocol: files and commands for host steps, a backend for agents.

A :class:`Workspace` is where a run keeps its files: the task snapshot, the programs, the paper
and everything the agents write. Host nodes use its file and command operations; agents get a
Deep Agents backend rooted at a sub-directory through :meth:`Workspace.agent_backend`. The
package ships :class:`~langgraph_agentic_scientist.workspace.LocalWorkspace`, a directory on
the host. A caller substitutes a git worktree, a container or a remote sandbox by implementing
this protocol; any Deep Agents sandbox backend can serve :meth:`Workspace.agent_backend`.
:class:`~langgraph_agentic_scientist.workspace.WorkspaceBase` derives every operation but a core
of six (``read_bytes``, ``write_bytes``, ``stat``, ``list_tree``, ``remove``, ``run``) and
``agent_backend``, so an implementation need only write those.

Paths are POSIX paths relative to the workspace root. ``""`` and ``"."`` name the root.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import Field

from ..core.state import StateModel

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

EntryKind = Literal["file", "dir", "symlink", "other"]


class FileEntry(StateModel):
    """One entry of a workspace tree listing.

    Attributes:
        path: POSIX path relative to the listed directory (for :meth:`Workspace.list_tree`) or
            to the workspace root (for :meth:`Workspace.stat`).
        size: Size in bytes of a regular file, otherwise 0.
        kind: ``file``, ``dir``, ``symlink`` or ``other``.
        sha256: ``sha256:<hex>`` of a regular file when the listing computed it.
        target: Link target of a symlink.
    """

    path: str
    size: int = 0
    kind: EntryKind = "file"
    sha256: str | None = None
    target: str | None = None


class CommandResult(StateModel):
    """The outcome of :meth:`Workspace.run`.

    ``stdout`` and ``stderr`` are bounded; ``truncated`` says whether output was dropped.
    A command killed by its timeout has ``timed_out=True`` and ``returncode=124``.
    """

    argv: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    duration_seconds: float = 0.0


class SnapshotRef(StateModel):
    """A reference to a snapshot taken by :meth:`Workspace.snapshot`; storable in state."""

    snapshot_id: str
    path: str
    digest: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class Workspace(Protocol):
    """Files and commands of a run, and the agents' view of them.

    Implementations must make every path relative to their root and refuse paths that escape it
    (:func:`~langgraph_agentic_scientist.workspace.paths.safe_relpath`).
    """

    @property
    def isolated(self) -> bool:
        """True when agent commands run isolated from the host (a container or remote sandbox).

        Upstream asks a person before pruning only when the agent is not containerised.
        """
        ...

    def agent_backend(
        self,
        root: str,
        *,
        read_only: bool = False,
        execute_timeout: int = 600,
        execute_max_timeout: int = 7200,
        env: Mapping[str, str] | None = None,
    ) -> BackendProtocol:
        """Return a Deep Agents backend rooted at ``root``.

        With ``read_only=False`` the backend is a ``SandboxBackendProtocol`` whose ``execute``
        uses ``execute_timeout`` when a command names none and refuses a requested timeout above
        ``execute_max_timeout``; ``env`` adds environment variables for commands. With
        ``read_only=True`` it is a :class:`~langgraph_agentic_scientist.workspace.ReadOnlyBackend`.
        """
        ...

    async def read_bytes(self, path: str) -> bytes:
        """Return the bytes of a regular file; raise ``FileNotFoundError`` when absent."""
        ...

    async def write_bytes(self, path: str, data: bytes, *, exclusive: bool = False) -> None:
        """Write a file atomically, creating parent directories.

        With ``exclusive=True`` raise ``FileExistsError`` when the file exists.
        """
        ...

    async def exists(self, path: str) -> bool:
        """Return whether anything exists at ``path`` (a dangling symlink counts)."""
        ...

    async def stat(self, path: str) -> FileEntry | None:
        """Return the entry at ``path`` without following a final symlink, or ``None``."""
        ...

    async def list_tree(self, path: str, *, with_digest: bool = False) -> list[FileEntry]:
        """List every entry below ``path`` recursively, sorted by path, without following links.

        Entry paths are relative to ``path``. With ``with_digest=True`` regular files carry their
        ``sha256`` (which reads every file); otherwise ``sha256`` is ``None``.
        """
        ...

    async def copy_tree(self, src: str, dst: str, *, ignore: Sequence[str] = ()) -> None:
        """Copy a directory tree; ``ignore`` takes ``.agentic_task_ignore`` patterns.

        Symlinks are copied as symlinks. ``dst`` must not exist.
        """
        ...

    async def remove(self, path: str) -> None:
        """Remove a file or a directory tree; a missing path is not an error."""
        ...

    async def digest(self, path: str) -> str:
        """Return ``sha256:<hex>`` of a file, or upstream's tree digest of a directory."""
        ...

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_output_bytes: int = 65536,
    ) -> CommandResult:
        """Run a command (no shell) with ``cwd`` relative to the workspace root.

        ``env=None`` inherits the implementation's default environment; a mapping is used as
        the whole environment. Output beyond ``max_output_bytes`` per stream is dropped.
        """
        ...

    def host_path(self, path: str) -> str:
        """Return the path of ``path`` as the processes started by :meth:`run` see it."""
        ...

    async def snapshot(self, path: str) -> SnapshotRef:
        """Snapshot a file or directory so that :meth:`restore` can bring it back."""
        ...

    async def restore(self, ref: SnapshotRef) -> None:
        """Restore a snapshot, replacing whatever is at its path now."""
        ...


async def is_file(workspace: Workspace, path: str) -> bool:
    """Return whether ``path`` is a regular file (a symlink is not)."""
    entry = await workspace.stat(path)
    return entry is not None and entry.kind == "file"


async def is_dir(workspace: Workspace, path: str) -> bool:
    """Return whether ``path`` is a directory (a symlink is not)."""
    entry = await workspace.stat(path)
    return entry is not None and entry.kind == "dir"


async def is_symlink(workspace: Workspace, path: str) -> bool:
    """Return whether ``path`` is a symlink."""
    entry = await workspace.stat(path)
    return entry is not None and entry.kind == "symlink"


async def read_text(workspace: Workspace, path: str) -> str:
    """Read a UTF-8 text file."""
    return (await workspace.read_bytes(path)).decode("utf-8")


async def write_text(
    workspace: Workspace, path: str, text: str, *, exclusive: bool = False
) -> None:
    """Write a UTF-8 text file atomically."""
    await workspace.write_bytes(path, text.encode("utf-8"), exclusive=exclusive)


__all__ = [
    "CommandResult",
    "EntryKind",
    "FileEntry",
    "SnapshotRef",
    "Workspace",
    "is_dir",
    "is_file",
    "is_symlink",
    "read_text",
    "write_text",
]
