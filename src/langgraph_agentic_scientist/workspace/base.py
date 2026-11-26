"""A base class that derives the optional workspace operations from a small core.

:class:`~langgraph_agentic_scientist.workspace.Workspace` is the whole interface the package calls.
Most of it can be derived from six operations, so a container or remote-sandbox workspace need
only implement those:

=====================  =========================================================================
Core (required)        ``read_bytes``, ``write_bytes``, ``stat``, ``list_tree``, ``remove``,
                       ``run``, and ``agent_backend`` for the agents
Derived (optional)     ``exists`` (from ``stat``), ``digest`` (from ``list_tree`` and
                       ``read_bytes``), ``copy_tree`` (from ``list_tree``, ``read_bytes`` and
                       ``write_bytes``), ``snapshot`` and ``restore`` (copies under
                       ``.snapshots/``), ``host_path`` (the path itself), ``isolated`` (true)
=====================  =========================================================================

Subclass :class:`WorkspaceBase`, implement the core, and override a derived operation when the
backend can do it better (a server-side copy, a native snapshot). The derived ``copy_tree`` copies
regular files; a tree holding a symlink is refused (``NotImplementedError``), and empty
directories are not carried, because the core has no operation for either.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from .digest import sha256_bytes, tree_digest
from .paths import join, safe_relpath, task_copy_path_is_ignored
from .protocol import CommandResult, FileEntry, SnapshotRef

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

SNAPSHOT_DIR = ".snapshots"


class WorkspaceBase(ABC):
    """Implements the derived operations of :class:`Workspace` over the core ones."""

    # --- core --------------------------------------------------------------------------------

    @abstractmethod
    def agent_backend(
        self,
        root: str,
        *,
        read_only: bool = False,
        execute_timeout: int = 600,
        execute_max_timeout: int = 7200,
        env: Mapping[str, str] | None = None,
        max_output_bytes: int = 100_000,
    ) -> BackendProtocol:
        """Return a Deep Agents backend rooted at ``root`` (see :class:`Workspace`)."""

    @abstractmethod
    async def read_bytes(self, path: str) -> bytes:
        """Return the bytes of a regular file; raise ``FileNotFoundError`` when absent."""

    @abstractmethod
    async def write_bytes(self, path: str, data: bytes, *, exclusive: bool = False) -> None:
        """Write a file atomically, creating parent directories."""

    @abstractmethod
    async def stat(self, path: str) -> FileEntry | None:
        """Return the entry at ``path`` without following a final symlink, or ``None``."""

    @abstractmethod
    async def list_tree(self, path: str, *, with_digest: bool = False) -> list[FileEntry]:
        """List every entry below ``path`` recursively, sorted by path."""

    @abstractmethod
    async def remove(self, path: str) -> None:
        """Remove a file or a directory tree; a missing path is not an error."""

    @abstractmethod
    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_output_bytes: int = 65536,
    ) -> CommandResult:
        """Run a command (no shell) with ``cwd`` relative to the workspace root."""

    # --- derived -----------------------------------------------------------------------------

    @property
    def isolated(self) -> bool:
        """True by default: a workspace other than the host directory is taken to isolate."""
        return True

    def host_path(self, path: str) -> str:
        """Return ``path`` unchanged (commands see workspace paths as they are)."""
        return safe_relpath(path)

    async def exists(self, path: str) -> bool:
        """Return whether anything exists at ``path``."""
        return await self.stat(path) is not None

    async def digest(self, path: str) -> str:
        """Return ``sha256:<hex>`` of a file, or upstream's tree digest of a directory."""
        entry = await self.stat(path)
        if entry is None:
            raise FileNotFoundError(path)
        if entry.kind != "dir":
            return sha256_bytes(await self.read_bytes(path))
        base = safe_relpath(path)

        async def read(relative: str) -> bytes:
            return await self.read_bytes(join(base, relative))

        return await tree_digest(await self.list_tree(path), read)

    async def copy_tree(self, src: str, dst: str, *, ignore: Sequence[str] = ()) -> None:
        """Copy the regular files of a tree; ``ignore`` takes ``.agentic_task_ignore`` patterns.

        Raises:
            FileExistsError: When ``dst`` exists.
            NotImplementedError: When the tree holds a symlink (override to copy links).
        """
        if await self.exists(dst):
            raise FileExistsError(dst)
        entries = await self.list_tree(src)
        kept: list[FileEntry] = []
        for entry in entries:
            parts = entry.path.split("/")
            if any(
                task_copy_path_is_ignored(
                    "/".join(parts[:depth]),
                    is_dir=depth < len(parts) or entry.kind == "dir",
                    patterns=ignore,
                )
                for depth in range(1, len(parts) + 1)
            ):
                continue
            if entry.kind == "symlink":
                raise NotImplementedError(
                    f"{type(self).__name__}.copy_tree cannot copy the symlink {entry.path}"
                )
            kept.append(entry)
        for entry in kept:
            if entry.kind == "file":
                data = await self.read_bytes(join(src, entry.path))
                await self.write_bytes(join(dst, entry.path), data)

    async def snapshot(self, path: str) -> SnapshotRef:
        """Copy ``path`` under ``.snapshots/<id>/data`` and return a reference to the copy."""
        relative = safe_relpath(path, allow_root=False)
        snapshot_id = uuid.uuid4().hex
        target = join(SNAPSHOT_DIR, snapshot_id, "data")
        entry = await self.stat(relative)
        if entry is None:
            return SnapshotRef(
                snapshot_id=snapshot_id, path=relative, metadata={"existed": "false"}
            )
        if entry.kind == "dir":
            await self.copy_tree(relative, target)
        else:
            await self.write_bytes(target, await self.read_bytes(relative))
        return SnapshotRef(
            snapshot_id=snapshot_id,
            path=relative,
            digest=await self.digest(relative),
            metadata={"existed": "true", "kind": entry.kind},
        )

    async def restore(self, ref: SnapshotRef) -> None:
        """Replace ``ref.path`` with the snapshot copy (or remove it if it did not exist)."""
        relative = safe_relpath(ref.path, allow_root=False)
        source = join(SNAPSHOT_DIR, safe_relpath(ref.snapshot_id), "data")
        await self.remove(relative)
        if ref.metadata.get("existed", "true") != "true":
            return
        entry = await self.stat(source)
        if entry is None:
            raise FileNotFoundError(f"snapshot {ref.snapshot_id} is missing")
        if entry.kind == "dir":
            await self.copy_tree(source, relative)
        else:
            await self.write_bytes(relative, await self.read_bytes(source))


__all__ = ["SNAPSHOT_DIR", "WorkspaceBase"]
