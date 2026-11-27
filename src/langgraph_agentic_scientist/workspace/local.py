"""A workspace that is a directory on the host.

:class:`LocalWorkspace` implements :class:`~langgraph_agentic_scientist.workspace.Workspace` over
a host directory. Agents get Deep Agents' ``LocalShellBackend`` rooted at their sub-directory.

.. warning::
   ``execute`` runs agent commands on the host with the permissions of the calling process.
   ``LocalWorkspace`` is not a sandbox; use a container or remote-sandbox workspace for code you
   do not trust. Upstream runs its agents in containers for the same reason.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import os
import shutil
import signal
import stat
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from deepagents.backends.protocol import BackendProtocol, ExecuteResponse, SandboxBackendProtocol

from ..core.errors import WorkspacePathError
from .digest import sha256_bytes, tree_digest
from .paths import safe_relpath, task_copy_path_is_ignored
from .protocol import CommandResult, EntryKind, FileEntry, SnapshotRef
from .readonly import ReadOnlyBackend

SNAPSHOT_DIR = ".snapshots"
AGENT_ARTIFACTS_DIR = ".agent_artifacts"
AGENT_ARTIFACTS_ROUTE = "/.agent_artifacts/"


def _within_root(cwd: Path, workspace_root: Path, key: str) -> str:
    """Map a host path inside ``cwd`` to the virtual path that names it under ``/``."""
    if not key.startswith("/"):
        return key
    path = Path(key)
    if path == cwd or cwd in path.parents:
        return "/" + path.relative_to(cwd).as_posix() if path != cwd else "/"
    if path == workspace_root or workspace_root in path.parents:
        raise PermissionError(f"{key} is outside the working directory {cwd}")
    return key


class _HostPathsFilesystemBackend(FilesystemBackend):
    """A virtual ``FilesystemBackend`` that also takes the host paths of its root."""

    def __init__(self, *args: Any, workspace_root: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._workspace_root = workspace_root

    def _resolve_path(self, key: str) -> Path:
        return super()._resolve_path(_within_root(self.cwd, self._workspace_root, key))


class _HostPathsShellBackend(LocalShellBackend):
    """A virtual ``LocalShellBackend`` whose file tools also take the host paths of its root.

    The shell runs on the host, in the root, and prints host paths; the file tools accept them
    as well as the virtual ``/``-rooted form, so that both name the same files.
    """

    def __init__(self, *args: Any, workspace_root: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._workspace_root = workspace_root

    def _resolve_path(self, key: str) -> Path:
        return super()._resolve_path(_within_root(self.cwd, self._workspace_root, key))


class _BoundedCompositeBackend(CompositeBackend):
    """``CompositeBackend`` that refuses a per-command timeout above a maximum and bounds output.

    The bounds sit on the composite so that the shell stays an unmodified ``LocalShellBackend``:
    Deep Agents gives cancellation cleanup only to a ``LocalShellBackend`` whose ``execute`` is not
    overridden, and that cleanup is what kills a command's process group when the session's wall
    clock cancels it. Output beyond ``max_output_bytes`` and the truncation note is cut to its
    first ``max_output_bytes``, so a shell that ignores the bound it was given is bounded too.
    """

    def __init__(self, *args: Any, max_timeout: int, max_output_bytes: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._max_timeout = max_timeout
        self._max_output_bytes = max_output_bytes

    def _bounded(self, response: ExecuteResponse) -> ExecuteResponse:
        note = f"\n\n... Output truncated at {self._max_output_bytes} bytes."
        data = response.output.encode("utf-8")
        if len(data) <= self._max_output_bytes + len(note.encode("utf-8")):
            return response
        head = data[: self._max_output_bytes].decode("utf-8", errors="ignore")
        return replace(response, output=head + note, truncated=True)

    def _refusal(self, timeout: int | None) -> ExecuteResponse | None:
        if timeout is not None and timeout > self._max_timeout:
            return ExecuteResponse(
                output=(
                    f"Error: timeout {timeout}s exceeds maximum allowed ({self._max_timeout}s)."
                ),
                exit_code=1,
            )
        return None

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run ``command`` unless its timeout exceeds the maximum."""
        return self._refusal(timeout) or self._bounded(super().execute(command, timeout=timeout))

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Run ``command`` unless its timeout exceeds the maximum."""
        refusal = self._refusal(timeout)
        if refusal is not None:
            return refusal
        return self._bounded(await super().aexecute(command, timeout=timeout))


class LocalWorkspace:
    """A :class:`~langgraph_agentic_scientist.workspace.Workspace` that is a host directory.

    Args:
        root: The workspace root; created when missing.
        inherit_env: Whether agent commands and :meth:`run` inherit the calling process's
            environment (upstream's agent CLIs do). When false only ``env`` is passed.
        env: Extra environment variables for agent commands and :meth:`run`.

    ``execute`` runs on the host and is not a sandbox (see the module docstring). A subclass
    whose agent commands run elsewhere overrides :meth:`shell_backend` (and, when that shell sees
    the workspace at another path, :meth:`agent_path`) and keeps the files and the rest.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        inherit_env: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.inherit_env = inherit_env
        self.env = dict(env or {})

    def __repr__(self) -> str:
        return f"LocalWorkspace({str(self.root)!r})"

    # --- protocol ---------------------------------------------------------------------------

    @property
    def isolated(self) -> bool:
        """Always false: agent commands run on the host."""
        return False

    def _path(self, path: str, *, follow: bool = False) -> Path:
        relative = safe_relpath(path)
        target = self.root / relative if relative else self.root
        # The parent chain must stay inside the root even when it contains symlinks.
        parent = target.parent.resolve() if relative else self.root
        if parent != self.root and self.root not in parent.parents:
            raise WorkspacePathError(f"workspace path escapes the workspace: {path!r}")
        if follow and relative:
            resolved = target.resolve()
            if resolved != self.root and self.root not in resolved.parents:
                raise WorkspacePathError(f"workspace path escapes the workspace: {path!r}")
        return target

    def host_path(self, path: str) -> str:
        """Return the absolute host path of a workspace path."""
        return str(self._path(path))

    def agent_path(self, root: str, *, read_only: bool = False) -> str:
        """Return the host path of ``root``: what an agent's shell prints, and its tools take.

        A subclass whose :meth:`shell_backend` sees the workspace at another path returns the
        path at which that shell and its file tools name ``root``.
        """
        return str(self._path(root, follow=True))

    def shell_backend(
        self,
        root_path: Path,
        *,
        timeout: int,
        max_output_bytes: int,
        env: Mapping[str, str],
    ) -> SandboxBackendProtocol:
        """Return the shell and file tools of a writable agent backend rooted at ``root_path``.

        ``root_path`` is the host directory of the agent's root, already created; ``timeout`` is
        the default timeout of a command, ``max_output_bytes`` the output to keep and ``env`` the
        workspace's ``env`` updated with the session's (``inherit_env`` is on ``self``). The
        returned backend runs commands in that directory and kills a command, with whatever it
        started, when it times out or its call is cancelled. :meth:`agent_backend` puts it under
        the route of Deep Agents' own files, the refusal of a timeout above the maximum and the
        output bound. The default runs ``LocalShellBackend`` on the host, whose file tools take
        the host paths of ``root_path`` as well as ``/``-rooted ones.
        """
        return _HostPathsShellBackend(
            root_dir=root_path,
            virtual_mode=True,
            workspace_root=self.root,
            timeout=timeout,
            max_output_bytes=max_output_bytes,
            env=dict(env),
            inherit_env=self.inherit_env,
        )

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
        """Return a Deep Agents backend rooted at the workspace directory ``root``.

        Writable backends execute commands with ``execute_timeout`` as the default timeout,
        refuse longer requests than ``execute_max_timeout`` and keep at most ``max_output_bytes``
        of a command's output. A command that times out or whose session is cancelled is killed
        with its whole process group. Deep Agents' own files (offloaded
        large tool results and conversation history) are routed to
        ``.agent_artifacts/<root>/`` outside the role's root, for read-only backends too, so that
        the role's directory holds only what the agent wrote on purpose. The shell and the file
        tools come from :meth:`shell_backend`.
        """
        root_path = self._path(root, follow=True)
        root_path.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(safe_relpath(root).encode()).hexdigest()[:16]
        artifacts = self.root / AGENT_ARTIFACTS_DIR / key
        artifacts.mkdir(parents=True, exist_ok=True)
        if read_only:
            return ReadOnlyBackend(
                _HostPathsFilesystemBackend(
                    root_dir=root_path, virtual_mode=True, workspace_root=self.root
                ),
                routes={
                    AGENT_ARTIFACTS_ROUTE: FilesystemBackend(root_dir=artifacts, virtual_mode=True)
                },
                artifacts_root=AGENT_ARTIFACTS_ROUTE.rstrip("/"),
            )
        shell = self.shell_backend(
            root_path,
            timeout=execute_timeout,
            max_output_bytes=max_output_bytes,
            env={**self.env, **dict(env or {})},
        )
        return _BoundedCompositeBackend(
            default=shell,
            routes={
                AGENT_ARTIFACTS_ROUTE: FilesystemBackend(root_dir=artifacts, virtual_mode=True)
            },
            artifacts_root=AGENT_ARTIFACTS_ROUTE.rstrip("/"),
            max_timeout=execute_max_timeout,
            max_output_bytes=max_output_bytes,
        )

    async def read_bytes(self, path: str) -> bytes:
        """Return the bytes of a regular file."""
        target = self._path(path, follow=True)
        return await asyncio.to_thread(target.read_bytes)

    async def write_bytes(self, path: str, data: bytes, *, exclusive: bool = False) -> None:
        """Write a file atomically (temporary file and rename), creating parent directories."""
        target = self._path(path)
        if not safe_relpath(path):
            raise WorkspacePathError("cannot write the workspace root as a file")
        await asyncio.to_thread(_write_file, target, data, exclusive)

    async def exists(self, path: str) -> bool:
        """Return whether anything exists at ``path``."""
        return await self.stat(path) is not None

    async def stat(self, path: str) -> FileEntry | None:
        """Return the entry at ``path`` without following a final symlink."""
        target = self._path(path)
        try:
            info = await asyncio.to_thread(os.lstat, target)
        except FileNotFoundError:
            return None
        return _entry(safe_relpath(path), target, info, with_digest=False)

    async def list_tree(self, path: str, *, with_digest: bool = False) -> list[FileEntry]:
        """List every entry below ``path`` recursively; ``with_digest`` hashes regular files."""
        base = self._path(path, follow=True)
        return await asyncio.to_thread(_list_tree, base, with_digest)

    async def copy_tree(self, src: str, dst: str, *, ignore: Sequence[str] = ()) -> None:
        """Copy a directory tree, honouring ``.agentic_task_ignore`` patterns."""
        source = self._path(src, follow=True)
        destination = self._path(dst)
        if not safe_relpath(dst):
            raise WorkspacePathError("cannot copy over the workspace root")
        await asyncio.to_thread(_copy_tree, source, destination, tuple(ignore))

    async def remove(self, path: str) -> None:
        """Remove a file or directory tree; a missing path is not an error."""
        if not safe_relpath(path):
            raise WorkspacePathError("cannot remove the workspace root")
        target = self._path(path)
        await asyncio.to_thread(_remove, target)

    async def digest(self, path: str) -> str:
        """Return ``sha256:<hex>`` of a file, or upstream's tree digest of a directory."""
        target = self._path(path, follow=True)
        if target.is_dir():
            base = safe_relpath(path)
            entries = await self.list_tree(path)

            async def read(relative: str) -> bytes:
                return await self.read_bytes(f"{base}/{relative}" if base else relative)

            return await tree_digest(entries, read)
        return sha256_bytes(await self.read_bytes(path))

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_output_bytes: int = 65536,
    ) -> CommandResult:
        """Run a command without a shell in its own process group.

        On timeout the whole process group is killed and ``returncode`` is 124.
        """
        workdir = self._path(cwd, follow=True)
        if env is None:
            environment = {**(os.environ if self.inherit_env else {}), **self.env}
        else:
            environment = dict(env)
        started = time.monotonic()
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(workdir),
            env=environment,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout_task = asyncio.ensure_future(_read_bounded(process.stdout, max_output_bytes))
        stderr_task = asyncio.ensure_future(_read_bounded(process.stderr, max_output_bytes))
        timed_out = False
        try:
            await asyncio.wait_for(process.wait(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            _kill_group(process.pid)
            await process.wait()
        except BaseException:
            _kill_group(process.pid)
            with contextlib.suppress(BaseException):
                await process.wait()
            raise
        stdout, stdout_truncated = await stdout_task
        stderr, stderr_truncated = await stderr_task
        return CommandResult(
            argv=list(argv),
            returncode=124 if timed_out else (process.returncode or 0),
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
            duration_seconds=time.monotonic() - started,
        )

    async def snapshot(self, path: str) -> SnapshotRef:
        """Copy ``path`` under ``.snapshots/<id>/`` and return a reference to the copy."""
        relative = safe_relpath(path, allow_root=False)
        source = self._path(relative)
        snapshot_id = uuid.uuid4().hex
        target = self.root / SNAPSHOT_DIR / snapshot_id / "data"
        exists = await self.exists(relative)
        await asyncio.to_thread(_snapshot_copy, source, target, exists)
        digest = await self.digest(relative) if exists else None
        return SnapshotRef(
            snapshot_id=snapshot_id,
            path=relative,
            digest=digest,
            metadata={"existed": "true" if exists else "false"},
        )

    async def restore(self, ref: SnapshotRef) -> None:
        """Replace ``ref.path`` with the snapshot copy (or remove it if it did not exist)."""
        relative = safe_relpath(ref.path, allow_root=False)
        target = self._path(relative)
        source = self.root / SNAPSHOT_DIR / safe_relpath(ref.snapshot_id) / "data"
        existed = ref.metadata.get("existed", "true") == "true"
        if existed and not source.exists() and not source.is_symlink():
            raise FileNotFoundError(f"snapshot {ref.snapshot_id} is missing")
        await asyncio.to_thread(_restore_copy, source, target, existed)


# --- helpers run in worker threads ----------------------------------------------------------


def _write_file(target: Path, data: bytes, exclusive: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o644)
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        return
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _kind(info: os.stat_result) -> EntryKind:
    if stat.S_ISLNK(info.st_mode):
        return "symlink"
    if stat.S_ISDIR(info.st_mode):
        return "dir"
    if stat.S_ISREG(info.st_mode):
        return "file"
    return "other"


def _entry(relative: str, target: Path, info: os.stat_result, *, with_digest: bool) -> FileEntry:
    kind = _kind(info)
    digest = None
    if with_digest and kind == "file":
        digest = sha256_bytes(target.read_bytes())
    return FileEntry(
        path=relative,
        size=info.st_size if kind == "file" else 0,
        kind=kind,
        sha256=digest,
        target=os.readlink(target) if kind == "symlink" else None,
    )


def _list_tree(base: Path, with_digest: bool) -> list[FileEntry]:
    if not base.is_dir():
        raise FileNotFoundError(f"not a directory: {base}")
    entries: list[FileEntry] = []
    for directory, dirnames, filenames in os.walk(base, followlinks=False):
        current = Path(directory)
        for name in (*dirnames, *filenames):
            path = current / name
            info = os.lstat(path)
            relative = path.relative_to(base).as_posix()
            entries.append(_entry(relative, path, info, with_digest=with_digest))
    entries.sort(key=lambda item: item.path)
    return entries


def _copy_tree(source: Path, destination: Path, patterns: tuple[str, ...]) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"not a directory: {source}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination exists: {destination}")

    def ignore(current: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        if not patterns:
            return ignored
        for name in names:
            full = Path(current) / name
            relative = full.relative_to(source).as_posix()
            if task_copy_path_is_ignored(
                relative, is_dir=stat.S_ISDIR(full.lstat().st_mode), patterns=patterns
            ):
                ignored.add(name)
        return ignored

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copytree(source, temporary, symlinks=True, ignore=ignore)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _remove(target: Path) -> None:
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)


def _snapshot_copy(source: Path, target: Path, exists: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not exists:
        return
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target, follow_symlinks=False)


def _restore_copy(source: Path, target: Path, existed: bool) -> None:
    _remove(target)
    if not existed:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(source, target, symlinks=True)
    else:
        shutil.copy2(source, target, follow_symlinks=False)


async def _read_bounded(stream: asyncio.StreamReader, limit: int) -> tuple[bytes, bool]:
    """Read a stream to its end, keeping its head and tail within ``limit`` bytes in total."""
    head = bytearray()
    tail = bytearray()
    half = max(limit // 2, 1)
    truncated = False
    while chunk := await stream.read(65536):
        if len(head) < half:
            room = half - len(head)
            head.extend(chunk[:room])
            chunk = chunk[room:]
        if chunk:
            tail.extend(chunk)
            if len(tail) > limit - half:
                truncated = True
                del tail[: len(tail) - (limit - half)]
    if truncated:
        return bytes(head) + b"\n...[output truncated]...\n" + bytes(tail), True
    return bytes(head) + bytes(tail), False


def _kill_group(pid: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)


__all__ = ["AGENT_ARTIFACTS_DIR", "AGENT_ARTIFACTS_ROUTE", "SNAPSHOT_DIR", "LocalWorkspace"]
