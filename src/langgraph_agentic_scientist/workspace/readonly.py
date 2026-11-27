"""A Deep Agents backend through which an agent can read but change nothing.

Upstream runs its read-only roles (judges, auditors, planners, selectors) on a read-only copy in
a container. The port gives them :class:`ReadOnlyBackend`, a ``CompositeBackend`` whose default
backend implements only ``BackendProtocol``, not ``SandboxBackendProtocol``, so Deep Agents'
``execute`` tool reports that execution is unavailable; reads, listings, glob and grep delegate to
the inner backend; writes, edits, deletes and uploads are refused with a message the agent reads.
Its routes, when given, hold only Deep Agents' own files outside the role's directory. Deep Agents'
own ``permissions`` are not used for this, because they apply only to the built-in file tools and
not to ``execute`` or to direct backend calls.
"""

from __future__ import annotations

from collections.abc import Mapping

from deepagents.backends.composite import CompositeBackend
from deepagents.backends.protocol import (
    PERMISSION_DENIED,
    BackendProtocol,
    DeleteResult,
    EditResult,
    FileDownloadResponse,
    FileUploadResponse,
    GlobResult,
    GrepResult,
    LsResult,
    ReadResult,
    WriteResult,
)

READ_ONLY_MESSAGE = (
    "Error: permission denied. This workspace is read-only for your role: files cannot be "
    "written, edited or deleted, and commands cannot be executed."
)


class _ReadOnlyView(BackendProtocol):
    """Wrap a backend so that every read works and every change is refused."""

    def __init__(self, inner: BackendProtocol) -> None:
        self.inner = inner

    # --- reads delegate -------------------------------------------------------------------

    def ls(self, path: str) -> LsResult:
        """List a directory through the inner backend."""
        return self.inner.ls(path)

    async def als(self, path: str) -> LsResult:
        """Async version of :meth:`ls`."""
        return await self.inner.als(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Read a file through the inner backend."""
        return self.inner.read(file_path, offset, limit)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        """Async version of :meth:`read`."""
        return await self.inner.aread(file_path, offset, limit)

    def grep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Search file contents through the inner backend."""
        return self.inner.grep(pattern, path, glob, max_count=max_count)

    async def agrep(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        *,
        max_count: int | None = None,
    ) -> GrepResult:
        """Async version of :meth:`grep`."""
        return await self.inner.agrep(pattern, path, glob, max_count=max_count)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Find files through the inner backend."""
        return self.inner.glob(pattern, path)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        """Async version of :meth:`glob`."""
        return await self.inner.aglob(pattern, path)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files through the inner backend."""
        return self.inner.download_files(paths)

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Async version of :meth:`download_files`."""
        return await self.inner.adownload_files(paths)

    # --- changes are refused --------------------------------------------------------------

    def write(self, file_path: str, content: str) -> WriteResult:
        """Refuse to write."""
        return WriteResult(error=READ_ONLY_MESSAGE)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        """Refuse to write."""
        return WriteResult(error=READ_ONLY_MESSAGE)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Refuse to edit."""
        return EditResult(error=READ_ONLY_MESSAGE)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Refuse to edit."""
        return EditResult(error=READ_ONLY_MESSAGE)

    def delete(self, file_path: str) -> DeleteResult:
        """Refuse to delete."""
        return DeleteResult(error=READ_ONLY_MESSAGE)

    async def adelete(self, file_path: str) -> DeleteResult:
        """Refuse to delete."""
        return DeleteResult(error=READ_ONLY_MESSAGE)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Refuse every upload."""
        return [FileUploadResponse(path=path, error=PERMISSION_DENIED) for path, _ in files]

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Refuse every upload."""
        return self.upload_files(files)


class ReadOnlyBackend(CompositeBackend):
    """A backend whose role can read ``inner`` and change nothing in it.

    Deep Agents keeps its own files (conversation history offloaded when a session is summarised,
    large tool results) under the ``artifacts_root`` of a ``CompositeBackend``; ``routes`` are the
    writable backends that hold them, outside the role's directory. Without routes nothing at all
    can be written.

    Args:
        inner: The backend that serves reads.
        routes: Writable backends for Deep Agents' own files, by path prefix.
        artifacts_root: Where Deep Agents puts its own files (a prefix of ``routes``).
    """

    def __init__(
        self,
        inner: BackendProtocol,
        *,
        routes: Mapping[str, BackendProtocol] | None = None,
        artifacts_root: str = "/",
    ) -> None:
        super().__init__(
            default=_ReadOnlyView(inner), routes=dict(routes or {}), artifacts_root=artifacts_root
        )
        self.inner = inner


__all__ = ["READ_ONLY_MESSAGE", "ReadOnlyBackend"]
