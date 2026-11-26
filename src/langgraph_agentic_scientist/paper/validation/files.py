"""A synchronous, read-only view of a paper directory for the validation gates.

Upstream's validators open the paper directory with ``os.open`` and refuse symbolic links at every
path component. The port reads the paper directory through the
:class:`~langgraph_agentic_scientist.Workspace` protocol, which a caller may implement on a
container or a remote sandbox, so the validators work on a :class:`PaperTree`: a synchronous view
with the same rules (a regular file whose every ancestor is a real directory). Two
implementations exist:

* :class:`SnapshotTree` is filled asynchronously from any workspace
  (:meth:`SnapshotTree.load`) and is what :class:`WorkspacePaperToolchain` validates;
* :class:`DirectoryTree` reads a host directory directly and serves the figure validator that the
  ``validate-paper`` skill runs inside a workspace.
"""

from __future__ import annotations

import os
import posixpath
import stat
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Protocol, runtime_checkable

from ...workspace.protocol import EntryKind, FileEntry, Workspace


@runtime_checkable
class PaperTree(Protocol):
    """A read-only view of a paper directory (POSIX paths relative to its root)."""

    def kind(self, rel: str) -> EntryKind | None:
        """Return the kind of the entry at ``rel`` without following a final symlink."""
        ...

    def read(self, rel: str, limit: int | None = None) -> bytes | None:
        """Return the content of a regular file (at most ``limit`` bytes), or ``None``.

        ``None`` means that the entry is absent, is not a regular file, or was not loaded.
        """
        ...

    def readlink(self, rel: str) -> str | None:
        """Return the target of a symlink, or ``None``."""
        ...

    def files(self) -> list[str]:
        """Return every regular file below the root, sorted, without following links."""
        ...

    def sha256(self, rel: str) -> str | None:
        """Return ``sha256:<hex>`` of a regular file when known."""
        ...


def as_tree(source: PaperTree | Path | str) -> PaperTree:
    """Return ``source`` as a tree; a path is read as a host directory."""
    if isinstance(source, str | Path):
        return DirectoryTree(source)
    return source


def clean_relative(rel: str) -> str:
    """Normalise a relative path (``./x`` to ``x``); raise ``ValueError`` when it escapes."""
    normal = posixpath.normpath(rel) if rel not in ("", ".") else "."
    if normal.startswith("/") or normal == ".." or normal.startswith("../"):
        raise ValueError(f"path escapes the paper workspace: {rel!r}")
    return "." if normal == "." else normal


def lexists(tree: PaperTree, rel: str) -> bool:
    """Return whether anything exists at ``rel`` (a dangling symlink counts)."""
    return tree.kind(clean_relative(rel)) is not None


def is_regular_nofollow(tree: PaperTree, rel: str) -> bool:
    """Return whether ``rel`` is a regular file whose ancestors are all real directories."""
    path = PurePosixPath(clean_relative(rel))
    parts = path.parts
    if not parts:
        return False
    for index in range(1, len(parts)):
        if tree.kind("/".join(parts[:index])) != "dir":
            return False
    return tree.kind(path.as_posix()) == "file"


def read_regular_nofollow(tree: PaperTree, rel: str, limit: int | None = None) -> bytes | None:
    """Read a regular file without following symlinks; ``None`` when absent or unsafe."""
    if not is_regular_nofollow(tree, rel):
        return None
    return tree.read(clean_relative(rel), limit)


def read_text(tree: PaperTree, rel: str) -> str | None:
    """Read a regular file as UTF-8 text (invalid bytes are replaced), or ``None``."""
    data = read_regular_nofollow(tree, rel)
    return None if data is None else data.decode("utf-8", errors="replace")


def resolve_inside(tree: PaperTree, rel: str, *, max_hops: int = 40) -> str | None:
    """Resolve symlinks textually and return the final relative path inside the tree.

    Returns ``None`` when a link escapes the root, loops, or dangles. A path without links
    resolves to itself.
    """
    pending = list(PurePosixPath(clean_relative(rel)).parts)
    resolved: list[str] = []
    hops = 0
    while pending:
        part = pending.pop(0)
        candidate = "/".join([*resolved, part])
        if tree.kind(candidate) == "symlink":
            hops += 1
            if hops > max_hops:
                return None
            target = tree.readlink(candidate)
            if target is None:
                return None
            joined = posixpath.normpath(posixpath.join("/".join(resolved), target))
            if joined.startswith("/") or joined == ".." or joined.startswith("../"):
                return None
            pending = list(PurePosixPath(joined).parts) + pending
            resolved = []
            continue
        resolved.append(part)
    return "/".join(resolved)


class DirectoryTree:
    """A :class:`PaperTree` over a host directory."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _path(self, rel: str) -> Path:
        return self.root / clean_relative(rel)

    def kind(self, rel: str) -> EntryKind | None:
        """Return the entry kind (``lstat``)."""
        try:
            mode = os.lstat(self._path(rel)).st_mode
        except OSError:
            return None
        if stat.S_ISLNK(mode):
            return "symlink"
        if stat.S_ISDIR(mode):
            return "dir"
        if stat.S_ISREG(mode):
            return "file"
        return "other"

    def read(self, rel: str, limit: int | None = None) -> bytes | None:
        """Read a regular file, or ``None``."""
        if self.kind(rel) != "file":
            return None
        try:
            with self._path(rel).open("rb") as handle:
                return handle.read() if limit is None else handle.read(limit)
        except OSError:
            return None

    def readlink(self, rel: str) -> str | None:
        """Return the target of a symlink."""
        try:
            return os.readlink(self._path(rel))
        except OSError:
            return None

    def files(self) -> list[str]:
        """Return every regular file below the root (links are not followed)."""
        found: list[str] = []
        for current, dirs, names in os.walk(self.root, followlinks=False):
            dirs.sort()
            for name in sorted(names):
                path = Path(current) / name
                if path.is_symlink() or not path.is_file():
                    continue
                found.append(path.relative_to(self.root).as_posix())
        return sorted(found)

    def sha256(self, rel: str) -> str | None:
        """Return the digest of a regular file."""
        import hashlib

        data = self.read(rel)
        return None if data is None else "sha256:" + hashlib.sha256(data).hexdigest()


class SnapshotTree:
    """A :class:`PaperTree` filled from a workspace listing and selected file contents."""

    def __init__(
        self, entries: Mapping[str, FileEntry], contents: Mapping[str, bytes] | None = None
    ) -> None:
        self._entries = dict(entries)
        self._contents = dict(contents or {})

    @classmethod
    async def load(
        cls,
        workspace: Workspace,
        root: str,
        *,
        include: Callable[[str, FileEntry], int | None],
    ) -> SnapshotTree:
        """List ``root`` in ``workspace`` and read the files that ``include`` selects.

        Args:
            workspace: The workspace holding the paper directory.
            root: Workspace-relative path of the paper directory.
            include: Called with each regular file's relative path and entry; returns the
                number of bytes to keep (``0`` to keep nothing but the listing, a large number
                for the whole file), or ``None`` to skip the file.
        """
        listing = await workspace.list_tree(root)
        entries = {entry.path: entry for entry in listing}
        contents: dict[str, bytes] = {}
        for rel, entry in entries.items():
            if entry.kind != "file":
                continue
            keep = include(rel, entry)
            if not keep:
                continue
            data = await workspace.read_bytes(f"{root.rstrip('/')}/{rel}" if root else rel)
            contents[rel] = data[:keep]
        return cls(entries, contents)

    def kind(self, rel: str) -> EntryKind | None:
        """Return the entry kind from the listing."""
        normal = clean_relative(rel)
        if normal == ".":
            return "dir"
        entry = self._entries.get(normal)
        return None if entry is None else entry.kind

    def read(self, rel: str, limit: int | None = None) -> bytes | None:
        """Return a loaded regular file."""
        normal = clean_relative(rel)
        entry = self._entries.get(normal)
        if entry is None or entry.kind != "file":
            return None
        data = self._contents.get(normal)
        if data is None:
            return None
        return data if limit is None else data[:limit]

    def readlink(self, rel: str) -> str | None:
        """Return the recorded target of a symlink."""
        entry = self._entries.get(clean_relative(rel))
        return None if entry is None or entry.kind != "symlink" else entry.target

    def files(self) -> list[str]:
        """Return every regular file of the listing."""
        return sorted(rel for rel, entry in self._entries.items() if entry.kind == "file")

    def sha256(self, rel: str) -> str | None:
        """Return the digest of the listing entry of a regular file."""
        entry = self._entries.get(clean_relative(rel))
        return None if entry is None or entry.kind != "file" else entry.sha256

    def size(self, rel: str) -> int | None:
        """Return the size of a regular file."""
        entry = self._entries.get(clean_relative(rel))
        return None if entry is None or entry.kind != "file" else entry.size

    def with_file(self, rel: str, data: bytes) -> SnapshotTree:
        """Return a copy with one more regular file (used for compile outputs)."""
        normal = clean_relative(rel)
        entries = dict(self._entries)
        entries[normal] = FileEntry(path=normal, size=len(data), kind="file")
        parent = PurePosixPath(normal).parent
        while parent.as_posix() != ".":
            entries.setdefault(parent.as_posix(), FileEntry(path=parent.as_posix(), kind="dir"))
            parent = parent.parent
        contents = dict(self._contents)
        contents[normal] = data
        return SnapshotTree(entries, contents)


__all__ = [
    "DirectoryTree",
    "PaperTree",
    "SnapshotTree",
    "as_tree",
    "clean_relative",
    "is_regular_nofollow",
    "lexists",
    "read_regular_nofollow",
    "read_text",
    "resolve_inside",
]
