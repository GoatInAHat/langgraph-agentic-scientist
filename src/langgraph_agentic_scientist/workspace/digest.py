"""Digests of files and trees in upstream's byte layout.

The tree digest is upstream ``final_evaluation/runner.py`` ``_tree_digest``: entries sorted by
their relative POSIX path; a directory contributes ``b"dir\\0" + path + b"\\0"``; a file
contributes ``b"file\\0" + path + b"\\0" + size + b"\\0" + content``. Upstream refuses symlinks in a
digest tree; the port hashes them as ``b"link\\0" + path + b"\\0" + target`` so that a program
with a symlink can still be frozen, and gives the same digest as upstream for any tree without
one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Iterable

from .protocol import FileEntry

SHA256_PREFIX = "sha256:"


def sha256_bytes(data: bytes) -> str:
    """Return ``sha256:<hex>`` of ``data``."""
    return SHA256_PREFIX + hashlib.sha256(data).hexdigest()


def sha256_hex(data: bytes) -> str:
    """Return the bare hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


async def tree_digest(
    entries: Iterable[FileEntry],
    read: Callable[[str], Awaitable[bytes]],
) -> str:
    """Compute upstream's tree digest from a listing and a reader of relative paths."""
    hasher = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.path):
        relative = entry.path.encode()
        if entry.kind == "dir":
            hasher.update(b"dir\0" + relative + b"\0")
        elif entry.kind == "symlink":
            hasher.update(b"link\0" + relative + b"\0" + (entry.target or "").encode())
        elif entry.kind == "file":
            content = await read(entry.path)
            hasher.update(b"file\0" + relative + b"\0" + str(len(content)).encode() + b"\0")
            hasher.update(content)
        else:
            raise ValueError(f"unsupported digest tree entry: {entry.path}")
    return SHA256_PREFIX + hasher.hexdigest()


__all__ = ["SHA256_PREFIX", "sha256_bytes", "sha256_hex", "tree_digest"]
