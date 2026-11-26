"""A workspace written against WorkspaceBase implements only the core and gets the rest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from langgraph_agentic_scientist.workspace import (
    CommandResult,
    FileEntry,
    LocalWorkspace,
    Workspace,
    WorkspaceBase,
    safe_relpath,
    sha256_bytes,
)


class DictWorkspace(WorkspaceBase):
    """The core operations over a dict of files (directories are implied by paths)."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    def agent_backend(self, root: str, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def read_bytes(self, path: str) -> bytes:
        try:
            return self.files[safe_relpath(path)]
        except KeyError:
            raise FileNotFoundError(path) from None

    async def write_bytes(self, path: str, data: bytes, *, exclusive: bool = False) -> None:
        key = safe_relpath(path)
        if exclusive and key in self.files:
            raise FileExistsError(path)
        self.files[key] = data

    def _below(self, key: str) -> list[str]:
        prefix = f"{key}/" if key else ""
        return [name for name in self.files if name.startswith(prefix)]

    async def stat(self, path: str) -> FileEntry | None:
        key = safe_relpath(path)
        if key in self.files:
            return FileEntry(path=key, size=len(self.files[key]))
        return FileEntry(path=key, kind="dir") if self._below(key) else None

    async def list_tree(self, path: str, *, with_digest: bool = False) -> list[FileEntry]:
        key = safe_relpath(path)
        entries: dict[str, FileEntry] = {}
        for name in self._below(key):
            relative = name[len(key) :].lstrip("/") if key else name
            parts = relative.split("/")
            for depth in range(1, len(parts)):
                entries["/".join(parts[:depth])] = FileEntry(
                    path="/".join(parts[:depth]), kind="dir"
                )
            data = self.files[name]
            entries[relative] = FileEntry(
                path=relative, size=len(data), sha256=sha256_bytes(data) if with_digest else None
            )
        return [entries[name] for name in sorted(entries)]

    async def remove(self, path: str) -> None:
        key = safe_relpath(path)
        self.files.pop(key, None)
        for name in self._below(key):
            del self.files[name]

    async def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        max_output_bytes: int = 65536,
    ) -> CommandResult:
        return CommandResult(argv=list(argv), returncode=0)


async def test_the_derived_operations_work_over_the_core(tmp_path: Any) -> None:
    ws = DictWorkspace()
    assert isinstance(ws, Workspace)
    await ws.write_bytes("a/x.txt", b"x")
    await ws.write_bytes("a/b/y.txt", b"y")
    await ws.write_bytes("a/skip/z.txt", b"z")
    assert await ws.exists("a/b") and not await ws.exists("nope")

    local = LocalWorkspace(tmp_path)
    for name, data in ws.files.items():
        await local.write_bytes(name, data)
    assert await ws.digest("a") == await local.digest("a")  # upstream's tree digest
    assert await ws.digest("a/x.txt") == sha256_bytes(b"x")

    await ws.copy_tree("a", "c", ignore=("skip/",))
    assert sorted(name for name in ws.files if name.startswith("c/")) == ["c/b/y.txt", "c/x.txt"]
    with pytest.raises(FileExistsError):
        await ws.copy_tree("a", "c")

    ref = await ws.snapshot("a")
    await ws.write_bytes("a/x.txt", b"changed")
    await ws.write_bytes("a/new.txt", b"n")
    await ws.restore(ref)
    assert ws.files["a/x.txt"] == b"x" and "a/new.txt" not in ws.files

    missing = await ws.snapshot("later")
    await ws.write_bytes("later/f", b"f")
    await ws.restore(missing)
    assert not await ws.exists("later")


async def test_list_tree_hashes_only_when_asked(tmp_path: Any) -> None:
    local = LocalWorkspace(tmp_path)
    await local.write_bytes("d/f.txt", b"data")
    (plain,) = [e for e in await local.list_tree("d") if e.kind == "file"]
    (hashed,) = [e for e in await local.list_tree("d", with_digest=True) if e.kind == "file"]
    assert plain.sha256 is None and hashed.sha256 == sha256_bytes(b"data")
