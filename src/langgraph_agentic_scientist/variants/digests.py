"""Digests of evaluator trees, strict program trees and bundles (upstream's sealing checks).

Upstream seals the evaluator of the task: it digests the evaluator source tree before the variants
run and after every evaluation (``evaluator_tree.managed_evaluator_tree_digest``), and it digests
the program and bundle trees of a brokered candidate that the retrofit copies unchanged
(``retrofit_preservation._strict_tree_digest``). These are the same digests, computed through a
:class:`~langgraph_agentic_scientist.Workspace` listing.

The evaluator-tree record layout is upstream's: a ``D`` record per directory, a masked ``I`` record
for ``.git``, ``.venv`` and ``results`` at the root, an ``F`` record per file (path and the SHA-256
of the content), ``__pycache__`` directories omitted, links and special files refused.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection

from ..workspace.digest import sha256_bytes
from ..workspace.paths import join
from ..workspace.protocol import FileEntry, Workspace
from .workspace_ops import list_children

MAX_EVALUATOR_TREE_BYTES = 512 * 1024 * 1024
MAX_EVALUATOR_TREE_ENTRIES = 100_000
MAX_TREE_BYTES = 512 * 1024 * 1024
MAX_TREE_ENTRIES = 100_000
MAX_TREE_DEPTH = 256

EVALUATOR_TREE_MASKED_DIRS = frozenset({".git", ".venv", "results"})
EVALUATOR_TREE_OMITTED_DIRS = frozenset({"__pycache__"})
MANAGED_RUNTIME_OMITTED_ROOT_ENTRIES = frozenset(
    {
        ".venv",
        ".agentic_runtime.json",
        ".agentic_resource_manifest.json",
        ".agentic_task_ignore",
        ".agentic_task_publication.json",
        ".agentic_task_qualification.json",
        ".agentic_task_snapshot.json",
        ".agentic_task_validation.json",
        "resources",
    }
)
"""Root entries that the managed evaluator digest leaves out (task-local runtime artifacts)."""


def _hex(sha: str | None) -> bytes:
    if sha is None or not sha.startswith("sha256:"):
        raise ValueError("a file entry has no SHA-256")
    return bytes.fromhex(sha[len("sha256:") :])


async def evaluator_tree_digest(
    workspace: Workspace,
    root: str,
    *,
    omitted_root_entries: Collection[str] = (),
) -> str:
    """Hash the bounded evaluator files below ``root`` (upstream ``evaluator_tree_digest``).

    Args:
        workspace: The workspace holding the tree.
        root: The task directory.
        omitted_root_entries: Root entries to leave out; pass
            :data:`MANAGED_RUNTIME_OMITTED_ROOT_ENTRIES` for upstream's *managed* digest, which
            ignores task-local runtime artifacts.

    Raises:
        ValueError: When the tree contains a link or special file, or exceeds a bound.
    """
    omitted = frozenset(omitted_root_entries)
    # Only the root's children are listed one by one: an omitted or masked entry is never read
    # (upstream skips it before it hashes anything), and the rest is listed as a subtree.
    entries: list[FileEntry] = []
    for name, _ in await list_children(workspace, root):
        if name in omitted:
            continue
        top = await workspace.stat(join(root, name))
        if top is None:
            continue
        if top.kind == "file":
            top = top.model_copy(
                update={"sha256": sha256_bytes(await workspace.read_bytes(join(root, name)))}
            )
        entries.append(top.model_copy(update={"path": name}))
        if top.kind == "dir" and name not in EVALUATOR_TREE_MASKED_DIRS:
            entries.extend(
                sub.model_copy(update={"path": f"{name}/{sub.path}"})
                for sub in await workspace.list_tree(join(root, name), with_digest=True)
            )
    children: dict[str, list[FileEntry]] = {}
    for entry in entries:
        parent, _, _ = entry.path.rpartition("/")
        children.setdefault(parent, []).append(entry)
    digest = hashlib.sha256()
    state = {"count": 0, "bytes": 0}

    def record(kind: bytes, relative: str, payload: bytes) -> None:
        path_bytes = relative.encode()
        digest.update(kind)
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(prefix: str) -> None:
        record(b"D", prefix, b"")
        for entry in sorted(
            children.get(prefix, []), key=lambda item: item.path.rsplit("/", 1)[-1]
        ):
            name = entry.path.rsplit("/", 1)[-1]
            if not prefix and name in omitted:
                continue
            if name in EVALUATOR_TREE_OMITTED_DIRS and entry.kind == "dir":
                continue
            if not prefix and name in EVALUATOR_TREE_MASKED_DIRS and entry.kind == "dir":
                record(b"I", name, b"masked")
                continue
            state["count"] += 1
            if state["count"] > MAX_EVALUATOR_TREE_ENTRIES:
                raise ValueError("evaluator source tree exceeds the entry limit")
            if entry.kind == "symlink":
                raise ValueError(f"evaluator source tree contains a symlink: {entry.path}")
            if entry.kind == "dir":
                visit(entry.path)
                continue
            if entry.kind != "file":
                raise ValueError(f"evaluator source tree contains a special file: {entry.path}")
            state["bytes"] += entry.size
            if state["bytes"] > MAX_EVALUATOR_TREE_BYTES:
                raise ValueError("evaluator source tree exceeds the byte limit")
            record(b"F", entry.path, _hex(entry.sha256))

    visit("")
    return digest.hexdigest()


async def managed_evaluator_tree_digest(workspace: Workspace, root: str) -> str:
    """Hash evaluator inputs without task-local managed-runtime artifacts (upstream's digest)."""
    return await evaluator_tree_digest(
        workspace, root, omitted_root_entries=MANAGED_RUNTIME_OMITTED_ROOT_ENTRIES
    )


async def strict_tree_digest(
    workspace: Workspace,
    root: str,
    *,
    omitted_root_entries: Collection[str] = (),
) -> str:
    """Hash every file and directory below ``root``, refusing links and special files.

    Port of upstream ``_strict_tree_digest`` (the bounds are the same: 100,000 entries, 512 MiB,
    depth 256); the record layout differs only in using the listing's file digests.
    """
    omitted = frozenset(omitted_root_entries)
    entries = await workspace.list_tree(root, with_digest=True)
    digest = hashlib.sha256()

    def record(kind: bytes, relative: str, payload: bytes = b"") -> None:
        for value in (kind, relative.encode(), payload):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

    record(b"D", "")
    count = 0
    total = 0
    for entry in sorted(entries, key=lambda item: item.path):
        first = entry.path.split("/", 1)[0]
        if first in omitted:
            continue
        count += 1
        if count > MAX_TREE_ENTRIES:
            raise ValueError("preserved bundle exceeds entry limit")
        if entry.path.count("/") + 1 > MAX_TREE_DEPTH:
            raise ValueError("preserved bundle exceeds directory depth limit")
        if entry.kind == "dir":
            record(b"D", entry.path)
        elif entry.kind == "file":
            total += entry.size
            if total > MAX_TREE_BYTES:
                raise ValueError("preserved bundle exceeds byte limit")
            record(b"F", entry.path, _hex(entry.sha256))
        else:
            raise ValueError(f"preserved bundle contains a link or special file: {entry.path}")
    return digest.hexdigest()


async def tree_fingerprint(
    workspace: Workspace, root: str, *, excluded: Collection[str] = ()
) -> str:
    """Return a deterministic SHA-256 of a tree, leaving out the given paths.

    Upstream's ``_fingerprint_tree`` (used for bundle and program fingerprints in the
    extraction manifest) records links as links; so does this one.
    """
    excluded_paths = frozenset(excluded)
    entries = await workspace.list_tree(root, with_digest=True)
    digest = hashlib.sha256()

    def update(kind: bytes, relative: str, payload: bytes = b"") -> None:
        for value in (kind, relative.encode("utf-8"), payload):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

    update(b"D", "")
    skipped: list[str] = []
    for entry in sorted(entries, key=lambda item: item.path):
        if entry.path in excluded_paths or any(entry.path.startswith(p + "/") for p in skipped):
            if entry.kind == "dir":
                skipped.append(entry.path)
            continue
        if entry.kind == "dir":
            update(b"D", entry.path)
        elif entry.kind == "symlink":
            update(b"L", entry.path, (entry.target or "").encode("utf-8"))
        elif entry.kind == "file":
            update(b"F", entry.path, _hex(entry.sha256))
        else:
            raise ValueError(f"unsupported entry in a fingerprinted tree: {entry.path}")
    return digest.hexdigest()


def file_sha256(data: bytes) -> str:
    """Return the bare hex SHA-256 of ``data``."""
    return sha256_bytes(data).removeprefix("sha256:")


__all__ = [
    "EVALUATOR_TREE_MASKED_DIRS",
    "MANAGED_RUNTIME_OMITTED_ROOT_ENTRIES",
    "evaluator_tree_digest",
    "file_sha256",
    "managed_evaluator_tree_digest",
    "strict_tree_digest",
    "tree_fingerprint",
]
