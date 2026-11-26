"""Small file operations of the variants phase over the :class:`Workspace` protocol.

Upstream's variants pipeline works on a directory tree with ``pathlib`` and ``shutil``; the port
works through a :class:`~langgraph_agentic_scientist.Workspace`, so that a caller can keep the run
in a container, a worktree or a remote sandbox. Everything here uses the protocol's own methods;
the one command it runs is ``ls``, to list the children of a directory without reading the tree
below them (a workspace without ``ls`` falls back to ``list_tree``).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from ..workspace.paths import join
from ..workspace.protocol import FileEntry, Workspace, is_dir, is_file, read_text, write_text
from .constants import (
    TASK_COPY_EXCLUDE_DIRS,
    TASK_COPY_EXCLUDE_FILES,
    TASK_COPY_EXCLUDE_SUFFIXES,
    TASK_ROOT_EXCLUDE_FILES,
)

PROGRAM_DIR_RE = re.compile(r"^program_(\d+)$")


async def read_text_or_none(workspace: Workspace, path: str) -> str | None:
    """Read a regular UTF-8 file; ``None`` when it is absent, not a file or not UTF-8."""
    if not await is_file(workspace, path):
        return None
    try:
        return await read_text(workspace, path)
    except (OSError, UnicodeError):
        return None


async def read_json_or_none(workspace: Workspace, path: str) -> Any | None:
    """Read a regular JSON file; ``None`` when it is absent or malformed."""
    text = await read_text_or_none(workspace, path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


async def read_json_object(workspace: Workspace, path: str) -> dict[str, Any] | None:
    """Read a JSON file that holds an object; ``None`` otherwise."""
    value = await read_json_or_none(workspace, path)
    return value if isinstance(value, dict) else None


async def write_json(
    workspace: Workspace,
    path: str,
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    trailing_newline: bool = False,
) -> None:
    """Write ``value`` as JSON (``json.dumps(value, indent=...)`` plus an optional newline)."""
    text = json.dumps(value, indent=indent, sort_keys=sort_keys)
    await write_text(workspace, path, text + ("\n" if trailing_newline else ""))


async def make_dir(workspace: Workspace, path: str) -> None:
    """Create a directory (the protocol creates parents with files, so write and remove one)."""
    if await is_dir(workspace, path):
        return
    keep = join(path, ".keep")
    await workspace.write_bytes(keep, b"")
    await workspace.remove(keep)


async def list_children(workspace: Workspace, path: str) -> list[tuple[str, bool]]:
    """Return ``(name, is_directory)`` for each direct child of a directory, sorted by name.

    Uses ``ls`` through :meth:`Workspace.run`; when that is unavailable the listing comes from
    :meth:`Workspace.list_tree`, which reads the whole tree below ``path``.
    """
    if not await is_dir(workspace, path):
        return []
    try:
        result = await workspace.run(["ls", "-1A", "-p", "."], cwd=path, timeout=60)
    except Exception:
        result = None
    if result is not None and result.returncode == 0 and not result.truncated:
        children: list[tuple[str, bool]] = []
        for line in result.stdout.splitlines():
            if not line:
                continue
            directory = line.endswith("/")
            children.append((line[:-1] if directory else line, directory))
        return sorted(children)
    entries = await workspace.list_tree(path)
    return sorted((entry.path, entry.kind == "dir") for entry in entries if "/" not in entry.path)


async def list_child_dirs(workspace: Workspace, path: str) -> list[str]:
    """Return the names of the direct child directories of a directory, sorted."""
    return [name for name, directory in await list_children(workspace, path) if directory]


async def program_indices(workspace: Workspace, bundle: str, *, upto: int) -> list[int]:
    """Return the indices ``n`` of the ``program_<n>`` directories of a bundle, ascending.

    Directories are probed by name from 0 up to ``upto + 1`` (a bundle holds at most one program
    per allowed attempt) and on while consecutive indices exist.
    """
    found: list[int] = []
    index = 0
    while True:
        exists = await is_dir(workspace, join(bundle, f"program_{index}"))
        if exists:
            found.append(index)
        elif index > upto:
            break
        index += 1
    return found


def anchored(pattern: str) -> str:
    """Anchor an ``.agentic_task_ignore`` pattern to the copied root (a leading slash)."""
    return pattern if pattern.startswith("/") else "/" + pattern


def variant_copy_patterns(user_patterns: Iterable[str]) -> list[str]:
    """Return the ignore patterns of upstream's variant-workspace task copy.

    Upstream's ``materialize._copy_task_files`` skips, among the top-level entries, program
    directories (``program_*`` and ``reference_program``), the directories of
    :data:`TASK_COPY_EXCLUDE_DIRS`, files of a previous run, runtime markers and log files, and
    whatever the task's ``.agentic_task_ignore`` names. The patterns are anchored to the root.
    """
    patterns = ["/program_*", "/reference_program"]
    patterns += [f"/{name}/" for name in sorted(TASK_COPY_EXCLUDE_DIRS)]
    patterns += [f"/{name}" for name in sorted(TASK_COPY_EXCLUDE_FILES | TASK_ROOT_EXCLUDE_FILES)]
    patterns += [f"/*{suffix}" for suffix in TASK_COPY_EXCLUDE_SUFFIXES]
    patterns += [anchored(pattern) for pattern in user_patterns]
    return patterns


def extract_copy_patterns(user_patterns: Iterable[str]) -> list[str]:
    """Return the ignore patterns of upstream's extraction task copy (any depth).

    Upstream's ``best_program_extraction._copy_task_files`` skips program directories
    (``program_<n>``), the directories of :data:`TASK_COPY_EXCLUDE_DIRS`, the root runtime
    marker and whatever ``.agentic_task_ignore`` names, at every depth.
    """
    patterns = ["program_[0-9]*/"]
    patterns += [f"{name}/" for name in sorted(TASK_COPY_EXCLUDE_DIRS)]
    patterns += [f"/{name}" for name in sorted(TASK_ROOT_EXCLUDE_FILES)]
    patterns += list(user_patterns)
    return patterns


async def copy_tree_replacing(
    workspace: Workspace, src: str, dst: str, *, ignore: Iterable[str] = ()
) -> None:
    """Copy a directory tree to ``dst``, replacing what is there (upstream ``rmtree`` + copy)."""
    await workspace.remove(dst)
    await workspace.copy_tree(src, dst, ignore=list(ignore))


async def copy_file(workspace: Workspace, src: str, dst: str) -> None:
    """Copy one regular file."""
    await workspace.write_bytes(dst, await workspace.read_bytes(src))


async def regular_entries(workspace: Workspace, path: str) -> list[FileEntry]:
    """List the entries below ``path`` and refuse links and special files."""
    entries = await workspace.list_tree(path)
    for entry in entries:
        if entry.kind not in ("file", "dir"):
            raise ValueError(f"{path} contains a link or special file: {entry.path}")
    return entries


async def remove_all(workspace: Workspace, paths: Iterable[str]) -> None:
    """Remove each path (a missing path is not an error)."""
    for path in paths:
        await workspace.remove(path)


__all__ = [
    "PROGRAM_DIR_RE",
    "anchored",
    "copy_file",
    "copy_tree_replacing",
    "extract_copy_patterns",
    "list_child_dirs",
    "list_children",
    "make_dir",
    "program_indices",
    "read_json_object",
    "read_json_or_none",
    "read_text_or_none",
    "regular_entries",
    "remove_all",
    "stage_active_files",
    "variant_copy_patterns",
    "write_json",
]


async def stage_active_files(
    workspace: Workspace,
    root: str,
    *,
    rubrics_from: str | None = None,
    idea_from: str | None = None,
) -> None:
    """Stage ``active_rubrics.json`` and ``active_idea.json`` into an agent's root.

    Upstream's ``variant_agent.sh`` writes both before every session: the rubrics file is a copy
    of the variant's ``target_rubrics.json`` (``[]`` when none is given); the idea file is a copy
    of the variant's ``variant_spec.json`` (``{}`` when none is given and none exists yet).
    """
    rubrics_text = await read_text_or_none(workspace, rubrics_from) if rubrics_from else None
    await write_text(workspace, join(root, "active_rubrics.json"), rubrics_text or "[]\n")
    idea_text = await read_text_or_none(workspace, idea_from) if idea_from else None
    if idea_text is not None:
        await write_text(workspace, join(root, "active_idea.json"), idea_text)
    elif await workspace.stat(join(root, "active_idea.json")) is None:
        await write_text(workspace, join(root, "active_idea.json"), "{}\n")
