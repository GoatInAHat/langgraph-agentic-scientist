"""Safe workspace paths and upstream's ``.agentic_task_ignore`` rules.

:func:`safe_relpath` is the one place that decides whether a path may be used inside a
workspace: POSIX, relative, no ``..`` component, no NUL byte. The task-copy ignore rules are a
port of upstream ``src/investigation/files.py`` (``task_copy_path_is_ignored`` and its helpers),
including the anchored ``/pattern`` and ``**`` segment semantics.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from pathlib import PurePosixPath

from ..core.errors import WorkspacePathError

TASK_COPY_IGNORE_FILENAME = ".agentic_task_ignore"


def safe_relpath(path: str, *, allow_root: bool = True) -> str:
    """Normalise a workspace path and refuse unsafe ones.

    Leading ``./`` and a single leading ``/`` (the virtual root of an agent backend) are
    accepted and removed. The result is ``""`` for the root.

    Raises:
        WorkspacePathError: For ``..`` components, backslashes, NUL bytes, ``~`` or a root path
            when ``allow_root`` is false.
    """
    if not isinstance(path, str):
        raise WorkspacePathError(f"workspace path must be a string: {path!r}")
    if "\x00" in path or "\\" in path:
        raise WorkspacePathError(f"unsafe workspace path: {path!r}")
    raw = path.strip()
    if raw.startswith("~"):
        raise WorkspacePathError(f"unsafe workspace path: {path!r}")
    while raw.startswith("/"):
        raw = raw[1:]
    parts: list[str] = []
    for part in PurePosixPath(raw).parts if raw else ():
        if part in ("", "."):
            continue
        if part == "..":
            raise WorkspacePathError(f"workspace path must not contain '..': {path!r}")
        parts.append(part)
    normalized = "/".join(parts)
    if not normalized and not allow_root:
        raise WorkspacePathError(f"workspace path must not be the root: {path!r}")
    return normalized


def join(*parts: str) -> str:
    """Join workspace path parts and normalise the result with :func:`safe_relpath`."""
    return safe_relpath("/".join(part.strip("/") for part in parts if part not in ("", ".")))


def relative_to(path: str, base: str) -> str:
    """Return ``path`` relative to ``base`` (both workspace paths), or raise ``ValueError``."""
    path = safe_relpath(path)
    base = safe_relpath(base)
    if not base:
        return path
    if path == base:
        return ""
    prefix = base + "/"
    if not path.startswith(prefix):
        raise ValueError(f"{path!r} is not inside {base!r}")
    return path[len(prefix) :]


def is_within(path: str, base: str) -> bool:
    """Return whether workspace path ``path`` is ``base`` or below it."""
    try:
        relative_to(path, base)
    except ValueError:
        return False
    return True


def parse_ignore_patterns(text: str) -> tuple[str, ...]:
    """Parse the lines of an ``.agentic_task_ignore`` file (comments and blanks skipped)."""
    patterns: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return tuple(patterns)


def task_copy_path_is_ignored(relative_path: str, *, is_dir: bool, patterns: Sequence[str]) -> bool:
    """Match a task-copy ignore pattern; a leading slash anchors it to the root (upstream rule)."""
    rel_path = PurePosixPath(relative_path).as_posix()
    if rel_path.startswith("./"):
        rel_path = rel_path[2:]
    name = PurePosixPath(rel_path).name
    for pattern in patterns:
        anchored = pattern.startswith("/")
        directory_only = pattern.endswith("/")
        normalized = pattern.strip("/") if anchored else pattern.rstrip("/")
        if directory_only and not is_dir:
            continue
        if anchored:
            if _root_ignore_pattern_matches(rel_path, normalized):
                return True
            continue
        if (
            fnmatch.fnmatch(name, normalized)
            or fnmatch.fnmatch(rel_path, normalized)
            or rel_path.startswith(f"{normalized}/")
        ):
            return True
    return False


def _root_ignore_pattern_matches(relative_path: str, pattern: str) -> bool:
    path_parts = PurePosixPath(relative_path.lstrip("/")).parts
    raw_pattern_parts = PurePosixPath(pattern).parts
    if not raw_pattern_parts:
        return False
    pattern_parts = tuple(
        part
        for index, part in enumerate(raw_pattern_parts)
        if part != "**" or index == 0 or raw_pattern_parts[index - 1] != "**"
    )
    return _path_segments_match(path_parts, pattern_parts)


def _path_segments_match(path_parts: tuple[str, ...], pattern_parts: tuple[str, ...]) -> bool:
    pending = [(0, 0)]
    visited: set[tuple[int, int]] = set()
    while pending:
        path_index, pattern_index = pending.pop()
        state = (path_index, pattern_index)
        if state in visited:
            continue
        visited.add(state)
        if pattern_index == len(pattern_parts):
            return True
        pattern_part = pattern_parts[pattern_index]
        if pattern_part == "**":
            pending.append((path_index, pattern_index + 1))
            if path_index < len(path_parts):
                pending.append((path_index + 1, pattern_index))
        elif path_index < len(path_parts) and fnmatch.fnmatchcase(
            path_parts[path_index], pattern_part
        ):
            pending.append((path_index + 1, pattern_index + 1))
    return False


__all__ = [
    "TASK_COPY_IGNORE_FILENAME",
    "is_within",
    "join",
    "parse_ignore_patterns",
    "relative_to",
    "safe_relpath",
    "task_copy_path_is_ignored",
]
