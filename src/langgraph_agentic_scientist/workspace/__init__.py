"""Workspaces: where a run keeps its files, and the agents' backends over them."""

from .base import WorkspaceBase
from .digest import sha256_bytes, tree_digest
from .local import LocalWorkspace
from .paths import (
    TASK_COPY_IGNORE_FILENAME,
    is_within,
    join,
    parse_ignore_patterns,
    relative_to,
    safe_relpath,
    task_copy_path_is_ignored,
)
from .protocol import (
    CommandResult,
    FileEntry,
    SnapshotRef,
    Workspace,
    is_dir,
    is_file,
    is_symlink,
    read_text,
    write_text,
)
from .readonly import ReadOnlyBackend

__all__ = [
    "TASK_COPY_IGNORE_FILENAME",
    "CommandResult",
    "FileEntry",
    "LocalWorkspace",
    "ReadOnlyBackend",
    "SnapshotRef",
    "Workspace",
    "WorkspaceBase",
    "is_dir",
    "is_file",
    "is_symlink",
    "is_within",
    "join",
    "parse_ignore_patterns",
    "read_text",
    "relative_to",
    "safe_relpath",
    "sha256_bytes",
    "task_copy_path_is_ignored",
    "tree_digest",
    "write_text",
]
