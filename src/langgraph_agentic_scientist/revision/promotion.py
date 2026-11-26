"""Publishing a validated revision over the published paper (upstream ``revision_publication.py``).

Upstream stages a copy of the published paper, snapshots the original into
``revisions/revision_01/before/``, overlays the revision's files, checks the result and exchanges
the directories atomically, with a journal for recovery. The port has no atomic exchange in the
workspace protocol; it makes the same change in place and puts the original back with the
workspace's snapshot and restore when anything fails, so the original paper is retained exactly
as upstream retains it. The checkpoint, not a journal, carries a resume: the snapshot reference
is state, and :func:`promote_revision` starts by restoring it when an earlier attempt left partial
work behind.

What is published, in upstream's order:

1. the original's files (``paper.tex``, ``paper.pdf``, ``paper_data.json``, the grounding ledger,
   the references, the audits, ...), its manifest as ``original_manifest.json``, the initial review
   and its figures go to ``revisions/revision_01/before/`` together with
   ``snapshot_complete.json``; the snapshot is created once and an existing one is an error;
2. the original's audited inputs that the revision's audit does not cover go to
   ``before/ancillary/`` with their relative paths and leave the paper, so that they are not part
   of the revised paper's audit inventory;
3. the revision's files and figures replace the paper's, and the inputs its audit binds are copied
   over;
4. the manifest of the revision, with its paths moved to the paper directory, becomes
   ``latest_manifest.json`` with a ``revision`` entry (``status`` ``promoted``) and the
   ``before_revision`` entry;
5. the published paper is checked against the snapshot as upstream checks it.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from ..core.errors import WorkspacePathError
from ..workspace import (
    SnapshotRef,
    Workspace,
    is_dir,
    is_file,
    is_within,
    join,
    relative_to,
    safe_relpath,
)
from .evidence import (
    AuditHashes,
    EvidenceError,
    ScopeAudit,
    published_audit_hashes,
    read_json_object,
    report_ok,
    validate_revision_evidence,
)

MAX_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_REVIEW_BYTES = 1024 * 1024
MAX_COPY_ENTRIES = 100_000
MAX_COPY_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_COPY_DEPTH = 64
MAX_BOUND_PATHS = 4096
MAX_BOUND_FILE_BYTES = 100 * 1024 * 1024

REQUIRED_REVISION_FILES = (
    "paper.tex",
    "paper.pdf",
    "claims_grounding.json",
    "references.bib",
    "references_provenance.json",
    "validation_report.json",
    "latest_manifest.json",
)
PROMOTED_FILES = (
    "paper.tex",
    "paper.pdf",
    "paper_data.json",
    "paper_prompt.md",
    "claims_grounding.json",
    "references.bib",
    "references_provenance.json",
    "figure_manifest.json",
    "validation_report.json",
    "validation_feedback.json",
    "validation_feedback.md",
    "paper_evidence_audit.json",
    "paper_evidence_audit.md",
    "paper_evidence_audit_feedback.json",
    "paper_evidence_audit_feedback.md",
    "method_scope_audit.json",
    "method_scope_audit.md",
)
SNAPSHOT_FILES = PROMOTED_FILES

REVISION_DIR_NAME = "revisions/revision_01"
BEFORE_DIR_NAME = "revisions/revision_01/before"
MANIFEST_NAME = "latest_manifest.json"
BEFORE_REVIEW_NAME = "ai_review.before_revision.json"
AFTER_REVIEW_NAME = "ai_review.after_revision.json"

_TREE_PREFIX = "evidence-tree/"
_FILE_PREFIXES = (
    "evidence-source/",
    "tex-source/",
    "compile-input/",
    "validator-input/",
    "figure-asset/",
)
_HEX64 = re.compile(r"[0-9a-f]{64}")


class PublicationError(ValueError):
    """A revision cannot be published, or a published revision is inconsistent."""


@dataclass
class CopyBudget:
    """Upstream's bounds on a publication copy: entries, bytes and depth."""

    entries: int = 0
    total_bytes: int = 0

    def add(self, *, size: int = 0) -> None:
        """Count one entry of ``size`` bytes; raise when a bound is exceeded."""
        self.entries += 1
        self.total_bytes += size
        if self.entries > MAX_COPY_ENTRIES:
            raise PublicationError("publication copy has too many entries")
        if self.total_bytes > MAX_COPY_TOTAL_BYTES:
            raise PublicationError("publication copy is too large")


@dataclass
class PromotionOutcome:
    """What :func:`promote_revision` did."""

    before_dir: str
    archived: list[str] = field(default_factory=list)
    skipped_checks: list[str] = field(default_factory=list)


# --- reading and copying ----------------------------------------------------------------------


async def read_regular(workspace: Workspace, path: str, *, max_bytes: int) -> bytes:
    """Read a regular file of at most ``max_bytes`` (upstream ``read_regular``).

    Raises:
        PublicationError: When the file is missing, not regular, or too large.
    """
    entry = await workspace.stat(path)
    if entry is None:
        raise PublicationError(f"missing regular file: {path}")
    if entry.kind != "file":
        raise PublicationError(f"expected regular file: {path}")
    if entry.size > max_bytes:
        raise PublicationError(f"file is too large: {path}")
    data = await workspace.read_bytes(path)
    if len(data) > max_bytes:
        raise PublicationError(f"file is too large: {path}")
    return data


async def read_json(workspace: Workspace, path: str, *, max_bytes: int) -> dict[str, Any]:
    """Read a regular JSON object file."""
    data = await read_regular(workspace, path, max_bytes=max_bytes)
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise PublicationError(f"invalid JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise PublicationError(f"expected JSON object: {path}")
    return payload


async def write_json(workspace: Workspace, path: str, payload: Mapping[str, Any]) -> None:
    """Write a JSON object with sorted keys and a two-space indent (upstream's manifest form)."""
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    await workspace.write_bytes(path, text.encode("utf-8"))


async def copy_regular(workspace: Workspace, source: str, target: str) -> int:
    """Copy one regular file; return its size."""
    data = await read_regular(workspace, source, max_bytes=MAX_ARTIFACT_BYTES)
    await workspace.write_bytes(target, data)
    return len(data)


async def copy_present_named_files(
    workspace: Workspace,
    source_root: str,
    target_root: str,
    names: Collection[str],
    budget: CopyBudget,
) -> list[str]:
    """Copy the named files that exist in ``source_root``; a name that is not regular is refused."""
    copied: list[str] = []
    for name in names:
        entry = await workspace.stat(join(source_root, name))
        if entry is None:
            continue
        if entry.kind != "file":
            raise PublicationError(f"expected regular file: {join(source_root, name)}")
        budget.add(
            size=await copy_regular(workspace, join(source_root, name), join(target_root, name))
        )
        copied.append(name)
    return copied


async def copy_directory_contents(
    workspace: Workspace, source_root: str, target_root: str, budget: CopyBudget
) -> list[str]:
    """Copy the files below ``source_root`` into ``target_root`` (merging into what is there).

    Symlinks and special files are refused, as upstream refuses them; the bounds on entries,
    bytes and depth apply.

    Returns:
        The relative paths of the copied files, sorted.
    """
    copied: list[str] = []
    for entry in await workspace.list_tree(source_root):
        parts = PurePosixPath(entry.path).parts
        if len(parts) > MAX_COPY_DEPTH:
            raise PublicationError("publication copy is too deep")
        source = join(source_root, entry.path)
        if entry.kind == "dir":
            budget.add()
        elif entry.kind == "file":
            budget.add(size=await copy_regular(workspace, source, join(target_root, entry.path)))
            copied.append(entry.path)
        elif entry.kind == "symlink":
            raise PublicationError(f"publication tree contains an unexpected symlink: {source}")
        else:
            raise PublicationError(f"publication tree contains a special file: {source}")
    return sorted(copied)


# --- paths in manifests -------------------------------------------------------------------------


def workspace_path(workspace: Workspace, value: str) -> str | None:
    """Return the workspace path that a manifest string names, or ``None`` if it names none.

    A manifest may hold workspace-relative paths (the package's form) or absolute host paths (as
    upstream's); both are understood.
    """
    root = workspace.host_path("").rstrip("/")
    candidate = value
    if value.startswith("/"):
        if value == root:
            return ""
        if not root or not value.startswith(root + "/"):
            return None
        candidate = value[len(root) + 1 :]
    try:
        return safe_relpath(candidate)
    except WorkspacePathError:
        return None


def names_path(workspace: Workspace, value: object, path: str) -> bool:
    """Whether the manifest string ``value`` names the workspace path ``path``."""
    return isinstance(value, str) and workspace_path(workspace, value) == safe_relpath(path)


def rebase_manifest(workspace: Workspace, value: object, *, old_root: str, new_root: str) -> object:
    """Move the paths under ``old_root`` to ``new_root`` in a manifest (upstream's rebase).

    Strings that name a path inside ``old_root`` (in either form, see :func:`workspace_path`) are
    rewritten in the form they had; a path that climbs out of the root is refused.
    """
    if isinstance(value, dict):
        return {
            key: rebase_manifest(workspace, item, old_root=old_root, new_root=new_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            rebase_manifest(workspace, item, old_root=old_root, new_root=new_root) for item in value
        ]
    if not isinstance(value, str) or not value:
        return value
    if ".." in PurePosixPath(value).parts:
        normalized = workspace_path(workspace, posixpath.normpath(value))
        if normalized is not None and is_within(normalized, old_root):
            raise PublicationError(f"manifest path escapes revision root: {value}")
        return value
    path = workspace_path(workspace, value)
    if path is None or not is_within(path, old_root):
        return value
    moved = join(new_root, relative_to(path, old_root))
    return workspace.host_path(moved) if value.startswith("/") else moved


# --- validating a revision workspace ------------------------------------------------------------


async def validate_revision_workspace(workspace: Workspace, revision_dir: str) -> dict[str, Any]:
    """Check that ``revision_dir`` is a complete, regular, validated revision; return its manifest.

    Upstream's ``_validate_revision_workspace``: the directory and its required and promoted
    files are regular, the figures tree has no symlink or special file, the manifest identifies
    the revised PDF, and the validation report passed.

    Raises:
        PublicationError: When any of that is not so.
    """
    entry = await workspace.stat(revision_dir)
    if entry is None or entry.kind != "dir":
        raise PublicationError(f"revision directory must be a regular directory: {revision_dir}")
    for name in REQUIRED_REVISION_FILES:
        await read_regular(workspace, join(revision_dir, name), max_bytes=MAX_ARTIFACT_BYTES)
    for name in PROMOTED_FILES:
        if await workspace.exists(join(revision_dir, name)):
            await read_regular(workspace, join(revision_dir, name), max_bytes=MAX_ARTIFACT_BYTES)
    figures = join(revision_dir, "figures")
    if await workspace.exists(figures):
        if not await is_dir(workspace, figures):
            raise PublicationError(f"expected regular directory: {figures}")
        for item in await workspace.list_tree(figures):
            if item.kind not in ("file", "dir"):
                raise PublicationError(f"figures tree contains an irregular entry: {item.path}")
    manifest = await read_json(
        workspace, join(revision_dir, MANIFEST_NAME), max_bytes=MAX_ARTIFACT_BYTES
    )
    if not names_path(workspace, manifest.get("paper_pdf"), join(revision_dir, "paper.pdf")):
        raise PublicationError("revision manifest does not identify the revised PDF")
    report = await read_json(
        workspace, join(revision_dir, "validation_report.json"), max_bytes=MAX_ARTIFACT_BYTES
    )
    if not report_ok(report):
        raise PublicationError("revision validation did not pass")
    return manifest


# --- audit-bound inputs -------------------------------------------------------------------------


def audit_bound_paths(input_sha256: object) -> tuple[list[str], list[str]]:
    """Return the files and trees that an audit's ``input_sha256`` binds (relative paths).

    Upstream's ``_audit_bound_paths``: keys under ``evidence-tree/`` are trees, other keys are
    files whose known prefixes (``evidence-source/``, ``tex-source/``, ...) are removed; paths
    must be safe, digests 64 lowercase hex digits, a path may not be both file and tree; nested
    trees are dropped and files inside a tree are covered by it.
    """
    if not isinstance(input_sha256, dict) or len(input_sha256) > MAX_BOUND_PATHS:
        raise PublicationError("paper evidence audit has invalid input_sha256 paths")
    files: set[PurePosixPath] = set()
    trees: set[PurePosixPath] = set()
    for key, digest in input_sha256.items():
        if not isinstance(key, str) or not isinstance(digest, str) or not _HEX64.fullmatch(digest):
            raise PublicationError("paper evidence audit has invalid input_sha256 entry")
        if key.startswith(_TREE_PREFIX):
            trees.add(_safe_bound_path(key.removeprefix(_TREE_PREFIX)))
            continue
        relative = key
        for prefix in _FILE_PREFIXES:
            if key.startswith(prefix):
                relative = key.removeprefix(prefix)
                break
        files.add(_safe_bound_path(relative))
    if files & trees:
        raise PublicationError("paper evidence audit binds a path as both file and tree")
    ordered_trees: list[PurePosixPath] = []
    for tree in sorted(trees, key=lambda path: (len(path.parts), path.as_posix())):
        if not any(tree.is_relative_to(parent) for parent in ordered_trees):
            ordered_trees.append(tree)
    ordered_files = sorted(
        (path for path in files if not any(path.is_relative_to(tree) for tree in ordered_trees)),
        key=lambda path: path.as_posix(),
    )
    return [path.as_posix() for path in ordered_files], [tree.as_posix() for tree in ordered_trees]


def _safe_bound_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if (
        not raw
        or "\\" in raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublicationError(f"unsafe audit-bound path: {raw!r}")
    return path


async def _copy_bound_file(
    workspace: Workspace, source: str, target: str, budget: CopyBudget
) -> None:
    entry = await workspace.stat(source)
    if entry is None or entry.kind != "file":
        raise PublicationError(f"audit-bound input is not a regular file: {source}")
    if entry.size > MAX_BOUND_FILE_BYTES:
        raise PublicationError(f"audit-bound input is too large: {source}")
    existing = await workspace.stat(target)
    if existing is not None and existing.kind != "file":
        raise PublicationError(f"audit-bound publication target is not regular: {target}")
    budget.add(size=await copy_regular(workspace, source, target))


async def copy_audit_bound_inputs(
    workspace: Workspace,
    source_root: str,
    target_root: str,
    input_sha256: object,
    budget: CopyBudget,
) -> None:
    """Copy every input that an audit binds from ``source_root`` to ``target_root``.

    Trees replace what is at their target; files are copied over (upstream's
    ``copy_audit_bound_inputs``).
    """
    files, trees = audit_bound_paths(input_sha256)
    for tree in trees:
        source = join(source_root, tree)
        entry = await workspace.stat(source)
        if entry is None or entry.kind != "dir":
            raise PublicationError(f"audit-bound tree is not a regular directory: {source}")
        target = join(target_root, tree)
        await workspace.remove(target)
        for item in await workspace.list_tree(source):
            if len(PurePosixPath(item.path).parts) > MAX_COPY_DEPTH:
                raise PublicationError("audit-bound publication tree is too deep")
            if item.kind == "dir":
                budget.add()
            elif item.kind == "file":
                await _copy_bound_file(
                    workspace, join(source, item.path), join(target, item.path), budget
                )
            else:
                raise PublicationError(f"audit-bound tree entry is not regular: {item.path}")
    for path in files:
        await _copy_bound_file(workspace, join(source_root, path), join(target_root, path), budget)


async def _tree_files(workspace: Workspace, root: str, tree: str, budget: CopyBudget) -> set[str]:
    found: set[str] = set()
    entry = await workspace.stat(join(root, tree))
    if entry is None:
        return found
    if entry.kind == "file":
        budget.add(size=entry.size)
        return {tree}
    if entry.kind != "dir":
        raise PublicationError(f"original evidence tree input is not regular: {tree}")
    budget.add()
    for item in await workspace.list_tree(join(root, tree)):
        if len(PurePosixPath(item.path).parts) > MAX_COPY_DEPTH:
            raise PublicationError("original evidence tree is too deep")
        budget.add(size=item.size)
        if item.kind == "file":
            found.add(posixpath.join(tree, item.path))
        elif item.kind != "dir":
            raise PublicationError(f"original evidence tree input is not regular: {item.path}")
    return found


async def _revision_contains_file(workspace: Workspace, revision_dir: str, path: str) -> bool:
    try:
        entry = await workspace.stat(join(revision_dir, path))
    except (OSError, WorkspacePathError):
        return False
    if entry is None or entry.kind == "dir":
        return False
    if entry.kind != "file":
        raise PublicationError(f"revision tree input is not regular: {path}")
    return True


async def archive_original_only_inputs(
    workspace: Workspace,
    *,
    paper_dir: str,
    revision_dir: str,
    archive_dir: str,
    revision_input_sha256: object,
    original_input_sha256: object,
) -> list[str]:
    """Move the original's audited inputs that the revision's audit does not cover to the archive.

    Upstream's ``archive_original_only_inputs``: the files the original's audit bound
    (excluding what publication changes) and that the revision's inputs do not contain are
    copied to ``archive_dir`` with their relative paths and removed from the paper, so that the
    revised paper's active audit inventory is the revision's own. The original's inventory is the
    one its audit recorded.

    Returns:
        The relative paths that were archived.
    """
    revision_files, revision_trees = audit_bound_paths(revision_input_sha256)
    protected = set(revision_files)
    original_hashes = published_audit_hashes(
        original_input_sha256 if isinstance(original_input_sha256, dict) else {}
    )
    _, original_trees = audit_bound_paths(original_hashes)
    original_files, _ = audit_bound_paths(
        {key: value for key, value in original_hashes.items() if not key.startswith(_TREE_PREFIX)}
    )
    inventory = CopyBudget()
    paths = set(original_files)
    for tree in original_trees:
        paths.update(await _tree_files(workspace, paper_dir, tree, inventory))
    archived: list[str] = []
    budget = CopyBudget()
    for path in sorted(paths):
        if path in protected:
            continue
        if any(
            PurePosixPath(path).is_relative_to(tree) for tree in revision_trees
        ) and await _revision_contains_file(workspace, revision_dir, path):
            continue
        source = join(paper_dir, path)
        target = join(archive_dir, path)
        if await workspace.exists(target):
            raise PublicationError(f"ancillary archive destination already exists: {path}")
        entry = await workspace.stat(source)
        if entry is None:
            continue
        if entry.kind != "file":
            raise PublicationError(f"original input is not regular: {path}")
        before = await workspace.digest(source)
        budget.add(size=await copy_regular(workspace, source, target))
        if await workspace.digest(source) != before:
            raise PublicationError(f"original input changed during archival: {path}")
        await workspace.remove(source)
        archived.append(path)
    return archived


# --- the original's snapshot ----------------------------------------------------------


async def snapshot_original(
    workspace: Workspace, *, paper_dir: str, before_review_path: str, budget: CopyBudget
) -> str:
    """Create ``revisions/revision_01/before/`` from the published paper (create once).

    Raises:
        PublicationError: When the snapshot already exists.
    """
    before = join(paper_dir, BEFORE_DIR_NAME)
    if await workspace.exists(before):
        raise PublicationError(f"revision snapshot already exists: {before}")
    present = await copy_present_named_files(workspace, paper_dir, before, SNAPSHOT_FILES, budget)
    await copy_regular(
        workspace, join(paper_dir, MANIFEST_NAME), join(before, "original_manifest.json")
    )
    review_name = posixpath.basename(before_review_path)
    await copy_regular(workspace, before_review_path, join(before, review_name))
    figure_files: list[str] = []
    figures = join(paper_dir, "figures")
    if await workspace.exists(figures):
        figure_files = await copy_directory_contents(
            workspace, figures, join(before, "figures"), budget
        )
    await write_json(
        workspace,
        join(before, "snapshot_complete.json"),
        {"files": present, "figure_files": figure_files, "status": "complete"},
    )
    return before


# --- replay of a promoted revision --------------------------------------------------------------


async def initial_review_must_be_preserved(workspace: Workspace, paper_dir: str) -> bool:
    """Whether a replay must retain the immutable before-revision review (upstream's rule).

    True when the manifest records a completed ``before_revision`` review at the named file, or a
    promoted revision, or a promotion that began and did not finish.
    """
    try:
        manifest = await read_json(
            workspace, join(paper_dir, MANIFEST_NAME), max_bytes=MAX_ARTIFACT_BYTES
        )
    except (PublicationError, OSError):
        manifest = {}
    named = join(paper_dir, BEFORE_REVIEW_NAME)
    before = manifest.get("before_revision")
    if (
        isinstance(before, dict)
        and before.get("status") == "completed"
        and names_path(workspace, before.get("review_json"), named)
    ):
        return True
    revision = manifest.get("revision")
    if isinstance(revision, dict) and revision.get("status") == "promoted":
        return True
    journal = join(paper_dir, REVISION_DIR_NAME, "promotion_state.json")
    if not await workspace.exists(journal):
        return False
    state = await read_json(workspace, journal, max_bytes=MAX_ARTIFACT_BYTES)
    return state.get("status") == "promoting"


async def promoted_original_snapshot(
    workspace: Workspace, paper_dir: str, revision_state: Mapping[str, Any]
) -> str:
    """Return the original snapshot recorded by a promoted revision, after checking it.

    Raises:
        PublicationError: When it is missing, outside ``<paper_dir>/revisions`` or not a
            directory.
    """
    value = revision_state.get("original_snapshot")
    if not isinstance(value, str) or not value:
        raise PublicationError("promoted revision manifest is missing original_snapshot")
    snapshot = workspace_path(workspace, value)
    root = join(paper_dir, "revisions")
    if snapshot is None or not is_within(snapshot, root):
        raise PublicationError(
            "promoted revision original_snapshot is outside the paper revision root"
        )
    current = root
    for part in PurePosixPath(relative_to(snapshot, root)).parts:
        current = join(current, part)
        entry = await workspace.stat(current)
        if entry is None:
            raise PublicationError("promoted revision original_snapshot is missing")
        if entry.kind == "symlink":
            raise PublicationError("promoted revision original_snapshot contains a symlink")
    if not await is_dir(workspace, snapshot):
        raise PublicationError("promoted revision original_snapshot is not a directory")
    return snapshot


# --- promotion -------------------------------------------------------------------------


async def promote_revision(
    workspace: Workspace,
    *,
    paper_dir: str,
    revision_dir: str,
    before_review_path: str,
    original: SnapshotRef,
    runtime: Any = None,
    audit_input_hashes: AuditHashes | None = None,
    scope_audit: ScopeAudit | None = None,
) -> PromotionOutcome:
    """Publish a validated revision over ``paper_dir``, keeping the original.

    Args:
        workspace: The workspace.
        paper_dir: The published paper.
        revision_dir: The validated revision (the paper graph's output).
        before_review_path: The initial review, inside ``paper_dir``.
        original: A snapshot of ``paper_dir`` taken before this call. It is restored first when
            an earlier attempt left a snapshot behind, and again when this attempt fails, so that
            a failure leaves the original paper as it was.
        runtime: Passed to the optional check functions.
        audit_input_hashes: See :func:`~.evidence.validate_passing_audits`.
        scope_audit: See :func:`~.evidence.validate_passing_audits`.

    Returns:
        The outcome: the snapshot directory and what was archived.

    Raises:
        PublicationError: When the revision cannot be published; the original is restored.
        EvidenceError: When the published paper fails the final evidence check; the original is
            restored.
    """
    paper_dir = safe_relpath(paper_dir, allow_root=False)
    revision_dir = safe_relpath(revision_dir, allow_root=False)
    before_review = safe_relpath(before_review_path, allow_root=False)
    if posixpath.dirname(before_review) != paper_dir:
        raise PublicationError("the initial review must be a file of the paper directory")
    if not await is_dir(workspace, paper_dir):
        raise PublicationError(f"paper directory must be a regular directory: {paper_dir}")
    await read_regular(workspace, before_review, max_bytes=MAX_REVIEW_BYTES)
    if await workspace.exists(join(paper_dir, BEFORE_DIR_NAME)):
        # an earlier attempt of this promotion left partial work: start from the original
        await workspace.restore(original)
    try:
        outcome = await _promote(
            workspace,
            paper_dir=paper_dir,
            revision_dir=revision_dir,
            before_review=before_review,
            runtime=runtime,
            audit_input_hashes=audit_input_hashes,
            scope_audit=scope_audit,
        )
    except BaseException:
        await workspace.restore(original)
        raise
    return outcome


async def _promote(
    workspace: Workspace,
    *,
    paper_dir: str,
    revision_dir: str,
    before_review: str,
    runtime: Any,
    audit_input_hashes: AuditHashes | None,
    scope_audit: ScopeAudit | None,
) -> PromotionOutcome:
    manifest = await validate_revision_workspace(workspace, revision_dir)
    evidence_audit = await read_json(
        workspace, join(revision_dir, "paper_evidence_audit.json"), max_bytes=MAX_ARTIFACT_BYTES
    )
    revision_hashes = evidence_audit.get("input_sha256")
    current_manifest = await read_json(
        workspace, join(paper_dir, MANIFEST_NAME), max_bytes=MAX_ARTIFACT_BYTES
    )
    original_audit = await _original_audit(workspace, paper_dir)
    budget = CopyBudget()
    before_dir = await snapshot_original(
        workspace, paper_dir=paper_dir, before_review_path=before_review, budget=budget
    )
    archived: list[str] = []
    if original_audit is not None:
        archived = await archive_original_only_inputs(
            workspace,
            paper_dir=paper_dir,
            revision_dir=revision_dir,
            archive_dir=join(before_dir, "ancillary"),
            revision_input_sha256=revision_hashes,
            original_input_sha256=original_audit.get("input_sha256"),
        )
    await copy_present_named_files(workspace, revision_dir, paper_dir, PROMOTED_FILES, budget)
    figures = join(revision_dir, "figures")
    if await workspace.exists(figures):
        await copy_directory_contents(workspace, figures, join(paper_dir, "figures"), budget)
    await copy_audit_bound_inputs(workspace, revision_dir, paper_dir, revision_hashes, budget)

    promoted = rebase_manifest(workspace, manifest, old_root=revision_dir, new_root=paper_dir)
    assert isinstance(promoted, dict)
    promoted["revision"] = {
        "status": "promoted",
        "source_dir": revision_dir,
        "original_snapshot": before_dir,
        "before_review": before_review,
    }
    before_status = current_manifest.get("before_revision")
    promoted["before_revision"] = (
        before_status
        if isinstance(before_status, dict)
        else {
            "status": "completed",
            "review_json": before_review,
            "log": join(paper_dir, "ai_review.before_revision.log"),
        }
    )
    await write_json(workspace, join(paper_dir, MANIFEST_NAME), promoted)
    await write_json(
        workspace,
        join(paper_dir, REVISION_DIR_NAME, "promotion_state.json"),
        {"status": "completed", "revision_dir": revision_dir},
    )
    skipped = await validate_revision_evidence(
        workspace,
        before_dir,
        paper_dir,
        published=True,
        runtime=runtime,
        audit_input_hashes=audit_input_hashes,
        scope_audit=scope_audit,
    )
    return PromotionOutcome(before_dir=before_dir, archived=archived, skipped_checks=skipped)


async def _original_audit(workspace: Workspace, paper_dir: str) -> dict[str, Any] | None:
    path = join(paper_dir, "paper_evidence_audit.json")
    if not await is_file(workspace, path):
        return None
    try:
        return await read_json_object(workspace, path, "original evidence audit")
    except EvidenceError:
        return None


__all__ = [
    "AFTER_REVIEW_NAME",
    "BEFORE_DIR_NAME",
    "BEFORE_REVIEW_NAME",
    "MANIFEST_NAME",
    "PROMOTED_FILES",
    "REQUIRED_REVISION_FILES",
    "REVISION_DIR_NAME",
    "SNAPSHOT_FILES",
    "CopyBudget",
    "PromotionOutcome",
    "PublicationError",
    "archive_original_only_inputs",
    "audit_bound_paths",
    "copy_audit_bound_inputs",
    "copy_directory_contents",
    "initial_review_must_be_preserved",
    "names_path",
    "promote_revision",
    "promoted_original_snapshot",
    "read_json",
    "read_regular",
    "rebase_manifest",
    "snapshot_original",
    "validate_revision_workspace",
    "workspace_path",
    "write_json",
]
