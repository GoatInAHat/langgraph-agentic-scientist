"""Host-side handling of a drafted scaffold: inspection, canonical files, fingerprints, repair
transactions, the judge's view, report validation and publication.

These are the deterministic steps that upstream's ``task.sh`` runs around the agents (the shell
functions ``inspect_staged_scaffold``, ``validate_staged_paths``, ``canonicalize_staged_scaffold``,
``prepare_alignment_judge_view``, the repair-scope and repair-transaction modules, and
``publication.py``), written over a :class:`~langgraph_agentic_scientist.workspace.Workspace`.

Layout inside the workspace, beside the run's task snapshot (``<run>/task``):

``<run>/_task_drafting/``
    the drafter's working directory (upstream's ``OUTPUT_DIR``): the staged copy of the task in
    ``task/``, the accepted design in ``_task_design/task_design.json``, the memory files the
    drafter writes, and the staged skills;
``<run>/_task_drafting/.harness/``
    host-only files that no agent role is rooted in: the judge's views of the scaffold and the
    authorisation of a dependency removal.

A transaction snapshots the staged task directory with ``Workspace.snapshot`` and restores it with
``Workspace.restore``. A *scoped* transaction (smoke repairs) also records the digest of every file
and the parsed contract, so that a repair that touches a file outside the scaffold's mutable paths
or changes the contract's meaning is detected and undone. A *tree* transaction (alignment and
qualification repairs) restores the whole authored tree.
"""

from __future__ import annotations

import copy
import hashlib
import json
import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from ..core.contract import (
    TASK_CONTRACT_FILENAME,
    TaskContract,
    load_task_contract,
    parse_task_contract_bytes,
    task_scaffold_fingerprint,
)
from ..core.errors import ProgramSpecError, TaskContractError
from ..core.render import json_text, task_sections
from ..core.state import ProgramRef, TaskRef
from ..evaluation.program_spec import (
    entrypoint_relpath,
    load_program_spec,
    render_default_program_spec_toml,
    resolve_entrypoint,
)
from ..workspace.digest import sha256_hex, tree_digest
from ..workspace.paths import join, parse_ignore_patterns
from ..workspace.protocol import FileEntry, Workspace
from .contract_lint import (
    ContractSyncError,
    contract_text,
    declared_dependencies_match_requirements,
    load_contract_object,
    retained_declared_dependencies,
    synchronize_declared_dependency_removals,
    synchronize_declared_digests,
)
from .history import collect_sources
from .models import (
    FINGERPRINTED_FILES,
    REQUIRED_ALIGNMENT_DIMENSIONS,
    AlignmentReport,
    RepairTransaction,
)

DRAFT_REQUIRED_FILES: tuple[str, ...] = (
    "capture.py",
    "evaluate.py",
    "evaluate_final.py",
    "task_contract.json",
)
"""The files the drafter must create besides the baseline entrypoint (upstream rule)."""

_RUNTIME_ROOT_ENTRIES = frozenset({".venv", ".agentic_runtime.json", ".agentic_repair_prompt.md"})
_REPAIR_PROMPT_NAME = ".agentic_repair_prompt.md"
_MAX_FINGERPRINT_ENTRIES = 100_000
MARKER_FILENAME = ".agentic_task_validation.json"
PUBLICATION_FILENAME = ".agentic_task_publication.json"
QUALIFICATION_FILENAME = ".agentic_task_qualification.json"
TASK_COPY_IGNORE_FILENAME = ".agentic_task_ignore"
RESOURCE_MANIFEST = ".agentic_resource_manifest.json"
_RUNTIME_FILE_MAX_BYTES = 1024 * 1024
_MAX_RUNTIME_PUBLICATION_BYTES = 64 * 1024 * 1024 * 1024

DEPENDENCY_AUTHORIZED = (
    "Harness authorized dependency removal: unused direct pins may now be removed."
)
"""The sentence the gate adds to a ``runtime_dependencies`` diagnostic (upstream text)."""


class ScaffoldError(ValueError):
    """A scaffold is unsafe or invalid.

    ``exit_code`` is upstream's: ``3`` for a safely missing canonical entrypoint or an incomplete
    scaffold, ``4`` for an unsafe artifact, ``1`` for every other refusal.
    """

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class RepairScopeError(ScaffoldError):
    """A repair changed data outside its harness-owned mutation boundary."""


class AlignmentReportError(ValueError):
    """An alignment report is malformed or does not match the scaffold."""


class PublicationError(ValueError):
    """A scaffold cannot be published."""


# --- locations -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftPaths:
    """Where a drafting run keeps its files (see the module docstring)."""

    root: str
    task: str
    design_dir: str
    design_file: str
    harness: str
    validation: str

    def judge_view(self, round_index: int) -> str:
        """Return the directory of the judge's view of round ``round_index``."""
        return join(self.harness, f"judge_view_{round_index}")

    def authorization(self, label: str) -> str:
        """Return the host-only authorisation file of repair ``label``."""
        return join(self.harness, f"authorization_{label}.json")


def draft_paths(task_root: str) -> DraftPaths:
    """Return the drafting locations beside the task snapshot at ``task_root``."""
    root = join(posixpath.dirname(task_root), "_task_drafting")
    return DraftPaths(
        root=root,
        task=join(root, "task"),
        design_dir=join(root, "_task_design"),
        design_file=join(root, "_task_design", "task_design.json"),
        harness=join(root, ".harness"),
        validation=join(root, "_task_validation"),
    )


def display_path(path: str) -> str:
    """Render a path the way upstream's ``str(Path(...))`` does (``./task`` becomes ``task``)."""
    return str(PurePosixPath(path))


def prompt_values(
    *, task_dir: str, results_dir: str = ".", task_md: str | None, extra: Mapping[str, str] | None
) -> dict[str, str]:
    """Return the placeholder values of ``load_instructions``: the five standard ones and ``extra``.

    ``task_dir`` and ``results_dir`` are rendered with ``str(Path(...))`` semantics, and
    ``TASK_MD_PATH`` is ``<task_dir>/task.md``.
    """
    goal, rubrics = task_sections(task_md)
    task_display = display_path(task_dir)
    values = {
        "TASK_DIR": task_display,
        "RESULTS_DIR": display_path(results_dir),
        "TASK_MD_PATH": str(PurePosixPath(task_dir) / "task.md"),
        "TASK_GOAL": goal,
        "TASK_RUBRICS": rubrics,
    }
    values.update(extra or {})
    return values


async def candidate_entrypoint(workspace: Workspace, task_root: str) -> str:
    """Return the baseline entrypoint of a task, relative to its root.

    It is the entrypoint declared by ``program_0/program_spec.toml`` of the task snapshot, and
    ``program_0/src/program.py`` when the snapshot declares none (upstream rule).
    """
    try:
        spec = await load_program_spec(workspace, join(task_root, "program_0"))
        return f"program_0/{entrypoint_relpath(spec)}"
    except ProgramSpecError:
        return "program_0/src/program.py"


def draft_values(entrypoint: str) -> dict[str, str]:
    """Return the extra placeholder values of the drafter's prompt and skills (``task.sh``:437-570).

    ``entrypoint`` is the baseline entrypoint relative to the task root (``program_0/...``).
    """
    return {
        "DRAFT_OUTPUT_DIR": ".",
        "DRAFT_CAPTURE_PATH": "./task/capture.py",
        "DRAFT_EVALUATE_PATH": "./task/evaluate.py",
        "DRAFT_FINAL_EVALUATE_PATH": "./task/evaluate_final.py",
        "DRAFT_TASK_CONTRACT_PATH": "./task/task_contract.json",
        "TASK_DESIGN_PATH": "./_task_design/task_design.json",
        "DRAFT_PROGRAM_DIR": "./task/program_0",
        "DRAFT_PROGRAM_SPEC_PATH": "./task/program_0/program_spec.toml",
        "DRAFT_CODE_ROOT": "./task/program_0/src",
        "DRAFT_ENTRYPOINT_PATH": f"./task/{entrypoint}",
        "GLOBAL_MEMORY_MD_FILE": "./memory.md",
        "GLOBAL_MEMORY_JSONL_FILE": "./memory.jsonl",
        "CONCLUSIONS_MD_FILE": "./conclusions.md",
        "CONCLUSIONS_HISTORY_FILE": "./conclusions_history.jsonl",
        "WORKER_INDEX": "0",
    }


# --- staging --------------------------------------------------------------------------------


async def stage_task_copy(workspace: Workspace, task_root: str, paths: DraftPaths) -> None:
    """Copy the task snapshot into the drafting directory (``ensure_task_copy_in_results``).

    Paths matched by the snapshot's ``.agentic_task_ignore`` are not copied. A copy left by an
    interrupted earlier attempt is replaced, so that the staged tree is always a complete copy.
    """
    patterns: tuple[str, ...] = ()
    ignore_file = join(task_root, TASK_COPY_IGNORE_FILENAME)
    entry = await workspace.stat(ignore_file)
    if entry is not None and entry.kind == "file":
        patterns = parse_ignore_patterns((await workspace.read_bytes(ignore_file)).decode("utf-8"))
    await workspace.remove(paths.task)
    await workspace.copy_tree(task_root, paths.task, ignore=patterns)


async def write_design_file(
    workspace: Workspace, paths: DraftPaths, design: Mapping[str, Any]
) -> None:
    """Write the accepted design where the drafter's prompt says it is (JSON, sorted keys)."""
    await workspace.write_bytes(paths.design_file, json_text(dict(design)).encode("utf-8"))


# --- fingerprints --------------------------------------------------------------------------


async def tree_fingerprint(
    workspace: Workspace,
    root: str,
    *,
    omitted_root_entries: frozenset[str] = frozenset(),
) -> str:
    """Hash every regular entry of a scaffold tree and refuse links and special files.

    The traversal order and the entry layout are upstream's ``task_tree_fingerprint`` (a
    directory entry, then its children sorted by name; ``dir`` and ``file`` records with the path
    and the size). A file contributes its SHA-256, taken from the workspace's listing, instead of
    its bytes, so the value differs from upstream's while carrying the same guarantee and without
    reading every byte through the protocol.

    Raises:
        ScaffoldError: For a symlink, a special file, or more than 100,000 entries.
    """
    entry = await workspace.stat(root)
    if entry is None or entry.kind != "dir":
        raise ScaffoldError(f"unsafe scaffold directory: {root}")
    listing = await workspace.list_tree(root)
    if len(listing) > _MAX_FINGERPRINT_ENTRIES:
        raise ScaffoldError("scaffold exceeds fingerprint entry limit")
    children: dict[str, list[FileEntry]] = {}
    for item in listing:
        parent = posixpath.dirname(item.path)
        children.setdefault(parent, []).append(item)
    digest = hashlib.sha256()

    def visit(parent: str) -> None:
        for item in sorted(
            children.get(parent, []), key=lambda value: posixpath.basename(value.path)
        ):
            name = posixpath.basename(item.path)
            if parent == "" and name in (_RUNTIME_ROOT_ENTRIES | omitted_root_entries):
                continue
            if item.kind == "symlink":
                raise ScaffoldError(f"scaffold contains a symlink: {item.path}")
            if item.kind == "dir":
                digest.update(b"dir\0" + item.path.encode() + b"\0")
                visit(item.path)
            elif item.kind == "file":
                sha = (item.sha256 or "").removeprefix("sha256:")
                digest.update(
                    b"file\0" + item.path.encode() + b"\0" + str(item.size).encode() + b"\0"
                )
                digest.update(sha.encode())
            else:
                raise ScaffoldError(f"scaffold contains a special file: {item.path}")

    visit("")
    return "sha256:" + digest.hexdigest()


async def safe_tree_fingerprint(workspace: Workspace, root: str) -> str:
    """Return :func:`tree_fingerprint`, or an all-zero fingerprint when the tree is unsafe."""
    try:
        return await tree_fingerprint(workspace, root)
    except (ScaffoldError, OSError, FileNotFoundError):
        return "sha256:" + "0" * 64


async def _reject_symlink_components(workspace: Workspace, root: str, relative: str) -> None:
    current = root
    for part in PurePosixPath(relative).parts:
        current = join(current, part)
        entry = await workspace.stat(current)
        if entry is not None and entry.kind == "symlink":
            raise ScaffoldError(f"scaffold contains a symlink: {relative}")


async def alignment_fingerprints(workspace: Workspace, root: str) -> dict[str, str]:
    """Fingerprint the task inputs whose semantics the judge reviews (upstream rule).

    The result holds the SHA-256 of the six files of :data:`FINGERPRINTED_FILES`, of the declared
    candidate entrypoint, and the ``declared_scaffold`` fingerprint of the contract's closure.

    Raises:
        ScaffoldError: When a file is missing or unsafe or the declared scaffold is invalid.
    """
    fingerprints: dict[str, str] = {}
    for relative in FINGERPRINTED_FILES:
        entry = await workspace.stat(join(root, relative))
        if entry is None or entry.kind != "file":
            raise ScaffoldError(f"missing safe scaffold file for alignment: {relative}")
        fingerprints[relative] = sha256_hex(await workspace.read_bytes(join(root, relative)))
    try:
        program_dir = join(root, "program_0")
        spec = await load_program_spec(workspace, program_dir)
        declared = f"program_0/{entrypoint_relpath(spec)}"
        await _reject_symlink_components(workspace, root, declared)
        relative_entrypoint = f"program_0/{await resolve_entrypoint(workspace, program_dir)}"
    except (ProgramSpecError, ValueError) as exc:
        raise ScaffoldError(f"invalid candidate program specification: {exc}") from exc
    entry = await workspace.stat(join(root, relative_entrypoint))
    if entry is None or entry.kind != "file":
        raise ScaffoldError(f"missing safe scaffold file for alignment: {relative_entrypoint}")
    fingerprints[relative_entrypoint] = sha256_hex(
        await workspace.read_bytes(join(root, relative_entrypoint))
    )
    try:
        fingerprints["declared_scaffold"] = await task_scaffold_fingerprint(workspace, root)
    except TaskContractError as exc:
        raise ScaffoldError(f"invalid declared scaffold: {exc}") from exc
    return fingerprints


# --- inspection, safety and canonical files -------------------------------------------------


async def validate_staged_paths(
    workspace: Workspace, task_dir: str, *, canonical_task_sha256: str | None
) -> None:
    """Refuse an unsafe staged tree (upstream ``validate_staged_paths``).

    The staged directory and ``task.md`` must be regular, ``task.md`` must equal the canonical
    task, the reserved repair-prompt path must be absent, ``program_0`` must be a real directory
    when present, and no required artifact may sit behind a symlink.

    Args:
        workspace: The workspace holding the staged scaffold.
        task_dir: The staged task directory.
        canonical_task_sha256: The SHA-256 of the canonical ``task.md`` (``None`` skips the check).

    Raises:
        ScaffoldError: With upstream's message.
    """
    root = await workspace.stat(task_dir)
    if root is None or root.kind != "dir":
        raise ScaffoldError(f"Unsafe staged task directory: {task_dir}")
    task_spec = await workspace.stat(join(task_dir, "task.md"))
    if task_spec is None or task_spec.kind != "file":
        raise ScaffoldError("Unsafe canonical task.md")
    if canonical_task_sha256 is not None:
        staged = sha256_hex(await workspace.read_bytes(join(task_dir, "task.md")))
        if staged != canonical_task_sha256:
            raise ScaffoldError("Draft or repair agent modified canonical task.md")
    if await workspace.exists(join(task_dir, _REPAIR_PROMPT_NAME)):
        raise ScaffoldError(
            f"Draft agent created reserved harness path: {join(task_dir, _REPAIR_PROMPT_NAME)}"
        )
    program = await workspace.stat(join(task_dir, "program_0"))
    if program is not None and program.kind != "dir":
        raise ScaffoldError(f"Unsafe drafted program directory: {join(task_dir, 'program_0')}")
    for relative in DRAFT_REQUIRED_FILES:
        current = task_dir
        for part in PurePosixPath(relative).parts:
            current = join(current, part)
            entry = await workspace.stat(current)
            if entry is not None and entry.kind == "symlink":
                raise ScaffoldError(f"Unsafe symlink in drafted artifact path: {current}")


@dataclass(frozen=True)
class Inspection:
    """The result of :func:`inspect_scaffold`: ``ok``, ``missing`` (retry) or ``unsafe`` (fail)."""

    kind: str
    diagnostic: str = ""
    missing: tuple[str, ...] = field(default_factory=tuple)


async def inspect_scaffold(workspace: Workspace, paths: DraftPaths, entrypoint: str) -> Inspection:
    """Check that every required drafted file exists and is a safe regular file.

    Mirrors upstream ``inspect_staged_scaffold`` (exit 0, 3 or 4): a missing file or a missing
    ``program_0`` is retried with feedback; a symlink, a non-regular file or an unsafe parent is
    fatal.
    """
    root = await workspace.stat(paths.task)
    if root is None or root.kind != "dir":
        return Inspection("unsafe", "Unsafe staged task directory")
    program = await workspace.stat(join(paths.task, "program_0"))
    program_missing = program is None
    if program is not None and program.kind != "dir":
        return Inspection("unsafe", "Unsafe drafted program directory: program_0")
    required = [*DRAFT_REQUIRED_FILES, entrypoint]
    missing: list[str] = []
    for relative in required:
        current = paths.task
        for part in PurePosixPath(relative).parts[:-1]:
            current = join(current, part)
            entry = await workspace.stat(current)
            if entry is not None and (entry.kind == "symlink" or entry.kind != "dir"):
                return Inspection("unsafe", f"Unsafe drafted artifact parent: {current}")
        if program_missing and relative == entrypoint:
            continue
        entry = await workspace.stat(join(paths.task, relative))
        if entry is None:
            missing.append(f"Missing required drafted artifact: {relative}")
            continue
        if entry.kind != "file":
            return Inspection("unsafe", f"Unsafe drafted artifact: {relative}")
    if program_missing:
        missing.append("Missing required drafted directory: program_0")
    if missing:
        return Inspection("missing", "\n".join(missing), tuple(missing))
    return Inspection("ok")


async def _prune_bytecode(workspace: Workspace, root: str) -> None:
    for item in await workspace.list_tree(root):
        if "__pycache__" in item.path.split("/") or item.path.endswith(".pyc"):
            await workspace.remove(join(root, item.path))


async def canonicalize_staged_scaffold(
    workspace: Workspace, paths: DraftPaths, source_root: str
) -> None:
    """Restore the harness-owned files of the staged scaffold (``canonicalize_staged_scaffold``).

    ``capture.py`` and the candidate entrypoint are taken from the canonical task when it has
    them; ``program_0/program_spec.toml`` is the canonical task's valid spec or the default spec.
    Generated bytecode is removed.

    Raises:
        ScaffoldError: With ``exit_code`` 3 when the canonical entrypoint safely does not exist
            (a repair then restores it), otherwise 1 for an unsafe artifact.
    """
    await _prune_bytecode(workspace, paths.task)
    source_capture = join(source_root, "capture.py")
    staged_capture = join(paths.task, "capture.py")
    entry = await workspace.stat(source_capture)
    if entry is not None:
        if entry.kind != "file":
            raise ScaffoldError(f"Unsafe canonical capture: {source_capture}")
        await workspace.write_bytes(staged_capture, await workspace.read_bytes(source_capture))
    else:
        staged_entry = await workspace.stat(staged_capture)
        if staged_entry is None or staged_entry.kind != "file":
            raise ScaffoldError(f"Unsafe drafted artifact: {staged_capture}")

    source_program = join(source_root, "program_0")
    source_spec_path = join(source_program, "program_spec.toml")
    for path, label in (
        (source_program, "Unsafe canonical program directory"),
        (source_spec_path, "Unsafe canonical program specification"),
    ):
        link = await workspace.stat(path)
        if link is not None and link.kind == "symlink":
            raise ScaffoldError(f"{label}: {path}")
    source_spec_loaded = None
    try:
        source_spec_loaded = await load_program_spec(workspace, source_program)
        canonical_spec = await workspace.read_bytes(source_spec_path)
    except ProgramSpecError:
        canonical_spec = render_default_program_spec_toml().encode("utf-8")

    staged_program = join(paths.task, "program_0")
    program_entry = await workspace.stat(staged_program)
    if program_entry is None or program_entry.kind != "dir":
        raise ScaffoldError(f"Unsafe drafted program directory: {staged_program}")
    await workspace.write_bytes(join(staged_program, "program_spec.toml"), canonical_spec)
    try:
        staged_spec = await load_program_spec(workspace, staged_program)
        declared_relative = f"program_0/{entrypoint_relpath(staged_spec)}"
        current = paths.task
        for part in PurePosixPath(declared_relative).parts:
            current = join(current, part)
            part_entry = await workspace.stat(current)
            if part_entry is not None and part_entry.kind == "symlink":
                raise ProgramSpecError(f"candidate entrypoint contains a symlink: {current}")
    except ProgramSpecError as exc:
        raise ScaffoldError(f"Invalid canonical program specification: {exc}") from exc
    staged_entrypoint = join(paths.task, declared_relative)
    entrypoint_entry = await workspace.stat(staged_entrypoint)
    if entrypoint_entry is not None and entrypoint_entry.kind != "file":
        raise ScaffoldError(f"Unsafe canonical candidate entrypoint: {staged_entrypoint}")
    if entrypoint_entry is None:
        raise ScaffoldError(
            f"Canonical candidate entrypoint does not exist: {declared_relative}", 3
        )
    if source_spec_loaded is not None:
        try:
            source_relative = await resolve_entrypoint(workspace, source_program)
        except ProgramSpecError:
            return
        await _reject_symlink_components(workspace, source_root, f"program_0/{source_relative}")
        await workspace.write_bytes(
            staged_entrypoint, await workspace.read_bytes(join(source_program, source_relative))
        )


async def validate_staged_resources(
    workspace: Workspace, task_dir: str, *, max_runtime_bytes: int
) -> int:
    """Validate the drafted resource tree and return its byte total (upstream rule).

    Raises:
        PublicationError: ``runtime_resources_exceed_limit: ...`` above ``max_runtime_bytes``, or
            a message about an unsafe resource tree.
    """
    if not 1 <= max_runtime_bytes <= _MAX_RUNTIME_PUBLICATION_BYTES:
        raise PublicationError("max_runtime_bytes must be between 1 byte and 64 GiB")
    root = join(task_dir, "resources")
    entry = await workspace.stat(root)
    if entry is None:
        return 0
    if entry.kind != "dir":
        raise PublicationError(f"unsafe drafted runtime resources: {root}")
    total = 0
    largest: list[tuple[int, str]] = []
    for item in await workspace.list_tree(root):
        if item.kind == "symlink" or item.kind == "other":
            raise PublicationError(f"unsafe drafted runtime resources: {root}/{item.path}")
        if item.kind != "file" or PurePosixPath(item.path).name == RESOURCE_MANIFEST:
            continue
        total += item.size
        largest = sorted([*largest, (item.size, item.path)], reverse=True)[:5]
        if total > max_runtime_bytes:
            contributors = ", ".join(f"{name}:{size}" for size, name in largest)
            raise PublicationError(
                "runtime_resources_exceed_limit: "
                f"observed={total} limit={max_runtime_bytes}; largest_observed=[{contributors}]"
            )
    return total


# --- repair scope and transactions ---------------------------------------------------------


def contract_policy_for_smoke_diagnostic(diagnostic: str) -> str:
    """Select the narrow contract policy of a smoke repair from its typed diagnostic.

    A ``runtime_dependencies`` diagnostic permits removing unused pins; any other permits only
    refreshing declared digests (upstream ``contract_policy_for_smoke_diagnostic``).
    """
    try:
        payload = json.loads(diagnostic)
    except (json.JSONDecodeError, TypeError):
        payload = {}
    if isinstance(payload, dict) and payload.get("category") == "runtime_dependencies":
        return "synchronized-dependency-removals"
    return "synchronized-digests"


async def task_files(workspace: Workspace, root: str) -> dict[str, str]:
    """Return the SHA-256 of every file of a scaffold tree, refusing links and special files."""
    files: dict[str, str] = {}
    for item in await workspace.list_tree(root):
        if item.kind == "symlink":
            raise RepairScopeError(f"repair scope contains symlink: {item.path}")
        if item.kind == "file":
            files[item.path] = (item.sha256 or "").removeprefix("sha256:")
        elif item.kind != "dir":
            raise RepairScopeError(f"repair scope contains special path: {item.path}")
    return files


async def mutable_paths(workspace: Workspace, root: str) -> list[str]:
    """Return the files a smoke repair may change: both evaluators, requirements, the entrypoint."""
    try:
        contract = parse_task_contract_bytes(
            await workspace.read_bytes(join(root, TASK_CONTRACT_FILENAME))
        )
        entrypoint = await resolve_entrypoint(workspace, join(root, "program_0"))
    except (TaskContractError, ProgramSpecError, OSError, ValueError) as exc:
        raise RepairScopeError("invalid scaffold metadata during repair") from exc
    return sorted(
        {
            contract.evaluators.search,
            contract.evaluators.final,
            "requirements.txt",
            f"program_0/{entrypoint}",
        }
    )


async def open_scoped_transaction(
    workspace: Workspace, paths: DraftPaths, *, label: str, policy: str
) -> RepairTransaction:
    """Snapshot the staged scaffold before a smoke repair and record its mutation boundary."""
    files = await task_files(workspace, paths.task)
    contract = load_contract_object(
        await workspace.read_bytes(join(paths.task, TASK_CONTRACT_FILENAME))
    )
    mutable = await mutable_paths(workspace, paths.task)
    if policy != "immutable":
        mutable.append(TASK_CONTRACT_FILENAME)
    snapshot = await workspace.snapshot(paths.task)
    return RepairTransaction(
        label=label,
        kind="scoped",
        snapshot=snapshot,
        base_fingerprint=await tree_fingerprint(workspace, paths.task),
        files=files,
        mutable_paths=sorted(set(mutable)),
        contract_policy=policy,
        contract_invariants=contract,
    )


async def open_tree_transaction(
    workspace: Workspace, paths: DraftPaths, *, label: str
) -> RepairTransaction:
    """Snapshot the whole authored tree before an alignment or qualification repair."""
    fingerprint = await tree_fingerprint(workspace, paths.task)
    sources, omitted = await collect_sources(workspace, paths.task)
    snapshot = await workspace.snapshot(paths.task)
    return RepairTransaction(
        label=label,
        kind="tree",
        snapshot=snapshot,
        base_fingerprint=fingerprint,
        sources_before=sources,
        sources_omitted=omitted,
    )


async def restore_transaction(workspace: Workspace, transaction: RepairTransaction) -> None:
    """Bring the staged scaffold back to the transaction's snapshot."""
    await workspace.restore(transaction.snapshot)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _validate_synchronized_contract_digests(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> None:
    before_semantics = copy.deepcopy(dict(before))
    after_semantics = copy.deepcopy(dict(after))
    try:
        for value in (before_semantics, after_semantics):
            value["canonical_task"]["sha256"] = "<digest>"
            for artifact in value["provenance"]["artifacts"]:
                artifact["sha256"] = "<digest>"
    except (KeyError, TypeError):
        raise RepairScopeError("repair changed task-contract semantics") from None
    if _canonical_json(before_semantics) != _canonical_json(after_semantics):
        raise RepairScopeError("repair changed task-contract semantics")


async def validate_repair_scope(
    workspace: Workspace,
    paths: DraftPaths,
    transaction: RepairTransaction,
    *,
    authorized: bool = False,
) -> tuple[tuple[str, ...], RepairTransaction]:
    """Reject out-of-scope mutations and protected contract changes after a smoke repair.

    Args:
        workspace: The workspace holding the staged scaffold.
        paths: The drafting locations.
        transaction: The scoped transaction taken before the repair.
        authorized: Whether the harness authorised dependency removal during the repair.

    Returns:
        The changed paths and the transaction (with the dependency journal updated).

    Raises:
        RepairScopeError: When the repair left its boundary; the caller restores the snapshot.
    """
    before = transaction.files
    after = await task_files(workspace, paths.task)
    changed = tuple(
        sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    )
    disallowed = tuple(path for path in changed if path not in transaction.mutable_paths)
    if disallowed:
        raise RepairScopeError("repair changed out-of-scope path(s): " + ", ".join(disallowed))
    contract = load_contract_object_or_scope_error(
        await workspace.read_bytes(join(paths.task, TASK_CONTRACT_FILENAME))
    )
    original = transaction.contract_invariants or {}
    policy = transaction.contract_policy
    if authorized:
        if policy == "immutable":
            raise RepairScopeError("dependency authorization requires a synchronized repair scope")
        policy = "synchronized-dependency-removals"
    if policy == "synchronized-digests":
        _validate_synchronized_contract_digests(original, contract)
        try:
            await synchronize_declared_digests(workspace, paths.task)
        except ContractSyncError as exc:
            raise RepairScopeError("repair left an invalid declared contract digest") from exc
        try:
            matches = await declared_dependencies_match_requirements(workspace, paths.task)
        except ContractSyncError as exc:
            raise RepairScopeError("repair left invalid dependency declarations") from exc
        if not matches:
            raise RepairScopeError(
                "dependency changes require a runtime_dependencies repair diagnostic"
            )
    elif policy == "synchronized-dependency-removals":
        transaction = await _validate_dependency_removals(
            workspace, paths, transaction, original, contract
        )
    elif contract != original:
        raise RepairScopeError("repair changed immutable task-contract fields")
    return changed, transaction


def load_contract_object_or_scope_error(raw: bytes) -> dict[str, Any]:
    """Parse a contract as an object; a parse failure is a scope error."""
    try:
        return load_contract_object(raw)
    except ContractSyncError as exc:
        raise RepairScopeError(f"invalid task contract during repair: {exc}") from exc


async def _validate_dependency_removals(
    workspace: Workspace,
    paths: DraftPaths,
    transaction: RepairTransaction,
    original: dict[str, Any],
    contract: dict[str, Any],
) -> RepairTransaction:
    allowed = original["provenance"]["dependencies"]
    journal = transaction.retained_dependencies
    if journal is None:
        _validate_synchronized_contract_digests(original, contract)
        try:
            journal = await retained_declared_dependencies(
                workspace, paths.task, allowed_dependencies=allowed
            )
        except (ContractSyncError, KeyError, TypeError) as exc:
            if "only remove pinned dependencies" in str(exc):
                raise RepairScopeError(str(exc)) from exc
            raise RepairScopeError("repair left invalid synchronized dependency removals") from exc
        transaction = transaction.model_copy(update={"retained_dependencies": journal})
    else:
        synchronized = copy.deepcopy(original)
        synchronized["provenance"]["dependencies"] = journal
        try:
            _validate_synchronized_contract_digests(original, contract)
        except RepairScopeError:
            _validate_synchronized_contract_digests(synchronized, contract)
    try:
        await synchronize_declared_dependency_removals(
            workspace, paths.task, allowed_dependencies=allowed
        )
    except (ContractSyncError, KeyError, TypeError) as exc:
        if "only remove pinned dependencies" in str(exc):
            raise RepairScopeError(str(exc)) from exc
        raise RepairScopeError("repair left invalid synchronized dependency removals") from exc
    refreshed = load_contract_object_or_scope_error(
        await workspace.read_bytes(join(paths.task, TASK_CONTRACT_FILENAME))
    )
    if refreshed["provenance"]["dependencies"] != journal:
        raise RepairScopeError("repair changed synchronized dependencies")
    return transaction


async def authorize_dependency_removal(
    workspace: Workspace,
    paths: DraftPaths,
    transaction: RepairTransaction,
    *,
    report_fingerprint: str,
) -> None:
    """Check that a ``runtime_dependencies`` failure inside a repair may authorise removal.

    Upstream authorises only when the repair opened with the digest-synchronised scope, nothing
    outside the mutable paths changed, ``requirements.txt`` is unchanged, the contract still
    differs from its original only in digests, and the failing fingerprint is the current one
    (``task_repair_authorization._validate_before_authorization``).

    Raises:
        RepairScopeError: With upstream's message when authorisation must be refused.
    """
    if transaction.contract_policy != "synchronized-digests":
        raise RepairScopeError("dependency transition requires an original digest repair scope")
    after = await task_files(workspace, paths.task)
    before = transaction.files
    changed = {path for path in before.keys() | after.keys() if before.get(path) != after.get(path)}
    if changed - set(transaction.mutable_paths):
        raise RepairScopeError("out-of-scope changes before dependency authorization")
    if before.get("requirements.txt") != after.get("requirements.txt"):
        raise RepairScopeError("dependencies changed before authorization")
    contract = load_contract_object_or_scope_error(
        await workspace.read_bytes(join(paths.task, TASK_CONTRACT_FILENAME))
    )
    _validate_synchronized_contract_digests(transaction.contract_invariants or {}, contract)
    if await tree_fingerprint(workspace, paths.task) != report_fingerprint:
        raise RepairScopeError("stale dependency failure fingerprint")


# --- the judge's view and the alignment report ---------------------------------------------


async def prepare_judge_view(
    workspace: Workspace, paths: DraftPaths, round_index: int, *, has_design: bool
) -> str:
    """Copy the staged scaffold into the judge's view and return its directory.

    The view omits ``.venv`` and ``.agentic_runtime.json`` and, when the accepted design exists,
    holds it as ``_accepted_task_design.json`` (upstream ``prepare_alignment_judge_view``).
    """
    view = paths.judge_view(round_index)
    await workspace.remove(view)
    await workspace.copy_tree(paths.task, view, ignore=("/.venv", "/.agentic_runtime.json"))
    if has_design:
        await workspace.write_bytes(
            join(view, "_accepted_task_design.json"), await workspace.read_bytes(paths.design_file)
        )
    return view


async def _validate_evidence(
    workspace: Workspace, evidence_root: str, dimension: str, value: Sequence[Any]
) -> None:
    if not value:
        raise AlignmentReportError(f"dimension {dimension} requires non-empty evidence")
    for item in value:
        raw = item.path.strip()
        path = PurePosixPath(raw)
        if not raw or path.is_absolute() or ".." in path.parts or raw.endswith("/"):
            raise AlignmentReportError("evidence path must be a safe relative file")
        target = join(evidence_root, *path.parts)
        entry = await workspace.stat(target)
        if entry is None or entry.kind != "file":
            raise AlignmentReportError("evidence path must be a safe relative file")
        if item.line < 1:
            raise AlignmentReportError(f"dimension {dimension} evidence line must be >= 1")
        try:
            text = (await workspace.read_bytes(target)).decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise AlignmentReportError("evidence path must be a readable UTF-8 file") from exc
        if item.line > len(text.splitlines()):
            raise AlignmentReportError(f"dimension {dimension} evidence line exceeds file length")
        if not item.detail.strip():
            raise AlignmentReportError(f"dimension {dimension} evidence detail is required")


async def validate_alignment_report(
    workspace: Workspace,
    report: AlignmentReport,
    *,
    expected_fingerprints: Mapping[str, str],
    evidence_root: str,
) -> tuple[str, ...]:
    """Validate a judge's report with upstream's rules and return the failed dimension ids.

    The report must be schema 1, carry a non-empty summary and a ``pass`` or ``fail`` verdict,
    quote exactly the expected fingerprints, contain exactly the eleven required dimensions, each
    with existing scaffold-relative evidence at an existing line (checked in the judge's view),
    a remediation for each failed dimension and none for a passed one, and a top-level verdict
    that matches the dimensions.

    Raises:
        AlignmentReportError: With upstream's message (shown to the judge on the report retry).
    """
    if report.report_schema_version != 1:
        raise AlignmentReportError("task alignment report schema_version must be 1")
    if not " ".join(report.summary.split()):
        raise AlignmentReportError("task alignment report requires a summary")
    verdict = report.verdict.strip()
    if verdict not in {"pass", "fail"}:
        raise AlignmentReportError("task alignment report verdict must be pass or fail")
    if dict(report.fingerprints) != dict(expected_fingerprints):
        raise AlignmentReportError("task alignment report fingerprints do not match scaffold")
    rows: dict[str, Any] = {}
    for row in report.dimensions:
        identifier = row.id.strip()
        if identifier in rows:
            raise AlignmentReportError(f"duplicate alignment dimension: {identifier}")
        rows[identifier] = row
    if set(rows) != set(REQUIRED_ALIGNMENT_DIMENSIONS):
        raise AlignmentReportError(
            "dimensions must contain exactly the required alignment dimensions"
        )
    failed: list[str] = []
    for identifier in REQUIRED_ALIGNMENT_DIMENSIONS:
        row = rows[identifier]
        dimension_verdict = row.verdict.strip()
        if dimension_verdict not in {"pass", "fail"}:
            raise AlignmentReportError(f"dimension {identifier} verdict must be pass or fail")
        await _validate_evidence(workspace, evidence_root, identifier, row.evidence)
        remediation = row.remediation.strip()
        if dimension_verdict == "fail":
            failed.append(identifier)
            if not remediation:
                raise AlignmentReportError(f"failed dimension {identifier} requires remediation")
        elif remediation:
            raise AlignmentReportError(
                f"passed dimension {identifier} must not include remediation"
            )
    expected_verdict = "fail" if failed else "pass"
    if verdict != expected_verdict:
        raise AlignmentReportError("top-level verdict does not match dimension verdicts")
    return tuple(failed)


def bounded_repair_report(report: AlignmentReport) -> str:
    """Return the bounded failed report that the alignment repair prompt embeds.

    Only failed dimensions are kept; each keeps its first eight evidence items (path 512 bytes,
    detail 768) and a remediation of 2048 bytes; the summary is cut at 2048 bytes. The text is the
    indented JSON of upstream's prompt and must stay within 96 KiB.

    Raises:
        ValueError: When the bounded report exceeds 96 KiB.
    """

    def bounded(value: object, limit: int) -> str:
        encoded = str(value).encode("utf-8")
        if len(encoded) <= limit:
            return str(value)
        return encoded[:limit].decode("utf-8", errors="ignore") + "…"

    failed: list[dict[str, Any]] = []
    for dimension in report.dimensions:
        if dimension.verdict != "fail":
            continue
        failed.append(
            {
                "id": dimension.id,
                "verdict": "fail",
                "evidence": [
                    {
                        "path": bounded(evidence.path, 512),
                        "line": evidence.line,
                        "detail": bounded(evidence.detail, 768),
                    }
                    for evidence in dimension.evidence[:8]
                ],
                "evidence_omitted": max(0, len(dimension.evidence) - 8),
                "remediation": bounded(dimension.remediation, 2048),
            }
        )
    text = json.dumps(
        {
            "schema_version": report.report_schema_version,
            "verdict": report.verdict,
            "summary": bounded(report.summary, 2048),
            "fingerprints": report.fingerprints,
            "dimensions": failed,
        },
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if len(text.encode("utf-8")) > 96 * 1024:
        raise ValueError("Bounded alignment repair report exceeds 96 KiB")
    return text


# --- publication ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicationPlan:
    """What publication copies, bound by a SHA-256 over its manifest (upstream plan)."""

    files: tuple[str, ...]
    file_sha256: tuple[tuple[str, str], ...]
    ignore_file: bytes
    resources: str | None
    resource_sha256: str | None
    manifest: bytes

    @property
    def sha256(self) -> str:
        """The digest a qualification receipt binds."""
        return hashlib.sha256(self.manifest).hexdigest()


def _is_below(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


async def _regular_files_below(workspace: Workspace, root: str, relative: str) -> set[str]:
    entry = await workspace.stat(join(root, relative))
    if entry is None or entry.kind == "symlink":
        raise PublicationError(f"missing safe candidate editable path: {relative}")
    if entry.kind == "file":
        return {relative}
    if entry.kind != "dir":
        raise PublicationError(f"unsafe candidate editable path: {relative}")
    files: set[str] = set()
    for item in await workspace.list_tree(join(root, relative)):
        if item.kind == "symlink" or item.kind == "other":
            raise PublicationError(f"unsafe candidate editable path: {relative}/{item.path}")
        if item.kind == "file":
            files.add(f"{relative}/{item.path}")
    return files


def _reject_reserved_artifact(relative: str) -> None:
    reserved = {
        MARKER_FILENAME,
        PUBLICATION_FILENAME,
        QUALIFICATION_FILENAME,
        TASK_COPY_IGNORE_FILENAME,
        ".agentic_runtime.json",
    }
    first = PurePosixPath(relative).parts[0]
    if (
        any(_is_below(relative, item) for item in reserved)
        or first == ".venv"
        or PurePosixPath(relative).name == RESOURCE_MANIFEST
    ):
        raise PublicationError(f"contract provenance artifact uses harness-owned path: {relative}")


async def _ignore_file_bytes(
    workspace: Workspace, published_dir: str, contract: TaskContract
) -> bytes:
    path = join(published_dir, TASK_COPY_IGNORE_FILENAME)
    entry = await workspace.stat(path)
    patterns = (
        (await workspace.read_bytes(path)).decode("utf-8").splitlines()
        if entry is not None and entry.kind == "file"
        else []
    )
    required = (
        ".venv/",
        "resources/",
        f"/{contract.evaluators.final}",
        f"/{contract.resources.final_root}/",
        "/task_contract.json",
        f"/{PUBLICATION_FILENAME}",
        "/.agentic_task_validation.json",
        f"/{QUALIFICATION_FILENAME}",
    )
    patterns.extend(pattern for pattern in required if pattern not in patterns)
    return ("\n".join(patterns) + "\n").encode()


async def _resource_digest(workspace: Workspace, root: str) -> str:
    entries = [
        item
        for item in await workspace.list_tree(root)
        if PurePosixPath(item.path).name != RESOURCE_MANIFEST
    ]

    async def read(relative: str) -> bytes:
        return await workspace.read_bytes(join(root, relative))

    return (await tree_digest(entries, read)).removeprefix("sha256:")


async def build_publication_plan(
    workspace: Workspace,
    staged_dir: str,
    published_dir: str,
    *,
    max_runtime_bytes: int,
) -> PublicationPlan:
    """Plan the publication of a staged scaffold into the task directory (upstream ``_build_plan``).

    The files are ``capture.py``, the canonical task, both evaluators, the contract, the program
    spec, the candidate entrypoint and every file below the editable paths, the declared artifacts
    outside ``resources/``, and ``requirements.txt``; the resource tree is published as a whole.

    Raises:
        PublicationError: For a generated helper artifact, a harness-owned path, an unsafe file,
            or resources above ``max_runtime_bytes``.
    """
    try:
        contract = await load_task_contract(workspace, staged_dir)
    except TaskContractError as exc:
        raise PublicationError(str(exc)) from exc
    for artifact in contract.provenance.artifacts:
        if artifact.role == "search_helper":
            raise PublicationError(
                "generated task-local evaluator helpers are unsupported; "
                f"inline helper {artifact.path} into {contract.evaluators.search}"
            )
    files = {
        "capture.py",
        contract.canonical_task.path,
        contract.evaluators.search,
        contract.evaluators.final,
        TASK_CONTRACT_FILENAME,
        "program_0/program_spec.toml",
    }
    try:
        files.add(f"program_0/{await resolve_entrypoint(workspace, join(staged_dir, 'program_0'))}")
    except (ProgramSpecError, ValueError) as exc:
        raise PublicationError(f"invalid drafted program specification: {exc}") from exc
    for editable in contract.candidate.editable_paths:
        files.update(await _regular_files_below(workspace, staged_dir, editable))
    files.update(
        artifact.path
        for artifact in contract.provenance.artifacts
        if not _is_below(artifact.path, "resources")
    )
    for artifact in contract.provenance.artifacts:
        _reject_reserved_artifact(artifact.path)
    requirements = await workspace.stat(join(staged_dir, "requirements.txt"))
    if requirements is not None and requirements.kind == "file":
        if requirements.size > _RUNTIME_FILE_MAX_BYTES:
            raise PublicationError(
                f"drafted runtime file exceeds 1 MiB: {join(staged_dir, 'requirements.txt')}"
            )
        files.add("requirements.txt")
    ignore_file = await _ignore_file_bytes(workspace, published_dir, contract)
    resources_entry = await workspace.stat(join(staged_dir, "resources"))
    resources = (
        "resources" if resources_entry is not None and resources_entry.kind == "dir" else None
    )
    file_hashes: list[tuple[str, str]] = []
    for relative in sorted(files):
        entry = await workspace.stat(join(staged_dir, relative))
        if entry is None or entry.kind != "file":
            raise PublicationError(f"unsafe regular file: {relative}")
        file_hashes.append(
            (relative, sha256_hex(await workspace.read_bytes(join(staged_dir, relative))))
        )
    resource_sha256: str | None = None
    resource_bytes: int | None = None
    if resources is not None:
        resource_bytes = await validate_staged_resources(
            workspace, staged_dir, max_runtime_bytes=max_runtime_bytes
        )
        resource_sha256 = await _resource_digest(workspace, join(staged_dir, resources))
    payload = {
        "schema_version": 1,
        "max_runtime_bytes": max_runtime_bytes,
        "files": dict(file_hashes),
        "generated": {TASK_COPY_IGNORE_FILENAME: hashlib.sha256(ignore_file).hexdigest()},
        "resources": (
            None
            if resources is None
            else {"path": resources, "sha256": resource_sha256, "bytes": resource_bytes}
        ),
    }
    manifest = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return PublicationPlan(
        files=tuple(relative for relative, _ in file_hashes),
        file_sha256=tuple(file_hashes),
        ignore_file=ignore_file,
        resources=resources,
        resource_sha256=resource_sha256,
        manifest=manifest,
    )


async def publish_scaffold(
    workspace: Workspace,
    plan: PublicationPlan,
    *,
    staged_dir: str,
    published_dir: str,
    alignment_report: bytes,
    qualification_receipt: bytes | None,
) -> None:
    """Publish the planned files, then the receipts, then the validation marker (last).

    Publication is idempotent: bytes already in place are left alone, so a resumed publication
    completes without changing what an interrupted one wrote. A file that exists with different
    content, and a stale qualification receipt without a qualification, are refused (upstream
    ``publish_task_scaffold``). The marker, which later phases read as proof of acceptance, is
    written last.

    Raises:
        PublicationError: On a conflict with the destination.
    """
    receipt_path = join(published_dir, QUALIFICATION_FILENAME)
    if qualification_receipt is None and await workspace.exists(receipt_path):
        raise PublicationError("task destination contains a stale qualification receipt")
    marker_path = join(published_dir, MARKER_FILENAME)
    if await workspace.exists(marker_path):
        existing = await workspace.read_bytes(marker_path)
        if existing != alignment_report:
            raise PublicationError("task destination has a different validation marker")

    async def put(target: str, content: bytes, *, replaceable: bool = False) -> None:
        entry = await workspace.stat(target)
        if entry is not None:
            if entry.kind != "file":
                raise PublicationError(f"task destination contains a non-file: {target}")
            current = await workspace.read_bytes(target)
            if current == content:
                return
            if not replaceable:
                raise PublicationError(f"task destination contains a conflicting file: {target}")
        await workspace.write_bytes(target, content)

    await put(join(published_dir, PUBLICATION_FILENAME), plan.manifest)
    for relative in plan.files:
        await put(
            join(published_dir, relative), await workspace.read_bytes(join(staged_dir, relative))
        )
    if plan.resources is not None:
        source = join(staged_dir, plan.resources)
        target = join(published_dir, plan.resources)
        if await workspace.exists(target):
            if await _resource_digest(workspace, target) != plan.resource_sha256:
                raise PublicationError("task destination contains different resources")
        else:
            await workspace.copy_tree(source, target)
            if await _resource_digest(workspace, target) != plan.resource_sha256:
                raise PublicationError("published resources differ from publication plan")
    await put(join(published_dir, TASK_COPY_IGNORE_FILENAME), plan.ignore_file, replaceable=True)
    if qualification_receipt is not None:
        await put(receipt_path, qualification_receipt)
    for relative, digest in plan.file_sha256:
        published = await workspace.read_bytes(join(published_dir, relative))
        if sha256_hex(published) != digest:
            raise PublicationError(f"published file differs from publication plan: {relative}")
    await put(marker_path, alignment_report)


async def published_task_ref(workspace: Workspace, source: TaskRef, published_dir: str) -> TaskRef:
    """Return the task reference of the published scaffold, with its contract loaded."""
    contract = await load_task_contract(workspace, published_dir)
    entrypoint = await resolve_entrypoint(workspace, join(published_dir, "program_0"))
    return source.model_copy(
        update={
            "root": published_dir,
            "contract": contract,
            "harness_evaluated": True,
            "seed_program": ProgramRef(
                path=join(published_dir, "program_0"), entrypoint=entrypoint
            ),
        }
    )


__all__ = [
    "DEPENDENCY_AUTHORIZED",
    "DRAFT_REQUIRED_FILES",
    "MARKER_FILENAME",
    "PUBLICATION_FILENAME",
    "QUALIFICATION_FILENAME",
    "AlignmentReportError",
    "DraftPaths",
    "Inspection",
    "PublicationError",
    "PublicationPlan",
    "RepairScopeError",
    "ScaffoldError",
    "alignment_fingerprints",
    "authorize_dependency_removal",
    "bounded_repair_report",
    "build_publication_plan",
    "candidate_entrypoint",
    "canonicalize_staged_scaffold",
    "contract_policy_for_smoke_diagnostic",
    "contract_text",
    "display_path",
    "draft_paths",
    "draft_values",
    "inspect_scaffold",
    "mutable_paths",
    "open_scoped_transaction",
    "open_tree_transaction",
    "prepare_judge_view",
    "prompt_values",
    "publish_scaffold",
    "published_task_ref",
    "restore_transaction",
    "safe_tree_fingerprint",
    "stage_task_copy",
    "task_files",
    "tree_fingerprint",
    "validate_alignment_report",
    "validate_repair_scope",
    "validate_staged_paths",
    "validate_staged_resources",
    "write_design_file",
]
