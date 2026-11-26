"""Stage 0: extract the best-program bundle and stage the evidence (upstream ``extract.py``).

The bundle ``<output>/_extracted_bundle/`` holds the task-level files of the run's task snapshot,
the best program as ``program_0``, the seed as ``reference_program`` and an
``extraction_manifest.json`` (upstream ``best_program_extraction.py``). When a completed ablations
directory is given, a bounded, hash-manifested allowlist of its files is staged as
``evidence/`` and ``evidence_manifest.json`` (upstream ``_stage_evidence``): the pipeline summary
and README, the generated variant proposals, the fixed aggregate CSVs and every validated
``evaluation_evidence.json`` of a variant program. Verification and restoration of the staged
evidence follow upstream, so the bootstrap agent can never change what later stages read.

Adaptations: symlinks are not carried (the workspace protocol has no operation that creates
them), staged evidence is not made read-only (the manifest verification detects any change and
:func:`restore_verified_evidence` restores it), and a task of a contract leaves its sealed files
(the final evaluator, the final resources and the contract) out of the bundle, because no agent of
the search side may read them.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from ..config import PosthocConfig
from ..core.contract import TaskContract
from ..core.errors import ProgramSpecError, WorkspacePathError
from ..core.state import TaskRef, utc_now
from ..evaluation.evidence import EVIDENCE_FILENAME, parse_evaluation_evidence
from ..evaluation.program_clone import clone_program
from ..evaluation.program_spec import load_program_spec, resolve_entrypoint
from ..workspace.paths import (
    TASK_COPY_IGNORE_FILENAME,
    join,
    parse_ignore_patterns,
    task_copy_path_is_ignored,
)
from ..workspace.protocol import FileEntry, Workspace
from .common import TASK_COPY_EXCLUDE_DIRS, json_file_text, read_text_or_empty

REQUIRED_BUNDLE_ENTRIES = ["program_0", "evaluate.py", "task.md", "extraction_manifest.json"]
MAX_EVIDENCE_TASK_DIRS = 64
MAX_EVIDENCE_SUMMARY_BYTES = 5 * 1024 * 1024
MAX_EVIDENCE_MANIFEST_BYTES = 1024 * 1024
MAX_EVIDENCE_TREE_ENTRIES = 1024
MAX_EVIDENCE_TREE_DEPTH = 4
MAX_DIRECTORY_ENTRIES = 512
MAX_PROGRAMS = 512
AGGREGATE_FILENAMES = ("run_summary.csv", "metric_values.csv", "timeseries.csv")
TASK_COPY_EXCLUDE_FILES = frozenset({".agentic_runtime.json"})
SEALED_CONTRACT_FILE = "task_contract.json"

_PROGRAM_DIR = re.compile(r"^program_(\d+)$")
_HEX = frozenset("0123456789abcdef")


class BundleValidationError(RuntimeError):
    """The extracted bundle or its staged evidence is incomplete or unsafe."""


class ExtractionError(RuntimeError):
    """Extraction validation or materialization failed (upstream ``ExtractionError``)."""


# --- the extracted bundle -----------------------------------------------------------------------


def sealed_task_paths(contract: TaskContract | None) -> tuple[str, ...]:
    """Return the task-relative paths a bundle must not carry for a task with a contract."""
    if contract is None:
        return ()
    return (SEALED_CONTRACT_FILE, contract.evaluators.final, contract.resources.final_root)


def _sealed(relpath: str, sealed: tuple[str, ...]) -> bool:
    return any(relpath == path or relpath.startswith(path.rstrip("/") + "/") for path in sealed)


async def copy_task_inputs(
    workspace: Workspace, task: TaskRef, bundle_dir: str, reference_dir: str
) -> list[str]:
    """Copy the task-level files and the seed program into the bundle (upstream rules).

    Program variant directories (``program_<n>``) and cache directories are pruned at any depth,
    files matching the task's ``.agentic_task_ignore`` patterns are skipped, and a task with a
    contract leaves its sealed files out. Returns the symlinks that were not carried.
    """
    root = task.root
    patterns = parse_ignore_patterns(
        await read_text_or_empty(workspace, join(root, TASK_COPY_IGNORE_FILENAME))
    )
    sealed = sealed_task_paths(task.contract)
    skipped_links: list[str] = []
    pruned: list[str] = []
    for entry in await workspace.list_tree(root):
        relpath = entry.path
        if any(relpath.startswith(prefix + "/") for prefix in pruned):
            continue
        name = PurePosixPath(relpath).name
        if entry.kind == "dir":
            if (
                _PROGRAM_DIR.match(name)
                or name in TASK_COPY_EXCLUDE_DIRS
                or task_copy_path_is_ignored(relpath, is_dir=True, patterns=patterns)
                or _sealed(relpath, sealed)
            ):
                pruned.append(relpath)
            continue
        if "/" not in relpath and name in TASK_COPY_EXCLUDE_FILES:
            continue
        if task_copy_path_is_ignored(relpath, is_dir=False, patterns=patterns) or _sealed(
            relpath, sealed
        ):
            continue
        if entry.kind == "symlink":
            skipped_links.append(relpath)
        elif entry.kind == "file":
            await workspace.write_bytes(
                join(bundle_dir, relpath), await workspace.read_bytes(join(root, relpath))
            )
    skipped_links.extend(
        (
            await clone_program(
                workspace, src_dir=reference_dir, dst_dir=join(bundle_dir, "reference_program")
            )
        ).skipped_symlinks
    )
    return skipped_links


async def extract_bundle(
    workspace: Workspace,
    *,
    task: TaskRef,
    best_program_dir: str,
    manifest: Mapping[str, Any],
    results_dir: str | None,
    reference_dir: str | None,
    output_dir: str,
    downstream_selection: Mapping[str, Any] | None,
    evidence_dir: str | None,
    config: PosthocConfig,
) -> dict[str, Any]:
    """Create ``<output_dir>/_extracted_bundle`` and return its extraction manifest.

    Args:
        workspace: The run's workspace.
        task: The run-local task snapshot.
        best_program_dir: The winning program directory.
        manifest: ``entrypoint_path``, ``combined_score``, ``correct``, ``source_attempt_id`` and
            ``parallel_worker_index`` of the winner (upstream's best-program manifest fields).
        results_dir: The investigation's results directory, when known.
        reference_dir: The seed program staged as ``reference_program`` (default: the task's
            ``program_0``).
        output_dir: The phase's output directory.
        downstream_selection: The downstream selection record, when the winner came from one.
        evidence_dir: A completed ablations directory staged as evidence, if any.
        config: Evidence limits.

    Raises:
        ExtractionError: When an input is missing or invalid.
        BundleValidationError: When the bundle or the evidence is incomplete or unsafe.
    """
    bundle_dir = join(output_dir, "_extracted_bundle")
    reference_dir = reference_dir or join(task.root, "program_0")
    task_entry = await workspace.stat(task.root)
    if task_entry is None or task_entry.kind != "dir":
        raise ExtractionError(f"Task directory does not exist: {task.root}")
    reference_entry = await workspace.stat(reference_dir)
    if reference_entry is None or reference_entry.kind != "dir":
        raise ExtractionError(f"Task directory missing program_0: {task.root}")
    best_entry = await workspace.stat(best_program_dir)
    if best_entry is None or best_entry.kind != "dir":
        raise ExtractionError(f"Best program directory does not exist: {best_program_dir}")
    try:
        await load_program_spec(workspace, best_program_dir)
    except ProgramSpecError as exc:
        raise ExtractionError(
            f"Invalid program_spec.toml in best program {best_program_dir}: {exc}"
        ) from exc
    try:
        await resolve_entrypoint(workspace, best_program_dir)
    except ProgramSpecError as exc:
        raise ExtractionError(
            f"Cannot resolve entrypoint in best program {best_program_dir}: {exc}"
        ) from exc

    await workspace.remove(bundle_dir)
    await copy_task_inputs(workspace, task, bundle_dir, reference_dir)
    await clone_program(workspace, src_dir=best_program_dir, dst_dir=join(bundle_dir, "program_0"))

    best_manifest_path = join(results_dir, "best_program_manifest.json") if results_dir else None
    extraction_manifest: dict[str, Any] = {
        "extraction_version": 1,
        "extracted_at": utc_now(),
        "source_task_dir": task.root,
        "source_results_dir": results_dir,
        "source_best_manifest_path": (
            best_manifest_path
            if best_manifest_path and await workspace.exists(best_manifest_path)
            else None
        ),
        "source_best_program_dir": best_program_dir,
        "reference_program_dir": join(bundle_dir, "reference_program"),
        "extracted_program_dir": join(bundle_dir, "program_0"),
        "entrypoint_path": manifest.get("entrypoint_path"),
        "combined_score": manifest.get("combined_score"),
        "correct": manifest.get("correct"),
        "source_attempt_id": manifest.get("source_attempt_id"),
        "parallel_worker_index": manifest.get("parallel_worker_index"),
        "downstream_selection": dict(downstream_selection) if downstream_selection else None,
    }
    await workspace.write_bytes(
        join(bundle_dir, "extraction_manifest.json"),
        (json.dumps(extraction_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    if evidence_dir is not None:
        await stage_evidence(workspace, evidence_dir, bundle_dir, config)
    await validate_bundle(workspace, bundle_dir, task)
    return extraction_manifest


async def validate_bundle(workspace: Workspace, bundle_dir: str, task: TaskRef) -> None:
    """Check that the extracted bundle holds the entries later stages read."""
    evaluator = task.contract.evaluators.search if task.contract is not None else "evaluate.py"
    required = [evaluator if name == "evaluate.py" else name for name in REQUIRED_BUNDLE_ENTRIES]
    missing = [name for name in required if not await workspace.exists(join(bundle_dir, name))]
    if missing:
        raise BundleValidationError(f"Extracted bundle at {bundle_dir} is missing: {missing}")


# --- staging the evidence -----------------------------------------------------------------------


def _children(entries: Mapping[str, FileEntry], prefix: str) -> list[str]:
    """Return the sorted names of the direct children of ``prefix`` in a listing."""
    start = prefix + "/" if prefix else ""
    names: set[str] = set()
    for path in entries:
        if path.startswith(start) and path != prefix:
            names.add(path[len(start) :].split("/", 1)[0])
    return sorted(names)


def _bounded_children(entries: Mapping[str, FileEntry], prefix: str) -> list[str]:
    names = _children(entries, prefix)
    if len(names) > MAX_DIRECTORY_ENTRIES:
        raise ValueError("Evaluation evidence directory entries exceed safety limit")
    return names


async def _load_evidence(workspace: Workspace, source_root: str, relative: str) -> bool:
    """Validate a variant program's ``evaluation_evidence.json``; false when it has none."""
    program = join(source_root, relative)
    entry = await workspace.stat(join(program, EVIDENCE_FILENAME))
    if entry is None:
        return False
    if entry.kind != "file":
        raise ValueError(f"Evaluation evidence is not a regular file: {program}")
    parse_evaluation_evidence(await workspace.read_bytes(join(program, EVIDENCE_FILENAME)))
    return True


async def variant_evaluation_evidence(
    workspace: Workspace, source_root: str, entries: Mapping[str, FileEntry]
) -> list[str]:
    """Find only the fixed evidence filename in non-symlinked variant program directories."""
    candidates: list[str] = []
    programs = 0
    for variant in _bounded_children(entries, "variants"):
        variant_path = f"variants/{variant}"
        variant_entry = entries.get(variant_path)
        if variant_entry is not None and variant_entry.kind == "symlink":
            raise ValueError(f"Evaluation evidence variant contains symlink: {variant_path}")
        if variant_entry is None or variant_entry.kind != "dir":
            continue
        for program in _bounded_children(entries, f"{variant_path}/task_bundle"):
            if not _PROGRAM_DIR.fullmatch(program):
                continue
            program_path = f"{variant_path}/task_bundle/{program}"
            program_entry = entries.get(program_path)
            if program_entry is not None and program_entry.kind == "symlink":
                raise ValueError(f"Evaluation evidence program contains symlink: {program_path}")
            if program_entry is None or program_entry.kind != "dir":
                continue
            programs += 1
            if programs > MAX_PROGRAMS:
                raise ValueError("Evaluation evidence programs exceed safety limit")
            if await _load_evidence(workspace, source_root, program_path):
                candidates.append(f"{program_path}/{EVIDENCE_FILENAME}")
    return candidates


async def _validate_evidence_anchor(workspace: Workspace, source_root: str) -> None:
    path = join(source_root, "pipeline_summary.json")
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        raise BundleValidationError(f"Evidence pipeline_summary.json is missing or unsafe: {path}")
    if entry.size > MAX_EVIDENCE_SUMMARY_BYTES:
        raise BundleValidationError("Evidence pipeline_summary.json exceeds safety limit")
    try:
        summary = json.loads((await workspace.read_bytes(path)).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError("Evidence pipeline_summary.json is invalid") from exc
    if not isinstance(summary, dict) or summary.get("schema_version") != 1:
        raise BundleValidationError("Evidence pipeline_summary.json has an unsupported schema")
    total_variants = summary.get("total_variants")
    total_completed = summary.get("total_completed")
    entries = summary.get("variants")
    if (
        isinstance(total_variants, bool)
        or not isinstance(total_variants, int)
        or total_variants < 1
        or isinstance(total_completed, bool)
        or not isinstance(total_completed, int)
        or total_completed < 1
        or total_completed > total_variants
        or not isinstance(entries, list)
        or len(entries) != total_variants
    ):
        raise BundleValidationError("Evidence pipeline_summary.json is incomplete")
    completed = 0
    for item in entries:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("variant_id"), str)
            or item.get("status") not in {"completed", "failed", "timeout"}
        ):
            raise BundleValidationError("Evidence pipeline_summary.json is incomplete")
        completed += int(item["status"] == "completed")
    if completed != total_completed:
        raise BundleValidationError("Evidence pipeline_summary.json is inconsistent")


async def _evidence_candidates(workspace: Workspace, source_root: str) -> list[str]:
    entries = {entry.path: entry for entry in await workspace.list_tree(source_root)}
    candidates: set[str] = set()
    try:
        candidates.update(await variant_evaluation_evidence(workspace, source_root, entries))
    except (OSError, ValueError) as exc:
        raise BundleValidationError(f"Invalid evaluation evidence: {exc}") from exc
    for relative in (
        "README.md",
        "pipeline_summary.json",
        "generation/variant_bundle.json",
        "generation/variant_selection.json",
    ):
        candidate = entries.get(relative)
        if candidate is None:
            continue
        if candidate.kind == "symlink":
            raise BundleValidationError(f"Evidence source contains symlink: {relative}")
        candidates.add(relative)
    aggregate = entries.get("aggregated")
    if aggregate is None or aggregate.kind != "dir":
        raise BundleValidationError(
            f"Evidence aggregate is not a real directory: {join(source_root, 'aggregated')}"
        )
    task_directories = _children(entries, "aggregated")
    if len(task_directories) > MAX_EVIDENCE_TASK_DIRS:
        raise BundleValidationError("Evidence task directory count exceeds safety limit")
    for name in task_directories:
        directory = entries.get(f"aggregated/{name}")
        if directory is None or directory.kind != "dir":
            continue
        for filename in AGGREGATE_FILENAMES:
            candidate = entries.get(f"aggregated/{name}/{filename}")
            if candidate is None:
                continue
            if candidate.kind == "symlink":
                raise BundleValidationError(
                    f"Evidence source contains symlink: aggregated/{name}/{filename}"
                )
            candidates.add(f"aggregated/{name}/{filename}")
    return sorted(candidates)


async def stage_evidence(
    workspace: Workspace, source_root: str, bundle_dir: str, config: PosthocConfig
) -> dict[str, Any]:
    """Copy a bounded, analysis-only allowlist of a completed ablations bundle.

    Writes ``<bundle>/evidence/`` and ``<bundle>/evidence_manifest.json`` and returns the
    manifest.

    Raises:
        BundleValidationError: When the source is not a completed ablations directory or exceeds
            a limit.
    """
    source_entry = await workspace.stat(source_root)
    if source_entry is None or source_entry.kind != "dir":
        raise BundleValidationError(f"Evidence source is not a real directory: {source_root}")
    await _validate_evidence_anchor(workspace, source_root)
    destination_root = join(bundle_dir, "evidence")
    manifest_path = join(bundle_dir, "evidence_manifest.json")
    if await workspace.exists(destination_root):
        raise BundleValidationError(f"Extracted bundle reserves evidence/: {destination_root}")
    if await workspace.exists(manifest_path):
        raise BundleValidationError(
            f"Extracted bundle reserves evidence_manifest.json: {manifest_path}"
        )
    candidates = await _evidence_candidates(workspace, source_root)
    if len(candidates) > config.evidence_max_files:
        raise BundleValidationError("Evidence file count exceeds safety limit")
    total_bytes = 0
    files: dict[str, dict[str, object]] = {}
    staged: list[tuple[str, bytes]] = []
    for relative in candidates:
        entry = await workspace.stat(join(source_root, relative))
        if entry is None or entry.kind != "file":
            raise BundleValidationError(f"Evidence entry is not a regular file: {relative}")
        if entry.size > config.evidence_max_file_bytes:
            raise BundleValidationError(f"Evidence file exceeds safety limit: {relative}")
        data = await workspace.read_bytes(join(source_root, relative))
        total_bytes += len(data)
        if total_bytes > config.evidence_max_total_bytes:
            raise BundleValidationError(f"Evidence file or bundle exceeds safety limit: {relative}")
        if PurePosixPath(relative).name == EVIDENCE_FILENAME:
            try:
                parse_evaluation_evidence(data)
            except ValueError as exc:
                raise BundleValidationError(f"Invalid evaluation evidence: {relative}") from exc
        files[f"evidence/{relative}"] = {
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        staged.append((relative, data))
    manifest = {
        "manifest_version": 1,
        "source_role": "completed_ablations_bundle",
        "files": files,
        "total_bytes": total_bytes,
    }
    for relative, data in staged:
        await workspace.write_bytes(join(destination_root, relative), data)
    await workspace.write_bytes(
        manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest


async def verify_staged_evidence(
    workspace: Workspace, bundle_dir: str, config: PosthocConfig | None = None
) -> dict[str, Any]:
    """Validate the evidence manifest, tree shape, sizes and SHA-256 digests.

    Raises:
        BundleValidationError: With upstream's message for the first rule that fails.
    """
    limits = config or PosthocConfig()
    manifest_path = join(bundle_dir, "evidence_manifest.json")
    evidence_root = join(bundle_dir, "evidence")
    manifest_entry = await workspace.stat(manifest_path)
    if manifest_entry is None or manifest_entry.kind != "file":
        raise BundleValidationError(f"Evidence manifest is missing or unsafe: {manifest_path}")
    root_entry = await workspace.stat(evidence_root)
    if root_entry is None or root_entry.kind != "dir":
        raise BundleValidationError(f"Evidence directory is missing or unsafe: {evidence_root}")
    if manifest_entry.size > MAX_EVIDENCE_MANIFEST_BYTES:
        raise BundleValidationError("Evidence manifest exceeds safety limit")
    try:
        manifest = json.loads((await workspace.read_bytes(manifest_path)).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError("Evidence manifest is invalid") from exc
    files = manifest.get("files") if isinstance(manifest, dict) else None
    total_bytes = manifest.get("total_bytes") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("manifest_version") != 1
        or manifest.get("source_role") != "completed_ablations_bundle"
        or not isinstance(files, dict)
        or len(files) > limits.evidence_max_files
        or isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes < 0
        or total_bytes > limits.evidence_max_total_bytes
    ):
        raise BundleValidationError("Evidence manifest has an invalid schema")

    expected_paths: set[str] = set()
    expected_total = 0
    for manifest_file, metadata in files.items():
        relative = (
            PurePosixPath(manifest_file) if isinstance(manifest_file, str) else PurePosixPath()
        )
        if (
            not isinstance(manifest_file, str)
            or relative.is_absolute()
            or relative.as_posix() != manifest_file
            or not relative.parts
            or relative.parts[0] != "evidence"
            or ".." in relative.parts
            or not isinstance(metadata, dict)
        ):
            raise BundleValidationError("Evidence manifest contains an unsafe path")
        size = metadata.get("size")
        digest = metadata.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or size > limits.evidence_max_file_bytes
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in _HEX for character in digest)
        ):
            raise BundleValidationError("Evidence manifest contains invalid metadata")
        expected_paths.add(manifest_file)
        expected_total += size
    if expected_total != total_bytes:
        raise BundleValidationError("Evidence manifest total_bytes is inconsistent")

    tree = await workspace.list_tree(evidence_root)
    if len(tree) > MAX_EVIDENCE_TREE_ENTRIES:
        raise BundleValidationError("Evidence tree exceeds entry safety limit")
    actual_paths: set[str] = set()
    actual: dict[str, FileEntry] = {}
    for entry in tree:
        depth = len(entry.path.split("/"))
        if entry.kind == "symlink":
            raise BundleValidationError(f"Evidence tree contains symlink: {entry.path}")
        if entry.kind == "dir":
            if depth > MAX_EVIDENCE_TREE_DEPTH:
                raise BundleValidationError("Evidence tree exceeds depth safety limit")
        elif entry.kind == "file":
            if depth - 1 > MAX_EVIDENCE_TREE_DEPTH:
                raise BundleValidationError("Evidence tree exceeds depth safety limit")
            actual_paths.add(f"evidence/{entry.path}")
            actual[f"evidence/{entry.path}"] = entry
        else:
            raise BundleValidationError(f"Evidence tree contains a special file: {entry.path}")
    if actual_paths != expected_paths:
        raise BundleValidationError("Evidence tree does not match manifest")
    for manifest_file, metadata in files.items():
        entry = actual[manifest_file]
        if entry.size != metadata["size"]:
            raise BundleValidationError(f"Evidence file size mismatch: {manifest_file}")
        digest = (entry.sha256 or "").removeprefix("sha256:")
        if digest != metadata["sha256"]:
            raise BundleValidationError(f"Evidence file digest mismatch: {manifest_file}")
    return manifest


async def restore_verified_evidence(
    workspace: Workspace,
    source_bundle: str,
    target_bundle: str,
    config: PosthocConfig | None = None,
) -> None:
    """Restore pristine evidence after bootstrap and verify every staged byte."""
    source_manifest = join(source_bundle, "evidence_manifest.json")
    source_evidence = join(source_bundle, "evidence")
    if not await workspace.exists(source_manifest) and not await workspace.exists(source_evidence):
        return
    manifest = await verify_staged_evidence(workspace, source_bundle, config)
    destination_root = join(target_bundle, "evidence")
    destination_manifest = join(target_bundle, "evidence_manifest.json")
    await workspace.remove(destination_root)
    await workspace.remove(destination_manifest)
    for manifest_path, metadata in sorted(manifest["files"].items()):
        relative = PurePosixPath(manifest_path)
        data = await workspace.read_bytes(join(source_bundle, manifest_path))
        if len(data) != metadata["size"] or hashlib.sha256(data).hexdigest() != metadata["sha256"]:
            raise BundleValidationError(f"Evidence file does not match manifest: {manifest_path}")
        await workspace.write_bytes(join(target_bundle, *relative.parts), data)
    await workspace.write_bytes(
        destination_manifest, (json_file_text_sorted(manifest)).encode("utf-8")
    )


def json_file_text_sorted(value: Any) -> str:
    """Return JSON as upstream writes the evidence manifest (sorted keys, indent 2)."""
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


__all__ = [
    "AGGREGATE_FILENAMES",
    "BundleValidationError",
    "ExtractionError",
    "WorkspacePathError",
    "copy_task_inputs",
    "extract_bundle",
    "json_file_text",
    "restore_verified_evidence",
    "sealed_task_paths",
    "stage_evidence",
    "validate_bundle",
    "variant_evaluation_evidence",
    "verify_staged_evidence",
]
