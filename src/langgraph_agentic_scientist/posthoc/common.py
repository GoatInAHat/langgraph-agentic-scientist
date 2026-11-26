"""Helpers shared by the stages of the post-hoc phase (upstream ``posthoc_pipeline/common.py``).

Upstream works on directories of the host; the port works on workspace paths. The rules are
upstream's: which task files a bundle carries, which files are checkpoints, how a checkpoint is
validated without deserialising model objects, how the primary metric is inferred and compared.

Checkpoint validation reads the file through the workspace protocol, which has no ranged reads, so
a checkpoint larger than :data:`CHECKPOINT_VALIDATION_MAX_BYTES` is accepted after the non-empty
check only (upstream's validation is best effort as well).
"""

from __future__ import annotations

import io
import json
import pickletools
import struct
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import PurePosixPath
from typing import Any

from ..core.errors import ProgramSpecError, WorkspacePathError
from ..evaluation.program_clone import clone_program
from ..evaluation.program_spec import resolve_entrypoint
from ..workspace.paths import join, safe_relpath
from ..workspace.protocol import FileEntry, Workspace

CHECKPOINT_VALIDATION_MAX_BYTES = 4 * 1024 * 1024 * 1024
"""Largest checkpoint that is read into memory for structural validation."""

TASK_COPY_EXCLUDE_DIRS: frozenset[str] = frozenset(
    {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".nox",
        ".venv",
        "node_modules",
        ".git",
    }
)

TASK_COPY_EXCLUDE_FILES: frozenset[str] = frozenset(
    {
        "correct.json",
        "metrics.json",
        "rubric_status.json",
        "stdout.log",
        "stderr.log",
    }
)

TASK_COPY_EXCLUDE_SUFFIXES: tuple[str, ...] = (".log", ".out", ".err")

CHECKPOINT_SUFFIXES: tuple[str, ...] = (
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".joblib",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
)

CHECKPOINT_NAME_HINTS: tuple[str, ...] = (
    "checkpoint",
    "ckpt",
    "model",
    "weights",
    "state",
    "best",
    "snapshot",
)

CHECKPOINT_NAME_EXCLUDES: tuple[str, ...] = (
    "train",
    "test",
    "data",
    "manifest",
    "split",
    "zca",
    "inv_sqrt",
    "x_mean",
    "labels",
    "images",
    "classes",
)

PRIMARY_METRIC_PRIORITY: tuple[str, ...] = (
    "val_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "accuracy",
    "acc",
    "f1",
    "auc",
    "reward",
    "score",
    "return",
    "lower_bound",
    "upper_bound",
    "loss",
    "perplexity",
)

PRIMARY_METRIC_EXCLUDE: frozenset[str] = frozenset(
    {
        "combined_score",
        "correct",
        "error",
        "terminated_early",
        "training_time",
        "time",
        "time_mean",
        "wall_time",
        "elapsed_time",
        "seed",
        "split_seed",
        "train_examples",
        "val_examples",
        "test_examples",
    }
)


# --- files --------------------------------------------------------------------------------------


async def read_json(workspace: Workspace, path: str) -> dict[str, Any] | None:
    """Read a JSON object file; ``None`` when absent, unsafe, invalid or not an object."""
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        return None
    try:
        value = json.loads((await workspace.read_bytes(path)).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


async def read_text_or_empty(workspace: Workspace, path: str) -> str:
    """Read a UTF-8 text file; the empty string when it does not exist."""
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        return ""
    return (await workspace.read_bytes(path)).decode("utf-8")


def json_file_text(data: Any) -> str:
    """Return upstream's JSON file text (``indent=2`` and a final newline)."""
    return json.dumps(data, indent=2) + "\n"


async def write_json(workspace: Workspace, path: str, data: Any) -> None:
    """Write a JSON file the way upstream's ``write_json`` does."""
    await workspace.write_bytes(path, json_file_text(data).encode("utf-8"))


async def copy_files(
    workspace: Workspace, src: str, dst: str, entries: Iterable[FileEntry]
) -> list[str]:
    """Copy the regular files among ``entries`` (paths relative to ``src``) to ``dst``.

    Directories are implied by their files; symlinks and special files are not carried (the
    workspace protocol has no operation that creates them). Returns the paths of the symlinks
    that were not carried.
    """
    skipped: list[str] = []
    for entry in entries:
        if entry.kind == "symlink":
            skipped.append(entry.path)
            continue
        if entry.kind != "file":
            continue
        await workspace.write_bytes(
            join(dst, entry.path), await workspace.read_bytes(join(src, entry.path))
        )
    return skipped


def _top_level(entries: Sequence[FileEntry]) -> dict[str, FileEntry]:
    """Return the top-level entries of a listing by name (directories implied by files)."""
    top: dict[str, FileEntry] = {}
    for entry in entries:
        head, _, rest = entry.path.partition("/")
        if not rest:
            top[head] = entry
        elif head not in top:
            top[head] = FileEntry(path=head, kind="dir")
    return top


async def copy_task_files(
    workspace: Workspace, src_dir: str, dst_dir: str, *, extra_skip: Iterable[str] = ()
) -> list[str]:
    """Copy task-level files of an analysis-ready bundle into a task bundle (upstream rules).

    The top-level entries named ``program_*`` or ``reference_program`` are skipped, as are cache
    directories, evaluator result files, and ``.log``, ``.out`` and ``.err`` files. Returns the
    symlinks that were not carried.
    """
    skip = set(extra_skip)
    entries = await workspace.list_tree(src_dir)
    skipped_links: list[str] = []
    selected: list[FileEntry] = []
    for name, top in sorted(_top_level(entries).items()):
        if name.startswith("program_") or name == "reference_program" or name in skip:
            continue
        if top.kind == "dir" and name in TASK_COPY_EXCLUDE_DIRS:
            continue
        if name in TASK_COPY_EXCLUDE_FILES or name.endswith(TASK_COPY_EXCLUDE_SUFFIXES):
            continue
        if top.kind == "other":
            continue
        if top.kind == "dir":
            selected.extend(
                entry
                for entry in entries
                if entry.path == name or entry.path.startswith(name + "/")
            )
        else:
            selected.append(top)
    skipped_links.extend(await copy_files(workspace, src_dir, dst_dir, selected))
    return skipped_links


async def remove_stale_program_files(workspace: Workspace, program_dir: str) -> None:
    """Remove evaluator-managed files from a cloned program directory."""
    for entry in await workspace.list_tree(program_dir):
        if entry.kind != "file":
            continue
        name = PurePosixPath(entry.path).name
        if name in TASK_COPY_EXCLUDE_FILES or name.endswith(TASK_COPY_EXCLUDE_SUFFIXES):
            await workspace.remove(join(program_dir, entry.path))


async def clone_program_bundle(workspace: Workspace, source_dir: str, target_dir: str) -> list[str]:
    """Clone a program with its copy rules, then drop stale evaluator files.

    Returns the symlinks that were not carried.
    """
    await workspace.remove(target_dir)
    report = await clone_program(workspace, src_dir=source_dir, dst_dir=target_dir)
    await remove_stale_program_files(workspace, target_dir)
    return report.skipped_symlinks


async def list_program_dirs(workspace: Workspace, root_dir: str) -> list[str]:
    """Return the ``program_<n>`` directories directly under ``root_dir``, ordered by index."""
    entries = await workspace.list_tree(root_dir) if await workspace.exists(root_dir) else []
    found: dict[int, str] = {}
    for entry in entries:
        if entry.kind != "dir" or "/" in entry.path:
            continue
        name = entry.path
        if name.startswith("program_") and name[len("program_") :].isdigit():
            found[int(name[len("program_") :])] = join(root_dir, name)
    return [found[index] for index in sorted(found)]


async def copy_selected_runtime_artifacts(
    workspace: Workspace, source_program_dir: str, target_program_dir: str
) -> dict[str, list[str]]:
    """Copy post-run artifacts of the original best program into the bootstrap bundle."""
    copied: dict[str, list[str]] = {"files": [], "directories": [], "checkpoints": []}
    entries = await workspace.list_tree(source_program_dir)
    by_path = {entry.path: entry for entry in entries}

    for filename in ("metrics.json", "correct.json", "_results.json", "rubric_status.json"):
        entry = by_path.get(filename)
        if entry is None or entry.kind != "file":
            continue
        await workspace.write_bytes(
            join(target_program_dir, filename),
            await workspace.read_bytes(join(source_program_dir, filename)),
        )
        copied["files"].append(filename)

    for dirname in ("analysis", "figures", "plots", "artifacts"):
        entry = by_path.get(dirname)
        if entry is None or entry.kind != "dir":
            continue
        await workspace.remove(join(target_program_dir, dirname))
        await copy_files(
            workspace,
            join(source_program_dir, dirname),
            join(target_program_dir, dirname),
            [
                FileEntry(
                    path=item.path[len(dirname) + 1 :],
                    kind=item.kind,
                    size=item.size,
                )
                for item in entries
                if item.path.startswith(dirname + "/")
            ],
        )
        copied["directories"].append(dirname)

    for relative, _size in await find_checkpoint_candidates(workspace, source_program_dir):
        await workspace.write_bytes(
            join(target_program_dir, relative),
            await workspace.read_bytes(join(source_program_dir, relative)),
        )
        copied["checkpoints"].append(relative)

    return copied


# --- checkpoints --------------------------------------------------------------------------------


def _candidate_paths_from_results_json(results: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key, value in results.items():
        key_text = str(key).lower()
        if not isinstance(value, str):
            continue
        if "path" in key_text or "checkpoint" in key_text or "model" in key_text:
            values.append(value)
    return values


def _is_checkpoint_candidate(relative: str) -> bool:
    path = PurePosixPath(relative)
    name = path.name.lower()
    if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        return False
    if any(exclude in name for exclude in CHECKPOINT_NAME_EXCLUDES):
        return False
    if any(hint in name for hint in CHECKPOINT_NAME_HINTS):
        return True
    parent_text = path.parent.as_posix().lower()
    return "checkpoint" in parent_text or "ckpt" in parent_text or "weights" in parent_text


async def find_checkpoint_candidates(
    workspace: Workspace, program_dir: str
) -> list[tuple[str, int]]:
    """Return ``(path relative to program_dir, size)`` of every checkpoint candidate.

    First the relative paths that ``_results.json`` names (a path inside the program, a regular
    file), then every file whose name and suffix look like a checkpoint, in path order.
    """
    entry = await workspace.stat(program_dir)
    if entry is None or entry.kind != "dir":
        return []
    entries = {item.path: item for item in await workspace.list_tree(program_dir)}
    candidates: list[tuple[str, int]] = []
    seen: set[str] = set()

    results = await read_json(workspace, join(program_dir, "_results.json")) or {}
    for value in _candidate_paths_from_results_json(results):
        if value.startswith("/"):
            continue
        try:
            relative = safe_relpath(value, allow_root=False)
        except WorkspacePathError:
            continue
        item = entries.get(relative)
        if item is not None and item.kind == "file" and relative not in seen:
            seen.add(relative)
            candidates.append((relative, item.size))

    for relative in sorted(entries):
        item = entries[relative]
        if item.kind != "file":
            continue
        if _is_checkpoint_candidate(relative) and relative not in seen:
            seen.add(relative)
            candidates.append((relative, item.size))
    return candidates


def _validate_npy(data: bytes) -> None:
    try:
        import numpy as np
    except ImportError:
        if not data.startswith(b"\x93NUMPY"):
            raise ValueError("not an npy file") from None
        return
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported npy version {version}")


def _validate_checkpoint_bytes(suffix: str, data: bytes) -> tuple[bool, str | None]:
    if suffix == ".json":
        json.loads(data.decode("utf-8"))
    elif suffix in {".pkl", ".pickle", ".joblib"}:
        for _ in pickletools.genops(io.BytesIO(data)):
            pass
    elif suffix == ".npy":
        _validate_npy(data)
    elif suffix == ".npz":
        if not zipfile.is_zipfile(io.BytesIO(data)):
            return False, "checkpoint archive is invalid"
    elif suffix == ".safetensors":
        header_length_bytes = data[:8]
        if len(header_length_bytes) != 8:
            return False, "safetensors checkpoint header is invalid"
        header_length = struct.unpack("<Q", header_length_bytes)[0]
        if header_length <= 1 or header_length > len(data) - 8:
            return False, "safetensors checkpoint header is invalid"
        json.loads(data[8 : 8 + header_length].decode("utf-8"))
    elif suffix in {".pt", ".pth", ".ckpt", ".bin"}:
        if zipfile.is_zipfile(io.BytesIO(data)):
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if not archive.namelist():
                    return False, "checkpoint archive is empty"
        else:
            for _ in pickletools.genops(io.BytesIO(data)):
                pass
    return True, None


async def validate_checkpoint_loadable(workspace: Workspace, path: str) -> tuple[bool, str | None]:
    """Validate a checkpoint's structure without deserialising model objects.

    Returns ``(loadable, error)``. See the module docstring for the size limit.
    """
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        return False, "checkpoint file missing"
    if entry.size <= 0:
        return False, "checkpoint file is empty"
    if entry.size > CHECKPOINT_VALIDATION_MAX_BYTES:
        return True, None
    try:
        data = await workspace.read_bytes(path)
        return _validate_checkpoint_bytes(PurePosixPath(path).suffix.lower(), data)
    except Exception as exc:  # best effort validation, as upstream
        return False, f"checkpoint not loadable: {exc}"


# --- primary metric -----------------------------------------------------------------------------


def infer_primary_metric(
    *,
    metrics: dict[str, Any] | None,
    results: dict[str, Any] | None = None,
    requested_key: str | None = None,
) -> dict[str, Any]:
    """Resolve the primary metric, preferring task metrics over ``combined_score``.

    Raises:
        RuntimeError: When no numeric metric can be resolved.
    """
    metrics = metrics or {}
    results = results or {}

    if requested_key:
        resolved = _lookup_metric(requested_key, metrics, results)
        if resolved is not None:
            key, value, source = resolved
            return _metric_info(key, value, source)

    for key in PRIMARY_METRIC_PRIORITY:
        resolved = _lookup_metric(key, metrics, results)
        if resolved is not None:
            metric_key, value, source = resolved
            if metric_key != "combined_score":
                return _metric_info(metric_key, value, source)

    for source_name, payload in (("metrics.json", metrics), ("_results.json", results)):
        for key, value in payload.items():
            if key in PRIMARY_METRIC_EXCLUDE:
                continue
            if not isinstance(value, int | float):
                continue
            return _metric_info(str(key), float(value), source_name)

    if isinstance(metrics.get("combined_score"), int | float):
        return _metric_info("combined_score", float(metrics["combined_score"]), "metrics.json")

    raise RuntimeError(
        "Unable to resolve a numeric primary metric from metrics.json or _results.json"
    )


def _lookup_metric(
    key: str,
    metrics: dict[str, Any],
    results: dict[str, Any],
) -> tuple[str, float, str] | None:
    for source_name, payload in (("metrics.json", metrics), ("_results.json", results)):
        value = payload.get(key)
        if isinstance(value, int | float):
            return key, float(value), source_name
    return None


def _metric_info(key: str, value: float, source: str) -> dict[str, Any]:
    lower_better_markers = ("loss", "error", "time", "latency", "perplexity", "upper_bound")
    higher_is_better = not any(marker in key.lower() for marker in lower_better_markers)
    return {
        "key": key,
        "value": value,
        "source": source,
        "higher_is_better": higher_is_better,
    }


def compare_metric_guardrail(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Compare the candidate's primary metric against the baseline's within ``tolerance``."""
    if baseline["key"] != candidate["key"]:
        return {
            "passed": False,
            "reason": (
                f"primary metric mismatch: baseline={baseline['key']} candidate={candidate['key']}"
            ),
            "baseline": baseline,
            "candidate": candidate,
            "tolerance": tolerance,
        }

    higher_is_better = bool(baseline["higher_is_better"])
    baseline_value = float(baseline["value"])
    candidate_value = float(candidate["value"])
    if higher_is_better:
        passed = candidate_value + tolerance >= baseline_value
    else:
        passed = candidate_value <= baseline_value + tolerance

    delta = candidate_value - baseline_value
    return {
        "passed": passed,
        "reason": None if passed else "primary metric regressed beyond tolerance",
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "tolerance": tolerance,
    }


# --- expected outputs ---------------------------------------------------------------------------


async def validate_expected_outputs(
    workspace: Workspace, root_dir: str, expected_outputs: Iterable[str]
) -> dict[str, Any]:
    """Check that every expected output is a safe, regular file below ``root_dir``."""
    found: list[str] = []
    missing: list[str] = []
    for relpath in expected_outputs:
        try:
            safe = not relpath.startswith("/") and bool(safe_relpath(relpath, allow_root=False))
        except WorkspacePathError:
            safe = False
        entry = await workspace.stat(join(root_dir, relpath)) if safe else None
        if entry is not None and entry.kind == "file":
            found.append(relpath)
        else:
            missing.append(relpath)
    return {"found": found, "missing": missing, "all_present": not missing}


async def resolve_program_entrypoint(workspace: Workspace, program_dir: str) -> str:
    """Return the workspace path of a program's entrypoint file.

    Raises:
        RuntimeError: When the program spec or the entrypoint is invalid (upstream's message).
    """
    try:
        return join(program_dir, await resolve_entrypoint(workspace, program_dir))
    except ProgramSpecError as exc:
        raise RuntimeError(f"Failed to resolve entrypoint for {program_dir}: {exc}") from exc


__all__ = [
    "CHECKPOINT_VALIDATION_MAX_BYTES",
    "clone_program_bundle",
    "compare_metric_guardrail",
    "copy_files",
    "copy_selected_runtime_artifacts",
    "copy_task_files",
    "find_checkpoint_candidates",
    "infer_primary_metric",
    "json_file_text",
    "list_program_dirs",
    "read_json",
    "read_text_or_empty",
    "remove_stale_program_files",
    "resolve_program_entrypoint",
    "validate_checkpoint_loadable",
    "validate_expected_outputs",
    "write_json",
]
