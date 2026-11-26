"""The trusted clone ``program_<idx-1>`` -> ``program_<idx>`` (upstream ``program_clone.py``).

Upstream agents run ``python -I -m src.investigation.program_clone --from ... --to ...``. The
port gives them the ``clone_program`` tool, which calls :func:`clone_program`; the rules are
upstream's: adjacent indices in the same experiment directory; for every source after
``program_0``, refuse until canonical evaluation and mirrored rubric artifacts prove the previous
iteration is complete and has not already passed; copy according to the source's
``program_spec.toml`` copy rules, never carrying evaluator artifacts.

File access goes through a :class:`~langgraph_agentic_scientist.workspace.Workspace`. The
workspace protocol has no operation that creates symlinks, so a symlink inside a program is
not carried by the port's clone; it is reported instead (upstream copies it as a symlink).
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from ..workspace.paths import safe_relpath
from .evidence import EVIDENCE_FILENAME
from .program_spec import load_program_spec

if TYPE_CHECKING:
    from ..workspace.protocol import Workspace

NONOVERRIDABLE_EXCLUDES: frozenset[str] = frozenset(
    {
        ".harness_results",
        ".evaluation_pending",
        ".metaevo_bbob_official",
        ".metaevo_official",
        "_analysis_extras.json",
        "_results.json",
        "analysis",
        "metrics.json",
        "correct.json",
        EVIDENCE_FILENAME,
        "rubric_status.json",
        "stdout.log",
        "stderr.log",
        "evaluation_manifest.json",
        "harness_evaluate_stdout.log",
        "harness_evaluate_stderr.log",
    }
)
"""Evaluator artifacts are managed by the pipeline and must not be carried forward."""

PROGRAM_DIR_PATTERN = re.compile(r"program_(\d+)")
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
SUPPORTED_EVALUATION_PRODUCERS = frozenset(
    {"harness_eval", "isolated_evaluator", "sandbox_host_eval"}
)
REQUIRED_SOURCE_ARTIFACTS = (
    "metrics.json",
    "correct.json",
    "evaluation_manifest.json",
    "rubric_status.json",
)


@dataclass
class CloneReport:
    """What a clone copied and what it could not carry."""

    copied: list[str] = field(default_factory=list)
    skipped_symlinks: list[str] = field(default_factory=list)


def _join(parent: str, name: str) -> str:
    return f"{parent}/{name}" if parent else name


async def _read_regular_json(workspace: Workspace, path: str) -> dict[str, Any]:
    name = PurePosixPath(path).name
    entry = await workspace.stat(path)
    if entry is None:
        raise ValueError(f"Candidate source not ready; missing required artifact: {name}")
    if entry.kind != "file":
        raise ValueError(f"Candidate source not ready; artifact must be a regular file: {name}")
    if entry.size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Candidate source not ready; artifact is too large: {name}")
    try:
        value = json.loads((await workspace.read_bytes(path)).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Candidate source not ready; invalid JSON artifact: {name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Candidate source not ready; artifact must contain a JSON object: {name}")
    return value


async def evaluation_results_complete(workspace: Workspace, program_dir: str) -> bool:
    """Require a valid completion manifest when an evaluator created one (upstream rule)."""
    if await workspace.stat(_join(program_dir, ".evaluation_pending")) is not None:
        return False
    manifest_path = _join(program_dir, "evaluation_manifest.json")
    entry = await workspace.stat(manifest_path)
    if entry is None:
        return True
    if entry.kind != "file":
        return False
    try:
        manifest = json.loads((await workspace.read_bytes(manifest_path)).decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    program_name = PurePosixPath(program_dir).name
    if manifest.get("producer") == "isolated_evaluator" or "published_summaries" in manifest:
        summaries = manifest.get("published_summaries")
        return (
            manifest.get("schema_version") == 1
            and manifest.get("returncode") == 0
            and isinstance(manifest.get("output_dir"), str)
            and bool(manifest["output_dir"].strip())
            and isinstance(summaries, list)
            and set(summaries) == {"metrics.json", "correct.json"}
        )
    if manifest.get("producer") == "sandbox_host_eval" or (
        manifest.get("execution_boundary") == "sandbox" and "returncode" in manifest
    ):
        return (
            manifest.get("producer") in (None, "sandbox_host_eval")
            and manifest.get("execution_boundary") == "sandbox"
            and manifest.get("program_id") == program_name
            and manifest.get("returncode") == 0
            and isinstance(manifest.get("command"), list)
            and bool(manifest["command"])
            and isinstance(manifest.get("started_at"), str)
            and isinstance(manifest.get("finished_at"), str)
        )
    required_harness_fields = {
        "attempt_id",
        "idea_id",
        "program_id",
        "command",
        "started_at",
        "finished_at",
        "exit_code",
    }
    return (
        manifest.get("producer") in (None, "harness_eval")
        and required_harness_fields <= manifest.keys()
        and manifest.get("program_id") == program_name
        and manifest.get("exit_code") == 0
        and isinstance(manifest.get("command"), list)
        and bool(manifest["command"])
    )


async def require_candidate_source_ready(
    workspace: Workspace, *, src_dir: str, dst_dir: str
) -> None:
    """Refuse a clone unless the source iteration is complete (upstream readiness rules).

    Raises:
        ValueError: With upstream's message.
    """
    src_dir = safe_relpath(src_dir, allow_root=False)
    dst_dir = safe_relpath(dst_dir, allow_root=False)
    src = PurePosixPath(src_dir)
    dst = PurePosixPath(dst_dir)
    dst_match = PROGRAM_DIR_PATTERN.fullmatch(dst.name)
    if dst_match is None:
        raise ValueError("Candidate clone destination must be named program_<index>")
    dst_index = int(dst_match.group(1))
    if dst_index == 0:
        return
    src_match = PROGRAM_DIR_PATTERN.fullmatch(src.name)
    if src_match is None:
        raise ValueError("Candidate clone source must be named program_<index>")
    if src.parent != dst.parent:
        raise ValueError("Candidate clones must remain in the same experiment directory")
    src_index = int(src_match.group(1))
    if src_index + 1 != dst_index:
        raise ValueError(f"Candidate clone must use adjacent indices: {src.name} -> {dst.name}")
    if src_index == 0:
        return

    artifacts = {
        filename: await _read_regular_json(workspace, _join(src_dir, filename))
        for filename in REQUIRED_SOURCE_ARTIFACTS
    }
    manifest = artifacts["evaluation_manifest.json"]
    if manifest.get("producer") not in SUPPORTED_EVALUATION_PRODUCERS:
        raise ValueError("Candidate source evaluation_manifest.json has an unsupported producer")
    if manifest.get("program_id") != src.name:
        raise ValueError(
            f"Candidate source evaluation_manifest.json program_id must match {src.name}"
        )
    if not await evaluation_results_complete(workspace, src_dir):
        raise ValueError("Candidate source not ready; evaluation manifest is incomplete")
    score = artifacts["metrics.json"].get("combined_score")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(float(score))
    ):
        raise ValueError("Candidate source metrics.json combined_score must be finite and numeric")
    correct = artifacts["correct.json"].get("correct")
    if not isinstance(correct, bool):
        raise ValueError("Candidate source correct.json correct must be boolean")
    rubric = artifacts["rubric_status.json"]
    attempt_id = rubric.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise ValueError("Candidate source rubric_status.json needs an attempt_id")
    idea_id = rubric.get("idea_id")
    if not isinstance(idea_id, str) or not idea_id.strip():
        raise ValueError("Candidate source rubric_status.json needs an idea_id")
    if not isinstance(rubric.get("overall_pass"), bool):
        raise ValueError("Candidate source rubric_status.json overall_pass must be boolean")
    rubric_entries = rubric.get("rubrics")
    if not isinstance(rubric_entries, list) or not rubric_entries:
        raise ValueError("Candidate source rubric_status.json needs rubric entries")
    for entry in rubric_entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("id"), str)
            or not entry["id"].strip()
            or not isinstance(entry.get("pass"), bool)
            or not isinstance(entry.get("evidence"), str)
        ):
            raise ValueError("Candidate source rubric_status.json contains an invalid rubric entry")
    derived_overall_pass = correct and all(entry["pass"] for entry in rubric_entries)
    if rubric["overall_pass"] != derived_overall_pass:
        raise ValueError("Candidate source rubric_status.json has inconsistent overall_pass")
    if derived_overall_pass:
        raise ValueError(
            "Candidate source already passes all rubrics; do not create another candidate"
        )
    experiment_dir = "" if str(src.parent) == "." else str(src.parent)
    latest_rubric = await _read_regular_json(workspace, _join(experiment_dir, "rubric_status.json"))
    if latest_rubric != artifacts["rubric_status.json"]:
        raise ValueError(
            "Candidate source not ready; root rubric_status.json does not match "
            f"{src.name}/rubric_status.json"
        )


def _matches_any(patterns: tuple[str, ...], relpath: str) -> bool:
    return any(fnmatch.fnmatchcase(relpath, pattern) for pattern in patterns)


def _excluded_root(relpath: str) -> bool:
    root_name = relpath.split("/", 1)[0]
    return root_name in NONOVERRIDABLE_EXCLUDES or root_name.startswith(".evaluation_stage_")


def should_copy(*, relpath: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    """Upstream ``_should_copy``: non-overridable excludes, then include, then exclude."""
    if _excluded_root(relpath):
        return False
    if include and _matches_any(include, relpath):
        return True
    return not (exclude and _matches_any(exclude, relpath))


def should_copy_dir(*, relpath: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    """Upstream ``_should_copy_dir``: directory patterns like ``node_modules/**`` match too."""
    relpath_dir = f"{relpath}/"
    if _excluded_root(relpath):
        return False
    if include and (_matches_any(include, relpath) or _matches_any(include, relpath_dir)):
        return True
    return not (exclude and (_matches_any(exclude, relpath) or _matches_any(exclude, relpath_dir)))


async def clone_program(workspace: Workspace, *, src_dir: str, dst_dir: str) -> CloneReport:
    """Clone a program directory following its ``program_spec.toml`` copy rules.

    Raises:
        FileNotFoundError: When the source is not a directory.
        FileExistsError: When the destination exists and is not empty.
        ValueError: When source and destination overlap.
        ProgramSpecError: When the source spec is missing or invalid.
    """
    src_dir = safe_relpath(src_dir, allow_root=False)
    dst_dir = safe_relpath(dst_dir, allow_root=False)
    source_entry = await workspace.stat(src_dir)
    if source_entry is None or source_entry.kind != "dir":
        raise FileNotFoundError(f"Source program directory not found: {src_dir}")
    src_parts = PurePosixPath(src_dir).parts
    dst_parts = PurePosixPath(dst_dir).parts
    shared = min(len(src_parts), len(dst_parts))
    if src_parts[:shared] == dst_parts[:shared]:
        raise ValueError("Source and destination program directories must not overlap")
    spec = await load_program_spec(workspace, src_dir)
    destination_entry = await workspace.stat(dst_dir)
    if destination_entry is not None and (
        destination_entry.kind != "dir" or await workspace.list_tree(dst_dir)
    ):
        raise FileExistsError(f"Destination directory is not empty: {dst_dir}")
    include = spec.copy_rules.include
    exclude = spec.copy_rules.exclude
    report = CloneReport()
    pruned: list[str] = []
    for entry in await workspace.list_tree(src_dir):
        relpath = entry.path
        if any(relpath.startswith(prefix + "/") for prefix in pruned):
            continue
        if entry.kind == "dir":
            if (
                not include
                and (_matches_any(exclude, f"{relpath}/") or _matches_any(exclude, relpath))
            ) or _excluded_root(relpath):
                pruned.append(relpath)
            continue
        if entry.kind == "symlink":
            if should_copy(relpath=relpath, include=include, exclude=exclude):
                report.skipped_symlinks.append(relpath)
            continue
        if entry.kind != "file" or not should_copy(
            relpath=relpath, include=include, exclude=exclude
        ):
            continue
        data = await workspace.read_bytes(_join(src_dir, relpath))
        await workspace.write_bytes(_join(dst_dir, relpath), data)
        report.copied.append(relpath)
    if not report.copied and destination_entry is None:
        # Upstream creates the destination directory even when nothing is copied.
        await workspace.write_bytes(_join(dst_dir, ".keep"), b"")
        await workspace.remove(_join(dst_dir, ".keep"))
    return report


__all__ = [
    "NONOVERRIDABLE_EXCLUDES",
    "CloneReport",
    "clone_program",
    "evaluation_results_complete",
    "require_candidate_source_ready",
    "should_copy",
    "should_copy_dir",
]
