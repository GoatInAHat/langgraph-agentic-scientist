"""Post-hoc ranking (R29): collecting the analyses, upstream's validation, and the merge.

Ports ``collect_posthoc_analyses``, ``validate_posthoc_ranking`` and
``augment_paper_data_with_posthoc`` of ``paper_data_posthoc.py``. Upstream discovers the analyses
on disk under ``<results>/posthoc``; the port reads them from the post-hoc phase's typed result
(:class:`~langgraph_agentic_scientist.core.results.PosthocResult`): each passed analysis's
directory holds ``analysis_summary.md``, ``interestingness_rubric.json`` and
``analysis_manifest.json`` as upstream's selected attempt does. The size bounds are upstream's.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from ..core.results import PosthocResult
from ..workspace.paths import join
from ..workspace.protocol import Workspace

MAX_POSTHOC_FILE_BYTES = 5 * 1024 * 1024
MAX_POSTHOC_ANALYSIS_BYTES = 20 * 1024 * 1024
MAX_POSTHOC_CONTEXT_BYTES = 50 * 1024 * 1024
MAX_POSTHOC_ANALYSES = 16
MAX_POSTHOC_DATA_FILES = 32
MAX_POSTHOC_CSV_ROWS = 10_000
MAX_POSTHOC_CSV_COLUMNS = 100
MAX_POSTHOC_CSV_CELL_CHARS = 10_000
CONTROL_JSON_NAMES = frozenset(
    {
        "active_idea.json",
        "active_rubrics.json",
        "analysis_manifest.json",
        "analysis_spec.json",
        "attempt_result.json",
        "interestingness_rubric.json",
        "rubric_contract.json",
        "rubric_status.json",
        "target_rubrics.json",
    }
)


async def _read_small(workspace: Workspace, path: str) -> bytes | None:
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file" or entry.size > MAX_POSTHOC_FILE_BYTES:
        return None
    return await workspace.read_bytes(path)


async def _read_json(workspace: Workspace, path: str) -> dict[str, Any] | None:
    data = await _read_small(workspace, path)
    if data is None:
        return None
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


def _safe_relative(relpath: object) -> str | None:
    if not isinstance(relpath, str) or not relpath:
        return None
    path = PurePosixPath(relpath)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path.as_posix()


async def _manifest_data(
    workspace: Workspace, directory: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    outputs = manifest.get("outputs", {})
    files = outputs.get("data_files", []) if isinstance(outputs, Mapping) else []
    data: dict[str, Any] = {}
    total = 0
    for raw in list(files if isinstance(files, list) else [])[:MAX_POSTHOC_DATA_FILES]:
        relative = _safe_relative(raw)
        if relative is None:
            continue
        suffix = PurePosixPath(relative).suffix.lower()
        if suffix not in {".json", ".csv"}:
            continue
        content = await _read_small(workspace, join(directory, relative))
        if content is None:
            continue
        total += len(content)
        if total > MAX_POSTHOC_ANALYSIS_BYTES:
            break
        try:
            text = content.decode("utf-8")
            if suffix == ".json":
                payload: Any = json.loads(text)
            else:
                reader = csv.DictReader(io.StringIO(text, newline=""))
                if len(reader.fieldnames or []) > MAX_POSTHOC_CSV_COLUMNS:
                    continue
                rows: list[dict[str, Any]] = []
                for index, row in enumerate(reader):
                    if index >= MAX_POSTHOC_CSV_ROWS or any(
                        len(str(value)) > MAX_POSTHOC_CSV_CELL_CHARS for value in row.values()
                    ):
                        rows = []
                        break
                    rows.append(row)
                payload = rows
        except (UnicodeDecodeError, ValueError, csv.Error, RecursionError):
            continue
        data[str(raw)] = payload
    return data


async def _directory_data(workspace: Workspace, directory: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    total = 0
    count = 0
    for entry in await workspace.list_tree(directory):
        if entry.kind != "file" or "/" in entry.path or not entry.path.endswith(".json"):
            continue
        if entry.path in CONTROL_JSON_NAMES:
            continue
        count += 1
        total += entry.size
        if count > MAX_POSTHOC_DATA_FILES or total > MAX_POSTHOC_ANALYSIS_BYTES:
            break
        payload = await _read_json(workspace, join(directory, entry.path))
        if payload is not None:
            data[PurePosixPath(entry.path).stem] = payload
    return data


async def collect_posthoc_analyses(
    workspace: Workspace, posthoc: PosthocResult | None
) -> list[dict[str, Any]]:
    """``collect_posthoc_analyses`` over the post-hoc phase's passed analyses."""
    if posthoc is None:
        return []
    analyses: list[dict[str, Any]] = []
    context_bytes = 0
    passed = [item for item in posthoc.analyses if item.status == "passed" and item.analysis_dir]
    for item in passed[:MAX_POSTHOC_ANALYSES]:
        directory = item.analysis_dir or ""
        summary_bytes = await _read_small(workspace, join(directory, "analysis_summary.md"))
        summary = summary_bytes.decode("utf-8", errors="replace") if summary_bytes else ""
        if not summary:
            continue
        rubric = await _read_json(workspace, join(directory, "interestingness_rubric.json")) or {}
        manifest_path = join(directory, "analysis_manifest.json")
        manifest_present = await workspace.exists(manifest_path)
        manifest = await _read_json(workspace, manifest_path) or dict(item.manifest)
        figures: list[str] = []
        if manifest_present:
            outputs = manifest.get("outputs", {})
            declared = outputs.get("figures", []) if isinstance(outputs, Mapping) else []
            for raw in list(declared if isinstance(declared, list) else [])[
                :MAX_POSTHOC_DATA_FILES
            ]:
                relative = _safe_relative(raw)
                if relative and relative.lower().endswith(".png"):
                    path = join(directory, relative)
                    if await _read_small(workspace, path) is not None:
                        figures.append(path)
            raw_data = await _manifest_data(workspace, directory, manifest)
        else:
            figures = [
                join(directory, entry.path)
                for entry in await workspace.list_tree(directory)
                if entry.kind == "file"
                and "/" not in entry.path
                and entry.path.lower().endswith(".png")
                and entry.size <= MAX_POSTHOC_FILE_BYTES
            ]
            raw_data = await _directory_data(workspace, directory)
        code = await _read_small(workspace, join(directory, "run_analysis.py"))
        analysis = {
            "analysis_id": item.analysis_id,
            "title": item.title or item.analysis_id,
            "family": item.family,
            "summary_text": summary,
            "rubric": rubric,
            "figure_paths": sorted(figures),
            "raw_data": raw_data,
            "run_analysis_code": code.decode("utf-8", errors="replace") if code else "",
            "attempt_dir": directory,
        }
        size = len(json.dumps(analysis, ensure_ascii=False).encode("utf-8"))
        if size > MAX_POSTHOC_ANALYSIS_BYTES:
            continue
        context_bytes += size
        if context_bytes > MAX_POSTHOC_CONTEXT_BYTES:
            break
        analyses.append(analysis)
    return analyses


def ranking_context_entries(analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The content of ``posthoc_analyses_context.json`` (the collected analyses)."""
    return [dict(item) for item in analyses]


__all__ = [
    "CONTROL_JSON_NAMES",
    "collect_posthoc_analyses",
    "ranking_context_entries",
]
