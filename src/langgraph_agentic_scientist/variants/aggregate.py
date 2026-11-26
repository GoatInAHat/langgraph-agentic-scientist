"""Stage 7: merge the per-variant analysis CSVs into combined files (upstream ``aggregate.py``).

Agents may have added or removed columns depending on the variant, so the merge takes the union
of columns: every column that appears in any variant's CSV is included, missing values are empty
strings. Results are written per task under ``aggregated/<task>/`` (for example
``aggregated/ed_conv/run_summary.csv``); the primary task's files are mirrored at
``aggregated/*.csv`` for existing consumers.

Three layouts are read:

* **structured**: ``variants/<id>/task_bundle/program_<n>/analysis/<csv>`` (the primary task, and
  the extra tasks the pipeline adapted: ``extra_tasks/<task>/variants/...``); per variant only
  the highest-numbered program that holds the file is used, because earlier programs are
  intermediate attempts;
* **flat** (added by hand): ``extra_tasks/<task>/<variant_id>/<csv>``; ``evaluation_id`` and
  ``variant_id`` are injected from the directory name;
* **nested replications** (added by hand):
  ``extra_tasks/<task>/<algorithm>/rep_<n>/run_timeseries.csv``.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Mapping, Sequence

from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace, is_dir, is_file
from .analysis import summary_qualification
from .constants import (
    AGGREGATED_SUBDIR,
    ANALYSIS_CSV_SCHEMAS,
    EXTRA_TASKS_SUBDIR,
    MAX_EVIDENCE_LENGTH,
    VARIANTS_SUBDIR,
)
from .workspace_ops import PROGRAM_DIR_RE, list_child_dirs, list_children, read_text_or_none

logger = logging.getLogger(__name__)

CSV_FILENAMES = list(ANALYSIS_CSV_SCHEMAS)
NESTED_REP_CSV_FILENAMES = ["run_timeseries.csv"]
REP_DIR_RE = re.compile(r"^rep_(.+)$")

CsvEntry = tuple[str, dict[str, str]]


async def _child_files(workspace: Workspace, path: str) -> list[str]:
    return [name for name, directory in await list_children(workspace, path) if not directory]


async def _is_flat_extra_task(workspace: Workspace, task_dir: str) -> bool:
    """A flat extra task has variant subdirectories that hold CSV files directly."""
    for name in await list_child_dirs(workspace, task_dir):
        if name.startswith("_"):
            continue
        if any(f.endswith(".csv") for f in await _child_files(workspace, join(task_dir, name))):
            return True
    return False


async def _is_nested_rep_extra_task(workspace: Workspace, task_dir: str) -> bool:
    """A nested-replication task has ``<algorithm>/rep_<n>/`` directories holding a known CSV."""
    for algo in await list_child_dirs(workspace, task_dir):
        if algo.startswith("_"):
            continue
        for rep in await list_child_dirs(workspace, join(task_dir, algo)):
            if not REP_DIR_RE.match(rep):
                continue
            for csv_name in NESTED_REP_CSV_FILENAMES:
                if await is_file(workspace, join(task_dir, algo, rep, csv_name)):
                    return True
    return False


async def _find_nested_rep_csvs(
    workspace: Workspace, task_dir: str, csv_filename: str
) -> list[CsvEntry]:
    entries: list[CsvEntry] = []
    for algo in await list_child_dirs(workspace, task_dir):
        if algo.startswith("_"):
            continue
        for rep in await list_child_dirs(workspace, join(task_dir, algo)):
            match = REP_DIR_RE.match(rep)
            if not match:
                continue
            path = join(task_dir, algo, rep, csv_filename)
            if not await is_file(workspace, path):
                continue
            variant_id = f"{algo}__rep_{match.group(1)}"
            entries.append(
                (
                    path,
                    {
                        "evaluation_id": variant_id,
                        "variant_id": variant_id,
                        "algorithm": algo,
                        "replication": match.group(1),
                    },
                )
            )
    return entries


async def _find_flat_csvs(workspace: Workspace, task_dir: str, csv_filename: str) -> list[CsvEntry]:
    entries: list[CsvEntry] = []
    for name in await list_child_dirs(workspace, task_dir):
        if name.startswith("_"):
            continue
        path = join(task_dir, name, csv_filename)
        if await is_file(workspace, path):
            entries.append((path, {"evaluation_id": name, "variant_id": name}))
    return entries


async def _find_analysis_csvs(
    workspace: Workspace, variants_dir: str, csv_filename: str
) -> list[str]:
    """Return the latest ``csv_filename`` per variant under its program analysis directories.

    ``program_0`` is the unmodified seed for every variant but the control, whose only program it
    is; the highest-numbered program that holds the file wins.
    """
    paths: list[str] = []
    for variant in await list_child_dirs(workspace, variants_dir):
        bundle = join(variants_dir, variant, "task_bundle")
        if not await is_dir(workspace, bundle):
            continue
        best_index = -1
        best_csv: str | None = None
        for name in await list_child_dirs(workspace, bundle):
            match = PROGRAM_DIR_RE.match(name)
            if match is None:
                continue
            csv_path = join(bundle, name, "analysis", csv_filename)
            index = int(match.group(1))
            if index > best_index and await is_file(workspace, csv_path):
                best_index = index
                best_csv = csv_path
        if best_csv is not None:
            paths.append(best_csv)
    return paths


async def _merge_csvs(
    workspace: Workspace, entries: Sequence[tuple[str, Mapping[str, str]]], output_path: str
) -> int:
    """Merge CSV files into one over the union of their columns; return the rows written.

    Columns are ordered: the injected columns first, then each file's columns in the order they
    first appear. Missing values are empty strings; a file that cannot be read is skipped.
    """
    if not entries:
        await workspace.write_bytes(output_path, b"")
        return 0
    loaded: list[tuple[list[str], list[dict[str, str]], Mapping[str, str]]] = []
    ordered: list[str] = []
    seen: set[str] = set()
    for _, extra in entries:
        for column in extra:
            if column not in seen:
                ordered.append(column)
                seen.add(column)
    for path, extra in entries:
        text = await read_text_or_none(workspace, path)
        if text is None:
            logger.error("Error reading %s; skipping this file", path)
            continue
        try:
            reader = csv.reader(io.StringIO(text, newline=""))
            headers = [h.strip() for h in next(reader, [])]
            rows = list(csv.DictReader(io.StringIO(text, newline="")))
        except csv.Error:
            logger.exception("Error reading %s; skipping this file", path)
            continue
        loaded.append((headers, rows, extra))
        for column in headers:
            if column not in seen:
                ordered.append(column)
                seen.add(column)
    if not ordered:
        logger.warning("No columns found in any CSV; writing empty file to %s", output_path)
        await workspace.write_bytes(output_path, b"")
        return 0
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=ordered, extrasaction="ignore")
    writer.writeheader()
    total = 0
    for _, rows, extra in loaded:
        for row in rows:
            filled = {column: row.get(column, "") for column in ordered}
            filled.update(extra)
            writer.writerow(filled)
            total += 1
    await workspace.write_bytes(output_path, stream.getvalue().encode("utf-8"))
    return total


async def aggregate_variant_csvs(
    workspace: Workspace,
    prompts: PromptRegistry,
    variants_run_dir: str,
    *,
    original_task_name: str = "original",
    max_evidence: int = MAX_EVIDENCE_LENGTH,
) -> dict[str, dict[str, str | None]]:
    """Aggregate analysis CSVs across all variant workspaces and extra tasks.

    Args:
        workspace: The run's workspace.
        prompts: The prompt registry (the rubric descriptions behind ``overall_pass``).
        variants_run_dir: The phase's output directory (``variants/`` and ``extra_tasks/``).
        original_task_name: The label of the primary task under ``aggregated/``.

    Returns:
        Task name to ``{csv filename: aggregated path or None}``; ``None`` where no data was found.
    """
    sources: list[tuple[str, str]] = []
    primary = join(variants_run_dir, VARIANTS_SUBDIR)
    if await is_dir(workspace, primary):
        sources.append((primary, original_task_name))
    flat_sources: list[tuple[str, str]] = []
    nested_sources: list[tuple[str, str]] = []
    extra_root = join(variants_run_dir, EXTRA_TASKS_SUBDIR)
    if await is_dir(workspace, extra_root):
        for name in await list_child_dirs(workspace, extra_root):
            task_dir = join(extra_root, name)
            if await is_dir(workspace, join(task_dir, VARIANTS_SUBDIR)):
                sources.append((join(task_dir, VARIANTS_SUBDIR), name))
            elif await _is_nested_rep_extra_task(workspace, task_dir):
                nested_sources.append((task_dir, name))
            elif await _is_flat_extra_task(workspace, task_dir):
                flat_sources.append((task_dir, name))
    if not sources and not flat_sources and not nested_sources:
        logger.warning("No variants directory found at %s; skipping aggregation", variants_run_dir)
        return {}

    aggregated_root = join(variants_run_dir, AGGREGATED_SUBDIR)
    results: dict[str, dict[str, str | None]] = {}

    for variants_dir, task_name in sources:
        task_results: dict[str, str | None] = {}
        for csv_filename in CSV_FILENAMES:
            csv_paths = await _find_analysis_csvs(workspace, variants_dir, csv_filename)
            if not csv_paths:
                logger.info("No %s files found for task %s; skipping", csv_filename, task_name)
                task_results[csv_filename] = None
                continue
            entries: list[tuple[str, Mapping[str, str]]] = []
            for path in csv_paths:
                extra: Mapping[str, str] = {}
                if csv_filename == "run_summary.csv":
                    program_dir = path.rsplit("/", 2)[0]
                    extra = await summary_qualification(
                        workspace, program_dir, prompts=prompts, max_evidence=max_evidence
                    )
                entries.append((path, extra))
            out_path = join(aggregated_root, task_name, csv_filename)
            rows = await _merge_csvs(workspace, entries, out_path)
            logger.info(
                "Aggregated %s/%s: %d files, %d total rows -> %s",
                task_name,
                csv_filename,
                len(csv_paths),
                rows,
                out_path,
            )
            task_results[csv_filename] = out_path
        if variants_dir == primary:
            await _sync_primary_compatibility_files(workspace, aggregated_root, task_results)
        results[task_name] = task_results

    for task_dir, task_name in flat_sources:
        flat_results: dict[str, str | None] = {}
        for csv_filename in CSV_FILENAMES:
            flat_entries = await _find_flat_csvs(workspace, task_dir, csv_filename)
            if not flat_entries:
                logger.info("No %s files found for flat task %s; skipping", csv_filename, task_name)
                flat_results[csv_filename] = None
                continue
            out_path = join(aggregated_root, task_name, csv_filename)
            await _merge_csvs(workspace, flat_entries, out_path)
            flat_results[csv_filename] = out_path
        results[task_name] = flat_results

    for task_dir, task_name in nested_sources:
        nested_results: dict[str, str | None] = {}
        for csv_filename in NESTED_REP_CSV_FILENAMES:
            nested_entries = await _find_nested_rep_csvs(workspace, task_dir, csv_filename)
            if not nested_entries:
                logger.info(
                    "No %s files found for nested-rep task %s; skipping", csv_filename, task_name
                )
                nested_results[csv_filename] = None
                continue
            out_path = join(aggregated_root, task_name, csv_filename)
            await _merge_csvs(workspace, nested_entries, out_path)
            nested_results[csv_filename] = out_path
        results[task_name] = nested_results
    return results


async def _sync_primary_compatibility_files(
    workspace: Workspace, aggregated_root: str, task_results: Mapping[str, str | None]
) -> None:
    """Mirror the primary task's CSVs at ``aggregated/*.csv``; a missing source removes the copy."""
    for csv_filename, source in task_results.items():
        compat = join(aggregated_root, csv_filename)
        if source is None:
            await workspace.remove(compat)
            continue
        await workspace.write_bytes(compat, await workspace.read_bytes(source))


__all__ = ["aggregate_variant_csvs"]
