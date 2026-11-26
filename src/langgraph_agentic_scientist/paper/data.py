"""Paper data: ``paper_data.json`` and its grounding sources, built from the results of the phases.

Upstream ``paper_data.py``, ``paper_data_ablation.py``, ``paper_data_posthoc.py``,
``paper_study_context.py``, ``paper_study_sources.py``, ``paper_evaluation_evidence.py`` and the
grounding part of ``paper_final_outcome.py``. Upstream scans the run directory; the port builds
the same JSON, with the same grounding ids, from the typed results of the phases
(:class:`~langgraph_agentic_scientist.core.results.InvestigationResult`, ``DownstreamSelection``,
``VariantsResult``, ``FinalEvaluationResult``) and reads the artifacts those results name through
the workspace.

The grounding ids a writer may cite:

* ``result.best_correct_score``, ``result.best_any_score``, ``result.total_ideas``,
  ``result.total_attempts``, ``result.total_evaluated_programs``,
  ``result.total_correct_programs``, ``result.top_program_NN`` and ``result.improvement_NNN``;
* ``result.total_ablation_variants``, ``result.total_ablation_runs`` and
  ``ablation.study_<study>.variant_<variant>.<metric>.<mean|std|min|max|n>`` and
  ``ablation.study_<study>.compare_<baseline>_vs_<variant>.<metric>.mean_delta``;
* ``result.<task>.<variant>.<metric>.<mean|std|min|max|n>`` (cross-task results);
* ``study.<fact>`` and ``study.final.<status|correct|outcome>`` (a downstream study without an
  investigation winner);
* ``final.metric.<name>``, ``final.evidence.episodes`` and
  ``final.qualification.<status|correct|outcome>`` (the sealed final evaluation);
* ``posthoc.<analysis_id>.summary`` and ``posthoc.<analysis_id>.data.<file>`` (ranked post-hoc
  analyses, added by :func:`augment_paper_data_with_posthoc`).

:func:`build_paper_data` is also the default node of the ``paper.data`` slot
(:func:`paper_data_node`).
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
from collections import defaultdict
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any, Literal

from langgraph.runtime import Runtime

from ..context import ScientistContext
from ..core.results import (
    DownstreamSelection,
    InvestigationResult,
    PaperData,
    VariantsResult,
)
from ..core.state import EvaluationRecord, utc_now
from ..evaluation.evidence import decode_evaluation_evidence, evidence_metadata
from ..evaluation.program_clone import NONOVERRIDABLE_EXCLUDES, evaluation_results_complete
from ..evaluation.protocol import FinalEvaluationResult
from ..workspace.paths import join, task_copy_path_is_ignored
from ..workspace.protocol import Workspace

PROGRAM_DIR_RE = re.compile(r"^program_(\d+)$")
VALID_PAPER_PROFILES = {"default", "standalone"}
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
EVIDENCE_FILE = "evaluation_evidence.json"
_IDENTIFIER_RE = re.compile(r"[^a-z0-9]+")

FinalOutcomePolicy = Literal["qualified_only", "allow_not_qualified"]
"""Whether a verified ``not_qualified`` final evaluation may support a paper."""


class PaperDataError(ValueError):
    """The paper data cannot be built (no evaluated programs, an unqualified final evaluation, ...)."""


class ResultsReader:
    """Reads the artifacts of the phases through a workspace (missing or unsafe means ``None``).

    A file is read only when it is a regular file (not a symlink); JSON that is not an object,
    and undecodable text, read as missing, as upstream's readers do.
    """

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace

    def host(self, path: str) -> str:
        """Return ``path`` as processes of the workspace see it (the paths in paper data)."""
        return self.workspace.host_path(path)

    async def is_file(self, path: str) -> bool:
        """Return whether ``path`` is a regular file."""
        entry = await self.workspace.stat(path)
        return entry is not None and entry.kind == "file"

    async def is_dir(self, path: str) -> bool:
        """Return whether ``path`` is a directory (not a symlink)."""
        entry = await self.workspace.stat(path)
        return entry is not None and entry.kind == "dir"

    async def read_bytes(self, path: str) -> bytes | None:
        """Return the bytes of a regular file, or ``None``."""
        if not await self.is_file(path):
            return None
        return await self.workspace.read_bytes(path)

    async def read_text(self, path: str) -> str | None:
        """Return the UTF-8 text of a regular file, or ``None``."""
        data = await self.read_bytes(path)
        if data is None:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    async def read_json(self, path: str) -> dict[str, Any] | None:
        """Return the JSON object in a regular file, or ``None``."""
        text = await self.read_text(path)
        if text is None:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    async def read_csv_rows(self, path: str) -> list[dict[str, str]]:
        """Return the rows of a CSV file as dictionaries (empty when the file is absent)."""
        text = await self.read_text(path)
        if text is None:
            return []
        return [dict(row) for row in csv.DictReader(io.StringIO(text, newline=""))]

    async def list_directories(self, path: str) -> list[str]:
        """Return the names of the directories directly below ``path``, sorted."""
        if not await self.is_dir(path):
            return []
        entries = await self.workspace.list_tree(path)
        return sorted(
            entry.path for entry in entries if entry.kind == "dir" and "/" not in entry.path
        )


EvidenceReader = Callable[[ResultsReader, str], Awaitable[bytes | None]]
"""Returns the bytes of the ``evaluation_evidence.json`` of an evaluated program directory."""


async def default_evidence_reader(reader: ResultsReader, program_dir: str) -> bytes | None:
    """Read ``<program_dir>/evaluation_evidence.json`` (upstream's location), or ``None``."""
    return await reader.read_bytes(join(program_dir, EVIDENCE_FILE))


async def _require_no_symlink(reader: ResultsReader, root: str, program: str) -> None:
    """Refuse an evidence directory that is outside ``root`` or whose path holds a symlink."""
    root_parts = PurePosixPath(root).parts
    parts = PurePosixPath(program).parts
    if parts[: len(root_parts)] != root_parts:
        raise ValueError("Evaluation evidence directory escapes its source root")
    current = root
    for part in ("", *parts[len(root_parts) :]):
        current = join(current, part) if part else current
        entry = await reader.workspace.stat(current) if current else None
        if entry is not None and entry.kind == "symlink":
            raise ValueError(
                f"Evaluation evidence directory contains symlink: {reader.host(current)}"
            )


async def paper_evaluation_evidence(
    reader: ResultsReader,
    root: str,
    program: str,
    *,
    evidence_reader: EvidenceReader | None = None,
) -> dict[str, Any]:
    """Load optional scientific data of an evaluated program without following links.

    Returns upstream's evidence metadata (``status``, and when available ``path``, ``sha256``,
    ``episode_count``, ``candidate_diagnostics_trust``) and, when available, the ``episodes``.

    Raises:
        ValueError: When the directory has a symlink in its path or the evidence breaks a rule of
            ``evaluation_evidence.json``.
    """
    await _require_no_symlink(reader, root, program)
    raw = await (evidence_reader or default_evidence_reader)(reader, program)
    metadata = evidence_metadata(raw, reader.host(join(program, EVIDENCE_FILE)) if raw else None)
    result = metadata.model_dump(exclude_none=True)
    if raw is not None:
        result["episodes"] = decode_evaluation_evidence(raw)["episodes"]
    return result


async def ablation_evaluation_evidence(
    reader: ResultsReader,
    root: str,
    variant_id: str,
    program_value: str | None,
    *,
    evidence_reader: EvidenceReader | None = None,
) -> dict[str, Any]:
    """Resolve CSV program labels only inside the corresponding variant workspace."""
    unavailable: dict[str, Any] = {"status": "unavailable"}
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", variant_id) or not program_value:
        return unavailable
    label = PurePosixPath(program_value)
    if not re.fullmatch(r"program_\d+", label.name):
        return unavailable
    program = join(root, "variants", variant_id, "task_bundle", label.name)
    if label != PurePosixPath(label.name) and label.as_posix() not in {
        program,
        reader.host(program),
    }:
        return unavailable
    return await paper_evaluation_evidence(reader, root, program, evidence_reader=evidence_reader)


def utc_now_string() -> str:
    """Return the current UTC timestamp string (upstream ``utc_now``)."""
    return utc_now()


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        compact = value.strip()
        if not compact:
            return None
        try:
            return float(compact)
        except ValueError:
            return None
    return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        compact = value.strip()
        if not compact:
            return None
        try:
            return int(compact)
        except ValueError:
            return None
    return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        compact = value.strip().lower()
        if compact in {"true", "1", "yes", "y"}:
            return True
        if compact in {"false", "0", "no", "n"}:
            return False
    return None


def _slug(value: str) -> str:
    compact = _IDENTIFIER_RE.sub("_", value.strip().lower()).strip("_")
    return compact or "unknown"


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = 0.0
    if len(values) > 1:
        variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "n": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def _best_by_score(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda item: (
            float(item.get("combined_score", float("-inf"))),
            -int(item.get("eval_index", 10**9)),
        ),
    )


def _stats(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0}
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(values),
        "max": max(values),
    }


def _float_column_values(rows: list[dict[str, str]], column_name: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _safe_float(row.get(column_name))
        if value is not None:
            values.append(value)
    return values


def _metric_float(value: Any) -> float | None:
    """Handle safe float logic for this module."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _metric_int(value: Any) -> int | None:
    """Handle safe int logic for this module."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _reindex_program_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return shallow-copied program rows with contiguous eval indices."""
    reindexed: list[dict[str, Any]] = []
    for eval_index, row in enumerate(rows, start=1):
        payload = dict(row)
        payload["eval_index"] = eval_index
        reindexed.append(payload)
    return reindexed


def _update_program_summary_fields(paper_data: dict[str, Any]) -> dict[str, Any]:
    """Recompute score-bearing paper fields from all_programs."""
    rows_raw = paper_data.get("all_programs")
    if not isinstance(rows_raw, list):
        return paper_data

    all_programs = [dict(row) for row in rows_raw if isinstance(row, dict)]
    all_programs.sort(key=lambda row: int(row.get("eval_index", 10**9)))
    all_programs = _reindex_program_rows(all_programs)
    correct_programs = [row for row in all_programs if row.get("correct") is True]
    top_programs = sorted(
        correct_programs,
        key=lambda row: (
            -float(row["combined_score"]),
            int(row["eval_index"]),
        ),
    )[:10]
    improvement_trace: list[dict[str, Any]] = []
    running_best = float("-inf")
    for row in correct_programs:
        score = float(row["combined_score"])
        if score <= running_best:
            continue
        running_best = score
        improvement_trace.append(
            {
                "id": f"result.improvement_{len(improvement_trace) + 1:03d}",
                "step_index": int(row["eval_index"]),
                "x_index": int(row["eval_index"]) - 1,
                "idea_id": row["idea_id"],
                "attempt_id": row["attempt_id"],
                "program_id": row["program_id"],
                "program_idx": int(row["program_idx"]),
                "combined_score": score,
                "program_dir": row["program_dir"],
            }
        )

    best_correct = _best_by_score(correct_programs)
    best_any = _best_by_score(all_programs)

    paper_data["all_programs"] = all_programs
    paper_data["best_correct"] = best_correct
    paper_data["best_any"] = best_any
    paper_data["top_programs"] = top_programs
    paper_data["improvement_trace"] = improvement_trace

    totals = paper_data.get("totals")
    if isinstance(totals, dict):
        totals["evaluated_programs"] = len(all_programs)
        totals["correct_programs"] = len(correct_programs)

    grounding_sources = paper_data.get("grounding_sources")
    if not isinstance(grounding_sources, dict):
        return paper_data

    for source_id in list(grounding_sources):
        if (
            source_id
            in {
                "result.best_correct_score",
                "result.best_any_score",
                "result.total_evaluated_programs",
                "result.total_correct_programs",
            }
            or source_id.startswith("result.top_program_")
            or source_id.startswith("result.improvement_")
        ):
            del grounding_sources[source_id]

    def register_source(source_id: str, payload: dict[str, Any]) -> None:
        """Register source."""
        grounding_sources[source_id] = {
            "id": source_id,
            **payload,
        }

    if isinstance(best_correct, dict):
        register_source(
            "result.best_correct_score",
            {
                "type": "metric",
                "value": float(best_correct["combined_score"]),
                "program_dir": best_correct.get("program_dir"),
                "idea_id": best_correct.get("idea_id"),
                "attempt_id": best_correct.get("attempt_id"),
                "program_id": best_correct.get("program_id"),
            },
        )

    if isinstance(best_any, dict):
        register_source(
            "result.best_any_score",
            {
                "type": "metric",
                "value": float(best_any["combined_score"]),
                "program_dir": best_any.get("program_dir"),
                "idea_id": best_any.get("idea_id"),
                "attempt_id": best_any.get("attempt_id"),
                "program_id": best_any.get("program_id"),
            },
        )

    register_source(
        "result.total_evaluated_programs", {"type": "count", "value": len(all_programs)}
    )
    register_source(
        "result.total_correct_programs", {"type": "count", "value": len(correct_programs)}
    )

    for idx, row in enumerate(top_programs, start=1):
        register_source(
            f"result.top_program_{idx:02d}",
            {
                "type": "metric",
                "value": float(row["combined_score"]),
                "program_dir": row.get("program_dir"),
                "idea_id": row.get("idea_id"),
                "attempt_id": row.get("attempt_id"),
                "program_id": row.get("program_id"),
            },
        )

    for row in improvement_trace:
        register_source(
            str(row["id"]),
            {
                "type": "metric",
                "value": row["combined_score"],
                "program_dir": row.get("program_dir"),
                "idea_id": row.get("idea_id"),
                "attempt_id": row.get("attempt_id"),
                "program_id": row.get("program_id"),
            },
        )

    return paper_data


async def _cross_task_variant_metadata(
    reader: ResultsReader, results_dir: str, task_name: str, variant_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Load optional variant metadata from extra_tasks/<task>/<variant>/variant_spec.json.

    Upstream lists the variant directories of the task; the port reads the specs of the variants
    that the task's aggregated tables name, which is the set the metadata is used for.
    """
    task_root = join(results_dir, "extra_tasks", task_name)
    metadata: dict[str, dict[str, Any]] = {}
    for dir_name in sorted({name for name in variant_ids if name}):
        if "/" in dir_name or dir_name in {".", ".."}:
            continue
        spec = await reader.read_json(join(task_root, dir_name, "variant_spec.json"))
        if not isinstance(spec, dict):
            continue
        variant_id = str(spec.get("variant_id") or dir_name).strip()
        if not variant_id:
            continue
        metadata[variant_id] = spec
    return metadata


def _summarize_cross_task_run_summary(
    *,
    rows: list[dict[str, str]],
    task_name: str,
    csv_path: str,
    variant_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Summarize per-variant stats from aggregated/<task>/run_summary.csv."""
    task_payload: dict[str, Any] = {}
    grounding_sources: dict[str, dict[str, Any]] = {}
    task_key = _slug(task_name)

    by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        variant_id = (row.get("variant_id") or "").strip()
        if not variant_id:
            continue
        by_variant[variant_id].append(row)

    for variant_id, variant_rows in by_variant.items():
        primary_metric_name = (variant_rows[0].get("primary_metric_name") or "").strip()
        metric_name = primary_metric_name or "combined_score"
        variant_kind = (variant_rows[0].get("variant_kind") or "").strip() or None
        if variant_kind is None:
            variant_kind = (
                str(variant_metadata.get(variant_id, {}).get("kind") or "").strip() or None
            )
        values = _float_column_values(variant_rows, "primary_metric_value")
        if not values:
            metric_name = "combined_score"
            values = _float_column_values(variant_rows, "combined_score")
        if not values:
            continue
        stats = _stats(values)
        correct_count = sum(1 for row in variant_rows if _safe_bool(row.get("correct")) is True)
        task_payload[variant_id] = {
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": metric_name,
            "n": int(stats["n"]),
            "correct_count": correct_count,
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
        }
        variant_key = _slug(variant_id)
        metric_key = _slug(metric_name)
        base = f"result.{task_key}.{variant_key}"
        for field in ("mean", "std", "min", "max"):
            value = stats.get(field)
            if value is None:
                continue
            grounding_sources[f"{base}.{metric_key}.{field}"] = {
                "id": f"{base}.{metric_key}.{field}",
                "type": "metric",
                "value": float(value),
                "task_name": task_name,
                "variant_id": variant_id,
                "variant_kind": variant_kind,
                "metric_name": metric_name,
                "summary_field": field,
                "source_csv": str(csv_path),
            }
        grounding_sources[f"{base}.{metric_key}.n"] = {
            "id": f"{base}.{metric_key}.n",
            "type": "count",
            "value": int(stats["n"]),
            "task_name": task_name,
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": metric_name,
            "source_csv": str(csv_path),
        }

    return task_payload, grounding_sources


def _summarize_cross_task_metric_values(
    *,
    rows: list[dict[str, str]],
    task_name: str,
    csv_path: str,
    task_payload: dict[str, Any],
    variant_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Register summaries for every numeric metric in metric_values.csv."""
    values_by_metric: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        variant_id = str(row.get("variant_id", "")).strip()
        metric_name = str(row.get("metric_name", "")).strip()
        metric_value = _safe_float(row.get("metric_value"))
        if variant_id and metric_name and metric_value is not None and math.isfinite(metric_value):
            values_by_metric[(variant_id, metric_name)].append(metric_value)

    grounding_sources: dict[str, dict[str, Any]] = {}
    source_identities: dict[str, tuple[str, str]] = {}
    task_key = _slug(task_name)
    for (variant_id, metric_name), values in sorted(values_by_metric.items()):
        try:
            stats = {
                "n": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        except (OverflowError, statistics.StatisticsError):
            continue
        if any(not math.isfinite(float(value)) for field, value in stats.items() if field != "n"):
            continue
        variant_kind = None
        variant_payload = task_payload.get(variant_id)
        if isinstance(variant_payload, dict):
            variant_kind = variant_payload.get("variant_kind")
        if variant_kind is None:
            variant_kind = variant_metadata.get(variant_id, {}).get("kind")
        base = f"result.{task_key}.{_slug(variant_id)}.{_slug(metric_name)}"
        identity = (variant_id, metric_name)
        previous_identity = source_identities.setdefault(base, identity)
        if previous_identity != identity:
            raise ValueError(
                "metric grounding source ID collision: "
                f"{base!r} maps to both {previous_identity!r} and {identity!r}"
            )
        for field in ("mean", "std", "min", "max"):
            value = stats.get(field)
            if value is None:
                continue
            source_id = f"{base}.{field}"
            grounding_sources[source_id] = {
                "id": source_id,
                "type": "metric",
                "value": float(value),
                "task_name": task_name,
                "variant_id": variant_id,
                "variant_kind": variant_kind,
                "metric_name": metric_name,
                "summary_field": field,
                "source_csv": str(csv_path),
            }
        count_source_id = f"{base}.n"
        grounding_sources[count_source_id] = {
            "id": count_source_id,
            "type": "count",
            "value": int(stats["n"]),
            "task_name": task_name,
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": metric_name,
            "source_csv": str(csv_path),
        }
    return grounding_sources


def _merge_cross_task_grounding_sources(
    target: dict[str, dict[str, Any]],
    incoming: dict[str, dict[str, Any]],
    *,
    preserve_existing: bool,
) -> None:
    """Merge sources while rejecting lossy identifier normalization."""
    identity_fields = ("task_name", "variant_id", "metric_name")
    for source_id, payload in incoming.items():
        existing = target.get(source_id)
        if existing is None:
            target[source_id] = payload
            continue
        existing_identity = tuple(existing.get(field) for field in identity_fields)
        incoming_identity = tuple(payload.get(field) for field in identity_fields)
        if existing_identity != incoming_identity:
            raise ValueError(
                "metric grounding source ID collision: "
                f"{source_id!r} maps to both {existing_identity!r} "
                f"and {incoming_identity!r}"
            )
        if not preserve_existing:
            target[source_id] = payload


def _preferred_cross_task_metric_name(metric_names: set[str]) -> str | None:
    """Choose a reader-facing primary metric from run_timeseries metric names."""
    if not metric_names:
        return None
    preferred_order = (
        "eval/episode_reward",
        "episode_return",
        "eval_return",
        "reward",
        "return",
    )
    normalized = {name.strip(): name.strip() for name in metric_names if name.strip()}
    for candidate in preferred_order:
        if candidate in normalized:
            return normalized[candidate]
    return sorted(normalized)[0] if normalized else None


def _summarize_cross_task_run_timeseries(
    *,
    rows: list[dict[str, str]],
    task_name: str,
    csv_path: str,
    variant_metadata: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Summarize per-variant stats from aggregated/<task>/run_timeseries.csv."""
    task_payload: dict[str, Any] = {}
    grounding_sources: dict[str, dict[str, Any]] = {}
    task_key = _slug(task_name)

    # Track the terminal value for each (variant, replication, metric).
    terminal_values: dict[tuple[str, str, str], tuple[int, float]] = {}
    metric_names_by_variant: dict[str, set[str]] = defaultdict(set)
    variant_kind_by_id: dict[str, str | None] = {}

    for row in rows:
        algorithm = str(row.get("algorithm", "")).strip()
        variant_id = algorithm or str(row.get("variant_id", "")).strip()
        if not variant_id:
            continue
        replication = (
            str(row.get("replication", "")).strip() or str(row.get("seed", "")).strip() or "0"
        )
        metric_name = str(row.get("metric", "")).strip()
        step = _safe_int(row.get("step"))
        value = _safe_float(row.get("value"))
        if not metric_name or step is None or value is None:
            continue

        metric_names_by_variant[variant_id].add(metric_name)
        if variant_id not in variant_kind_by_id:
            variant_kind_by_id[variant_id] = (
                str(variant_metadata.get(variant_id, {}).get("kind") or "").strip() or None
            )
        key = (variant_id, replication, metric_name)
        current = terminal_values.get(key)
        if current is None or step >= current[0]:
            terminal_values[key] = (step, value)

    for variant_id in sorted(metric_names_by_variant):
        selected_metric_name = _preferred_cross_task_metric_name(
            metric_names_by_variant[variant_id]
        )
        if selected_metric_name is None:
            continue
        values = [
            value
            for (current_variant_id, _replication, current_metric_name), (
                _step,
                value,
            ) in terminal_values.items()
            if current_variant_id == variant_id and current_metric_name == selected_metric_name
        ]
        if not values:
            continue
        stats = _stats(values)
        variant_kind = variant_kind_by_id.get(variant_id)
        task_payload[variant_id] = {
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": selected_metric_name,
            "n": int(stats["n"]),
            # RL cross-task timeseries lacks a correctness concept; preserve the field
            # for schema compatibility while using the completed replication count.
            "correct_count": int(stats["n"]),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "source_layout": "run_timeseries_terminal",
        }
        variant_key = _slug(variant_id)
        metric_key = _slug(selected_metric_name)
        base = f"result.{task_key}.{variant_key}"
        for field in ("mean", "std", "min", "max"):
            value = stats.get(field)
            if value is None:
                continue
            grounding_sources[f"{base}.{metric_key}.{field}"] = {
                "id": f"{base}.{metric_key}.{field}",
                "type": "metric",
                "value": float(value),
                "task_name": task_name,
                "variant_id": variant_id,
                "variant_kind": variant_kind,
                "metric_name": selected_metric_name,
                "summary_field": field,
                "source_csv": str(csv_path),
                "source_layout": "run_timeseries_terminal",
            }
        grounding_sources[f"{base}.{metric_key}.n"] = {
            "id": f"{base}.{metric_key}.n",
            "type": "count",
            "value": int(stats["n"]),
            "task_name": task_name,
            "variant_id": variant_id,
            "variant_kind": variant_kind,
            "metric_name": selected_metric_name,
            "source_csv": str(csv_path),
            "source_layout": "run_timeseries_terminal",
        }

    return task_payload, grounding_sources


async def collect_cross_task_results(reader: ResultsReader, results_dir: str) -> dict[str, Any]:
    """Scan aggregated/<task>/run_summary.csv for every task and emit per-variant stats.

    Returns a dict with:
      - task_names: sorted list of task names discovered
      - cross_task_results: {task: {variant_id: {variant_kind, metric_name, stats...}}}
      - grounding_sources: {result.<task>.<variant>.<stat>: payload}
    """
    aggregated_root = join(results_dir, "aggregated")
    task_names: list[str] = []
    cross_task_results: dict[str, dict[str, Any]] = {}
    grounding_sources: dict[str, dict[str, Any]] = {}
    if not await reader.is_dir(aggregated_root):
        return {"task_names": [], "cross_task_results": {}, "grounding_sources": {}}

    for task_name in await reader.list_directories(aggregated_root):
        task_dir = join(aggregated_root, task_name)
        task_payload: dict[str, Any] = {}
        task_sources: dict[str, dict[str, Any]] = {}
        summary_csv_path = join(task_dir, "run_summary.csv")
        summary_rows = await reader.read_csv_rows(summary_csv_path)
        timeseries_csv_path = join(task_dir, "run_timeseries.csv")
        metric_values_path = join(task_dir, "metric_values.csv")
        metric_rows = await reader.read_csv_rows(metric_values_path)
        timeseries_rows = [] if summary_rows else await reader.read_csv_rows(timeseries_csv_path)
        variant_ids = [
            *(str(row.get("variant_id", "")).strip() for row in summary_rows),
            *(str(row.get("variant_id", "")).strip() for row in metric_rows),
            *(
                str(row.get("algorithm", "")).strip() or str(row.get("variant_id", "")).strip()
                for row in timeseries_rows
            ),
        ]
        variant_metadata = await _cross_task_variant_metadata(
            reader, results_dir, task_name, variant_ids
        )
        if summary_rows:
            task_payload, task_sources = _summarize_cross_task_run_summary(
                rows=summary_rows,
                task_name=task_name,
                csv_path=reader.host(summary_csv_path),
                variant_metadata=variant_metadata,
            )
        elif timeseries_rows:
            task_payload, task_sources = _summarize_cross_task_run_timeseries(
                rows=timeseries_rows,
                task_name=task_name,
                csv_path=reader.host(timeseries_csv_path),
                variant_metadata=variant_metadata,
            )

        metric_sources = _summarize_cross_task_metric_values(
            rows=metric_rows,
            task_name=task_name,
            csv_path=reader.host(metric_values_path),
            task_payload=task_payload,
            variant_metadata=variant_metadata,
        )

        if task_payload:
            task_names.append(task_name)
            cross_task_results[task_name] = task_payload
        _merge_cross_task_grounding_sources(
            grounding_sources,
            task_sources,
            preserve_existing=False,
        )
        _merge_cross_task_grounding_sources(
            grounding_sources,
            metric_sources,
            preserve_existing=True,
        )

    return {
        "task_names": task_names,
        "cross_task_results": cross_task_results,
        "grounding_sources": grounding_sources,
    }


async def _merge_nested_ablation_payload(
    paper_data: dict[str, Any],
    *,
    reader: ResultsReader,
    nested_bundle_dir: str,
    paper_profile: str,
    variant_ids: Sequence[str] = (),
    evidence_reader: EvidenceReader | None = None,
) -> dict[str, Any]:
    """Augment normal paper data with ablation-only fields from a nested bundle."""
    if await has_multitask_ablation_bundle(reader, nested_bundle_dir):
        ablation_data = await build_multitask_ablation_paper_data(
            reader,
            nested_bundle_dir,
            paper_profile=paper_profile,
            variant_ids=variant_ids,
            evidence_reader=evidence_reader,
        )
    else:
        ablation_data = await build_ablation_paper_data(
            reader,
            nested_bundle_dir,
            paper_profile=paper_profile,
            variant_ids=variant_ids,
            evidence_reader=evidence_reader,
        )

    ablation_studies = ablation_data.get("ablation_studies")
    if isinstance(ablation_studies, list) and ablation_studies:
        paper_data["ablation_studies"] = ablation_studies

    idea_titles = paper_data.get("idea_titles")
    ablation_titles = ablation_data.get("idea_titles")
    if isinstance(idea_titles, list) and isinstance(ablation_titles, list):
        paper_data["idea_titles"] = list(
            dict.fromkeys(
                str(title).strip()
                for title in [*idea_titles, *ablation_titles]
                if isinstance(title, str) and str(title).strip()
            )
        )

    totals = paper_data.get("totals")
    ablation_totals = ablation_data.get("totals")
    if isinstance(totals, dict) and isinstance(ablation_totals, dict):
        for key in (
            "ablation_studies",
            "ablation_variants",
            "ablation_runs",
            "completed_ablation_variants",
            "failed_ablation_variants",
            "timed_out_ablation_variants",
        ):
            value = ablation_totals.get(key)
            if isinstance(value, int):
                totals[key] = value

    grounding_sources = paper_data.get("grounding_sources")
    ablation_sources = ablation_data.get("grounding_sources")
    if isinstance(grounding_sources, dict) and isinstance(ablation_sources, dict):
        for source_id, payload in ablation_sources.items():
            source_key = str(source_id)
            if source_key.startswith("ablation.") or source_key in {
                "result.total_ablation_studies",
                "result.total_ablation_variants",
                "result.total_ablation_runs",
            }:
                grounding_sources[source_key] = payload

    base_rows = paper_data.get("all_programs")
    ablation_rows = ablation_data.get("all_programs")
    if isinstance(base_rows, list) and isinstance(ablation_rows, list):
        merged_rows = [dict(row) for row in base_rows if isinstance(row, dict)]
        merged_rows.extend(dict(row) for row in ablation_rows if isinstance(row, dict))
        paper_data["all_programs"] = _reindex_program_rows(merged_rows)
        _update_program_summary_fields(paper_data)

    return paper_data


def _choose_anchor_variant(variants: list[dict[str, Any]]) -> str | None:
    if not variants:
        return None
    by_id = {str(item.get("variant_id")): item for item in variants if item.get("variant_id")}
    if "control_best" in by_id:
        return "control_best"
    for item in variants:
        if str(item.get("kind", "")).strip().lower() == "control":
            return str(item["variant_id"])
    for item in variants:
        if str(item.get("kind", "")).strip().lower() == "baseline":
            return str(item["variant_id"])
    return str(variants[0].get("variant_id")) if variants else None


async def build_ablation_paper_data(
    reader: ResultsReader,
    results_dir: str,
    *,
    paper_profile: str = "default",
    study_id_override: str | None = None,
    study_title_override: str | None = None,
    variant_ids: Sequence[str] = (),
    evidence_reader: EvidenceReader | None = None,
) -> dict[str, Any]:
    """Build paper_data.json from a variants-with-reps ablation bundle.

    ``variant_ids`` names the variants of the bundle (the specs of the variants phase); the
    variants named by the pipeline summary and the aggregated tables are added, and each must be
    a directory under ``variants/``. Upstream lists the ``variants/`` directory instead.
    """
    resolved = results_dir
    pipeline_summary = await reader.read_json(join(resolved, "pipeline_summary.json")) or {}
    pipeline_variant_rows = pipeline_summary.get("variants")
    pipeline_variants: dict[str, dict[str, Any]] = {}
    if isinstance(pipeline_variant_rows, list):
        for item in pipeline_variant_rows:
            if not isinstance(item, dict):
                continue
            variant_id = item.get("variant_id")
            if isinstance(variant_id, str) and variant_id.strip():
                pipeline_variants[variant_id.strip()] = item

    run_rows = await reader.read_csv_rows(join(resolved, "aggregated", "run_summary.csv"))
    metric_rows = await reader.read_csv_rows(join(resolved, "aggregated", "metric_values.csv"))
    candidate_ids = {str(name) for name in variant_ids} | set(pipeline_variants)
    candidate_ids |= {str(row.get("variant_id", "")).strip() for row in run_rows}
    variant_dirs = [
        join(resolved, "variants", name)
        for name in sorted(candidate_ids)
        if name and "/" not in name and await reader.is_dir(join(resolved, "variants", name))
    ]

    runs_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    metrics_by_variant: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    primary_metric_flags: dict[str, dict[str, bool]] = defaultdict(dict)
    all_programs: list[dict[str, Any]] = []

    for eval_index, row in enumerate(run_rows, start=1):
        variant_id = str(row.get("variant_id", "")).strip()
        if not variant_id:
            continue
        run_id = str(row.get("run_id", "")).strip() or f"run_{eval_index:03d}"
        evaluation_id = str(row.get("evaluation_id", "")).strip() or None
        program_dir = str(row.get("program_dir", "")).strip() or None
        results_dir_value = str(row.get("results_dir", "")).strip() or None
        primary_metric_name = str(row.get("primary_metric_name", "")).strip() or "combined_score"
        primary_metric_value = _safe_float(row.get("primary_metric_value"))
        combined_score = _safe_float(row.get("combined_score"))
        if combined_score is None:
            combined_score = primary_metric_value
        if combined_score is None:
            continue
        correct = _safe_bool(row.get("correct"))
        overall_pass = _safe_bool(row.get("overall_pass"))
        program_idx = len(runs_by_variant[variant_id])
        evaluation_evidence = await ablation_evaluation_evidence(
            reader, resolved, variant_id, program_dir, evidence_reader=evidence_reader
        )
        run_payload = {
            "run_id": run_id,
            "evaluation_id": evaluation_id,
            "variant_id": variant_id,
            "variant_kind": str(row.get("variant_kind", "")).strip() or None,
            "base_program": str(row.get("base_program", "")).strip() or None,
            "agent_name": str(row.get("agent_name", "")).strip() or None,
            "correct": correct,
            "overall_pass": overall_pass,
            "combined_score": combined_score,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "program_dir": program_dir,
            "results_dir": results_dir_value,
            "eval_index": eval_index,
            "evaluation_evidence": evaluation_evidence,
        }
        runs_by_variant[variant_id].append(run_payload)
        all_programs.append(
            {
                "idea_id": variant_id,
                "idea_dir_name": variant_id,
                "attempt_id": f"{variant_id}_attempt_01",
                "attempt_dir_name": "variant_bundle",
                "program_id": run_id,
                "program_idx": program_idx,
                "combined_score": combined_score,
                "correct": correct,
                "eligible": (correct is True),
                "program_dir": program_dir,
                "eval_index": eval_index,
                "variant_id": variant_id,
                "variant_kind": str(row.get("variant_kind", "")).strip() or None,
                "run_id": run_id,
                "evaluation_id": evaluation_id,
                "results_dir": results_dir_value,
                "source_layout": "ablation_bundle",
                "evaluation_evidence": evaluation_evidence,
                "primary_metric_name": primary_metric_name,
                "primary_metric_value": primary_metric_value,
            }
        )

    for row in metric_rows:
        variant_id = str(row.get("variant_id", "")).strip()
        metric_name = str(row.get("metric_name", "")).strip()
        metric_value = _safe_float(row.get("metric_value"))
        if not variant_id or not metric_name or metric_value is None:
            continue
        metrics_by_variant[variant_id][metric_name].append(metric_value)
        primary_flag = _safe_bool(row.get("is_primary_metric"))
        if primary_flag is True:
            primary_metric_flags[variant_id][metric_name] = True
        elif metric_name not in primary_metric_flags[variant_id]:
            primary_metric_flags[variant_id][metric_name] = False

    variants: list[dict[str, Any]] = []
    idea_titles: list[str] = []
    for variant_dir in variant_dirs:
        variant_id = PurePosixPath(variant_dir).name
        variant_spec = await reader.read_json(join(variant_dir, "variant_spec.json")) or {}
        run_manifest = await reader.read_json(join(variant_dir, "run_manifest.json")) or {}
        rubric_contract = await reader.read_json(join(variant_dir, "rubric_contract.json")) or {}
        pipeline_variant = pipeline_variants.get(variant_id, {})
        variant_runs = runs_by_variant.get(variant_id, [])
        metric_summaries: dict[str, dict[str, Any]] = {}
        for metric_name, values in sorted(metrics_by_variant.get(variant_id, {}).items()):
            summary = _summary(values)
            if summary is None:
                continue
            metric_summaries[metric_name] = {
                **summary,
                "is_primary_metric": bool(
                    primary_metric_flags.get(variant_id, {}).get(metric_name, False)
                ),
            }

        title = None
        for candidate in (
            variant_spec.get("title"),
            pipeline_variant.get("title"),
            run_manifest.get("title"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                title = candidate.strip()
                break
        if title is not None:
            idea_titles.append(title)

        kind = None
        for candidate in (
            variant_spec.get("kind"),
            pipeline_variant.get("kind"),
            run_manifest.get("variant_kind"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                kind = candidate.strip()
                break

        selected_primary_metric_name: str | None = None
        if variant_runs:
            first_metric_name = str(variant_runs[0].get("primary_metric_name") or "").strip()
            selected_primary_metric_name = first_metric_name or None
        if selected_primary_metric_name is None:
            for metric_name, is_primary in primary_metric_flags.get(variant_id, {}).items():
                if is_primary:
                    selected_primary_metric_name = metric_name
                    break
        if selected_primary_metric_name is None and metric_summaries:
            selected_primary_metric_name = next(iter(metric_summaries.keys()))

        if selected_primary_metric_name and selected_primary_metric_name not in metric_summaries:
            values = [
                float(run["primary_metric_value"])
                for run in variant_runs
                if _safe_float(run.get("primary_metric_value")) is not None
            ]
            summary = _summary(values)
            if summary is not None:
                metric_summaries[selected_primary_metric_name] = {
                    **summary,
                    "is_primary_metric": True,
                }
        primary_metric_summary = (
            metric_summaries.get(selected_primary_metric_name)
            if selected_primary_metric_name
            else None
        )
        correct_run_count = sum(1 for run in variant_runs if run.get("correct") is True)
        overall_pass_count = sum(1 for run in variant_runs if run.get("overall_pass") is True)

        variant_payload = {
            "variant_id": variant_id,
            "title": title,
            "kind": kind,
            "status": run_manifest.get("status") or pipeline_variant.get("status"),
            "rubric_passed": pipeline_variant.get("rubric_passed"),
            "eval_count": run_manifest.get("eval_count", pipeline_variant.get("eval_count")),
            "exit_code": pipeline_variant.get("exit_code"),
            "variant_dir": reader.host(variant_dir),
            "task_bundle_dir": reader.host(join(variant_dir, "task_bundle"))
            if await reader.is_dir(join(variant_dir, "task_bundle"))
            else None,
            "primary_metric_name": selected_primary_metric_name,
            "primary_metric_summary": primary_metric_summary,
            "metric_summaries": metric_summaries,
            "run_count": len(variant_runs),
            "correct_run_count": correct_run_count,
            "overall_pass_count": overall_pass_count,
            "variant_spec": variant_spec,
            "run_manifest": run_manifest,
            "rubric_contract": rubric_contract,
            "target_rubrics": await reader.read_json(join(variant_dir, "target_rubrics.json")),
            "pipeline_variant_summary": pipeline_variant,
            "runs": variant_runs,
            "artifacts": {
                "variant_spec": reader.host(join(variant_dir, "variant_spec.json"))
                if await reader.is_file(join(variant_dir, "variant_spec.json"))
                else None,
                "run_manifest": reader.host(join(variant_dir, "run_manifest.json"))
                if await reader.is_file(join(variant_dir, "run_manifest.json"))
                else None,
                "rubric_contract": reader.host(join(variant_dir, "rubric_contract.json"))
                if await reader.is_file(join(variant_dir, "rubric_contract.json"))
                else None,
                "target_rubrics": reader.host(join(variant_dir, "target_rubrics.json"))
                if await reader.is_file(join(variant_dir, "target_rubrics.json"))
                else None,
                "aggregated_run_summary": reader.host(
                    join(resolved, "aggregated", "run_summary.csv")
                ),
                "aggregated_metric_values": reader.host(
                    join(resolved, "aggregated", "metric_values.csv")
                ),
                "aggregated_timeseries": reader.host(join(resolved, "aggregated", "timeseries.csv"))
                if await reader.is_file(join(resolved, "aggregated", "timeseries.csv"))
                else None,
            },
        }
        variants.append(variant_payload)

    variants.sort(key=lambda item: str(item.get("variant_id", "")))
    idea_titles = list(dict.fromkeys(title for title in idea_titles if title))

    anchor_variant_id = _choose_anchor_variant(variants)
    anchor_variant = next(
        (item for item in variants if item.get("variant_id") == anchor_variant_id), None
    )
    comparisons: list[dict[str, Any]] = []
    if anchor_variant is not None:
        anchor_metric_name = anchor_variant.get("primary_metric_name")
        anchor_summary = anchor_variant.get("primary_metric_summary")
        anchor_mean = None
        if isinstance(anchor_summary, dict):
            anchor_mean = _safe_float(anchor_summary.get("mean"))
        for variant in variants:
            variant_id = variant.get("variant_id")
            if not isinstance(variant_id, str) or variant_id == anchor_variant_id:
                continue
            variant_summary = variant.get("primary_metric_summary")
            variant_mean = None
            if isinstance(variant_summary, dict):
                variant_mean = _safe_float(variant_summary.get("mean"))
            if anchor_mean is None or variant_mean is None:
                continue
            metric_name = str(
                variant.get("primary_metric_name") or anchor_metric_name or "combined_score"
            )
            comparisons.append(
                {
                    "comparison_id": f"{anchor_variant_id}__vs__{variant_id}",
                    "baseline_variant_id": anchor_variant_id,
                    "variant_id": variant_id,
                    "metric_name": metric_name,
                    "baseline_mean": anchor_mean,
                    "variant_mean": variant_mean,
                    "mean_delta": variant_mean - anchor_mean,
                }
            )

    all_programs.sort(key=lambda row: int(row["eval_index"]))
    correct_programs = [row for row in all_programs if row.get("correct") is True]
    top_programs = sorted(
        correct_programs,
        key=lambda row: (
            -float(row["combined_score"]),
            int(row["eval_index"]),
        ),
    )[:10]
    best_correct = _best_by_score(correct_programs)
    best_any = _best_by_score(all_programs)
    improvement_trace: list[dict[str, Any]] = []

    study_id = _slug(study_id_override or PurePosixPath(resolved).name)
    completed_variants = sum(1 for item in variants if item.get("status") == "completed")
    failed_variants = sum(1 for item in variants if item.get("status") == "failed")
    timed_out_variants = sum(1 for item in variants if item.get("status") == "timeout")
    totals = {
        "ideas": len(variants),
        "attempts": len(variants),
        "evaluated_programs": len(all_programs),
        "correct_programs": len(correct_programs),
        "ablation_studies": 1,
        "ablation_variants": len(variants),
        "ablation_runs": len(all_programs),
        "completed_ablation_variants": completed_variants,
        "failed_ablation_variants": failed_variants,
        "timed_out_ablation_variants": timed_out_variants,
    }

    grounding_sources: dict[str, dict[str, Any]] = {}

    def register_source(source_id: str, payload: dict[str, Any]) -> None:
        grounding_sources[source_id] = {"id": source_id, **payload}

    if isinstance(best_correct, dict):
        register_source(
            "result.best_correct_score",
            {
                "type": "metric",
                "value": float(best_correct["combined_score"]),
                "program_dir": best_correct.get("program_dir"),
                "idea_id": best_correct.get("idea_id"),
                "attempt_id": best_correct.get("attempt_id"),
                "program_id": best_correct.get("program_id"),
            },
        )

    if isinstance(best_any, dict):
        register_source(
            "result.best_any_score",
            {
                "type": "metric",
                "value": float(best_any["combined_score"]),
                "program_dir": best_any.get("program_dir"),
                "idea_id": best_any.get("idea_id"),
                "attempt_id": best_any.get("attempt_id"),
                "program_id": best_any.get("program_id"),
            },
        )

    register_source("result.total_ideas", {"type": "count", "value": len(variants)})
    register_source("result.total_attempts", {"type": "count", "value": len(variants)})
    register_source(
        "result.total_evaluated_programs", {"type": "count", "value": len(all_programs)}
    )
    register_source(
        "result.total_correct_programs", {"type": "count", "value": len(correct_programs)}
    )
    register_source("result.total_ablation_variants", {"type": "count", "value": len(variants)})
    register_source("result.total_ablation_runs", {"type": "count", "value": len(all_programs)})

    for idx, row in enumerate(top_programs, start=1):
        register_source(
            f"result.top_program_{idx:02d}",
            {
                "type": "metric",
                "value": float(row["combined_score"]),
                "program_dir": row.get("program_dir"),
                "idea_id": row.get("idea_id"),
                "attempt_id": row.get("attempt_id"),
                "program_id": row.get("program_id"),
            },
        )

    for variant in variants:
        variant_id = str(variant.get("variant_id") or "")
        if not variant_id:
            continue
        metric_name = str(variant.get("primary_metric_name") or "combined_score")
        metric_key = _slug(metric_name)
        variant_key = _slug(variant_id)
        summary = variant.get("primary_metric_summary")
        if isinstance(summary, dict):
            for field in ("mean", "std", "min", "max"):
                value = _safe_float(summary.get(field))
                if value is None:
                    continue
                register_source(
                    f"ablation.study_{study_id}.variant_{variant_key}.{metric_key}.{field}",
                    {
                        "type": "metric",
                        "value": value,
                        "study_id": study_id,
                        "variant_id": variant_id,
                        "metric_name": metric_name,
                        "summary_field": field,
                    },
                )
            count_value = _safe_int(summary.get("n"))
            if count_value is not None:
                register_source(
                    f"ablation.study_{study_id}.variant_{variant_key}.{metric_key}.n",
                    {
                        "type": "count",
                        "value": count_value,
                        "study_id": study_id,
                        "variant_id": variant_id,
                        "metric_name": metric_name,
                    },
                )

    for comparison in comparisons:
        baseline_variant_id = str(comparison.get("baseline_variant_id") or "")
        variant_id = str(comparison.get("variant_id") or "")
        metric_name = str(comparison.get("metric_name") or "combined_score")
        delta = _safe_float(comparison.get("mean_delta"))
        if not baseline_variant_id or not variant_id or delta is None:
            continue
        register_source(
            (
                f"ablation.study_{study_id}.compare_{_slug(baseline_variant_id)}"
                f"_vs_{_slug(variant_id)}.{_slug(metric_name)}.mean_delta"
            ),
            {
                "type": "metric",
                "value": delta,
                "study_id": study_id,
                "baseline_variant_id": baseline_variant_id,
                "variant_id": variant_id,
                "metric_name": metric_name,
                "summary_field": "mean_delta",
            },
        )

    return {
        "generated_at": utc_now_string(),
        "paper_profile": paper_profile,
        "presentation_contract": _presentation_contract(paper_profile),
        "results_dir": reader.host(resolved),
        "best_correct": best_correct,
        "best_any": best_any,
        "totals": totals,
        "top_programs": top_programs,
        "improvement_trace": improvement_trace,
        "all_programs": all_programs,
        "idea_titles": idea_titles,
        "grounding_sources": grounding_sources,
        "ablation_studies": [
            {
                "study_id": study_id,
                "title": str(
                    study_title_override
                    or pipeline_summary.get("title")
                    or PurePosixPath(resolved).name
                ),
                "source_dir": reader.host(resolved),
                "pipeline_summary": pipeline_summary,
                "primary_metric_name": (
                    anchor_variant.get("primary_metric_name")
                    if isinstance(anchor_variant, dict)
                    else None
                ),
                "anchor_variant_id": anchor_variant_id,
                "variant_count": len(variants),
                "run_count": len(all_programs),
                "variants": variants,
                "comparisons": comparisons,
            }
        ],
    }


async def has_ablation_bundle(reader: ResultsReader, results_dir: str) -> bool:
    """Return whether the results dir looks like a variants-with-reps bundle."""
    return await reader.is_dir(join(results_dir, "variants")) and await reader.is_file(
        join(results_dir, "aggregated", "run_summary.csv")
    )


def _is_within_bundle_root(bundle_root: str, candidate: str) -> bool:
    root_parts = PurePosixPath(bundle_root).parts
    return PurePosixPath(candidate).parts[: len(root_parts)] == root_parts


async def _resolve_multitask_bundle_dir(
    reader: ResultsReader,
    bundle_root: str,
    *,
    task_name: str,
    task_entry: dict[str, Any],
) -> str | None:
    role = str(task_entry.get("role") or "").strip()
    if role == "primary_root_bundle" and await has_ablation_bundle(reader, bundle_root):
        return bundle_root

    task_dir = task_entry.get("task_dir")
    if isinstance(task_dir, str) and task_dir.strip():
        try:
            candidate = join(bundle_root, task_dir.strip())
        except ValueError:
            candidate = None
        if (
            candidate is not None
            and _is_within_bundle_root(bundle_root, candidate)
            and await has_ablation_bundle(reader, candidate)
        ):
            return candidate

    fallback = join(bundle_root, "extra_tasks", task_name)
    if _is_within_bundle_root(bundle_root, fallback) and await has_ablation_bundle(
        reader, fallback
    ):
        return fallback
    return None


async def _iter_multitask_bundle_dirs(
    reader: ResultsReader,
    bundle_root: str,
    manifest: dict[str, Any] | None,
) -> list[tuple[str, dict[str, Any], str]]:
    tasks = manifest.get("tasks") if isinstance(manifest, dict) else None
    if not isinstance(tasks, dict):
        return []

    resolved_entries: list[tuple[str, dict[str, Any], str]] = []
    seen_dirs: set[str] = set()
    for task_name_raw, task_entry_raw in tasks.items():
        if not isinstance(task_name_raw, str) or not isinstance(task_entry_raw, dict):
            continue
        task_name = task_name_raw.strip()
        if not task_name:
            continue
        bundle_dir = await _resolve_multitask_bundle_dir(
            reader,
            bundle_root,
            task_name=task_name,
            task_entry=task_entry_raw,
        )
        if bundle_dir is None or bundle_dir in seen_dirs:
            continue
        seen_dirs.add(bundle_dir)
        resolved_entries.append((task_name, task_entry_raw, bundle_dir))
    return resolved_entries


async def has_multitask_ablation_bundle(reader: ResultsReader, results_dir: str) -> bool:
    """Return whether the results dir declares multiple canonical task bundles."""
    manifest = await reader.read_json(join(results_dir, "paper_bundle_manifest.json"))
    return bool(await _iter_multitask_bundle_dirs(reader, results_dir, manifest))


def _merge_unique_titles(existing: list[str], additions: list[str]) -> list[str]:
    return list(
        dict.fromkeys(
            str(title).strip()
            for title in [*existing, *additions]
            if isinstance(title, str) and str(title).strip()
        )
    )


async def build_multitask_ablation_paper_data(
    reader: ResultsReader,
    results_dir: str,
    *,
    paper_profile: str = "default",
    variant_ids: Sequence[str] = (),
    evidence_reader: EvidenceReader | None = None,
) -> dict[str, Any]:
    """Build one paper_data payload from multiple canonical task bundles."""
    resolved = results_dir
    manifest = await reader.read_json(join(resolved, "paper_bundle_manifest.json")) or {}
    task_entries = await _iter_multitask_bundle_dirs(reader, resolved, manifest)
    if not task_entries:
        return await build_ablation_paper_data(
            reader,
            resolved,
            paper_profile=paper_profile,
            variant_ids=variant_ids,
            evidence_reader=evidence_reader,
        )

    studies: list[dict[str, Any]] = []
    all_programs: list[dict[str, Any]] = []
    idea_titles: list[str] = []
    grounding_sources: dict[str, dict[str, Any]] = {}
    completed_variants = 0
    failed_variants = 0
    timed_out_variants = 0

    for task_name, task_entry_raw, bundle_dir in task_entries:
        role = str(task_entry_raw.get("role") or "supporting_task").strip() or "supporting_task"
        task_payload = await build_ablation_paper_data(
            reader,
            bundle_dir,
            paper_profile=paper_profile,
            study_id_override=task_name,
            variant_ids=variant_ids if bundle_dir == resolved else (),
            evidence_reader=evidence_reader,
        )
        task_studies = task_payload.get("ablation_studies")
        if not isinstance(task_studies, list) or not task_studies:
            continue
        study = dict(task_studies[0])
        study_id = str(study.get("study_id") or task_name).strip() or task_name
        study["task_name"] = task_name
        study["task_role"] = role
        study["manifest_entry"] = task_entry_raw
        studies.append(study)

        idea_titles = _merge_unique_titles(
            idea_titles,
            [str(title) for title in task_payload.get("idea_titles", []) if isinstance(title, str)],
        )

        for row in task_payload.get("all_programs", []):
            if not isinstance(row, dict):
                continue
            payload = dict(row)
            variant_id = str(payload.get("variant_id") or payload.get("idea_id") or "").strip()
            run_id = str(payload.get("run_id") or payload.get("program_id") or "").strip()
            namespaced_prefix = f"{study_id}:{variant_id}" if variant_id else study_id
            payload["study_id"] = study_id
            payload["task_name"] = task_name
            payload["task_role"] = role
            payload["idea_id"] = namespaced_prefix
            payload["idea_dir_name"] = namespaced_prefix
            if run_id:
                payload["attempt_id"] = f"{namespaced_prefix}:{run_id}"
                payload["program_id"] = f"{namespaced_prefix}:{run_id}"
            else:
                payload["attempt_id"] = namespaced_prefix
                payload["program_id"] = namespaced_prefix
            all_programs.append(payload)

        task_sources = task_payload.get("grounding_sources")
        if isinstance(task_sources, dict):
            for source_id, payload in task_sources.items():
                if not isinstance(payload, dict):
                    continue
                source_key = str(source_id)
                if source_key.startswith("ablation."):
                    grounding_sources[source_key] = dict(payload)

        task_totals = task_payload.get("totals")
        if isinstance(task_totals, dict):
            completed_variants += int(task_totals.get("completed_ablation_variants", 0) or 0)
            failed_variants += int(task_totals.get("failed_ablation_variants", 0) or 0)
            timed_out_variants += int(task_totals.get("timed_out_ablation_variants", 0) or 0)

    all_programs.sort(
        key=lambda row: (str(row.get("study_id") or ""), int(row.get("eval_index", 10**9)))
    )
    for eval_index, row in enumerate(all_programs, start=1):
        row["eval_index"] = eval_index

    correct_programs = [row for row in all_programs if row.get("correct") is True]
    studies.sort(
        key=lambda row: (
            str(row.get("task_role") != "primary_root_bundle"),
            str(row.get("study_id") or ""),
        )
    )

    totals = {
        "ideas": sum(int(study.get("variant_count") or 0) for study in studies),
        "attempts": sum(int(study.get("variant_count") or 0) for study in studies),
        "evaluated_programs": len(all_programs),
        "correct_programs": len(correct_programs),
        "ablation_studies": len(studies),
        "ablation_variants": sum(int(study.get("variant_count") or 0) for study in studies),
        "ablation_runs": sum(int(study.get("run_count") or 0) for study in studies),
        "completed_ablation_variants": completed_variants,
        "failed_ablation_variants": failed_variants,
        "timed_out_ablation_variants": timed_out_variants,
    }

    grounding_sources["result.total_ideas"] = {
        "id": "result.total_ideas",
        "type": "count",
        "value": totals["ideas"],
    }
    grounding_sources["result.total_attempts"] = {
        "id": "result.total_attempts",
        "type": "count",
        "value": totals["attempts"],
    }
    grounding_sources["result.total_evaluated_programs"] = {
        "id": "result.total_evaluated_programs",
        "type": "count",
        "value": totals["evaluated_programs"],
    }
    grounding_sources["result.total_correct_programs"] = {
        "id": "result.total_correct_programs",
        "type": "count",
        "value": totals["correct_programs"],
    }
    grounding_sources["result.total_ablation_variants"] = {
        "id": "result.total_ablation_variants",
        "type": "count",
        "value": totals["ablation_variants"],
    }
    grounding_sources["result.total_ablation_runs"] = {
        "id": "result.total_ablation_runs",
        "type": "count",
        "value": totals["ablation_runs"],
    }
    grounding_sources["result.total_ablation_studies"] = {
        "id": "result.total_ablation_studies",
        "type": "count",
        "value": totals["ablation_studies"],
    }

    for study in studies:
        study_id = str(study.get("study_id") or "").strip()
        if not study_id:
            continue
        grounding_sources[f"ablation.study_{study_id}.variant_count"] = {
            "id": f"ablation.study_{study_id}.variant_count",
            "type": "count",
            "value": int(study.get("variant_count") or 0),
            "study_id": study_id,
        }
        grounding_sources[f"ablation.study_{study_id}.run_count"] = {
            "id": f"ablation.study_{study_id}.run_count",
            "type": "count",
            "value": int(study.get("run_count") or 0),
            "study_id": study_id,
        }

    return {
        "generated_at": utc_now_string(),
        "paper_profile": paper_profile,
        "presentation_contract": _presentation_contract(paper_profile),
        "results_dir": reader.host(resolved),
        "best_correct": None,
        "best_any": None,
        "totals": totals,
        "top_programs": [],
        "improvement_trace": [],
        "all_programs": all_programs,
        "idea_titles": idea_titles,
        "grounding_sources": grounding_sources,
        "ablation_studies": studies,
        "paper_bundle_manifest": manifest,
    }


POSTHOC_RELEVANCE_VALUES = frozenset({"high", "medium", "low"})
POSTHOC_USAGE_VALUES = frozenset({"main_results", "supporting", "appendix", "skip"})
MAX_POSTHOC_FILE_BYTES = 5 * 1024 * 1024
MAX_POSTHOC_ANALYSIS_BYTES = 20 * 1024 * 1024
MAX_POSTHOC_CONTEXT_BYTES = 50 * 1024 * 1024
MAX_POSTHOC_ANALYSES = 16
MAX_POSTHOC_DATA_FILES = 32
MAX_POSTHOC_CSV_ROWS = 10_000
MAX_POSTHOC_CSV_COLUMNS = 100
MAX_POSTHOC_CSV_CELL_CHARS = 10_000


async def _posthoc_json(reader: ResultsReader, path: str) -> dict[str, Any] | None:
    """Read a JSON file, returning None on missing or invalid data."""
    entry = await reader.workspace.stat(path)
    if entry is None or entry.kind != "file" or entry.size > MAX_POSTHOC_FILE_BYTES:
        return None
    try:
        payload = json.loads((await reader.workspace.read_bytes(path)).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    return payload if isinstance(payload, dict) else None


async def _posthoc_text(reader: ResultsReader, path: str) -> str | None:
    """Read a text file, returning None on missing or error."""
    entry = await reader.workspace.stat(path)
    if entry is None or entry.kind != "file" or entry.size > MAX_POSTHOC_FILE_BYTES:
        return None
    try:
        return (await reader.workspace.read_bytes(path)).decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


async def _is_posthoc_session(reader: ResultsReader, path: str) -> bool:
    ranking = await reader.workspace.stat(join(path, "ranking.json"))
    analyses = await reader.workspace.stat(join(path, "analyses"))
    return (
        await reader.is_dir(path)
        and ranking is not None
        and ranking.kind == "file"
        and analyses is not None
        and analyses.kind == "dir"
    )


async def discover_posthoc_session(
    reader: ResultsReader, results_dir: str, posthoc_dir: str | None = None
) -> str | None:
    """Find the posthoc analysis directory containing a valid ranking.json.

    Expects the layout ``results_dir/posthoc/{ranking.json, analyses/}`` (or ``posthoc_dir``
    itself, or the ``output_dir`` that ``posthoc/latest_manifest.json`` names). Returns the
    ``posthoc/`` directory, or ``None`` if it does not exist.
    """
    if posthoc_dir is not None:
        return posthoc_dir if await _is_posthoc_session(reader, posthoc_dir) else None

    posthoc_root = join(results_dir, "posthoc")
    entry = await reader.workspace.stat(posthoc_root)
    if entry is None or entry.kind == "symlink":
        return None
    stable_manifest = await _posthoc_json(reader, join(posthoc_root, "latest_manifest.json"))
    if stable_manifest is not None:
        output_dir = stable_manifest.get("output_dir")
        if isinstance(output_dir, str) and output_dir:
            try:
                candidate = (
                    join(posthoc_root, output_dir) if not output_dir.startswith("/") else None
                )
            except ValueError:
                candidate = None
            if (
                candidate is not None
                and _is_within_bundle_root(posthoc_root, candidate)
                and await _is_posthoc_session(reader, candidate)
            ):
                return candidate
    return posthoc_root if await _is_posthoc_session(reader, posthoc_root) else None


def _validate_path_component(component: str, name: str) -> str:
    """Validate an untrusted ranking.json path component."""
    if not component or component in {".", ".."} or "/" in component or "\\" in component:
        raise ValueError(f"Invalid {name} in ranking.json: {component!r}")
    return component


async def _resolve_attempt_dir(
    reader: ResultsReader, posthoc_root: str, analysis_id: str, selected_attempt: str
) -> str:
    """Resolve the attempt directory structurally, rejecting traversal from ranking.json."""
    analyses_root = join(posthoc_root, "analyses")
    safe_analysis_id = _validate_path_component(analysis_id, "analysis_id")
    safe_selected_attempt = _validate_path_component(selected_attempt, "selected_attempt")
    unresolved_analysis = join(analyses_root, safe_analysis_id)
    unresolved_attempt = join(unresolved_analysis, safe_selected_attempt)
    for path in (unresolved_analysis, unresolved_attempt):
        entry = await reader.workspace.stat(path)
        if entry is not None and entry.kind == "symlink":
            raise ValueError("Attempt directory must not be a symlink")
    return unresolved_attempt


async def _list_directory_files(reader: ResultsReader, directory: str) -> list[tuple[str, int]]:
    """List the regular files directly inside ``directory`` as ``(name, size)``."""
    if not await reader.is_dir(directory):
        return []
    entries = await reader.workspace.list_tree(directory)
    return sorted(
        (entry.path, entry.size)
        for entry in entries
        if entry.kind == "file" and "/" not in entry.path
    )


async def _collect_png_paths(reader: ResultsReader, directory: str) -> list[str]:
    """List PNG files in a directory as host path strings."""
    return sorted(
        reader.host(join(directory, name))
        for name, size in await _list_directory_files(reader, directory)
        if PurePosixPath(name).suffix.lower() == ".png" and size <= MAX_POSTHOC_FILE_BYTES
    )


async def _resolve_declared_file(
    reader: ResultsReader, directory: str, relpath: object
) -> str | None:
    if not isinstance(relpath, str) or not relpath:
        return None
    raw_path = PurePosixPath(relpath)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        return None
    current = directory
    for component in raw_path.parts:
        current = join(current, component)
        entry = await reader.workspace.stat(current)
        if entry is not None and entry.kind == "symlink":
            return None
    entry = await reader.workspace.stat(current)
    if entry is None or entry.kind != "file" or entry.size > MAX_POSTHOC_FILE_BYTES:
        return None
    return current


def _manifest_output_paths(manifest: dict[str, Any], key: str) -> list[object]:
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict):
        return []
    values = outputs.get(key, [])
    return values if isinstance(values, list) else []


async def _collect_manifest_figures(
    reader: ResultsReader, directory: str, manifest: dict[str, Any]
) -> list[str]:
    figures: list[str] = []
    for relpath in _manifest_output_paths(manifest, "figures")[:MAX_POSTHOC_DATA_FILES]:
        path = await _resolve_declared_file(reader, directory, relpath)
        if path is not None and PurePosixPath(path).suffix.lower() == ".png":
            figures.append(reader.host(path))
    return sorted(figures)


async def _collect_manifest_data(
    reader: ResultsReader, directory: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    data: dict[str, Any] = {}
    total_bytes = 0
    for relpath in _manifest_output_paths(manifest, "data_files")[:MAX_POSTHOC_DATA_FILES]:
        path = await _resolve_declared_file(reader, directory, relpath)
        if path is None or PurePosixPath(path).suffix.lower() not in {".json", ".csv"}:
            continue
        entry = await reader.workspace.stat(path)
        total_bytes += entry.size if entry is not None else 0
        if total_bytes > MAX_POSTHOC_ANALYSIS_BYTES:
            break
        try:
            raw = (await reader.workspace.read_bytes(path)).decode("utf-8")
            if PurePosixPath(path).suffix.lower() == ".json":
                payload = json.loads(raw)
            else:
                csv_reader = csv.DictReader(io.StringIO(raw, newline=""))
                if len(csv_reader.fieldnames or []) > MAX_POSTHOC_CSV_COLUMNS:
                    continue
                rows: list[dict[str, str | None]] = []
                for row_index, row in enumerate(csv_reader):
                    if row_index >= MAX_POSTHOC_CSV_ROWS:
                        rows = []
                        break
                    if any(len(str(value)) > MAX_POSTHOC_CSV_CELL_CHARS for value in row.values()):
                        rows = []
                        break
                    rows.append(row)
                payload = rows
        except (OSError, UnicodeDecodeError, ValueError, csv.Error, RecursionError):
            continue
        data[str(relpath)] = payload
    return data


# Infrastructure / control JSON files that are NOT raw analysis data.
# These are either loaded into dedicated fields (e.g. rubric) or are
# pipeline bookkeeping and should not appear in ``raw_data``.
_CONTROL_JSON_NAMES: frozenset[str] = frozenset(
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


async def _collect_data_files(reader: ResultsReader, directory: str) -> dict[str, Any]:
    """Read raw-data JSON files in *directory*, skipping control/metadata files.

    Returns a dict mapping ``filename_stem`` -> parsed JSON content.
    """
    data: dict[str, Any] = {}
    total_bytes = 0
    file_count = 0
    for name, size in await _list_directory_files(reader, directory):
        if PurePosixPath(name).suffix.lower() != ".json":
            continue
        if name in _CONTROL_JSON_NAMES:
            continue
        file_count += 1
        total_bytes += size
        if file_count > MAX_POSTHOC_DATA_FILES or total_bytes > MAX_POSTHOC_ANALYSIS_BYTES:
            break
        payload = await _posthoc_json(reader, join(directory, name))
        if payload is not None:
            data[PurePosixPath(name).stem] = payload
    return data


async def collect_posthoc_analyses(
    reader: ResultsReader, results_dir: str, posthoc_dir: str | None = None
) -> list[dict[str, Any]]:
    """Load all valid posthoc analyses of a posthoc session directory.

    Returns a list of dicts, each containing:
    - ``analysis_id``, ``title``, ``family``
    - ``summary_text``: the full markdown summary
    - ``rubric``: the parsed interestingness_rubric.json dict
    - ``figure_paths``: list of absolute PNG paths of the attempt
    - ``raw_data``, ``run_analysis_code`` and ``attempt_dir``
    """
    posthoc_root = await discover_posthoc_session(reader, results_dir, posthoc_dir)
    if posthoc_root is None:
        return []

    ranking = await _posthoc_json(reader, join(posthoc_root, "ranking.json"))
    if ranking is None:
        return []

    valid_entries = ranking.get("valid")
    if not isinstance(valid_entries, list):
        return []

    analyses: list[dict[str, Any]] = []
    context_bytes = 0
    for entry in valid_entries[:MAX_POSTHOC_ANALYSES]:
        if not isinstance(entry, dict):
            continue
        analysis_id = entry.get("analysis_id")
        selected_attempt = entry.get("selected_attempt")
        if not isinstance(analysis_id, str) or not isinstance(selected_attempt, str):
            continue

        try:
            attempt_dir = await _resolve_attempt_dir(
                reader, posthoc_root, analysis_id, selected_attempt
            )
        except ValueError:
            continue
        if not await reader.is_dir(attempt_dir):
            continue

        summary_text = await _posthoc_text(reader, join(attempt_dir, "analysis_summary.md"))
        if not summary_text:
            continue

        rubric = await _posthoc_json(reader, join(attempt_dir, "interestingness_rubric.json")) or {}
        manifest_path = join(attempt_dir, "analysis_manifest.json")
        manifest_present = await reader.workspace.exists(manifest_path)
        manifest = await _posthoc_json(reader, manifest_path) or {}
        figure_paths = await _collect_manifest_figures(reader, attempt_dir, manifest)
        if not manifest_present:
            figure_paths = await _collect_png_paths(reader, attempt_dir)
        raw_data = await _collect_manifest_data(reader, attempt_dir, manifest)
        if not manifest_present:
            raw_data = await _collect_data_files(reader, attempt_dir)
        run_analysis_code = await _posthoc_text(reader, join(attempt_dir, "run_analysis.py")) or ""

        analysis = {
            "analysis_id": analysis_id,
            "title": entry.get("title", analysis_id),
            "family": entry.get("family", ""),
            "summary_text": summary_text,
            "rubric": rubric,
            "figure_paths": figure_paths,
            "raw_data": raw_data,
            "run_analysis_code": run_analysis_code,
            "attempt_dir": reader.host(attempt_dir),
        }
        analysis_bytes = len(json.dumps(analysis, ensure_ascii=False).encode("utf-8"))
        if analysis_bytes > MAX_POSTHOC_ANALYSIS_BYTES:
            continue
        context_bytes += analysis_bytes
        if context_bytes > MAX_POSTHOC_CONTEXT_BYTES:
            break
        analyses.append(analysis)

    return analyses


def copy_posthoc_figures(analyses: Sequence[Mapping[str, Any]]) -> list[str]:
    """Skip pre-computed PNGs: the paper writer generates its own figures from raw data.

    Returns the ids of the analyses that have pre-computed PNGs, which are intentionally not
    copied (upstream logs a warning for each).
    """
    return [
        str(analysis.get("analysis_id", ""))
        for analysis in analyses
        if analysis.get("figure_paths")
    ]


def validate_posthoc_ranking(
    ranking_result: dict[str, Any], analyses: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate and rank every collected analysis exactly once."""
    errors: list[str] = []
    expected_ids: list[str] = []
    for index, analysis in enumerate(analyses):
        analysis_id = analysis.get("analysis_id")
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            errors.append(f"analyses[{index}].analysis_id must be a non-empty string")
            continue
        expected_ids.append(analysis_id)
    duplicate_source_ids = sorted(
        analysis_id for analysis_id in set(expected_ids) if expected_ids.count(analysis_id) > 1
    )
    if duplicate_source_ids:
        errors.append(f"collected analyses contain duplicate IDs: {duplicate_source_ids}")

    rationale = ranking_result.get("ranking_rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("ranking_rationale must be a non-empty string")

    ranked_entries = ranking_result.get("ranked_analyses")
    if not isinstance(ranked_entries, list):
        errors.append("ranked_analyses must be a list")
        raise ValueError("Invalid posthoc ranking: " + "; ".join(errors))

    ranked_ids: list[str] = []
    ranks: list[int] = []
    validated_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(ranked_entries):
        if not isinstance(entry, dict):
            errors.append(f"ranked_analyses[{index}] must be an object")
            continue
        validated_entries.append(entry)
        analysis_id = entry.get("analysis_id")
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            errors.append(f"ranked_analyses[{index}].analysis_id must be a non-empty string")
        else:
            ranked_ids.append(analysis_id)

        rank = entry.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool):
            errors.append(f"ranked_analyses[{index}].rank must be an integer")
        else:
            ranks.append(rank)

        relevance = entry.get("relevance_to_thesis")
        if relevance not in POSTHOC_RELEVANCE_VALUES:
            errors.append(
                f"ranked_analyses[{index}].relevance_to_thesis must be one of "
                f"{sorted(POSTHOC_RELEVANCE_VALUES)}"
            )
        usage = entry.get("recommended_usage")
        if usage not in POSTHOC_USAGE_VALUES:
            errors.append(
                f"ranked_analyses[{index}].recommended_usage must be one of "
                f"{sorted(POSTHOC_USAGE_VALUES)}"
            )
        for field in ("relevance_justification", "key_finding_for_paper"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"ranked_analyses[{index}].{field} must be a non-empty string")

    duplicate_ranked_ids = sorted(
        analysis_id for analysis_id in set(ranked_ids) if ranked_ids.count(analysis_id) > 1
    )
    if duplicate_ranked_ids:
        errors.append(f"duplicate analysis IDs: {duplicate_ranked_ids}")
    expected_id_set = set(expected_ids)
    ranked_id_set = set(ranked_ids)
    unknown_ids = sorted(ranked_id_set - expected_id_set)
    missing_ids = sorted(expected_id_set - ranked_id_set)
    if unknown_ids:
        errors.append(f"unknown analysis IDs: {unknown_ids}")
    if missing_ids:
        errors.append(f"missing analysis IDs: {missing_ids}")
    if sorted(ranks) != list(range(1, len(expected_ids) + 1)):
        errors.append(f"ranks must be the contiguous integers 1..{len(expected_ids)} exactly once")

    if errors:
        raise ValueError("Invalid posthoc ranking: " + "; ".join(errors))
    return sorted(validated_entries, key=lambda entry: entry["rank"])


def augment_paper_data_with_posthoc(
    paper_data: dict[str, Any],
    ranking_result: dict[str, Any],
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Add ranked posthoc analyses without changing existing evidence.

    ``ranking_result`` is the parsed ``posthoc_ranking.json`` produced by the
    ranking agent.  ``analyses`` is the output of ``collect_posthoc_analyses``.
    Existing inventory entries, totals, and grounding records remain authoritative.
    """
    analyses_by_id = {a["analysis_id"]: a for a in analyses}

    ranked_entries = validate_posthoc_ranking(ranking_result, analyses)

    existing_posthoc = paper_data.get("posthoc_analyses")
    posthoc_list = list(existing_posthoc) if isinstance(existing_posthoc, list) else []
    existing_ids = {
        item.get("analysis_id")
        for item in posthoc_list
        if isinstance(item, dict) and isinstance(item.get("analysis_id"), str)
    }
    for entry in ranked_entries:
        aid = entry["analysis_id"]
        if aid in existing_ids:
            continue
        source = analyses_by_id[aid]
        posthoc_list.append(
            {
                "rank": entry.get("rank"),
                "analysis_id": aid,
                "title": source.get("title", aid),
                "family": source.get("family", ""),
                "summary_text": source.get("summary_text", ""),
                "rubric": source.get("rubric", {}),
                "relevance_to_thesis": entry.get("relevance_to_thesis", ""),
                "relevance_justification": entry.get("relevance_justification", ""),
                "recommended_usage": entry.get("recommended_usage", ""),
                "key_finding_for_paper": entry.get("key_finding_for_paper", ""),
                "figure_paths": source.get("figure_paths", []),
                "raw_data": source.get("raw_data", {}),
                "run_analysis_code": source.get("run_analysis_code", ""),
            }
        )

    paper_data["posthoc_analyses"] = posthoc_list

    # Update totals
    totals = paper_data.get("totals")
    if isinstance(totals, dict):
        totals.setdefault("posthoc_analyses_total", len(analyses))
        totals.setdefault("posthoc_analyses_ranked", len(posthoc_list))

    # Register grounding sources
    grounding_sources = paper_data.get("grounding_sources")
    if isinstance(grounding_sources, dict):
        for item in posthoc_list:
            if not isinstance(item, dict):
                continue
            aid = item.get("analysis_id")
            if not isinstance(aid, str) or not aid:
                continue
            grounding_sources.setdefault(
                f"posthoc.{aid}.summary",
                {
                    "id": f"posthoc.{aid}.summary",
                    "type": "posthoc_analysis",
                    "analysis_id": aid,
                    "title": item.get("title", ""),
                    "family": item.get("family", ""),
                },
            )
            # Register each raw data file as a grounding source.
            raw_data = item.get("raw_data")
            if not isinstance(raw_data, dict):
                continue
            for data_stem in raw_data:
                if not isinstance(data_stem, str) or not data_stem:
                    continue
                source_id = f"posthoc.{aid}.data.{data_stem}"
                grounding_sources.setdefault(
                    source_id,
                    {
                        "id": source_id,
                        "type": "posthoc_data",
                        "analysis_id": aid,
                        "data_file": data_stem,
                    },
                )

    return paper_data


# --- the presentation contract --------------------------------------------------------------


def _presentation_contract(paper_profile: str) -> dict[str, Any]:
    """Return profile-specific writing guidance embedded in paper_data.json."""
    if paper_profile == "standalone":
        return {
            "variant_unit": "model_variant",
            "run_unit": "seeded_run",
            "internal_fields": ["correct", "eligible"],
            "forbidden_visible_terms": [
                "eligible",
                "correct and eligible",
                "correctness rate",
                "worker_01",
                "worker_02",
                "combined_score",
                "rubric",
            ],
            "preferred_terms": {
                "idea": "model variant",
                "ideas": "model variants",
                "program": "seeded run",
                "programs": "seeded runs",
                "implementation": "configuration",
                "implementations": "configurations",
            },
            "threshold_rate_rule": (
                "If a thresholded rate appears in the visible paper, keep one reader-facing "
                "meaning per figure, such as target attainment, successful-run fraction, or "
                "non-collapse rate. Do not reuse a single label for different task-specific thresholds."
            ),
        }
    return {
        "variant_unit": "idea",
        "run_unit": "program",
        "internal_fields": ["correct", "eligible"],
    }


# --- study context and the sealed final evaluation ------------------------------------------

STUDY_SELECTION_KEYS = (
    "selection_kind",
    "investigation_outcome",
    "combined_score",
    "baseline_score",
    "source_attempt_id",
    "source_program_dir",
    "program_dir",
    "entrypoint_path",
    "program_tree_sha256",
    "source_tree_sha256",
    "baseline_program_dir",
    "baseline_tree_sha256",
    "attempt_counts",
)
"""The keys of a frozen downstream selection that the study context carries."""


def merge_study_context(paper_data: dict[str, Any], selection: Mapping[str, Any] | None) -> None:
    """Attach only host-verified selection facts, independently of final evidence."""
    if selection is None:
        return
    context = {key: selection[key] for key in STUDY_SELECTION_KEYS}
    baseline = context["baseline_score"] if context["baseline_program_dir"] is not None else None
    context["baseline_score"] = baseline
    context["search_score_delta"] = (
        context["combined_score"] - baseline if baseline is not None else None
    )
    context["final_evaluation"] = None
    context["limitations"] = [
        "Investigation found no qualifying winner; selection used search evidence only.",
        "Unscored attempts are missing performance evidence, not measured losses.",
        "Search scores and sealed final metrics belong to different evaluation scopes.",
        "An unqualified final outcome may reflect validity or other criteria; do not infer "
        "that failure means only an inferior score. Interpret metrics within those limitations.",
    ]
    if baseline is None:
        context["limitations"].append(
            "A valid scored baseline is unavailable; no baseline performance comparison is supported."
        )
    paper_data["downstream_study"] = context
    grounding = paper_data.setdefault("grounding_sources", {})
    facts = {
        "selection_kind": context["selection_kind"],
        "investigation_outcome": context["investigation_outcome"],
        "search_score": context["combined_score"],
        "baseline_score": baseline,
        "search_score_delta": context["search_score_delta"],
        **{f"attempts.{name}": value for name, value in context["attempt_counts"].items()},
    }
    for name, value in facts.items():
        if value is None:
            continue
        source_id = f"study.{name}"
        grounding[source_id] = {
            "id": source_id,
            "type": "study_context",
            "value": value,
            "evaluation_scope": "search",
            "program_tree_sha256": context["program_tree_sha256"],
        }


def merge_study_final_outcome(paper_data: dict[str, Any], manifest: Mapping[str, Any]) -> None:
    """Preserve qualification status without treating it as a validity diagnosis."""
    context = paper_data.get("downstream_study")
    if not isinstance(context, dict):
        return
    outcome = {
        "status": manifest["status"],
        "correct": manifest["correct"],
        "outcome": manifest.get("outcome", "qualified"),
    }
    context["final_evaluation"] = outcome
    for name, value in outcome.items():
        source_id = f"study.final.{name}"
        paper_data["grounding_sources"][source_id] = {
            "id": source_id,
            "type": "study_context",
            "value": value,
            "evaluation_scope": "sealed_final",
            "program_tree_sha256": manifest["program_tree_sha256"],
            "results_tree_sha256": manifest["results_tree_sha256"],
        }


def ground_final_outcome(paper_data: dict[str, Any]) -> None:
    """Expose verified qualification facts independently of scientific metrics."""
    final = paper_data["final_evaluation"]
    if final["correct"] is not True:
        final["limitations"] = [
            "Final evaluation completed but did not qualify; search success does not "
            "establish final qualification.",
            "Non-qualification may reflect performance, uncertainty, validity, adaptivity, "
            "or other criteria; report only causes supported by evaluator evidence.",
            "Retained metrics do not independently establish claims whose qualification "
            "or validity checks failed. Missing gate diagnostics remain unknown.",
        ]
    grounding = paper_data.setdefault("grounding_sources", {})
    for name in ("status", "correct", "outcome"):
        source_id = f"final.qualification.{name}"
        grounding[source_id] = {
            "id": source_id,
            "type": "final_qualification",
            "value": final[name],
            "evaluation_scope": "sealed_final",
            "manifest_path": final["manifest_path"],
            "program_tree_sha256": final["program_tree_sha256"],
            "results_tree_sha256": final["results_tree_sha256"],
        }


def check_final_outcome_policy(
    paper_data: Mapping[str, Any], policy: FinalOutcomePolicy = "qualified_only"
) -> None:
    """Apply the final-outcome gate to paper data.

    A verified ``not_qualified`` final evaluation may support a paper only under
    ``allow_not_qualified`` (or in a downstream study, which frames its own limitations).

    Raises:
        PaperDataError: When the paper data holds a final evaluation that did not qualify and
            the policy does not allow it.
    """
    final = paper_data.get("final_evaluation")
    if not isinstance(final, Mapping):
        return
    if (
        final.get("correct") is not True
        and "downstream_study" not in paper_data
        and policy != "allow_not_qualified"
    ):
        raise PaperDataError(
            f"Final evaluation did not qualify for publication: {final.get('manifest_path')}"
        )


async def merge_final_evaluation(
    paper_data: dict[str, Any],
    final: FinalEvaluationResult,
    *,
    reader: ResultsReader,
    final_outcome_policy: FinalOutcomePolicy = "qualified_only",
    evidence_reader: EvidenceReader | None = None,
) -> None:
    """Register completed sealed-final metrics as publication grounding.

    ``final`` is the verified result of the final evaluation (its once-only and digest rules are
    the final-evaluation phase's); this function binds it into the paper data.

    Raises:
        PaperDataError: When the final evaluation did not qualify and the policy does not allow
            a paper on it, or when its metric names collide after grounding-id normalisation.
    """
    out = final.output_dir or ""
    manifest_path = reader.host(join(out, "final_manifest.json")) if out else ""
    if (
        final.correct is not True
        and "downstream_study" not in paper_data
        and final_outcome_policy != "allow_not_qualified"
    ):
        raise PaperDataError(f"Final evaluation did not qualify for publication: {manifest_path}")
    manifest: dict[str, Any] = {
        "status": final.status,
        "correct": final.correct,
        "outcome": final.outcome,
        "metrics": dict(final.metrics),
        "task_tree_sha256": final.task_tree_sha256,
        "evaluator_sha256": final.evaluator_sha256,
        "program_tree_sha256": final.program_tree_sha256,
        "results_tree_sha256": final.results_tree_sha256,
    }
    merge_study_final_outcome(paper_data, manifest)
    results_dir = join(out, "results") if out else ""
    evidence = (
        await paper_evaluation_evidence(reader, out, results_dir, evidence_reader=evidence_reader)
        if out
        else {"status": "unavailable"}
    )
    paper_data["final_evaluation"] = {
        "status": manifest.get("status", "completed"),
        "correct": manifest["correct"],
        "outcome": manifest.get("outcome", "qualified"),
        "metrics": manifest["metrics"],
        "task_tree_sha256": manifest["task_tree_sha256"],
        "evaluator_sha256": manifest["evaluator_sha256"],
        "program_tree_sha256": manifest["program_tree_sha256"],
        "results_tree_sha256": manifest["results_tree_sha256"],
        "manifest_path": manifest_path,
        "evaluation_evidence": evidence,
    }
    ground_final_outcome(paper_data)
    grounding = paper_data.get("grounding_sources")
    if not isinstance(grounding, dict):
        grounding = {}
        paper_data["grounding_sources"] = grounding
    if evidence["status"] == "available":
        source_id = "final.evidence.episodes"
        grounding[source_id] = {
            "id": source_id,
            "type": "evaluation_evidence",
            "evaluation_scope": "sealed_final",
            "path": evidence["path"],
            "sha256": evidence["sha256"],
            "json_pointer": "/episodes",
            "episode_count": evidence["episode_count"],
            "candidate_diagnostics_trust": evidence["candidate_diagnostics_trust"],
            "manifest_path": manifest_path,
            "program_tree_sha256": manifest["program_tree_sha256"],
            "results_tree_sha256": manifest["results_tree_sha256"],
        }
    normalized_metric_names: set[str] = set()
    for name, value in manifest["metrics"].items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        safe_name = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
        if not safe_name or safe_name in normalized_metric_names:
            raise PaperDataError("Final metric names collide after grounding-ID normalization")
        normalized_metric_names.add(safe_name)
        source_id = f"final.metric.{safe_name}"
        grounding[source_id] = {
            "id": source_id,
            "type": "metric",
            "value": value,
            "evaluation_scope": "sealed_final",
            "manifest_path": manifest_path,
            "results_tree_sha256": manifest["results_tree_sha256"],
        }


async def load_downstream_selection(
    reader: ResultsReader, downstream: DownstreamSelection | None
) -> dict[str, Any] | None:
    """Read the frozen selection record that ``downstream`` names (``None`` when there is none).

    The record is the ``.downstream_selection.json`` of the downstream-selection step, already
    verified there.

    Raises:
        PaperDataError: When a selected program's record is missing or lacks a key the study
            context carries.
    """
    if downstream is None or downstream.status != "selected" or not downstream.selection_path:
        return None
    selection = await reader.read_json(downstream.selection_path)
    if selection is None:
        raise PaperDataError(
            f"invalid downstream selection manifest: {reader.host(downstream.selection_path)}"
        )
    missing = [key for key in STUDY_SELECTION_KEYS if key not in selection]
    if missing:
        raise PaperDataError(f"downstream selection lacks keys: {missing}")
    return selection


# --- staged study sources --------------------------------------------------------------------

MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_SOURCE_ENTRIES = 5000
CODE_EXCLUDES = (
    *(f"/{name}" for name in sorted(NONOVERRIDABLE_EXCLUDES)),
    "/.evaluation_stage_*",
    "__pycache__",
    "*.pyc",
    ".venv",
    "node_modules",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "dist",
    "build",
)
"""The patterns that keep a staged source tree to code (upstream ``_CODE_EXCLUDES``)."""


async def _read_source_tree(reader: ResultsReader, directory: str) -> dict[str, bytes]:
    """Read a source tree as ``{relative path: bytes}``, refusing non-regular entries."""
    tree: dict[str, bytes] = {}
    remaining = MAX_SOURCE_BYTES
    entries = await reader.workspace.list_tree(directory)
    if len(entries) > MAX_SOURCE_ENTRIES:
        raise ValueError("paper source entry limit exceeded")
    for entry in entries:
        if entry.kind == "dir":
            continue
        if entry.kind != "file":
            raise ValueError(f"paper source contains a non-regular entry: {entry.path}")
        if entry.size > remaining:
            raise ValueError("paper source byte limit exceeded")
        data = await reader.workspace.read_bytes(join(directory, entry.path))
        remaining -= len(data)
        tree[entry.path] = data
    return tree


def _code_only(tree: Mapping[str, bytes]) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    ignored_dirs: set[str] = set()
    for relative, data in sorted(tree.items()):
        path = PurePosixPath(relative)
        skip = False
        for depth in range(1, len(path.parts)):
            parent = "/".join(path.parts[:depth])
            if parent in ignored_dirs:
                skip = True
                break
            if task_copy_path_is_ignored(parent, is_dir=True, patterns=CODE_EXCLUDES):
                ignored_dirs.add(parent)
                skip = True
                break
        if skip or task_copy_path_is_ignored(relative, is_dir=False, patterns=CODE_EXCLUDES):
            continue
        result[relative] = data
    return result


async def stage_study_sources(
    paper_data: dict[str, Any],
    workspace: Workspace,
    output_dir: str,
    selection: Mapping[str, Any] | None,
) -> None:
    """Preserve verified code at stable relative paths; never overwrite prior evidence.

    Copies the downstream program (and its baseline) into
    ``<output_dir>/source_evidence/downstream/<role>/`` (code files only) after checking their
    tree digests against the frozen selection, and records the paths in the study context.
    Staging the same sources again is a no-op; different content is refused.

    Raises:
        ValueError: When the sources no longer match the study context or the frozen selection.
    """
    context = paper_data.get("downstream_study")
    if not isinstance(context, dict):
        return
    if selection is None or any(
        context.get(key) != selection.get(key)
        for key in (
            "program_dir",
            "program_tree_sha256",
            "baseline_program_dir",
            "baseline_tree_sha256",
        )
    ):
        raise ValueError("paper source selection no longer matches study context")
    reader = ResultsReader(workspace)
    combined: dict[str, bytes] = {}
    paths: dict[str, str] = {}
    for role, source_key, digest_key in (
        ("program", "program_dir", "program_tree_sha256"),
        ("baseline", "baseline_program_dir", "baseline_tree_sha256"),
    ):
        if selection.get(source_key) is None:
            continue
        source = str(selection[source_key])
        if await workspace.digest(source) != selection[digest_key]:
            raise ValueError(f"paper {role} source digest differs from frozen selection")
        tree = _code_only(await _read_source_tree(reader, source))
        combined.update({f"{role}/{path}": data for path, data in tree.items()})
        paths[role] = f"source_evidence/downstream/{role}"
    if len(combined) > MAX_SOURCE_ENTRIES or sum(map(len, combined.values())) > MAX_SOURCE_BYTES:
        raise ValueError("combined paper source limit exceeded")
    destination = join(output_dir, "source_evidence", "downstream")
    if await reader.is_dir(destination):
        if await _read_source_tree(reader, destination) != combined:
            raise ValueError("existing downstream paper sources differ from frozen selection")
    else:
        for relative, data in sorted(combined.items()):
            await workspace.write_bytes(join(destination, relative), data)
    context["source_evidence"] = paths


async def verify_study_sources(
    workspace: Workspace, paper_dir: str, selection: Mapping[str, Any] | None
) -> None:
    """Reject author mutations before auditing; never repair or recreate missing evidence.

    Raises:
        ValueError: When the staged sources or the study context in ``paper_data.json`` differ
            from the frozen selection.
    """
    reader = ResultsReader(workspace)
    paper_data = await reader.read_json(join(paper_dir, "paper_data.json"))
    if paper_data is None:
        raise ValueError("could not verify frozen paper sources: paper data must contain an object")
    context = paper_data.get("downstream_study")
    if selection is None:
        if context is not None:
            raise ValueError("paper declares a downstream study without a frozen selection")
        return
    if not isinstance(context, dict):
        raise ValueError("paper is missing downstream study context")
    for key in STUDY_SELECTION_KEYS:
        if key not in selection:
            continue
        expected = selection[key]
        if key == "baseline_score" and selection.get("baseline_program_dir") is None:
            expected = None
        if key not in context or context[key] != expected:
            raise ValueError(f"paper study context differs from frozen selection: {key}")
    if "combined_score" in selection and "baseline_score" in selection:
        baseline = selection["baseline_score"] if selection.get("baseline_program_dir") else None
        delta = selection["combined_score"] - baseline if baseline is not None else None
        if context.get("search_score_delta") != delta:
            raise ValueError("paper study score difference differs from frozen selection")
    expected_tree: dict[str, bytes] = {}
    paths: dict[str, str] = {}
    for role, source_key, digest_key in (
        ("program", "program_dir", "program_tree_sha256"),
        ("baseline", "baseline_program_dir", "baseline_tree_sha256"),
    ):
        if selection.get(source_key) is None:
            continue
        source = str(selection[source_key])
        if await workspace.digest(source) != selection[digest_key]:
            raise ValueError(f"paper {role} source digest differs from frozen selection")
        expected_tree.update(
            {
                f"{role}/{path}": data
                for path, data in _code_only(await _read_source_tree(reader, source)).items()
            }
        )
        paths[role] = f"source_evidence/downstream/{role}"
    if context.get("source_evidence") != paths:
        raise ValueError("paper source paths differ from frozen selection")
    actual = await _read_source_tree(reader, join(paper_dir, "source_evidence", "downstream"))
    if actual != expected_tree:
        raise ValueError("existing downstream paper sources differ from frozen selection")


# --- building paper data ---------------------------------------------------------------------


def _program_number(path: str) -> tuple[int, str] | None:
    name = PurePosixPath(path).name
    match = PROGRAM_DIR_RE.match(name)
    return (int(match.group(1)), name) if match else None


async def _investigation_rows(
    reader: ResultsReader,
    investigation: InvestigationResult,
    evidence_reader: EvidenceReader | None,
) -> tuple[list[dict[str, Any]], int, int, list[str]]:
    """Return the evaluated-program rows, the idea and attempt counts and the idea titles."""
    ideas = {record.idea_id: record for record in investigation.ideas}
    attempts = {record.attempt_id: record for record in investigation.attempts}
    ordered: list[tuple[tuple[str, str, int], EvaluationRecord, str, str, str, str, int]] = []
    for evaluation in investigation.evaluations:
        if evaluation.plane != "search" or not evaluation.idea_id:
            continue
        if evaluation.status != "completed":
            continue
        score = _metric_float(evaluation.metrics.get("combined_score"))
        if score is None:
            continue
        number = _program_number(evaluation.program.path)
        if number is None:
            continue
        if not await evaluation_results_complete(reader.workspace, evaluation.program.path):
            continue
        idea = ideas.get(evaluation.idea_id)
        idea_dir = (
            idea.idea_dir
            if idea is not None
            else str(PurePosixPath(evaluation.program.path).parent.parent)
        )
        idea_dir_name = PurePosixPath(idea_dir).name
        attempt = attempts.get(evaluation.attempt_id or "")
        attempt_dir = attempt.attempt_dir if attempt is not None else idea_dir
        attempt_dir_name = (
            "idea_root" if attempt_dir == idea_dir else PurePosixPath(attempt_dir).name
        )
        idea_id = (
            idea.bundle.idea_id.strip()
            if idea is not None and idea.bundle.idea_id.strip()
            else evaluation.idea_id
        )
        attempt_id = evaluation.attempt_id or f"{idea_id}_attempt_01"
        ordered.append(
            (
                (idea_dir_name, attempt_dir_name, number[0]),
                evaluation,
                idea_id,
                idea_dir_name,
                attempt_id,
                attempt_dir_name,
                number[0],
            )
        )
    ordered.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    for eval_index, (
        _,
        evaluation,
        idea_id,
        idea_dir_name,
        attempt_id,
        attempt_dir_name,
        idx,
    ) in enumerate(ordered, start=1):
        program_dir = evaluation.program.path
        correct = evaluation.correct
        rows.append(
            {
                "idea_id": idea_id,
                "idea_dir_name": idea_dir_name,
                "attempt_id": attempt_id,
                "attempt_dir_name": attempt_dir_name,
                "program_id": PurePosixPath(program_dir).name,
                "program_idx": idx,
                "combined_score": float(evaluation.metrics["combined_score"]),
                "correct": correct,
                "eligible": (correct is True),
                "program_dir": reader.host(program_dir),
                "evaluation_evidence": await paper_evaluation_evidence(
                    reader,
                    investigation.results_dir or "",
                    program_dir,
                    evidence_reader=evidence_reader,
                ),
                "eval_index": eval_index,
            }
        )
    titles: list[str] = []
    for record in sorted(investigation.ideas, key=lambda item: PurePosixPath(item.idea_dir).name):
        title = record.bundle.title.strip()
        if title:
            titles.append(title)
    return rows, len(investigation.ideas), len(investigation.attempts), titles


def _register(sources: dict[str, dict[str, Any]], source_id: str, payload: dict[str, Any]) -> None:
    sources[source_id] = {"id": source_id, **payload}


def _investigation_payload(
    rows: list[dict[str, Any]],
    *,
    idea_count: int,
    attempt_count: int,
    idea_titles: list[str],
    paper_profile: str,
    results_dir: str,
) -> dict[str, Any]:
    """Return the paper data of an investigation's evaluated programs (upstream body)."""
    all_programs = sorted(rows, key=lambda row: int(row["eval_index"]))
    correct_programs = [row for row in all_programs if row.get("correct") is True]
    top_programs = sorted(
        correct_programs,
        key=lambda row: (-float(row["combined_score"]), int(row["eval_index"])),
    )[:10]

    improvement_trace: list[dict[str, Any]] = []
    running_best = float("-inf")
    for row in correct_programs:
        score = float(row["combined_score"])
        if score <= running_best:
            continue
        running_best = score
        improvement_trace.append(
            {
                "id": f"result.improvement_{len(improvement_trace) + 1:03d}",
                "step_index": int(row["eval_index"]),
                "x_index": int(row["eval_index"]) - 1,
                "idea_id": row["idea_id"],
                "attempt_id": row["attempt_id"],
                "program_id": row["program_id"],
                "program_idx": int(row["program_idx"]),
                "combined_score": score,
                "program_dir": row["program_dir"],
            }
        )

    best_correct = _best_by_score(correct_programs)
    best_any = _best_by_score(all_programs)

    grounding_sources: dict[str, dict[str, Any]] = {}
    for source_id, entry in (
        ("result.best_correct_score", best_correct),
        ("result.best_any_score", best_any),
    ):
        if isinstance(entry, dict):
            _register(
                grounding_sources,
                source_id,
                {
                    "type": "metric",
                    "value": float(entry["combined_score"]),
                    "program_dir": entry.get("program_dir"),
                    "idea_id": entry.get("idea_id"),
                    "attempt_id": entry.get("attempt_id"),
                    "program_id": entry.get("program_id"),
                },
            )
    _register(grounding_sources, "result.total_ideas", {"type": "count", "value": idea_count})
    _register(grounding_sources, "result.total_attempts", {"type": "count", "value": attempt_count})
    _register(
        grounding_sources,
        "result.total_evaluated_programs",
        {"type": "count", "value": len(all_programs)},
    )
    _register(
        grounding_sources,
        "result.total_correct_programs",
        {"type": "count", "value": len(correct_programs)},
    )
    for idx, row in enumerate(top_programs, start=1):
        _register(
            grounding_sources,
            f"result.top_program_{idx:02d}",
            {
                "type": "metric",
                "value": float(row["combined_score"]),
                "program_dir": row.get("program_dir"),
                "idea_id": row.get("idea_id"),
                "attempt_id": row.get("attempt_id"),
                "program_id": row.get("program_id"),
            },
        )
    for row in improvement_trace:
        _register(
            grounding_sources,
            str(row["id"]),
            {
                "type": "metric",
                "value": row["combined_score"],
                "program_dir": row.get("program_dir"),
                "idea_id": row.get("idea_id"),
                "attempt_id": row.get("attempt_id"),
                "program_id": row.get("program_id"),
            },
        )

    return {
        "generated_at": utc_now_string(),
        "paper_profile": paper_profile,
        "presentation_contract": _presentation_contract(paper_profile),
        "results_dir": results_dir,
        "best_correct": best_correct,
        "best_any": best_any,
        "totals": {
            "ideas": idea_count,
            "attempts": attempt_count,
            "evaluated_programs": len(all_programs),
            "correct_programs": len(correct_programs),
        },
        "top_programs": top_programs,
        "improvement_trace": improvement_trace,
        "all_programs": all_programs,
        "idea_titles": idea_titles,
        "grounding_sources": grounding_sources,
        "ablation_studies": [],
    }


async def build_paper_data(
    workspace: Workspace,
    *,
    investigation: InvestigationResult | None = None,
    downstream: DownstreamSelection | None = None,
    variants: VariantsResult | None = None,
    final: FinalEvaluationResult | None = None,
    paper_profile: Literal["default", "standalone"] = "default",
    final_outcome_policy: FinalOutcomePolicy = "qualified_only",
    evidence_reader: EvidenceReader | None = None,
    allow_empty: bool = False,
) -> PaperData:
    """Build the paper data of a run from the results of its phases.

    The same JSON as upstream ``build_paper_data``, with the same grounding ids (see the module
    docstring), built from typed phase results; the artifacts they name are read through
    ``workspace``. Paths in the result are the paths processes of the workspace see
    (``Workspace.host_path``), as upstream's absolute paths are.

    Args:
        workspace: The workspace holding the artifacts.
        investigation: The investigation result; its search evaluations become the evaluated
            programs. Without one, the paper data is built from the ablation bundle of
            ``variants`` (upstream's results directory that is itself a bundle).
        downstream: The downstream selection of a run without an investigation winner (adds the
            study context and its grounding).
        variants: The ablation phase's result; its output directory is the ablation bundle that
            is merged (``ablation.*`` grounding, cross-task results).
        final: The verified sealed final evaluation (``final.*`` grounding).
        paper_profile: ``default`` or ``standalone`` (adds the presentation contract).
        final_outcome_policy: Whether a ``not_qualified`` final evaluation may support a paper.
        evidence_reader: Replaces the default reader of ``evaluation_evidence.json`` (the
            ``evaluation_evidence.json`` of an evaluated program directory).
        allow_empty: Do not raise when no program was evaluated.

    Raises:
        PaperDataError: When the paper profile is unknown, no program was evaluated (and there
            is no downstream study), the final evaluation did not qualify under the policy, or
            a grounding id would collide.
        ValueError: When an evidence file breaks a rule of ``evaluation_evidence.json``.
    """
    if paper_profile not in VALID_PAPER_PROFILES:
        raise PaperDataError(
            f"paper_profile must be one of {sorted(VALID_PAPER_PROFILES)}, got {paper_profile!r}"
        )
    reader = ResultsReader(workspace)
    variant_ids = [spec.variant_id for spec in variants.specs] if variants is not None else []
    bundle_dir = variants.output_dir if variants is not None else None
    selection = await load_downstream_selection(reader, downstream)

    has_programs = investigation is not None and any(
        evaluation.plane == "search" and evaluation.idea_id and evaluation.status == "completed"
        for evaluation in investigation.evaluations
    )
    paper_data: dict[str, Any]
    if not has_programs and bundle_dir is not None:
        if await has_multitask_ablation_bundle(reader, bundle_dir):
            paper_data = await build_multitask_ablation_paper_data(
                reader,
                bundle_dir,
                paper_profile=paper_profile,
                variant_ids=variant_ids,
                evidence_reader=evidence_reader,
            )
        elif await has_ablation_bundle(reader, bundle_dir):
            paper_data = await build_ablation_paper_data(
                reader,
                bundle_dir,
                paper_profile=paper_profile,
                variant_ids=variant_ids,
                evidence_reader=evidence_reader,
            )
            _merge_cross_task_results(
                paper_data, await collect_cross_task_results(reader, bundle_dir)
            )
        else:
            paper_data = {}
        if paper_data:
            merge_study_context(paper_data, selection)
            await _finish(
                paper_data,
                reader=reader,
                final=final,
                final_outcome_policy=final_outcome_policy,
                evidence_reader=evidence_reader,
            )
            return _validated(paper_data, allow_empty=allow_empty)

    rows: list[dict[str, Any]] = []
    idea_count = attempt_count = 0
    titles: list[str] = []
    if investigation is not None:
        rows, idea_count, attempt_count, titles = await _investigation_rows(
            reader, investigation, evidence_reader
        )
    results_dir = investigation.results_dir if investigation and investigation.results_dir else ""
    paper_data = _investigation_payload(
        rows,
        idea_count=idea_count,
        attempt_count=attempt_count,
        idea_titles=titles,
        paper_profile=paper_profile,
        results_dir=reader.host(results_dir) if results_dir else "",
    )

    evidence_sources = paper_data.setdefault("evidence_sources", {})
    if variants is not None and bundle_dir is not None:
        host_bundle = reader.host(bundle_dir)
        if not await reader.is_dir(bundle_dir):
            evidence_sources["ablations"] = {"status": "missing", "path": host_bundle}
        elif await has_ablation_bundle(reader, bundle_dir) or await has_multitask_ablation_bundle(
            reader, bundle_dir
        ):
            paper_data = await _merge_nested_ablation_payload(
                paper_data,
                reader=reader,
                nested_bundle_dir=bundle_dir,
                paper_profile=paper_profile,
                variant_ids=variant_ids,
                evidence_reader=evidence_reader,
            )
            evidence_sources = paper_data.setdefault("evidence_sources", {})
            evidence_sources["ablations"] = {"status": "included", "path": host_bundle}
        else:
            evidence_sources["ablations"] = {
                "status": "degraded",
                "path": host_bundle,
                "reason": "no validated ablation bundle",
            }
    paper_data.setdefault("evidence_sources", {})["investigation"] = {
        "status": "included",
        "path": reader.host(results_dir) if results_dir else "",
    }

    if bundle_dir is not None:
        _merge_cross_task_results(paper_data, await collect_cross_task_results(reader, bundle_dir))
    merge_study_context(paper_data, selection)
    await _finish(
        paper_data,
        reader=reader,
        final=final,
        final_outcome_policy=final_outcome_policy,
        evidence_reader=evidence_reader,
    )
    return _validated(paper_data, allow_empty=allow_empty)


async def _finish(
    paper_data: dict[str, Any],
    *,
    reader: ResultsReader,
    final: FinalEvaluationResult | None,
    final_outcome_policy: FinalOutcomePolicy,
    evidence_reader: EvidenceReader | None,
) -> None:
    if final is not None:
        await merge_final_evaluation(
            paper_data,
            final,
            reader=reader,
            final_outcome_policy=final_outcome_policy,
            evidence_reader=evidence_reader,
        )


def _validated(paper_data: dict[str, Any], *, allow_empty: bool) -> PaperData:
    totals = paper_data.get("totals", {})
    if (
        not allow_empty
        and int(totals.get("evaluated_programs", 0)) <= 0
        and not paper_data.get("downstream_study")
    ):
        raise PaperDataError(
            "No evaluated programs found in results directory; cannot write paper."
        )
    return PaperData.model_validate(paper_data)


def _merge_cross_task_results(paper_data: dict[str, Any], cross: Mapping[str, Any]) -> None:
    """Merge aggregated/<task>/run_summary.csv stats into paper_data in-place."""
    task_names = cross.get("task_names") or []
    cross_results = cross.get("cross_task_results") or {}
    cross_sources = cross.get("grounding_sources") or {}
    if task_names:
        existing = paper_data.get("task_names")
        merged = list(existing) if isinstance(existing, list) else []
        for name in task_names:
            if name not in merged:
                merged.append(name)
        paper_data["task_names"] = merged
    if cross_results:
        paper_data["cross_task_results"] = cross_results
    if cross_sources:
        grounding = paper_data.get("grounding_sources")
        if not isinstance(grounding, dict):
            grounding = {}
            paper_data["grounding_sources"] = grounding
        for key, payload in cross_sources.items():
            grounding.setdefault(key, payload)


async def write_paper_data(
    workspace: Workspace, paper_dir: str, paper_data: PaperData, *, name: str = "paper_data.json"
) -> str:
    """Write ``<paper_dir>/paper_data.json`` (sorted keys, as upstream) and return its path."""
    path = join(paper_dir, name)
    text = json.dumps(paper_data.model_dump(mode="json"), indent=2, sort_keys=True)
    await workspace.write_bytes(path, text.encode("utf-8"))
    return path


async def paper_data_node(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """The default node of the ``paper.data`` slot: build the paper data from the phase results.

    Reads ``investigation``, ``downstream``, ``variants`` and ``final`` from the state, the paper
    profile and the final-outcome policy from the recorded configuration (``run_config``), and
    returns ``{"paper_data": PaperData}``; when ``paper_dir`` is set, ``paper_data.json`` is
    written there. A prebuilt ``paper_data`` in the state is kept (the final-outcome gate is
    applied to it either way). A paper data that cannot be built returns ``{"failure": Failure}``
    and no ``paper_data``.
    """
    from ..core.runconfig import method_config
    from ..core.state import Failure

    config = method_config(state)
    policy: FinalOutcomePolicy = config.final_outcome_policy
    existing = state.get("paper_data")
    try:
        if existing is not None:
            paper_data = (
                existing if isinstance(existing, PaperData) else PaperData.model_validate(existing)
            )
            check_final_outcome_policy(paper_data.model_dump(mode="json"), policy)
            return {}
        paper_data = await build_paper_data(
            runtime.context.workspace,
            investigation=state.get("investigation"),
            downstream=state.get("downstream"),
            variants=state.get("variants"),
            final=state.get("final"),
            paper_profile=config.paper.profile,
            final_outcome_policy=policy,
        )
        paper_dir = state.get("paper_dir")
        if paper_dir:
            await write_paper_data(runtime.context.workspace, paper_dir, paper_data)
    except PaperDataError as exc:
        return {"failure": Failure(category="paper_data_failed", scope="method", detail=str(exc))}
    return {"paper_data": paper_data}


__all__ = [
    "CODE_EXCLUDES",
    "EvidenceReader",
    "FinalOutcomePolicy",
    "PaperDataError",
    "ResultsReader",
    "ablation_evaluation_evidence",
    "augment_paper_data_with_posthoc",
    "build_ablation_paper_data",
    "build_multitask_ablation_paper_data",
    "build_paper_data",
    "check_final_outcome_policy",
    "collect_cross_task_results",
    "collect_posthoc_analyses",
    "copy_posthoc_figures",
    "default_evidence_reader",
    "discover_posthoc_session",
    "ground_final_outcome",
    "load_downstream_selection",
    "merge_final_evaluation",
    "merge_study_context",
    "merge_study_final_outcome",
    "paper_data_node",
    "paper_evaluation_evidence",
    "stage_study_sources",
    "validate_posthoc_ranking",
    "verify_study_sources",
    "write_paper_data",
]
