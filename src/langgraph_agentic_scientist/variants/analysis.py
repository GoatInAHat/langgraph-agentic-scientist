"""The host's canonical analysis of an evaluation (upstream ``sandbox_outputs.py`` and
``qualification_summary.py``).

After an evaluation the host does not trust CSVs that the candidate wrote. It replaces them with
three canonical files under ``program_<n>/analysis/``, built from what the evaluator reported
(``metrics.json``, ``correct.json``, the validated ``evaluation_evidence.json``) and the identity
the host knows (evaluation id, variant id, kind, base program, agent name):

* ``run_summary.csv``: one row; ``overall_pass`` is left empty until the rubric status is final;
* ``metric_values.csv``: one row per numeric metric (nested metrics flattened with dots,
  ``combined_score`` first);
* ``timeseries.csv``: the evidence's candidate time series (namespaced ``candidate/seed_<seed>/``)
  followed by the rows of every ``analysis/timeseries.csv`` the program wrote below the
  evaluator's output directory, validated row by row.

Bounds are upstream's: 20,000 entries, 64 MiB and depth 8 of evaluator output; 200,000 time
series rows; 5,000 metrics; fields of at most 512 characters. A violation raises ``RuntimeError``
and the evaluation does not count. After the verifier's rubric status is canonicalised,
:func:`publish_summary_qualification` fills ``overall_pass`` in ``run_summary.csv``.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any

from ..evaluation.protocol import EvaluationResult
from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import FileEntry, Workspace, is_dir
from .constants import (
    ANALYSIS_CSV_SCHEMAS,
    MAX_ANALYSIS_ROWS,
    MAX_EVIDENCE_LENGTH,
    MAX_FIELD_LENGTH,
    MAX_JSON_DEPTH,
    MAX_METRICS,
    MAX_SANDBOX_DEPTH,
    MAX_SANDBOX_ENTRIES,
    MAX_SANDBOX_OUTPUT_BYTES,
    SAFE_LABEL,
)
from .rubrics import (
    canonicalize_variant_rubric_status,
    required_rubric_ids,
    variant_attempt_id,
)
from .workspace_ops import read_json_object, read_text_or_none

_PROGRAM_DIR = re.compile(r"program_(\d+)")


def _safe_label(value: str, kind: str) -> str:
    if not SAFE_LABEL.fullmatch(value):
        raise RuntimeError(f"sandbox produced invalid {kind}")
    return value


def trusted_label(environment: Mapping[str, str], key: str) -> str:
    """Return an identity label of the host environment, validated (``unknown`` when unset)."""
    return _safe_label(environment.get(key) or "unknown", key.lower())


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("sandbox produced a non-numeric analysis value") from exc
    if not math.isfinite(number):
        raise RuntimeError("sandbox produced a non-finite analysis value")
    return number


def _optional_timeseries_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        boolean_value = value.strip().casefold()
        if boolean_value == "true":
            return 1.0
        if boolean_value == "false":
            return 0.0
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("sandbox produced a non-numeric analysis value") from exc
    if math.isnan(number):
        return None
    if not math.isfinite(number):
        raise RuntimeError("sandbox produced a non-finite analysis value")
    return number


def numeric_metrics(
    payload: Mapping[str, Any], prefix: str = "", depth: int = 0
) -> list[tuple[str, float]]:
    """Flatten the numeric entries of ``metrics.json`` (dots for nesting), ``combined_score`` first.

    Booleans, strings, nulls and lists are skipped; anything else that is not a number or an
    object is an error, as are non-finite numbers, unsafe names and more than 5,000 metrics.
    """
    if depth > MAX_JSON_DEPTH:
        raise RuntimeError("sandbox metrics.json exceeds the nesting limit")
    metrics: list[tuple[str, float]] = []
    for key, value in payload.items():
        if not isinstance(key, str):
            raise RuntimeError("sandbox metrics.json contains a non-string key")
        name = _safe_label(f"{prefix}.{key}" if prefix else key, "metric name")
        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        if isinstance(value, int | float):
            metrics.append((name, _finite_float(value)))
        elif isinstance(value, dict):
            metrics.extend(numeric_metrics(value, name, depth + 1))
        elif isinstance(value, list):
            continue
        else:
            raise RuntimeError("sandbox metrics.json contains an invalid value")
        if len(metrics) > MAX_METRICS:
            raise RuntimeError("sandbox metrics.json exceeds the metric limit")
    metrics.sort(key=lambda item: (item[0] != "combined_score", item[0]))
    return metrics


def csv_text(columns: list[str], rows: Iterable[Mapping[str, Any]]) -> str:
    """Write rows as CSV with upstream's dialect (header first, ``\\r\\n`` line ends)."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def read_csv_rows(text: str, schema_name: str, max_rows: int) -> list[dict[str, str]]:
    """Parse an analysis CSV of one of the three schemas, with upstream's checks.

    The header must equal the schema's columns; at most ``max_rows`` rows; no row with extra
    fields; no field longer than 512 characters.
    """
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames != ANALYSIS_CSV_SCHEMAS[schema_name]:
            raise RuntimeError(f"sandbox produced invalid analysis CSV: {schema_name}")
        rows: list[dict[str, str]] = []
        for row in reader:
            if len(rows) >= max_rows or None in row:
                raise RuntimeError(f"sandbox produced invalid analysis CSV: {schema_name}")
            if any(len(value or "") > MAX_FIELD_LENGTH for value in row.values()):
                raise RuntimeError(f"sandbox analysis field exceeds limit: {schema_name}")
            rows.append({key: value or "" for key, value in row.items()})
        return rows
    except csv.Error as exc:
        raise RuntimeError(f"cannot read sandbox analysis CSV: {schema_name}") from exc


def validate_bounded_tree(entries: list[FileEntry]) -> None:
    """Check evaluator output against upstream's bounds: entries, bytes, depth and links."""
    if len(entries) > MAX_SANDBOX_ENTRIES:
        raise RuntimeError("sandbox output exceeds the entry limit")
    total = 0
    for entry in entries:
        if entry.kind == "symlink" or entry.kind == "other":
            raise RuntimeError(f"sandbox produced unsafe output: {entry.path}")
        if entry.kind == "dir" and entry.path.count("/") + 1 > MAX_SANDBOX_DEPTH:
            raise RuntimeError(f"sandbox output exceeds the depth limit: {entry.path}")
        if entry.kind == "file":
            total += entry.size
            if total > MAX_SANDBOX_OUTPUT_BYTES:
                raise RuntimeError("sandbox output exceeds the 64 MiB promotion limit")


def evidence_rows(
    result: EvaluationResult, evaluation_id: str, variant_id: str
) -> list[dict[str, Any]]:
    """Return the evidence's candidate time series as rows (upstream ``evidence_timeseries``)."""
    if result.evidence is None:
        return []
    rows: list[dict[str, Any]] = []
    for episode in result.evidence.episodes:
        for row in episode.candidate_timeseries:
            rows.append(
                {
                    **row,
                    "series_name": f"candidate/seed_{episode.seed}/{row['series_name']}",
                    "evaluation_id": evaluation_id,
                    "variant_id": variant_id,
                }
            )
    return rows


async def validated_timeseries(
    workspace: Workspace,
    output_dir: str | None,
    entries: list[FileEntry],
    evaluation_id: str,
    variant_id: str,
    from_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the canonical time series rows: the evidence's, then the program's own."""
    rows: list[dict[str, Any]] = list(from_evidence)
    row_count = len(rows)
    sources = sorted(
        entry.path
        for entry in entries
        if entry.kind == "file"
        and (
            entry.path == "analysis/timeseries.csv"
            or entry.path.endswith("/analysis/timeseries.csv")
        )
    )
    for relative in sources:
        assert output_dir is not None
        parts = PurePosixPath(relative).parts
        scratch_name = "/".join(parts[:-2])
        raw = await workspace.read_bytes(join(output_dir, relative))
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise RuntimeError("cannot read sandbox analysis CSV: timeseries.csv") from exc
        for row in read_csv_rows(text, "timeseries.csv", MAX_ANALYSIS_ROWS):
            row_count += 1
            if row_count > MAX_ANALYSIS_ROWS:
                raise RuntimeError("sandbox timeseries exceeds the row limit")
            series_name = _safe_label(row["series_name"], "series name")
            if from_evidence and not scratch_name and series_name.startswith("candidate/seed_"):
                # The fixed evidence artifact is the source for this reserved namespace.
                continue
            if scratch_name:
                series_name = _safe_label(
                    f"{_safe_label(scratch_name, 'scratch name')}/{series_name}", "series name"
                )
            step = _finite_float(row["step"])
            if step < 0 or not step.is_integer():
                raise RuntimeError("sandbox timeseries contains an invalid step")
            value = _optional_timeseries_float(row["value"])
            if value is None:
                continue
            rows.append(
                {
                    "evaluation_id": evaluation_id,
                    "variant_id": variant_id,
                    "series_name": series_name,
                    "step": int(step),
                    "value": value,
                }
            )
    return rows


async def write_canonical_analysis(
    workspace: Workspace,
    *,
    program_dir: str,
    result: EvaluationResult,
    environment: Mapping[str, str],
) -> None:
    """Replace ``<program_dir>/analysis`` with the canonical analysis of an evaluation.

    Args:
        workspace: The run's workspace.
        program_dir: The evaluated ``program_<n>`` directory.
        result: The completed evaluation. Its ``metrics`` must hold a finite ``combined_score``
            and its ``correct`` must be a boolean.
        environment: The trusted identity: ``EVALUATION_ID``, ``VARIANT_ID``, ``VARIANT_KIND``,
            ``BASE_PROGRAM`` and ``AGENT_NAME``.

    Raises:
        RuntimeError: When the evaluator's output violates upstream's bounds or contracts.
    """
    metrics = numeric_metrics(result.metrics)
    combined_score = dict(metrics).get("combined_score")
    if combined_score is None:
        raise RuntimeError("sandbox metrics.json must contain finite combined_score")
    if not isinstance(result.correct, bool):
        raise RuntimeError("sandbox correct.json must contain a boolean 'correct'")
    evaluation_id = trusted_label(environment, "EVALUATION_ID")
    variant_id = trusted_label(environment, "VARIANT_ID")
    program_name = _safe_label(PurePosixPath(program_dir).name, "program name")

    entries: list[FileEntry] = []
    output_dir = result.output_dir
    if output_dir is not None and await is_dir(workspace, output_dir):
        entries = await workspace.list_tree(output_dir)
        validate_bounded_tree(entries)
    else:
        output_dir = None
    series = await validated_timeseries(
        workspace,
        output_dir,
        entries,
        evaluation_id,
        variant_id,
        evidence_rows(result, evaluation_id, variant_id),
    )

    summary_row = {
        "evaluation_id": evaluation_id,
        "variant_id": variant_id,
        "variant_kind": trusted_label(environment, "VARIANT_KIND"),
        "base_program": trusted_label(environment, "BASE_PROGRAM"),
        "agent_name": trusted_label(environment, "AGENT_NAME"),
        "correct": result.correct,
        "combined_score": combined_score,
        "overall_pass": "",
        "primary_metric_name": "combined_score",
        "primary_metric_value": combined_score,
        "program_dir": program_name,
        "results_dir": "analysis",
    }
    metric_rows = [
        {
            "evaluation_id": evaluation_id,
            "variant_id": variant_id,
            "metric_name": name,
            "metric_value": value,
            "is_primary_metric": name == "combined_score",
        }
        for name, value in metrics
    ]
    schemas = ANALYSIS_CSV_SCHEMAS
    analysis = join(program_dir, "analysis")
    await workspace.remove(analysis)
    await workspace.write_bytes(
        join(analysis, "run_summary.csv"),
        csv_text(schemas["run_summary.csv"], [summary_row]).encode("utf-8"),
    )
    await workspace.write_bytes(
        join(analysis, "metric_values.csv"),
        csv_text(schemas["metric_values.csv"], metric_rows).encode("utf-8"),
    )
    await workspace.write_bytes(
        join(analysis, "timeseries.csv"),
        csv_text(schemas["timeseries.csv"], series).encode("utf-8"),
    )


# --- rubric qualification of run_summary.csv ----------------------------------------------------


async def summary_qualification(
    workspace: Workspace,
    program_dir: str,
    *,
    prompts: PromptRegistry,
    max_evidence: int = MAX_EVIDENCE_LENGTH,
) -> dict[str, str]:
    """Reconcile a structured evaluation with its canonical rubric status, or mark it unknown.

    Returns ``{"overall_pass": "True" | "False"}`` when ``program_<n>/analysis/run_summary.csv``,
    ``rubric_status.json``, ``correct.json`` and ``metrics.json`` agree with the host-owned
    canonical status, and ``{"overall_pass": ""}`` otherwise (port of upstream
    ``summary_qualification``).
    """
    unknown = {"overall_pass": ""}
    match = _PROGRAM_DIR.fullmatch(PurePosixPath(program_dir).name)
    if match is None:
        return unknown
    try:
        summary_text = await read_text_or_none(
            workspace, join(program_dir, "analysis/run_summary.csv")
        )
        status = await read_json_object(workspace, join(program_dir, "rubric_status.json"))
        correctness = await read_json_object(workspace, join(program_dir, "correct.json"))
        metrics = await read_json_object(workspace, join(program_dir, "metrics.json"))
        if summary_text is None or status is None or correctness is None or metrics is None:
            return unknown
        rows = list(csv.DictReader(io.StringIO(summary_text)))
        if len(rows) != 1:
            return unknown
        row = rows[0]
        variant_id = row["variant_id"]
        correct = correctness.get("correct")
        if not isinstance(correct, bool):
            return unknown
        attempt_id = variant_attempt_id(variant_id, int(match[1]))
        if (
            status.get("variant_id") != variant_id
            or status.get("idea_id") != variant_id
            or status.get("attempt_id") != attempt_id
        ):
            return unknown
        required = required_rubric_ids(
            prompts, variant_id, row.get("variant_kind", ""), row.get("base_program", "best")
        )
        canonical, _ = canonicalize_variant_rubric_status(
            status,
            variant_id=variant_id,
            idea_id=variant_id,
            attempt_id=attempt_id,
            required_rubric_ids=required,
            combined_score=metrics.get("combined_score"),
            correct=correct,
            max_evidence=max_evidence,
        )
        if (
            canonical is None
            or status.get("overall_pass") is not canonical["overall_pass"]
            or status.get("metrics") != canonical["metrics"]
        ):
            return unknown
        return {"overall_pass": str(canonical["overall_pass"])}
    except (OSError, ValueError, UnicodeError, KeyError, csv.Error, json.JSONDecodeError):
        return unknown


async def publish_summary_qualification(
    workspace: Workspace,
    program_dir: str,
    *,
    prompts: PromptRegistry,
    max_evidence: int = MAX_EVIDENCE_LENGTH,
) -> None:
    """Update a saved ``run_summary.csv`` after the host has finalised rubric qualification."""
    relative = join(program_dir, "analysis/run_summary.csv")
    text = await read_text_or_none(workspace, relative)
    if text is None:
        return
    reader = csv.DictReader(io.StringIO(text, newline=""))
    rows = list(reader)
    if not reader.fieldnames or len(rows) != 1:
        return
    row = rows[0]
    row.update(
        await summary_qualification(
            workspace, program_dir, prompts=prompts, max_evidence=max_evidence
        )
    )
    columns = list(dict.fromkeys([*reader.fieldnames, "overall_pass"]))
    await workspace.write_bytes(relative, csv_text(columns, [row]).encode("utf-8"))


__all__ = [
    "csv_text",
    "evidence_rows",
    "numeric_metrics",
    "publish_summary_qualification",
    "read_csv_rows",
    "summary_qualification",
    "trusted_label",
    "validate_bounded_tree",
    "validated_timeseries",
    "write_canonical_analysis",
]
