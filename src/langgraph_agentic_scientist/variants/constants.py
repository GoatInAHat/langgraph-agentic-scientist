"""Constants of the variants phase: CSV schemas, rubric ids, file names and limits.

Ports of upstream ``variants_pipeline/constants.py`` and the module-level constants of the other
variants modules. The CSV schemas and the required rubric ids are the contract between the
retrofitted program, the host's canonical analysis, the rubric verifier and the aggregation.
"""

from __future__ import annotations

import re

RUN_SUMMARY_COLUMNS: list[str] = [
    "evaluation_id",
    "variant_id",
    "variant_kind",
    "base_program",
    "agent_name",
    "correct",
    "combined_score",
    "overall_pass",
    "primary_metric_name",
    "primary_metric_value",
    "program_dir",
    "results_dir",
]
"""Columns of ``analysis/run_summary.csv``."""

METRIC_VALUES_COLUMNS: list[str] = [
    "evaluation_id",
    "variant_id",
    "metric_name",
    "metric_value",
    "is_primary_metric",
]
"""Columns of ``analysis/metric_values.csv``."""

TIMESERIES_COLUMNS: list[str] = [
    "evaluation_id",
    "variant_id",
    "series_name",
    "step",
    "value",
]
"""Columns of ``analysis/timeseries.csv``."""

ANALYSIS_CSV_SCHEMAS: dict[str, list[str]] = {
    "run_summary.csv": RUN_SUMMARY_COLUMNS,
    "metric_values.csv": METRIC_VALUES_COLUMNS,
    "timeseries.csv": TIMESERIES_COLUMNS,
}
"""Analysis filename to its required columns (the on-disk contract)."""

REQUIRED_RUBRIC_IDS: list[str] = [
    "analysis_files_present",
    "run_summary_schema_valid",
    "metric_values_schema_valid",
    "timeseries_schema_valid",
    "metrics_consistent_with_metrics_json",
    "correctness_recorded",
    "variant_intent_preserved",
    "single_change_scope",
    "paper_analysis_ready",
]
"""The nine rubric families every variant must satisfy."""

REFERENCE_RUBRIC_IDS: list[str] = [
    "hyperparameter_audit_complete",
    "reference_conditions_aligned",
    "reference_performance_check",
]
"""The three extra rubrics of a baseline adapted from an external reference implementation."""

VALID_VARIANT_KINDS: frozenset[str] = frozenset({"control", "baseline", "ablation"})
"""Variant kinds of the study (the core's ``VariantKind`` also names two this phase never uses)."""

CONTROL_BEST_ID = "control_best"
BASELINE_REFERENCE_ID = "baseline_reference"
RESERVED_VARIANT_IDS: frozenset[str] = frozenset({BASELINE_REFERENCE_ID, CONTROL_BEST_ID})

CONTRACT_VERSION = 1
MAX_RUBRIC_STATUS_BYTES = 1024 * 1024
MAX_RUBRICS = 256
MAX_EVIDENCE_LENGTH = 4096
MAX_RECOMMENDATION_LENGTH = 4096
MAX_VERIFICATION_ERROR_LENGTH = 256
SAFE_RUBRIC_ID = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
VARIANT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
"""Syntax of a variant id (upstream checks it for custom baselines; generated ones are screened
by the same rule)."""
MAX_VARIANT_ID_LENGTH = 64

RETROFIT_COMPLETE_FILENAME = ".retrofit_complete.json"
MAX_EVALUATOR_BYTES = 16 * 1024 * 1024
EXTRACTION_MANIFEST_FILENAME = "extraction_manifest.json"
EXTRACTION_VERSION = 1
EXTRACTED_BUNDLE_DIR = "_extracted_bundle"
RETROFITTED_BUNDLE_DIR = "_retrofitted_bundle"
GENERATION_DIR = "generation"
VARIANTS_SUBDIR = "variants"
EXTRA_TASKS_SUBDIR = "extra_tasks"
AGGREGATED_SUBDIR = "aggregated"
README_WORK_DIR = "_readme_work"
PIPELINE_SUMMARY_FILENAME = "pipeline_summary.json"
TASK_SNAPSHOT_METADATA_FILENAME = ".agentic_task_snapshot.json"

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
"""Upstream ``investigation.constants.TASK_COPY_EXCLUDE_DIRS``."""

# Files of a previous run that must not be copied into a variant workspace (materialize.py).
TASK_COPY_EXCLUDE_FILES: frozenset[str] = frozenset(
    {
        "correct.json",
        "evaluation_evidence.json",
        "evaluation_manifest.json",
        "metrics.json",
        "rubric_status.json",
        "_results.json",
        "extraction_manifest.json",
    }
)
TASK_ROOT_EXCLUDE_FILES: frozenset[str] = frozenset({".agentic_runtime.json"})
TASK_COPY_EXCLUDE_SUFFIXES: tuple[str, ...] = (".log", ".out", ".err")

# Evaluator artifacts a candidate must not hold before its evaluation (host_eval.py).
EVALUATOR_ARTIFACTS: tuple[str, ...] = (
    "_results.json",
    "analysis",
    "metrics.json",
    "correct.json",
    "evaluation_evidence.json",
    "rubric_status.json",
    "evaluation_manifest.json",
    "harness_evaluate_stdout.log",
    "harness_evaluate_stderr.log",
    "runs",
)
EVALUATION_PENDING_FILENAME = ".evaluation_pending"

# Limits of the host's canonical analysis (sandbox_outputs.py).
MAX_SANDBOX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_SANDBOX_ENTRIES = 20_000
MAX_ANALYSIS_ROWS = 200_000
MAX_SANDBOX_DEPTH = 8
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 8
MAX_METRICS = 5_000
MAX_FIELD_LENGTH = 512
SAFE_LABEL = re.compile(r"[A-Za-z0-9_.:/|\-]{1,160}\Z")

__all__ = [
    "AGGREGATED_SUBDIR",
    "ANALYSIS_CSV_SCHEMAS",
    "BASELINE_REFERENCE_ID",
    "CONTRACT_VERSION",
    "CONTROL_BEST_ID",
    "EVALUATION_PENDING_FILENAME",
    "EVALUATOR_ARTIFACTS",
    "EXTRACTED_BUNDLE_DIR",
    "EXTRACTION_MANIFEST_FILENAME",
    "EXTRACTION_VERSION",
    "EXTRA_TASKS_SUBDIR",
    "GENERATION_DIR",
    "MAX_ANALYSIS_ROWS",
    "MAX_EVALUATOR_BYTES",
    "MAX_EVIDENCE_LENGTH",
    "MAX_FIELD_LENGTH",
    "MAX_JSON_BYTES",
    "MAX_JSON_DEPTH",
    "MAX_METRICS",
    "MAX_RECOMMENDATION_LENGTH",
    "MAX_RUBRICS",
    "MAX_RUBRIC_STATUS_BYTES",
    "MAX_SANDBOX_DEPTH",
    "MAX_SANDBOX_ENTRIES",
    "MAX_SANDBOX_OUTPUT_BYTES",
    "MAX_VARIANT_ID_LENGTH",
    "MAX_VERIFICATION_ERROR_LENGTH",
    "METRIC_VALUES_COLUMNS",
    "PIPELINE_SUMMARY_FILENAME",
    "README_WORK_DIR",
    "REFERENCE_RUBRIC_IDS",
    "REQUIRED_RUBRIC_IDS",
    "RESERVED_VARIANT_IDS",
    "RETROFITTED_BUNDLE_DIR",
    "RETROFIT_COMPLETE_FILENAME",
    "RUN_SUMMARY_COLUMNS",
    "SAFE_LABEL",
    "SAFE_RUBRIC_ID",
    "TASK_COPY_EXCLUDE_DIRS",
    "TASK_COPY_EXCLUDE_FILES",
    "TASK_COPY_EXCLUDE_SUFFIXES",
    "TASK_ROOT_EXCLUDE_FILES",
    "TASK_SNAPSHOT_METADATA_FILENAME",
    "TIMESERIES_COLUMNS",
    "VALID_VARIANT_KINDS",
    "VARIANTS_SUBDIR",
    "VARIANT_ID_RE",
]
