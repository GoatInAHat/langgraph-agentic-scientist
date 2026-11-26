"""Validation of evaluator artifacts, with upstream's limits and messages.

Ports of upstream ``src/investigation/evaluation_evidence.py`` (``evaluation_evidence.json``
schema 1: 16 MiB, nesting depth 12, at most 10,000 episodes and 100,000 timeseries rows, unique
signed 64-bit seeds, named finite diagnostics, no duplicate keys), ``evaluator_outputs.py``
(the contract-bound shape of ``metrics.json`` and ``correct.json``), the summary checks of
``evaluate_program.py`` (a finite numeric ``combined_score`` and a boolean ``correct``) and
``evaluation_usage.py`` (``compute_usage``). They work on bytes, so that an evaluator anywhere
(a container, a remote executor) can be validated by the host.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Collection, Mapping
from typing import Any

from ..core.errors import EvaluatorOutputContractError
from .protocol import ComputeUsage, EvaluationEvidence, EvidenceMetadata

EVIDENCE_FILENAME = "evaluation_evidence.json"
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
MAX_RESULT_JSON_BYTES = 1024 * 1024
MAX_SUMMARY_BYTES = 16 * 1024 * 1024
_MAX_DEPTH = 12
_MAX_EPISODES = 10_000
_MAX_TIMESERIES_ROWS = 100_000
_NAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,79}\Z")
_REQUIRED_EPISODE = {"seed", "score", "correct", "duration_seconds"}
_OPTIONAL_EPISODE = {"solution", "candidate_diagnostics", "candidate_timeseries"}
_COMPUTE_USAGE_FIELDS = {
    "wall_clock_limit_seconds",
    "elapsed_seconds",
    "max_rpc_calls",
    "rpc_calls",
    "outcome",
    "objective_counts",
}


def _finite(value: object) -> bool:
    try:
        return (
            not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)
        )
    except OverflowError:
        return False


# --- evaluation_evidence.json ------------------------------------------------------------


def _unique_evidence_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("evaluation evidence contains duplicate keys")
        result[key] = value
    return result


def _validate_json(value: object, depth: int = 0) -> None:
    if depth > _MAX_DEPTH:
        raise ValueError("evaluation evidence exceeds the nesting depth limit")
    if isinstance(value, dict):
        for key, item in value.items():
            if len(key) > 512:
                raise ValueError("evaluation evidence key exceeds length limit")
            _validate_json(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _validate_json(item, depth + 1)
    elif isinstance(value, int | float) and not isinstance(value, bool) and not _finite(value):
        raise ValueError("evaluation evidence numbers must be finite")


def _validate_episode(episode: Any) -> int:
    if not isinstance(episode, dict) or not episode.keys() >= _REQUIRED_EPISODE:
        raise ValueError("evaluation evidence episode is missing required fields")
    if episode.keys() - (_REQUIRED_EPISODE | _OPTIONAL_EPISODE):
        raise ValueError("evaluation evidence episode has undeclared fields")
    seed = episode["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not -(2**63) <= seed < 2**63:
        raise ValueError("evaluation evidence seed must be a signed 64-bit integer")
    if not _finite(episode["score"]) or not isinstance(episode["correct"], bool):
        raise ValueError("evaluation evidence score/correctness is invalid")
    duration = episode["duration_seconds"]
    if not _finite(duration) or duration < 0:
        raise ValueError("evaluation evidence duration must be finite and nonnegative")
    diagnostics = episode.get("candidate_diagnostics", {})
    if not isinstance(diagnostics, dict) or any(
        not _NAME.fullmatch(name) or not _finite(value) for name, value in diagnostics.items()
    ):
        raise ValueError("evaluation evidence candidate diagnostics must be named finite numbers")
    rows = episode.get("candidate_timeseries", [])
    if not isinstance(rows, list):
        raise ValueError("evaluation evidence candidate timeseries must be a list")
    for row in rows:
        if not isinstance(row, dict) or row.keys() != {"series_name", "step", "value"}:
            raise ValueError("evaluation evidence timeseries fields are invalid")
        name, step = row["series_name"], row["step"]
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise ValueError("evaluation evidence series name is invalid")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("evaluation evidence series step must be a nonnegative integer")
        if not _finite(row["value"]):
            raise ValueError("evaluation evidence series value must be finite")
    return int(seed)


def decode_evaluation_evidence(raw: bytes) -> dict[str, Any]:
    """Validate the bytes of an ``evaluation_evidence.json`` and return the decoded payload.

    Raises:
        ValueError: With upstream's message when any rule fails.
    """
    if len(raw) > MAX_EVIDENCE_BYTES:
        raise ValueError("evaluation evidence exceeds the 16 MiB limit")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_evidence_object)
        _validate_json(payload)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("evaluation evidence must be bounded valid JSON") from exc
    if not isinstance(payload, dict) or payload.keys() != {"schema_version", "episodes"}:
        raise ValueError("evaluation evidence has undeclared or missing fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("evaluation evidence schema_version must be 1")
    episodes = payload["episodes"]
    if not isinstance(episodes, list) or not 1 <= len(episodes) <= _MAX_EPISODES:
        raise ValueError("evaluation evidence episodes exceed count limit or are empty")
    seeds: set[int] = set()
    row_count = 0
    for episode in episodes:
        seed = _validate_episode(episode)
        if seed in seeds:
            raise ValueError("evaluation evidence has duplicate seeds")
        seeds.add(seed)
        row_count += len(episode.get("candidate_timeseries", []))
        if row_count > _MAX_TIMESERIES_ROWS:
            raise ValueError("evaluation evidence timeseries exceed row limit")
    return payload


def parse_evaluation_evidence(raw: bytes) -> EvaluationEvidence:
    """Validate the bytes of an ``evaluation_evidence.json`` and return the typed record."""
    return EvaluationEvidence.model_validate(decode_evaluation_evidence(raw))


def evidence_metadata(raw: bytes | None, path: str | None = None) -> EvidenceMetadata:
    """Return upstream's evidence metadata for evidence bytes (``None``: no evidence file)."""
    if raw is None:
        return EvidenceMetadata(status="unavailable")
    payload = decode_evaluation_evidence(raw)
    return EvidenceMetadata(
        status="available",
        path=path,
        sha256="sha256:" + hashlib.sha256(raw).hexdigest(),
        episode_count=len(payload["episodes"]),
        candidate_diagnostics_trust="unverified",
    )


# --- metrics.json and correct.json -------------------------------------------------------


def _unique_result_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluatorOutputContractError("evaluator JSON contains duplicate keys")
        result[key] = value
    return result


def _load_result_mapping(raw: bytes | None) -> dict[str, Any]:
    if raw is None:
        raise EvaluatorOutputContractError("required evaluator result is missing or unsafe")
    if len(raw) > MAX_RESULT_JSON_BYTES:
        raise EvaluatorOutputContractError("evaluator result exceeds 1 MiB")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_result_object)
    except EvaluatorOutputContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise EvaluatorOutputContractError("evaluator result must be bounded valid JSON") from error
    if not isinstance(payload, dict):
        raise EvaluatorOutputContractError("evaluator result must be a JSON object")
    return payload


def validate_evaluator_outputs(
    metrics_raw: bytes | None,
    correct_raw: bytes | None,
    evidence_raw: bytes | None,
    metric_names: Collection[str],
) -> tuple[dict[str, Any], bool]:
    """Validate the shape of a contracted evaluator's outputs (upstream ``evaluator_outputs``).

    This validates scientific output shape, not scientific acceptance or feasibility.

    Raises:
        EvaluatorOutputContractError: The terminal ``evaluator_output_contract_invalid`` failure.
    """
    metrics = _load_result_mapping(metrics_raw)
    if metrics.keys() != set(metric_names):
        raise EvaluatorOutputContractError(
            "search result metric names must exactly match the task contract"
        )
    for value in metrics.values():
        if not _finite(value):
            raise EvaluatorOutputContractError("search result metrics must be finite numbers")
    correct = _load_result_mapping(correct_raw)
    if correct.keys() != {"correct"} or not isinstance(correct["correct"], bool):
        raise EvaluatorOutputContractError(
            "correct.json must contain exactly one boolean key: correct"
        )
    if evidence_raw is not None:
        try:
            decode_evaluation_evidence(evidence_raw)
        except ValueError as error:
            raise EvaluatorOutputContractError(str(error)) from error
    return metrics, bool(correct["correct"])


def _read_summary(raw: bytes | None, filename: str) -> dict[str, Any]:
    if raw is None:
        raise RuntimeError(f"Evaluator did not produce a safe summary: {filename}")
    if len(raw) > MAX_SUMMARY_BYTES:
        raise RuntimeError(f"Evaluator summary exceeds {MAX_SUMMARY_BYTES} bytes: {filename}")
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Evaluator produced invalid JSON: {filename}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Evaluator summary must contain a JSON object: {filename}")
    return parsed


def validate_summaries(
    metrics_raw: bytes | None, correct_raw: bytes | None
) -> tuple[dict[str, Any], bool]:
    """Apply upstream's summary checks of the evaluation wrapper (``_validated_summaries``).

    Raises:
        RuntimeError: With upstream's message when a summary is missing or invalid.
    """
    metrics = _read_summary(metrics_raw, "metrics.json")
    correct = _read_summary(correct_raw, "correct.json")
    score = metrics.get("combined_score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        raise RuntimeError("metrics.json combined_score must be numeric")
    if not math.isfinite(float(score)):
        raise RuntimeError("metrics.json combined_score must be finite")
    if not isinstance(correct.get("correct"), bool):
        raise RuntimeError("correct.json correct must be a boolean")
    return metrics, bool(correct["correct"])


def parse_qualification(raw: bytes | None) -> bool | None:
    """Return ``feasible`` of a ``qualification.json``.

    The file is ``{"schema_version": 1, "feasible": <bool>}``.
    """
    if raw is None:
        return None
    payload = _load_result_mapping(raw)
    if payload.get("schema_version") != 1 or not isinstance(payload.get("feasible"), bool):
        raise EvaluatorOutputContractError(
            'qualification.json must be {"schema_version": 1, "feasible": <bool>}'
        )
    return bool(payload["feasible"])


# --- compute usage ------------------------------------------------------------------------


def validate_compute_usage(value: Any) -> None:
    """Validate a compute-usage record with upstream's rules (``validate_compute_usage``)."""
    if not isinstance(value, Mapping) or set(value.keys()) != _COMPUTE_USAGE_FIELDS:
        raise ValueError("compute usage fields are invalid")
    for field in ("wall_clock_limit_seconds", "elapsed_seconds"):
        number = value[field]
        if not (_finite(number) and number >= 0):
            raise ValueError(f"compute usage {field} must be finite and nonnegative")
    if value["wall_clock_limit_seconds"] == 0:
        raise ValueError("compute usage time limit must be positive")
    if type(value["max_rpc_calls"]) is not int or value["max_rpc_calls"] < 1:
        raise ValueError("compute usage RPC limit must be positive")
    calls = value["rpc_calls"]
    if calls is not None and (type(calls) is not int or calls < 0):
        raise ValueError("compute usage RPC calls must be nonnegative or unknown")
    if not isinstance(value["outcome"], str) or value["outcome"] not in {
        "completed",
        "timeout",
        "rpc_limit",
        "failed",
    }:
        raise ValueError("compute usage outcome is invalid")
    if value["outcome"] == "completed" and calls is not None and calls > value["max_rpc_calls"]:
        raise ValueError("completed compute usage exceeds RPC limit")
    if value["objective_counts"] != "advisory":
        raise ValueError("internal objective counts cannot certify hard budget compliance")


def compute_usage(
    *,
    wall_clock_limit_seconds: float,
    elapsed_seconds: float,
    max_rpc_calls: int,
    rpc_calls: int | None,
    outcome: str,
) -> ComputeUsage:
    """Build and validate a compute-usage record (upstream ``compute_usage``)."""
    record = {
        "wall_clock_limit_seconds": wall_clock_limit_seconds,
        "elapsed_seconds": elapsed_seconds,
        "max_rpc_calls": max_rpc_calls,
        "rpc_calls": rpc_calls,
        "outcome": outcome,
        "objective_counts": "advisory",
    }
    validate_compute_usage(record)
    return ComputeUsage.model_validate(record)


__all__ = [
    "EVIDENCE_FILENAME",
    "MAX_EVIDENCE_BYTES",
    "compute_usage",
    "decode_evaluation_evidence",
    "evidence_metadata",
    "parse_evaluation_evidence",
    "parse_qualification",
    "validate_compute_usage",
    "validate_evaluator_outputs",
    "validate_summaries",
]
