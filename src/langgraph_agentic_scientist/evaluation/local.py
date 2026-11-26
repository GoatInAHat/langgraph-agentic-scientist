"""Evaluators that run the task's evaluator scripts as local processes.

:class:`LocalCommandEvaluator` runs ``<task>/evaluate.py --program_path <entrypoint>
--results_dir <dir>`` (``evaluate_final.py`` for the final evaluator of a drafting smoke check)
and :class:`LocalFinalEvaluator` runs the sealed ``evaluate_final.py`` once per id, writing
upstream's ``invocation.json`` and final manifest (schema 3). Both work through a
:class:`~langgraph_agentic_scientist.workspace.Workspace` and keep their files in an evaluator
area (``.evaluations/<id>/``, ``.final_evaluations/<id>/``) that no agent root contains.

.. warning::
   These evaluators run the candidate as a local process **without isolation** (their
   ``isolated`` is false). Upstream runs contracted tasks only in containers; the package refuses
   to search or finally evaluate a contracted task with an evaluator that is not isolated unless
   ``config.allow_unisolated_contracted`` is set. A local process cannot outlive its caller, so an
   id whose evaluation started without being recorded comes back ``interrupted`` (search planes)
   or raises :class:`~langgraph_agentic_scientist.core.errors.FinalEvaluationAlreadyStarted`.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

from ..core.errors import (
    EvaluatorOutputContractError,
    FinalEvaluationAlreadyStarted,
    FinalEvaluationError,
    ProgramSpecError,
)
from ..core.state import TaskRef, utc_now
from ..workspace.digest import sha256_bytes
from ..workspace.protocol import Workspace
from .evidence import (
    EVIDENCE_FILENAME,
    compute_usage,
    evidence_metadata,
    parse_evaluation_evidence,
    parse_qualification,
    validate_compute_usage,
    validate_evaluator_outputs,
    validate_summaries,
)
from .ids import id_to_path
from .program_spec import resolve_entrypoint
from .protocol import (
    EvaluationEvidence,
    EvaluationRequest,
    EvaluationResult,
    FinalEvaluationRequest,
    FinalEvaluationResult,
)

DEFAULT_TIMEOUT_SECONDS = 21_600.0
MAX_OUTPUT_BYTES = 64 * 1024
SUMMARY_BYTES = 2048
_SENSITIVE_ENV_TOKENS = ("AUTH", "CREDENTIAL", "KEY", "PASSWORD", "PROXY", "SECRET", "TOKEN")
_SAFE_ENV_NAMES = {
    "AGENTIC_DATA_DIR",
    "DATA_DIR",
    "FINEWEB_DATA_DIR",
    "GRPO_TEST_DATASET_PATH",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PATH",
    "TASK_SOURCE_DIR",
    "TERM",
    "TZ",
    "USER",
    "VENV_DIR",
}
_SAFE_ENV_PREFIXES = (
    "CUDA_",
    "MKL_",
    "NCCL_",
    "NUMEXPR_",
    "NVIDIA_",
    "OMP_",
    "ROCR_",
    "TORCH_",
    "XLA_",
)
_PYTHON_ENV_BLOCKLIST = {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"}
_RESULT_FILES = {"metrics.json", "correct.json", "evaluation_manifest.json", ".evaluation_pending"}
_FINAL_MANIFEST_V1_FIELDS = {
    "schema_version",
    "status",
    "started_at",
    "completed_at",
    "correct",
    "metrics",
    "declared_final_metrics",
    "task_tree_sha256",
    "evaluator_sha256",
    "program_tree_sha256",
    "results_tree_sha256",
}
_FINAL_MANIFEST_V2_FIELDS = _FINAL_MANIFEST_V1_FIELDS | {"outcome"}
_FINAL_MANIFEST_V3_FIELDS = _FINAL_MANIFEST_V2_FIELDS | {"evaluation_evidence", "compute_usage"}
_INVOCATION_FIELDS = {
    "schema_version",
    "status",
    "started_at",
    "task_tree_sha256",
    "evaluator_sha256",
    "program_tree_sha256",
}


def evaluation_environment(
    home: str, tmp: str, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return the sanitized evaluator environment of upstream's evaluation wrapper.

    Only allowlisted variables of the calling process pass; names that look like credentials
    never do. ``HOME`` and ``TMPDIR`` point into the evaluation's own runtime directory.
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in _PYTHON_ENV_BLOCKLIST
        and not any(token in key.upper() for token in _SENSITIVE_ENV_TOKENS)
        and (key in _SAFE_ENV_NAMES or key.startswith(_SAFE_ENV_PREFIXES))
    }
    environment.update(
        HOME=home,
        TMPDIR=tmp,
        PIP_CONFIG_FILE=os.devnull,
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
    )
    environment.update(dict(extra or {}))
    return environment


def _bounded(text: str, limit: int = SUMMARY_BYTES) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return "..." + data[-limit:].decode("utf-8", errors="ignore")


async def _optional_bytes(workspace: Workspace, path: str) -> bytes | None:
    entry = await workspace.stat(path)
    if entry is None:
        return None
    if entry.kind != "file":
        raise EvaluatorOutputContractError("required evaluator result is missing or unsafe")
    return await workspace.read_bytes(path)


async def _make_dir(workspace: Workspace, path: str) -> None:
    keep = f"{path}/.keep"
    await workspace.write_bytes(keep, b"")
    await workspace.remove(keep)


async def _program_fingerprint(workspace: Workspace, program_dir: str) -> dict[str, str]:
    fingerprint: dict[str, str] = {}
    for entry in await workspace.list_tree(program_dir, with_digest=True):
        root_name = entry.path.split("/", 1)[0]
        if root_name in _RESULT_FILES or root_name.startswith(".evaluation_stage_"):
            continue
        fingerprint[entry.path] = f"{entry.kind}:{entry.sha256 or entry.target or ''}"
    return fingerprint


def _evaluator_file(task: TaskRef, which: str) -> str:
    if task.contract is not None:
        return (
            task.contract.evaluators.search if which == "search" else task.contract.evaluators.final
        )
    return "evaluate.py" if which == "search" else "evaluate_final.py"


def _metric_names(task: TaskRef, which: str) -> list[str] | None:
    if task.contract is None:
        return None
    metrics = task.contract.metrics.search if which == "search" else task.contract.metrics.final
    return [metric.name for metric in metrics]


def _join(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part not in ("", "."))


class LocalCommandEvaluator:
    """Run the task's evaluator as a local process (see the module docstring).

    Args:
        workspace: The run's workspace; programs and tasks are read from it and evaluator
            outputs are kept under ``area``.
        python: Interpreter for the evaluator (default: the current interpreter). Upstream uses
            the task runtime's interpreter; pass that interpreter here.
        area: Workspace directory of the evaluator's own files.
        env: Extra environment variables for every evaluation.
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        python: str | None = None,
        area: str = ".evaluations",
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.workspace = workspace
        self.python = python or sys.executable
        self.area = area
        self.env = dict(env or {})

    @property
    def isolated(self) -> bool:
        """Always false: the candidate runs as a local process."""
        return False

    async def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Evaluate a program, or return the recorded result for a known id."""
        workspace = self.workspace
        area = _join(self.area, id_to_path(request.evaluation_id))
        result_path = f"{area}/result.json"
        recorded = await _optional_bytes(workspace, result_path)
        if recorded is not None:
            return EvaluationResult.model_validate_json(recorded)
        invocation = {
            "evaluation_id": request.evaluation_id,
            "status": "started",
            "started_at": utc_now(),
            "plane": request.plane,
            "program": request.program.path,
        }
        try:
            await workspace.write_bytes(
                f"{area}/invocation.json",
                (json.dumps(invocation, indent=2, sort_keys=True) + "\n").encode(),
                exclusive=True,
            )
        except FileExistsError:
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="interrupted",
                summary=(
                    "An evaluation with this id started earlier and its result can no longer be "
                    "obtained; it was not started again."
                ),
                output_dir=f"{area}/results",
            )
        started = time.monotonic()
        try:
            result = await self._run(request, area)
        except Exception as exc:
            result = EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="infrastructure_error",
                summary=_bounded(f"{type(exc).__name__}: {exc}"),
                output_dir=f"{area}/results",
            )
        result = result.model_copy(update={"duration_seconds": time.monotonic() - started})
        await workspace.write_bytes(result_path, result.model_dump_json(by_alias=True).encode())
        return result

    async def _run(self, request: EvaluationRequest, area: str) -> EvaluationResult:
        workspace = self.workspace
        task_root = request.task.root
        evaluator_rel = _evaluator_file(request.task, request.evaluator)
        evaluator_path = _join(task_root, evaluator_rel)
        evaluator_entry = await workspace.stat(evaluator_path)
        if evaluator_entry is None or evaluator_entry.kind != "file":
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="infrastructure_error",
                summary=f"Evaluator must be a real file: {evaluator_path}",
            )
        program_dir = request.program.path
        try:
            entrypoint = await resolve_entrypoint(workspace, program_dir)
        except ProgramSpecError as exc:
            return EvaluationResult(
                evaluation_id=request.evaluation_id,
                status="failed",
                failure_category="invalid_program_spec",
                summary=_bounded(str(exc)),
            )
        results_dir = f"{area}/results"
        home_dir = f"{area}/runtime/home"
        tmp_dir = f"{area}/runtime/tmp"
        for directory in (results_dir, home_dir, tmp_dir):
            await _make_dir(workspace, directory)
        before = await _program_fingerprint(workspace, program_dir)
        argv = [
            self.python,
            workspace.host_path(evaluator_path),
            "--program_path",
            workspace.host_path(_join(program_dir, entrypoint)),
            "--results_dir",
            workspace.host_path(results_dir),
        ]
        if request.num_replications > 1:
            argv += ["--num_replications", str(request.num_replications)]
        env = evaluation_environment(
            workspace.host_path(home_dir),
            workspace.host_path(tmp_dir),
            {**self.env, **request.environment},
        )
        timeout = request.timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        completed = await workspace.run(
            argv, cwd=task_root, env=env, timeout=timeout, max_output_bytes=MAX_OUTPUT_BYTES
        )
        await workspace.write_bytes(f"{area}/stdout.log", completed.stdout.encode("utf-8"))
        await workspace.write_bytes(f"{area}/stderr.log", completed.stderr.encode("utf-8"))
        base: dict[str, Any] = {
            "evaluation_id": request.evaluation_id,
            "output_dir": results_dir,
            "returncode": completed.returncode,
        }
        if completed.timed_out:
            return EvaluationResult(
                **base,
                status="timeout",
                summary=_bounded(f"evaluation timed out after {timeout:g} s\n{completed.stderr}"),
            )
        if completed.returncode != 0:
            return EvaluationResult(
                **base,
                status="failed",
                failure_category="evaluator_failed",
                summary=_bounded(
                    f"evaluator exited with {completed.returncode}\n"
                    f"{completed.stderr or completed.stdout}"
                ),
            )
        after = await _program_fingerprint(workspace, program_dir)
        if any(after.get(path) != digest for path, digest in before.items()):
            return EvaluationResult(
                **base,
                status="failed",
                failure_category="candidate_changed",
                summary="Candidate program changed during evaluation",
            )
        return await self._read_results(request, results_dir, base)

    async def _read_results(
        self, request: EvaluationRequest, results_dir: str, base: dict[str, Any]
    ) -> EvaluationResult:
        workspace = self.workspace
        try:
            metrics_raw = await _optional_bytes(workspace, f"{results_dir}/metrics.json")
            correct_raw = await _optional_bytes(workspace, f"{results_dir}/correct.json")
            evidence_raw = await _optional_bytes(workspace, f"{results_dir}/{EVIDENCE_FILENAME}")
            names = _metric_names(request.task, request.evaluator)
            if names is not None:
                metrics, correct = validate_evaluator_outputs(
                    metrics_raw, correct_raw, evidence_raw, names
                )
            else:
                metrics, correct = validate_summaries(metrics_raw, correct_raw)
            evidence: EvaluationEvidence | None = None
            if evidence_raw is not None:
                try:
                    evidence = parse_evaluation_evidence(evidence_raw)
                except ValueError as error:
                    raise EvaluatorOutputContractError(str(error)) from error
            feasible = None
            if request.plane == "qualification":
                feasible = parse_qualification(
                    await _optional_bytes(workspace, f"{results_dir}/qualification.json")
                )
        except EvaluatorOutputContractError as exc:
            return EvaluationResult(
                **base,
                status="failed",
                failure_category=EvaluatorOutputContractError.category,
                summary=_bounded(str(exc)),
            )
        except RuntimeError as exc:
            return EvaluationResult(
                **base,
                status="failed",
                failure_category="invalid_summary",
                summary=_bounded(str(exc)),
            )
        score = metrics.get("combined_score")
        return EvaluationResult(
            **base,
            status="completed",
            metrics=metrics,
            correct=correct,
            feasible=feasible,
            evidence=evidence,
            summary=f"returncode=0; combined_score={score}; correct={str(correct).lower()}",
        )


def _check_exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - value.keys()
    unexpected = value.keys() - expected
    if missing:
        raise FinalEvaluationError(f"{label} missing fields: {sorted(missing)!r}")
    if unexpected:
        raise FinalEvaluationError(f"{label} has unexpected fields: {sorted(unexpected)!r}")


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def validate_final_manifest(manifest: Mapping[str, Any]) -> FinalEvaluationResult:
    """Apply upstream's field rules for final manifests (schemas 1-3) and return the result.

    These are the rules of upstream ``verify_final_manifest`` that concern the manifest itself;
    the digests of results, task and program are compared by the caller that holds them.

    Raises:
        FinalEvaluationError: With upstream's message.
    """
    schema_version = manifest.get("schema_version")
    if schema_version == 1:
        _check_exact_fields(manifest, _FINAL_MANIFEST_V1_FIELDS, "final manifest")
    elif schema_version == 2:
        _check_exact_fields(manifest, _FINAL_MANIFEST_V2_FIELDS, "final manifest")
    elif schema_version == 3:
        _check_exact_fields(manifest, _FINAL_MANIFEST_V3_FIELDS, "final manifest")
    else:
        raise FinalEvaluationError("final manifest has an unsupported schema version")
    if manifest["status"] != "completed":
        raise FinalEvaluationError("final manifest is not a completed artifact")
    correct = manifest["correct"]
    if not isinstance(correct, bool):
        raise FinalEvaluationError("final manifest correct must be boolean")
    if schema_version == 1 and correct is not True:
        raise FinalEvaluationError("schema-v1 final manifest must report correct=true")
    outcome = "qualified" if correct else "not_qualified"
    if schema_version in (2, 3) and manifest["outcome"] != outcome:
        raise FinalEvaluationError("final manifest outcome does not match correct")
    for field in ("started_at", "completed_at"):
        if not isinstance(manifest[field], str) or not manifest[field].strip():
            raise FinalEvaluationError(f"invalid final manifest {field}")
    declared_raw = manifest["declared_final_metrics"]
    if (
        not isinstance(declared_raw, list)
        or not declared_raw
        or any(not isinstance(name, str) or not name for name in declared_raw)
        or len(set(declared_raw)) != len(declared_raw)
    ):
        raise FinalEvaluationError("invalid declared_final_metrics")
    metrics = manifest["metrics"]
    if not isinstance(metrics, dict):
        raise FinalEvaluationError("invalid final manifest metrics")
    for name in declared_raw:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise FinalEvaluationError(f"declared final metric is not numeric: {name}")
        if not math.isfinite(float(value)):
            raise FinalEvaluationError(f"declared final metric is not finite: {name}")
    if set(metrics) != set(declared_raw):
        raise FinalEvaluationError("manifest metrics do not exactly match declared metrics")
    for field in (
        "task_tree_sha256",
        "evaluator_sha256",
        "program_tree_sha256",
        "results_tree_sha256",
    ):
        value = manifest[field]
        if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
            raise FinalEvaluationError(f"invalid {field}")
    usage = manifest.get("compute_usage")
    if schema_version == 3:
        try:
            validate_compute_usage(usage)
        except ValueError as exc:
            raise FinalEvaluationError(str(exc)) from exc
        if not isinstance(usage, dict) or usage["outcome"] != "completed":
            raise FinalEvaluationError("completed final manifest has failed compute usage")
    return FinalEvaluationResult.model_validate(
        {
            "status": "completed",
            "correct": correct,
            "outcome": outcome,
            "metrics": {name: float(metrics[name]) for name in declared_raw},
            "declared_final_metrics": list(declared_raw),
            "task_tree_sha256": manifest["task_tree_sha256"],
            "evaluator_sha256": manifest["evaluator_sha256"],
            "program_tree_sha256": manifest["program_tree_sha256"],
            "results_tree_sha256": manifest["results_tree_sha256"],
            "started_at": manifest["started_at"],
            "completed_at": manifest["completed_at"],
            "evaluation_evidence": manifest.get("evaluation_evidence"),
            "compute_usage": usage,
            "manifest_schema_version": schema_version,
        }
    )


class LocalFinalEvaluator:
    """Run the sealed ``evaluate_final.py`` once per id as a local process.

    Args:
        workspace: The run's workspace.
        python: Interpreter for the evaluator (default: the current interpreter).
        area: Workspace directory of the final evaluations' own files.
        env: Extra environment variables for the evaluator.
        default_max_rpc_calls: ``max_rpc_calls`` recorded in ``compute_usage`` for tasks
            without a contract (upstream's default for contracts).
    """

    def __init__(
        self,
        workspace: Workspace,
        *,
        python: str | None = None,
        area: str = ".final_evaluations",
        env: Mapping[str, str] | None = None,
        default_max_rpc_calls: int = 100_000,
    ) -> None:
        self.workspace = workspace
        self.python = python or sys.executable
        self.area = area
        self.env = dict(env or {})
        self.default_max_rpc_calls = default_max_rpc_calls

    @property
    def isolated(self) -> bool:
        """Always false: the candidate runs as a local process."""
        return False

    async def _verify_recorded(self, out: str, manifest_raw: bytes) -> FinalEvaluationResult:
        try:
            manifest = json.loads(manifest_raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise FinalEvaluationError(f"invalid final manifest: {out}") from exc
        result = validate_final_manifest(manifest)
        results_digest = await self.workspace.digest(f"{out}/results")
        if results_digest != result.results_tree_sha256:
            raise FinalEvaluationError("final results digest mismatch")
        invocation = json.loads(await self.workspace.read_bytes(f"{out}/invocation.json"))
        _check_exact_fields(invocation, _INVOCATION_FIELDS, "final invocation")
        expected = {
            "schema_version": 1,
            "status": "started",
            "started_at": result.started_at,
            "task_tree_sha256": result.task_tree_sha256,
            "evaluator_sha256": result.evaluator_sha256,
            "program_tree_sha256": result.program_tree_sha256,
        }
        if invocation != expected:
            raise FinalEvaluationError("final invocation does not match final manifest")
        return result.model_copy(update={"output_dir": out})

    async def _task_digest(self, task: TaskRef) -> str:
        if task.contract is not None:
            from ..core.contract import task_scaffold_fingerprint

            return await task_scaffold_fingerprint(self.workspace, task.root)
        return await self.workspace.digest(task.root)

    async def evaluate_sealed(self, request: FinalEvaluationRequest) -> FinalEvaluationResult:
        """Run the sealed evaluation once, or replay the verified manifest of a known id."""
        workspace = self.workspace
        out = _join(self.area, id_to_path(request.evaluation_id))
        manifest_raw = await _optional_bytes(workspace, f"{out}/final_manifest.json")
        if manifest_raw is not None:
            return await self._verify_recorded(out, manifest_raw)
        if await workspace.exists(f"{out}/invocation.json"):
            raise FinalEvaluationAlreadyStarted(request.evaluation_id)
        task = request.task
        program = request.frozen.program
        evaluator_path = _join(task.root, _evaluator_file(task, "final"))
        entry = await workspace.stat(evaluator_path)
        if entry is None or entry.kind != "file":
            raise FinalEvaluationError(f"invalid final evaluator: {evaluator_path}")
        program_digest = await workspace.digest(program.path)
        if program_digest != request.frozen.program_tree_sha256:
            raise FinalEvaluationError("final program digest mismatch")
        evaluator_digest = sha256_bytes(await workspace.read_bytes(evaluator_path))
        task_digest = await self._task_digest(task)
        entrypoint = await resolve_entrypoint(workspace, program.path)
        started_at = utc_now()
        invocation = {
            "schema_version": 1,
            "status": "started",
            "started_at": started_at,
            "task_tree_sha256": task_digest,
            "evaluator_sha256": evaluator_digest,
            "program_tree_sha256": program_digest,
        }
        await workspace.write_bytes(
            f"{out}/invocation.json",
            (json.dumps(invocation, indent=2, sort_keys=True) + "\n").encode(),
            exclusive=True,
        )
        results_dir = f"{out}/results"
        home_dir = f"{out}/runtime/home"
        tmp_dir = f"{out}/runtime/tmp"
        for directory in (results_dir, home_dir, tmp_dir):
            await _make_dir(workspace, directory)
        contract = task.contract
        limit = (
            float(contract.compute.final.wall_clock_seconds)
            if contract
            else DEFAULT_TIMEOUT_SECONDS
        )
        if request.timeout_seconds is not None:
            limit = min(limit, float(request.timeout_seconds))
        max_rpc_calls = (
            contract.compute.final.max_candidate_rpc_calls
            if contract
            else self.default_max_rpc_calls
        )
        extra_env = dict(self.env)
        if request.sealed_inputs is not None:
            extra_env["AGENTIC_SEALED_INPUTS_REF"] = request.sealed_inputs.ref
        started = time.monotonic()
        completed = await workspace.run(
            [
                self.python,
                workspace.host_path(evaluator_path),
                "--program_path",
                workspace.host_path(_join(program.path, entrypoint)),
                "--results_dir",
                workspace.host_path(results_dir),
            ],
            cwd=task.root,
            env=evaluation_environment(
                workspace.host_path(home_dir), workspace.host_path(tmp_dir), extra_env
            ),
            timeout=limit,
            max_output_bytes=MAX_OUTPUT_BYTES,
        )
        elapsed = time.monotonic() - started
        await workspace.write_bytes(f"{out}/stdout.log", completed.stdout.encode("utf-8"))
        await workspace.write_bytes(f"{out}/stderr.log", completed.stderr.encode("utf-8"))
        if completed.timed_out:
            raise FinalEvaluationError("final evaluation deadline exceeded")
        if completed.returncode != 0:
            raise FinalEvaluationError(
                f"final evaluator exited with {completed.returncode}: "
                f"{_bounded(completed.stderr or completed.stdout, 512)}"
            )
        names = [metric.name for metric in contract.metrics.final] if contract else None
        result_entries = {item.path for item in await workspace.list_tree(results_dir)}
        required = {"metrics.json", "correct.json"}
        if not required <= result_entries or result_entries - (required | {EVIDENCE_FILENAME}):
            raise FinalEvaluationError("final evaluator emitted undeclared result artifacts")
        metrics_raw = await workspace.read_bytes(f"{results_dir}/metrics.json")
        correct_raw = await workspace.read_bytes(f"{results_dir}/correct.json")
        evidence_raw = await _optional_bytes(workspace, f"{results_dir}/{EVIDENCE_FILENAME}")
        try:
            metrics_value = json.loads(metrics_raw)
            correct_value = json.loads(correct_raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise FinalEvaluationError("final results are not valid JSON") from exc
        if (
            not isinstance(correct_value, dict)
            or set(correct_value) != {"correct"}
            or not isinstance(correct_value["correct"], bool)
        ):
            raise FinalEvaluationError(
                "final correctness must contain exactly one boolean field: correct"
            )
        if not isinstance(metrics_value, dict):
            raise FinalEvaluationError("invalid final metrics")
        declared = names if names is not None else sorted(metrics_value)
        if set(metrics_value) != set(declared):
            raise FinalEvaluationError(
                "final metric names must exactly match declared metrics: "
                f"declared={sorted(declared)!r}, actual={sorted(metrics_value)!r}"
            )
        for name in declared:
            if not _finite_number(metrics_value[name]):
                raise FinalEvaluationError(f"declared final metric is not numeric: {name}")
        try:
            evidence = evidence_metadata(evidence_raw)
        except ValueError as exc:
            raise FinalEvaluationError(str(exc)) from exc
        if await workspace.digest(program.path) != program_digest:
            raise FinalEvaluationError("final program digest mismatch")
        if await self._task_digest(task) != task_digest:
            raise FinalEvaluationError("final task digest mismatch")
        usage = compute_usage(
            wall_clock_limit_seconds=limit,
            elapsed_seconds=elapsed,
            max_rpc_calls=max_rpc_calls,
            rpc_calls=None,
            outcome="completed",
        )
        correct = bool(correct_value["correct"])
        manifest = {
            "schema_version": 3,
            "status": "completed",
            "started_at": started_at,
            "completed_at": utc_now(),
            "correct": correct,
            "outcome": "qualified" if correct else "not_qualified",
            "metrics": {name: metrics_value[name] for name in declared},
            "declared_final_metrics": list(declared),
            "task_tree_sha256": task_digest,
            "evaluator_sha256": evaluator_digest,
            "program_tree_sha256": program_digest,
            "results_tree_sha256": await workspace.digest(results_dir),
            "evaluation_evidence": evidence.model_dump(exclude={"path"}, exclude_none=True),
            "compute_usage": usage.model_dump(),
        }
        result = validate_final_manifest(manifest)
        await workspace.write_bytes(
            f"{out}/final_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            exclusive=True,
        )
        return result.model_copy(update={"output_dir": out})


__all__ = [
    "LocalCommandEvaluator",
    "LocalFinalEvaluator",
    "evaluation_environment",
    "validate_final_manifest",
]
