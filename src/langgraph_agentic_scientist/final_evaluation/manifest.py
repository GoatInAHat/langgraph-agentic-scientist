"""Resolving, freezing and verifying: the host rules of the sealed final evaluation.

Upstream's ``final_evaluation/runner.py`` mixes these rules with the container broker. The broker
(launch, retries, GPU leases, reset-transcript enforcement) belongs to the caller's
:class:`~langgraph_agentic_scientist.evaluation.FinalEvaluator`; what stays here is the method:

* :func:`resolve_incumbent` is ``resolve_frozen_incumbent``: a downstream selection first, else
  the investigation's best program, validated for a safe entrypoint;
* :func:`freeze_incumbent` computes the digests of the program, the task scaffold and the final
  evaluator that bind the evaluation (``scaffold_digest``, ``program_digest``,
  ``evaluator_digest`` of ``run_final_evaluation``);
* :func:`verify_final_result` applies the manifest rules of ``verify_final_manifest`` to the
  evaluator's result: the field rules of schemas 1 to 3 (through
  :func:`~langgraph_agentic_scientist.evaluation.validate_final_manifest`), the digests against the
  frozen inputs, the declared metrics and the compute limits against the task contract, and, when
  the evaluator kept its results where the workspace can read them, the digest of the results.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from ..core.contract import task_scaffold_fingerprint
from ..core.errors import FinalEvaluationError, TaskContractError
from ..core.results import DownstreamSelection, InvestigationResult
from ..core.state import BestManifest, ProgramRef, TaskRef, utc_now
from ..evaluation.local import validate_final_manifest
from ..evaluation.protocol import EvidenceMetadata, FinalEvaluationResult, FreezeRecord
from ..workspace.digest import sha256_bytes
from ..workspace.paths import is_within, join, safe_relpath
from ..workspace.protocol import Workspace

DEFAULT_FINAL_EVALUATOR = "evaluate_final.py"
"""The final evaluator of a task without a contract."""


@dataclass(frozen=True)
class Incumbent:
    """The program the downstream phases freeze, with what is known about it.

    Attributes:
        program: The program directory and entrypoint.
        source: ``explicit`` (the caller named it), ``downstream`` (the no-winner selection) or
            ``best`` (the investigation's best program).
        combined_score: The program's score when known.
        correct: Whether the program was correct when known.
        source_attempt_id: The attempt that produced it.
        extra: Further provenance (the parallel worker index, the downstream selection record).
    """

    program: ProgramRef
    source: Literal["explicit", "downstream", "best"]
    combined_score: float | None = None
    correct: bool | None = None
    source_attempt_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _entrypoint(program_dir: str, entrypoint: str, label: str) -> str:
    relative = PurePosixPath(entrypoint)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise FinalEvaluationError(f"invalid {label} entrypoint")
    return posixpath.normpath(entrypoint)


async def _validated_program(
    workspace: Workspace, program_dir: str, entrypoint: str, label: str
) -> ProgramRef:
    try:
        directory = safe_relpath(program_dir, allow_root=False)
    except ValueError as exc:
        raise FinalEvaluationError(f"invalid {label}: {exc}") from exc
    entry = await workspace.stat(directory)
    if entry is None or entry.kind != "dir":
        raise FinalEvaluationError(f"{label} is not a directory: {directory}")
    relative = _entrypoint(directory, entrypoint, label)
    file_entry = await workspace.stat(join(directory, relative))
    if file_entry is None or file_entry.kind != "file":
        raise FinalEvaluationError(f"invalid {label} entrypoint")
    return ProgramRef(path=directory, entrypoint=relative)


async def resolve_incumbent(
    workspace: Workspace,
    *,
    incumbent: ProgramRef | None = None,
    incumbent_manifest: BestManifest | None = None,
    downstream: DownstreamSelection | None = None,
    investigation: InvestigationResult | None = None,
) -> Incumbent:
    """Resolve the frozen incumbent (upstream ``resolve_frozen_incumbent``).

    An explicit ``incumbent`` is used as given. Otherwise a selected downstream program wins over
    the investigation's best program, which must lie inside the investigation's results
    directory.

    Raises:
        FinalEvaluationError: With upstream's messages when there is no incumbent or it is unsafe.
    """
    if incumbent is not None:
        program = await _validated_program(
            workspace, incumbent.path, incumbent.entrypoint, "frozen incumbent"
        )
        manifest = incumbent_manifest
        return Incumbent(
            program=program,
            source="explicit",
            combined_score=manifest.combined_score if manifest else None,
            correct=manifest.correct if manifest else None,
            source_attempt_id=manifest.source_attempt_id if manifest else None,
        )
    if downstream is not None and downstream.status == "selected" and downstream.program:
        program = await _validated_program(
            workspace, downstream.program.path, downstream.program.entrypoint, "frozen incumbent"
        )
        return Incumbent(
            program=program,
            source="downstream",
            combined_score=downstream.combined_score,
            correct=True,
            source_attempt_id=downstream.evaluation_id,
            extra={"downstream_selection": downstream.model_dump(mode="json")},
        )
    best = investigation.best if investigation is not None else None
    if best is None:
        raise FinalEvaluationError("best-program manifest has no frozen incumbent")
    directory = safe_relpath(best.program_dir, allow_root=False)
    results_dir = investigation.results_dir if investigation is not None else None
    if results_dir and not is_within(directory, results_dir):
        raise FinalEvaluationError("frozen incumbent escapes investigation results")
    program = await _validated_program(
        workspace, directory, best.entrypoint_path, "frozen incumbent"
    )
    extra = dict(best.model_extra or {})
    return Incumbent(
        program=program,
        source="best",
        combined_score=best.combined_score,
        correct=best.correct,
        source_attempt_id=best.source_attempt_id,
        extra=extra,
    )


def final_evaluator_path(task: TaskRef) -> str:
    """Return the task-relative path of the sealed final evaluator."""
    if task.contract is not None:
        return task.contract.evaluators.final
    return DEFAULT_FINAL_EVALUATOR


async def task_digest(workspace: Workspace, task: TaskRef) -> str:
    """Return the digest that binds a task: a contract's scaffold fingerprint, else the tree."""
    if task.contract is not None:
        try:
            return await task_scaffold_fingerprint(workspace, task.root)
        except TaskContractError as exc:
            raise FinalEvaluationError(f"task scaffold became invalid: {exc}") from exc
    return await workspace.digest(task.root)


async def evaluator_digest(workspace: Workspace, task: TaskRef) -> str:
    """Return the digest of the task's sealed final evaluator file."""
    path = join(task.root, final_evaluator_path(task))
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        raise FinalEvaluationError(f"invalid final evaluator: {path}")
    return sha256_bytes(await workspace.read_bytes(path))


async def freeze_incumbent(
    workspace: Workspace, task: TaskRef, program: ProgramRef
) -> FreezeRecord:
    """Digest the program, the task scaffold and the final evaluator, and record the freeze."""
    program_digest = await workspace.digest(program.path)
    return FreezeRecord(
        program=program.model_copy(update={"digest": program_digest}),
        program_tree_sha256=program_digest,
        task_tree_sha256=await task_digest(workspace, task),
        evaluator_sha256=await evaluator_digest(workspace, task),
        frozen_at=utc_now(),
    )


def manifest_from_result(result: FinalEvaluationResult) -> dict[str, Any]:
    """Return the final manifest (of the result's schema version) that the result mirrors."""
    version = result.manifest_schema_version
    manifest: dict[str, Any] = {
        "schema_version": version,
        "status": result.status,
        "started_at": result.started_at,
        "completed_at": result.completed_at,
        "correct": result.correct,
        "metrics": dict(result.metrics),
        "declared_final_metrics": list(result.declared_final_metrics),
        "task_tree_sha256": result.task_tree_sha256,
        "evaluator_sha256": result.evaluator_sha256,
        "program_tree_sha256": result.program_tree_sha256,
        "results_tree_sha256": result.results_tree_sha256,
    }
    if version >= 2:
        manifest["outcome"] = result.outcome
    if version >= 3:
        evidence = result.evaluation_evidence or EvidenceMetadata(status="unavailable")
        manifest["evaluation_evidence"] = evidence.model_dump(exclude={"path"}, exclude_none=True)
        manifest["compute_usage"] = (
            result.compute_usage.model_dump() if result.compute_usage is not None else None
        )
    return manifest


async def verify_final_result(
    workspace: Workspace,
    *,
    task: TaskRef,
    frozen: FreezeRecord,
    result: FinalEvaluationResult,
) -> FinalEvaluationResult:
    """Verify an evaluator's result as upstream verifies a completed final artifact.

    The field rules of schemas 1 to 3 come first, then the digests: the program and the task must
    be unchanged since the freeze and the result must be bound to them and to the evaluator the
    freeze recorded; with a contract the declared final metrics and the compute limits must be
    the contract's.

    Raises:
        FinalEvaluationError: With upstream's message for the first rule that fails.
    """
    verified = validate_final_manifest(manifest_from_result(result))
    verified = verified.model_copy(
        update={
            "output_dir": result.output_dir,
            "manifest_schema_version": verified.manifest_schema_version,
        }
    )
    if await workspace.digest(frozen.program.path) != frozen.program_tree_sha256:
        raise FinalEvaluationError("frozen incumbent changed during final evaluation")
    if await task_digest(workspace, task) != frozen.task_tree_sha256:
        raise FinalEvaluationError("task scaffold changed during final evaluation")
    if result.task_tree_sha256 != frozen.task_tree_sha256:
        raise FinalEvaluationError("final task digest mismatch")
    if result.evaluator_sha256 != frozen.evaluator_sha256:
        raise FinalEvaluationError("final evaluator digest mismatch")
    if result.program_tree_sha256 != frozen.program_tree_sha256:
        raise FinalEvaluationError("final program digest mismatch")
    contract = task.contract
    if contract is not None:
        contract_metrics = [metric.name for metric in contract.metrics.final]
        if contract_metrics != result.declared_final_metrics:
            raise FinalEvaluationError("final manifest metrics differ from task contract")
        usage = result.compute_usage
        if (
            result.manifest_schema_version >= 3
            and usage is not None
            and (
                usage.max_rpc_calls != contract.compute.final.max_candidate_rpc_calls
                or usage.wall_clock_limit_seconds > contract.compute.final.wall_clock_seconds
            )
        ):
            raise FinalEvaluationError("final compute usage limits differ from task contract")
    if result.output_dir:
        results_dir = join(result.output_dir, "results")
        entry = await workspace.stat(results_dir)
        if (
            entry is not None
            and entry.kind == "dir"
            and await workspace.digest(results_dir) != result.results_tree_sha256
        ):
            raise FinalEvaluationError("final results digest mismatch")
    return verified


__all__ = [
    "DEFAULT_FINAL_EVALUATOR",
    "Incumbent",
    "evaluator_digest",
    "final_evaluator_path",
    "freeze_incumbent",
    "manifest_from_result",
    "resolve_incumbent",
    "task_digest",
    "verify_final_result",
]
