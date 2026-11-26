"""The drafting gates: contract preflight, smoke, production entry and qualification.

:class:`LocalTaskGate` is the default :class:`~langgraph_agentic_scientist.TaskGate`. It is a
rewrite of upstream's gate logic (``task_drafting/gate.py`` and ``checks.py``, and the shell
function ``validate_staged_contract`` of ``task.sh``) over a ``Workspace`` and an ``Evaluator``:
the decisions are upstream's, the container, broker and socket mechanics are not ported (they are
the caller's executor's, see ``docs/adaptations/drafting_scaffold.md``).

Checks (a :data:`~langgraph_agentic_scientist.core.interfaces.GateCheck`):

``preflight``
    the host's contract validation: staged paths safe, ``task.md`` unchanged, declared digests
    refreshed, the contract loadable, the contract lint and the RPC-transport lint clean;
``contract``
    the same inside the gate, plus the candidate entrypoint and the harness-capability lint;
``runtime``, ``search-smoke``, ``final-smoke``
    the search or final evaluator on the baseline with ``AGENTIC_TASK_DRAFT_SMOKE=1`` (reduced
    workload, evaluation plane ``smoke``); the results must hold exactly the declared metrics and
    a boolean ``correct``;
``production-entry``
    both evaluators in production topology, stopped after the first completed declared reset (plane
    ``production_entry``, ``AGENTIC_TASK_PRODUCTION_ENTRY=1``); it needs an isolated evaluator, as
    upstream's needs containers, and is an infrastructure failure otherwise;
``acceptance-smoke``
    the resource tree, ``contract``, both smokes and (when the evaluator is isolated)
    ``production-entry``, in that order, stopping at the first failure;
``qualification``
    the full production search workload (plane ``qualification``,
    ``AGENTIC_TASK_QUALIFICATION=1``): the baseline must complete and its ``qualification.json``
    must report ``feasible``.

Every report states whether the scaffold passed, whether a failure was the harness's
(``infrastructure``) and its typed category, stage and component, with the scaffold fingerprint the
check ran against. An evaluator that modifies the scaffold fails the check.

A caller whose harness differs (a container runtime, a broker, a hosted service) substitutes its own
:class:`~langgraph_agentic_scientist.TaskGate`; an isolated evaluator that serves the
``production_entry`` plane returns ``completed`` after the first declared reset.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import unicodedata
from typing import Any, Literal

from langgraph.errors import GraphBubbleUp

from ..config import DraftingConfig
from ..core.contract import (
    TaskContract,
    load_task_contract,
    task_scaffold_fingerprint,
)
from ..core.errors import ProgramSpecError, TaskContractError
from ..core.interfaces import GateCheck, GateReport, TaskGate
from ..core.state import EvaluationPlane, ProgramRef, TaskRef
from ..evaluation.program_spec import resolve_entrypoint
from ..evaluation.protocol import EvaluationRequest, EvaluationResult, Evaluator
from ..workspace.digest import sha256_hex
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .contract_lint import (
    ContractSyncError,
    isolated_rpc_transport_findings,
    preflight_findings,
    synchronize_declared_digests,
    unavailable_harness_environment_findings,
)
from .history import has_transport_credential, redact_sensitive_text
from .models import RepairTransaction
from .scaffold import (
    DEPENDENCY_AUTHORIZED,
    DraftPaths,
    PublicationError,
    RepairScopeError,
    ScaffoldError,
    authorize_dependency_removal,
    safe_tree_fingerprint,
    tree_fingerprint,
    validate_staged_paths,
    validate_staged_resources,
)

ALLOWED_GATES = frozenset(
    {
        "contract",
        "runtime",
        "search-smoke",
        "final-smoke",
        "production-entry",
        "acceptance-smoke",
    }
)
"""The operations of upstream's gate service (``task_drafting/gate.py`` ``ALLOWED_GATES``)."""

RUNTIME_REPAIRABLE_CATEGORIES = frozenset({"runtime_dependencies", "runtime_baseline_validation"})
"""Failure categories by which an evaluator that prepares its own runtime reports an authored
fault of the scaffold's dependencies or baseline (upstream ``checks._prepare_runtime``); a
``runtime_dependencies`` failure is the one that can authorise removing an unused pin."""

ZERO_FINGERPRINT = "sha256:" + "0" * 64
_EXIT_CODES = {"pass": 0, "authored_failure": 3, "infrastructure_failure": 4}
_SMOKE_ENVIRONMENT = {"AGENTIC_TASK_DRAFT_SMOKE": "1"}
_ACCEPTANCE_MAX_SECONDS = 7170


# --- sanitisation (upstream task_drafting/gate.py and task.sh) ----------------------------------


def sanitize_diagnostic(value: str, *, max_bytes: int = 16 * 1024) -> str:
    """Make a gate diagnostic safe to show an agent (upstream ``sanitize_diagnostic``).

    Control characters other than newline and tab become ``\\xNN`` escapes; a diagnostic that
    carries a transport credential or any recognisable secret is replaced by a digest marker;
    the result is cut to ``max_bytes`` with an ellipsis. Because the redaction escapes every
    non-printable character, the result is a single line.
    """
    normalized = "".join(
        character
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
        else f"\\x{ord(character):02x}"
        for character in value
    )
    if has_transport_credential(normalized):
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]
        safe = f"<redacted-sensitive-text:{digest}>"
    else:
        safe = redact_sensitive_text(normalized)
    encoded = safe.encode("utf-8")
    if len(encoded) <= max_bytes:
        return safe
    marker = "…".encode()
    return encoded[: max_bytes - len(marker)].decode("utf-8", errors="ignore") + "…"


def sanitize_preflight_diagnostic(raw: str) -> str:
    """Bound and sanitise the output of the contract preflight (``sanitize_preflight_diagnostic``).

    The last 32 KiB are kept, control characters other than newline and tab are escaped, and the
    text is redacted.
    """
    data = raw.encode("utf-8")[-32768:]
    text = data.decode("utf-8", errors="replace")
    safe = "".join(
        character
        if character in "\n\t" or not unicodedata.category(character).startswith("C")
        else f"\\x{ord(character):02x}"
        for character in text
    )
    return redact_sensitive_text(safe)


def typed_smoke_diagnostic(report: GateReport, *, scaffold_fingerprint: str | None = None) -> str:
    """Render the typed diagnostic a smoke repair prompt carries (``typed_smoke_diagnostic``).

    The JSON object (sorted keys, one line) names the gate, the category, the component and
    stage, the scaffold fingerprint and the exit code. For an authored failure the text of the
    diagnostic is a fixed sentence: the task-authored output is shown only through the in-session
    gate, never inlined.
    """
    status = (
        "pass"
        if report.passed
        else "infrastructure_failure"
        if report.infrastructure
        else "authored_failure"
    )
    component = str(report.details.get("component", "harness"))
    raw = report.diagnostic.encode("utf-8")
    if status == "pass":
        diagnostic = sanitize_diagnostic(report.diagnostic)
    elif status == "infrastructure_failure":
        diagnostic = sanitize_diagnostic(
            report.diagnostic
            or "Task infrastructure failed; inspect the named component and stage."
        )
    else:
        diagnostic = sanitize_diagnostic(
            f"{report.check} failed with {report.category}; inspect the named component and stage."
        )
    return json.dumps(
        {
            "schema_version": 1,
            "status": status,
            "gate": report.check,
            "category": report.category or "",
            "component": component,
            "stage": report.stage or "",
            "scaffold_fingerprint": scaffold_fingerprint or report.fingerprint or ZERO_FINGERPRINT,
            "exit_code": _EXIT_CODES[status],
            "output_bytes": len(raw),
            "output_sha256": hashlib.sha256(raw).hexdigest(),
            "diagnostic": diagnostic,
            "message": (
                "deterministic search and final evaluator smoke checks passed"
                if status == "pass"
                else "task evaluator smoke failed"
            ),
        },
        sort_keys=True,
    )


# --- reports -------------------------------------------------------------------------------


def make_report(
    check: str,
    status: str,
    category: str,
    component: str,
    stage: str,
    fingerprint: str,
    diagnostic: str = "",
    elapsed_seconds: float = 0.0,
    **details: Any,
) -> GateReport:
    """Build a :class:`GateReport` the way upstream's typed report is built.

    ``status`` is ``pass``, ``authored_failure`` or ``infrastructure_failure``; ``component`` and
    ``elapsed_seconds`` are kept in ``details``.
    """
    return GateReport(
        check=check,
        passed=status == "pass",
        infrastructure=status == "infrastructure_failure",
        category=category,
        stage=stage,
        diagnostic=sanitize_diagnostic(diagnostic),
        fingerprint=fingerprint,
        details={
            "component": component,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "status": status,
            **details,
        },
    )


class _Failure(Exception):
    def __init__(
        self, status: str, category: str, component: str, stage: str, diagnostic: str = ""
    ) -> None:
        super().__init__(diagnostic or category)
        self.status = status
        self.category = category
        self.component = component
        self.stage = stage
        self.diagnostic = diagnostic


def _authored(category: str, component: str, stage: str, diagnostic: str = "") -> _Failure:
    return _Failure("authored_failure", category, component, stage, diagnostic)


def _infrastructure(category: str, component: str, stage: str, diagnostic: str = "") -> _Failure:
    return _Failure("infrastructure_failure", category, component, stage, diagnostic)


# --- the local gate ---------------------------------------------------------------------------


class LocalTaskGate:
    """The default drafting gate: evaluators through an ``Evaluator``, lints through the workspace.

    Args:
        evaluator: Runs the evaluators on the baseline; required by every check except
            ``preflight`` and ``contract``.
        config: The drafting limits (smoke, production-entry and qualification timeouts, the
            resource byte limit); upstream defaults when omitted.
    """

    def __init__(
        self, evaluator: Evaluator | None = None, *, config: DraftingConfig | None = None
    ) -> None:
        self.evaluator = evaluator
        self.config = config or DraftingConfig()

    async def run_check(
        self,
        check: GateCheck,
        *,
        task: TaskRef,
        workspace: Workspace,
        evaluation_id: str,
    ) -> GateReport:
        """Run one check of the scaffold at ``task.root``.

        ``task.root`` is the staged task directory and ``task.task_md_sha256`` the digest of the
        canonical ``task.md`` that the staged copy must equal. ``evaluation_id`` prefixes the ids
        of the evaluations the check requests (``<id>/search``, ``<id>/final``, ...), so that a
        resumed check re-attaches to a running evaluation instead of starting another.
        """
        started = time.monotonic()
        try:
            if check == "preflight":
                return await self._preflight(task, workspace, started)
            if check == "acceptance-smoke":
                return await self._acceptance(task, workspace, evaluation_id, started)
            if check == "qualification":
                return await self._qualification(task, workspace, evaluation_id, started)
            if check not in ALLOWED_GATES:
                return make_report(
                    str(check),
                    "infrastructure_failure",
                    "unknown_check",
                    "harness",
                    str(check),
                    ZERO_FINGERPRINT,
                    f"unknown check {check!r}",
                )
            return await self._gate(check, task, workspace, evaluation_id, started)
        except GraphBubbleUp:
            raise
        except Exception:
            return make_report(
                str(check),
                "infrastructure_failure",
                "unexpected_harness_failure",
                "harness",
                str(check),
                await safe_tree_fingerprint(workspace, task.root),
                "Harness gate execution failed.",
                time.monotonic() - started,
            )

    # --- preflight ---------------------------------------------------------------------

    async def _preflight(self, task: TaskRef, workspace: Workspace, started: float) -> GateReport:
        """Upstream ``validate_staged_contract``; the diagnostic is the sanitised preflight text."""
        root = task.root

        def fail(category: str, message: str) -> GateReport:
            report = make_report(
                "preflight",
                "authored_failure",
                category,
                "contract",
                "contract",
                ZERO_FINGERPRINT,
                "",
                time.monotonic() - started,
            )
            return report.model_copy(
                update={"diagnostic": sanitize_preflight_diagnostic(message + "\n")}
            )

        try:
            await validate_staged_paths(workspace, root, canonical_task_sha256=task.task_md_sha256)
        except ScaffoldError as exc:
            return fail("scaffold_unsafe", str(exc))
        try:
            await synchronize_declared_digests(workspace, root)
            await load_task_contract(workspace, root)
            fingerprint = await task_scaffold_fingerprint(workspace, root)
        except (ContractSyncError, TaskContractError) as exc:
            findings = (
                *await preflight_findings(workspace, root),
                *await isolated_rpc_transport_findings(workspace, root),
            )
            extra = ("\nContract findings:\n- " + "\n- ".join(findings)) if findings else ""
            return fail("contract_invalid", f"Invalid drafted task contract: {exc}{extra}")
        findings = (
            *await preflight_findings(workspace, root),
            *await isolated_rpc_transport_findings(workspace, root),
        )
        if findings:
            return fail(
                "contract_preflight_invalid",
                "Invalid drafted task contract:\nContract findings:\n- " + "\n- ".join(findings),
            )
        return make_report(
            "preflight",
            "pass",
            "gate_passed",
            "contract",
            "contract",
            fingerprint,
            "",
            time.monotonic() - started,
        )

    # --- acceptance --------------------------------------------------------------------

    async def _acceptance(
        self, task: TaskRef, workspace: Workspace, evaluation_id: str, started: float
    ) -> GateReport:
        root = task.root
        try:
            await validate_staged_resources(
                workspace, root, max_runtime_bytes=self.config.max_runtime_bytes
            )
        except PublicationError as exc:
            diagnostic = sanitize_diagnostic(str(exc))
            category = (
                "runtime_resources_exceed_limit"
                if diagnostic.startswith("runtime_resources_exceed_limit:")
                else "runtime_resources_invalid"
            )
            return make_report(
                "runtime",
                "authored_failure",
                category,
                "runtime",
                "resource-validation",
                await safe_tree_fingerprint(workspace, root),
                diagnostic,
                time.monotonic() - started,
            )
        operations = ["contract", "search-smoke", "final-smoke"]
        if self.evaluator is not None and self.evaluator.isolated:
            operations.append("production-entry")
        for operation in operations:
            report = await self._gate(
                operation, task, workspace, f"{evaluation_id}/{operation}", time.monotonic()
            )
            if not report.passed:
                return report
        return make_report(
            "acceptance-smoke",
            "pass",
            "gate_passed",
            "harness",
            "acceptance-smoke",
            await tree_fingerprint(workspace, root),
            "",
            time.monotonic() - started,
        )

    # --- one gate (upstream _run_gate) ---------------------------------------------------

    async def _gate(
        self,
        operation: str,
        task: TaskRef,
        workspace: Workspace,
        evaluation_id: str,
        started: float,
    ) -> GateReport:
        root = task.root
        diagnostic = ""
        try:
            if operation == "contract":
                staged = sha256_hex(await workspace.read_bytes(join(root, "task.md")))
                if task.task_md_sha256 is not None and staged != task.task_md_sha256:
                    raise _authored(
                        "canonical_task_modified",
                        "contract",
                        "contract",
                        "task.md must match the canonical source task exactly",
                    )
            await synchronize_declared_digests(workspace, root)
            before = await tree_fingerprint(workspace, root)
            contract = await load_task_contract(workspace, root)
            if operation == "contract":
                findings = await preflight_findings(workspace, root)
                if findings:
                    raise _authored(
                        "contract_preflight_invalid", "contract", "contract", "\n".join(findings)
                    )
            try:
                entrypoint = await resolve_entrypoint(workspace, join(root, "program_0"))
            except ProgramSpecError:
                raise _authored(
                    "candidate_entrypoint_invalid",
                    "candidate",
                    "contract",
                    "program_0/program_spec.toml must declare an existing regular entrypoint "
                    "inside program_0 (default: program_0/src/program.py)",
                ) from None
            transport = await isolated_rpc_transport_findings(workspace, root)
            if transport:
                raise _authored(
                    "rpc_transport_invalid", "candidate", "contract", "\n".join(transport)
                )
            capability = await unavailable_harness_environment_findings(
                workspace,
                root,
                (
                    ("search_evaluator", contract.evaluators.search),
                    ("final_evaluator", contract.evaluators.final),
                ),
            )
            if capability:
                raise _authored(
                    "harness_capability_invalid",
                    capability[0].component,
                    "contract",
                    "\n".join(item.diagnostic for item in capability),
                )
            if operation in {"runtime", "search-smoke"}:
                await self._smoke("search", task, contract, entrypoint, evaluation_id)
            elif operation == "final-smoke":
                await self._smoke("final", task, contract, entrypoint, evaluation_id)
            elif operation == "production-entry":
                diagnostic = await self._production_entry(task, entrypoint, evaluation_id)
            after = await tree_fingerprint(workspace, root)
            if before != after:
                raise _authored("evaluator_modified_scaffold", "harness", operation)
        except _Failure as exc:
            return make_report(
                operation,
                exc.status,
                exc.category,
                exc.component,
                exc.stage,
                await safe_tree_fingerprint(workspace, root),
                exc.diagnostic,
                time.monotonic() - started,
            )
        except (ContractSyncError, TaskContractError) as exc:
            return make_report(
                operation,
                "authored_failure",
                "contract_invalid",
                "contract",
                "contract",
                await safe_tree_fingerprint(workspace, root),
                str(exc),
                time.monotonic() - started,
            )
        return make_report(
            operation,
            "pass",
            "gate_passed",
            "harness",
            operation,
            await tree_fingerprint(workspace, root),
            diagnostic,
            time.monotonic() - started,
        )

    def _request(
        self,
        evaluation_id: str,
        plane: EvaluationPlane,
        evaluator: Literal["search", "final"],
        task: TaskRef,
        entrypoint: str,
        environment: dict[str, str],
        timeout: float,
    ) -> EvaluationRequest:
        return EvaluationRequest(
            evaluation_id=evaluation_id,
            plane=plane,
            program=ProgramRef(path=join(task.root, "program_0"), entrypoint=entrypoint),
            task=task,
            evaluator=evaluator,
            num_replications=1,
            timeout_seconds=float(timeout),
            environment=environment,
        )

    async def _evaluate(self, request: EvaluationRequest, evaluator_name: str) -> EvaluationResult:
        if self.evaluator is None:
            raise _infrastructure(
                f"{evaluator_name}_no_evaluator",
                f"{evaluator_name}_evaluator",
                "execution",
                "no evaluator is configured for this run",
            )
        return await self.evaluator.evaluate(request)

    async def _smoke(
        self,
        plane: Literal["search", "final"],
        task: TaskRef,
        contract: TaskContract,
        entrypoint: str,
        evaluation_id: str,
    ) -> None:
        """Run one evaluator in the reduced smoke profile and check its results."""
        request = self._request(
            f"{evaluation_id}/{plane}",
            "smoke",
            plane,
            task,
            entrypoint,
            {**_SMOKE_ENVIRONMENT, "TASK_EVALUATION_PLANE": plane},
            self.config.evaluator_smoke_timeout_seconds,
        )
        result = await self._evaluate(request, plane)
        component = f"{plane}_evaluator"
        if result.status == "timeout":
            raise _authored(f"{plane}_timeout", component, "execution")
        if result.status in {"infrastructure_error", "interrupted"}:
            category = result.failure_category or (
                "evaluation_interrupted"
                if result.status == "interrupted"
                else f"{plane}_process_runtime"
            )
            raise _infrastructure(category, component, "execution", result.summary)
        if result.status == "failed":
            if result.failure_category in RUNTIME_REPAIRABLE_CATEGORIES:
                raise _authored(result.failure_category, "runtime", "bootstrap", result.summary)
            if result.failure_category == "output_limit_exceeded":
                raise _authored(f"{plane}_excessive_output", component, "execution")
            if result.failure_category == "evaluator_output_contract_invalid":
                raise _authored(f"{plane}_invalid_results", component, "results")
            raise _authored(
                f"{plane}_nonzero_exit",
                component,
                "execution",
                sanitize_diagnostic(result.summary if plane == "search" else ""),
            )
        self._require_score(plane, contract, result)

    @staticmethod
    def _require_score(plane: str, contract: TaskContract, result: EvaluationResult) -> None:
        """Upstream ``_require_score``: exactly the declared metrics, finite, and a boolean."""
        declared = {
            metric.name
            for metric in (contract.metrics.search if plane == "search" else contract.metrics.final)
        }
        metrics = result.metrics
        invalid = (
            set(metrics) != declared
            or not isinstance(result.correct, bool)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
        )
        if not invalid and plane == "search":
            score = metrics.get("combined_score")
            invalid = (
                isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(float(score))
            )
        if invalid:
            raise _authored(f"{plane}_invalid_results", f"{plane}_evaluator", "results")

    async def _production_entry(self, task: TaskRef, entrypoint: str, evaluation_id: str) -> str:
        if self.evaluator is None or not self.evaluator.isolated:
            raise _infrastructure(
                "production_entry_requires_container", "harness", "production-entry"
            )
        entries: list[str] = []
        for plane in ("search", "final"):
            request = self._request(
                f"{evaluation_id}/{plane}",
                "production_entry",
                plane,
                task,
                entrypoint,
                {"AGENTIC_TASK_PRODUCTION_ENTRY": "1", "TASK_EVALUATION_PLANE": plane},
                self.config.production_entry_timeout_seconds,
            )
            result = await self.evaluator.evaluate(request)
            component = f"{plane}_evaluator"
            if result.status == "timeout":
                raise _authored(f"{plane}_production_timeout", component, "production-entry")
            if result.status in {"infrastructure_error", "interrupted"}:
                raise _infrastructure(
                    result.failure_category or f"{plane}_production_infrastructure",
                    component,
                    "production-entry",
                    result.summary,
                )
            if result.status == "failed":
                raise _authored(
                    result.failure_category or f"{plane}_production_nonzero_exit",
                    component,
                    "production-entry",
                    result.summary,
                )
            entries.append(f"{plane}_entry_status={result.status}")
        return " ".join(entries)

    # --- qualification -----------------------------------------------------------------

    async def _qualification(
        self, task: TaskRef, workspace: Workspace, evaluation_id: str, started: float
    ) -> GateReport:
        root = task.root
        fingerprint = await safe_tree_fingerprint(workspace, root)
        stage = "production-search"
        try:
            contract = await load_task_contract(workspace, root)
            entrypoint = await resolve_entrypoint(workspace, join(root, "program_0"))
        except (TaskContractError, ProgramSpecError) as exc:
            return make_report(
                "qualification",
                "authored_failure",
                "qualification_publication_plan",
                "contract",
                "publication-plan",
                fingerprint,
                str(exc),
                time.monotonic() - started,
            )
        request = self._request(
            f"{evaluation_id}/qualification",
            "qualification",
            "search",
            task,
            entrypoint,
            {"AGENTIC_TASK_QUALIFICATION": "1", "TASK_EVALUATION_PLANE": "search"},
            self.config.qualification_timeout_seconds,
        )
        try:
            result = await self._evaluate(request, "qualification")
        except _Failure as exc:
            return make_report(
                "qualification",
                exc.status,
                exc.category,
                exc.component,
                stage,
                fingerprint,
                exc.diagnostic,
                time.monotonic() - started,
            )
        elapsed = (
            result.duration_seconds
            if result.duration_seconds is not None
            else (time.monotonic() - started)
        )
        if result.status == "completed":
            try:
                self._require_score("search", contract, result)
            except _Failure as exc:
                return make_report(
                    "qualification",
                    exc.status,
                    "qualification_invalid_results",
                    exc.component,
                    stage,
                    fingerprint,
                    "",
                    elapsed,
                )
            if result.feasible is not True:
                return make_report(
                    "qualification",
                    "authored_failure",
                    "qualification_infeasible",
                    "search_evaluator",
                    stage,
                    fingerprint,
                    result.summary or "the baseline did not report a feasible production run",
                    elapsed,
                )
            return make_report(
                "qualification",
                "pass",
                "qualification_passed",
                "search_evaluator",
                stage,
                fingerprint,
                "",
                elapsed,
                baseline_combined_score=float(result.metrics["combined_score"]),
                baseline_correct=result.correct,
                baseline_feasible=True,
                evaluation_id=request.evaluation_id,
            )
        if result.status == "timeout":
            return make_report(
                "qualification",
                "authored_failure",
                "qualification_timeout",
                "search_evaluator",
                stage,
                fingerprint,
                result.summary,
                elapsed,
            )
        category = result.failure_category or (
            "evaluation_interrupted"
            if result.status == "interrupted"
            else "qualification_runtime_failure"
        )
        status = "authored_failure" if result.status == "failed" else "infrastructure_failure"
        return make_report(
            "qualification",
            status,
            category,
            "search_evaluator",
            stage,
            fingerprint,
            result.summary,
            elapsed,
        )


# --- the in-session gate of a repair ----------------------------------------------------------


class RepairSessionGate:
    """Wraps a gate for the ``task_gate`` tool of one smoke repair session.

    When the wrapped gate reports an authored ``runtime_dependencies`` failure, the repair is
    still inside the scope it opened with, ``requirements.txt`` is unchanged and the failing
    fingerprint is current, the harness authorises the removal of unused direct pins: it records
    the authorisation in a host-only file (outside the agent's root) and prefixes the diagnostic
    with :data:`~langgraph_agentic_scientist.drafting.scaffold.DEPENDENCY_AUTHORIZED`. The
    graph reads that file after the session, so neither the gate's output nor the agent's own
    text is an authorisation record (upstream ``DependencyRepairExecutor``).

    Args:
        inner: The gate that runs the checks.
        paths: The drafting locations.
        transaction: The scoped transaction of the repair.
        label: The repair's label.
    """

    def __init__(
        self,
        inner: TaskGate,
        *,
        paths: DraftPaths,
        transaction: RepairTransaction | None,
        label: str,
    ) -> None:
        self.inner = inner
        self.paths = paths
        self.transaction = transaction
        self.label = label

    async def run_check(
        self, check: GateCheck, *, task: TaskRef, workspace: Workspace, evaluation_id: str
    ) -> GateReport:
        """Run the check; add the dependency authorisation when its conditions hold."""
        report = await self.inner.run_check(
            check, task=task, workspace=workspace, evaluation_id=evaluation_id
        )
        if (
            report.passed
            or report.infrastructure
            or report.category != "runtime_dependencies"
            or self.transaction is None
        ):
            return report
        authorization = self.paths.authorization(self.label)
        try:
            if not await workspace.exists(authorization):
                if self.transaction.contract_policy == "synchronized-dependency-removals":
                    return report
                await authorize_dependency_removal(
                    workspace,
                    self.paths,
                    self.transaction,
                    report_fingerprint=report.fingerprint or "",
                )
                await workspace.write_bytes(
                    authorization,
                    (
                        json.dumps(
                            {
                                "label": self.label,
                                "category": "runtime_dependencies",
                                "scaffold_fingerprint": report.fingerprint,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode(),
                )
        except (RepairScopeError, OSError, ValueError) as exc:
            return make_report(
                str(report.check),
                "infrastructure_failure",
                "repair_authorization_failed",
                "harness",
                "repair_authorization",
                report.fingerprint or ZERO_FINGERPRINT,
                str(exc),
            )
        return report.model_copy(
            update={
                "diagnostic": DEPENDENCY_AUTHORIZED + "\n" + sanitize_diagnostic(report.diagnostic)
            }
        )


def acceptance_budget_seconds(config: DraftingConfig) -> int:
    """Return the shared deadline of one acceptance check (upstream ``smoke_required_budget``)."""
    return min(
        _ACCEPTANCE_MAX_SECONDS,
        2 * config.evaluator_smoke_timeout_seconds + config.production_entry_timeout_seconds,
    )


__all__ = [
    "ALLOWED_GATES",
    "ZERO_FINGERPRINT",
    "LocalTaskGate",
    "RepairSessionGate",
    "acceptance_budget_seconds",
    "make_report",
    "sanitize_diagnostic",
    "sanitize_preflight_diagnostic",
    "typed_smoke_diagnostic",
]
