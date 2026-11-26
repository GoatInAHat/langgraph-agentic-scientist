"""The scaffold drafting loop as a LangGraph graph (upstream phases A2-A7).

After the task has a design (or none, in the legacy workflow), upstream's ``task.sh`` drafts the
scaffold, validates it and repairs it until an independent judge and a full production run both
accept the same files, then publishes. Its control flow is nested shell functions that return
exit codes. In the port every step is a graph node, every loop has its counter in state, and
every agent session is one node execution (design decisions D1 and D6), so a crash or an
interrupt resumes without repeating a completed session, gate or evaluation.

The method, in the order the graph runs it, with upstream's limits (``DraftingConfig``):

``prepare``
    stage a copy of the task snapshot and the accepted design in the drafting directory;
``draft`` (R7, skills ``draft-task`` and ``memory-log``) and ``inspect_scaffold``
    the drafter creates the missing files; an incomplete scaffold is retried with the harness
    feedback, up to ``draft_max_attempts`` (3) sessions; an unsafe file ends drafting;
``canonicalize``
    ``capture.py``, ``program_0/program_spec.toml`` and the candidate entrypoint are restored from
    the canonical task;
``preflight`` and ``preflight_repair`` (R8)
    the contract gate; a failure is repaired up to ``contract_max_repairs`` (3) times;
``smoke`` and ``smoke_repair`` (R9)
    the reduced search and final smoke and, with an isolated evaluator, the production entry;
    infrastructure failures retry up to ``smoke_max_infrastructure_attempts`` (3) without
    spending a repair; authored failures are repaired up to ``smoke_max_repairs`` (3) times
    inside a scoped transaction that is committed only when every smoke check passes and rolled
    back on an unsafe edit, a terminal failure or an exhausted budget;
``alignment_view``, ``alignment_judge`` (R10) and ``check_alignment``
    an independent judge reviews a read-only view of the scaffold against the canonical task on
    eleven dimensions; a malformed report is retried once; at most ``2 * (alignment_max_repairs +
    qualification_max_repairs + 1)`` judge calls are made;
``alignment_repair`` (R11)
    a failed report is repaired up to ``alignment_max_repairs`` (10) times inside a tree
    transaction that stays provisional until the judge accepts;
``qualify``, ``qualification_repair`` (R12) and the qualification transaction
    the full production workload must complete feasibly; a failure is repaired up to
    ``qualification_max_repairs`` (4) times as a *pending* transaction with a failure history, an
    oscillation check and a calibration record that the judge sees; the transaction is committed
    only when alignment and qualification accept the same scaffold, and rolled back otherwise;
``publish``
    the accepted scaffold, the alignment report and the qualification receipt are published into
    the task directory, after a credential scan.

The graph reads every limit through
:func:`~langgraph_agentic_scientist.core.runconfig.method_config`
and keeps all of its records in typed state. Because a run takes many supersteps, a caller sets a
high ``recursion_limit`` in the run configuration.
"""

from __future__ import annotations

import base64
import json
import operator
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..agents.runner import (
    RoleOutcome,
    agent_node_retry_policy,
    run_structured_role,
    run_workspace_role,
)
from ..agents.staging import skills_for_step
from ..agents.tools import make_nl_tool, make_report_blocker_tool, make_task_gate_tool
from ..config import DraftingConfig
from ..context import GraphOverrides, ScientistContext
from ..core.contract import load_task_contract, task_scaffold_fingerprint
from ..core.errors import TaskContractError
from ..core.interfaces import GateReport, TaskGate
from ..core.render import json_text, truncate_utf8
from ..core.results import DraftingResult
from ..core.runconfig import method_config, record_run_config
from ..core.state import (
    Failure,
    FailureScope,
    ProgramRef,
    RunConfigRecord,
    TaskRef,
    merge_dicts,
)
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.digest import sha256_hex
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .design_models import Capacity
from .gate import (
    LocalTaskGate,
    RepairSessionGate,
    sanitize_preflight_diagnostic,
    typed_smoke_diagnostic,
)
from .history import (
    capture_alignment_report,
    capture_rejected_repair,
    collect_sources,
    contains_secret,
    is_sensitive_path,
    redact_sensitive_text,
    render_alignment_history,
    render_repair_history,
)
from .models import (
    AlignmentHistoryEntry,
    AlignmentReport,
    DependencyTransaction,
    PendingQualificationRepair,
    QualificationCalibration,
    QualificationFailure,
    QualificationReceipt,
    QualificationRecord,
    RejectedRepair,
    RepairTransaction,
)
from .qualification import (
    ALIGNMENT_FAILURE_CATEGORY,
    POST_VALIDATION_CATEGORY,
    TERMINAL_GPU_CATEGORIES,
    QualificationError,
    accept_calibration,
    alignment_context,
    create_calibration,
    qualification_failure_projection,
    qualification_success_context,
    record_failure,
    record_from_gate,
    reject_repeated_candidate,
    render_qualification_repair_prompt,
    report_text,
    sha256_text,
)
from .scaffold import (
    AlignmentReportError,
    DraftPaths,
    PublicationError,
    RepairScopeError,
    ScaffoldError,
    alignment_fingerprints,
    bounded_repair_report,
    build_publication_plan,
    candidate_entrypoint,
    canonicalize_staged_scaffold,
    contract_policy_for_smoke_diagnostic,
    draft_paths,
    draft_values,
    evaluator_tree_sha256,
    inspect_scaffold,
    open_scoped_transaction,
    open_tree_transaction,
    prepare_judge_view,
    prompt_values,
    publish_scaffold,
    published_task_ref,
    restore_transaction,
    safe_tree_fingerprint,
    stage_task_copy,
    tree_fingerprint,
    validate_alignment_report,
    validate_repair_scope,
    validate_staged_paths,
    write_design_file,
)

PHASE = "drafting"

STEP_PREPARE = "prepare"
STEP_DRAFT = "draft"
STEP_INSPECT = "inspect_scaffold"
STEP_CANONICALIZE = "canonicalize"
STEP_PREFLIGHT = "preflight"
STEP_PREFLIGHT_REPAIR = "preflight_repair"
STEP_SMOKE = "smoke"
STEP_SMOKE_REPAIR = "smoke_repair"
STEP_ALIGNMENT_VIEW = "alignment_view"
STEP_ALIGNMENT_JUDGE = "alignment_judge"
STEP_CHECK_ALIGNMENT = "check_alignment"
STEP_ALIGNMENT_REPAIR = "alignment_repair"
STEP_QUALIFY = "qualify"
STEP_QUALIFICATION_REPAIR = "qualification_repair"
STEP_CHECK_CANDIDATE = "check_qualification_candidate"
STEP_COMMIT_QUALIFICATION = "commit_qualification_repair"
STEP_PUBLISH = "publish"
SCAFFOLD_STEPS: tuple[str, ...] = (
    STEP_PREPARE,
    STEP_DRAFT,
    STEP_INSPECT,
    STEP_CANONICALIZE,
    STEP_PREFLIGHT,
    STEP_PREFLIGHT_REPAIR,
    STEP_SMOKE,
    STEP_SMOKE_REPAIR,
    STEP_ALIGNMENT_VIEW,
    STEP_ALIGNMENT_JUDGE,
    STEP_CHECK_ALIGNMENT,
    STEP_ALIGNMENT_REPAIR,
    STEP_QUALIFY,
    STEP_QUALIFICATION_REPAIR,
    STEP_CHECK_CANDIDATE,
    STEP_COMMIT_QUALIFICATION,
    STEP_PUBLISH,
)
"""The steps that carry hook nodes, under the keys ``drafting.<step>``."""

SMOKE_RETRYABLE_CATEGORIES = frozenset(
    {"runtime_bootstrap", "gpu_allocation_busy", "evaluation_interrupted"}
)
"""Infrastructure categories of a smoke check that are retried (upstream: the first two)."""

ACCEPTANCE = "acceptance-smoke"
_GATE_ID_FORMAT = "{prefix}/gate_{seq:03d}_{check}"


# --- state -------------------------------------------------------------------------------------


class ScaffoldInput(TypedDict):
    """Input of the scaffold graph.

    Attributes:
        task: The run-local task snapshot (``task.md`` and, optionally, a canonical ``capture.py``
            and ``program_0``).
        accepted_design: The accepted TaskDesign v1 as JSON, or ``None`` (legacy workflow).
        accepted_review: The review that accepted the design, as JSON.
        run_config: The recorded configuration of a parent run, when mounted.
    """

    task: TaskRef
    accepted_design: NotRequired[dict[str, Any] | None]
    accepted_review: NotRequired[dict[str, Any] | None]
    run_config: NotRequired[RunConfigRecord | None]
    capacity: NotRequired[Capacity]


class ScaffoldOutput(TypedDict):
    """Output of the scaffold graph: the drafting result and the hook outputs."""

    drafting: DraftingResult
    hook_outputs: dict[str, Any]


class ScaffoldState(TypedDict, total=False):
    """State of the scaffold graph (see the module docstring for the steps that use it)."""

    task: TaskRef
    capacity: Capacity
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    accepted_design: dict[str, Any] | None
    accepted_review: dict[str, Any] | None
    # layout
    staged: TaskRef
    entrypoint: str
    prefix: str
    # drafting
    draft_attempts: int
    missing_outputs: list[str]
    retry_feedback: str | None
    inspection: str
    # budgets
    repairs_used: int
    contract_repairs: int
    smoke_repairs: int
    smoke_infra_attempts: int
    gate_checks: int
    alignment_repairs: int
    judge_calls: int
    report_retries: int
    qualification_repairs: int
    qualification_infra_attempts: int
    qualification_runs: int
    # control
    phase: str
    canon_error: dict[str, Any] | None
    gate: GateReport | None
    gate_seq: int
    smoke_op: str
    smoke_followup: str | None
    smoke_continue: bool
    last_contract_diagnostic: str
    last_smoke_diagnostic: str
    last_alignment_diagnostic: str
    smoke_gate: str
    smoke_prompt_active: str
    # transactions
    transaction: RepairTransaction | None
    dependency_transaction: DependencyTransaction | None
    pending_qualification: PendingQualificationRepair | None
    # alignment
    review_round: int
    fingerprints: dict[str, str]
    view_dir: str
    view_fingerprint: str
    staged_fingerprint: str
    judge_prompt: str
    judge_suffix: str
    judge_report: AlignmentReport | None
    judge_failure: Failure | None
    report_attempt: int
    alignment: AlignmentReport | None
    accepted_alignment: AlignmentReport | None
    alignment_prompt_base: str
    alignment_prompt: str
    alignment_history: Annotated[list[AlignmentHistoryEntry], operator.add]
    rejected_repairs: Annotated[list[RejectedRepair], operator.add]
    # qualification
    qualification: QualificationRecord | None
    accepted_qualification: QualificationRecord | None
    failed_qualification: QualificationRecord | None
    qualification_failures: Annotated[list[QualificationFailure], operator.add]
    calibration: QualificationCalibration | None
    qualification_alignment_report: str
    qualification_post_validation: str
    qualification_failure_stage: str
    # result
    blocker: str | None
    status: str
    failure: Failure | None
    drafting: DraftingResult


# --- helpers -----------------------------------------------------------------------------------


def _need(state: Mapping[str, Any], key: str) -> Any:
    """Return a state value that the graph's order guarantees to be set."""
    value = state.get(key)
    if value is None:
        raise KeyError(f"{key} is not set; the graph reached a step out of order")
    return value


def _cfg(state: Mapping[str, Any]) -> DraftingConfig:
    return method_config(state).drafting


def _paths(state: Mapping[str, Any]) -> DraftPaths:
    return draft_paths(state["task"].root)


def _failed(category: str, detail: str = "", *, scope: FailureScope = "task") -> dict[str, Any]:
    return {
        "status": "failed",
        "failure": Failure(
            category=category,
            scope=scope,
            detail=truncate_utf8(detail, 4096),
        ),
    }


def _is_failed(state: Mapping[str, Any]) -> bool:
    return state.get("status") == "failed"


def _limits(cfg: DraftingConfig) -> dict[str, int]:
    """The effective repair limits; the legacy workflow gives every pool the same number."""
    if cfg.workflow == "legacy":
        legacy = cfg.legacy_validation_max_repairs
        return {
            "contract": legacy,
            "smoke": legacy,
            "alignment": legacy,
            "qualification": cfg.qualification_max_repairs,
        }
    return {
        "contract": cfg.contract_max_repairs,
        "smoke": cfg.smoke_max_repairs,
        "alignment": cfg.alignment_max_repairs,
        "qualification": cfg.qualification_max_repairs,
    }


def _exhausted(state: Mapping[str, Any], kind: str) -> bool:
    """Upstream ``repair_budget_exhausted``: the scoped pool, or the one pool of the legacy flow."""
    cfg = _cfg(state)
    limit = _limits(cfg)[kind]
    if cfg.workflow == "legacy" and kind != "qualification":
        return int(state.get("repairs_used", 0)) >= limit
    return int(state.get(f"{kind}_repairs", 0)) >= limit


def _used(state: Mapping[str, Any], kind: str) -> int:
    if _cfg(state).workflow == "legacy" and kind != "qualification":
        return int(state.get("repairs_used", 0))
    return int(state.get(f"{kind}_repairs", 0))


def _isolated(ctx: ScientistContext) -> bool:
    return bool(ctx.evaluator is not None and ctx.evaluator.isolated)


def _task_gate(ctx: ScientistContext, cfg: DraftingConfig) -> TaskGate:
    return ctx.task_gate or LocalTaskGate(ctx.evaluator, config=cfg)


def _bounded_diagnostic(text: str) -> str:
    """Redact and bound a diagnostic to 2048 bytes with an ellipsis (upstream retry prompts)."""
    diagnostic = redact_sensitive_text(text).strip()
    encoded = diagnostic.encode("utf-8")
    if len(encoded) > 2048:
        diagnostic = encoded[:2048].decode("utf-8", errors="ignore") + "…"
    return diagnostic


def _b64(text: str) -> str:
    return base64.b64encode(text.encode("utf-8")).decode("ascii")


async def _read_text(workspace: Workspace, path: str) -> str | None:
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        return None
    return (await workspace.read_bytes(path)).decode("utf-8", errors="replace")


async def _design_text(workspace: Workspace, state: Mapping[str, Any]) -> str:
    """The accepted design JSON, or the canonical task text when the run has none."""
    paths = _paths(state)
    text = await _read_text(workspace, paths.design_file)
    if text is not None:
        return text
    return (await _read_text(workspace, join(state["task"].root, "task.md"))) or ""


async def _staged_task_md(workspace: Workspace, paths: DraftPaths) -> str | None:
    return await _read_text(workspace, join(paths.task, "task.md"))


def _repair_values(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Values of a repair prompt: ``load_instructions(template, Path("."), Path("."), extra)``."""
    return prompt_values(task_dir=".", results_dir=".", task_md=None, extra=extra)


def _with_gate_text(ctx: ScientistContext, prompt: str) -> str:
    """Append the harness-owned validation gate text (upstream ``task_gate_prompt``)."""
    return prompt.rstrip("\n") + ctx.prompts.text("workflow/drafting/task_gate").rstrip("\n")


def _after_text(prompt: str, text: str) -> str:
    """Append a retry section the way upstream's ``print(prompt.rstrip()); print(text)`` does."""
    return prompt.rstrip() + "\n" + text


def _tools(
    state: Mapping[str, Any],
    runtime: Runtime[ScientistContext],
    *,
    label: str,
    gate: TaskGate,
) -> list[BaseTool]:
    ctx = runtime.context
    cfg = _cfg(state)
    return [
        make_task_gate_tool(
            ctx.workspace,
            gate,
            state["staged"],
            id_prefix=f"{state['prefix']}/{label}",
            max_checks=cfg.max_gate_checks_per_repair,
            prompts=ctx.prompts,
        ),
        make_report_blocker_tool(prompts=ctx.prompts),
    ]


def _repair_failure(outcome: RoleOutcome) -> tuple[str, str] | None:
    """Return ``(category, detail)`` when a repair session failed or reported a blocker."""
    if outcome.blocker:
        return "repair_blocked", outcome.blocker
    if outcome.failure is not None:
        return outcome.failure.category, outcome.failure.detail
    if outcome.timed_out:
        return "repair_timed_out", "the repair session ran out of time"
    return None


async def _fingerprint(state: Mapping[str, Any], workspace: Workspace) -> str:
    return await tree_fingerprint(workspace, _paths(state).task)


# --- prepare, draft and inspect ----------------------------------------------------------------


async def prepare(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Stage the task copy and the accepted design (upstream ``ensure_staged_task_copy``).

    The task snapshot is copied into the drafting directory (honouring ``.agentic_task_ignore``),
    the accepted design is written as ``_task_design/task_design.json``, and the staged task
    reference (root, canonical ``task.md`` digest, baseline program) is recorded. A copy left by
    an interrupted earlier run is replaced, so the step is safe to run again.
    """
    ctx = runtime.context
    workspace = ctx.workspace
    task: TaskRef = state["task"]
    paths = _paths(state)
    canonical = await _read_text(workspace, join(task.root, "task.md"))
    if canonical is None:
        return _failed("task_missing", f"Missing task spec: {join(task.root, 'task.md')}")
    await stage_task_copy(workspace, task.root, paths)
    design = state.get("accepted_design")
    if design:
        await write_design_file(workspace, paths, design)
    entrypoint = await candidate_entrypoint(workspace, task.root)
    staged = task.model_copy(
        update={
            "root": paths.task,
            "task_md_sha256": sha256_hex(canonical.encode("utf-8")),
            "contract": None,
            "seed_program": ProgramRef(
                path=join(paths.task, "program_0"),
                entrypoint=entrypoint.removeprefix("program_0/"),
            ),
        }
    )
    thread = (runtime.execution_info.thread_id if runtime.execution_info else None) or "run"
    return {
        "staged": staged,
        "entrypoint": entrypoint,
        "prefix": f"{thread}/drafting",
        "draft_attempts": 0,
        "missing_outputs": [],
        "retry_feedback": None,
        "repairs_used": 0,
        "contract_repairs": 0,
        "smoke_repairs": 0,
        "smoke_infra_attempts": 0,
        "alignment_repairs": 0,
        "judge_calls": 0,
        "report_retries": 0,
        "qualification_repairs": 0,
        "qualification_infra_attempts": 0,
        "qualification_runs": 0,
        "gate_seq": 0,
        "phase": "initial",
        "smoke_op": ACCEPTANCE,
        "smoke_followup": None,
        "smoke_continue": False,
        "review_round": -1,
        "status": "running",
        "failure": None,
        "blocker": None,
        "canon_error": None,
        "transaction": None,
        "dependency_transaction": None,
        "pending_qualification": None,
        "accepted_alignment": None,
        "accepted_qualification": None,
        "failed_qualification": None,
        "calibration": None,
        "last_contract_diagnostic": "",
        "last_smoke_diagnostic": "",
        "last_alignment_diagnostic": "",
        "qualification_alignment_report": "",
        "qualification_post_validation": "",
        "qualification_failure_stage": "",
    }


async def draft(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run the task drafter (R7): create the missing scaffold files, one session per attempt.

    The prompt is ``drafting/draft_task`` with the placeholder values of upstream's ``task.sh``;
    a retry appends the harness retry feedback (``workflow/drafting/draft_retry_feedback``) with
    the diagnostic of the last inspection. The skills ``draft-task`` and ``memory-log`` are staged
    in the drafting directory, which only this role works in, so only this role sees them.
    """
    ctx = runtime.context
    paths = _paths(state)
    values = prompt_values(
        task_dir="./task",
        results_dir=".",
        task_md=await _staged_task_md(ctx.workspace, paths),
        extra=draft_values(state["entrypoint"]),
    )
    prompt = ctx.prompts.render("drafting/draft_task", values).rstrip("\n")
    feedback = state.get("retry_feedback")
    if feedback:
        prompt += ctx.prompts.render(
            "workflow/drafting/draft_retry_feedback", {"DIAGNOSTIC": feedback}
        )
    outcome = await run_workspace_role(
        role=Role.TASK_DRAFTER,
        runtime=runtime,
        root=paths.root,
        prompt=prompt,
        skills=skills_for_step("task"),
        skill_values=values,
        tools=[make_report_blocker_tool(prompts=ctx.prompts)],
    )
    return {
        "draft_attempts": int(state.get("draft_attempts", 0)) + 1,
        "blocker": outcome.blocker,
    }


async def inspect_scaffold_node(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Check that the drafter created every required file safely (``inspect_staged_scaffold``).

    Missing files are retried with feedback while ``draft_attempts < draft_max_attempts``; an unsafe
    artifact ends drafting.
    """
    cfg = _cfg(state)
    inspection = await inspect_scaffold(
        runtime.context.workspace, _paths(state), state["entrypoint"]
    )
    if inspection.kind == "unsafe":
        return {
            **_failed("draft_unsafe", inspection.diagnostic, scope="infrastructure"),
            "inspection": "failed",
        }
    if inspection.kind == "missing":
        if int(state.get("draft_attempts", 0)) >= cfg.draft_max_attempts:
            attempts = state.get("draft_attempts", 0)
            blocker = f" (the drafter reported: {state['blocker']})" if state.get("blocker") else ""
            return {
                **_failed(
                    "draft_incomplete",
                    f"Task scaffold remained incomplete after {attempts} draft attempt(s)"
                    f"{blocker}: " + inspection.diagnostic,
                ),
                "inspection": "failed",
            }
        return {
            "inspection": "missing",
            "missing_outputs": list(inspection.missing),
            "retry_feedback": inspection.diagnostic,
        }
    return {"inspection": "ok", "missing_outputs": [], "retry_feedback": None}


def _route_inspection(state: ScaffoldState) -> str:
    kind = str(state.get("inspection"))
    return kind if kind in ("ok", "missing") else "fail"


# --- canonical files ---------------------------------------------------------------------------


async def canonicalize(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Check the staged tree is safe and restore the harness-owned files.

    Upstream's ``validate_staged_paths`` and ``canonicalize_staged_scaffold``. The result is
    recorded for the router: after an alignment repair a missing canonical entrypoint (code 3) is
    repaired with the "canonical candidate boundary repair" text, after a qualification repair
    it counts as a post-repair validation failure, and every other refusal ends drafting.
    """
    workspace = runtime.context.workspace
    paths = _paths(state)
    phase = state.get("phase", "initial")
    try:
        await validate_staged_paths(
            workspace, paths.task, canonical_task_sha256=state["staged"].task_md_sha256
        )
        await canonicalize_staged_scaffold(workspace, paths, state["task"].root)
    except ScaffoldError as exc:
        diagnostic = sanitize_preflight_diagnostic(str(exc) + "\n")
        if exc.exit_code == 3 and phase in ("alignment_repair", "qualification_repair"):
            return {
                "canon_error": {"code": 3, "message": str(exc)},
                "last_alignment_diagnostic": diagnostic,
            }
        return {
            **_failed("scaffold_unsafe", str(exc), scope="infrastructure"),
            "canon_error": {"code": exc.exit_code, "message": str(exc)},
        }
    return {"canon_error": None, "last_alignment_diagnostic": ""}


def _route_canonicalize(state: ScaffoldState) -> str:
    if state.get("canon_error") is None:
        return "preflight"
    if _is_failed(state):
        return "fail"
    if state.get("phase") == "alignment_repair":
        return "boundary"
    return "qualification_validation_failed"


async def alignment_boundary_retry(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Repair an invalid canonical candidate boundary with the remaining alignment budget."""
    if _exhausted(state, "alignment"):
        return _failed(
            "alignment_boundary_invalid",
            "Task alignment repair left an invalid canonical candidate boundary after "
            f"{_used(state, 'alignment')} repair attempt(s)",
        )
    ctx = runtime.context
    text = ctx.prompts.render(
        "workflow/drafting/candidate_boundary_repair",
        {
            "DIAGNOSTIC_BASE64": _b64(
                _bounded_diagnostic(state.get("last_alignment_diagnostic", ""))
            )
        },
    )
    return {"alignment_prompt": _after_text(state["alignment_prompt"], text), "canon_error": None}


# --- preflight ---------------------------------------------------------------------------------


async def _run_gate(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext], check: Any
) -> GateReport:
    ctx = runtime.context
    gate = _task_gate(ctx, _cfg(state))
    return await gate.run_check(
        check,
        task=state["staged"],
        workspace=ctx.workspace,
        evaluation_id=_GATE_ID_FORMAT.format(
            prefix=state["prefix"], seq=int(state.get("gate_seq", 0)), check=check
        ),
    )


async def preflight(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """The contract gate (``validate_staged_contract``): ``TaskGate.run_check("preflight")``."""
    report = await _run_gate(state, runtime, "preflight")
    return {
        "gate": report,
        "gate_seq": int(state.get("gate_seq", 0)) + 1,
        "last_contract_diagnostic": "" if report.passed else report.diagnostic,
    }


def _route_preflight(state: ScaffoldState) -> str:
    report = _need(state, "gate")
    if report.passed:
        return "smoke"
    phase = state.get("phase", "initial")
    if phase == "alignment_repair":
        return "alignment_validation_failed"
    if phase == "qualification_repair":
        return "qualification_validation_failed"
    if phase == "initial" and not _exhausted(state, "contract"):
        return "preflight_repair"
    return "preflight_failed"


async def preflight_failed(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """End drafting after a contract failure that no repair budget or transaction can mend."""
    if state.get("phase") == "smoke_repair":
        return _failed(
            "smoke_repair_contract_invalid",
            "Provisional smoke repair failed contract validation; repair aborted",
        )
    return _failed(
        "contract_invalid",
        f"Task scaffold preflight still failed after {_used(state, 'contract')} repair attempt(s)",
    )


async def preflight_repair(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the preflight repairer (R8) on the staged task with the base64 diagnostic.

    The repair works directly on the staged scaffold (upstream scopes no transaction to it);
    a blocker or a failed session ends drafting.
    """
    ctx = runtime.context
    paths = _paths(state)
    gate_report = state.get("gate")
    diagnostic = gate_report.diagnostic if gate_report is not None else ""
    prompt = ctx.prompts.render(
        "drafting/repair_task_preflight",
        _repair_values({"TASK_PREFLIGHT_DIAGNOSTIC": _b64(diagnostic)}),
    )
    label = f"preflight_repair_{int(state.get('contract_repairs', 0))}"
    outcome = await run_workspace_role(
        role=Role.PREFLIGHT_REPAIRER,
        runtime=runtime,
        root=paths.task,
        prompt=_with_gate_text(ctx, prompt),
        tools=_tools(state, runtime, label=label, gate=_task_gate(ctx, _cfg(state))),
    )
    failure = _repair_failure(outcome)
    update: dict[str, Any] = {"blocker": outcome.blocker}
    if failure is not None:
        update.update(_failed(failure[0], failure[1], scope="infrastructure"))
    return update


async def close_preflight_repair(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Count the repair, then validate the staged paths and restore the canonical files."""
    update: dict[str, Any] = {
        "contract_repairs": int(state.get("contract_repairs", 0)) + 1,
        "repairs_used": int(state.get("repairs_used", 0)) + 1,
    }
    return update


def _route_failure_or(next_node: str) -> Any:
    def route(state: ScaffoldState) -> str:
        return "fail" if _is_failed(state) else next_node

    return route


# --- smoke -------------------------------------------------------------------------------------


def smoke_target_operations(gate: str) -> tuple[str, str | None]:
    """Choose the validation to repeat after a smoke repair (upstream ``smoke_target_operations``).

    A failure of the contract, runtime or one smoke is re-checked alone and then in the full
    acceptance check; a production-entry failure is re-checked after the runtime check.
    """
    if gate in {"contract", "runtime", "search-smoke", "final-smoke"}:
        return gate, None
    if gate == "production-entry":
        return "runtime", "production-entry"
    return ACCEPTANCE, None


async def smoke(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run one smoke check (``acceptance-smoke`` or a targeted one) and plan what follows.

    A passing targeted check advances to its follow-up and finally to the full acceptance check
    (``smoke_continue``); infrastructure failures are counted so that the router can retry them
    without spending a repair; the typed diagnostic of a failure is kept for the repair prompt.
    """
    operation = state.get("smoke_op") or ACCEPTANCE
    report = await _run_gate(state, runtime, operation)
    update: dict[str, Any] = {
        "gate": report,
        "gate_seq": int(state.get("gate_seq", 0)) + 1,
        "smoke_continue": False,
    }
    if report.passed:
        update["last_smoke_diagnostic"] = ""
        update["smoke_infra_attempts"] = 0
        if operation != ACCEPTANCE:
            update["smoke_op"] = state.get("smoke_followup") or ACCEPTANCE
            update["smoke_followup"] = None
            update["smoke_continue"] = True
        return update
    update["last_smoke_diagnostic"] = typed_smoke_diagnostic(report)
    update["smoke_gate"] = report.check
    if report.infrastructure:
        update["smoke_infra_attempts"] = int(state.get("smoke_infra_attempts", 0)) + 1
    else:
        update["smoke_infra_attempts"] = 0
    return update


def _route_smoke(state: ScaffoldState) -> str:
    report = _need(state, "gate")
    if state.get("smoke_continue"):
        return "smoke"
    if report.passed:
        return "smoke_accepted"
    cfg = _cfg(state)
    if report.infrastructure:
        if (
            report.category in SMOKE_RETRYABLE_CATEGORIES
            and int(state.get("smoke_infra_attempts", 0)) < cfg.smoke_max_infrastructure_attempts
        ):
            return "smoke"
        return "smoke_failed"
    phase = state.get("phase", "initial")
    if phase == "alignment_repair":
        return "alignment_validation_failed"
    if phase == "qualification_repair":
        return "qualification_validation_failed"
    if _exhausted(state, "smoke"):
        return "smoke_failed"
    return "open_smoke_transaction"


async def smoke_failed(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """End drafting after a smoke failure that cannot be repaired (the transaction rolls back)."""
    report = _need(state, "gate")
    if report.infrastructure:
        return _failed(
            "smoke_infrastructure_failure",
            "Task evaluator smoke failed for an infrastructure reason; repair skipped: "
            + (report.category or ""),
            scope="infrastructure",
        )
    return _failed(
        "smoke_failed",
        f"Task evaluator smoke still failed after {_used(state, 'smoke')} repair attempt(s)",
    )


async def smoke_accepted(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Every smoke check passed: commit a provisional smoke repair chain (upstream commit)."""
    if state.get("phase") == "smoke_repair":
        return {"dependency_transaction": None, "phase": "initial", "smoke_infra_attempts": 0}
    return {}


def _route_smoke_accepted(state: ScaffoldState) -> str:
    if state.get("phase") == "qualification_repair":
        return "check_qualification_candidate"
    return "alignment_view"


async def open_smoke_transaction(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Snapshot the scaffold before a smoke repair and record its mutation boundary.

    One transaction is kept across the smoke repairs of a chain; each round records its own scope
    (the files, the contract and the policy of its diagnostic) while the snapshot to roll back to
    stays the chain's first. A ``runtime_dependencies`` diagnostic permits removing an unused pin.
    """
    workspace = runtime.context.workspace
    paths = _paths(state)
    chain: DependencyTransaction | None = state.get("dependency_transaction")
    diagnostic = state.get("last_smoke_diagnostic", "")
    policy = contract_policy_for_smoke_diagnostic(diagnostic)
    label = f"smoke_repair_{int(state.get('smoke_repairs', 0))}"
    try:
        scope = await open_scoped_transaction(workspace, paths, label=label, policy=policy)
    except ScaffoldError as exc:
        return _failed("smoke_repair_scope_invalid", str(exc), scope="infrastructure")
    if chain is not None:
        scope = scope.model_copy(
            update={
                "snapshot": chain.transaction.snapshot,
                "base_fingerprint": chain.transaction.base_fingerprint,
            }
        )
    active = bool(chain.dependency_repair_active) if chain is not None else False
    return {
        "dependency_transaction": DependencyTransaction(
            transaction=scope,
            dependency_repair_active=active or policy == "synchronized-dependency-removals",
        ),
        "smoke_prompt_active": "active" if active else "inactive",
        "phase": "smoke_repair",
    }


async def smoke_repair(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run the smoke repairer (R9) inside the scoped transaction.

    The prompt carries the typed smoke diagnostic and the state of the provisional
    dependency-removal transaction; the in-session ``task_gate`` tool authorises dependency
    removal when its conditions hold (:class:`~.gate.RepairSessionGate`).
    """
    ctx = runtime.context
    paths = _paths(state)
    chain: DependencyTransaction = _need(state, "dependency_transaction")
    prompt = ctx.prompts.render(
        "drafting/repair_task_smoke",
        _repair_values(
            {
                "TASK_SMOKE_DIAGNOSTIC": state.get("last_smoke_diagnostic", ""),
                "PROVISIONAL_DEPENDENCY_TRANSACTION": str(
                    state.get("smoke_prompt_active", "inactive")
                ),
            }
        ),
    )
    label = chain.transaction.label
    gate = RepairSessionGate(
        _task_gate(ctx, _cfg(state)), paths=paths, transaction=chain.transaction, label=label
    )
    outcome = await run_workspace_role(
        role=Role.SMOKE_REPAIRER,
        runtime=runtime,
        root=paths.task,
        prompt=_with_gate_text(ctx, prompt),
        tools=_tools(state, runtime, label=label, gate=gate),
    )
    failure = _repair_failure(outcome)
    update: dict[str, Any] = {"blocker": outcome.blocker}
    if failure is not None:
        update.update(_failed(failure[0], failure[1], scope="infrastructure"))
    return update


async def close_smoke_repair(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Validate the scope of a smoke repair and plan the targeted re-validation.

    A repair that changed a file outside the mutable paths or the contract's meaning is detected
    and the failure ends drafting (the transaction rolls back); a repair that the harness
    authorised to remove a dependency makes the removal permitted for the rest of the chain.
    """
    workspace = runtime.context.workspace
    paths = _paths(state)
    chain: DependencyTransaction = _need(state, "dependency_transaction")
    authorized = await workspace.exists(paths.authorization(chain.transaction.label))
    try:
        _, transaction = await validate_repair_scope(
            workspace, paths, chain.transaction, authorized=authorized
        )
    except RepairScopeError as exc:
        return _failed("smoke_repair_out_of_scope", str(exc))
    except ScaffoldError as exc:
        return _failed("smoke_repair_scope_invalid", str(exc), scope="infrastructure")
    operation, followup = smoke_target_operations(str(state.get("smoke_gate", ACCEPTANCE)))
    return {
        "dependency_transaction": DependencyTransaction(
            transaction=transaction,
            dependency_repair_active=chain.dependency_repair_active or authorized,
        ),
        "smoke_repairs": int(state.get("smoke_repairs", 0)) + 1,
        "repairs_used": int(state.get("repairs_used", 0)) + 1,
        "smoke_op": operation,
        "smoke_followup": followup,
    }


# --- alignment ---------------------------------------------------------------------------------


async def alignment_view(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Prepare a judge round: fingerprints, a read-only view of the scaffold, the prompt.

    The view omits runtime state and holds the accepted design as ``_accepted_task_design.json``.
    The prompt is ``drafting/validate_task`` with the harness attestation
    (:func:`~.qualification.alignment_context`), which carries the qualification calibration of a
    pending repair, the current qualification evidence and the historical failure projection, and
    the retained alignment history after it.
    """
    ctx = runtime.context
    workspace = ctx.workspace
    paths = _paths(state)
    round_index = int(state.get("review_round", -1)) + 1
    try:
        fingerprints = await alignment_fingerprints(workspace, paths.task)
        view = await prepare_judge_view(
            workspace, paths, round_index, has_design=await workspace.exists(paths.design_file)
        )
        staged_fp = await tree_fingerprint(workspace, paths.task)
        view_fp = await tree_fingerprint(workspace, view)
    except ScaffoldError as exc:
        return _failed("alignment_view_invalid", str(exc))
    pending: PendingQualificationRepair | None = state.get("pending_qualification")
    isolated = _isolated(ctx)
    current: dict[str, object] | None = None
    projection: dict[str, object] | None = None
    if isolated:
        accepted: QualificationRecord | None = state.get("accepted_qualification")
        try:
            if accepted is not None:
                plan = await build_publication_plan(
                    workspace,
                    paths.task,
                    state["task"].root,
                    max_runtime_bytes=_cfg(state).max_runtime_bytes,
                )
                current = qualification_success_context(
                    accepted, current_fingerprint=staged_fp, current_plan_sha256=plan.sha256
                )
            else:
                current = {
                    "status": "pending",
                    "stage": "production-search",
                    "task_fingerprint": staged_fp,
                }
        except (QualificationError, PublicationError) as exc:
            return _failed("qualification_stale", str(exc), scope="infrastructure")
        if pending is not None and pending.source_report is not None:
            projection = qualification_failure_projection(
                pending.source_report, current_fingerprint=staged_fp
            )
    context = alignment_context(
        isolated=isolated,
        calibration=pending.calibration if pending is not None else None,
        current_qualification=current,
        qualification_failure=projection,
    )
    design_exists = await workspace.exists(paths.design_file)
    task_md = await _read_text(workspace, join(view, "task.md"))
    extra = {
        "TASK_CONTRACT_PATH": "task_contract.json",
        "SEARCH_EVALUATOR_PATH": "evaluate.py",
        "FINAL_EVALUATOR_PATH": "evaluate_final.py",
        "TASK_ALIGNMENT_FINGERPRINTS": json.dumps(fingerprints, sort_keys=True),
        "TASK_ALIGNMENT_CONTEXT": context,
        "TASK_ALIGNMENT_REPORT_PATH": "./task_alignment.json",
        "TASK_DESIGN_PATH": "./_accepted_task_design.json" if design_exists else "./task.md",
    }
    values = prompt_values(task_dir=".", results_dir=".", task_md=task_md, extra=extra)
    rendered = ctx.prompts.render("drafting/validate_task", values)
    history = render_alignment_history(
        state.get("alignment_history", []),
        current_fingerprint=staged_fp,
        heading=ctx.prompts.text("workflow/drafting/alignment_history_heading"),
    )
    return {
        "review_round": round_index,
        "fingerprints": fingerprints,
        "view_dir": view,
        "view_fingerprint": view_fp,
        "staged_fingerprint": staged_fp,
        "judge_prompt": (rendered + "\n" + history).rstrip("\n"),
        "judge_suffix": "",
        "judge_report": None,
        "judge_failure": None,
        "report_attempt": 0,
    }


async def alignment_judge(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the independent alignment judge (R10) on the read-only view; its answer is the report.

    At most ``2 * (alignment_max_repairs + qualification_max_repairs + 1)`` judge calls are made
    over the whole drafting (upstream's bound). The judge has no file or execution tools that can
    change anything: its backend is read-only, and ``nl`` gives it the line-numbered read its
    prompt names.
    """
    ctx = runtime.context
    limits = _limits(_cfg(state))
    budget = 2 * (limits["alignment"] + limits["qualification"] + 1)
    if int(state.get("judge_calls", 0)) >= budget:
        return _failed(
            "alignment_judge_budget_exhausted", "Task alignment judge call budget exhausted"
        )
    view = state["view_dir"]
    outcome = await run_structured_role(
        role=Role.ALIGNMENT_JUDGE,
        schema=AlignmentReport,
        runtime=runtime,
        root=view,
        read_only=True,
        prompt=state["judge_prompt"] + state.get("judge_suffix", ""),
        tools=[make_nl_tool(ctx.workspace, view, prompts=ctx.prompts)],
    )
    report = outcome.structured if isinstance(outcome.structured, AlignmentReport) else None
    return {
        "judge_calls": int(state.get("judge_calls", 0)) + 1,
        "judge_report": report,
        "judge_failure": outcome.failure,
    }


async def check_alignment(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Validate the judge's report with upstream's rules; retry a malformed one once.

    A scaffold or view that changed during the review fails drafting. A valid report (pass or
    fail) is retained in the alignment history and written, as audit, under
    ``_task_validation/round_<n>/``. A malformed report is retried with the output-contract
    correction text (``workflow/drafting/alignment_report_correction``); a second malformed
    report ends drafting.
    """
    ctx = runtime.context
    workspace = ctx.workspace
    paths = _paths(state)
    if (
        await tree_fingerprint(workspace, paths.task) != state["staged_fingerprint"]
        or await tree_fingerprint(workspace, state["view_dir"]) != state["view_fingerprint"]
    ):
        return _failed("judge_modified_scaffold", "Independent task judge modified the scaffold")
    report: AlignmentReport | None = state.get("judge_report")
    diagnostic: str | None = None
    if report is None:
        diagnostic = "invalid alignment report: requested report file is missing"
    else:
        try:
            await validate_alignment_report(
                workspace,
                report,
                expected_fingerprints=state["fingerprints"],
                evidence_root=state["view_dir"],
            )
        except AlignmentReportError as exc:
            diagnostic = f"invalid alignment report: {exc}"
    if diagnostic is not None:
        if int(state.get("report_attempt", 0)) >= 1:
            return _failed("alignment_report_invalid", diagnostic)
        text = ctx.prompts.render(
            "workflow/drafting/alignment_report_correction",
            {"DIAGNOSTIC_BASE64": _b64(_bounded_diagnostic(diagnostic))},
        )
        return {
            "judge_suffix": "\n" + text,
            "report_attempt": int(state.get("report_attempt", 0)) + 1,
            "alignment": None,
        }
    assert report is not None
    round_index = int(state["review_round"])
    entry = capture_alignment_report(
        report,
        task_fingerprint=state["staged_fingerprint"],
        name=f"alignment_{round_index}.json",
    )
    await workspace.write_bytes(
        join(paths.validation, f"round_{round_index}", "task_alignment.json"),
        json_text(report.model_dump(mode="json", by_alias=True)).encode("utf-8"),
    )
    return {
        "alignment": report,
        "alignment_history": [entry],
        "last_alignment_diagnostic": "",
        "report_attempt": int(state.get("report_attempt", 0)),
    }


def _route_check_alignment(state: ScaffoldState) -> str:
    if _is_failed(state):
        return "fail"
    report: AlignmentReport | None = state.get("alignment")
    if report is None:
        return "judge"
    return "passed" if report.verdict.strip() == "pass" else "failed"


async def alignment_passed(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """The judge accepted: commit a provisional alignment repair, remember the accepted report."""
    return {
        "accepted_alignment": state["alignment"],
        "transaction": None,
        "phase": (
            "qualification_repair" if state.get("pending_qualification") is not None else "initial"
        ),
    }


def _route_alignment_passed(state: ScaffoldState) -> str:
    return (
        "validate_qualification" if state.get("accepted_qualification") is not None else "qualify"
    )


async def alignment_failed(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """The judge rejected the scaffold.

    With a pending qualification repair the rejection is recorded as a failed fingerprint (and
    the rejected edits kept as history), acceptance is cleared, and the qualification repair
    loop continues from the repair's source report. Otherwise the alignment budget decides:
    spent, drafting fails; else an alignment repair follows.
    """
    ctx = runtime.context
    workspace = ctx.workspace
    pending: PendingQualificationRepair | None = state.get("pending_qualification")
    report: AlignmentReport = _need(state, "alignment")
    if pending is None:
        if _exhausted(state, "alignment"):
            return _failed(
                "alignment_failed",
                f"Task alignment still failed after {_used(state, 'alignment')} repair attempt(s)",
            )
        return {}
    paths = _paths(state)
    fingerprint = await tree_fingerprint(workspace, paths.task)
    report_json = json_text(report.model_dump(mode="json", by_alias=True))
    try:
        failures = record_failure(
            state.get("qualification_failures", []),
            fingerprint=fingerprint,
            category=ALIGNMENT_FAILURE_CATEGORY,
            evidence=report_json,
        )
        retained = await _retain_rejected(
            state, workspace, pending, fingerprint, [("task_alignment.json", report_json)]
        )
    except ValueError as exc:
        return _failed("qualification_history_invalid", str(exc), scope="infrastructure")
    return {
        "qualification_failures": failures,
        "rejected_repairs": [retained],
        "qualification_alignment_report": report_json,
        "accepted_alignment": None,
        "accepted_qualification": None,
        "failed_qualification": pending.source_report,
    }


def _route_alignment_failed(state: ScaffoldState) -> str:
    if _is_failed(state):
        return "fail"
    if state.get("pending_qualification") is not None:
        return "qualification_budget"
    return "open_alignment_transaction"


async def _retain_rejected(
    state: Mapping[str, Any],
    workspace: Workspace,
    pending: PendingQualificationRepair,
    rejected_fingerprint: str,
    reports: Sequence[tuple[str, str]],
) -> RejectedRepair:
    """Capture the provisional edits of a rejected qualification repair before any restore."""
    paths = _paths(state)
    after, omitted_after = await collect_sources(workspace, paths.task)
    return capture_rejected_repair(
        pending.transaction.sources_before,
        after,
        omitted_files=pending.transaction.sources_omitted + omitted_after,
        base_fingerprint=pending.base_fingerprint,
        reports=[(name, text.encode("utf-8")) for name, text in reports[:3]],
        rejected_fingerprint=rejected_fingerprint,
    )


# --- alignment repair --------------------------------------------------------------------------


async def open_alignment_transaction(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Open (or keep) the tree transaction of an alignment repair and build its base prompt.

    The prompt is ``drafting/repair_task`` with the bounded failed report and the immutable
    accepted design, followed by the retained alignment history. The transaction stays
    provisional until the judge accepts; it is kept across repair rounds.
    """
    ctx = runtime.context
    workspace = ctx.workspace
    paths = _paths(state)
    transaction: RepairTransaction | None = state.get("transaction")
    if transaction is None:
        try:
            transaction = await open_tree_transaction(
                workspace, paths, label=f"alignment_{int(state.get('alignment_repairs', 0))}"
            )
        except ScaffoldError as exc:
            return _failed("alignment_transaction_failed", str(exc), scope="infrastructure")
    report: AlignmentReport = _need(state, "alignment")
    try:
        bounded = redact_sensitive_text(bounded_repair_report(report))
    except ValueError as exc:
        return _failed("alignment_prompt_too_large", str(exc))
    design_file = await _read_text(workspace, paths.design_file)
    design = redact_sensitive_text(design_file) if design_file is not None else "{}"
    if len(design.encode("utf-8")) > 96 * 1024:
        return _failed("alignment_prompt_too_large", "Accepted task design exceeds 96 KiB")
    rendered = ctx.prompts.render(
        "drafting/repair_task",
        _repair_values({"TASK_ALIGNMENT_REPORT": bounded, "TASK_DESIGN": design}),
    )
    history = render_alignment_history(
        state.get("alignment_history", []),
        current_fingerprint=await tree_fingerprint(workspace, paths.task),
        heading=ctx.prompts.text("workflow/drafting/alignment_history_heading"),
    )
    base = (rendered + "\n" + history).rstrip("\n")
    return {
        "transaction": transaction,
        "phase": "alignment_repair",
        "alignment_prompt_base": base,
        "alignment_prompt": base,
        "accepted_alignment": None,
        "accepted_qualification": None,
        "last_contract_diagnostic": "",
        "last_smoke_diagnostic": "",
        "last_alignment_diagnostic": "",
        "canon_error": None,
        "smoke_op": ACCEPTANCE,
        "smoke_followup": None,
    }


async def alignment_repair(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the alignment repairer (R11) on the provisional tree."""
    ctx = runtime.context
    paths = _paths(state)
    label = f"repair_{int(state.get('review_round', 0))}_{int(state.get('alignment_repairs', 0))}"
    outcome = await run_workspace_role(
        role=Role.ALIGNMENT_REPAIRER,
        runtime=runtime,
        root=paths.task,
        prompt=_with_gate_text(ctx, state["alignment_prompt"]),
        tools=_tools(state, runtime, label=label, gate=_task_gate(ctx, _cfg(state))),
    )
    failure = _repair_failure(outcome)
    update: dict[str, Any] = {"blocker": outcome.blocker}
    if failure is not None:
        update.update(_failed(failure[0], failure[1], scope="infrastructure"))
    return update


async def close_alignment_repair(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Count the repair (also when it failed, as upstream does)."""
    return {
        "alignment_repairs": int(state.get("alignment_repairs", 0)) + 1,
        "repairs_used": int(state.get("repairs_used", 0)) + 1,
    }


async def alignment_validation_failed(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """A provisional alignment repair failed contract or smoke validation; keep the edits.

    While the alignment budget lasts, the next repair continues on the same tree with the
    "continue provisional alignment repair" text and the typed diagnostic; when it is spent the
    transaction rolls back and drafting fails.
    """
    if _exhausted(state, "alignment"):
        return _failed(
            "alignment_repair_validation_failed",
            "Task alignment repair validation failed after "
            f"{_used(state, 'alignment')} repair attempt(s)",
        )
    ctx = runtime.context
    diagnostic = (
        state.get("last_alignment_diagnostic")
        or state.get("last_smoke_diagnostic")
        or state.get("last_contract_diagnostic")
        or '{"schema_version":1,"status":"infrastructure_failure","category":'
        '"alignment_repair_validation","component":"harness","stage":"alignment-repair",'
        '"diagnostic":"Provisional alignment repair failed structural or contract validation."}'
    )
    text = ctx.prompts.render(
        "workflow/drafting/continue_provisional_repair",
        {"DIAGNOSTIC_BASE64": _b64(_bounded_diagnostic(diagnostic))},
    )
    return {
        "alignment_prompt": _after_text(state["alignment_prompt_base"], text),
        "last_contract_diagnostic": "",
        "last_smoke_diagnostic": "",
        "last_alignment_diagnostic": "",
        "canon_error": None,
        "phase": "alignment_repair",
        "smoke_op": ACCEPTANCE,
        "smoke_followup": None,
    }


# --- qualification -----------------------------------------------------------------------------


async def qualify(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run the full production search qualification and bind a receipt to the result.

    The publication plan is computed before and after the gate: a qualification that changed the
    publishable scaffold is an authored failure (``qualification_modified_scaffold``). A passing
    gate yields a :class:`QualificationReceipt` binding the scaffold fingerprint, the plan, the
    evaluators, ``requirements.txt`` and the baseline's result.
    """
    ctx = runtime.context
    workspace = ctx.workspace
    cfg = _cfg(state)
    paths = _paths(state)
    root = paths.task
    try:
        plan_before = await build_publication_plan(
            workspace, root, state["task"].root, max_runtime_bytes=cfg.max_runtime_bytes
        )
    except PublicationError as exc:
        record = QualificationRecord(
            status="authored_failure",
            category="qualification_publication_plan",
            stage="publication-plan",
            diagnostic=str(exc),
            task_fingerprint=await safe_tree_fingerprint(workspace, root),
        )
        return {
            "qualification": record,
            "qualification_runs": int(state.get("qualification_runs", 0)) + 1,
        }
    runs = int(state.get("qualification_runs", 0))
    gate = _task_gate(ctx, cfg)
    report = await gate.run_check(
        "qualification",
        task=state["staged"],
        workspace=workspace,
        evaluation_id=f"{state['prefix']}/qualification_{runs:02d}",
    )
    update: dict[str, Any] = {"qualification_runs": runs + 1}
    if not report.passed:
        update["qualification"] = record_from_gate(report)
        update["qualification_infra_attempts"] = (
            int(state.get("qualification_infra_attempts", 0)) + 1 if report.infrastructure else 0
        )
        return update
    fingerprint = await safe_tree_fingerprint(workspace, root)
    try:
        plan_after = await build_publication_plan(
            workspace, root, state["task"].root, max_runtime_bytes=cfg.max_runtime_bytes
        )
        contract = await load_task_contract(workspace, root)
        evaluator_sha = await evaluator_tree_sha256(workspace, root, contract)
        requirements = await workspace.stat(join(root, "requirements.txt"))
        requirements_sha = (
            sha256_hex(await workspace.read_bytes(join(root, "requirements.txt")))
            if requirements is not None and requirements.kind == "file"
            else sha256_hex(b"")
        )
    except (PublicationError, TaskContractError) as exc:
        update["qualification"] = QualificationRecord(
            status="authored_failure",
            category="qualification_modified_scaffold",
            stage="production-search",
            diagnostic=str(exc),
            task_fingerprint=fingerprint,
        )
        return update
    if plan_after.sha256 != plan_before.sha256 or fingerprint != report.fingerprint:
        update["qualification"] = QualificationRecord(
            status="authored_failure",
            category="qualification_modified_scaffold",
            stage="production-search",
            diagnostic="Full search qualification modified the publishable scaffold.",
            task_fingerprint=fingerprint,
        )
        return update
    compute = contract.compute.search
    receipt = QualificationReceipt(
        task_fingerprint=fingerprint,
        publication_plan_sha256=plan_before.sha256,
        evaluator_tree_sha256=evaluator_sha,
        requirements_sha256=requirements_sha,
        baseline_combined_score=float(report.details.get("baseline_combined_score", 0.0)),
        baseline_correct=report.details.get("baseline_correct"),
        baseline_feasible=True,
        accelerator_model=compute.accelerator_model,
        evaluator_gpu_count=compute.evaluator_gpu_count,
        candidate_gpu_count=compute.candidate_gpu_count,
        evaluation_id=str(report.details.get("evaluation_id", "")),
    )
    update["qualification"] = record_from_gate(report, receipt=receipt)
    update["qualification_infra_attempts"] = 0
    return update


def _route_qualify(state: ScaffoldState) -> str:
    record: QualificationRecord = _need(state, "qualification")
    if record.status == "pass":
        return "passed"
    if record.status == "authored_failure":
        return "failed"
    cfg = _cfg(state)
    if (
        record.category not in TERMINAL_GPU_CATEGORIES
        and int(state.get("qualification_infra_attempts", 0))
        < cfg.qualification_max_infrastructure_attempts
    ):
        return "retry"
    return "infrastructure"


async def qualification_infrastructure_failed(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """End drafting after an infrastructure failure of the qualification (the draft rolls back)."""
    record: QualificationRecord = _need(state, "qualification")
    return _failed(
        "qualification_infrastructure_failure",
        "Full production search qualification failed for an infrastructure reason: "
        f"{record.category}",
        scope="infrastructure",
    )


async def qualification_passed(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Accept the qualification report; a pending repair goes back to the judge."""
    return {
        "accepted_qualification": state["qualification"],
        "failed_qualification": None,
        **({"accepted_alignment": None} if state.get("pending_qualification") is not None else {}),
    }


def _route_qualification_passed(state: ScaffoldState) -> str:
    return (
        "alignment_view"
        if state.get("pending_qualification") is not None
        else "validate_qualification"
    )


async def validate_qualification(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Reject stale qualification evidence before it is reused."""
    ctx = runtime.context
    workspace = ctx.workspace
    paths = _paths(state)
    accepted: QualificationRecord = _need(state, "accepted_qualification")
    try:
        plan = await build_publication_plan(
            workspace,
            paths.task,
            state["task"].root,
            max_runtime_bytes=_cfg(state).max_runtime_bytes,
        )
        qualification_success_context(
            accepted,
            current_fingerprint=await tree_fingerprint(workspace, paths.task),
            current_plan_sha256=plan.sha256,
        )
    except (QualificationError, PublicationError, ScaffoldError) as exc:
        return _failed(
            "qualification_stale",
            "Current production qualification evidence is missing, invalid, or stale: " + str(exc),
            scope="infrastructure",
        )
    return {}


def _route_validated(state: ScaffoldState) -> str:
    if _is_failed(state):
        return "fail"
    return "commit" if state.get("pending_qualification") is not None else "publish"


async def commit_qualification_repair(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Alignment and qualification accepted the same scaffold: accept the calibration, commit.

    The calibration record is bound to the accepted alignment report, the qualification report
    and the receipt (upstream ``qualification_calibration accept``), then the pending
    transaction is committed (its snapshot is dropped).
    """
    pending: PendingQualificationRepair = _need(state, "pending_qualification")
    alignment: AlignmentReport = _need(state, "accepted_alignment")
    qualification: QualificationRecord = _need(state, "accepted_qualification")
    if pending.calibration is None or qualification.receipt is None:
        return _failed(
            "qualification_commit_failed",
            "the pending qualification repair has no calibration or receipt",
            scope="infrastructure",
        )
    try:
        accepted = accept_calibration(
            pending.calibration,
            alignment_report_sha256=sha256_text(
                json_text(alignment.model_dump(mode="json", by_alias=True))
            ),
            qualification_report_sha256=sha256_text(report_text(qualification)),
            qualification_receipt_sha256=sha256_text(report_text(qualification.receipt)),
        )
    except ValueError as exc:
        return _failed("qualification_commit_failed", str(exc), scope="infrastructure")
    return {
        "calibration": accepted,
        "pending_qualification": None,
        "transaction": None,
        "phase": "initial",
    }


async def qualification_failed(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Record an authored qualification failure and keep the provisional edits as history.

    The failed report's scaffold fingerprint and category enter the failure history (so that a
    repair reproducing the fingerprint is an oscillation); acceptance is cleared. For a failure of
    a repaired scaffold the rejected edits are retained as source-only history.
    """
    workspace = runtime.context.workspace
    record: QualificationRecord = _need(state, "qualification")
    pending: PendingQualificationRepair | None = state.get("pending_qualification")
    update: dict[str, Any] = {
        "failed_qualification": record,
        "accepted_alignment": None,
        "accepted_qualification": None,
        "qualification_infra_attempts": 0,
        "qualification_failure_stage": "production-search",
    }
    try:
        update["qualification_failures"] = record_failure(
            state.get("qualification_failures", []),
            fingerprint=record.task_fingerprint,
            category=record.category,
            evidence=report_text(record),
        )
        if pending is not None:
            update["rejected_repairs"] = [
                await _retain_rejected(
                    state,
                    workspace,
                    pending,
                    record.task_fingerprint,
                    [
                        ("qualification.json", report_text(record)),
                        ("post_validation.json", state.get("qualification_post_validation", "")),
                        ("task_alignment.json", state.get("qualification_alignment_report", "")),
                    ],
                )
            ]
    except ValueError as exc:
        return _failed("qualification_history_invalid", str(exc), scope="infrastructure")
    return update


def _route_qualification_budget(state: ScaffoldState) -> str:
    if _is_failed(state):
        return "fail"
    if int(state.get("qualification_repairs", 0)) < _limits(_cfg(state))["qualification"]:
        return "open_qualification_transaction"
    return "qualification_exhausted"


async def qualification_exhausted(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """The qualification repair budget is spent (the pending transaction rolls back)."""
    used = int(state.get("qualification_repairs", 0))
    limit = _limits(_cfg(state))["qualification"]
    return _failed(
        "qualification_repair_exhausted",
        f"Qualification repairs {used}/{limit}: repair budget exhausted; last failed gate: "
        f"{state.get('qualification_failure_stage') or 'production-search'}",
    )


async def open_qualification_transaction(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Open (or keep) the pending tree transaction of the qualification repair chain.

    The first repair snapshots the aligned scaffold; later repairs continue on the provisional
    draft. Acceptance is cleared before editing.
    """
    workspace = runtime.context.workspace
    paths = _paths(state)
    pending: PendingQualificationRepair | None = state.get("pending_qualification")
    if pending is None:
        try:
            transaction = await open_tree_transaction(
                workspace,
                paths,
                label=f"qualification_{int(state.get('qualification_repairs', 0))}",
            )
        except ScaffoldError as exc:
            return _failed("qualification_transaction_failed", str(exc), scope="infrastructure")
        pending = PendingQualificationRepair(
            transaction=transaction,
            base_fingerprint=transaction.base_fingerprint or "",
            index=int(state.get("qualification_repairs", 0)),
        )
    return {
        "pending_qualification": pending.model_copy(
            update={"source_report": state.get("failed_qualification")}
        ),
        "phase": "qualification_repair",
        "accepted_alignment": None,
        "accepted_qualification": None,
    }


async def qualification_repair(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the qualification repairer (R12) on the provisional draft.

    The prompt is assembled as upstream's ``task.sh`` does (:func:`~.qualification
    .render_qualification_repair_prompt`): the paragraph and the five base64 blocks, the shared
    repair policy, the alignment history and the rejected-repair history; it carries the prior
    calibration record and the latest rejected repair validation.
    """
    ctx = runtime.context
    workspace = ctx.workspace
    paths = _paths(state)
    pending: PendingQualificationRepair = _need(state, "pending_qualification")
    failed: QualificationRecord = state["failed_qualification"] or pending.source_report  # type: ignore[assignment]
    before = await tree_fingerprint(workspace, paths.task)
    prompt = render_qualification_repair_prompt(
        ctx.prompts,
        diagnostic=report_text(failed),
        design=await _design_text(workspace, state),
        alignment_report=state.get("qualification_alignment_report", ""),
        calibration=report_text(pending.calibration) if pending.calibration is not None else "",
        post_validation=state.get("qualification_post_validation", ""),
        alignment_history=render_alignment_history(
            state.get("alignment_history", []),
            current_fingerprint=before,
            heading=ctx.prompts.text("workflow/drafting/alignment_history_heading"),
        ),
        repair_history=render_repair_history(
            state.get("rejected_repairs", []),
            current_fingerprint=before,
            heading=ctx.prompts.text("workflow/drafting/repair_history_heading"),
        ),
    )
    label = f"qualification_repair_{int(state.get('qualification_repairs', 0))}"
    outcome = await run_workspace_role(
        role=Role.QUALIFICATION_REPAIRER,
        runtime=runtime,
        root=paths.task,
        prompt=_with_gate_text(ctx, prompt),
        tools=_tools(state, runtime, label=label, gate=_task_gate(ctx, _cfg(state))),
    )
    failure = _repair_failure(outcome)
    update: dict[str, Any] = {"blocker": outcome.blocker}
    if failure is not None:
        update.update(_failed(failure[0], failure[1], scope="infrastructure"))
        update["qualification_failure_stage"] = "agent"
    return update


async def close_qualification_repair(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Count the repair and reset the validation diagnostics before its post-repair validation."""
    return {
        "qualification_repairs": int(state.get("qualification_repairs", 0)) + 1,
        "repairs_used": int(state.get("repairs_used", 0)) + 1,
        "last_contract_diagnostic": "",
        "last_smoke_diagnostic": "",
        "last_alignment_diagnostic": "",
        "canon_error": None,
        "smoke_op": ACCEPTANCE,
        "smoke_followup": None,
        "qualification_post_validation": "",
        "phase": "qualification_repair",
    }


def _failed_report_text(state: Mapping[str, Any]) -> str:
    failed: QualificationRecord | None = state.get("failed_qualification")
    return report_text(failed) if failed is not None else ""


async def qualification_validation_failed(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """A qualification repair failed its post-repair validation (contract, smoke, canonical files).

    The rejected fingerprint is recorded in the failure history under
    ``qualification_post_validation_failure`` with the validation evidence, the rejected edits are
    retained as history, and the repair loop continues with the same source report while its
    budget lasts.
    """
    workspace = runtime.context.workspace
    pending: PendingQualificationRepair = _need(state, "pending_qualification")
    after = await safe_tree_fingerprint(workspace, _paths(state).task)
    evidence = (
        state.get("last_alignment_diagnostic")
        or state.get("last_smoke_diagnostic")
        or state.get("last_contract_diagnostic")
        or ""
    )
    try:
        failures = record_failure(
            state.get("qualification_failures", []),
            fingerprint=after,
            category=POST_VALIDATION_CATEGORY,
            evidence=evidence,
        )
        retained = await _retain_rejected(
            state,
            workspace,
            pending,
            after,
            [
                ("qualification.json", _failed_report_text(state)),
                ("post_validation.json", evidence),
                ("task_alignment.json", state.get("qualification_alignment_report", "")),
            ],
        )
    except ValueError as exc:
        return _failed("qualification_history_invalid", str(exc), scope="infrastructure")
    return {
        "qualification_failures": failures,
        "rejected_repairs": [retained],
        "qualification_post_validation": evidence,
        "qualification_failure_stage": "post-repair-validation",
        "canon_error": None,
    }


async def check_qualification_candidate(
    state: ScaffoldState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Reject a repeated candidate, then create the calibration (upstream steps 3 and 4).

    A repaired tree whose fingerprint is already in the failure history is an oscillation: drafting
    ends with the typed report ``qualification_repair_oscillation`` and the transaction rolls
    back. Otherwise a calibration record binds the original and calibrated fingerprints, the
    accepted design and the qualification failure; the transaction stays pending.
    """
    workspace = runtime.context.workspace
    pending: PendingQualificationRepair = _need(state, "pending_qualification")
    after = await tree_fingerprint(workspace, _paths(state).task)
    repeated = reject_repeated_candidate(state.get("qualification_failures", []), fingerprint=after)
    if repeated is not None:
        return {
            **_failed(repeated.category, repeated.diagnostic),
            "qualification_failure_stage": "oscillation",
            "qualification": repeated,
        }
    failed: QualificationRecord | None = state.get("failed_qualification") or pending.source_report
    design = await _design_text(workspace, state)
    try:
        calibration = create_calibration(
            original_fingerprint=pending.base_fingerprint,
            calibrated_fingerprint=after,
            accepted_design_sha256=sha256_text(design),
            qualification_failure_sha256=sha256_text(report_text(failed))
            if failed
            else sha256_text(""),
        )
    except ValueError as exc:
        return _failed("qualification_calibration_failed", str(exc), scope="infrastructure")
    return {
        "pending_qualification": pending.model_copy(
            update={
                "calibrated_fingerprint": after,
                "calibration": calibration,
                "source_report": failed,
            }
        ),
        "qualification_failure_stage": "",
    }


# --- publication and failure -------------------------------------------------------------------


MAX_SCANNED_RESOURCE_BYTES = 16 * 1024 * 1024
"""Resource files above this size are not read by the pre-publication credential scan."""


async def _scan_publication_secrets(
    workspace: Workspace, root: str, files: Sequence[str], report: bytes
) -> None:
    """Refuse a scaffold with a credential-like path or content (a pre-publication scan).

    Upstream scans every file below the staged task (``scan_task_publication_secrets``): the planned
    files here, and the files below ``resources/`` up to 16 MiB each (a larger one is a data file
    that the workspace protocol cannot stream; it is not read).
    """
    candidates = list(files)
    resources = await workspace.stat(join(root, "resources"))
    if resources is not None and resources.kind == "dir":
        candidates.extend(
            f"resources/{entry.path}"
            for entry in await workspace.list_tree(join(root, "resources"))
            if entry.kind == "file" and entry.size <= MAX_SCANNED_RESOURCE_BYTES
        )
    for relative in candidates:
        if is_sensitive_path(relative):
            raise PublicationError(f"Refusing to publish sensitive task path: {relative}")
        if contains_secret(await workspace.read_bytes(join(root, relative))):
            raise PublicationError(f"Refusing to publish credential-like content: {relative}")
    if contains_secret(report):
        raise PublicationError(
            "Refusing to publish credential-like content: .agentic_task_validation.json"
        )


async def publish(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Publish the accepted scaffold, the alignment report and the qualification receipt.

    The planned files are scanned for credentials first; publication is idempotent (a resumed
    publication completes what an interrupted one began), and the validation marker that later
    phases read as proof of acceptance is written last.
    """
    ctx = runtime.context
    workspace = ctx.workspace
    paths = _paths(state)
    task: TaskRef = state["task"]
    alignment: AlignmentReport = _need(state, "accepted_alignment")
    qualification: QualificationRecord | None = state.get("accepted_qualification")
    report_bytes = json_text(alignment.model_dump(mode="json", by_alias=True)).encode("utf-8")
    try:
        plan = await build_publication_plan(
            workspace, paths.task, task.root, max_runtime_bytes=_cfg(state).max_runtime_bytes
        )
        receipt_bytes: bytes | None = None
        if qualification is not None:
            if qualification.receipt is None or (
                qualification.receipt.publication_plan_sha256 != plan.sha256
            ):
                raise PublicationError(
                    "task qualification receipt has a different publication plan"
                )
            receipt_bytes = report_text(qualification.receipt).encode("utf-8")
        await _scan_publication_secrets(workspace, paths.task, plan.files, report_bytes)
        await publish_scaffold(
            workspace,
            plan,
            staged_dir=paths.task,
            published_dir=task.root,
            alignment_report=report_bytes,
            qualification_receipt=receipt_bytes,
        )
        published = await published_task_ref(workspace, task, task.root)
        fingerprint = await task_scaffold_fingerprint(workspace, task.root)
    except (PublicationError, ScaffoldError, TaskContractError, OSError) as exc:
        return _failed("publication_failed", str(exc), scope="infrastructure")
    calibration: QualificationCalibration | None = state.get("calibration")
    receipts: dict[str, Any] = {
        "alignment_report": alignment.model_dump(mode="json", by_alias=True),
        "qualification": qualification.model_dump(mode="json") if qualification else None,
        "calibration": calibration.model_dump(mode="json", by_alias=True) if calibration else None,
        "publication_plan_sha256": plan.sha256,
        "design": state.get("accepted_design"),
        "design_review": state.get("accepted_review"),
    }
    return {
        "status": "published",
        "drafting": DraftingResult(
            status="published",
            task=published,
            scaffold_fingerprint=fingerprint,
            receipts=receipts,
        ),
    }


async def fail(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """End a failed drafting: roll back whatever is provisional and report the failure.

    A smoke chain, an alignment repair and a pending qualification repair are each restored to
    their snapshot (upstream: unsafe edits, terminal failures, an exhausted budget and
    interruption restore the original scaffold); the report states why drafting failed.
    """
    workspace = runtime.context.workspace
    update: dict[str, Any] = {}
    chain: DependencyTransaction | None = state.get("dependency_transaction")
    transaction: RepairTransaction | None = state.get("transaction")
    pending: PendingQualificationRepair | None = state.get("pending_qualification")
    for candidate, key in (
        (chain.transaction if chain else None, "dependency_transaction"),
        (transaction, "transaction"),
        (pending.transaction if pending else None, "pending_qualification"),
    ):
        if candidate is not None:
            await restore_transaction(workspace, candidate)
            update[key] = None
    failure = state.get("failure") or Failure(category="drafting_failed", scope="task")
    update["status"] = "failed"
    update["drafting"] = DraftingResult(status="failed", failure=failure)
    return update


# --- the factory -------------------------------------------------------------------------------


def build_task_scaffold_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "task_scaffold",
) -> CompiledStateGraph[ScaffoldState, ScientistContext, ScaffoldInput, ScaffoldOutput]:
    """Build the scaffold drafting graph (upstream phases A2-A7).

    The graph drafts, validates, aligns and qualifies a task scaffold, repairing it within
    upstream's budgets and transactions, and publishes it into the task directory (see the
    module docstring).
    It can run alone on a task snapshot (with or without an accepted design) or be mounted as a
    subgraph of :func:`~langgraph_agentic_scientist.drafting.build_task_drafting_graph`.

    Args:
        checkpointer: The caller's checkpointer; ``None`` inherits the parent's when mounted.
        store: The caller's store; only a top-level caller passes one.
        overrides: Replacement graphs for slots (the scaffold graph has no slots of its own).
        name: The compiled graph's name.

    Returns:
        A compiled graph with input :class:`ScaffoldInput`, state :class:`ScaffoldState` and output
        :class:`ScaffoldOutput`. Hooks run before and after the steps of :data:`SCAFFOLD_STEPS`
        under the keys ``drafting.<step>``. A run takes many supersteps; set ``recursion_limit``
        in the run configuration accordingly.
    """
    g = StepGraph(
        PHASE,
        ScaffoldState,
        input_schema=ScaffoldInput,
        output_schema=ScaffoldOutput,
        overrides=overrides,
    )
    retry = agent_node_retry_policy()

    g.add_node("record_run_config", record_run_config)
    g.step(STEP_PREPARE, prepare)
    g.step(STEP_DRAFT, draft, retry_policy=retry)
    g.step(STEP_INSPECT, inspect_scaffold_node)
    g.step(STEP_CANONICALIZE, canonicalize)
    g.add_node("alignment_boundary_retry", alignment_boundary_retry)
    g.step(STEP_PREFLIGHT, preflight)
    g.add_node("preflight_failed", preflight_failed)
    g.step(STEP_PREFLIGHT_REPAIR, preflight_repair, retry_policy=retry)
    g.add_node("close_preflight_repair", close_preflight_repair)
    g.step(STEP_SMOKE, smoke)
    g.add_node("smoke_failed", smoke_failed)
    g.add_node("smoke_accepted", smoke_accepted)
    g.add_node("open_smoke_transaction", open_smoke_transaction)
    g.step(STEP_SMOKE_REPAIR, smoke_repair, retry_policy=retry)
    g.add_node("close_smoke_repair", close_smoke_repair)
    g.step(STEP_ALIGNMENT_VIEW, alignment_view)
    g.step(STEP_ALIGNMENT_JUDGE, alignment_judge, retry_policy=retry)
    g.step(STEP_CHECK_ALIGNMENT, check_alignment)
    g.add_node("alignment_passed", alignment_passed)
    g.add_node("alignment_failed", alignment_failed)
    g.add_node("open_alignment_transaction", open_alignment_transaction)
    g.step(STEP_ALIGNMENT_REPAIR, alignment_repair, retry_policy=retry)
    g.add_node("close_alignment_repair", close_alignment_repair)
    g.add_node("alignment_validation_failed", alignment_validation_failed)
    g.step(STEP_QUALIFY, qualify)
    g.add_node("qualification_infrastructure_failed", qualification_infrastructure_failed)
    g.add_node("qualification_passed", qualification_passed)
    g.add_node("qualification_failed", qualification_failed)
    g.add_node("qualification_exhausted", qualification_exhausted)
    g.add_node("validate_qualification", validate_qualification)
    g.step(STEP_COMMIT_QUALIFICATION, commit_qualification_repair)
    g.add_node("open_qualification_transaction", open_qualification_transaction)
    g.step(STEP_QUALIFICATION_REPAIR, qualification_repair, retry_policy=retry)
    g.add_node("close_qualification_repair", close_qualification_repair)
    g.add_node("qualification_validation_failed", qualification_validation_failed)
    g.step(STEP_CHECK_CANDIDATE, check_qualification_candidate)
    g.step(STEP_PUBLISH, publish)
    g.add_node("fail", fail)

    g.add_edge(START, "record_run_config")
    g.add_edge("record_run_config", STEP_PREPARE)
    g.add_conditional_edges(
        STEP_PREPARE, _route_failure_or(STEP_DRAFT), {"fail": "fail", STEP_DRAFT: STEP_DRAFT}
    )
    g.add_conditional_edges(
        STEP_DRAFT, _route_failure_or(STEP_INSPECT), {"fail": "fail", STEP_INSPECT: STEP_INSPECT}
    )
    g.add_conditional_edges(
        STEP_INSPECT,
        _route_inspection,
        {"ok": STEP_CANONICALIZE, "missing": STEP_DRAFT, "fail": "fail"},
    )
    g.add_conditional_edges(
        STEP_CANONICALIZE,
        _route_canonicalize,
        {
            "preflight": STEP_PREFLIGHT,
            "fail": "fail",
            "boundary": "alignment_boundary_retry",
            "qualification_validation_failed": "qualification_validation_failed",
        },
    )
    g.add_conditional_edges(
        "alignment_boundary_retry",
        _route_failure_or(STEP_ALIGNMENT_REPAIR),
        {"fail": "fail", STEP_ALIGNMENT_REPAIR: STEP_ALIGNMENT_REPAIR},
    )
    g.add_conditional_edges(
        STEP_PREFLIGHT,
        _route_preflight,
        {
            "smoke": STEP_SMOKE,
            "preflight_repair": STEP_PREFLIGHT_REPAIR,
            "preflight_failed": "preflight_failed",
            "alignment_validation_failed": "alignment_validation_failed",
            "qualification_validation_failed": "qualification_validation_failed",
        },
    )
    g.add_edge("preflight_failed", "fail")
    g.add_conditional_edges(
        STEP_PREFLIGHT_REPAIR,
        _route_failure_or("close_preflight_repair"),
        {"fail": "fail", "close_preflight_repair": "close_preflight_repair"},
    )
    g.add_edge("close_preflight_repair", STEP_CANONICALIZE)
    g.add_conditional_edges(
        STEP_SMOKE,
        _route_smoke,
        {
            "smoke": STEP_SMOKE,
            "smoke_accepted": "smoke_accepted",
            "smoke_failed": "smoke_failed",
            "alignment_validation_failed": "alignment_validation_failed",
            "qualification_validation_failed": "qualification_validation_failed",
            "open_smoke_transaction": "open_smoke_transaction",
        },
    )
    g.add_edge("smoke_failed", "fail")
    g.add_conditional_edges(
        "smoke_accepted",
        _route_smoke_accepted,
        {
            "alignment_view": STEP_ALIGNMENT_VIEW,
            "check_qualification_candidate": STEP_CHECK_CANDIDATE,
        },
    )
    g.add_conditional_edges(
        "open_smoke_transaction",
        _route_failure_or(STEP_SMOKE_REPAIR),
        {"fail": "fail", STEP_SMOKE_REPAIR: STEP_SMOKE_REPAIR},
    )
    g.add_conditional_edges(
        STEP_SMOKE_REPAIR,
        _route_failure_or("close_smoke_repair"),
        {"fail": "fail", "close_smoke_repair": "close_smoke_repair"},
    )
    g.add_conditional_edges(
        "close_smoke_repair",
        _route_failure_or(STEP_CANONICALIZE),
        {"fail": "fail", STEP_CANONICALIZE: STEP_CANONICALIZE},
    )
    g.add_conditional_edges(
        STEP_ALIGNMENT_VIEW,
        _route_failure_or(STEP_ALIGNMENT_JUDGE),
        {"fail": "fail", STEP_ALIGNMENT_JUDGE: STEP_ALIGNMENT_JUDGE},
    )
    g.add_conditional_edges(
        STEP_ALIGNMENT_JUDGE,
        _route_failure_or(STEP_CHECK_ALIGNMENT),
        {"fail": "fail", STEP_CHECK_ALIGNMENT: STEP_CHECK_ALIGNMENT},
    )
    g.add_conditional_edges(
        STEP_CHECK_ALIGNMENT,
        _route_check_alignment,
        {
            "fail": "fail",
            "judge": STEP_ALIGNMENT_JUDGE,
            "passed": "alignment_passed",
            "failed": "alignment_failed",
        },
    )
    g.add_conditional_edges(
        "alignment_passed",
        _route_alignment_passed,
        {"validate_qualification": "validate_qualification", "qualify": STEP_QUALIFY},
    )
    g.add_conditional_edges(
        "alignment_failed",
        _route_alignment_failed,
        {
            "fail": "fail",
            "qualification_budget": "qualification_budget",
            "open_alignment_transaction": "open_alignment_transaction",
        },
    )
    g.add_conditional_edges(
        "open_alignment_transaction",
        _route_failure_or(STEP_ALIGNMENT_REPAIR),
        {"fail": "fail", STEP_ALIGNMENT_REPAIR: STEP_ALIGNMENT_REPAIR},
    )
    g.add_conditional_edges(
        STEP_ALIGNMENT_REPAIR,
        _route_failure_or("close_alignment_repair"),
        {"fail": "fail", "close_alignment_repair": "close_alignment_repair"},
    )
    g.add_edge("close_alignment_repair", STEP_CANONICALIZE)
    g.add_conditional_edges(
        "alignment_validation_failed",
        _route_failure_or(STEP_ALIGNMENT_REPAIR),
        {"fail": "fail", STEP_ALIGNMENT_REPAIR: STEP_ALIGNMENT_REPAIR},
    )
    g.add_conditional_edges(
        STEP_QUALIFY,
        _route_qualify,
        {
            "passed": "qualification_passed",
            "failed": "qualification_failed",
            "retry": STEP_QUALIFY,
            "infrastructure": "qualification_infrastructure_failed",
        },
    )
    g.add_edge("qualification_infrastructure_failed", "fail")
    g.add_conditional_edges(
        "qualification_passed",
        _route_qualification_passed,
        {
            "alignment_view": STEP_ALIGNMENT_VIEW,
            "validate_qualification": "validate_qualification",
        },
    )
    g.add_conditional_edges(
        "validate_qualification",
        _route_validated,
        {"fail": "fail", "commit": STEP_COMMIT_QUALIFICATION, "publish": STEP_PUBLISH},
    )
    g.add_conditional_edges(
        STEP_COMMIT_QUALIFICATION,
        _route_failure_or(STEP_PUBLISH),
        {"fail": "fail", STEP_PUBLISH: STEP_PUBLISH},
    )
    g.add_conditional_edges(
        "qualification_failed",
        _route_qualification_budget,
        {
            "fail": "fail",
            "open_qualification_transaction": "open_qualification_transaction",
            "qualification_exhausted": "qualification_exhausted",
        },
    )
    g.add_node("qualification_budget", _noop)
    g.add_conditional_edges(
        "qualification_budget",
        _route_qualification_budget,
        {
            "fail": "fail",
            "open_qualification_transaction": "open_qualification_transaction",
            "qualification_exhausted": "qualification_exhausted",
        },
    )
    g.add_edge("qualification_exhausted", "fail")
    g.add_conditional_edges(
        "open_qualification_transaction",
        _route_failure_or(STEP_QUALIFICATION_REPAIR),
        {"fail": "fail", STEP_QUALIFICATION_REPAIR: STEP_QUALIFICATION_REPAIR},
    )
    g.add_conditional_edges(
        STEP_QUALIFICATION_REPAIR,
        _route_failure_or("close_qualification_repair"),
        {"fail": "fail", "close_qualification_repair": "close_qualification_repair"},
    )
    g.add_edge("close_qualification_repair", STEP_CANONICALIZE)
    g.add_conditional_edges(
        "qualification_validation_failed",
        _route_qualification_budget,
        {
            "fail": "fail",
            "open_qualification_transaction": "open_qualification_transaction",
            "qualification_exhausted": "qualification_exhausted",
        },
    )
    g.add_conditional_edges(
        STEP_CHECK_CANDIDATE,
        _route_failure_or(STEP_QUALIFY),
        {"fail": "fail", STEP_QUALIFY: STEP_QUALIFY},
    )
    g.add_conditional_edges(STEP_PUBLISH, _route_failure_or(END), {"fail": "fail", END: END})
    g.add_edge("fail", END)
    return g.compile(checkpointer=checkpointer, store=store, name=name)


async def _noop(state: ScaffoldState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """A routing node with no update."""
    return {}


__all__ = [
    "ACCEPTANCE",
    "PHASE",
    "SCAFFOLD_STEPS",
    "SMOKE_RETRYABLE_CATEGORIES",
    "ScaffoldInput",
    "ScaffoldOutput",
    "ScaffoldState",
    "build_task_scaffold_graph",
    "smoke_target_operations",
]
