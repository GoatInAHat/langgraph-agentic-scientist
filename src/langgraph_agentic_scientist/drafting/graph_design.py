"""The task design loop as a LangGraph graph (upstream phase A1).

Upstream's ``run_design_workflow`` (``src/scripts/lib/task_design.sh``) turns the task into a
plan-only design before any scaffold exists: a designer writes a TaskDesign v1, a judge reviews it
against eleven dimensions, a patch repairer fixes what failed with replace-only patches confined
to the repair paths the judge authorised, and a delta auditor decides whether a repair that fixed
something while the review found new failures caused them. The loop keeps a *repair frontier*
(the best design so far) and a *repair memory* of rejected regressions, rotates the focus onto one
dimension after two rounds without improvement, and stops after ``design_max_repairs`` semantic
repairs.

In the port every step is a graph node and every retry is a graph loop with its counter in state,
so each agent session has its own checkpoint namespace and a resumed run never repeats a
completed session (design decisions D1 and D6):

* ``design_generate``: the task designer (R1) answers with a :class:`TaskDesign`.
* ``validate_design``: upstream's loader rules; a violation sends the design to
  ``design_structure_repair`` (R2), at most ``design_max_structure_repairs`` times, which does
  not spend the semantic budget.
* ``design_review``: the judge (R3), one session per attempt. ``check_review`` applies the
  review contract; an invalid review is retried up to ``design_review_attempts`` times, each
  retry's prompt carrying the "# Retry feedback" section of every failed attempt.
* ``decide``: a passing review publishes the design (``design_publish``); a spent budget fails;
  otherwise the repair feedback is built (repair memory, protected dimensions, the previous
  candidate's progress) and, after ``design_non_improving_before_focus`` rounds without
  improvement, narrowed to one dimension.
* ``design_patch_repair`` (R4) and ``check_patch``: an invalid patch is retried up to
  ``design_format_attempts`` times with the "# Format retry feedback" section and spends nothing
  of the semantic budget.
* ``begin_focused_review``, ``design_review``, ``check_review``, ``compare_reviews``: the focused
  review of the candidate (the judge prompt plus "# Focused repair verification"); when every
  parent failure is cleared a full review follows (``begin_full_review``).
* ``design_delta_audit`` (R5) and ``check_delta``: up to ``design_delta_attempts`` attempts.
* ``promote_or_reject``: promotion, a quarantined frontier advance, or rejection with repair
  memory and latent-finding rebinding; then ``decide`` again.

One judge reviews, as upstream's driver runs. The three-judge quorum and its adjudicator are
dormant upstream and are not built.

Every limit is read through :func:`~langgraph_agentic_scientist.core.runconfig.method_config`.
The files that upstream's prompts name are written into per-session directories of the design
directory, so the prompts stay true: ``task.md``, ``task_design.json``, ``parent_review.json``,
``parent_design.json``, ``changed_paths.json``, ``parent_task_design.json``,
``candidate_review.json`` and the answer file of the role that was answered.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.tools import BaseTool
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..agents.runner import agent_node_retry_policy, run_structured_role
from ..agents.tools import make_file_sha256_tool
from ..context import GraphOverrides, ScientistContext
from ..core.render import json_text
from ..core.runconfig import method_config, record_run_config
from ..core.state import Failure, RunConfigRecord, TaskRef, append, merge_dicts
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.paths import join
from ..workspace.protocol import Workspace, write_text
from .design import (
    TaskDesignError,
    apply_design_patch,
    check_delta,
    cleared_parent_failures,
    compare_reviews,
    enrich_progress,
    focus_repair_payload,
    focused_authorization,
    has_only_indirect_regressions,
    rebind_latent_failures,
    regressions_were_already_failed,
    review_failure_payload,
    target_dimensions_text,
    task_design_changed_paths,
    update_repair_memory,
    validate_design,
    validate_review,
    validation_failure_payload,
)
from .design_models import (
    Capacity,
    DeltaAudit,
    DesignLedgerEntry,
    DesignPatch,
    DesignProgress,
    DesignReview,
    RepairMemory,
    TaskDesign,
)

PHASE = "drafting"
"""The phase name of the hook keys (``drafting.<step>``)."""

STEP_GENERATE = "design_generate"
STEP_STRUCTURE_REPAIR = "design_structure_repair"
STEP_REVIEW = "design_review"
STEP_PATCH_REPAIR = "design_patch_repair"
STEP_DELTA_AUDIT = "design_delta_audit"
STEP_PUBLISH = "design_publish"
DESIGN_STEPS = (
    STEP_GENERATE,
    STEP_STRUCTURE_REPAIR,
    STEP_REVIEW,
    STEP_PATCH_REPAIR,
    STEP_DELTA_AUDIT,
    STEP_PUBLISH,
)
"""The steps that run a role or publish, and so carry hook nodes (``drafting.<step>.before`` and
``.after``)."""

TASK_MD = "./task.md"
TASK_DESIGN_JSON = "./task_design.json"
TASK_DESIGN_REVIEW_JSON = "./task_design_review.json"
TASK_DESIGN_PATCH_JSON = "./task_design_patch.json"
PARENT_DESIGN_JSON = "./parent_task_design.json"
PARENT_REVIEW_JSON = "./parent_review.json"
CANDIDATE_REVIEW_JSON = "./candidate_review.json"
FORMAT_DIAGNOSTIC_BYTES = 4096
"""Upstream keeps the first 4096 bytes of each of the patch tool's two outputs."""
FORMAT_FEEDBACK_CHARS = 8192
"""The part of the diagnostic that the format retry prompt carries."""


class DesignInput(TypedDict):
    """Input of the design graph.

    Attributes:
        task: The run-local task snapshot (its ``root`` holds ``task.md``).
        capacity: The investigation capacity the design must fit; the configured investigation
            limits when absent.
        design_dir: Workspace directory for the design's files; ``_task_design`` next to the
            task snapshot when absent (upstream ``<output>/_task_design``).
        run_config: The recorded configuration of a parent run, when mounted.
    """

    task: TaskRef
    capacity: NotRequired[Capacity]
    design_dir: NotRequired[str]
    run_config: NotRequired[RunConfigRecord | None]


class DesignOutput(TypedDict):
    """Output of the design graph.

    Attributes:
        status: ``accepted`` (the design passed an independent review) or ``failed``.
        design: The accepted design (or the last design, when the loop failed).
        accepted_review: The review that accepted it.
        receipt: Upstream's ``design_receipt.json`` content (design and review digests).
        design_dir: Where the files were written.
        ledger: Upstream's candidate ledger.
        failure: Why the loop failed.
    """

    status: Literal["running", "accepted", "failed"]
    design: TaskDesign | None
    accepted_review: DesignReview | None
    receipt: dict[str, Any] | None
    design_dir: str
    ledger: list[DesignLedgerEntry]
    failure: Failure | None


class DesignState(TypedDict, total=False):
    """State of the design graph (see the module docstring for the steps that use each key)."""

    task: TaskRef
    capacity: Capacity | None
    design_dir: str
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    # the design under validation and the repair frontier
    design: TaskDesign | None
    design_diagnostic: str | None
    fingerprint: str | None
    structure_repairs: int
    semantic_repairs: int
    non_improving: int
    focus_history: list[str]
    best_review: DesignReview | None
    repair_base: TaskDesign | None
    repair_memory: RepairMemory
    last_progress: DesignProgress | None
    # a review in flight
    review_stage: Literal["initial", "focused", "full"] | None
    review_round: int
    review_design: TaskDesign | None
    review_attempts: int
    review_feedback: list[str]
    review_report: DesignReview | None
    review_failure: str | None
    review_valid: bool
    # a repair in flight
    decision: Literal["accept", "fail", "repair"] | None
    repair_feedback: str | None
    repair_review: DesignReview | None
    patch_round: int
    format_attempts: int
    patch_diagnostic: str | None
    candidate_patch: DesignPatch | None
    patch_failure: str | None
    candidate: TaskDesign | None
    candidate_review: DesignReview | None
    progress: DesignProgress | None
    full_failed: bool
    promotion_allowed: bool
    delta_required: bool
    delta_attempts: int
    delta_feedback: str | None
    delta_audit: DeltaAudit | None
    delta_failure: str | None
    delta_retry: bool
    delta_completed: bool
    frontier_allowed: bool
    frontier_continuation_allowed: bool
    # the record
    ledger: Annotated[list[DesignLedgerEntry], append]
    status: Literal["running", "accepted", "failed"]
    accepted_review: DesignReview | None
    receipt: dict[str, Any] | None
    failure: Failure | None


# --- helpers ------------------------------------------------------------------------------------


def _design_dir(state: Mapping[str, Any]) -> str:
    explicit = state.get("design_dir")
    if explicit:
        return join(str(explicit))
    parent = PurePosixPath(state["task"].root).parent.as_posix()
    return join(parent if parent != "." else "", "_task_design")


def _capacity(state: Mapping[str, Any]) -> Capacity:
    capacity = state.get("capacity")
    if capacity is not None:
        return capacity if isinstance(capacity, Capacity) else Capacity.model_validate(capacity)
    investigation = method_config(state).investigation
    return Capacity(
        max_idea_cycles=investigation.max_idea_cycles,
        max_program_evals_per_attempt=investigation.max_program_evals_per_attempt,
    )


async def _task_md(workspace: Workspace, state: Mapping[str, Any]) -> bytes:
    return await workspace.read_bytes(join(state["task"].root, "task.md"))


async def _write_json(workspace: Workspace, path: str, payload: Any) -> None:
    await write_text(workspace, path, json_text(payload))


async def _stage(
    workspace: Workspace, session: str, state: Mapping[str, Any], files: Mapping[str, Any]
) -> None:
    """Write ``task.md`` and the given JSON payloads into a role's session directory."""
    await workspace.write_bytes(join(session, "task.md"), await _task_md(workspace, state))
    for name, payload in files.items():
        await _write_json(workspace, join(session, name), payload)


def _failed(category: str, detail: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "failure": Failure(category=category, scope="method", detail=detail[:4096]),
    }


def _is_failed(state: Mapping[str, Any]) -> bool:
    return state.get("status") == "failed"


def _tools(ctx: ScientistContext, session: str) -> list[BaseTool]:
    return [make_file_sha256_tool(ctx.workspace, session, prompts=ctx.prompts)]


def _session_failure(failure: Failure | None, fallback: str) -> str:
    if failure is None:
        return fallback
    return f"{fallback}: {failure.category}: {failure.detail}"


async def _write_ledger(workspace: Workspace, design_dir: str, state: Mapping[str, Any]) -> None:
    entries = state.get("ledger") or []
    if entries:
        await write_text(
            workspace,
            join(design_dir, "candidate_ledger.jsonl"),
            "".join(entry.line() + "\n" for entry in entries),
        )


def _entry(event: str, payload: Mapping[str, Any]) -> DesignLedgerEntry:
    return DesignLedgerEntry(event=event, payload=dict(payload))


# --- generation and structural repair -----------------------------------------------------------


async def design_generate(state: DesignState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run the task designer (R1): answer with a TaskDesign v1 for the canonical task."""
    ctx = runtime.context
    capacity = _capacity(state)
    design_dir = _design_dir(state)
    session = join(design_dir, "generate")
    await _stage(ctx.workspace, session, state, {})
    outcome = await run_structured_role(
        role=Role.TASK_DESIGNER,
        schema=TaskDesign,
        runtime=runtime,
        root=session,
        read_only=True,
        prompt_key="drafting/design_task",
        values={
            "TASK_MD_PATH": TASK_MD,
            "TASK_DESIGN_PATH": TASK_DESIGN_JSON,
            "MAX_IDEA_CYCLES": str(capacity.max_idea_cycles),
            "MAX_PROGRAM_EVALS_PER_ATTEMPT": str(capacity.max_program_evals_per_attempt),
        },
        fragments=False,
        tools=_tools(ctx, session),
    )
    if not isinstance(outcome.structured, TaskDesign):
        return _failed(
            "task_design_generation_failed",
            _session_failure(outcome.failure, "the task designer produced no design"),
        )
    design = outcome.structured
    await _write_json(ctx.workspace, join(design_dir, "task_design.json"), design.payload())
    return {"design_dir": design_dir, "design": design, "status": "running"}


async def validate_design_node(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Validate the current design with upstream's loader rules (``validate-design``)."""
    design = state.get("design")
    if design is None:
        return _failed("task_design_structure_invalid", "there is no design to validate")
    task_md = await _task_md(runtime.context.workspace, state)
    try:
        fingerprint = validate_design(design, task_md)
    except TaskDesignError as exc:
        diagnostic = f"invalid task design artifact: {exc}"
        used = state.get("structure_repairs", 0)
        limit = method_config(state).drafting.design_max_structure_repairs
        if used >= limit:
            return {
                "design_diagnostic": diagnostic,
                **_failed(
                    "task_design_structure_invalid",
                    f"Task design remained structurally invalid after {limit} structural repair "
                    f"attempt(s): {diagnostic}",
                ),
            }
        return {"design_diagnostic": diagnostic, "fingerprint": None}
    return {"design_diagnostic": None, "fingerprint": fingerprint}


async def design_structure_repair(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the design structure repairer (R2); it never spends the semantic budget."""
    ctx = runtime.context
    used = state.get("structure_repairs", 0)
    design = state.get("design")
    feedback = validation_failure_payload(used, state.get("design_diagnostic") or "")
    session = join(_design_dir(state), f"structure_repair_{used}")
    current = design.payload() if design is not None else {}
    await _stage(ctx.workspace, session, state, {"task_design.json": current})
    outcome = await run_structured_role(
        role=Role.DESIGN_STRUCTURE_REPAIRER,
        schema=TaskDesign,
        runtime=runtime,
        root=session,
        read_only=True,
        prompt_key="drafting/repair_task_design_structure",
        values={
            "TASK_MD_PATH": TASK_MD,
            "TASK_DESIGN_PATH": TASK_DESIGN_JSON,
            "TASK_DESIGN_REVIEW": json.dumps(feedback, ensure_ascii=False, sort_keys=True),
        },
        fragments=False,
        tools=_tools(ctx, session),
    )
    update: dict[str, Any] = {"structure_repairs": used + 1}
    if isinstance(outcome.structured, TaskDesign):
        update["design"] = outcome.structured
        await _write_json(
            ctx.workspace,
            join(_design_dir(state), "task_design.json"),
            outcome.structured.payload(),
        )
    return update


def route_validation(state: DesignState) -> str:
    """Route after validation: review, structural repair, or the end."""
    if _is_failed(state):
        return "fail"
    return "repair" if state.get("design_diagnostic") else "review"


# --- reviews ------------------------------------------------------------------------------------


def _review_reset() -> dict[str, Any]:
    return {
        "review_attempts": 0,
        "review_feedback": [],
        "review_report": None,
        "review_failure": None,
        "review_valid": False,
    }


async def begin_initial_review(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Start the first (round 0) review of the generated design."""
    return {
        "review_stage": "initial",
        "review_round": 0,
        "review_design": state["design"],
        "semantic_repairs": state.get("semantic_repairs", 0),
        "non_improving": state.get("non_improving", 0),
        "focus_history": list(state.get("focus_history") or []),
        "repair_memory": state.get("repair_memory") or RepairMemory(),
        **_review_reset(),
    }


async def begin_focused_review(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Start the focused review of a repair candidate against its parent."""
    return {
        "review_stage": "focused",
        "review_round": state.get("review_round", 0) + 1,
        "review_design": state["candidate"],
        **_review_reset(),
    }


async def begin_full_review(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Start the full review of a candidate whose focused review cleared every parent failure."""
    return {
        "review_stage": "full",
        "review_round": state.get("review_round", 0) + 1,
        "review_design": state["candidate"],
        **_review_reset(),
    }


async def design_review(state: DesignState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run the design judge (R3): one session, one review attempt.

    The session directory holds ``task.md`` and ``task_design.json``; a focused review also holds
    the parent's review and design and the harness-derived change manifest. The prompt carries
    the "# Focused repair verification" section of a focused review and one "# Retry feedback"
    section for every earlier attempt whose review failed its contract.
    """
    ctx = runtime.context
    design = state["review_design"]
    assert design is not None
    stage = state["review_stage"]
    review_round = state.get("review_round", 0)
    fingerprint = design.fingerprint()
    session = join(_design_dir(state), f"review_{review_round}_a")
    files: dict[str, Any] = {"task_design.json": design.payload()}
    suffix = ""
    if stage == "focused":
        parent = state["repair_base"]
        parent_review = state["best_review"]
        assert parent is not None and parent_review is not None
        changed = list(task_design_changed_paths(parent, design))
        if not changed:
            return _failed(
                "task_design_changed_paths_invalid", "Could not derive task design changed paths"
            )
        files.update(
            {
                "parent_review.json": parent_review.payload(),
                "parent_design.json": parent.payload(),
                "changed_paths.json": {"schema_version": 1, "paths": changed},
            }
        )
        suffix += ctx.prompts.render(
            "workflow/drafting/focused_repair_verification",
            {"TARGET_DIMENSIONS": target_dimensions_text(parent_review)},
            fragments=False,
        )
    for diagnostic in state.get("review_feedback") or []:
        suffix += ctx.prompts.render(
            "workflow/drafting/review_retry_feedback", {"DIAGNOSTIC": diagnostic}, fragments=False
        )
    await _stage(ctx.workspace, session, state, files)
    outcome = await run_structured_role(
        role=Role.DESIGN_JUDGE,
        schema=DesignReview,
        runtime=runtime,
        root=session,
        read_only=True,
        prompt_key="drafting/validate_task_design",
        values={
            "TASK_MD_PATH": TASK_MD,
            "TASK_DESIGN_PATH": TASK_DESIGN_JSON,
            "TASK_DESIGN_REVIEW_PATH": TASK_DESIGN_REVIEW_JSON,
            "TASK_DESIGN_FINGERPRINT": fingerprint,
        },
        fragments=False,
        suffix=suffix,
    )
    if isinstance(outcome.structured, DesignReview):
        await _write_json(
            ctx.workspace, join(session, "task_design_review.json"), outcome.structured.payload()
        )
        return {"review_report": outcome.structured, "review_failure": None}
    return {
        "review_report": None,
        "review_failure": _session_failure(
            outcome.failure, "task design review must be a structured response"
        ),
    }


async def check_review(state: DesignState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Apply the review contract (``validate-review --require-schema-version 3``).

    A valid review, passing or failing, becomes the frontier's review (round 0) or the
    candidate's review. An invalid one is retried up to ``design_review_attempts`` attempts in
    all.
    """
    design = state["review_design"]
    assert design is not None
    report = state.get("review_report")
    diagnostic: str | None = None
    if report is None:
        diagnostic = (
            state.get("review_failure") or "task design review must be a structured response"
        )
    else:
        try:
            validate_review(report, design)
        except TaskDesignError as exc:
            diagnostic = f"invalid task design artifact: {exc}"
    if diagnostic is not None:
        attempts = state.get("review_attempts", 0) + 1
        if attempts >= method_config(state).drafting.design_review_attempts:
            return {
                "review_attempts": attempts,
                **_failed(
                    "task_design_review_invalid",
                    f"Independent task design judge produced an invalid report: {diagnostic}",
                ),
            }
        return {
            "review_valid": False,
            "review_attempts": attempts,
            "review_feedback": [*(state.get("review_feedback") or []), diagnostic],
        }
    assert report is not None
    if state["review_stage"] == "initial":
        return {
            "review_valid": True,
            "best_review": report,
            "repair_base": design,
            "fingerprint": design.fingerprint(),
            "ledger": [_entry("reviewed", report.payload())],
        }
    update: dict[str, Any] = {"review_valid": True, "candidate_review": report}
    if state["review_stage"] == "full":
        update["ledger"] = [_entry("full_review", report.payload())]
    return update


def route_review_check(state: DesignState) -> str:
    """Route after ``check_review``: retry an invalid review, else decide (round 0) or compare."""
    if _is_failed(state):
        return "fail"
    if not state.get("review_valid"):
        return "retry"
    return "decide" if state.get("review_stage") == "initial" else "compare"


# --- the decision, the patch ----------------------------------------------------------------------


async def decide(state: DesignState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """The head of upstream's decision loop.

    A passing frontier review is accepted. A spent semantic budget fails. Otherwise the repair
    feedback is built from the frontier's failed review and, after
    ``design_non_improving_before_focus`` rounds without improvement, narrowed to one dimension
    chosen by rotation; the repair authority of a focused round is that dimension's repair paths
    only.
    """
    best = state["best_review"]
    assert best is not None
    repairs = state.get("semantic_repairs", 0)
    cfg = method_config(state).drafting
    reset: dict[str, Any] = {
        "candidate": None,
        "candidate_review": None,
        "candidate_patch": None,
        "progress": None,
        "full_failed": False,
        "promotion_allowed": False,
        "delta_required": False,
        "delta_completed": False,
        "frontier_allowed": False,
        "frontier_continuation_allowed": False,
        "delta_audit": None,
        "delta_attempts": 0,
        "delta_feedback": None,
        "delta_failure": None,
        "delta_retry": False,
        "patch_failure": None,
        "patch_diagnostic": None,
        "format_attempts": 0,
        **_review_reset(),
    }
    if best.verdict == "pass":
        return {"decision": "accept", **reset}
    if repairs >= cfg.design_max_repairs:
        return {
            "decision": "fail",
            **reset,
            **_failed(
                "task_design_not_accepted",
                f"Task design still failed after {cfg.design_max_repairs} semantic repair "
                "attempt(s)",
            ),
        }
    feedback = review_failure_payload(
        best, repairs, state.get("last_progress"), state.get("repair_memory")
    )
    repair_review = best
    focus_history = list(state.get("focus_history") or [])
    if state.get("non_improving", 0) >= cfg.design_non_improving_before_focus:
        feedback, focused_dimension = focus_repair_payload(feedback, focus_history)
        focus_history.append(focused_dimension)
        repair_review = focused_authorization(best, feedback)
    return {
        "decision": "repair",
        "repair_feedback": json.dumps(feedback, ensure_ascii=False, sort_keys=True),
        "repair_review": repair_review,
        "focus_history": focus_history,
        "patch_round": repairs,
        **reset,
    }


def route_decision(state: DesignState) -> str:
    """Route after ``decide``: publish, repair, or end."""
    if _is_failed(state):
        return "fail"
    return "publish" if state.get("decision") == "accept" else "repair"


async def design_patch_repair(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the design patch repairer (R4): one session, one patch attempt.

    The prompt carries the "# Format retry feedback" section when the previous patch of this
    round failed its contract.
    """
    ctx = runtime.context
    base = state["repair_base"]
    assert base is not None
    attempt = state.get("format_attempts", 0)
    name = f"repair_{state.get('patch_round', 0)}" + (f"_format_{attempt}" if attempt else "")
    session = join(_design_dir(state), name)
    await _stage(ctx.workspace, session, state, {"task_design.json": base.payload()})
    suffix = ""
    if state.get("patch_diagnostic"):
        suffix = ctx.prompts.render(
            "workflow/drafting/patch_format_retry_feedback",
            {"DIAGNOSTIC": (state.get("patch_diagnostic") or "")[:FORMAT_FEEDBACK_CHARS]},
            fragments=False,
        )
    outcome = await run_structured_role(
        role=Role.DESIGN_PATCH_REPAIRER,
        schema=DesignPatch,
        runtime=runtime,
        root=session,
        read_only=True,
        prompt_key="drafting/repair_task_design",
        values={
            "TASK_MD_PATH": TASK_MD,
            "TASK_DESIGN_PATH": TASK_DESIGN_JSON,
            "TASK_DESIGN_FINGERPRINT": base.fingerprint(),
            "TASK_DESIGN_PATCH_PATH": TASK_DESIGN_PATCH_JSON,
            "TASK_DESIGN_REVIEW": state.get("repair_feedback") or "",
        },
        fragments=False,
        suffix=suffix,
    )
    if isinstance(outcome.structured, DesignPatch):
        await _write_json(
            ctx.workspace, join(session, "task_design_patch.json"), outcome.structured.payload()
        )
        return {"candidate_patch": outcome.structured, "patch_failure": None}
    return {
        "candidate_patch": None,
        "patch_failure": _session_failure(
            outcome.failure, "the repairer produced no structured patch"
        ),
    }


async def check_patch(state: DesignState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Validate the patch and apply it into a candidate design (upstream ``apply-patch``).

    An invalid patch is retried up to ``design_format_attempts`` attempts in all and spends
    nothing of the semantic budget; after the last the loop fails, as upstream's does.
    """
    base = state["repair_base"]
    repair_review = state["repair_review"]
    assert base is not None and repair_review is not None
    patch = state.get("candidate_patch")
    diagnostic: str | None = None
    candidate: TaskDesign | None = None
    if patch is None:
        diagnostic = state.get("patch_failure") or "task design patch must be a structured response"
    else:
        try:
            validate_review(repair_review, base)
            candidate = apply_design_patch(
                patch,
                parent=base,
                canonical_task=await _task_md(runtime.context.workspace, state),
                repair_review=repair_review,
            )
        except TaskDesignError as exc:
            diagnostic = f"invalid task design artifact: {exc}\n"
    if candidate is not None:
        return {"candidate": candidate, "patch_diagnostic": None}
    assert diagnostic is not None
    attempts = state.get("format_attempts", 0) + 1
    cut = diagnostic.encode("utf-8")[:FORMAT_DIAGNOSTIC_BYTES].decode("utf-8", errors="ignore")
    if attempts >= method_config(state).drafting.design_format_attempts:
        progress = DesignProgress(diagnostic=cut)
        return {
            "format_attempts": attempts,
            "non_improving": state.get("non_improving", 0) + 1,
            "last_progress": progress,
            "ledger": [_entry("repair_failed", progress.payload())],
            **_failed(
                "task_design_patch_invalid",
                f"Task design repair artifact remained invalid after {attempts} format "
                f"attempt(s): {cut}",
            ),
        }
    return {"format_attempts": attempts, "patch_diagnostic": cut}


def route_patch(state: DesignState) -> str:
    """Route after ``check_patch``: review the candidate, retry the patch, or end."""
    if _is_failed(state):
        return "fail"
    return "review" if state.get("candidate") is not None else "retry"


# --- comparing the candidate ----------------------------------------------------------------------


async def compare_reviews_node(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Compare the candidate's review with the frontier's (``compare-reviews``) and decide.

    After the focused review: a fix with every parent failure cleared leads to a full review; a
    fix with failures left leads to the delta audit; no fix rejects the candidate. After the full
    review: an improvement that passes promotes the candidate, one that still fails needs the
    delta audit, and a non-improvement needs it only if something was fixed or the full review
    failed with parent failures left.
    """
    base = state["repair_base"]
    best = state["best_review"]
    candidate = state["candidate"]
    candidate_review = state["candidate_review"]
    assert base and best and candidate and candidate_review
    progress = enrich_progress(
        compare_reviews(best, candidate_review, parent_design=base, candidate_design=candidate),
        candidate_review,
    )
    update: dict[str, Any] = {"progress": progress}
    if state["review_stage"] == "focused":
        if progress.fixed:
            if cleared_parent_failures(progress):
                update["delta_required"] = False
            else:
                update["delta_required"] = True
        return update
    full_failed = candidate_review.verdict != "pass"
    update["full_failed"] = full_failed
    if progress.improved:
        if not full_failed:
            update["promotion_allowed"] = True
        else:
            update["delta_required"] = True
    elif progress.fixed or (full_failed and progress.still_failed):
        update["delta_required"] = True
    return update


def route_compare(state: DesignState) -> str:
    """Route after a comparison: full review, delta audit, or promotion or rejection."""
    progress = state["progress"]
    assert progress is not None
    if state["review_stage"] == "focused" and progress.fixed and cleared_parent_failures(progress):
        return "full"
    if state.get("delta_required"):
        return "audit"
    return "settle"


# --- the delta audit ------------------------------------------------------------------------------


async def design_delta_audit(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the design delta auditor (R5): one session, one audit attempt.

    The session directory holds the parent and candidate designs and reviews. The retry prompt
    carries the "# Retry feedback" section for the previous attempt's failure only, as upstream
    builds each attempt's prompt from the base prompt.
    """
    ctx = runtime.context
    base = state["repair_base"]
    best = state["best_review"]
    candidate = state["candidate"]
    candidate_review = state["candidate_review"]
    assert base and best and candidate and candidate_review
    session = join(_design_dir(state), f"delta_{state.get('review_round', 0)}")
    await _stage(
        ctx.workspace,
        session,
        state,
        {
            "parent_task_design.json": base.payload(),
            "task_design.json": candidate.payload(),
            "parent_review.json": best.payload(),
            "candidate_review.json": candidate_review.payload(),
        },
    )
    suffix = ""
    if state.get("delta_feedback"):
        suffix = ctx.prompts.render(
            "workflow/drafting/delta_retry_feedback",
            {"DIAGNOSTIC": state.get("delta_feedback") or ""},
            fragments=False,
        )
    outcome = await run_structured_role(
        role=Role.DESIGN_DELTA_AUDITOR,
        schema=DeltaAudit,
        runtime=runtime,
        root=session,
        read_only=True,
        prompt_key="drafting/audit_task_design_delta",
        values={
            "TASK_MD_PATH": TASK_MD,
            "TASK_DESIGN_PATH": TASK_DESIGN_JSON,
            "TASK_DESIGN_REVIEW_PATH": TASK_DESIGN_REVIEW_JSON,
            "TASK_DESIGN_FINGERPRINT": candidate.fingerprint(),
            "TASK_PARENT_DESIGN_PATH": PARENT_DESIGN_JSON,
            "TASK_PARENT_REVIEW_PATH": PARENT_REVIEW_JSON,
            "TASK_CANDIDATE_REVIEW_PATH": CANDIDATE_REVIEW_JSON,
        },
        fragments=False,
        suffix=suffix,
    )
    if isinstance(outcome.structured, DeltaAudit):
        await _write_json(
            ctx.workspace, join(session, "task_design_review.json"), outcome.structured.payload()
        )
        return {"delta_audit": outcome.structured, "delta_failure": None}
    return {
        "delta_audit": None,
        "delta_failure": _session_failure(
            outcome.failure, "task design delta review must be a structured response"
        ),
    }


async def check_delta_node(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Apply the delta contract (``validate-delta``) and classify the audited candidate.

    A pass promotes the candidate. A fail with only indirect regressions advances the quarantined
    frontier; a fail whose regressions were already failing in the parent, after a failed full
    review, continues it. An audit that stays invalid after ``design_delta_attempts`` attempts
    leaves the candidate to be rejected.
    """
    base = state["repair_base"]
    best = state["best_review"]
    candidate = state["candidate"]
    candidate_review = state["candidate_review"]
    progress = state["progress"]
    assert base and best and candidate and candidate_review and progress
    delta = state.get("delta_audit")
    diagnostic: str | None = None
    if delta is None:
        diagnostic = state.get("delta_failure") or "task design delta review must be structured"
    else:
        try:
            check_delta(
                delta,
                parent=base,
                candidate=candidate,
                parent_review=best,
                candidate_review=candidate_review,
            )
        except TaskDesignError as exc:
            diagnostic = f"invalid task design artifact: {exc}"
    if diagnostic is not None:
        attempts = state.get("delta_attempts", 0) + 1
        if attempts >= method_config(state).drafting.design_delta_attempts:
            failed = progress.model_copy(deep=True)
            failed.improved = False
            failed.audit_failure = "delta audit invocation or artifact validation failed"
            return {
                "delta_attempts": attempts,
                "delta_retry": False,
                "delta_completed": False,
                "progress": failed,
            }
        return {"delta_attempts": attempts, "delta_retry": True, "delta_feedback": diagnostic}
    assert delta is not None
    consensus = progress
    enriched = enrich_progress(consensus, candidate_review, delta)
    update: dict[str, Any] = {
        "delta_retry": False,
        "delta_completed": True,
        "progress": enriched,
        "ledger": [_entry("delta_review", enriched.payload())],
    }
    full_failed = bool(state.get("full_failed"))
    if delta.verdict == "pass":
        update["promotion_allowed"] = True
    elif has_only_indirect_regressions(
        enriched,
        parent=base,
        candidate=candidate,
        parent_review=best,
        candidate_review=candidate_review,
        delta=delta,
    ):
        update["frontier_allowed"] = True
    elif full_failed and enriched.still_failed and regressions_were_already_failed(enriched):
        update["frontier_continuation_allowed"] = True
    return update


def route_delta(state: DesignState) -> str:
    """Route after ``check_delta``: another attempt, or promotion or rejection."""
    return "retry" if state.get("delta_retry") else "settle"


# --- promotion or rejection -----------------------------------------------------------------------


async def promote_or_reject(
    state: DesignState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Promote the candidate to the repair frontier, or reject it; then the loop goes on.

    Promotion (a passing delta audit) and the two quarantined frontier advances set the new
    repair base and reset the non-improving counter and the focus history; a quarantined advance
    keeps its progress as the next repair's feedback and adds the rejected regressions to the
    repair memory. A rejected candidate raises the non-improving counter, adds its regressions to
    the repair memory and, when the full review failed and the delta audit completed, rebinds
    unchanged full-review findings to the retained parent (``latent_review``). Every outcome
    spends one semantic repair.
    """
    base = state["repair_base"]
    best = state["best_review"]
    candidate = state["candidate"]
    candidate_review = state["candidate_review"]
    progress = state["progress"]
    assert base and best and candidate and candidate_review and progress
    memory = state.get("repair_memory") or RepairMemory()
    completed = bool(state.get("delta_completed"))
    update: dict[str, Any] = {"semantic_repairs": state.get("semantic_repairs", 0) + 1}
    ledger: list[DesignLedgerEntry] = []
    promoted = False
    last_progress: DesignProgress | None = None
    if state.get("promotion_allowed"):
        promoted = True
        ledger.append(_entry("promoted", progress.payload()))
    elif state.get("frontier_allowed") and completed:
        promoted = True
        memory = update_repair_memory(memory, progress)
        last_progress = progress
        ledger.append(_entry("frontier_advanced", progress.payload()))
    elif state.get("frontier_continuation_allowed") and completed:
        promoted = True
        memory = update_repair_memory(memory, progress)
        last_progress = progress
        ledger.append(_entry("frontier_continued", progress.payload()))
    if promoted:
        update.update(
            {
                "repair_base": candidate,
                "best_review": candidate_review,
                "fingerprint": candidate.fingerprint(),
                "non_improving": 0,
                "focus_history": [],
                "last_progress": last_progress,
            }
        )
    else:
        memory = update_repair_memory(memory, progress)
        update.update(
            {"non_improving": state.get("non_improving", 0) + 1, "last_progress": progress}
        )
        ledger.append(_entry("rejected", progress.payload()))
        delta = state.get("delta_audit")
        if state.get("full_failed") and completed and delta is not None:
            consensus = compare_reviews(
                best, candidate_review, parent_design=base, candidate_design=candidate
            )
            rebound = rebind_latent_failures(
                parent=base,
                candidate=candidate,
                parent_review=best,
                candidate_review=candidate_review,
                delta=delta,
                consensus_regressions=consensus.regressed,
            )
            if rebound is not None:
                try:
                    validate_review(rebound, base)
                except TaskDesignError as exc:
                    return _failed(
                        "task_design_review_invalid",
                        f"Could not validate rebound parent review: {exc}",
                    )
                update["best_review"] = rebound
                ledger.append(_entry("latent_findings_rebound", rebound.payload()))
    update["repair_memory"] = memory
    update["ledger"] = ledger
    return update


def route_settle(state: DesignState) -> str:
    """Route after promotion or rejection: back to ``decide``, or the end on failure."""
    return "fail" if _is_failed(state) else "decide"


# --- the end --------------------------------------------------------------------------------------


async def design_publish(state: DesignState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Publish the accepted design, its review and the receipt (upstream ``design_receipt.json``).

    Writes ``task_design.json``, ``accepted_review.json``, ``design_receipt.json`` and the
    candidate ledger into the design directory.
    """
    workspace = runtime.context.workspace
    design_dir = _design_dir(state)
    design = state["repair_base"]
    review = state["best_review"]
    assert design is not None and review is not None
    design_text = json_text(design.payload())
    review_text = json_text(review.payload())
    await write_text(workspace, join(design_dir, "task_design.json"), design_text)
    await write_text(workspace, join(design_dir, "accepted_review.json"), review_text)
    receipt = {
        "schema_version": 1,
        "design_sha256": hashlib.sha256(design_text.encode("utf-8")).hexdigest(),
        "review_sha256": hashlib.sha256(review_text.encode("utf-8")).hexdigest(),
        "verdict": "pass",
    }
    await write_text(
        workspace,
        join(design_dir, "design_receipt.json"),
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    )
    await _write_ledger(workspace, design_dir, state)
    return {
        "status": "accepted",
        "design_dir": design_dir,
        "design": design,
        "accepted_review": review,
        "receipt": receipt,
        "failure": None,
    }


async def fail_design(state: DesignState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """The end of a failed loop: keep the ledger as a file and report the failure."""
    design_dir = _design_dir(state)
    await _write_ledger(runtime.context.workspace, design_dir, state)
    return {"status": "failed", "design_dir": design_dir}


# --- the factory ----------------------------------------------------------------------------------


def build_task_design_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "task_design",
) -> CompiledStateGraph[DesignState, ScientistContext, DesignInput, DesignOutput]:
    """Build the task design graph (upstream phase A1).

    The graph turns ``task.md`` into an accepted TaskDesign v1: generate, validate (structural
    repair), review, and repair until an independent review passes or a bound is reached (see the
    module docstring). It can run alone or be mounted as a subgraph of the task drafting graph.

    Args:
        checkpointer: The caller's checkpointer; ``None`` inherits the parent's when mounted.
        store: The caller's store; only a top-level caller passes one.
        overrides: Replacement graphs (the design graph has no slots of its own).
        name: The compiled graph's name.

    Returns:
        A compiled graph with input :class:`DesignInput`, state :class:`DesignState` and output
        :class:`DesignOutput`. Limits come from ``ScientistConfig.drafting``: ``design_max_repairs``
        (8), ``design_max_structure_repairs`` (3), ``design_review_attempts`` (3),
        ``design_format_attempts`` (3), ``design_delta_attempts`` (3) and
        ``design_non_improving_before_focus`` (2). Hooks run before and after the steps listed in
        :data:`DESIGN_STEPS`, under the keys ``drafting.<step>``.
    """
    g = StepGraph(
        PHASE,
        DesignState,
        input_schema=DesignInput,
        output_schema=DesignOutput,
        overrides=overrides,
    )
    retry = agent_node_retry_policy()
    g.add_node("record_run_config", record_run_config)
    g.step(STEP_GENERATE, design_generate, retry_policy=retry)
    g.add_node("validate_design", validate_design_node)
    g.step(STEP_STRUCTURE_REPAIR, design_structure_repair, retry_policy=retry)
    g.add_node("begin_initial_review", begin_initial_review)
    g.step(STEP_REVIEW, design_review, retry_policy=retry)
    g.add_node("check_review", check_review)
    g.add_node("decide", decide)
    g.step(STEP_PATCH_REPAIR, design_patch_repair, retry_policy=retry)
    g.add_node("check_patch", check_patch)
    g.add_node("begin_focused_review", begin_focused_review)
    g.add_node("begin_full_review", begin_full_review)
    g.add_node("compare_reviews", compare_reviews_node)
    g.step(STEP_DELTA_AUDIT, design_delta_audit, retry_policy=retry)
    g.add_node("check_delta", check_delta_node)
    g.add_node("promote_or_reject", promote_or_reject)
    g.step(STEP_PUBLISH, design_publish)
    g.add_node("fail_design", fail_design)

    g.add_edge(START, "record_run_config")
    g.add_edge("record_run_config", STEP_GENERATE)
    g.add_conditional_edges(
        STEP_GENERATE,
        lambda state: "fail" if _is_failed(state) else "validate",
        {"fail": "fail_design", "validate": "validate_design"},
    )
    g.add_conditional_edges(
        "validate_design",
        route_validation,
        {"fail": "fail_design", "repair": STEP_STRUCTURE_REPAIR, "review": "begin_initial_review"},
    )
    g.add_edge(STEP_STRUCTURE_REPAIR, "validate_design")
    g.add_edge("begin_initial_review", STEP_REVIEW)
    g.add_conditional_edges(
        STEP_REVIEW,
        lambda state: "fail" if _is_failed(state) else "check",
        {"fail": "fail_design", "check": "check_review"},
    )
    g.add_conditional_edges(
        "check_review",
        route_review_check,
        {
            "fail": "fail_design",
            "retry": STEP_REVIEW,
            "decide": "decide",
            "compare": "compare_reviews",
        },
    )
    g.add_conditional_edges(
        "decide",
        route_decision,
        {"fail": "fail_design", "publish": STEP_PUBLISH, "repair": STEP_PATCH_REPAIR},
    )
    g.add_edge(STEP_PATCH_REPAIR, "check_patch")
    g.add_conditional_edges(
        "check_patch",
        route_patch,
        {"fail": "fail_design", "retry": STEP_PATCH_REPAIR, "review": "begin_focused_review"},
    )
    g.add_edge("begin_focused_review", STEP_REVIEW)
    g.add_edge("begin_full_review", STEP_REVIEW)
    g.add_conditional_edges(
        "compare_reviews",
        route_compare,
        {"full": "begin_full_review", "audit": STEP_DELTA_AUDIT, "settle": "promote_or_reject"},
    )
    g.add_edge(STEP_DELTA_AUDIT, "check_delta")
    g.add_conditional_edges(
        "check_delta",
        route_delta,
        {"retry": STEP_DELTA_AUDIT, "settle": "promote_or_reject"},
    )
    g.add_conditional_edges(
        "promote_or_reject",
        route_settle,
        {"fail": "fail_design", "decide": "decide"},
    )
    g.add_edge(STEP_PUBLISH, END)
    g.add_edge("fail_design", END)
    return g.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "DESIGN_STEPS",
    "PHASE",
    "DesignInput",
    "DesignOutput",
    "DesignState",
    "build_task_design_graph",
    "check_delta_node",
    "check_patch",
    "check_review",
    "decide",
    "design_delta_audit",
    "design_generate",
    "design_patch_repair",
    "design_publish",
    "design_review",
    "design_structure_repair",
    "promote_or_reject",
    "validate_design_node",
]
