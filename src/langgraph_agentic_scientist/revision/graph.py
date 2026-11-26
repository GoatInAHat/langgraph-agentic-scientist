"""The revision graph: revise a published paper against its AI review (upstream ``revision.py``).

The method, as upstream defines it: a completed first review is required; the revision writer
edits the manuscript from the review's feedback and from existing evidence only (no experiments,
no new references, no new numbers), through the paper-edit loop with fixed options (references
preserved, figures refreshed as ``auto`` decides, no post-draft reference enrichment); the revised
paper must pass the paper validation, the evidence audit, the retention check against the original
and a method-scope audit before it replaces the published paper; the original is kept under
``revisions/revision_01/before/``; a second review assesses the published revision. A failed edit,
validation or publication retains the original; a failed second review retains the revision and
degrades the phase.

The graph::

    record_run_config -> check_preconditions -> stage_revision -> paper -> retention_check
        -> snapshot_original -> promote -> second_review -> finalize

* ``check_preconditions`` finds the completed first review and checks it can serve as feedback; a
  paper whose revision was already promoted is checked and, if its second review is still
  missing, goes straight to ``second_review`` (upstream's replay);
* ``stage_revision`` fixes the edit options and picks the attempt directory, reusing a saved
  attempt that still passes the checks;
* ``paper`` runs the paper graph in revision mode (slot ``revision.paper``);
* ``retention_check`` checks that the revision keeps the original's evidence and has passing
  audits;
* ``snapshot_original`` and ``promote`` publish it with the original kept (see
  :mod:`~langgraph_agentic_scientist.revision.promotion`);
* ``second_review`` runs the review graph (slot ``review``) on the published paper.

Step names for hooks are ``revision.<node>`` for the nodes above and ``revision.finalize``.
"""

from __future__ import annotations

import importlib
import inspect
import json
import posixpath
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..context import GraphOverrides, ScientistContext
from ..core.render import truncate_utf8
from ..core.results import AIReview, EditOptions, PaperResult, ReviewResult, RevisionResult
from ..core.runconfig import record_run_config
from ..core.state import Event, Failure, FailureScope
from ..core.steps import StepGraph
from ..review import build_review_graph
from ..review.publish import ManifestError, read_manifest, read_review_json, write_manifest
from ..workspace import Workspace, is_file, is_within, join, safe_relpath, write_text
from .evidence import (
    MAX_JSON_BYTES,
    AuditHashes,
    EvidenceError,
    ScopeAudit,
    validate_revision_evidence,
)
from .feedback import (
    MAX_REVIEW_BYTES,
    FeedbackError,
    RevisionFeedback,
    load_revision_feedback,
    render_revision_feedback_json,
    render_revision_feedback_markdown,
)
from .models import (
    SLOT_AUDIT_INPUT_HASHES,
    SLOT_PAPER,
    SLOT_REVIEW,
    SLOT_SCOPE_AUDIT,
    PromotionRecord,
    RetentionReport,
    RevisionInput,
    RevisionOutput,
    RevisionState,
)
from .promotion import (
    AFTER_REVIEW_NAME,
    BEFORE_REVIEW_NAME,
    MANIFEST_NAME,
    MAX_ARTIFACT_BYTES,
    PublicationError,
    initial_review_must_be_preserved,
    names_path,
    promote_revision,
    promoted_original_snapshot,
    read_regular,
    validate_revision_workspace,
    workspace_path,
)
from .summary import revision_review_fields

FAILURE_DETAIL_BYTES = 1200
"""Upstream bounds a failure detail at 1,200 characters."""

MAX_ATTEMPTS = 100
"""Upstream numbers revision attempts ``attempt_01`` to ``attempt_99``."""

PASSTHROUGH_KEYS = (
    "run_config",
    "task",
    "run_id",
    "investigation",
    "downstream",
    "variants",
    "final",
    "posthoc",
    "paper_data",
    "artifact_paths",
)
"""Input keys handed to the paper graph unchanged."""

AFTER_REVIEW_STEM = "ai_review.after_revision"
AFTER_REVIEW_KEY = "after_revision"


# --- helpers ------------------------------------------------------------------------------------


def fixed_edit_options(options: EditOptions | None) -> EditOptions:
    """Return the edit options of a revision: upstream's three fixed values over the caller's.

    Upstream's revision command passes ``--reference-mode preserve``, ``--figure-refresh-mode
    auto`` and ``--skip-postdraft-reference-enrichment`` to the paper edit (``revision.py``).
    """
    return (options or EditOptions()).model_copy(
        update={
            "reference_mode": "preserve",
            "figure_refresh_mode": "auto",
            "skip_postdraft_enrichment": True,
        }
    )


def _failure(category: str, detail: object, *, scope: FailureScope = "task") -> Failure:
    text = truncate_utf8(str(detail), FAILURE_DETAIL_BYTES)
    return Failure(category=category, scope=scope, detail=text)


def _revision_root(state: Mapping[str, Any]) -> str:
    explicit = state.get("revision_root")
    if explicit:
        return safe_relpath(str(explicit), allow_root=False)
    return join(posixpath.dirname(safe_relpath(str(state["paper_dir"]))), "revision")


def _function_slot(overrides: GraphOverrides, key: str) -> Any:
    slot = overrides.slots.get(key)
    if slot is None or isinstance(slot, Runnable):
        return None
    return slot


async def call_slot(
    impl: Any, payload: Any, config: RunnableConfig, runtime: Runtime[ScientistContext]
) -> Any:
    """Run a slot implementation: a compiled graph is invoked as a subgraph of the calling node.

    A callable slot is called as ``impl(payload, runtime)`` and may be a coroutine function.
    """
    if isinstance(impl, Runnable | CompiledStateGraph):
        return await impl.ainvoke(payload, config, context=runtime.context)
    result = impl(payload, runtime)
    return await result if inspect.isawaitable(result) else result


def _typed(output: Any, cls: type[Any], *keys: str) -> Any:
    """Pick a ``cls`` instance from a slot's output (the object itself, or under ``keys``)."""
    if isinstance(output, cls):
        return output
    if isinstance(output, Mapping):
        for key in keys:
            value = output.get(key)
            if isinstance(value, cls):
                return value
            if isinstance(value, Mapping):
                return cls.model_validate(value)
    return None


def _paper_function(module: str, name: str) -> Any:
    """Return a function of the paper package, or ``None`` when the package lacks it."""
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None


def _paper_audit_hashes() -> AuditHashes | None:
    """The paper package's recomputation of an evidence audit's input hashes, if it has one."""
    function = _paper_function("langgraph_agentic_scientist.paper.audit", "audit_input_sha256")
    if function is None:
        return None

    async def hashes(directory: str, runtime: Any) -> Mapping[str, str]:
        return await function(runtime.context.workspace, directory)  # type: ignore[no-any-return]

    return hashes


def _paper_scope_audit() -> ScopeAudit | None:
    """The paper package's validation of a method-scope audit, if it has one."""
    function = _paper_function(
        "langgraph_agentic_scientist.paper.validation.scope_audit", "validate_method_scope_audit_in"
    )
    if function is None:
        return None

    async def valid(directory: str, runtime: Any) -> bool:
        result = await function(runtime.context.workspace, directory)
        return isinstance(result, Mapping) and result.get("ok") is True

    return valid


def _default_paper_graph(overrides: GraphOverrides) -> Any:
    try:
        module = importlib.import_module("langgraph_agentic_scientist.paper")
        factory = module.build_paper_graph
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "build_revision_graph needs the paper graph: install a release that includes "
            f"langgraph_agentic_scientist.paper, or give GraphOverrides.slots[{SLOT_PAPER!r}]"
        ) from exc
    return factory(overrides=overrides, name="revision_paper")


# --- the review file ----------------------------------------------------------------------------


async def require_manifest(workspace: Workspace, paper_dir: str) -> dict[str, Any]:
    """Return the published paper's manifest.

    Raises:
        PublicationError: When it is missing, unreadable or not an object.
    """
    try:
        manifest = await read_manifest(workspace, paper_dir)
    except ManifestError as exc:
        raise PublicationError("published paper manifest is invalid") from exc
    if manifest is None:
        raise PublicationError("published paper manifest is missing")
    return manifest


async def initial_review_path(
    workspace: Workspace, paper_dir: str, review: ReviewResult | None = None
) -> str:
    """Return the immutable first review of a paper (upstream ``_initial_review_path``).

    The named file ``ai_review.before_revision.json`` is the review. When it does not exist and
    nothing requires it to, the latest review ``ai_review.json`` is copied to the named file (with
    its log), and the manifest records it under ``before_revision``.

    Raises:
        PublicationError: When no completed first review is available.
    """
    if review is not None and review.status != "completed":
        raise PublicationError("completed initial AI review is unavailable")
    named = join(paper_dir, BEFORE_REVIEW_NAME)
    if await is_file(workspace, named):
        return named
    if await initial_review_must_be_preserved(workspace, paper_dir):
        raise PublicationError("immutable initial AI review is missing or invalid")
    legacy = join(paper_dir, "ai_review.json")
    if not await is_file(workspace, legacy):
        raise PublicationError("completed initial AI review is unavailable")
    named_log = join(paper_dir, "ai_review.before_revision.log")
    legacy_log = join(paper_dir, "ai_review.log")
    await workspace.write_bytes(
        named, await read_regular(workspace, legacy, max_bytes=MAX_ARTIFACT_BYTES)
    )
    if await is_file(workspace, legacy_log):
        await workspace.write_bytes(
            named_log, await read_regular(workspace, legacy_log, max_bytes=MAX_ARTIFACT_BYTES)
        )
    manifest = await require_manifest(workspace, paper_dir)
    manifest["before_revision"] = {"status": "completed", "review_json": named, "log": named_log}
    await write_manifest(workspace, paper_dir, manifest)
    return named


async def _source_paper_data(workspace: Workspace, paper_dir: str) -> dict[str, Any]:
    """Read the published paper's ``paper_data.json``; ``{}`` when it is missing or unusable."""
    try:
        data = await read_regular(
            workspace, join(paper_dir, "paper_data.json"), max_bytes=MAX_JSON_BYTES
        )
        payload = json.loads(data.decode("utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


async def next_attempt_dir(workspace: Workspace, revision_root: str) -> str:
    """Return the first ``attempt_NN`` directory of ``revision_root`` that does not exist."""
    for attempt in range(1, MAX_ATTEMPTS):
        candidate = join(revision_root, f"attempt_{attempt:02d}")
        if await workspace.stat(candidate) is None:
            return candidate
    raise PublicationError("revision attempt limit reached")


async def reusable_attempt(
    workspace: Workspace,
    paper_dir: str,
    revision_root: str,
    *,
    runtime: Any = None,
    audit_input_hashes: AuditHashes | None = None,
    scope_audit: ScopeAudit | None = None,
) -> str | None:
    """Return a saved attempt that is a valid revision of ``paper_dir``, if there is one."""
    for attempt in range(MAX_ATTEMPTS):
        candidate = join(revision_root, f"attempt_{attempt:02d}")
        entry = await workspace.stat(candidate)
        if entry is None or entry.kind != "dir":
            continue
        try:
            await validate_revision_workspace(workspace, candidate)
            await validate_revision_evidence(
                workspace,
                paper_dir,
                candidate,
                runtime=runtime,
                audit_input_hashes=audit_input_hashes,
                scope_audit=scope_audit,
            )
        except (OSError, UnicodeError, ValueError, RecursionError):
            continue
        return candidate
    return None


async def record_rereview_status(
    workspace: Workspace, paper_dir: str, review: ReviewResult
) -> None:
    """Record the second review's outcome in the promoted manifest (upstream's rereview status)."""
    manifest = await require_manifest(workspace, paper_dir)
    revision_state = manifest.get("revision")
    if not isinstance(revision_state, dict):
        raise PublicationError("published paper manifest is missing revision state")
    revision_state["rereview_status"] = review.status
    revision_state["after_review"] = review.review_path
    if review.failure is not None and review.failure.detail:
        revision_state["rereview_error"] = review.failure.detail
    else:
        revision_state.pop("rereview_error", None)
    await write_manifest(workspace, paper_dir, manifest)


# --- the graph ----------------------------------------------------------------------------------


def build_revision_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "revision",
) -> CompiledStateGraph[RevisionState, ScientistContext, RevisionInput, RevisionOutput]:
    """Build the revision graph.

    The input is a :class:`~langgraph_agentic_scientist.revision.models.RevisionInput` (the
    published paper directory, optionally the review and what the paper graph needs); the output is
    a :class:`~langgraph_agentic_scientist.core.results.RevisionResult` under ``result``. The
    ``revision_writer`` role of the paper graph writes the revision; ``Role.AI_REVIEWER`` and
    ``Role.AREA_CHAIR`` assess it.

    Replaceable parts (``GraphOverrides.slots``):

    * ``revision.paper``: the paper graph in revision mode (default: the package's paper graph);
      it receives a :class:`~langgraph_agentic_scientist.revision.models.RevisionPaperInput`;
    * ``review``: the review graph of the second assessment (default: the package's review graph),
      given a ``ReviewInput``;
    * ``revision.audit_input_hashes`` and ``revision.scope_audit``: async functions
      ``(directory, runtime)`` that recompute an evidence audit's input hashes and validate the
      method-scope audit. The defaults are the paper package's own checks
      (``paper.audit.audit_input_sha256`` and
      ``paper.validation.scope_audit.validate_method_scope_audit_in``); without either the
      evidence check compares the recorded files only and says which parts it did not make.

    Args:
        checkpointer: The checkpointer, when the graph is the top level graph.
        store: The store, when the graph is the top level graph.
        overrides: Replacement graphs and functions.
        name: The graph name (traces nest by it).

    Returns:
        The compiled graph; use ``ainvoke``.
    """
    overrides = overrides or GraphOverrides()
    paper_impl = overrides.slots.get(SLOT_PAPER) or _default_paper_graph(overrides)
    review_impl = overrides.slots.get(SLOT_REVIEW) or build_review_graph(overrides=overrides)
    audit_hashes: AuditHashes | None = (
        _function_slot(overrides, SLOT_AUDIT_INPUT_HASHES) or _paper_audit_hashes()
    )
    scope_audit: ScopeAudit | None = (
        _function_slot(overrides, SLOT_SCOPE_AUDIT) or _paper_scope_audit()
    )

    async def check_preconditions(
        state: RevisionState, runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        workspace = runtime.context.workspace
        paper_dir = safe_relpath(state["paper_dir"], allow_root=False)
        try:
            before = await initial_review_path(workspace, paper_dir, state.get("review"))
            feedback = load_revision_feedback(
                await read_regular(workspace, before, max_bytes=MAX_REVIEW_BYTES)
            )
            manifest = await require_manifest(workspace, paper_dir)
            update: dict[str, Any] = {
                "before_review_path": before,
                "feedback": feedback,
                "replay": "none",
            }
            revision_state = manifest.get("revision")
            if isinstance(revision_state, dict) and revision_state.get("status") == "promoted":
                update.update(
                    await _replay(
                        workspace,
                        paper_dir,
                        _revision_root(state),
                        manifest,
                        revision_state,
                        runtime,
                    )
                )
        except (PublicationError, FeedbackError, EvidenceError, OSError, ValueError) as exc:
            return {"failure": _failure("revision_precondition_failed", exc)}
        return update

    async def _replay(
        workspace: Workspace,
        paper_dir: str,
        revision_root: str,
        manifest: Mapping[str, Any],
        revision_state: Mapping[str, Any],
        runtime: Runtime[ScientistContext],
    ) -> dict[str, Any]:
        snapshot = await promoted_original_snapshot(workspace, paper_dir, revision_state)
        await validate_revision_workspace(workspace, paper_dir)
        await validate_revision_evidence(
            workspace,
            snapshot,
            paper_dir,
            published=True,
            runtime=runtime,
            audit_input_hashes=audit_hashes,
            scope_audit=scope_audit,
        )
        source = revision_state.get("source_dir")
        revision_dir = workspace_path(workspace, source) if isinstance(source, str) else None
        if revision_dir is None:
            raise PublicationError("promoted revision manifest is missing source_dir")
        if not is_within(revision_dir, revision_root):
            raise PublicationError("promoted revision source is outside the revision root")
        await validate_revision_workspace(workspace, revision_dir)
        await validate_revision_evidence(
            workspace,
            snapshot,
            revision_dir,
            runtime=runtime,
            audit_input_hashes=audit_hashes,
            scope_audit=scope_audit,
        )
        replay = "rereview"
        after_state = manifest.get("after_revision")
        expected = join(paper_dir, AFTER_REVIEW_NAME)
        if (
            isinstance(after_state, dict)
            and after_state.get("status") == "completed"
            and names_path(workspace, after_state.get("review_json"), expected)
        ):
            try:
                await read_review_json(workspace, expected)
            except (OSError, ValueError):
                pass
            else:
                replay = "completed"
        return {
            "replay": replay,
            "revision_dir": revision_dir,
            "promotion": PromotionRecord(before_dir=snapshot),
        }

    def route_preconditions(state: Mapping[str, Any]) -> str:
        if state.get("failure") is not None:
            return "failed"
        replay = state.get("replay", "none")
        return {"completed": "done", "rereview": "rereview"}.get(replay, "edit")

    async def stage_revision(
        state: RevisionState, runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        workspace = runtime.context.workspace
        paper_dir = safe_relpath(state["paper_dir"], allow_root=False)
        revision_root = _revision_root(state)
        try:
            reusable = await reusable_attempt(
                workspace,
                paper_dir,
                revision_root,
                runtime=runtime,
                audit_input_hashes=audit_hashes,
                scope_audit=scope_audit,
            )
            attempt = reusable or await next_attempt_dir(workspace, revision_root)
        except (PublicationError, OSError) as exc:
            return {"failure": _failure("revision_staging_failed", exc)}
        return {
            "attempt_dir": attempt,
            "reuse_attempt": reusable is not None,
            "revision_dir": attempt,
            "edit_options": fixed_edit_options(state.get("edit_options")),
        }

    def route_staged(state: Mapping[str, Any]) -> str:
        if state.get("failure") is not None:
            return "failed"
        return "reuse" if state.get("reuse_attempt") else "edit"

    async def paper(
        state: RevisionState, runtime: Runtime[ScientistContext], config: RunnableConfig
    ) -> dict[str, Any]:
        paper_dir = safe_relpath(state["paper_dir"], allow_root=False)
        review_path = state["before_review_path"]
        review = state.get("review")
        if review is None or review.status != "completed":
            review = await _review_result(runtime.context.workspace, review_path)
        payload: dict[str, Any] = {
            "mode": "revision",
            "previous_paper_dir": paper_dir,
            "output_dir": state["attempt_dir"],
            "manifest_dir": state["attempt_dir"],
            "external_review": review,
            "edit_options": fixed_edit_options(state.get("edit_options")),
        }
        feedback = state.get("feedback")
        if feedback is not None:
            payload["revision_feedback"] = feedback
        for key in PASSTHROUGH_KEYS:
            if state.get(key) is not None:
                payload[key] = state[key]  # type: ignore[literal-required]
        # as upstream's revision command: post-hoc analyses and the final evaluation are given to
        # the edit only when the paper being revised carries them
        source = await _source_paper_data(runtime.context.workspace, paper_dir)
        if source.get("posthoc_analyses") in (None, {}, []):
            payload.pop("posthoc", None)
        if source.get("final_evaluation") in (None, {}, []):
            payload.pop("final", None)
        try:
            output = await call_slot(paper_impl, payload, config, runtime)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            return {
                "failure": _failure(
                    "paper_revision_failed",
                    f"paper revision failed: {type(exc).__name__}: {exc}",
                    scope="method",
                )
            }
        result: PaperResult | None = _typed(output, PaperResult, "result", "paper")
        if result is None:
            return {
                "failure": _failure(
                    "paper_revision_failed", "the paper graph returned no paper result"
                )
            }
        update: dict[str, Any] = {"paper": result}
        if result.status != "complete":
            reason = result.incomplete_reason or (result.failure.detail if result.failure else "")
            update["failure"] = _failure(
                "paper_revision_failed",
                f"paper revision did not complete ({result.status})"
                + (f": {reason}" if reason else ""),
                scope="method",
            )
            return update
        update["revision_dir"] = safe_relpath(result.paper_dir or state["attempt_dir"])
        return update

    def route_failed(state: Mapping[str, Any]) -> str:
        return "failed" if state.get("failure") is not None else "ok"

    async def retention_check(
        state: RevisionState, runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        workspace = runtime.context.workspace
        try:
            await validate_revision_workspace(workspace, state["revision_dir"])
            skipped = await validate_revision_evidence(
                workspace,
                safe_relpath(state["paper_dir"], allow_root=False),
                state["revision_dir"],
                runtime=runtime,
                audit_input_hashes=audit_hashes,
                scope_audit=scope_audit,
            )
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            return {
                "failure": _failure(
                    "revision_evidence_invalid",
                    f"paper revision evidence validation failed: {exc}",
                )
            }
        return {"retention": RetentionReport(ok=True, skipped_checks=skipped)}

    async def snapshot_original(
        state: RevisionState, runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        paper_dir = safe_relpath(state["paper_dir"], allow_root=False)
        return {"snapshot": await runtime.context.workspace.snapshot(paper_dir)}

    async def promote(state: RevisionState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        snapshot = state.get("snapshot")
        assert snapshot is not None
        try:
            outcome = await promote_revision(
                runtime.context.workspace,
                paper_dir=state["paper_dir"],
                revision_dir=state["revision_dir"],
                before_review_path=state["before_review_path"],
                original=snapshot,
                runtime=runtime,
                audit_input_hashes=audit_hashes,
                scope_audit=scope_audit,
            )
        except (OSError, UnicodeError, ValueError, RecursionError) as exc:
            return {"failure": _failure("revision_publication_failed", exc, scope="infrastructure")}
        record = PromotionRecord(
            before_dir=outcome.before_dir,
            archived=outcome.archived,
            skipped_checks=outcome.skipped_checks,
        )
        return {"promotion": record}

    async def second_review(
        state: RevisionState, runtime: Runtime[ScientistContext], config: RunnableConfig
    ) -> dict[str, Any]:
        paper_dir = safe_relpath(state["paper_dir"], allow_root=False)
        payload = {
            "pdf_path": join(paper_dir, "paper.pdf"),
            "paper_dir": paper_dir,
            "artifact_stem": AFTER_REVIEW_STEM,
            "manifest_key": AFTER_REVIEW_KEY,
            "run_config": state.get("run_config"),
        }
        try:
            output = await call_slot(review_impl, payload, config, runtime)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            result = ReviewResult(
                status="failed",
                failure=_failure("second_review_failed", f"{type(exc).__name__}: {exc}"),
            )
        else:
            result = _typed(output, ReviewResult, "result", "review") or ReviewResult(
                status="failed",
                failure=_failure("second_review_failed", "the review graph returned no result"),
            )
        return {"second_review": result}

    async def finalize(state: RevisionState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        workspace = runtime.context.workspace
        paper_dir = safe_relpath(state["paper_dir"], allow_root=False)
        failure = state.get("failure")
        promotion = state.get("promotion")
        before_dir = promotion.before_dir if promotion is not None else None
        attempted = state.get("paper")
        if failure is not None:
            result = RevisionResult(
                status="failed", paper=attempted, before_dir=before_dir, failure=failure
            )
            event = Event(
                kind="revision_result",
                phase="revision",
                step="finalize",
                data={"status": "failed", "failure": failure.category},
            )
            return {"result": result, "events": [event]}

        second = state.get("second_review")
        if state.get("replay") == "completed":
            second = await _review_result(workspace, join(paper_dir, AFTER_REVIEW_NAME))
        status: Literal["promoted", "degraded"] = "promoted"
        problem: Failure | None = None
        if second is None or second.status != "completed":
            status = "degraded"
            second = second or ReviewResult(status="skipped")
            problem = second.failure or _failure("second_review_failed", "second AI review failed")
        if state.get("replay") != "completed":
            try:
                await record_rereview_status(workspace, paper_dir, second)
            except (PublicationError, OSError, ValueError) as exc:
                status = "degraded"
                problem = problem or _failure("manifest_update_failed", exc)
        published = _published(attempted, paper_dir, state.get("replay") == "completed")
        result = RevisionResult(
            status=status,
            paper=published,
            second_review=second,
            before_dir=before_dir,
            failure=problem,
        )
        fields = await _review_fields(workspace, state.get("before_review_path"), second)
        event = Event(
            kind="revision_result",
            phase="revision",
            step="finalize",
            data={"status": status, "before_dir": before_dir, **fields},
        )
        return {"result": result, "events": [event]}

    sg = StepGraph(
        "revision", RevisionState, input_schema=RevisionInput, output_schema=RevisionOutput
    )
    sg.add_node("record_run_config", record_run_config)
    sg.step("check_preconditions", check_preconditions)
    sg.step("stage_revision", stage_revision, hook_writable=("edit_options",))
    sg.step("paper", paper)
    sg.slots.append(SLOT_PAPER)
    sg.step("retention_check", retention_check)
    sg.step("snapshot_original", snapshot_original)
    sg.step("promote", promote)
    sg.step("second_review", second_review)
    sg.slots.append(SLOT_REVIEW)
    sg.step("finalize", finalize)

    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "check_preconditions")
    sg.add_conditional_edges(
        "check_preconditions",
        route_preconditions,
        {
            "failed": "finalize",
            "done": "finalize",
            "rereview": "second_review",
            "edit": "stage_revision",
        },
    )
    sg.add_conditional_edges(
        "stage_revision",
        route_staged,
        {"failed": "finalize", "reuse": "retention_check", "edit": "paper"},
    )
    sg.add_conditional_edges("paper", route_failed, {"failed": "finalize", "ok": "retention_check"})
    sg.add_conditional_edges(
        "retention_check", route_failed, {"failed": "finalize", "ok": "snapshot_original"}
    )
    sg.add_edge("snapshot_original", "promote")
    sg.add_conditional_edges("promote", route_failed, {"failed": "finalize", "ok": "second_review"})
    sg.add_edge("second_review", "finalize")
    sg.add_edge("finalize", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


# --- finalize helpers ---------------------------------------------------------------------------


async def _review_result(workspace: Workspace, review_path: str) -> ReviewResult:
    """Build the external review that drives the paper edit from the first review's file."""
    try:
        payload = await read_review_json(workspace, review_path)
        aggregate: AIReview | None = AIReview.model_validate(payload)
    except (OSError, ValueError):
        aggregate = None
    return ReviewResult(status="completed", aggregate=aggregate, review_path=review_path)


def _published(attempted: PaperResult | None, paper_dir: str, replayed: bool) -> PaperResult:
    """The revised paper as published: its files are the paper directory's."""
    base = attempted or PaperResult(status="complete")
    return base.model_copy(
        update={
            "status": "complete",
            "paper_dir": paper_dir,
            "pdf_path": join(paper_dir, "paper.pdf"),
            "tex_path": join(paper_dir, "paper.tex"),
            "manifest_path": join(paper_dir, MANIFEST_NAME),
            "failure": None,
            "incomplete_reason": None,
        }
    )


async def _review_fields(
    workspace: Workspace, before_path: str | None, second: ReviewResult | None
) -> dict[str, Any]:
    async def load(path: str | None) -> dict[str, Any] | None:
        if not path:
            return None
        try:
            return await read_review_json(workspace, path)
        except (OSError, ValueError):
            return None

    after = second.review_path if second is not None and second.status == "completed" else None
    return dict(revision_review_fields(await load(before_path), await load(after)))


async def write_feedback_files(
    workspace: Workspace, directory: str, feedback: RevisionFeedback
) -> None:
    """Write ``revision_feedback.json`` and ``revision_feedback.md`` into ``directory``.

    A paper graph that stages its own feedback does not need this; it is the same rendering as
    upstream's, for one that is given the feedback as data
    (:class:`~langgraph_agentic_scientist.revision.models.RevisionPaperInput`).
    """
    await write_text(
        workspace,
        join(directory, "revision_feedback.json"),
        render_revision_feedback_json(feedback),
    )
    await write_text(
        workspace,
        join(directory, "revision_feedback.md"),
        render_revision_feedback_markdown(feedback),
    )


__all__ = [
    "build_revision_graph",
    "call_slot",
    "fixed_edit_options",
    "initial_review_path",
    "next_attempt_dir",
    "record_rereview_status",
    "require_manifest",
    "reusable_attempt",
    "write_feedback_files",
]
