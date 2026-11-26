"""The writer stage (R30, R35, R38) and the post-draft reference enrichment (R31, R32).

The writer stage runs one agent session per loop round, as upstream's ``run_prompt_stage_with_
feedback`` does: the stage is the draft (round 0), a manuscript repair (the same writer prompt;
the repair context is in the feedback files of the workspace), or the targeted figure-quality
repair. The session's role is the paper writer, the revision writer in revision mode, or the
figure-quality repairer. A session that fails, or leaves a required output missing or empty, is a
failed stage: upstream then writes stage-execution feedback into ``validation_feedback.*``.

The post-draft enrichment (``run_postdraft_reference_enrichment``) runs once after the first
draft: the draft-gap analyst (repair rounds on invalid gaps), and when it reports ``ready`` the
post-draft reference planner (repair rounds with the reference repair context), retrieval of the
new references, their merge into the bibliography, and a "citation weave" run of the writer
prompt. Its failure ends the paper phase. It is the replaceable slot
``paper.postdraft_enrichment``; :func:`build_postdraft_graph` is its default.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired, TypedDict

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
from ..context import GraphOverrides, ScientistContext
from ..core.interfaces import ReferenceQuery, ReferenceRecord
from ..core.runconfig import method_config
from ..core.state import Event, Failure, RunConfigRecord, TaskRef, merge_dicts
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.paths import join
from .contexts import (
    render_stage_prompt,
    write_json,
    write_stage_execution_feedback,
    write_text,
)
from .models import DraftGapsAnswer, PaperMode, Stage

FRESH_OUTPUTS = ("paper.tex",)
EDIT_OUTPUTS = (
    "paper.tex",
    "claims_grounding.json",
    "method_scope_audit.json",
    "method_scope_audit.md",
)
POSTDRAFT_MARKER = ".postdraft_reference_enrichment_complete"


def expected_outputs(mode: str) -> tuple[str, ...]:
    """The outputs a writer stage must leave non-empty (``paper.sh`` vs ``paper_edit.sh``)."""
    return FRESH_OUTPUTS if mode == "fresh" else EDIT_OUTPUTS


def stage_prompt_name(stage: Stage, repair_round: int) -> str:
    """The prompt artefact upstream keeps for the stage."""
    if stage == "figure_quality_repair":
        return "paper_figure_quality_repair_prompt.md"
    if stage == "draft":
        return "paper_prompt.md"
    return f"paper_repair_prompt_round_{repair_round}.md"


def stage_name(stage: Stage, repair_round: int) -> str:
    """Upstream's ``CURRENT_PAPER_STAGE_NAME``."""
    if stage == "figure_quality_repair":
        return "paper_figure_quality_repair"
    if stage == "draft":
        return "paper"
    return f"paper_repair_round_{repair_round}"


def writer_role(mode: str, stage: Stage) -> Role:
    """The role of a writer stage."""
    if stage == "figure_quality_repair":
        return Role.FIGURE_QUALITY_REPAIRER
    if mode == "revision":
        return Role.REVISION_WRITER
    return Role.PAPER_WRITER


async def missing_outputs(
    ctx: ScientistContext, paper_dir: str, outputs: Sequence[str]
) -> list[str]:
    """The required outputs that are missing or empty."""
    missing: list[str] = []
    for name in outputs:
        entry = await ctx.workspace.stat(join(paper_dir, name))
        if entry is None or entry.kind != "file" or entry.size <= 0:
            missing.append(name)
    return missing


def available_skills(ctx: ScientistContext, names: Sequence[str]) -> list[str]:
    """The skills of ``names`` that the registry ships (others cannot be staged)."""
    shipped = set(ctx.prompts.skill_names())
    return [name for name in names if name in shipped]


async def run_writer_session(
    state: Mapping[str, Any],
    runtime: Runtime[ScientistContext],
    *,
    stage: Stage,
    prompt_key: str,
    prompt_name: str,
    name: str,
    feedback: tuple[str, str] = ("validation_feedback.json", "validation_feedback.md"),
) -> tuple[bool, Failure | None, RoleOutcome]:
    """Run one writer-type session and check its outputs (``run_prompt_stage_with_feedback``).

    Returns ``(ok, failure, outcome)``. On failure the stage-execution feedback is written to
    ``feedback`` (``validation_feedback.*`` for the writer).
    """
    ctx = runtime.context
    cfg = method_config(state).paper
    paper_dir = state["paper_dir"]
    mode = state.get("mode", "fresh")
    prompt = await render_stage_prompt(state, runtime, prompt_key)
    prompt_path = join(paper_dir, prompt_name)
    await write_text(ctx.workspace, prompt_path, prompt)
    skills = available_skills(ctx, state.get("selected_skills") or [])
    if cfg.review_rounds > 0 and "review-paper" not in skills:
        skills += available_skills(ctx, ["review-paper"])
    outcome = await run_workspace_role(
        role=writer_role(mode, stage),
        runtime=runtime,
        root=paper_dir,
        prompt=prompt,
        skills=skills,
        timeout=cfg.timeout_seconds,
        stage=False,  # staged (and watermarked) when the paper directory was prepared
    )
    outputs = expected_outputs(mode)
    missing = await missing_outputs(ctx, paper_dir, outputs)
    if outcome.ok and not missing:
        return True, None, outcome
    detail = (
        outcome.failure.detail
        if outcome.failure is not None
        else ("the session ran out of its wall clock" if outcome.timed_out else "")
    )
    await write_stage_execution_feedback(
        ctx.workspace,
        paper_dir,
        feedback_json=feedback[0],
        feedback_md=feedback[1],
        stage_name=name,
        failure_detail=detail or "the session failed",
        prompt_path=prompt_path,
        expected_outputs=outputs,
        log_text=outcome.final_text,
        missing_outputs_failure=outcome.ok and bool(missing),
    )
    category = "stage_outputs_missing" if outcome.ok else "stage_failed"
    failure = Failure(
        category=category,
        scope="method",
        detail=f"stage '{name}' "
        + (f"left {', '.join(missing)} missing or empty" if outcome.ok else f"failed: {detail}"),
    )
    return False, failure, outcome


# --- post-draft enrichment ---------------------------------------------------------------------


class PostdraftInput(TypedDict):
    """What the post-draft enrichment reads from the paper loop."""

    run_config: RunConfigRecord | None
    task: TaskRef
    paper_dir: str
    mode: NotRequired[PaperMode]
    writer_prompt_key: NotRequired[str]
    writer_skill: NotRequired[str]
    selected_skills: NotRequired[list[str]]
    paper_format: NotRequired[str]
    two_stage: NotRequired[bool]
    repair_round: NotRequired[int]


PostdraftStatus = Literal["done", "no_gaps", "failed"]


class PostdraftOutput(TypedDict):
    """What the post-draft enrichment returns."""

    postdraft_status: PostdraftStatus
    postdraft_detail: str
    events: list[Event]


class PostdraftState(PostdraftInput, total=False):
    """The enrichment's own state."""

    postdraft_status: PostdraftStatus
    postdraft_detail: str
    events: Annotated[list[Event], operator.add]
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    gaps_round: int
    gaps_ok: bool
    gaps_ready: bool
    plan_round: int
    plan_ok: bool
    queries: list[str]
    retrieval_ok: bool


async def _gaps(state: PostdraftState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    from .references.gaps import write_draft_reference_gaps_validation_feedback

    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).paper
    paper_dir = state["paper_dir"]
    round_ = int(state.get("gaps_round") or 0)
    name = (
        "draft_reference_gaps_prompt.md"
        if round_ == 0
        else f"draft_reference_gaps_repair_prompt_round_{round_}.md"
    )
    prompt = await render_stage_prompt(state, runtime, "paper/paper_reference_gaps")
    await write_text(ws, join(paper_dir, name), prompt)
    outcome = await run_structured_role(
        role=Role.DRAFT_GAP_ANALYST,
        schema=DraftGapsAnswer,
        runtime=runtime,
        root=paper_dir,
        read_only=True,
        prompt=prompt,
        timeout=cfg.timeout_seconds,
    )
    if not isinstance(outcome.structured, DraftGapsAnswer):
        await write_stage_execution_feedback(
            ws,
            paper_dir,
            feedback_json="draft_reference_gaps_validation_feedback.json",
            feedback_md="draft_reference_gaps_validation_feedback.md",
            stage_name=name.removesuffix(".md").replace("_prompt", ""),
            failure_detail=outcome.failure.detail if outcome.failure else "no structured answer",
            prompt_path=join(paper_dir, name),
            expected_outputs=("draft_reference_gaps.md", "draft_reference_gaps.json"),
            log_text=outcome.final_text,
        )
        return {"gaps_ok": False, "gaps_ready": False}
    answer = outcome.structured
    await write_text(ws, join(paper_dir, "draft_reference_gaps.md"), answer.memo)
    gaps = answer.gaps.model_dump(mode="json")
    await write_json(ws, join(paper_dir, "draft_reference_gaps.json"), gaps)
    feedback = await write_draft_reference_gaps_validation_feedback(ws, paper_dir)
    ok = bool(feedback.get("ok"))
    return {"gaps_ok": ok, "gaps_ready": ok and str(gaps.get("status", "")).strip() == "ready"}


def _after_gaps(state: PostdraftState) -> str:
    if state.get("gaps_ok"):
        return "plan" if state.get("gaps_ready") else "no_gaps"
    round_ = int(state.get("gaps_round") or 0)
    if round_ >= method_config(state).paper.max_repair_rounds:
        return "failed_gaps"
    return "repair_gaps"


async def _repair_gaps(state: PostdraftState) -> dict[str, Any]:
    return {"gaps_round": int(state.get("gaps_round") or 0) + 1}


async def _no_gaps(state: PostdraftState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    await runtime.context.workspace.write_bytes(join(state["paper_dir"], POSTDRAFT_MARKER), b"")
    return {
        "postdraft_status": "no_gaps",
        "postdraft_detail": (
            "draft gap analysis found no targeted citation gaps; skipping postdraft retrieval"
        ),
    }


async def _failed_gaps(state: PostdraftState) -> dict[str, Any]:
    round_ = int(state.get("gaps_round") or 0)
    return {
        "postdraft_status": "failed",
        "postdraft_detail": (
            f"exhausted draft reference gap repair rounds after {round_} attempts; see "
            f"{join(state['paper_dir'], 'draft_reference_gaps_validation_feedback.json')}"
        ),
    }


async def _plan(state: PostdraftState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    from .references.diagnostics import reference_repair_context
    from .references.plan import (
        ReferencePlanAnswer,
        reference_query_texts,
        write_plan_files,
        write_reference_query_plan_validation_feedback,
    )

    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).paper
    paper_dir = state["paper_dir"]
    stem = "postdraft_reference_query_plan"
    round_ = int(state.get("plan_round") or 0)
    prompt = await render_stage_prompt(state, runtime, "paper/reference_query_plan_postdraft")
    if round_ > 0:
        prompt += await reference_repair_context(ws, paper_dir, stem, prompts=ctx.prompts)
    name = f"{stem}_prompt.md" if round_ == 0 else f"{stem}_repair_prompt_round_{round_}.md"
    await write_text(ws, join(paper_dir, name), prompt)
    outcome = await run_structured_role(
        role=Role.POSTDRAFT_REFERENCE_PLANNER,
        schema=ReferencePlanAnswer,
        runtime=runtime,
        root=paper_dir,
        read_only=True,
        prompt=prompt,
        timeout=cfg.timeout_seconds,
    )
    if not isinstance(outcome.structured, ReferencePlanAnswer):
        await write_stage_execution_feedback(
            ws,
            paper_dir,
            feedback_json=f"{stem}_validation_feedback.json",
            feedback_md=f"{stem}_validation_feedback.md",
            stage_name=name.removesuffix(".md").replace("_prompt", ""),
            failure_detail=outcome.failure.detail if outcome.failure else "no structured answer",
            prompt_path=join(paper_dir, name),
            expected_outputs=(f"{stem}.md", f"{stem}.json"),
            log_text=outcome.final_text,
        )
        return {"plan_ok": False}
    plan = outcome.structured.plan.model_dump(mode="json")
    await write_plan_files(ws, paper_dir, stem=stem, memo=outcome.structured.memo, plan=plan)
    feedback = await write_reference_query_plan_validation_feedback(
        ws, paper_dir, stem=stem, max_queries=cfg.max_queries
    )
    ok = bool(feedback.get("ok"))
    return {"plan_ok": ok, "queries": reference_query_texts(plan) if ok else []}


async def _check_plan(state: PostdraftState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    from .references.diagnostics import record_reference_round

    repeated = await record_reference_round(
        runtime.context.workspace,
        state["paper_dir"],
        "postdraft_reference_query_plan",
        int(state.get("plan_round") or 0),
    )
    if repeated:
        return {
            "postdraft_status": "failed",
            "postdraft_detail": "Reference plan and validation errors unchanged after repair",
        }
    return {}


def _after_check_plan(state: PostdraftState) -> str:
    if state.get("postdraft_status") == "failed":
        return "end"
    if state.get("plan_ok"):
        return "retrieve"
    if int(state.get("plan_round") or 0) >= method_config(state).paper.max_repair_rounds:
        return "failed_plan"
    return "repair_plan"


async def _repair_plan(state: PostdraftState) -> dict[str, Any]:
    return {"plan_round": int(state.get("plan_round") or 0) + 1}


async def _failed_plan(state: PostdraftState) -> dict[str, Any]:
    round_ = int(state.get("plan_round") or 0)
    return {
        "postdraft_status": "failed",
        "postdraft_detail": (
            f"exhausted postdraft reference query plan repair rounds after {round_} attempts; see "
            f"{join(state['paper_dir'], 'postdraft_reference_query_plan_validation_feedback.json')}"
        ),
    }


async def retrieve_records(
    ctx: ScientistContext, queries: Sequence[str], *, paper_dir: str, min_refs: int, max_refs: int
) -> tuple[list[ReferenceRecord], str | None]:
    """Search every query with the context's retriever, deduplicated, up to ``max_refs``; ask
    the retriever's fallbacks when too few were found. Returns the records and an error."""
    from .references.graph import SEARCH_LIMIT_PER_QUERY, FallbackRetriever
    from .references.semantic_scholar import SemanticScholarRetriever

    retriever = ctx.retriever or SemanticScholarRetriever()
    records: list[ReferenceRecord] = []
    seen: set[str] = set()
    errors: list[str] = []
    for text in queries:
        if len(records) >= max_refs:
            break
        try:
            found = await retriever.search(
                ReferenceQuery(query=text, limit=SEARCH_LIMIT_PER_QUERY), workspace=ctx.workspace
            )
        except Exception as exc:
            errors.append(f"{text}: {exc}")
            continue
        for record in found:
            key = (record.doi or record.key or record.title).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
            if len(records) >= max_refs:
                break
    if len(records) < min_refs and isinstance(retriever, FallbackRetriever):
        outcome = await retriever.fallback(
            list(queries),
            min_refs=min_refs,
            max_refs=max_refs,
            paper_dir=paper_dir,
            workspace=ctx.workspace,
            s2_error="; ".join(errors) or None,
            s2_count=len(records),
        )
        for record in getattr(outcome, "records", []) or []:
            key = (record.doi or record.key or record.title).strip().lower()
            if key not in seen and len(records) < max_refs:
                seen.add(key)
                records.append(record)
    if not records:
        return [], "postdraft reference retrieval found no references: " + (
            "; ".join(errors) or "no results"
        )
    if len(records) < min_refs:
        return records, (
            f"postdraft reference retrieval found {len(records)} references < required {min_refs}"
        )
    return records, None


async def _retrieve(state: PostdraftState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    from .references.merge import merge_reference_artifacts
    from .references.retrieve import reference_provenance_payload, write_reference_outputs

    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).paper
    paper_dir = state["paper_dir"]
    for source, target in (
        ("references.bib", "references.pre_postdraft.bib"),
        ("references_provenance.json", "references.pre_postdraft.json"),
    ):
        if await ws.exists(join(paper_dir, source)):
            await ws.write_bytes(
                join(paper_dir, target), await ws.read_bytes(join(paper_dir, source))
            )
    records, error = await retrieve_records(
        ctx,
        state.get("queries") or [],
        paper_dir=paper_dir,
        min_refs=cfg.min_refs,
        max_refs=cfg.max_refs,
    )
    if error is not None:
        return {
            "postdraft_status": "failed",
            "postdraft_detail": f"postdraft reference retrieval failed: {error}",
        }
    payload = reference_provenance_payload(
        records,
        generated_at=None,
        results_dir="./results",
        campaign_results_dir="./results",
        queries=list(state.get("queries") or []),
        plan_metadata={},
    )
    try:
        await write_reference_outputs(
            ws,
            paper_dir,
            records,
            payload,
            bib_name="postdraft_references.bib",
            provenance_name="postdraft_references_provenance.json",
        )
        merged = await merge_reference_artifacts(
            ws, paper_dir, results_dir="./results", campaign_results_dir="./results"
        )
    except ValueError as exc:
        return {
            "postdraft_status": "failed",
            "postdraft_detail": f"postdraft reference retrieval failed: {exc}",
        }
    return {
        "retrieval_ok": True,
        "events": [
            Event(
                kind="postdraft_references_merged",
                phase="paper",
                step="postdraft_enrichment",
                data={"count": merged.get("count") if isinstance(merged, Mapping) else None},
            )
        ],
    }


def _after_retrieve(state: PostdraftState) -> str:
    return "weave" if state.get("retrieval_ok") else "end"


async def _weave(state: PostdraftState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ok, failure, _ = await run_writer_session(
        state,
        runtime,
        stage="draft",
        prompt_key=state.get("writer_prompt_key") or "paper/paper",
        prompt_name="postdraft_citation_weave_prompt.md",
        name="postdraft_citation_weave",
    )
    if not ok:
        return {
            "postdraft_status": "failed",
            "postdraft_detail": failure.detail if failure else "citation weave failed",
        }
    await runtime.context.workspace.write_bytes(join(state["paper_dir"], POSTDRAFT_MARKER), b"")
    return {"postdraft_status": "done", "postdraft_detail": "postdraft references woven"}


def build_postdraft_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "paper.postdraft_enrichment",
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Return the default ``paper.postdraft_enrichment`` slot.

    It runs mounted in the paper graph or alone (its input schema is the slot's input).

    Args:
        checkpointer: The caller's checkpointer when run alone; ``None`` inherits the parent's.
        store: The caller's store when run alone.
        overrides: Replacement slots (the steps carry hooks under the graph's phase name).
        name: The compiled graph's name.
    """
    graph = StepGraph(
        "paper.postdraft_enrichment",
        PostdraftState,
        input_schema=PostdraftInput,
        output_schema=PostdraftOutput,
        overrides=overrides,
    )
    retry = agent_node_retry_policy()
    graph.step("gaps", _gaps, retry_policy=retry)
    graph.add_node("repair_gaps", _repair_gaps)
    graph.step("no_gaps", _no_gaps)
    graph.add_node("failed_gaps", _failed_gaps)
    graph.step("plan", _plan, retry_policy=retry)
    graph.step("check_plan", _check_plan)
    graph.add_node("repair_plan", _repair_plan)
    graph.add_node("failed_plan", _failed_plan)
    graph.step("retrieve", _retrieve)
    graph.step("weave", _weave, retry_policy=retry)
    graph.add_edge(START, "gaps")
    graph.add_conditional_edges(
        "gaps",
        _after_gaps,
        {
            "plan": "plan",
            "no_gaps": "no_gaps",
            "failed_gaps": "failed_gaps",
            "repair_gaps": "repair_gaps",
        },
    )
    graph.add_edge("repair_gaps", "gaps")
    graph.add_edge("no_gaps", END)
    graph.add_edge("failed_gaps", END)
    graph.add_edge("plan", "check_plan")
    graph.add_conditional_edges(
        "check_plan",
        _after_check_plan,
        {
            "end": END,
            "retrieve": "retrieve",
            "failed_plan": "failed_plan",
            "repair_plan": "repair_plan",
        },
    )
    graph.add_edge("repair_plan", "plan")
    graph.add_edge("failed_plan", END)
    graph.add_conditional_edges("retrieve", _after_retrieve, {"weave": "weave", "end": END})
    graph.add_edge("weave", END)
    return graph.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "EDIT_OUTPUTS",
    "FRESH_OUTPUTS",
    "POSTDRAFT_MARKER",
    "PostdraftInput",
    "PostdraftOutput",
    "PostdraftState",
    "available_skills",
    "build_postdraft_graph",
    "expected_outputs",
    "missing_outputs",
    "retrieve_records",
    "run_writer_session",
    "stage_name",
    "stage_prompt_name",
    "writer_role",
]
