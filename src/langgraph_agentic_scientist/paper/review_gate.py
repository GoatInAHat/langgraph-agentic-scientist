"""The in-loop review gate (R34): review packet, reviewer session, review feedback.

Ports ``run_paper_review_gate`` (``paper.sh`` and ``paper_edit.sh``) and ``paper_review.py``.
Each contract attempt builds ``paper_review_packet.md``, sets the previous ``paper_review.json``
aside, and runs the paper reviewer with the ``review-paper`` skill. The reviewer answers with a
:class:`~langgraph_agentic_scientist.paper.models.ReviewPayload` (adaptation A3; its validators
are the checks of ``validate_review_payload``); the host writes ``paper_review.json`` and derives
``paper_review.md`` and ``paper_review_feedback.json``/``.md``.

Outcomes, as upstream's return codes: ``completed`` (0), ``stage_failed`` (the reviewer session
failed: the previous review is put back and the paper loop stops as incomplete), and
``contract_exhausted`` (two attempts without a valid review: the review feedback is cleared and
the loop continues into manuscript repair without it, which upstream also returns as 0).

The gate is the replaceable slot ``paper.review_gate``; :func:`build_review_gate_graph` is its
default.
"""

from __future__ import annotations

import json
import operator
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from ..agents.runner import agent_node_retry_policy, run_structured_role
from ..context import GraphOverrides, ScientistContext
from ..core.runconfig import method_config
from ..core.state import Event, RunConfigRecord, TaskRef, merge_dicts
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .audit import final_outcome_summary, json_excerpt, split_manuscript, truncate
from .contexts import (
    read_json_or_none,
    read_text_or_none,
    render_stage_prompt,
    workflow_text,
    write_json,
    write_stage_contract_feedback,
    write_stage_execution_feedback,
    write_text,
)
from .models import InLoopReview, ReviewIssue, ReviewPayload

MAX_TEX_CHARS = 30000
MAX_JSON_CHARS = 12000
MAX_CONTRACT_ATTEMPTS = 2
"""Upstream ``max_review_contract_attempts``."""

REVIEW_FEEDBACK_ARTIFACTS = (
    "paper_review_feedback.json",
    "paper_review_feedback.md",
    "paper_review.md",
)
STALE_REVIEW_ARTIFACTS = (
    "paper_review_stage_feedback.json",
    "paper_review_stage_feedback.md",
    "paper_review.json",
    *REVIEW_FEEDBACK_ARTIFACTS,
)


def _error_summary(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping) and isinstance(payload.get("_json_error"), str):
        return {"available": False, "error": payload["_json_error"]}
    return None


def _paper_data_summary(payload: Any) -> dict[str, Any]:
    error = _error_summary(payload)
    if error is not None:
        return error
    if not isinstance(payload, Mapping):
        return {"available": False}
    grounding = payload.get("grounding_sources")
    titles = payload.get("idea_titles")
    ablations = payload.get("ablation_studies")
    totals = payload.get("totals")
    return {
        "available": True,
        "paper_profile": payload.get("paper_profile", "default"),
        "downstream_study": payload.get("downstream_study"),
        "final_evaluation": final_outcome_summary(payload),
        "totals": totals if isinstance(totals, Mapping) else {},
        "idea_titles_preview": titles[:10] if isinstance(titles, list) else [],
        "grounding_source_count": len(grounding) if isinstance(grounding, Mapping | list) else 0,
        "ablation_count": len(ablations) if isinstance(ablations, list) else 0,
    }


def _references_summary(payload: Any) -> dict[str, Any]:
    error = _error_summary(payload)
    if error is not None:
        return error
    if not isinstance(payload, Mapping):
        return {"available": False}
    references = payload.get("references")
    preview = [
        {"id": row.get("id"), "citation_key": row.get("citation_key"), "title": row.get("title")}
        for row in (references[:10] if isinstance(references, list) else [])
        if isinstance(row, Mapping)
    ]
    return {
        "available": True,
        "count": len(references) if isinstance(references, list) else 0,
        "queries": payload.get("queries") if isinstance(payload.get("queries"), list) else [],
        "preview": preview,
    }


def _validation_summary(payload: Any) -> dict[str, Any]:
    error = _error_summary(payload)
    if error is not None:
        return error
    if not isinstance(payload, Mapping):
        return {"available": False}
    raw_feedback = payload.get("feedback")
    feedback: Mapping[str, Any] = raw_feedback if isinstance(raw_feedback, Mapping) else {}

    def count(key: str) -> int:
        value = feedback.get(key)
        return len(value) if isinstance(value, list) else 0

    return {
        "available": True,
        "paper_profile": payload.get("paper_profile"),
        "pdf_path": payload.get("pdf_path"),
        "hard_error_count": count("errors"),
        "figure_warning_count": count("figure_validation_warnings"),
        "figure_error_count": count("figure_validation_errors"),
    }


async def build_review_packet(ctx: ScientistContext, paper_dir: str) -> str:
    """``build_review_packet``: write ``paper_review_packet.md`` and return its path.

    Raises:
        FileNotFoundError: When ``paper.tex`` is missing.
    """
    ws = ctx.workspace
    tex = await read_text_or_none(ws, join(paper_dir, "paper.tex"))
    if tex is None:
        raise FileNotFoundError(
            f"Missing required manuscript artifact: {join(paper_dir, 'paper.tex')}"
        )
    lines = [
        "# Paper review packet",
        "",
        f"- generated_at: `{datetime.now(UTC).isoformat()}`",
        f"- paper_dir: `{paper_dir}`",
        "",
        *workflow_text(ctx, "review_packet_intro.md").split("\n"),
        "",
        "## Artifact checklist",
        "",
    ]
    for name in (
        "paper.tex",
        "claims_grounding.json",
        "paper_data.json",
        "references.bib",
        "references_provenance.json",
        "validation_report.json",
        "paper_strategy.json",
        "figure_manifest.json",
        "paper.pdf",
    ):
        state = "present" if await ws.exists(join(paper_dir, name)) else "missing"
        lines.append(f"- `{name}` — `{state}`")
    strategy = await read_json_or_none(ws, join(paper_dir, "paper_strategy.json"))
    manifest = await read_json_or_none(ws, join(paper_dir, "figure_manifest.json"))

    def excerpt(payload: Any) -> str:
        return json_excerpt(payload, max_chars=MAX_JSON_CHARS)

    lines.extend(
        [
            "",
            "## paper_data summary",
            "",
            "```json",
            excerpt(
                _paper_data_summary(await read_json_or_none(ws, join(paper_dir, "paper_data.json")))
            ),
            "```",
            "",
            "## claims_grounding excerpt",
            "",
            "```json",
            excerpt(await read_json_or_none(ws, join(paper_dir, "claims_grounding.json"))),
            "```",
            "",
            "## references provenance summary",
            "",
            "```json",
            excerpt(
                _references_summary(
                    await read_json_or_none(ws, join(paper_dir, "references_provenance.json"))
                )
            ),
            "```",
            "",
            "## validation report summary",
            "",
            "```json",
            excerpt(
                _validation_summary(
                    await read_json_or_none(ws, join(paper_dir, "validation_report.json"))
                )
            ),
            "```",
        ]
    )
    if strategy is not None:
        lines.extend(["", "## paper_strategy excerpt", "", "```json", excerpt(strategy), "```"])
    if manifest is not None:
        lines.extend(["", "## figure_manifest excerpt", "", "```json", excerpt(manifest), "```"])
    manuscript, supplement = split_manuscript(tex)
    lines.extend(
        [
            "",
            "## manuscript source excerpt",
            "",
            "```tex",
            truncate(manuscript, max_chars=MAX_TEX_CHARS),
            "```",
            "",
        ]
    )
    if supplement is not None:
        lines.extend(
            [
                "## supplementary information excerpt",
                "",
                "```tex",
                truncate(supplement, max_chars=MAX_TEX_CHARS),
                "```",
                "",
            ]
        )
    path = join(paper_dir, "paper_review_packet.md")
    await write_text(ws, path, "\n".join(lines))
    return path


def render_review_markdown(review: ReviewPayload) -> str:
    """Upstream ``render_review_markdown``."""
    lines = [
        "# Paper review",
        "",
        f"- status: `{review.status}`",
        f"- score: `{review.score.value}: {review.score.label}`",
        "",
        "## Summary",
        review.summary,
        "",
        "## Strengths",
    ]
    lines.extend(f"- {item}" for item in review.strengths) if review.strengths else lines.append(
        "- None recorded."
    )
    lines.append("")

    def block(title: str, issues: list[ReviewIssue]) -> None:
        lines.append(f"## {title}")
        if not issues:
            lines.extend(["- None.", ""])
            return
        for index, issue in enumerate(issues, start=1):
            lines.append(f"### {index}. {issue.id} — {issue.section}")
            lines.append(f"- issue: {issue.issue}")
            lines.append(f"- requested_change: {issue.requested_change}")
            if issue.evidence:
                lines.append("- evidence:")
                lines.extend(f"  - {item}" for item in issue.evidence)
            lines.append("")

    block("Major comments", review.major_comments)
    block("Minor comments", review.minor_comments)
    lines.append("## Questions")
    lines.extend(f"- {item}" for item in review.questions) if review.questions else lines.append(
        "- None."
    )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


async def write_review_feedback(
    workspace: Workspace, paper_dir: str, review: ReviewPayload
) -> None:
    """``finalize_review``: ``paper_review.md`` and ``paper_review_feedback.json``/``.md``."""
    json_path = join(paper_dir, "paper_review.json")
    md_path = join(paper_dir, "paper_review.md")
    await write_text(workspace, md_path, render_review_markdown(review))

    def summaries(issues: list[ReviewIssue]) -> list[dict[str, Any]]:
        return [
            {
                "id": issue.id,
                "section": issue.section,
                "issue": issue.issue,
                "requested_change": issue.requested_change,
                "evidence": list(issue.evidence),
            }
            for issue in issues
        ]

    major = summaries(review.major_comments)
    minor = summaries(review.minor_comments)
    feedback = {
        "generated_at": datetime.now(UTC).isoformat(),
        "ok": not major,
        "status": review.status,
        "summary": review.summary,
        "major_comment_count": len(major),
        "minor_comment_count": len(minor),
        "review_json_path": json_path,
        "review_markdown_path": md_path,
        "major_comments": major,
        "minor_comments": minor,
    }
    await write_json(workspace, join(paper_dir, "paper_review_feedback.json"), feedback)
    lines = [
        "# Paper review feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- ok: `{feedback['ok']}`",
        f"- status: `{review.status}`",
        f"- major_comment_count: `{len(major)}`",
        f"- minor_comment_count: `{len(minor)}`",
        f"- review_json_path: `{json_path}`",
        f"- review_markdown_path: `{md_path}`",
        "",
        "## Summary",
        review.summary,
        "",
        "## Major comments",
    ]

    def items(rows: list[dict[str, Any]]) -> None:
        if not rows:
            lines.append("- None.")
            return
        for item in rows:
            lines.append(f"- `{item['id']}` ({item['section']}): {item['issue']}")
            lines.append(f"  - requested_change: {item['requested_change']}")
            if item["evidence"]:
                lines.append("  - evidence:")
                lines.extend(f"    - {entry}" for entry in item["evidence"])

    items(major)
    lines.extend(["", "## Minor comments"])
    items(minor)
    lines.append("")
    await write_text(
        workspace, join(paper_dir, "paper_review_feedback.md"), "\n".join(lines).rstrip() + "\n"
    )


class ReviewGateInput(TypedDict):
    """What the review gate reads from the paper loop."""

    run_config: RunConfigRecord | None
    task: TaskRef
    paper_dir: str
    mode: NotRequired[str]
    writer_prompt_key: NotRequired[str]
    writer_skill: NotRequired[str]
    paper_format: NotRequired[str]
    two_stage: NotRequired[bool]
    review_round: NotRequired[int]


class ReviewGateOutput(TypedDict):
    """What the review gate returns."""

    in_loop_review: InLoopReview
    events: list[Event]


class ReviewGateState(ReviewGateInput, total=False):
    """The gate's own state across its contract attempts."""

    in_loop_review: InLoopReview
    events: Annotated[list[Event], operator.add]
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    contract_attempt: int
    prompt: str
    prompt_path: str
    payload: ReviewPayload | None
    session_failure: str | None
    contract_failure: str | None


BACKUP = ".paper_review_previous.json"


async def _prepare(state: ReviewGateState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ctx = runtime.context
    ws = ctx.workspace
    paper_dir = state["paper_dir"]
    attempt = int(state.get("contract_attempt") or 0) + 1
    review_attempt = int(state.get("review_round") or 0) + 1
    if attempt == 1:
        for name in STALE_REVIEW_ARTIFACTS:
            await ws.remove(join(paper_dir, name))
    await build_review_packet(ctx, paper_dir)
    if review_attempt == 1 and attempt == 1:
        prompt_name = "paper_review_prompt.md"
    elif attempt == 1:
        prompt_name = f"paper_review_prompt_round_{review_attempt}.md"
    else:
        prompt_name = f"paper_review_prompt_round_{review_attempt}_contract_retry_{attempt}.md"
    prompt = await render_stage_prompt(state, runtime, "paper/review_json")
    prompt_path = join(paper_dir, prompt_name)
    await write_text(ws, prompt_path, prompt)
    backup = join(paper_dir, BACKUP)
    await ws.remove(backup)
    current = join(paper_dir, "paper_review.json")
    if await ws.exists(current):
        await ws.write_bytes(backup, await ws.read_bytes(current))
        await ws.remove(current)
    return {
        "contract_attempt": attempt,
        "prompt": prompt,
        "prompt_path": prompt_path,
        "payload": None,
        "session_failure": None,
        "contract_failure": None,
    }


async def _session(state: ReviewGateState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    cfg = method_config(state).paper
    outcome = await run_structured_role(
        role=Role.PAPER_REVIEWER,
        schema=ReviewPayload,
        runtime=runtime,
        root=state["paper_dir"],
        read_only=True,
        prompt=state["prompt"],
        skills=("review-paper",) if "review-paper" in runtime.context.prompts.skill_names() else (),
        timeout=cfg.timeout_seconds,
    )
    if isinstance(outcome.structured, ReviewPayload):
        return {"payload": outcome.structured}
    assert outcome.failure is not None
    if outcome.failure.category == "structured_response_missing":
        return {"contract_failure": outcome.failure.detail or "no structured review"}
    return {"session_failure": f"{outcome.failure.category}: {outcome.failure.detail}"}


async def _finalize(state: ReviewGateState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ws = runtime.context.workspace
    paper_dir = state["paper_dir"]
    attempt = int(state.get("contract_attempt") or 1)
    current = join(paper_dir, "paper_review.json")
    backup = join(paper_dir, BACKUP)
    prompt_path = state.get("prompt_path") or join(paper_dir, "paper_review_prompt.md")
    stage_name = PurePosixPath(prompt_path).name.removesuffix(".md").replace("_prompt", "")
    payload = state.get("payload")
    if state.get("session_failure"):
        await write_stage_execution_feedback(
            ws,
            paper_dir,
            feedback_json="paper_review_stage_feedback.json",
            feedback_md="paper_review_stage_feedback.md",
            stage_name=stage_name,
            failure_detail=str(state["session_failure"]),
            prompt_path=prompt_path,
            expected_outputs=("paper_review.json",),
        )
        if not await ws.exists(current) and await ws.exists(backup):
            await ws.write_bytes(current, await ws.read_bytes(backup))
            await ws.remove(backup)
        return {
            "in_loop_review": InLoopReview(
                outcome="stage_failed", attempts=attempt, detail=str(state["session_failure"])
            )
        }
    await ws.remove(backup)
    if payload is not None:
        await write_json(ws, current, payload.model_dump(mode="json"))
        await write_review_feedback(ws, paper_dir, payload)
        await ws.remove(join(paper_dir, "paper_review_stage_feedback.json"))
        await ws.remove(join(paper_dir, "paper_review_stage_feedback.md"))
        return {
            "in_loop_review": InLoopReview(outcome="completed", review=payload, attempts=attempt),
            "events": [
                Event(
                    kind="paper_review_completed",
                    phase="paper",
                    step="review_gate",
                    data={"major_comment_count": len(payload.major_comments)},
                )
            ],
        }
    await write_stage_contract_feedback(
        ws,
        paper_dir,
        feedback_json="paper_review_stage_feedback.json",
        feedback_md="paper_review_stage_feedback.md",
        stage_name=stage_name,
        prompt_path=prompt_path,
        expected_outputs=("paper_review.json",),
        hard_errors=[
            f"review stage '{stage_name}' wrote an invalid paper review artifact: "
            f"{state.get('contract_failure') or 'no structured review'}"
        ],
    )
    if attempt >= MAX_CONTRACT_ATTEMPTS:
        for name in REVIEW_FEEDBACK_ARTIFACTS:
            await ws.remove(join(paper_dir, name))
        return {
            "in_loop_review": InLoopReview(
                outcome="contract_exhausted",
                attempts=attempt,
                detail=str(state.get("contract_failure") or ""),
            )
        }
    return {}


def _after_finalize(state: ReviewGateState) -> str:
    return "done" if state.get("in_loop_review") is not None else "retry"


def build_review_gate_graph(
    *, overrides: GraphOverrides | None = None, name: str = "paper.review_gate"
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Return the default ``paper.review_gate`` slot (at most two contract attempts)."""
    graph = StepGraph(
        "paper.review_gate",
        ReviewGateState,
        input_schema=ReviewGateInput,
        output_schema=ReviewGateOutput,
        overrides=overrides,
    )
    graph.step("prepare", _prepare)
    graph.step("session", _session, retry_policy=agent_node_retry_policy())
    graph.step("finalize", _finalize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "session")
    graph.add_edge("session", "finalize")
    graph.add_conditional_edges("finalize", _after_finalize, {"done": END, "retry": "prepare"})
    return graph.compile(name=name)


def review_json_text(review: ReviewPayload) -> str:
    """The ``paper_review.json`` the host writes for an accepted review."""
    return json.dumps(review.model_dump(mode="json"), indent=2, sort_keys=True)


__all__ = [
    "MAX_CONTRACT_ATTEMPTS",
    "ReviewGateInput",
    "ReviewGateOutput",
    "ReviewGateState",
    "build_review_gate_graph",
    "build_review_packet",
    "render_review_markdown",
    "review_json_text",
    "write_review_feedback",
]
