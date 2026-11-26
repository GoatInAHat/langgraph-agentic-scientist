"""Prompt values and feedback files of the paper stages.

:func:`prompt_values` reproduces the ``extra`` replacement table of upstream's ``render_prompt``
(``paper.sh`` and ``paper_edit.sh``) together with ``paper_workspace_context_replacements``
(``paper_prompt_context.py``): every path is rendered as the agent sees it from the paper
workspace (``./paper.tex``, ``./task``, ``./results``), and the context paragraphs are chosen by the
files present in the workspace, as upstream chooses them. The paragraphs are package texts
derived from upstream (``prompts/workflow/paper/*``).

The feedback writers port ``paper_feedback.py`` (validation, stage-execution and stage-contract
feedback) with upstream's JSON keys and markdown layout. Paths in feedback are workspace paths
(upstream writes absolute host paths); the agents read them relative to the paper workspace.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..core.interfaces import FigureAvailability
from ..core.render import task_sections
from ..core.state import utc_now
from ..workspace.paths import join
from ..workspace.protocol import Workspace

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

    from ..context import ScientistContext

LOG_TAIL_LINES = 40
"""Upstream ``_LOG_TAIL_LINES``."""

GLOBAL_MEMORY_RECENT_LINES = 80
"""Upstream ``tail_jsonl(..., max_lines=80)`` of the global memory."""

WORKFLOW = "workflow/paper"


def workflow_text(ctx: ScientistContext, name: str, values: Mapping[str, str] | None = None) -> str:
    """Return a paper workflow text from the registry, rendered with ``values``."""
    key = f"{WORKFLOW}/{name.removesuffix('.md')}"
    if values:
        return ctx.prompts.render(key, values, fragments=False)
    return ctx.prompts.text(key)


async def read_text_or_none(workspace: Workspace, path: str) -> str | None:
    """Return a UTF-8 file's text, or ``None`` when it is not a regular file."""
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        return None
    return (await workspace.read_bytes(path)).decode("utf-8", errors="replace")


async def read_json_or_none(workspace: Workspace, path: str) -> Any:
    """Return parsed JSON of a file, ``None`` when absent, and ``{"_json_error": ...}`` when invalid
    (upstream ``_read_json`` of the packet builders)."""
    text = await read_text_or_none(workspace, path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        return {"_json_error": f"Invalid JSON in {path.rsplit('/', 1)[-1]}: {exc}"}


async def is_regular_file(workspace: Workspace, path: str) -> bool:
    """Return whether ``path`` is a regular file."""
    entry = await workspace.stat(path)
    return entry is not None and entry.kind == "file"


async def path_exists(workspace: Workspace, path: str) -> bool:
    """Return whether anything exists at ``path``."""
    return await workspace.exists(path)


async def write_json(workspace: Workspace, path: str, payload: Any) -> None:
    """Write JSON with upstream's layout (``indent=2, sort_keys=True``)."""
    await workspace.write_bytes(path, json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))


async def write_text(workspace: Workspace, path: str, text: str) -> None:
    """Write a UTF-8 text file."""
    await workspace.write_bytes(path, text.encode("utf-8"))


def tail_jsonl(text: str | None, *, max_lines: int) -> list[Any]:
    """Return the parsed objects of the last ``max_lines`` lines (upstream ``tail_jsonl``)."""
    if not text:
        return []
    entries: list[Any] = []
    for line in text.splitlines()[-max_lines:]:
        raw = line.strip()
        if not raw:
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return entries


# --- context renderers (paper_prompt_context.py) ------------------------------------------


def render_posthoc_analyses_context(ctx: ScientistContext, analyses: Sequence[Any] | None) -> str:
    """``_render_posthoc_analyses_context`` for the analyses staged in the workspace."""
    if analyses is None:
        return workflow_text(ctx, "posthoc_context_none.md")
    lines: list[str] = []
    for analysis in analyses:
        if not isinstance(analysis, Mapping):
            continue
        lines.append(f"## {analysis.get('title', analysis.get('analysis_id', 'unknown'))}")
        lines.append(f"- **ID**: `{analysis.get('analysis_id', '')}`")
        lines.append(f"- **Family**: {analysis.get('family', '')}")
        summary = str(analysis.get("summary_text", "") or "").strip()
        if summary:
            lines.append(f"\n### Summary\n{summary}")
        rubric = analysis.get("rubric", {})
        if rubric and isinstance(rubric, Mapping):
            lines.append("\n### Interestingness Rubric (self-assessed, treat as noisy context)")
            for key, value in rubric.items():
                lines.append(f"- {key}: {value}")
        lines.append("")
    return "\n".join(lines)


def render_draft_gaps_context(ctx: ScientistContext, payload: Any, *, present: bool) -> str:
    """``_render_draft_reference_gaps_context`` from the parsed ``draft_reference_gaps.json``."""
    if not present:
        return workflow_text(ctx, "draft_gaps_context_none.md")
    if not isinstance(payload, Mapping) or "_json_error" in payload:
        return workflow_text(ctx, "draft_gaps_context_unreadable.md")
    lines = [f"Status: `{payload.get('status', 'unknown')}`"]
    summary = str(payload.get("summary", "")).strip()
    if summary:
        lines.append(f"Summary: {summary}")
    missing = payload.get("missing_literatures", [])
    if isinstance(missing, list) and missing:
        lines.append("Missing literatures:")
        lines.extend(f"- {topic}" for topic in missing)
    targets = payload.get("rewrite_targets", [])
    if isinstance(targets, list) and targets:
        lines.append("Rewrite targets:")
        for row in targets:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                "- "
                + " | ".join(
                    value
                    for value in (
                        str(row.get("target_id", "")).strip(),
                        str(row.get("section", "")).strip(),
                        str(row.get("priority", "")).strip(),
                        str(row.get("citation_intent", "")).strip(),
                    )
                    if value
                )
            )
            reason = str(row.get("reason", "")).strip()
            if reason:
                lines.append(f"  reason: {reason}")
            anchor = str(row.get("anchor_text", "")).strip()
            if anchor:
                lines.append(f"  anchor: {anchor}")
            topics = row.get("search_topics", [])
            if isinstance(topics, list) and topics:
                lines.append("  search_topics: " + ", ".join(str(t).strip() for t in topics))
    return "\n".join(lines)


def venue_context(ctx: ScientistContext, present: bool) -> str:
    """``venue_context_text``."""
    if not present:
        return workflow_text(ctx, "venue_context_absent.md")
    return workflow_text(ctx, "venue_context_present.md", {"VENUE_PATH": "./VENUE.md"})


def contributions_context(ctx: ScientistContext, present: bool) -> str:
    """``contributions_context_text``."""
    if not present:
        return workflow_text(ctx, "contributions_context_absent.md")
    return workflow_text(
        ctx, "contributions_context_present.md", {"CONTRIBUTIONS_PATH": "./contributions.md"}
    )


def author_block_context(ctx: ScientistContext, present: bool) -> str:
    """``author_block_context_text``."""
    if not present:
        return workflow_text(ctx, "author_block_context_absent.md")
    return workflow_text(
        ctx, "author_block_context_present.md", {"AUTHOR_PATH": "./paper_authors.tex"}
    )


def figure_availability(ctx: ScientistContext) -> FigureAvailability:
    """The concept-figure backends of the run (A8): the renderer's, or none configured."""
    if ctx.figure_renderer is None:
        return FigureAvailability()
    return ctx.figure_renderer.availability()


# --- the replacement table of render_prompt -----------------------------------------------

STATIC_PATHS: dict[str, str] = {
    "PAPER_DIR": ".",
    "PAPER_DATA_PATH": "./paper_data.json",
    "PAPER_BIB_PATH": "./references.bib",
    "PAPER_REFERENCES_PROVENANCE_PATH": "./references_provenance.json",
    "PAPER_FIGURES_DIR": "./figures",
    "PAPER_FIGURE_PATH": "./figures/run_progression.png",
    "PAPER_AUTHOR_BLOCK_PATH": "./paper_authors.tex",
    "PAPER_TEX_PATH": "./paper.tex",
    "PAPER_CLAIMS_PATH": "./claims_grounding.json",
    "PAPER_SCOPE_AUDIT_JSON_PATH": "./method_scope_audit.json",
    "PAPER_SCOPE_AUDIT_MD_PATH": "./method_scope_audit.md",
    "REFERENCE_QUERY_PLAN_PATH": "./reference_query_plan.md",
    "REFERENCE_QUERY_PLAN_JSON_PATH": "./reference_query_plan.json",
    "PAPER_STRATEGY_PATH": "./paper_strategy.md",
    "PAPER_STRATEGY_JSON_PATH": "./paper_strategy.json",
    "CONTRIBUTIONS_FILE_PATH": "./contributions.md",
    "VENUE_FILE_PATH": "./VENUE.md",
    "PAPER_REVIEW_PACKET_PATH": "./paper_review_packet.md",
    "PAPER_REVIEW_JSON_PATH": "./paper_review.json",
    "PAPER_REVIEW_MD_PATH": "./paper_review.md",
    "PAPER_REVIEW_FEEDBACK_JSON_PATH": "./paper_review_feedback.json",
    "PAPER_REVIEW_FEEDBACK_MD_PATH": "./paper_review_feedback.md",
    "PAPER_EVIDENCE_AUDIT_PACKET_PATH": "./paper_evidence_audit_packet.md",
    "PAPER_EVIDENCE_AUDIT_RAW_OUTPUT_PATH": "./paper_evidence_audit_raw.txt",
    "PAPER_EVIDENCE_AUDIT_JSON_PATH": "./paper_evidence_audit.json",
    "PAPER_EVIDENCE_AUDIT_MD_PATH": "./paper_evidence_audit.md",
    "PAPER_EVIDENCE_AUDIT_FEEDBACK_JSON_PATH": "./paper_evidence_audit_feedback.json",
    "PAPER_EVIDENCE_AUDIT_FEEDBACK_MD_PATH": "./paper_evidence_audit_feedback.md",
    "PAPER_EVIDENCE_AUDIT_STAGE_FEEDBACK_JSON_PATH": "./paper_evidence_audit_stage_feedback.json",
    "PAPER_EVIDENCE_AUDIT_STAGE_FEEDBACK_MD_PATH": "./paper_evidence_audit_stage_feedback.md",
    "SCIENTIFIC_EVIDENCE_VERIFICATION_REPORT_PATH": (
        "./scientific-evidence/verification-report.json"
    ),
    "GLOBAL_MEMORY_MD_FILE": "./global_memory/memory.md",
    "GLOBAL_MEMORY_JSONL_FILE": "./global_memory/memory.jsonl",
    "GLOBAL_MEMORY_SUMMARY_PATH": "./global_memory_summary.md",
    "PAPER_EDIT_CONTEXT_PATH": "./paper_edit_context.md",
    "PAPER_EDIT_CONTEXT_JSON_PATH": "./paper_edit_context.json",
    "POSTHOC_RANKING_PATH": "./posthoc_ranking.json",
    "DRAFT_REFERENCE_GAPS_PATH": "./draft_reference_gaps.md",
    "DRAFT_REFERENCE_GAPS_JSON_PATH": "./draft_reference_gaps.json",
    "POSTDRAFT_REFERENCE_QUERY_PLAN_PATH": "./postdraft_reference_query_plan.md",
    "POSTDRAFT_REFERENCE_QUERY_PLAN_JSON_PATH": "./postdraft_reference_query_plan.json",
}
"""The fixed paths of upstream's ``extra`` table, as the agent sees them."""


async def prompt_values(
    ctx: ScientistContext,
    *,
    paper_dir: str,
    template_key: str,
    main_prompt_key: str,
    writer_skill: str,
    paper_format: str,
    min_refs: int,
    max_refs: int,
    two_stage: bool,
    review_rounds: int,
    task_goal: str = "",
    task_rubrics: str = "",
) -> dict[str, str]:
    """Return the placeholder values of a paper-stage prompt (upstream ``render_prompt``).

    Args:
        ctx: The run context.
        paper_dir: The paper workspace directory.
        template_key: Registry key of the prompt being rendered (the strategy prompt reads the
            strategy validation feedback, every other prompt the paper validation feedback).
        main_prompt_key: Registry key of the current writer prompt (``PAPER_MAIN_PROMPT_PATH``).
        writer_skill: The writer skill of the current family.
        paper_format: The venue name.
        min_refs: ``PAPER_MIN_REFS``.
        max_refs: ``PAPER_MAX_REFS``.
        two_stage: Whether two-stage writing is enabled.
        review_rounds: ``PAPER_REVIEW_ROUNDS``.
        task_goal: Fallback ``Goal`` when the staged ``task/task.md`` is missing.
        task_rubrics: Fallback ``Rubrics``.
    """
    ws = ctx.workspace

    async def present(name: str) -> bool:
        return await is_regular_file(ws, join(paper_dir, name))

    task_md = await read_text_or_none(ws, join(paper_dir, "task/task.md"))
    if task_md is not None:
        goal, rubrics = task_sections(task_md)
    else:
        goal, rubrics = task_sections(None)
        if task_goal:
            goal = task_goal
        if task_rubrics:
            rubrics = task_rubrics

    if two_stage and await present("paper_strategy.md") and await present("paper_strategy.json"):
        strategy_context = workflow_text(ctx, "strategy_context_ready.md")
    elif two_stage:
        strategy_context = workflow_text(ctx, "strategy_context_pending.md")
    else:
        strategy_context = workflow_text(ctx, "strategy_context_none.md")

    if template_key.endswith("/paper_strategy"):
        feedback_json, feedback_md = (
            "strategy_validation_feedback.json",
            "strategy_validation_feedback.md",
        )
    else:
        feedback_json, feedback_md = "validation_feedback.json", "validation_feedback.md"
    if await present(feedback_json) or await present(feedback_md):
        validation_context = workflow_text(
            ctx,
            "validation_feedback_context_present.md",
            {"FEEDBACK_JSON_NAME": feedback_json, "FEEDBACK_MD_NAME": feedback_md},
        )
    else:
        validation_context = workflow_text(ctx, "validation_feedback_context_absent.md")

    if await present("paper_review_feedback.json") or await present("paper_review_feedback.md"):
        review_context = workflow_text(ctx, "review_feedback_context_present.md")
    elif review_rounds > 0:
        review_context = workflow_text(ctx, "review_feedback_context_pending.md")
    else:
        review_context = workflow_text(ctx, "review_feedback_context_disabled.md")

    edit_context_present = await present("paper_edit_context.md")
    memory_recent = tail_jsonl(
        await read_text_or_none(ws, join(paper_dir, "global_memory/memory.jsonl")),
        max_lines=GLOBAL_MEMORY_RECENT_LINES,
    )
    availability = figure_availability(ctx)

    values: dict[str, str] = {
        "TASK_DIR": "./task",
        "RESULTS_DIR": "./results",
        "TASK_MD_PATH": "./task/task.md",
        "TASK_GOAL": goal,
        "TASK_RUBRICS": rubrics,
        **STATIC_PATHS,
        "PAPER_FORMAT": paper_format,
        "PAPER_MIN_REFS": str(min_refs),
        "PAPER_MAX_REFS": str(max_refs),
        "PAPER_WRITER_SKILL": writer_skill,
        "PAPER_MAIN_PROMPT_PATH": main_prompt_key.rsplit("/", 1)[-1] + ".md",
        "PAPER_STRATEGY_CONTEXT": strategy_context,
        "PAPER_VALIDATION_FEEDBACK_CONTEXT": validation_context,
        "PAPER_AUTHOR_BLOCK_CONTEXT": author_block_context(ctx, await present("paper_authors.tex")),
        "VENUE_CONTEXT": venue_context(ctx, await present("VENUE.md")),
        "PAPER_REVIEW_CONTEXT": review_context,
        "GLOBAL_MEMORY_RECENT_JSON": json.dumps(memory_recent, indent=2, sort_keys=True),
        "PAPER_OPENAI_API_KEY_AVAILABLE": "true" if availability.openai else "false",
        "PAPER_OPENAI_API_KEY_CONTEXT": availability.openai_context,
        "PAPER_GEMINI_API_KEY_AVAILABLE": "true" if availability.gemini else "false",
        "PAPER_GEMINI_API_KEY_CONTEXT": availability.gemini_context,
        "PAPER_EDIT_STRATEGY_CONTEXT": workflow_text(
            ctx,
            "edit_strategy_context_edit.md"
            if edit_context_present
            else "edit_strategy_context_fresh.md",
        ),
        "PAPER_EDIT_CONTEXT": workflow_text(
            ctx, "edit_context_edit.md" if edit_context_present else "edit_context_fresh.md"
        ),
    }
    # paper_workspace_context_replacements
    analyses = await read_json_or_none(ws, join(paper_dir, "posthoc_analyses_context.json"))
    gaps_path = join(paper_dir, "draft_reference_gaps.json")
    readme = await read_text_or_none(ws, join(paper_dir, "results_readme.md"))
    values.update(
        {
            "POSTHOC_ANALYSES_CONTEXT": render_posthoc_analyses_context(
                ctx, analyses if isinstance(analyses, list) else None
            ),
            "DRAFT_REFERENCE_GAPS_CONTEXT": render_draft_gaps_context(
                ctx,
                await read_json_or_none(ws, gaps_path),
                present=await is_regular_file(ws, gaps_path),
            ),
            "CONTRIBUTIONS_CONTEXT": contributions_context(ctx, await present("contributions.md")),
            "PAPER_AGGREGATED_DIR": "./aggregated"
            if await ws.exists(join(paper_dir, "aggregated"))
            else "",
            "RESULTS_README": readme.strip()
            if readme is not None
            else workflow_text(ctx, "results_readme_missing.md"),
        }
    )
    return values


async def render_stage_prompt(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext], key: str
) -> str:
    """Render a paper-stage prompt from the loop state (upstream ``render_prompt <template>
    <current writer prompt>``); the caller keeps the prompt artefact."""
    from ..core.runconfig import method_config

    ctx = runtime.context
    cfg = method_config(state).paper
    task = state.get("task")
    values = await prompt_values(
        ctx,
        paper_dir=state["paper_dir"],
        template_key=key,
        main_prompt_key=state.get("writer_prompt_key") or "paper/paper",
        writer_skill=state.get("writer_skill") or "write-paper",
        paper_format=state.get("paper_format") or cfg.paper_format,
        min_refs=cfg.min_refs,
        max_refs=cfg.max_refs,
        two_stage=bool(state.get("two_stage", cfg.two_stage)),
        review_rounds=cfg.review_rounds,
        task_goal=task.goal if task is not None else "",
        task_rubrics=task.rubrics if task is not None else "",
    )
    return ctx.prompts.render(key, values)


# --- feedback writers (paper_feedback.py) ------------------------------------------------------


def _append_unique(messages: list[str], message: object) -> None:
    text = str(message).strip()
    if text and text not in messages:
        messages.append(text)


def validation_report_hard_errors(report: Mapping[str, Any] | None) -> list[str]:
    """``_validation_report_hard_errors``: top-level failures repair prompts must see."""
    if not isinstance(report, Mapping):
        return []
    messages: list[str] = []
    if report.get("ok") is False and report.get("error"):
        _append_unique(messages, report["error"])
        return messages
    if report.get("ok") is not False:
        return messages
    for block_name in (
        "figure_validation",
        "page_validation",
        "evidence_validation",
        "reference_validation",
        "standalone_language_validation",
        "venue_validation",
        "supplementary_information_validation",
    ):
        block = report.get(block_name)
        if not isinstance(block, Mapping):
            continue
        for key in ("hard_errors", "errors"):
            values = block.get(key)
            if isinstance(values, list):
                for value in values:
                    _append_unique(messages, value)
    return messages


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def build_validation_feedback(
    *,
    paper_dir: str,
    report: Mapping[str, Any] | None,
    error: str | None = None,
    extra_hard_errors: Sequence[str] = (),
    extra_soft_warnings: Sequence[str] = (),
    auxiliary_checks: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Return the validation feedback (JSON object and markdown) of ``write_validation_feedback``.

    ``report`` is the validation report as JSON (``validation_report.json``). The port also reads
    the package's own report fields (``passed``, ``errors``, ``figure_warnings``) when upstream's
    blocks are absent, so a replaced toolchain that reports only those fields still drives repair.
    """
    report_map = dict(report) if isinstance(report, Mapping) else None
    page = _dict(report_map.get("page_validation")) if report_map else {}
    figure_feedback = (
        _list(_dict(report_map.get("feedback")).get("figure_validation_warnings"))
        if report_map
        else []
    )
    if report_map and not figure_feedback:
        figure_feedback = _list(report_map.get("figure_warnings"))
    evidence = _dict(report_map.get("evidence_validation")) if report_map else {}
    audit = _dict(report_map.get("agentic_evidence_audit")) if report_map else {}
    supplement = _dict(report_map.get("supplementary_information_validation")) if report_map else {}

    hard_errors: list[str] = []
    if error is not None:
        _append_unique(hard_errors, error)
    for message in validation_report_hard_errors(report_map):
        _append_unique(hard_errors, message)
    if report_map and report_map.get("passed") is False:
        for message in _list(report_map.get("errors")):
            _append_unique(hard_errors, message)
    for message in extra_hard_errors:
        _append_unique(hard_errors, message)
    report_ok = None
    if report_map is not None:
        if isinstance(report_map.get("ok"), bool):
            report_ok = report_map["ok"]
        elif isinstance(report_map.get("passed"), bool):
            report_ok = report_map["passed"]
    feedback_ok = error is None and report_ok is not False and not hard_errors
    pdf_path = report_map.get("pdf_path") if report_map else None
    soft_warnings = [str(message) for message in figure_feedback]
    soft_warnings.extend(str(message) for message in extra_soft_warnings)

    evidence_summary = _dict(evidence.get("summary"))
    dangling = _list(evidence.get("dangling"))
    evidence_hard = _list(evidence.get("hard_errors"))
    audit_findings = _list(audit.get("findings"))

    def pick(item: Any, key: str) -> Any:
        return item.get(key) if isinstance(item, Mapping) else None

    feedback: dict[str, Any] = {
        "generated_at": utc_now(),
        "paper_dir": paper_dir,
        "ok": feedback_ok,
        "hard_errors": hard_errors,
        "soft_warnings": soft_warnings,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "evidence_validation": {
            "checked": bool(evidence.get("checked", False)),
            "bundle_path": evidence.get("bundle_path"),
            "bundle_present": bool(evidence.get("bundle_present", False)),
            "verification_report_path": evidence.get("verification_report_path"),
            "paper_token_count": evidence_summary.get(
                "total_references", evidence.get("paper_token_count", 0)
            ),
            "summary": {
                key: evidence_summary.get(key, 0)
                for key in (
                    "total_references",
                    "resolved_references",
                    "anchored_references",
                    "dangling_references",
                    "manifests_checked",
                )
            },
            "dangling": [
                {
                    "subject_path": pick(item, "subject_path"),
                    "subject_kind": pick(item, "subject_kind"),
                    "raw_reference": pick(item, "raw_reference"),
                    "target": pick(item, "target"),
                    "locator": pick(item, "locator"),
                    "messages": [str(m) for m in _list(pick(item, "messages"))],
                }
                for item in dangling
            ],
            "errors": [str(message) for message in evidence_hard],
        },
        "agentic_evidence_audit": {
            "checked": bool(audit.get("checked", False)),
            "verdict": audit.get("verdict"),
            "summary": audit.get("summary"),
            "finding_count": int(audit.get("finding_count", 0) or 0),
            "findings": [
                {
                    key: pick(item, key)
                    for key in ("id", "section", "claim", "issue", "evidence", "fix")
                }
                for item in audit_findings
            ],
        },
        "supplementary_information_validation": {
            **{
                key: bool(supplement.get(key, False))
                for key in (
                    "checked",
                    "appendix_present",
                    "heading_present",
                    "placed_after_references",
                    "pre_bibliography_clearpage_present",
                    "clearpage_present",
                )
            },
            "visible_character_count": int(supplement.get("visible_character_count", 0) or 0),
            "substantive_element_count": int(supplement.get("substantive_element_count", 0) or 0),
            "rendered_checked": bool(supplement.get("rendered_checked", False)),
            "rendered_heading_present": bool(supplement.get("rendered_heading_present", False)),
            "rendered_visible_character_count": int(
                supplement.get("rendered_visible_character_count", 0) or 0
            ),
            "errors": [str(m) for m in _list(supplement.get("hard_errors"))],
        },
        "page_validation": {
            "checked": bool(page.get("checked", False)),
            **{
                key: page.get(key)
                for key in (
                    "paper_format",
                    "min_body_pages",
                    "min_last_body_page_fill_ratio",
                    "max_body_pages",
                )
            },
            "exclude_references": bool(page.get("exclude_references", False)),
            **{
                key: page.get(key)
                for key in ("total_pages", "reference_start_page", "supplement_start_page")
            },
            "supplement_pages": page.get("supplement_pages", 0),
            **{
                key: page.get(key)
                for key in (
                    "body_pages",
                    "last_body_page_number",
                    "last_body_page_fill_ratio",
                    "last_body_page_counted",
                )
            },
            "errors": [str(m) for m in _list(page.get("errors"))],
        },
    }
    if auxiliary_checks:
        feedback["auxiliary_checks"] = dict(auxiliary_checks)

    lines = [
        "# Paper validation feedback",
        "",
        f"- generated_at: {feedback['generated_at']}",
        f"- paper_dir: `{paper_dir}`",
        f"- ok: `{feedback['ok']}`",
        "",
    ]
    if hard_errors:
        lines.append("## Hard errors")
        lines.extend(f"- {message}" for message in hard_errors)
        lines.append("")
    block = feedback["supplementary_information_validation"]
    if block["checked"]:
        lines.append("## Supplementary Information validation")
        lines.extend(f"- {key}: `{block[key]}`" for key in list(block)[1:-1])
        lines.append("")
    page_block = feedback["page_validation"]
    if page_block["checked"]:
        lines.append("## Page budget")
        for key in (
            "paper_format",
            "body_pages",
            "min_body_pages",
            "min_last_body_page_fill_ratio",
            "max_body_pages",
            "exclude_references",
            "total_pages",
            "reference_start_page",
            "supplement_start_page",
            "supplement_pages",
            "last_body_page_number",
            "last_body_page_fill_ratio",
            "last_body_page_counted",
        ):
            lines.append(f"- {key}: `{page_block[key]}`")
        lines.extend(f"- page_error: {message}" for message in page_block["errors"])
        lines.append("")
    ev = feedback["evidence_validation"]
    if ev["checked"] or ev["dangling"] or ev["errors"] or ev["bundle_present"]:
        lines.append("## Evidence validation")
        lines.extend(
            [
                f"- bundle_path: `{ev['bundle_path']}`",
                f"- bundle_present: `{ev['bundle_present']}`",
                f"- verification_report: `{ev['verification_report_path']}`",
                *(f"- {key}: `{value}`" for key, value in ev["summary"].items()),
            ]
        )
        if ev["dangling"]:
            lines.append("")
            lines.append("### Dangling `@evidence[[...]]` links to fix")
            for item in ev["dangling"]:
                subject = item.get("subject_path") or "<unknown-subject>"
                token = item.get("raw_reference") or item.get("target") or "<unknown-token>"
                detail = "; ".join(item.get("messages") or []) or "dangling link"
                lines.append(f"- `{subject}` {token} -> {detail}")
        if ev["errors"]:
            lines.append("")
            lines.append("### Evidence hard errors")
            lines.extend(f"- {message}" for message in ev["errors"])
        lines.append("")
    audit_block = feedback["agentic_evidence_audit"]
    if audit_block["checked"] or audit_block["findings"]:
        lines.append("## Agentic evidence audit")
        lines.extend(
            [
                f"- checked: `{audit_block['checked']}`",
                f"- verdict: `{audit_block['verdict']}`",
                f"- finding_count: `{audit_block['finding_count']}`",
            ]
        )
        if audit_block.get("summary"):
            lines.append(f"- summary: {audit_block['summary']}")
        if audit_block["findings"]:
            lines.append("")
            lines.append("### Audit findings to fix")
            for item in audit_block["findings"]:
                lines.append(f"- `{item.get('id')}` ({item.get('section')}): {item.get('issue')}")
                if item.get("claim"):
                    lines.append(f"  - claim: {item['claim']}")
                if item.get("evidence"):
                    lines.append(f"  - evidence: {item['evidence']}")
                if item.get("fix"):
                    lines.append(f"  - requested_fix: {item['fix']}")
        lines.append("")
    if auxiliary_checks:
        lines.append("## Auxiliary checks")
        for name, payload in auxiliary_checks.items():
            if not isinstance(payload, Mapping):
                lines.append(f"- {name}: `{payload}`")
                continue
            lines.append(f"- {name}.ok: `{payload.get('ok')}`")
            for message in payload.get("hard_errors", []) or []:
                lines.append(f"- {name}.hard_error: {message}")
            for message in payload.get("soft_warnings", []) or []:
                lines.append(f"- {name}.soft_warning: {message}")
        lines.append("")
    if soft_warnings:
        lines.append("## Soft warnings")
        lines.extend(f"- {message}" for message in soft_warnings)
        lines.append("")
    return feedback, "\n".join(lines).strip() + "\n"


async def write_validation_feedback(
    workspace: Workspace,
    paper_dir: str,
    *,
    report: Mapping[str, Any] | None,
    error: str | None = None,
    extra_hard_errors: Sequence[str] = (),
    extra_soft_warnings: Sequence[str] = (),
    auxiliary_checks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``validation_feedback.json`` and ``.md`` (upstream ``write_validation_feedback``)."""
    feedback, markdown = build_validation_feedback(
        paper_dir=paper_dir,
        report=report,
        error=error,
        extra_hard_errors=extra_hard_errors,
        extra_soft_warnings=extra_soft_warnings,
        auxiliary_checks=auxiliary_checks,
    )
    await write_json(workspace, join(paper_dir, "validation_feedback.json"), feedback)
    await write_text(workspace, join(paper_dir, "validation_feedback.md"), markdown)
    return feedback


async def write_validation_feedback_from_current_report(
    workspace: Workspace, paper_dir: str, error: str | None = None
) -> dict[str, Any]:
    """Upstream ``write_validation_feedback_from_current_report``."""
    report = await read_json_or_none(workspace, join(paper_dir, "validation_report.json"))
    return await write_validation_feedback(
        workspace,
        paper_dir,
        report=report if isinstance(report, Mapping) and "_json_error" not in report else None,
        error=error or None,
    )


async def _expected_output_records(
    workspace: Workspace, paper_dir: str, expected_outputs: Sequence[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in expected_outputs:
        path = join(paper_dir, raw)
        entry = await workspace.stat(path)
        size = entry.size if entry is not None and entry.kind == "file" else None
        record = {
            "path": path,
            "source": raw,
            "exists": entry is not None,
            "is_file": entry is not None and entry.kind == "file",
            "non_empty": bool(size and size > 0),
            "size_bytes": size,
        }
        records.append(record)
        if not record["non_empty"]:
            missing.append(path)
    return records, missing


def _stage_markdown(
    title: str,
    feedback: Mapping[str, Any],
    header_keys: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    log_tail: Sequence[str],
) -> str:
    lines = [f"# {title}", ""]
    lines.extend(f"- {key}: `{feedback.get(key)}`" for key in header_keys)
    lines.extend(["", "## Hard errors"])
    lines.extend(f"- {message}" for message in feedback["hard_errors"])
    lines.append("")
    if records:
        lines.append("## Required outputs")
        for record in records:
            state = "ok" if record["non_empty"] else "missing_or_empty"
            lines.append(f"- `{record['path']}` — `{state}`")
        lines.append("")
    if log_tail:
        lines.extend(["## Log tail", "```text", *log_tail, "```", ""])
    return "\n".join(lines).strip() + "\n"


async def write_stage_execution_feedback(
    workspace: Workspace,
    paper_dir: str,
    *,
    feedback_json: str,
    feedback_md: str,
    stage_name: str,
    failure_detail: str,
    prompt_path: str,
    expected_outputs: Sequence[str],
    log_text: str = "",
    missing_outputs_failure: bool = False,
) -> dict[str, Any]:
    """Feedback for a stage that failed before validation (``write_stage_execution_feedback``).

    Upstream classifies by the agent's exit status (70 empty log, 71 missing outputs, otherwise
    the exit code). A session has no exit status: ``missing_outputs_failure`` stands for 71, and
    ``failure_detail`` names why the session failed otherwise.
    """
    records, missing = await _expected_output_records(workspace, paper_dir, expected_outputs)
    hard_errors: list[str] = []
    if missing_outputs_failure:
        if missing:
            hard_errors.append(
                f"stage '{stage_name}' did not leave all required outputs non-empty: "
                f"{', '.join(missing)}"
            )
        else:
            hard_errors.append(
                f"stage '{stage_name}' reported missing required outputs, but none were detected "
                "after the fact"
            )
    else:
        hard_errors.append(f"stage '{stage_name}' failed: {failure_detail}")
    if not await workspace.exists(prompt_path):
        hard_errors.append(f"stage prompt file is missing: {prompt_path}")
    log_tail = log_text.splitlines()[-LOG_TAIL_LINES:] if log_text else []
    feedback: dict[str, Any] = {
        "generated_at": utc_now(),
        "paper_dir": paper_dir,
        "ok": False,
        "failure_kind": "stage_execution",
        "stage_name": stage_name,
        "stage_return_code": 71 if missing_outputs_failure else 1,
        "prompt_path": prompt_path,
        "log_path": None,
        "hard_errors": hard_errors,
        "soft_warnings": [],
        "expected_outputs": records,
        "log_tail": log_tail,
    }
    markdown = _stage_markdown(
        "Paper stage execution feedback",
        feedback,
        (
            "generated_at",
            "paper_dir",
            "ok",
            "failure_kind",
            "stage_name",
            "stage_return_code",
            "prompt_path",
        ),
        records,
        log_tail,
    )
    await write_json(workspace, join(paper_dir, feedback_json), feedback)
    await write_text(workspace, join(paper_dir, feedback_md), markdown)
    return feedback


async def write_stage_contract_feedback(
    workspace: Workspace,
    paper_dir: str,
    *,
    feedback_json: str,
    feedback_md: str,
    stage_name: str,
    prompt_path: str,
    expected_outputs: Sequence[str],
    hard_errors: Sequence[str],
    log_text: str = "",
) -> dict[str, Any]:
    """Feedback for a stage whose artifact broke its contract (upstream function)."""
    records, _ = await _expected_output_records(workspace, paper_dir, expected_outputs)
    persisted = list(hard_errors)
    if not await workspace.exists(prompt_path):
        persisted.append(f"stage prompt file is missing: {prompt_path}")
    log_tail = log_text.splitlines()[-LOG_TAIL_LINES:] if log_text else []
    feedback: dict[str, Any] = {
        "generated_at": utc_now(),
        "paper_dir": paper_dir,
        "ok": False,
        "failure_kind": "stage_contract",
        "stage_name": stage_name,
        "prompt_path": prompt_path,
        "log_path": None,
        "hard_errors": persisted,
        "soft_warnings": [],
        "expected_outputs": records,
        "log_tail": log_tail,
    }
    markdown = _stage_markdown(
        "Paper stage contract feedback",
        feedback,
        ("generated_at", "paper_dir", "ok", "failure_kind", "stage_name", "prompt_path"),
        records,
        log_tail,
    )
    await write_json(workspace, join(paper_dir, feedback_json), feedback)
    await write_text(workspace, join(paper_dir, feedback_md), markdown)
    return feedback


def global_memory_summary(ctx: ScientistContext, entries: Sequence[Any]) -> str:
    """``global_memory_summary.md`` of ``paper.sh``'s grounding step."""
    lines = [*workflow_text(ctx, "global_memory_summary_header.md").split("\n")]
    if not entries:
        lines.append("- No global memory entries found yet.")
    else:
        lines.append("## Recent Entries")
        lines.append("")
        for entry in list(entries)[-20:]:
            if not isinstance(entry, Mapping):
                continue
            timestamp = str(entry.get("timestamp", "")).strip() or "unknown_ts"
            step = str(entry.get("step", "")).strip() or "unknown_step"
            scope = str(entry.get("scope", "")).strip() or "unknown_scope"
            summary = str(entry.get("summary", "")).strip() or "(missing summary)"
            evidence = str(entry.get("evidence", "")).strip() or "(missing evidence)"
            lines.append(f"- {timestamp} | step={step} | scope={scope}")
            lines.append(f"  - summary: {summary}")
            lines.append(f"  - evidence: {evidence}")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "GLOBAL_MEMORY_RECENT_LINES",
    "LOG_TAIL_LINES",
    "STATIC_PATHS",
    "author_block_context",
    "build_validation_feedback",
    "contributions_context",
    "figure_availability",
    "global_memory_summary",
    "is_regular_file",
    "path_exists",
    "prompt_values",
    "read_json_or_none",
    "read_text_or_none",
    "render_draft_gaps_context",
    "render_posthoc_analyses_context",
    "render_stage_prompt",
    "tail_jsonl",
    "validation_report_hard_errors",
    "venue_context",
    "workflow_text",
    "write_json",
    "write_stage_contract_feedback",
    "write_stage_execution_feedback",
    "write_text",
    "write_validation_feedback",
    "write_validation_feedback_from_current_report",
]
