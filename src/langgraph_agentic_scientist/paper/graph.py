"""``build_paper_graph``: the paper phase (``paper.sh`` and ``paper_edit.sh``) as one graph.

Preparation: ``record_run_config`` -> ``prepare`` (modes, output directory, aliases) ->
``paper.data`` (slot; skipped when ``paper_data`` was given) -> ``check_final_outcome`` (the
final-outcome gate; ``paper_data.json``) -> ``seed_edit`` (edit and revision modes) ->
``stage_inputs`` (venue brief, author block, contributions, memory summary, writer skills) ->
``paper.references`` (slot) -> ``strategy`` (two-stage only, repair rounds) ->
``resolve_family`` -> ``rank_posthoc`` (when analyses exist, repair rounds) ->
``augment_paper_data`` -> ``init_concept_figure`` -> the loop.

The loop is transcribed from ``paper.sh`` (fresh, 26 rows) and ``paper_edit.sh`` (edit and
revision); both tables are in ``docs/adaptations/paper_loop.md``. Each decision is a pure
function of this module (``decide_*``), so every row is tested on its own; the nodes apply the
decision's effects (restoring or discarding the figure-quality snapshot, saving revision PDFs,
authorising a concept-figure repair) and route by the decision's ``next``. Every counter and flag
is in state, so a resume re-enters the loop at the row that was running.

An incomplete paper ends without finalisation, as upstream's exit status 2; a failed paper ends
with its failure (upstream exit status 1, and in edit mode every exit of ``paper_edit.sh``).
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from langgraph.config import get_config
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer

from ..agents.runner import agent_node_retry_policy, run_structured_role
from ..context import GraphOverrides, ScientistContext
from ..core.results import EditOptions, PaperData, ValidationReport
from ..core.runconfig import method_config, record_run_config
from ..core.state import Event, Failure
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.paths import join
from .audit import build_audit_graph
from .contexts import (
    read_json_or_none,
    read_text_or_none,
    render_stage_prompt,
    write_json,
    write_stage_contract_feedback,
    write_stage_execution_feedback,
    write_text,
    write_validation_feedback,
    write_validation_feedback_from_current_report,
)
from .data import augment_paper_data_with_posthoc, validate_posthoc_ranking
from .figures import (
    authorize_concept_figure_repair,
    initial_state,
    prepare_concept_figure,
    preserve_fingerprint,
)
from .finalize import finalize_paper_artifacts, paper_result
from .models import (
    AuditResult,
    ConceptFigureState,
    FigureQualityState,
    PaperInput,
    PaperMode,
    PaperOutput,
    PaperState,
    PosthocRanking,
    StrategyAnswer,
)
from .ranking import collect_posthoc_analyses
from .references.graph import build_references_graph
from .review_gate import build_review_gate_graph
from .strategy import strategy_resolution, validate_strategy_files, write_strategy_artifacts
from .workspace import (
    PaperPreparationError,
    default_paper_dir,
    effective_edit_options,
    external_review_bytes,
    profile_skills,
    read_external_review,
    resolve_final_stage,
    resolve_reference_mode,
    restore_revision_references,
    seed_edit_workspace,
    stage_aliases,
    stage_documents,
    stage_external_review,
    stage_writer_skills,
    writer_for,
)
from .writer import (
    POSTDRAFT_MARKER,
    build_postdraft_graph,
    run_writer_session,
    stage_name,
    stage_prompt_name,
)

FIGURE_QUALITY_WARNING_MARKERS = (
    "PNG width too small",
    "PNG height too small",
    "references raster PNG despite available vector PDF",
)
"""Upstream ``FIGURE_QUALITY_WARNING_MARKERS``."""


# --- decisions (pure; one per transition table row group) ---------------------------------------


@dataclass(frozen=True)
class Decision:
    """A transition of the loop: the next node, the state updates, and an effect to apply.

    ``action`` is one of ``restore_fq`` (restore the figure-quality snapshot), ``discard_fq``,
    ``snapshot_fq``, ``save_audit_pdf``, ``save_review_pdf``; ``incomplete`` and ``failed``
    carry upstream's message.
    """

    next: str
    updates: Mapping[str, Any] = field(default_factory=dict)
    actions: tuple[str, ...] = ()
    incomplete: str | None = None
    failed: str | None = None


def decide_select_stage(*, fq_pending: bool, repair_round: int) -> Decision:
    """Rows 1-3 (edit E1-E3): what ``write`` runs this round."""
    if fq_pending:
        stage = "figure_quality_repair"
    elif repair_round == 0:
        stage = "draft"
    else:
        stage = "manuscript_repair"
    return Decision(
        "write",
        {
            "stage": stage,
            "figure_quality_pending": False,
            "reverted": False,
            "failure_repeated": False,
            "stage_rc": None,
        },
    )


def decide_after_write(
    *, mode: str, stage: str, ok: bool, repair_round: int, max_rounds: int
) -> Decision:
    """Rows 4-7 (edit E5-E8)."""
    if ok:
        return Decision("postdraft_gate", {"stage_rc": "ok"})
    if stage == "figure_quality_repair":
        return Decision("postdraft_gate", {"stage_rc": "failed", "reverted": True}, ("restore_fq",))
    if repair_round >= max_rounds:
        message = (
            f"exhausted repair rounds after {repair_round} attempts; see validation_feedback.json"
        )
        if mode == "fresh":
            return Decision("outcome", {"stage_rc": "failed"}, incomplete=message)
        return Decision("outcome", {"stage_rc": "failed"}, failed=message)
    return Decision("select_stage", {"stage_rc": "failed", "repair_round": repair_round + 1})


def decide_postdraft(*, repair_round: int, enrichment_done: bool, skip: bool) -> Decision:
    """Row 8 (edit E9): run the post-draft enrichment once, after the first draft."""
    if repair_round == 0 and not enrichment_done and not skip:
        return Decision("postdraft")
    return Decision("figures")


def decide_after_postdraft(*, status: str | None, detail: str) -> Decision:
    """Row 8: an enrichment failure ends the phase."""
    if status == "failed":
        return Decision("outcome", failed=detail or "postdraft reference enrichment failed")
    return Decision("figures")


def decide_after_validate(
    *,
    mode: str,
    stage: str,
    passed: bool,
    reverted: bool,
    fingerprint: str | None,
    previous: str | None,
) -> Decision:
    """Rows 11-12 (edit E12-E14)."""
    is_fq = stage == "figure_quality_repair"
    if is_fq and not passed and not reverted:
        return Decision("validate", {"reverted": True}, ("restore_fq",))
    actions: tuple[str, ...] = ()
    if mode != "fresh" and is_fq and passed and not reverted:
        actions = ("discard_fq",)
    if passed:
        updates: dict[str, Any] = {"previous_failure_fingerprint": None, "failure_repeated": False}
    else:
        repeated = fingerprint is not None and previous is not None and fingerprint == previous
        updates = {"previous_failure_fingerprint": fingerprint, "failure_repeated": repeated}
    return Decision("audit" if mode == "fresh" else "review_check", updates, actions)


def decide_after_audit(
    *,
    mode: str,
    stage: str,
    reverted: bool,
    outcome: str,
    repair_round: int,
    max_rounds: int,
) -> Decision:
    """Rows 14-19 (fresh) and E17-E19 (edit)."""
    if mode == "fresh":
        actions: tuple[str, ...] = ()
        updates: dict[str, Any] = {}
        effective = outcome
        if stage == "figure_quality_repair" and not reverted:
            if outcome != "pass":
                actions = ("restore_fq",)
                updates = {"reverted": True, "audit_effective": "pass"}
                effective = "pass"
            else:
                actions = ("discard_fq",)
        if effective == "findings":
            if repair_round >= max_rounds:
                return Decision(
                    "outcome",
                    updates,
                    actions,
                    incomplete=(
                        "paper evidence audit reported revision findings after exhausting repair "
                        "rounds; see paper_evidence_audit_feedback.json"
                    ),
                )
            return Decision(
                "select_stage",
                {**updates, "repair_round": repair_round + 1},
                (*actions, "authorize_concept", "save_audit_pdf"),
            )
        if effective == "operational_failure":
            return Decision(
                "outcome",
                updates,
                actions,
                incomplete=(
                    "paper evidence audit failed operationally; see "
                    "paper_evidence_audit_stage_feedback.json"
                ),
            )
        return Decision("review_check", updates, actions)
    if outcome == "findings":
        if repair_round >= max_rounds:
            return Decision(
                "outcome",
                failed=f"evidence audit findings remain after {repair_round} repair rounds",
            )
        return Decision("select_stage", {"repair_round": repair_round + 1})
    if outcome != "pass":
        return Decision("outcome", failed="required evidence audit failed operationally")
    return Decision("figure_quality_gate")


def decide_review_check(
    *, mode: str, review_round: int, review_rounds: int, passed: bool
) -> Decision:
    """Whether the review gate runs (row 20; edit E15) and, in edit mode, whether the audit
    runs (E16)."""
    if review_round < review_rounds:
        return Decision("review_gate")
    if mode == "fresh":
        return Decision("figure_quality_gate")
    return Decision("audit" if passed else "figure_quality_gate")


def decide_after_review(
    *, mode: str, outcome: str, review_round: int, repair_round: int, max_rounds: int
) -> Decision:
    """Row 20 (edit E15) after the review gate ran."""
    if outcome == "stage_failed":
        message = "paper review stage failed with exit code 1; see paper_review_stage_feedback.json"
        if mode == "fresh":
            return Decision("outcome", incomplete=message)
        return Decision("outcome", failed=message)
    next_review = review_round + 1
    if repair_round >= max_rounds:
        message = (
            f"exhausted repair rounds after {repair_round} attempts; see validation_feedback.json"
        )
        if mode == "fresh":
            return Decision("outcome", {"review_round": next_review}, incomplete=message)
        return Decision("outcome", {"review_round": next_review}, failed=message)
    return Decision(
        "select_stage",
        {"review_round": next_review, "repair_round": repair_round + 1},
        ("save_review_pdf",),
    )


def decide_figure_quality(
    *,
    passed: bool,
    actionable: bool,
    attempted: bool,
    warning_reported: bool,
    repair_round: int,
    max_rounds: int,
) -> Decision:
    """Rows 21-22 (edit E20-E21). ``snapshot_fq`` is decided by the node: a snapshot that
    succeeds makes the repair pending and spends a round; one that fails reports the warning."""
    if not (passed and actionable):
        return Decision("loop_decision")
    if not attempted and repair_round < max_rounds:
        return Decision("select_stage", {"fq_attempted": True}, ("snapshot_fq",))
    if attempted and not warning_reported:
        return Decision("loop_decision", {"fq_warning_reported": True})
    return Decision("loop_decision")


def decide_loop(
    *,
    mode: str,
    passed: bool,
    repeated: bool,
    repair_round: int,
    max_rounds: int,
    audit_outcome: str | None,
) -> Decision:
    """Rows 23-26 (edit E22-E25)."""
    if passed:
        return Decision("finalize")
    if repeated:
        message = (
            "paper validation errors and inputs were unchanged after repair; see "
            "validation_feedback.json"
        )
        if mode == "fresh":
            return Decision("outcome", incomplete=message)
        return Decision(
            "outcome",
            failed=(
                "validation errors and inputs were unchanged after repair; see "
                "validation_feedback.json"
            ),
        )
    if mode == "fresh":
        if repair_round < max_rounds:
            return Decision("select_stage", {"repair_round": repair_round + 1})
        reasons = ["paper validation failed; see validation_feedback.json"]
        if audit_outcome == "findings":
            reasons.append(
                "paper evidence audit reported revision findings; see "
                "paper_evidence_audit_feedback.json"
            )
        elif audit_outcome == "operational_failure":
            reasons.append(
                "paper evidence audit failed operationally; see "
                "paper_evidence_audit_stage_feedback.json"
            )
        return Decision("outcome", incomplete="; ".join(reasons))
    if repair_round >= max_rounds:
        return Decision(
            "outcome",
            failed=(
                f"exhausted repair rounds after {repair_round} attempts; see "
                "validation_feedback.json"
            ),
        )
    return Decision("select_stage", {"repair_round": repair_round + 1})


def has_actionable_figure_quality_warnings(warnings: list[str]) -> bool:
    """Upstream ``has_actionable_figure_quality_warnings``."""
    return any(marker in w for w in warnings for marker in FIGURE_QUALITY_WARNING_MARKERS)


# --- helpers -----------------------------------------------------------------------------


def _mode(state: Mapping[str, Any]) -> str:
    return str(state.get("mode") or "fresh")


def _route(state: Mapping[str, Any]) -> str:
    return str(state.get("next") or "outcome")


async def _apply(
    decision: Decision, state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Apply a decision's effects and return the state update (``next`` included)."""
    ws = runtime.context.workspace
    paper_dir = state["paper_dir"]
    update: dict[str, Any] = {"next": decision.next}
    fq = (state.get("figure_quality") or FigureQualityState()).model_copy()
    updates = dict(decision.updates)
    if "figure_quality_pending" in updates:
        fq.pending = bool(updates.pop("figure_quality_pending"))
    if "fq_attempted" in updates:
        fq.attempted = bool(updates.pop("fq_attempted"))
    if "fq_warning_reported" in updates:
        fq.warning_reported = bool(updates.pop("fq_warning_reported"))
    effective = updates.pop("audit_effective", None)
    events: list[Event] = []
    for action in decision.actions:
        if action == "restore_fq" and fq.snapshot is not None:
            await ws.restore(fq.snapshot)
            fq.snapshot = None
            events.append(Event(kind="figure_quality_rolled_back", phase="paper"))
        elif action == "discard_fq":
            fq.snapshot = None
        elif action == "snapshot_fq":
            try:
                fq.snapshot = await ws.snapshot(paper_dir)
            except Exception as exc:
                fq.warning_reported = True
                update["next"] = "loop_decision"
                events.append(
                    Event(
                        kind="figure_quality_snapshot_failed",
                        phase="paper",
                        data={"detail": str(exc)[:500]},
                    )
                )
            else:
                fq.pending = True
                updates["repair_round"] = int(state.get("repair_round") or 0) + 1
        elif action == "save_audit_pdf":
            saved = await _save_pdf(
                ws,
                paper_dir,
                "evidence_audit_before_repair_round_"
                f"{int(state.get('repair_round') or 0) + 1:02d}.pdf",
            )
            if saved:
                update["revisions"] = [saved]
        elif action == "save_review_pdf":
            review_round = int(updates.get("review_round", state.get("review_round") or 0))
            next_round = int(state.get("repair_round") or 0) + 1
            saved = await _save_pdf(
                ws,
                paper_dir,
                f"review_round_{review_round:02d}_before_repair_round_{next_round:02d}.pdf",
            )
            if saved:
                update["revisions"] = [saved]
        elif action == "authorize_concept":
            concept = state.get("concept") or ConceptFigureState()
            update["concept"] = await authorize_concept_figure_repair(ws, paper_dir, concept)
    update["figure_quality"] = fq
    if effective is not None and state.get("audit") is not None:
        audit: AuditResult = state["audit"]
        update["audit"] = audit.model_copy(update={"outcome": effective})
    update.update(updates)
    if decision.incomplete:
        reasons = [*(state.get("incomplete_reasons") or []), decision.incomplete]
        update["incomplete_reasons"] = reasons
        update["status"] = "incomplete"
        if not (
            await ws.exists(join(paper_dir, "validation_feedback.json"))
            and await ws.exists(join(paper_dir, "validation_feedback.md"))
        ):
            await write_validation_feedback_from_current_report(ws, paper_dir, decision.incomplete)
    if decision.failed:
        update["status"] = "failed"
        update["failure"] = Failure(category="paper_failed", scope="method", detail=decision.failed)
    if events:
        update["events"] = events
    return update


async def _save_pdf(ws: Any, paper_dir: str, name: str) -> str | None:
    source = join(paper_dir, "paper.pdf")
    if not await ws.exists(source):
        return None
    target = join(paper_dir, "revisions", name)
    await ws.write_bytes(target, await ws.read_bytes(source))
    return target


def _max_rounds(state: Mapping[str, Any]) -> int:
    return method_config(state).paper.max_repair_rounds


def _failed(detail: str, category: str = "paper_failed") -> dict[str, Any]:
    return {
        "status": "failed",
        "failure": Failure(category=category, scope="method", detail=detail),
        "next": "outcome",
    }


# --- preparation nodes ---------------------------------------------------------------------------


async def _prepare(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ctx = runtime.context
    cfg = method_config(state)
    mode = cast(PaperMode, _mode(state))
    task = state["task"]
    paper_dir = (
        state.get("output_dir")
        or state.get("paper_dir")
        or default_paper_dir(task.root, state["run_id"])
    )
    base = {
        "paper_dir": paper_dir,
        "manifest_root": state.get("manifest_dir")
        or (paper_dir.rsplit("/", 1)[0] if "/" in paper_dir else "."),
        "repair_round": 0,
        "review_round": 0,
        "figure_quality": FigureQualityState(),
        "incomplete_reasons": [],
        "status": "running",
        "strategy_round": 0,
        "ranking_round": 0,
    }
    try:
        options = effective_edit_options(mode, state.get("edit_options"))
        previous = state.get("previous_paper_dir")
        if mode != "fresh" and (not previous or not await ctx.workspace.exists(previous)):
            raise PaperPreparationError(f"Missing edit paper directory: {previous}")
        if (
            mode == "revision"
            and state.get("external_review") is None
            and not state.get("external_review_path")
        ):
            raise PaperPreparationError("revision mode needs an external review")
        reference_mode = await resolve_reference_mode(
            ctx.workspace,
            mode,
            previous,
            options.reference_mode,
        )
        profile = cfg.paper.profile
        family, prompt_key, writer_skill = writer_for(mode, profile)
        events = await stage_aliases(ctx, state, paper_dir)
        await ctx.workspace.write_bytes(join(paper_dir, "figures/.keep"), b"")
    except PaperPreparationError as exc:
        return {**base, **_failed(str(exc), "paper_preparation_failed")}
    two_stage = cfg.paper.two_stage and mode != "revision"
    skills = [
        writer_skill if name.startswith("write-paper") else name for name in profile_skills(profile)
    ]
    return {
        **base,
        "next": "paper_data",
        "edit_options": options,
        "profile": profile,
        "two_stage": two_stage,
        "paper_format": cfg.paper.paper_format,
        "reference_mode": reference_mode,
        "figure_refresh_mode": options.figure_refresh_mode if mode != "fresh" else "refresh",
        "skip_postdraft_enrichment": bool(options.skip_postdraft_enrichment)
        if mode != "fresh"
        else False,
        "writer_family": family,
        "writer_prompt_key": prompt_key,
        "writer_skill": writer_skill,
        "selected_skills": skills,
        "events": events,
    }


async def _default_paper_data(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """The default ``paper.data`` slot: the host side's builder (``paper_data_node``), which
    keeps a given ``paper_data`` and builds it from the earlier phases' results otherwise."""
    from .data import paper_data_node

    return dict(await paper_data_node(state, runtime))


async def _check_final_outcome(
    state: PaperState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    if state.get("status") == "failed":
        return {"next": "outcome"}
    data = state.get("paper_data")
    failure = state.get("failure")
    if data is None or failure is not None:
        if failure is not None:
            return _failed(failure.detail, failure.category)
        return _failed("paper data is missing", "paper_data_unavailable")
    payload = data.model_dump(mode="json", by_alias=True)
    totals = payload.get("totals") or {}
    if (
        isinstance(totals, Mapping)
        and "evaluated_programs" in totals
        and int(totals.get("evaluated_programs") or 0) <= 0
        and not payload.get("downstream_study")
    ):
        return _failed("No evaluated programs found in results directory; cannot write paper.")
    final = payload.get("final_evaluation")
    policy = method_config(state).final_outcome_policy
    if (
        isinstance(final, Mapping)
        and final.get("correct") is not True
        and "downstream_study" not in payload
        and policy != "allow_not_qualified"
    ):
        return _failed("Final evaluation did not qualify for publication", "final_outcome_blocked")
    await write_json(
        runtime.context.workspace, join(state["paper_dir"], "paper_data.json"), payload
    )
    return {"next": "seed_edit"}


async def _seed_edit(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    mode = _mode(state)
    if mode == "fresh":
        return {}
    ctx = runtime.context
    ws = ctx.workspace
    paper_dir = state["paper_dir"]
    previous = state.get("previous_paper_dir") or ""
    options = state.get("edit_options") or EditOptions()
    try:
        seeded = await seed_edit_workspace(
            ctx,
            paper_dir=paper_dir,
            previous_paper_dir=previous,
            options=options,
            reference_mode=state.get("reference_mode", "refresh"),
            paper_data=(
                state["paper_data"].model_dump(mode="json", by_alias=True)  # type: ignore[union-attr]
                if state.get("paper_data")
                else None
            ),
        )
        update: dict[str, Any] = {}
        if seeded.get("paper_data") is not None:
            data = PaperData.model_validate(seeded["paper_data"])
            await write_json(ws, join(paper_dir, "paper_data.json"), seeded["paper_data"])
            update["paper_data"] = data
        if mode == "revision":
            review_path = state.get("external_review_path")
            if review_path:
                raw = await read_external_review(ws, review_path)
            else:
                review = state.get("external_review")
                assert review is not None
                raw = external_review_bytes(review)
            await stage_external_review(ws, paper_dir, raw)
            baseline = join(
                paper_dir.rsplit("/", 1)[0] if "/" in paper_dir else ".",
                ".revision_references",
                paper_dir.rsplit("/", 1)[-1],
            )
            for name in ("references.bib", "references_provenance.json"):
                source = join(previous, name)
                if not await ws.exists(source):
                    raise PaperPreparationError(
                        f"Missing preserved revision reference artifact: {source}"
                    )
                await ws.write_bytes(join(baseline, name), await ws.read_bytes(source))
            update["reference_baseline_dir"] = baseline
        return update
    except PaperPreparationError as exc:
        return _failed(str(exc), "paper_preparation_failed")


async def _stage_inputs(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    if state.get("status") == "failed":
        return {"next": "outcome"}
    ctx = runtime.context
    cfg = method_config(state).paper
    paper_dir = state["paper_dir"]
    task_md = await read_text_or_none(ctx.workspace, join(paper_dir, "task/task.md"))
    paths = state.get("artifact_paths") or {}
    try:
        events = await stage_documents(
            ctx,
            paper_dir=paper_dir,
            paper_format=state.get("paper_format") or cfg.paper_format,
            min_refs=cfg.min_refs,
            max_refs=cfg.max_refs,
            contributions_source=paths.get("contributions"),
            task_md=task_md,
            venue_file=(state.get("edit_options") or EditOptions()).venue_file,
        )
    except PaperPreparationError as exc:
        return _failed(str(exc), "paper_preparation_failed")
    marker = join(paper_dir, ".paper_two_stage_enabled")
    if state.get("two_stage"):
        await ctx.workspace.write_bytes(marker, b"")
    else:
        await ctx.workspace.remove(marker)
    _, missing = await _restage(state, runtime)
    if missing:
        events.append(Event(kind="paper_skills_missing", phase="paper", data={"skills": missing}))
    return {"events": events, "next": "references"}


async def _restage(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> tuple[list[str], list[str]]:
    from .contexts import prompt_values

    ctx = runtime.context
    cfg = method_config(state).paper
    values = await prompt_values(
        ctx,
        paper_dir=state["paper_dir"],
        template_key=state.get("writer_prompt_key") or "paper/paper",
        main_prompt_key=state.get("writer_prompt_key") or "paper/paper",
        writer_skill=state.get("writer_skill") or "write-paper",
        paper_format=state.get("paper_format") or cfg.paper_format,
        min_refs=cfg.min_refs,
        max_refs=cfg.max_refs,
        two_stage=bool(state.get("two_stage")),
        review_rounds=cfg.review_rounds,
    )
    return await stage_writer_skills(
        ctx,
        paper_dir=state["paper_dir"],
        skills=state.get("selected_skills") or [],
        writer_skill=state.get("writer_skill") or "write-paper",
        include_review_skill=cfg.review_rounds > 0,
        values=values,
        paper_format=state.get("paper_format") or cfg.paper_format,
        watermark=bool(cfg.watermark),
    )


def references_runner(
    graph: CompiledStateGraph[Any, Any, Any, Any],
) -> Any:
    """Return the default ``paper.references`` slot: run ``graph`` (the references graph).

    The graph is built once by :func:`build_paper_graph`, with the paper graph's overrides, and
    invoked with the node's config, so that it runs as a subgraph of the node.
    """
    from .contexts import prompt_values

    async def run_references(
        state: Mapping[str, Any], runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        ctx = runtime.context
        cfg = method_config(state).paper
        values = await prompt_values(
            ctx,
            paper_dir=state["paper_dir"],
            template_key="paper/reference_query_plan",
            main_prompt_key=state.get("writer_prompt_key") or "paper/paper",
            writer_skill=state.get("writer_skill") or "write-paper",
            paper_format=state.get("paper_format") or cfg.paper_format,
            min_refs=cfg.min_refs,
            max_refs=cfg.max_refs,
            two_stage=bool(state.get("two_stage")),
            review_rounds=cfg.review_rounds,
        )
        result = await graph.ainvoke(
            {
                "task": state["task"],
                "paper_dir": state["paper_dir"],
                "run_config": state.get("run_config"),
                "reference_mode": state.get("reference_mode", "refresh"),
                "variant": "initial" if _mode(state) == "fresh" else "edit",
                "values": values,
                "results_dir": "./results",
                "campaign_results_dir": "./results",
            },
            get_config(),
            context=ctx,
        )
        return {"references": result.get("references")}

    return run_references


def _after_references(state: PaperState) -> str:
    if state.get("status") == "failed":
        return "outcome"
    references = state.get("references")
    status = getattr(references, "status", None)
    if status is None and isinstance(references, Mapping):
        status = references.get("status")
    if status == "failed":
        return "references_failed"
    return "strategy" if state.get("two_stage") else "rank_posthoc"


async def _references_failed(state: PaperState) -> dict[str, Any]:
    references = state.get("references")
    failure = getattr(references, "failure", None)
    detail = failure.detail if failure is not None else "reference retrieval failed"
    return _failed(detail, "references_failed")


# --- strategy and ranking ----------------------------------------------------------------------


async def _strategy(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).paper
    paper_dir = state["paper_dir"]
    round_ = int(state.get("strategy_round") or 0)
    name = (
        "paper_strategy_prompt.md"
        if round_ == 0
        else f"paper_strategy_repair_prompt_round_{round_}.md"
    )
    prompt = await render_stage_prompt(state, runtime, "paper/paper_strategy")
    await write_text(ws, join(paper_dir, name), prompt)
    from .writer import available_skills

    outcome = await run_structured_role(
        role=Role.PAPER_STRATEGIST,
        schema=StrategyAnswer,
        runtime=runtime,
        root=paper_dir,
        read_only=False,
        prompt=prompt,
        skills=available_skills(ctx, state.get("selected_skills") or []),
        timeout=cfg.timeout_seconds,
        stage=False,  # staged (and watermarked) when the paper directory was prepared
    )
    if not isinstance(outcome.structured, StrategyAnswer):
        await write_stage_execution_feedback(
            ws,
            paper_dir,
            feedback_json="strategy_validation_feedback.json",
            feedback_md="strategy_validation_feedback.md",
            stage_name=name.removesuffix(".md").replace("_prompt", ""),
            failure_detail=outcome.failure.detail if outcome.failure else "no structured answer",
            prompt_path=join(paper_dir, name),
            expected_outputs=("paper_strategy.md", "paper_strategy.json"),
            log_text=outcome.final_text,
        )
        return {"strategy_ok": False}
    await write_strategy_artifacts(ws, paper_dir, outcome.structured)
    errors = await validate_strategy_files(ws, paper_dir)
    if errors:
        return {
            "strategy_ok": False,
            "events": [Event(kind="strategy_invalid", phase="paper", data={"errors": errors})],
        }
    return {"strategy_ok": True, "strategy": outcome.structured.strategy}


def _after_strategy(state: PaperState) -> str:
    if state.get("strategy_ok"):
        return "resolve_family"
    if int(state.get("strategy_round") or 0) >= _max_rounds(state):
        return "strategy_exhausted"
    return "strategy_repair"


async def _strategy_repair(state: PaperState) -> dict[str, Any]:
    return {"strategy_round": int(state.get("strategy_round") or 0) + 1}


async def _strategy_exhausted(state: PaperState) -> dict[str, Any]:
    round_ = int(state.get("strategy_round") or 0)
    return _failed(
        f"exhausted strategy repair rounds after {round_} attempts; see "
        f"{join(state['paper_dir'], 'strategy_validation_feedback.json')}",
        "strategy_failed",
    )


async def _resolve_family(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    strategy = state.get("strategy")
    selected = strategy.selected_prompt_family if strategy is not None else ""
    try:
        family, prompt_key, writer_skill, skills = resolve_final_stage(
            current_prompt_key=state.get("writer_prompt_key") or "paper/paper",
            current_skills=state.get("selected_skills") or [],
            selected_family=selected,
        )
    except ValueError as exc:
        return _failed(str(exc), "strategy_failed")
    await write_json(
        runtime.context.workspace,
        join(state["paper_dir"], "paper_strategy_resolution.json"),
        strategy_resolution(family, prompt_key, writer_skill, skills),
    )
    update = {
        "writer_family": family,
        "writer_prompt_key": prompt_key,
        "writer_skill": writer_skill,
        "selected_skills": skills,
    }
    await _restage({**state, **update}, runtime)
    return {**update, "next": "rank_posthoc"}


async def _rank_posthoc(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """One ranking session (or the decision that there is nothing to rank)."""
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).paper
    paper_dir = state["paper_dir"]
    analyses = state.get("posthoc_analyses")
    if analyses is None:
        if not (state.get("two_stage") or state.get("posthoc") is not None):
            return {"posthoc_analyses": [], "ranking_ok": True, "ranking": None}
        analyses = await collect_posthoc_analyses(ws, state.get("posthoc"))
        if analyses:
            await write_json(ws, join(paper_dir, "posthoc_analyses_context.json"), analyses)
    if not analyses:
        return {"posthoc_analyses": [], "ranking_ok": True, "ranking": None}
    round_ = int(state.get("ranking_round") or 0)
    prompt = await render_stage_prompt(state, runtime, "paper/paper_posthoc_ranking")
    if round_ > 0:
        prompt = f"{prompt}\n\n{ctx.prompts.text('workflow/paper/ranking_repair')}"
        name = f"posthoc_ranking_repair_prompt_round_{round_}.md"
        stage = f"posthoc_ranking_repair_round_{round_}"
    else:
        name = "posthoc_ranking_prompt.md"
        stage = "posthoc_ranking"
    await write_text(ws, join(paper_dir, name), prompt)
    await ws.remove(join(paper_dir, "posthoc_ranking.json"))
    from .writer import available_skills

    outcome = await run_structured_role(
        role=Role.POSTHOC_RANKER,
        schema=PosthocRanking,
        runtime=runtime,
        root=paper_dir,
        read_only=True,
        prompt=prompt,
        skills=available_skills(ctx, state.get("selected_skills") or []),
        timeout=cfg.timeout_seconds,
        stage=False,  # staged (and watermarked) when the paper directory was prepared
    )
    if not isinstance(outcome.structured, PosthocRanking):
        await write_stage_execution_feedback(
            ws,
            paper_dir,
            feedback_json="posthoc_ranking_validation_feedback.json",
            feedback_md="posthoc_ranking_validation_feedback.md",
            stage_name=stage,
            failure_detail=outcome.failure.detail if outcome.failure else "no structured answer",
            prompt_path=join(paper_dir, name),
            expected_outputs=("posthoc_ranking.json",),
            log_text=outcome.final_text,
        )
        return {"posthoc_analyses": analyses, "ranking_ok": False}
    ranking = outcome.structured.model_dump(mode="json", exclude_none=True)
    await write_json(ws, join(paper_dir, "posthoc_ranking.json"), ranking)
    try:
        validate_posthoc_ranking(ranking, analyses)
    except ValueError as exc:
        await write_stage_contract_feedback(
            ws,
            paper_dir,
            feedback_json="posthoc_ranking_validation_feedback.json",
            feedback_md="posthoc_ranking_validation_feedback.md",
            stage_name=stage,
            prompt_path=join(paper_dir, name),
            expected_outputs=("posthoc_ranking.json",),
            hard_errors=[f"invalid posthoc ranking artifact: {exc}"],
        )
        return {"posthoc_analyses": analyses, "ranking_ok": False}
    await ws.remove(join(paper_dir, "posthoc_ranking_validation_feedback.json"))
    await ws.remove(join(paper_dir, "posthoc_ranking_validation_feedback.md"))
    return {"posthoc_analyses": analyses, "ranking_ok": True, "ranking": outcome.structured}


def _after_ranking(state: PaperState) -> str:
    if state.get("ranking_ok"):
        return "augment_paper_data"
    if int(state.get("ranking_round") or 0) >= _max_rounds(state):
        return "ranking_exhausted"
    return "ranking_repair"


async def _ranking_repair(state: PaperState) -> dict[str, Any]:
    return {"ranking_round": int(state.get("ranking_round") or 0) + 1}


async def _ranking_exhausted(state: PaperState) -> dict[str, Any]:
    round_ = int(state.get("ranking_round") or 0)
    return _failed(
        f"exhausted posthoc ranking repair rounds after {round_} attempts; see "
        f"{join(state['paper_dir'], 'posthoc_ranking_validation_feedback.json')}",
        "posthoc_ranking_failed",
    )


async def _augment_paper_data(
    state: PaperState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    ranking = state.get("ranking")
    analyses = state.get("posthoc_analyses") or []
    data = state.get("paper_data")
    if ranking is None or not analyses or data is None:
        return {}
    augmented = augment_paper_data_with_posthoc(
        data.model_dump(mode="json", by_alias=True),
        ranking.model_dump(mode="json", exclude_none=True),
        analyses,
    )
    await write_json(
        runtime.context.workspace, join(state["paper_dir"], "paper_data.json"), augmented
    )
    return {"paper_data": PaperData.model_validate(augmented)}


async def _init_concept_figure(
    state: PaperState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    cfg = method_config(state).paper
    preserve = await preserve_fingerprint(runtime.context.workspace, state["paper_dir"])
    concept = initial_state(
        runtime.context,
        preserve=preserve,
        budget_seconds=None,
        full=cfg.concept_figure_full_min_seconds,
        medium=cfg.concept_figure_medium_min_seconds,
        light=cfg.concept_figure_light_min_seconds,
    )
    return {"concept": concept}


# --- loop nodes ---------------------------------------------------------------------------------


async def _select_stage(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    if _mode(state) == "revision":
        await restore_revision_references(
            runtime.context.workspace, state["paper_dir"], state.get("reference_baseline_dir")
        )
    fq = state.get("figure_quality") or FigureQualityState()
    decision = decide_select_stage(
        fq_pending=fq.pending, repair_round=int(state.get("repair_round") or 0)
    )
    return await _apply(decision, state, runtime)


async def _write(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    stage = state.get("stage") or "draft"
    round_ = int(state.get("repair_round") or 0)
    key = (
        "paper/paper_figure_quality_repair"
        if stage == "figure_quality_repair"
        else state.get("writer_prompt_key") or "paper/paper"
    )
    ok, failure, _ = await run_writer_session(
        state,
        runtime,
        stage=stage,
        prompt_key=key,
        prompt_name=stage_prompt_name(stage, round_),
        name=stage_name(stage, round_),
    )
    return {"stage_rc": "ok" if ok else "failed", "stage_failure": failure}


async def _after_write(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    decision = decide_after_write(
        mode=_mode(state),
        stage=state.get("stage") or "draft",
        ok=state.get("stage_rc") == "ok",
        repair_round=int(state.get("repair_round") or 0),
        max_rounds=_max_rounds(state),
    )
    update = await _apply(decision, state, runtime)
    if _mode(state) == "revision" and decision.next == "postdraft_gate":
        await restore_revision_references(
            runtime.context.workspace, state["paper_dir"], state.get("reference_baseline_dir")
        )
    return update


async def _postdraft_gate(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    done = await runtime.context.workspace.exists(join(state["paper_dir"], POSTDRAFT_MARKER))
    decision = decide_postdraft(
        repair_round=int(state.get("repair_round") or 0),
        enrichment_done=done,
        skip=bool(state.get("skip_postdraft_enrichment")),
    )
    return {"next": decision.next}


async def _after_postdraft(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    decision = decide_after_postdraft(
        status=state.get("postdraft_status"), detail=state.get("postdraft_detail") or ""
    )
    return await _apply(decision, state, runtime)


async def _default_figures(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """The default ``paper.figures`` slot (row 9): the concept-figure transaction."""
    if state.get("reverted"):
        return {}
    mode = state.get("figure_refresh_mode") or "refresh"
    concept, events = await prepare_concept_figure(
        runtime.context,
        paper_dir=state["paper_dir"],
        concept=state.get("concept") or ConceptFigureState(),
        refresh_mode=mode if _mode(state) != "fresh" else "refresh",
    )
    return {"concept": concept, "events": events}


async def _default_validation(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """The default ``paper.validation`` slot (row 10): the toolchain's gates, the report, the
    feedback files, the round's archive and the failure fingerprint."""
    ctx = runtime.context
    ws = ctx.workspace
    cfg = method_config(state).paper
    paper_dir = state["paper_dir"]
    toolchain = ctx.paper_toolchain
    if toolchain is None:
        from .validation import WorkspacePaperToolchain

        toolchain = WorkspacePaperToolchain()
    options = {
        "mode": _mode(state),
        "paper_format": state.get("paper_format") or cfg.paper_format,
        "paper_profile": state.get("profile") or cfg.profile,
        "watermark": cfg.watermark,
        "require_scope_audit": _mode(state) != "fresh",
        "final_outcome_policy": method_config(state).final_outcome_policy,
    }
    report: ValidationReport | None
    error: str | None = None
    report_path = join(paper_dir, "validation_report.json")
    await ws.remove(report_path)
    try:
        report = await toolchain.validate(paper_dir, workspace=ws, options=options)
    except Exception as exc:
        report = None
        error = f"{type(exc).__name__}: {exc}"
    on_disk = await read_json_or_none(ws, report_path)
    written = isinstance(on_disk, dict) and "_json_error" not in on_disk
    report_json: dict[str, Any] | None = on_disk if written else None
    if report is not None:
        returned = report.model_dump(mode="json", by_alias=True)
        report_json = {**returned, **(report_json or {})}
        if not written:
            await write_json(ws, report_path, report_json)
    auxiliary = dict((report_json or {}).get("auxiliary_checks") or {})
    extra_errors: list[str] = []
    extra_warnings: list[str] = []
    if _mode(state) != "fresh":
        from .validation.scope_audit import validate_method_scope_audit_in

        scope = await validate_method_scope_audit_in(ws, paper_dir)
        extra_errors = [str(m) for m in scope.get("hard_errors") or []]
        extra_warnings = [str(m) for m in scope.get("soft_warnings") or []]
        auxiliary["method_scope_audit"] = scope
    feedback = await write_validation_feedback(
        ws,
        paper_dir,
        report=report_json,
        error=error,
        extra_hard_errors=extra_errors,
        extra_soft_warnings=extra_warnings,
        auxiliary_checks=auxiliary or None,
    )
    passed = bool(feedback.get("ok")) and not feedback.get("hard_errors")
    round_ = int(state.get("repair_round") or 0)
    try:
        from .references.diagnostics import archive_round

        await archive_round(
            ws,
            paper_dir,
            "validation",
            round_,
            ["validation_report.json", "validation_feedback.json", "validation_feedback.md"],
        )
    except ImportError:  # pragma: no cover - the host side ships it
        pass
    fingerprint = None
    if not passed:
        from .validation.fingerprint import validation_failure_fingerprint
        from .validation.validator import load_validation_tree

        try:
            fingerprint = validation_failure_fingerprint(
                list(feedback.get("hard_errors") or []),
                await load_validation_tree(ws, paper_dir),
            )
        except RuntimeError:
            fingerprint = None
    validation = report or ValidationReport(
        passed=False, errors=list(feedback.get("hard_errors") or [])
    )
    return {
        "validation": validation,
        "validation_passed": passed,
        "validation_fingerprint": fingerprint,
        "validation_soft_warnings": list(feedback.get("soft_warnings") or []),
    }


async def _after_validate(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    decision = decide_after_validate(
        mode=_mode(state),
        stage=state.get("stage") or "draft",
        passed=bool(state.get("validation_passed")),
        reverted=bool(state.get("reverted")),
        fingerprint=state.get("validation_fingerprint"),
        previous=state.get("previous_failure_fingerprint"),
    )
    return await _apply(decision, state, runtime)


async def _after_audit(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    audit = state.get("audit")
    decision = decide_after_audit(
        mode=_mode(state),
        stage=state.get("stage") or "draft",
        reverted=bool(state.get("reverted")),
        outcome=audit.outcome if audit is not None else "operational_failure",
        repair_round=int(state.get("repair_round") or 0),
        max_rounds=_max_rounds(state),
    )
    return await _apply(decision, state, runtime)


async def _review_check(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    decision = decide_review_check(
        mode=_mode(state),
        review_round=int(state.get("review_round") or 0),
        review_rounds=method_config(state).paper.review_rounds,
        passed=bool(state.get("validation_passed")),
    )
    return {"next": decision.next}


async def _after_review(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    review = state.get("in_loop_review")
    decision = decide_after_review(
        mode=_mode(state),
        outcome=review.outcome if review is not None else "stage_failed",
        review_round=int(state.get("review_round") or 0),
        repair_round=int(state.get("repair_round") or 0),
        max_rounds=_max_rounds(state),
    )
    return await _apply(decision, state, runtime)


async def _figure_quality_gate(
    state: PaperState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    fq = state.get("figure_quality") or FigureQualityState()
    warnings = list(state.get("validation_soft_warnings") or [])
    if not warnings:
        feedback = await read_json_or_none(
            runtime.context.workspace, join(state["paper_dir"], "validation_feedback.json")
        )
        if isinstance(feedback, Mapping) and isinstance(feedback.get("soft_warnings"), list):
            warnings = [str(w) for w in feedback["soft_warnings"]]
    decision = decide_figure_quality(
        passed=bool(state.get("validation_passed")),
        actionable=has_actionable_figure_quality_warnings(warnings),
        attempted=fq.attempted,
        warning_reported=fq.warning_reported,
        repair_round=int(state.get("repair_round") or 0),
        max_rounds=_max_rounds(state),
    )
    return await _apply(decision, state, runtime)


async def _loop_decision(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    audit = state.get("audit")
    decision = decide_loop(
        mode=_mode(state),
        passed=bool(state.get("validation_passed")),
        repeated=bool(state.get("failure_repeated")),
        repair_round=int(state.get("repair_round") or 0),
        max_rounds=_max_rounds(state),
        audit_outcome=audit.outcome if audit is not None else None,
    )
    return await _apply(decision, state, runtime)


async def _finalize(state: PaperState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    paper_dir = state["paper_dir"]
    paths = state.get("artifact_paths") or {}
    investigation = state.get("investigation")
    sources = [join(paper_dir, "global_memory")]
    source = (
        paths.get("global_memory")
        or paths.get("results")
        or (investigation.results_dir if investigation else None)
    )
    if source:
        sources.append(source)
    manifest_path, _ = await finalize_paper_artifacts(
        runtime.context,
        paper_dir=paper_dir,
        manifest_root=state.get("manifest_root") or ".",
        memory_dirs=sources,
    )
    return {
        "status": "complete",
        "paper": paper_result(
            status="complete",
            paper_dir=paper_dir,
            validation=state.get("validation"),
            manifest_path=manifest_path,
        ),
    }


async def _outcome(state: PaperState) -> dict[str, Any]:
    status = state.get("status") or "failed"
    if status == "running":
        status = "failed"
    return {
        "paper": paper_result(
            status="incomplete" if status == "incomplete" else "failed",
            paper_dir=state.get("paper_dir"),
            validation=state.get("validation"),
            reasons=state.get("incomplete_reasons") or [],
            failure=state.get("failure"),
        )
    }


# --- the graph ---------------------------------------------------------------------------------


def build_paper_graph(
    *,
    checkpointer: Checkpointer = None,
    store: BaseStore | None = None,
    overrides: GraphOverrides | None = None,
    name: str = "paper",
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Return the paper phase as a compiled graph.

    The graph runs alone (with the results of earlier phases, with a prebuilt ``paper_data``, or
    in ``edit``/``revision`` mode with ``previous_paper_dir``) or mounted in the pipeline. The
    replaceable slots are ``paper.data``, ``paper.references``, ``paper.postdraft_enrichment``,
    ``paper.figures``, ``paper.validation``, ``paper.audit`` and ``paper.review_gate``; a
    replacement graph shares the keys of :class:`~langgraph_agentic_scientist.paper.models.
    PaperState` and returns the slot's output keys (``paper_data``; ``references``;
    ``postdraft_status`` and ``postdraft_detail``; ``concept``; ``validation``,
    ``validation_passed`` and ``validation_fingerprint``; ``audit``; ``in_loop_review``).

    Args:
        checkpointer: The caller's checkpointer (``None`` when mounted: the parent's is used).
        store: The caller's store (only at the top level).
        overrides: Replacement slots.
        name: The graph's name.
    """
    overrides = overrides or GraphOverrides()
    g = StepGraph(
        "paper", PaperState, input_schema=PaperInput, output_schema=PaperOutput, overrides=overrides
    )
    retry = agent_node_retry_policy()
    g.add_node("record_run_config", record_run_config)
    g.step("prepare", _prepare)
    g.slot("paper_data", _default_paper_data, key="paper.data")
    g.step("check_final_outcome", _check_final_outcome)
    g.step("seed_edit", _seed_edit)
    g.step("stage_inputs", _stage_inputs)
    g.slot(
        "references",
        references_runner(build_references_graph(overrides=overrides)),
        key="paper.references",
    )
    g.add_node("references_failed", _references_failed)
    g.step("strategy", _strategy, retry_policy=retry)
    g.add_node("strategy_repair", _strategy_repair)
    g.add_node("strategy_exhausted", _strategy_exhausted)
    g.step("resolve_family", _resolve_family)
    g.step("rank_posthoc", _rank_posthoc, retry_policy=retry)
    g.add_node("ranking_repair", _ranking_repair)
    g.add_node("ranking_exhausted", _ranking_exhausted)
    g.step("augment_paper_data", _augment_paper_data)
    g.step("init_concept_figure", _init_concept_figure)
    g.step("select_stage", _select_stage)
    g.step("write", _write, retry_policy=retry)
    g.step("after_write", _after_write)
    g.add_node("postdraft_gate", _postdraft_gate)
    g.slot(
        "postdraft", build_postdraft_graph(overrides=overrides), key="paper.postdraft_enrichment"
    )
    g.add_node("after_postdraft", _after_postdraft)
    g.slot("figures", _default_figures, key="paper.figures")
    g.slot("validate", _default_validation, key="paper.validation")
    g.step("after_validate", _after_validate)
    g.slot("audit", build_audit_graph(overrides=overrides), key="paper.audit")
    g.step("after_audit", _after_audit)
    g.add_node("review_check", _review_check)
    g.slot("review_gate", build_review_gate_graph(overrides=overrides), key="paper.review_gate")
    g.step("after_review", _after_review)
    g.step("figure_quality_gate", _figure_quality_gate)
    g.add_node("loop_decision", _loop_decision)
    g.step("finalize", _finalize)
    g.add_node("outcome", _outcome)

    def via_next(*targets: str) -> dict[Hashable, str]:
        return {target: target for target in targets}

    g.add_edge(START, "record_run_config")
    g.add_edge("record_run_config", "prepare")
    g.add_conditional_edges("prepare", _route, via_next("paper_data", "outcome"))
    g.add_edge("paper_data", "check_final_outcome")
    g.add_conditional_edges("check_final_outcome", _route, via_next("seed_edit", "outcome"))
    g.add_edge("seed_edit", "stage_inputs")
    g.add_conditional_edges("stage_inputs", _route, via_next("references", "outcome"))
    g.add_conditional_edges(
        "references",
        _after_references,
        via_next("outcome", "references_failed", "strategy", "rank_posthoc"),
    )
    g.add_edge("references_failed", "outcome")
    g.add_conditional_edges(
        "strategy",
        _after_strategy,
        via_next("resolve_family", "strategy_exhausted", "strategy_repair"),
    )
    g.add_edge("strategy_repair", "strategy")
    g.add_edge("strategy_exhausted", "outcome")
    g.add_conditional_edges("resolve_family", _route, via_next("rank_posthoc", "outcome"))
    g.add_conditional_edges(
        "rank_posthoc",
        _after_ranking,
        via_next("augment_paper_data", "ranking_exhausted", "ranking_repair"),
    )
    g.add_edge("ranking_repair", "rank_posthoc")
    g.add_edge("ranking_exhausted", "outcome")
    g.add_edge("augment_paper_data", "init_concept_figure")
    g.add_edge("init_concept_figure", "select_stage")
    g.add_edge("select_stage", "write")
    g.add_edge("write", "after_write")
    g.add_conditional_edges(
        "after_write", _route, via_next("postdraft_gate", "select_stage", "outcome")
    )
    g.add_conditional_edges("postdraft_gate", _route, via_next("postdraft", "figures"))
    g.add_edge("postdraft", "after_postdraft")
    g.add_conditional_edges("after_postdraft", _route, via_next("figures", "outcome"))
    g.add_edge("figures", "validate")
    g.add_edge("validate", "after_validate")
    g.add_conditional_edges("after_validate", _route, via_next("validate", "audit", "review_check"))
    g.add_edge("audit", "after_audit")
    g.add_conditional_edges(
        "after_audit",
        _route,
        via_next("select_stage", "outcome", "review_check", "figure_quality_gate"),
    )
    g.add_conditional_edges(
        "review_check", _route, via_next("review_gate", "audit", "figure_quality_gate")
    )
    g.add_edge("review_gate", "after_review")
    g.add_conditional_edges("after_review", _route, via_next("select_stage", "outcome"))
    g.add_conditional_edges(
        "figure_quality_gate", _route, via_next("select_stage", "loop_decision")
    )
    g.add_conditional_edges(
        "loop_decision", _route, via_next("finalize", "select_stage", "outcome")
    )
    g.add_edge("finalize", END)
    g.add_edge("outcome", END)
    return g.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "FIGURE_QUALITY_WARNING_MARKERS",
    "Decision",
    "build_paper_graph",
    "decide_after_audit",
    "decide_after_postdraft",
    "decide_after_review",
    "decide_after_validate",
    "decide_after_write",
    "decide_figure_quality",
    "decide_loop",
    "decide_postdraft",
    "decide_review_check",
    "decide_select_stage",
    "has_actionable_figure_quality_warnings",
]
