"""The ideation graph: one cycle's idea (upstream ``idea.sh`` and ``campaign_loop``'s idea loop).

``stage`` prepares a fresh staging directory and the placeholder values of ``idea.md``, ``ideate``
runs the ideator (one session per node execution), ``validate`` applies upstream's artifact checks
and the harness rubric rules. An invalid or failed attempt is a loop back to ``stage`` with the
validation feedback appended to the next prompt, up to ``max_idea_generation_attempts``; when all
fail the graph returns ``exhausted`` and the worker ends the campaign.

The ideator answers through structured output, so the two JSON files upstream's agent wrote are
written here by the host from the validated answer. The agent keeps writing memory itself (the
idea-local ``memory.md``, the global memory files, and through the staged helper the conclusions);
its root is the worker's results root so that those files are within its reach.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..agents.runner import agent_node_retry_policy, run_structured_role
from ..agents.staging import SKILLS_BY_STEP
from ..context import GraphOverrides, ScientistContext
from ..core.render import standard_values
from ..core.runconfig import method_config, record_run_config
from ..core.state import Failure, utc_now
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.paths import join
from .context import (
    BEST_MANIFEST_FILE,
    idea_values,
    read_json_object,
    read_text_or,
)
from .ideas import (
    IdeaArtifactsError,
    IdeaDraft,
    bundle_json,
    correct_incumbent_score,
    dump_json,
    render_validation_feedback,
    sanitize_validation_feedback,
    validate_idea,
)
from .memory_sync import append_memory_event_once, format_memory_markdown_entry
from .models import (
    IdeaFailure,
    IdeationInput,
    IdeationOutput,
    IdeationResult,
    IdeationState,
)


def _staging_dir(state: Mapping[str, Any], attempt: int) -> str:
    cycle = state["cycle"]
    return join(
        state["results_dir"], "_idea_generation", f"cycle_{cycle.index:04d}_attempt_{attempt:02d}"
    )


async def stage_idea_workspace(
    state: IdeationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Create the attempt's staging directory and compute the prompt's placeholder values.

    Paths in the values are relative to the worker's results root (the ideator's root). The
    histories, the best manifest and the conclusions the prompt names are files the worker wrote
    before the cycle (``cycle_start``).
    """
    ctx = runtime.context
    cfg = method_config(state).investigation
    attempt = int(state.get("generation_attempt") or 1)
    staging = _staging_dir(state, attempt)
    results_dir = state["results_dir"]
    task = state["task"]
    await ctx.workspace.write_bytes(join(staging, ".staging"), b"")
    values = await idea_values(
        ctx.workspace,
        ctx.prompts,
        results_dir=results_dir,
        staging_dir=staging,
        cycle_index=state["cycle"].index,
        worker_index=state["worker_index"],
        harness=task.harness_evaluated,
        contract=task.contract,
        max_pattern_chars=cfg.max_source_pattern_chars,
    )
    task_md = await read_text_or(ctx.workspace, join(task.root, "task.md"), "") or None
    values.update(
        standard_values(
            task_dir="task", results_dir=".", task_md=task_md, task_md_path="task/task.md"
        )
    )
    return {"generation_attempt": attempt, "staging_dir": staging, "prompt_values": values}


async def ideate(state: IdeationState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run one ideator session (R13) and keep its structured answer."""
    cfg = method_config(state).investigation
    values = state["prompt_values"]
    cycle = state["cycle"]
    suffix = render_validation_feedback(
        runtime.context.prompts,
        state.get("validation_feedback", ""),
        max_bytes=cfg.validation_feedback_max_bytes,
    )
    outcome = await run_structured_role(
        role=Role.IDEATOR,
        schema=IdeaDraft,
        runtime=runtime,
        root=state["results_dir"],
        read_only=False,
        prompt_key="investigation/idea",
        values=values,
        suffix=suffix,
        skills=SKILLS_BY_STEP["idea"],
        skill_values=values,
        timeout=cfg.idea_timeout_seconds,
        env={
            "CYCLE_INDEX": str(cycle.index),
            "WORKER_INDEX": str(state["worker_index"]),
        },
        pool_index=cycle.model_index,
    )
    draft = outcome.structured.model_dump() if outcome.structured is not None else None
    detail = (
        "" if outcome.failure is None else f"{outcome.failure.category}: {outcome.failure.detail}"
    )
    return {
        "draft": draft,
        "session_failed": draft is None,
        "session_detail": detail,
        "timed_out": outcome.timed_out,
    }


async def validate_idea_node(
    state: IdeationState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Validate the ideator's answer; write the idea files; or schedule a retry, or give up."""
    ctx = runtime.context
    cfg = method_config(state).investigation
    cycle = state["cycle"]
    attempt = int(state.get("generation_attempt") or 1)
    task = state["task"]
    harness = task.harness_evaluated
    prior = list(state.get("failures") or [])
    failure_reason = ""
    detail = ""
    feedback = ""
    draft = state.get("draft")
    if state.get("session_failed") or draft is None:
        failure_reason = "idea_generation_exit_nonzero"
        detail = str(state.get("session_detail") or "")
    else:
        best = await read_json_object(ctx.workspace, join(state["results_dir"], BEST_MANIFEST_FILE))
        try:
            bundle = validate_idea(
                draft,
                cycle.index,
                harness=harness,
                incumbent_score=correct_incumbent_score(best, harness=harness),
                max_harness_rubrics=cfg.max_harness_rubrics,
                max_pattern_chars=cfg.max_source_pattern_chars,
            )
        except IdeaArtifactsError as exc:
            failure_reason = exc.reason
            detail = exc.detail
            feedback = sanitize_validation_feedback(
                str(exc), max_bytes=cfg.validation_feedback_max_bytes
            )
        else:
            staging = state["staging_dir"]
            payload = bundle_json(bundle)
            await ctx.workspace.write_bytes(
                join(staging, "idea_bundle.json"), dump_json(payload).encode("utf-8")
            )
            await ctx.workspace.write_bytes(
                join(staging, "target_rubrics.json"),
                dump_json(payload["target_rubrics"]).encode("utf-8"),
            )
            event = {
                "timestamp": utc_now(),
                "step": "idea",
                "scope": "idea",
                "summary": "generated idea bundle (auto-healed by idea.sh)",
                "evidence": "idea_bundle.json + target_rubrics.json",
                "next_action": "run attempt",
                "idea_id": bundle.idea_id,
            }
            await append_memory_event_once(
                ctx.workspace,
                state["results_dir"],
                event,
                markdown=format_memory_markdown_entry(event),
                match_keys=("step", "idea_id"),
                marker=f"- idea_id: {bundle.idea_id}",
            )
            return {
                "ideation": IdeationResult(
                    status="accepted",
                    bundle=bundle,
                    idea_workspace=staging,
                    failures=prior,
                    attempts_used=attempt,
                )
            }
    failure = IdeaFailure(
        cycle=cycle.index,
        generation_attempt=attempt,
        reason=failure_reason,
        detail=detail,
        timed_out=bool(state.get("timed_out")),
    )
    if attempt < cfg.max_idea_generation_attempts:
        return {
            "failures": [failure],
            "generation_attempt": attempt + 1,
            "validation_feedback": feedback,
        }
    return {
        "failures": [failure],
        "ideation": IdeationResult(
            status="exhausted",
            failures=[*prior, failure],
            attempts_used=attempt,
            failure=Failure(
                category="campaign_failed",
                scope="method",
                detail="idea_generation_attempts_exhausted",
            ),
        ),
    }


def _route(state: IdeationState) -> str:
    return END if state.get("ideation") is not None else "stage"


def build_ideation_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "ideation",
) -> CompiledStateGraph[Any, ScientistContext, IdeationInput, IdeationOutput]:
    """Build the graph of one cycle's ideation (upstream ``idea.sh`` with its retries).

    Steps (hook keys ``ideation.<step>``): ``stage``, ``ideate``, ``validate``. The graph can run
    alone given a worker's results root and a cycle; the worker mounts it as the slot
    ``investigation.ideation``. A replacement must accept :class:`IdeationInput` and return
    :class:`IdeationOutput`.

    Args:
        checkpointer: The checkpointer when run at the top level; ``None`` inherits the parent's.
        store: The store when run at the top level.
        overrides: Replacement slots and hooks' graph overrides.
        name: The compiled graph's name.
    """
    sg = StepGraph(
        "ideation",
        IdeationState,
        input_schema=IdeationInput,
        output_schema=IdeationOutput,
        overrides=overrides,
    )
    sg.add_node("record_run_config", record_run_config)
    sg.step("stage", stage_idea_workspace, hook_writable=("prompt_values",))
    sg.step("ideate", ideate, retry_policy=agent_node_retry_policy())
    sg.step("validate", validate_idea_node)
    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "stage")
    sg.add_edge("stage", "ideate")
    sg.add_edge("ideate", "validate")
    sg.add_conditional_edges("validate", _route, {END: END, "stage": "stage"})
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = ["build_ideation_graph", "ideate", "stage_idea_workspace", "validate_idea_node"]
