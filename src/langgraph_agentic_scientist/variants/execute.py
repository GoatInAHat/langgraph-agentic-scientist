"""Stage 5: author variants with agents and evaluate them on the host (upstream ``execute.py``
and the evaluation semantics of ``host_eval.py``).

One variant is one run of :func:`build_run_variant_graph`. Per program index (``program_0`` for the
control, ``program_1..max_evals_per_variant`` for every other variant):

1. **edit** (not for the control): the *variant editor* (R19, skills ``edit-program`` and
   ``memory-log``, the ``clone_program`` tool) creates ``program_<n>`` from ``program_<n-1>``. A
   session that ends without the directory is tried again, at most ``edit_max_tries`` (3) times;
   each try is its own node execution, so it has its own checkpoint namespace.
2. **evaluate**: the host evaluates through ``ctx.evaluator`` (plane ``variant``, the configured
   replications and timeout), checks that the sealed evaluator tree did not change before or
   after, and publishes the evaluation: ``metrics.json``, ``correct.json``, the evidence, the
   canonical analysis CSVs and ``evaluation_manifest.json`` under ``program_<n>/``.
3. **scaffold, verify, canonicalise**: the host pre-creates ``rubric_status.json`` with the exact
   rubric ids; the *variant verifier* (R20, skills ``verify-rubrics`` and ``memory-log``) fills
   each rubric's ``pass`` and ``evidence``; the host builds the canonical status from those
   judgments (at most ``verify_max_tries`` (3) tries) and writes it, with the root mirror and
   ``overall_pass`` in ``run_summary.csv``. A status that never became valid is replaced by an
   all-fail status.
4. The variant stops at the first program that passes, at a timeout, at an evaluation that did not
   complete, or after its last program index.

Every retry is a graph loop with its counter in state; every evaluation has a deterministic id,
so a resumed run asks the evaluator again for the same evaluation and never runs one twice. A
crash after the evaluator answered and before the files were published is recognised by the
``.evaluation_pending`` marker and the publication is redone.
"""

from __future__ import annotations

import json
import logging
import operator
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from ..agents.runner import agent_node_retry_policy, run_workspace_role
from ..agents.tools import make_clone_program_tool
from ..context import GraphOverrides, ScientistContext
from ..core.errors import ProgramSpecError
from ..core.render import truncate_utf8
from ..core.results import VariantResult, VariantSpec
from ..core.runconfig import method_config
from ..core.state import (
    EvaluationRecord,
    Failure,
    ProgramRef,
    RunConfigRecord,
    TaskRef,
    merge_dicts,
    utc_now,
)
from ..core.steps import StepGraph
from ..evaluation.ids import evaluation_id
from ..evaluation.program_spec import resolve_entrypoint
from ..evaluation.protocol import EvaluationRequest, EvaluationResult
from ..models import Role
from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace, is_dir, is_file, write_text
from .analysis import publish_summary_qualification, write_canonical_analysis
from .constants import (
    ANALYSIS_CSV_SCHEMAS,
    EVALUATION_PENDING_FILENAME,
    EVALUATOR_ARTIFACTS,
    MAX_VERIFICATION_ERROR_LENGTH,
    SAFE_RUBRIC_ID,
)
from .digests import managed_evaluator_tree_digest
from .models import (
    VariantRun,
    spec_detail,
    spec_to_dict,
)
from .preflight import validate_candidate, validate_reference_workspace
from .roles import (
    agent_environment,
    agent_name,
    model_label,
    trusted_environment,
    variant_skill_values,
)
from .rubrics import (
    canonicalize_variant_rubric_status,
    load_rubric_status,
    required_rubric_ids,
    rubric_status_record,
    variant_attempt_id,
)
from .workspace_ops import (
    program_dirs,
    read_json_object,
    read_text_or_none,
    stage_active_files,
    write_json,
)

logger = logging.getLogger(__name__)

EDIT_PROMPT_KEY = "variants/variant_edit"
VERIFY_PROMPT_KEY = "variants/variant_verify"
EDIT_SKILLS = ("edit-program", "memory-log")
VERIFY_SKILLS = ("verify-rubrics", "memory-log")
NOTE_BYTES = 4096
DEFAULT_PREVIOUS_STATUS = "(none; first candidate)"
DEFAULT_PREVIOUS_ERROR = "(none; first verification attempt)"
PASSED_STATUS = "completed"

Route = Literal["edit", "evaluate", "scaffold", "verified", "next", "finish"]


class RunVariantInput(TypedDict):
    """What :func:`build_run_variant_graph` is given (the ``Send`` argument of the phase).

    Attributes:
        spec: The materialised variant.
        task: The task whose evaluator evaluates the variant (the original task, or an extra task
            in a cross-task adaptation).
        task_bundle: Workspace path of the variant's ``task_bundle``.
        id_parts: Components of the evaluation ids, before the variant id (for example
            ``["variants"]`` or ``["extra", "<task>"]``).
        run_id: The run id the evaluation ids start with.
        run_config: The recorded method configuration.
        evaluator_digest: The sealed evaluator tree digest to hold (``None`` skips the check).
        agent_timeout: Wall clock of each agent session in seconds (``None``: unbounded).
        task_name: The extra task's name, in a cross-task adaptation.
    """

    spec: VariantSpec
    task: TaskRef
    task_bundle: str
    id_parts: list[str]
    run_id: str
    run_config: RunConfigRecord | None
    evaluator_digest: str | None
    agent_timeout: float | None
    task_name: NotRequired[str | None]


class RunVariantOutput(TypedDict):
    """What the run graph returns: the variant's run."""

    variant_runs: list[VariantRun]


class RunVariantState(RunVariantInput, total=False):
    """State of one variant's run (see the module docstring for the loop)."""

    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    target_rubric_ids: list[str]
    first_index: int
    last_index: int
    program_index: int
    edit_attempt: int
    verify_attempt: int
    eval_count: int
    exit_code: int
    timed_out: bool
    route: Route
    notes: Annotated[list[str], operator.add]
    evaluations: Annotated[list[EvaluationRecord], operator.add]
    current_evaluation: EvaluationResult | None
    verification_error: str | None
    rubric_valid: bool
    rubric_passed: bool
    best_index: int | None
    last_status: dict[str, Any] | None
    infrastructure_failure: Failure | None
    variant_runs: Annotated[list[VariantRun], operator.add]


def run_variant_input(
    *,
    spec: VariantSpec,
    task: TaskRef,
    run_id: str,
    run_config: RunConfigRecord | None,
    evaluator_digest: str | None,
    agent_timeout: float | None,
    id_parts: list[str],
    task_name: str | None = None,
) -> RunVariantInput:
    """Build the ``Send`` argument of one variant run."""
    return {
        "spec": spec,
        "task": task,
        "task_bundle": join(spec.variant_dir, "task_bundle"),
        "id_parts": id_parts,
        "run_id": run_id,
        "run_config": run_config,
        "evaluator_digest": evaluator_digest,
        "agent_timeout": agent_timeout,
        "task_name": task_name,
    }


# --- manifest ------------------------------------------------------------------------------------


async def update_manifest_status(
    workspace: Workspace,
    variant_dir: str,
    status: str,
    eval_count: int | None = None,
    *,
    agent: str | None = None,
    agent_model: str | None = None,
) -> None:
    """Update the status fields of ``run_manifest.json`` (upstream ``_update_manifest_status``)."""
    path = join(variant_dir, "run_manifest.json")
    manifest = await read_json_object(workspace, path)
    if manifest is None:
        return
    now = datetime.now(UTC).isoformat()
    manifest["status"] = status
    if status == "running":
        manifest["started_at"] = now
        manifest["completed_at"] = None
    elif status in ("completed", "failed", "timeout"):
        manifest["completed_at"] = now
    if eval_count is not None:
        manifest["eval_count"] = eval_count
    if agent is not None:
        manifest.update(
            {
                "agent": agent,
                "agent_model": agent_model,
                "agent_profile": None,
                "reasoning_effort": None,
            }
        )
    await write_json(workspace, path, manifest)


# --- prompts -------------------------------------------------------------------------------------


def verification_error_context(error: str) -> str:
    """Return a one-line error of at most 256 characters for the next verify prompt."""
    single_line = " ".join(error.split())
    return single_line[:MAX_VERIFICATION_ERROR_LENGTH]


async def previous_rubric_summary(
    workspace: Workspace,
    prompts: PromptRegistry,
    task_bundle: str,
    program_index: int,
    spec: VariantSpec,
) -> str:
    """Summarise the previous program's rubric status for the next edit prompt."""
    status = await load_rubric_status(
        workspace, join(task_bundle, f"program_{program_index}", "rubric_status.json")
    )
    if status is None:
        return DEFAULT_PREVIOUS_STATUS
    trusted = required_rubric_ids(
        prompts, spec.variant_id, spec.kind, str(spec_detail(spec, "base_program"))
    )
    entries = status.get("rubrics")
    by_id: dict[object, object] = {}
    if isinstance(entries, list):
        by_id = {entry.get("id"): entry.get("pass") for entry in entries if isinstance(entry, dict)}
    summary = {
        "overall_pass": status.get("overall_pass") is True,
        "failed_rubric_ids": [
            rubric_id for rubric_id in trusted if by_id.get(rubric_id) is not True
        ],
    }
    return json.dumps(summary, sort_keys=True)


def build_edit_prompt(
    prompts: PromptRegistry,
    *,
    spec: VariantSpec,
    program_index: int,
    edit_attempt: int,
    previous_status: str,
) -> str:
    """Render the variant editor's prompt (upstream ``_build_host_edit_prompt``)."""
    return str(
        prompts.render(
            EDIT_PROMPT_KEY,
            {
                "VARIANT_SPEC_JSON": json.dumps(spec_to_dict(spec), indent=2),
                "PROGRAM_INDEX": str(program_index),
                "PREVIOUS_INDEX": str(program_index - 1),
                "EDIT_ATTEMPT": str(edit_attempt),
                "PREVIOUS_STATUS": previous_status or DEFAULT_PREVIOUS_STATUS,
            },
            fragments=False,
        )
    )


def build_verify_prompt(
    prompts: PromptRegistry,
    *,
    spec: VariantSpec,
    program_index: int,
    evaluation_correct: bool,
    target_rubric_ids: list[str],
    previous_error: str | None,
) -> str:
    """Render the variant verifier's prompt (upstream ``_build_host_verify_prompt``)."""
    return str(
        prompts.render(
            VERIFY_PROMPT_KEY,
            {
                "PROGRAM_INDEX": str(program_index),
                "VARIANT_ID": spec.variant_id,
                "VARIANT_KIND": spec.kind,
                "EVALUATION_CORRECT": str(evaluation_correct),
                "REQUIRED_RUBRIC_IDS": json.dumps(target_rubric_ids, indent=2),
                "PREVIOUS_ERROR": previous_error or DEFAULT_PREVIOUS_ERROR,
            },
            fragments=False,
        )
    )


# --- rubric artifacts ----------------------------------------------------------------------------


async def target_rubric_ids(
    workspace: Workspace, variant_dir: str, expected: list[str]
) -> list[str]:
    """Check ``target_rubrics.json`` against the trusted contract and return the ids.

    Raises:
        ValueError: When the file is unreadable, duplicated, unsafe or not the contract's set.
    """
    path = join(variant_dir, "target_rubrics.json")
    text = await read_text_or_none(workspace, path)
    try:
        payload = json.loads(text) if text is not None else None
    except json.JSONDecodeError:
        payload = None
    if payload is None:
        raise ValueError(f"invalid target_rubrics.json: {path}")
    if not isinstance(payload, list):
        raise ValueError("target_rubrics.json must contain a list")
    rubric_ids = [
        item["id"] for item in payload if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if (
        len(rubric_ids) != len(payload)
        or len(set(rubric_ids)) != len(rubric_ids)
        or any(SAFE_RUBRIC_ID.fullmatch(rubric_id) is None for rubric_id in rubric_ids)
        or set(rubric_ids) != set(expected)
    ):
        raise ValueError("target_rubrics.json does not match the trusted rubric contract")
    return expected


async def clear_rubric_status(workspace: Workspace, task_bundle: str, program_index: int) -> None:
    """Remove the root and the program's ``rubric_status.json``."""
    await workspace.remove(join(task_bundle, "rubric_status.json"))
    await workspace.remove(join(task_bundle, f"program_{program_index}", "rubric_status.json"))


def _metrics_block(evaluation: EvaluationResult) -> dict[str, Any]:
    return {
        "combined_score": evaluation.metrics.get("combined_score"),
        "correct": bool(evaluation.correct),
    }


async def write_rubric_scaffold(
    workspace: Workspace,
    task_bundle: str,
    program_index: int,
    variant_id: str,
    rubric_ids: list[str],
    evaluation: EvaluationResult,
) -> None:
    """Pre-create the program's ``rubric_status.json`` with every rubric failing and empty."""
    scaffold = {
        "variant_id": variant_id,
        "attempt_id": variant_attempt_id(variant_id, program_index),
        "idea_id": variant_id,
        "overall_pass": False,
        "rubrics": [{"id": rubric_id, "pass": False, "evidence": ""} for rubric_id in rubric_ids],
        "metrics": _metrics_block(evaluation),
        "next_recommendation": "",
    }
    await write_text(
        workspace,
        join(task_bundle, f"program_{program_index}", "rubric_status.json"),
        json.dumps(scaffold, indent=2, sort_keys=True) + "\n",
    )


def failed_rubric_status(
    variant_id: str,
    program_index: int,
    rubric_ids: list[str],
    evaluation: EvaluationResult,
    verification_error: str | None,
) -> dict[str, Any]:
    """Return the all-fail status written when verification never produced a valid one."""
    context = verification_error or "verifier did not produce valid rubric judgments"
    return {
        "variant_id": variant_id,
        "attempt_id": variant_attempt_id(variant_id, program_index),
        "idea_id": variant_id,
        "overall_pass": False,
        "rubrics": [
            {"id": rubric_id, "pass": False, "evidence": f"Verification incomplete: {context}"}
            for rubric_id in rubric_ids
        ],
        "metrics": _metrics_block(evaluation),
        "next_recommendation": "Repair rubric verification before another evaluation.",
    }


async def write_canonical_rubric_status(
    workspace: Workspace,
    prompts: PromptRegistry,
    task_bundle: str,
    program_index: int,
    rubric_status: Mapping[str, Any],
    max_evidence: int,
) -> None:
    """Write the host-canonical status to the program and to the workspace mirror, then qualify."""
    content = json.dumps(rubric_status, indent=2, sort_keys=True) + "\n"
    await write_text(
        workspace, join(task_bundle, f"program_{program_index}", "rubric_status.json"), content
    )
    await write_text(workspace, join(task_bundle, "rubric_status.json"), content)
    await publish_summary_qualification(
        workspace,
        join(task_bundle, f"program_{program_index}"),
        prompts=prompts,
        max_evidence=max_evidence,
    )


# --- evaluation publication ----------------------------------------------------------------------


async def discard_evaluation_artifacts(workspace: Workspace, program_dir: str) -> None:
    """Remove every host-owned evaluation output of a program directory."""
    for artifact in (".harness_results", EVALUATION_PENDING_FILENAME, *EVALUATOR_ARTIFACTS):
        await workspace.remove(join(program_dir, artifact))


async def published_for(workspace: Workspace, program_dir: str, request_id: str) -> bool:
    """Whether the program already holds a complete publication of evaluation ``request_id``."""
    if await workspace.stat(join(program_dir, EVALUATION_PENDING_FILENAME)) is not None:
        return False
    manifest = await read_json_object(workspace, join(program_dir, "evaluation_manifest.json"))
    return manifest is not None and manifest.get("evaluation_id") == request_id


async def publish_evaluation(
    workspace: Workspace,
    program_dir: str,
    result: EvaluationResult,
    *,
    request_id: str,
    environment: Mapping[str, str],
) -> None:
    """Publish a completed evaluation under ``program_dir`` (the manifest comes last).

    ``.evaluation_pending`` marks the publication as unfinished until the manifest is written, so
    a crash in between is recognised and the publication is redone.

    Raises:
        RuntimeError: When the evaluator's output violates upstream's contracts (see
            :func:`~langgraph_agentic_scientist.variants.analysis.write_canonical_analysis`).
    """
    await workspace.write_bytes(
        join(program_dir, EVALUATION_PENDING_FILENAME), request_id.encode("utf-8")
    )
    await write_canonical_analysis(
        workspace, program_dir=program_dir, result=result, environment=environment
    )
    payloads = {
        "metrics.json": json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
        "correct.json": json.dumps({"correct": result.correct}, indent=2, sort_keys=True) + "\n",
    }
    if result.evidence is not None:
        payloads["evaluation_evidence.json"] = (
            result.evidence.model_dump_json(by_alias=True, indent=2) + "\n"
        )
    for name, text in payloads.items():
        await write_text(workspace, join(program_dir, name), text)
    manifest = {
        "producer": "isolated_evaluator",
        "schema_version": 1,
        "program_id": program_dir.rsplit("/", 1)[-1],
        "output_dir": result.output_dir or f"evaluator:{request_id}",
        "returncode": 0,
        "published_summaries": ["metrics.json", "correct.json"],
        "evaluation_id": request_id,
        "variant_evaluation_id": environment.get("EVALUATION_ID", ""),
        "started_at": utc_now(),
    }
    await write_text(
        workspace,
        join(program_dir, "evaluation_manifest.json"),
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    await workspace.remove(join(program_dir, EVALUATION_PENDING_FILENAME))


async def check_analysis_files(
    workspace: Workspace, task_bundle: str
) -> dict[str, bool]:
    """Report which analysis files exist across the ``program_<n>`` directories."""
    result: dict[str, bool] = {}
    for name in await program_dirs(workspace, task_bundle):
        program = join(task_bundle, name)
        for filename in ANALYSIS_CSV_SCHEMAS:
            exists = await is_file(workspace, join(program, "analysis", filename))
            if exists:
                result[filename] = True
            elif filename not in result:
                result[filename] = False
        if await is_file(workspace, join(program, "rubric_status.json")):
            result["rubric_status.json"] = True
        elif "rubric_status.json" not in result:
            result["rubric_status.json"] = False
    return result


# --- the graph -----------------------------------------------------------------------------------


def _cfg(state: Mapping[str, Any]) -> Any:
    return method_config(state).variants


def _note(text: str) -> list[str]:
    return [truncate_utf8(text, NOTE_BYTES)]


def _spec_base_program(spec: VariantSpec) -> str:
    return str(spec_detail(spec, "base_program"))


async def init_variant(
    state: RunVariantState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Prepare a variant run: manifest, rubric ids, reference workspace and the first index."""
    ctx = runtime.context
    ws = ctx.workspace
    spec = state["spec"]
    cfg = _cfg(state)
    await update_manifest_status(
        ws,
        spec.variant_dir,
        "running",
        agent="langgraph-deepagents",
        agent_model=model_label(ctx, Role.VARIANT_EDITOR),
    )
    base_program = _spec_base_program(spec)
    expected = required_rubric_ids(ctx.prompts, spec.variant_id, spec.kind, base_program)
    try:
        ids = await target_rubric_ids(ws, spec.variant_dir, expected)
        await validate_reference_workspace(ws, state["task_bundle"], spec.variant_id, base_program)
    except ValueError as exc:
        return {
            "route": "finish",
            "exit_code": 1,
            "notes": _note(str(exc)),
            "target_rubric_ids": expected,
            "first_index": 0,
            "last_index": 0,
            "program_index": 0,
            "eval_count": 0,
        }
    first = 0 if spec.kind == "control" else 1
    last = first if spec.kind == "control" else first + cfg.max_evals_per_variant - 1
    return {
        "target_rubric_ids": ids,
        "first_index": first,
        "last_index": last,
        "program_index": first,
        "edit_attempt": 1,
        "verify_attempt": 1,
        "eval_count": 0,
        "exit_code": 1,
        "timed_out": False,
        "rubric_valid": False,
        "rubric_passed": False,
        "route": "evaluate" if first == 0 else "edit",
    }


async def edit(state: RunVariantState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run one session of the variant editor (R19): create ``program_<n>``."""
    ctx = runtime.context
    ws = ctx.workspace
    spec = state["spec"]
    cfg = _cfg(state)
    index = state["program_index"]
    bundle = state["task_bundle"]
    previous = await previous_rubric_summary(ws, ctx.prompts, bundle, index - 1, spec)
    prompt = build_edit_prompt(
        ctx.prompts,
        spec=spec,
        program_index=index,
        edit_attempt=state["edit_attempt"],
        previous_status=previous,
    )
    await stage_active_files(
        ws,
        bundle,
        rubrics_from=join(spec.variant_dir, "target_rubrics.json"),
        idea_from=join(spec.variant_dir, "variant_spec.json"),
    )
    values = variant_skill_values(
        task_md=await read_text_or_none(ws, join(bundle, "task.md")),
        attempt_id=f"variant_{spec.variant_id}",
        variant_id=spec.variant_id,
        max_evals=cfg.max_evals_per_variant,
        active_idea=(await read_text_or_none(ws, join(bundle, "active_idea.json")) or "{}").strip(),
        active_rubrics=(
            await read_text_or_none(ws, join(bundle, "active_rubrics.json")) or "[]"
        ).strip(),
    )
    env = agent_environment(
        variant_id=spec.variant_id,
        variant_kind=spec.kind,
        base_program=_spec_base_program(spec),
        attempt_id=f"variant_{spec.variant_id}",
        max_evals=cfg.max_evals_per_variant,
        num_replications=cfg.num_replications,
        evaluation_id=_last_label(state),
    )
    outcome = await run_workspace_role(
        role=Role.VARIANT_EDITOR,
        runtime=runtime,
        root=bundle,
        prompt=prompt,
        skills=EDIT_SKILLS,
        skill_values=values,
        tools=[make_clone_program_tool(ws, bundle, prompts=ctx.prompts)],
        timeout=state["agent_timeout"],
        env=env,
    )
    update: dict[str, Any] = {"timed_out": outcome.timed_out}
    if outcome.timed_out:
        update["exit_code"] = 124
        update["notes"] = _note(f"variant editor timed out ({spec.variant_id}, program_{index})")
    elif outcome.failure is not None:
        update["exit_code"] = 1
        update["notes"] = _note(
            f"variant editor failed ({outcome.failure.category}): {outcome.failure.detail}"
        )
    else:
        update["exit_code"] = 0
        if outcome.blocker:
            update["notes"] = _note(f"variant editor reported a blocker: {outcome.blocker}")
    return update


def _last_label(state: Mapping[str, Any]) -> str | None:
    count = state.get("eval_count", 0)
    if not count:
        return None
    return f"{state['spec'].variant_id}_eval_{count}"


async def after_edit(state: RunVariantState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Decide after an edit session: evaluate, try again, or finish (upstream's edit loop)."""
    ws = runtime.context.workspace
    cfg = _cfg(state)
    if state.get("timed_out"):
        return {"route": "finish", "exit_code": 124}
    created = await is_dir(ws, join(state["task_bundle"], f"program_{state['program_index']}"))
    if created:
        return {"route": "evaluate"}
    if state["edit_attempt"] < cfg.edit_max_tries:
        return {"route": "edit", "edit_attempt": state["edit_attempt"] + 1}
    return {
        "route": "finish",
        "notes": _note(
            f"program_{state['program_index']} was not created after {cfg.edit_max_tries} tries"
        ),
    }


async def evaluate(state: RunVariantState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Evaluate ``program_<n>`` through the context's evaluator and publish the evaluation."""
    ctx = runtime.context
    ws = ctx.workspace
    spec = state["spec"]
    cfg = _cfg(state)
    index = state["program_index"]
    bundle = state["task_bundle"]
    program_name = f"program_{index}"
    program_dir = join(bundle, program_name)
    label = f"{spec.variant_id}_eval_{state['eval_count'] + 1}"
    request_id = evaluation_id(state["run_id"], *state["id_parts"], spec.variant_id, program_name)
    evaluator = ctx.evaluator
    if evaluator is None:
        return {
            "route": "finish",
            "exit_code": 1,
            "notes": _note("no evaluator is configured for this run"),
        }
    ready = await published_for(ws, program_dir, request_id)
    pending_ours = (
        await read_text_or_none(ws, join(program_dir, EVALUATION_PENDING_FILENAME)) == request_id
    )
    try:
        if not ready and not pending_ours:
            _, entrypoint = await validate_candidate(ws, bundle, index)
        else:
            entrypoint = await resolve_entrypoint(ws, program_dir)
        await _verify_evaluator_source(ws, state)
    except (ValueError, ProgramSpecError, OSError) as exc:
        logger.error("Variant %s: %s", spec.variant_id, exc)
        return {"route": "finish", "exit_code": 1, "notes": _note(str(exc))}

    environment = dict(
        trusted_environment(
            variant_id=spec.variant_id,
            variant_kind=spec.kind,
            base_program=_spec_base_program(spec),
            agent=agent_name(ctx, Role.VARIANT_EDITOR),
            evaluation_id=label,
            num_replications=cfg.num_replications,
        )
    )
    timeout = (
        state["agent_timeout"]
        if cfg.evaluation_timeout_seconds is None
        else float(cfg.evaluation_timeout_seconds)
    )
    request = EvaluationRequest(
        evaluation_id=request_id,
        plane="variant",
        program=ProgramRef(path=program_dir, entrypoint=entrypoint, digest=None),
        task=state["task"],
        num_replications=cfg.num_replications,
        timeout_seconds=timeout,
        environment=environment,
    )
    # The evaluator is keyed by the id: a resumed node gets the recorded (or re-attached) result.
    result = await evaluator.evaluate(request)
    record = EvaluationRecord(
        evaluation_id=request_id,
        program=request.program,
        plane="variant",
        status=result.status,
        metrics=result.metrics,
        correct=result.correct,
        evidence_available=result.evidence is not None,
        failure_category=result.failure_category,
        summary=result.summary,
        attempt_id=variant_attempt_id(spec.variant_id, index),
        idea_id=spec.variant_id,
    )
    base: dict[str, Any] = {"evaluations": [record], "current_evaluation": result}
    if result.status == "timeout":
        return {
            **base,
            "route": "finish",
            "timed_out": True,
            "exit_code": 124,
            "notes": _note(result.summary or f"evaluation timed out ({request_id})"),
        }
    if result.status == "infrastructure_error":
        failure = Failure(
            category=result.failure_category or "evaluator_infrastructure_error",
            scope="infrastructure",
            detail=truncate_utf8(result.summary, NOTE_BYTES),
        )
        return {
            **base,
            "route": "finish",
            "exit_code": 1,
            "infrastructure_failure": failure,
            "notes": _note(f"evaluator infrastructure failure: {result.summary}"),
        }
    if result.status != "completed":
        return {
            **base,
            "route": "finish",
            "exit_code": 1,
            "notes": _note(
                f"evaluation {result.status} ({result.failure_category or 'no category'}): "
                f"{result.summary}"
            ),
        }
    try:
        await publish_evaluation(
            ws, program_dir, result, request_id=request_id, environment=environment
        )
        await _verify_evaluator_source(ws, state)
    except (OSError, RuntimeError, ValueError) as exc:
        await discard_evaluation_artifacts(ws, program_dir)
        logger.error("Variant %s host evaluation failed: %s", spec.variant_id, exc)
        return {**base, "route": "finish", "exit_code": 1, "notes": _note(str(exc))}
    return {
        **base,
        "route": "scaffold",
        "eval_count": state["eval_count"] + 1,
        "exit_code": result.returncode or 0,
        "verify_attempt": 1,
        "verification_error": None,
        "rubric_valid": False,
        "rubric_passed": False,
        "best_index": index,
    }


async def _verify_evaluator_source(ws: Workspace, state: Mapping[str, Any]) -> None:
    expected = state.get("evaluator_digest")
    if expected is None:
        return
    if await managed_evaluator_tree_digest(ws, state["task"].root) != expected:
        raise ValueError("sealed evaluator source tree changed before evaluation")


async def scaffold_status(
    state: RunVariantState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Clear the rubric status and pre-create the scaffold the verifier fills in."""
    ws = runtime.context.workspace
    evaluation = state["current_evaluation"]
    assert evaluation is not None
    index = state["program_index"]
    await clear_rubric_status(ws, state["task_bundle"], index)
    await write_rubric_scaffold(
        ws,
        state["task_bundle"],
        index,
        state["spec"].variant_id,
        state["target_rubric_ids"],
        evaluation,
    )
    return {}


async def verify(state: RunVariantState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Run one session of the variant verifier (R20): fill the scaffold's judgments."""
    ctx = runtime.context
    ws = ctx.workspace
    spec = state["spec"]
    cfg = _cfg(state)
    bundle = state["task_bundle"]
    evaluation = state["current_evaluation"]
    assert evaluation is not None
    prompt = build_verify_prompt(
        ctx.prompts,
        spec=spec,
        program_index=state["program_index"],
        evaluation_correct=bool(evaluation.correct),
        target_rubric_ids=state["target_rubric_ids"],
        previous_error=state.get("verification_error"),
    )
    await stage_active_files(
        ws,
        bundle,
        rubrics_from=join(spec.variant_dir, "target_rubrics.json"),
        idea_from=join(spec.variant_dir, "variant_spec.json"),
    )
    values = variant_skill_values(
        task_md=await read_text_or_none(ws, join(bundle, "task.md")),
        attempt_id=f"variant_{spec.variant_id}",
        variant_id=spec.variant_id,
        max_evals=cfg.max_evals_per_variant,
        active_idea=(await read_text_or_none(ws, join(bundle, "active_idea.json")) or "{}").strip(),
        active_rubrics=(
            await read_text_or_none(ws, join(bundle, "active_rubrics.json")) or "[]"
        ).strip(),
    )
    env = agent_environment(
        variant_id=spec.variant_id,
        variant_kind=spec.kind,
        base_program=_spec_base_program(spec),
        attempt_id=f"variant_{spec.variant_id}",
        max_evals=cfg.max_evals_per_variant,
        num_replications=cfg.num_replications,
        evaluation_id=_last_label(state),
    )
    outcome = await run_workspace_role(
        role=Role.VARIANT_VERIFIER,
        runtime=runtime,
        root=bundle,
        prompt=prompt,
        skills=VERIFY_SKILLS,
        skill_values=values,
        timeout=state["agent_timeout"],
        env=env,
    )
    if outcome.timed_out:
        return {
            "timed_out": True,
            "exit_code": 124,
            "route": "next",
            "notes": _note(f"variant verifier timed out ({spec.variant_id})"),
        }
    if outcome.failure is not None:
        return {
            "exit_code": 1,
            "route": "scaffold",
            "verification_error": verification_error_context("verifier exited with code 1"),
            "notes": _note(
                f"variant verifier failed ({outcome.failure.category}): {outcome.failure.detail}"
            ),
        }
    return {"route": "verified"}


async def after_verify(
    state: RunVariantState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Canonicalise the verifier's judgments and decide: pass, another try, or next program."""
    ctx = runtime.context
    ws = ctx.workspace
    spec = state["spec"]
    cfg = _cfg(state)
    bundle = state["task_bundle"]
    index = state["program_index"]
    evaluation = state["current_evaluation"]
    assert evaluation is not None
    route = state.get("route")
    update: dict[str, Any] = {}
    rubric_valid = False
    rubric_passed = False
    verification_error = state.get("verification_error")
    if route == "verified":
        rubric = await load_rubric_status(
            ws, join(bundle, f"program_{index}", "rubric_status.json")
        )
        canonical, rubric_error = canonicalize_variant_rubric_status(
            rubric,
            variant_id=spec.variant_id,
            attempt_id=variant_attempt_id(spec.variant_id, index),
            idea_id=spec.variant_id,
            required_rubric_ids=state["target_rubric_ids"],
            combined_score=evaluation.metrics.get("combined_score"),
            correct=bool(evaluation.correct),
        )
        rubric_valid = canonical is not None
        if canonical is not None:
            await write_canonical_rubric_status(
                ws, ctx.prompts, bundle, index, canonical, cfg.rubric_evidence_max_chars
            )
            update["last_status"] = canonical
        rubric_passed = bool(
            evaluation.correct and rubric_valid and canonical and canonical["overall_pass"] is True
        )
        if not rubric_valid:
            verification_error = verification_error_context(
                rubric_error or "invalid rubric_status.json"
            )
            update["notes"] = _note(
                f"verifier attempt {state['verify_attempt']} wrote invalid rubric status: "
                f"{rubric_error}"
            )
    # Whether another verification try follows (upstream's inner loop).
    if (
        not state.get("timed_out")
        and not rubric_valid
        and state["verify_attempt"] < cfg.verify_max_tries
    ):
        return {
            **update,
            "route": "scaffold",
            "verify_attempt": state["verify_attempt"] + 1,
            "verification_error": verification_error,
        }
    if not rubric_valid:
        failed = failed_rubric_status(
            spec.variant_id, index, state["target_rubric_ids"], evaluation, verification_error
        )
        await write_canonical_rubric_status(
            ws, ctx.prompts, bundle, index, failed, cfg.rubric_evidence_max_chars
        )
        update["last_status"] = failed
    return {
        **update,
        "route": "next",
        "rubric_valid": rubric_valid,
        "rubric_passed": rubric_passed,
        "verification_error": verification_error,
    }


async def next_program(
    state: RunVariantState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Stop at a pass, a timeout or the last index; otherwise go on with the next program."""
    if state.get("timed_out") or state.get("rubric_passed"):
        return {"route": "finish"}
    if state["program_index"] >= state["last_index"]:
        return {"route": "finish"}
    return {
        "route": "edit",
        "program_index": state["program_index"] + 1,
        "edit_attempt": 1,
        "verify_attempt": 1,
        "verification_error": None,
        "rubric_valid": False,
    }


async def finish_variant(
    state: RunVariantState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Record the variant's run (upstream's ``VariantRunResult``) and update its manifest."""
    ws = runtime.context.workspace
    spec = state["spec"]
    cfg = _cfg(state)
    timed_out = bool(state.get("timed_out"))
    rubric_passed = bool(state.get("rubric_passed"))
    status: Literal["completed", "failed", "timeout"] = (
        "timeout" if timed_out else ("completed" if rubric_passed else "failed")
    )
    eval_count = state.get("eval_count", 0)
    await update_manifest_status(ws, spec.variant_dir, status, eval_count=eval_count)
    last = state.get("last_status")
    infrastructure = state.get("infrastructure_failure")
    failure = infrastructure
    notes = "\n".join(note for note in state.get("notes", []) if note)
    if failure is None and not rubric_passed:
        failure = Failure(
            category="variant_timeout" if timed_out else "variant_failed",
            scope="method",
            detail=truncate_utf8(notes, NOTE_BYTES),
        )
    result = VariantResult(
        variant_id=spec.variant_id,
        status="timeout" if timed_out else ("passed" if rubric_passed else "failed"),
        passed=rubric_passed,
        best_program_index=state.get("best_index"),
        evaluations=list(state.get("evaluations", [])),
        rubric_status=rubric_status_record(last) if last else None,
        task_name=state.get("task_name"),
        failure=None if rubric_passed else failure,
    )
    run = VariantRun(
        result=result,
        status=status,
        eval_count=eval_count,
        exit_code=state.get("exit_code", 1),
        output=truncate_utf8(notes, NOTE_BYTES * 4),
        analysis_files=await check_analysis_files(ws, state["task_bundle"]),
        spec=spec,
        task_bundle=state["task_bundle"],
        infrastructure_failure=infrastructure,
    )
    return {"variant_runs": [run]}


def _route(state: Mapping[str, Any]) -> str:
    return str(state.get("route", "finish"))


def build_run_variant_graph(
    *,
    checkpointer: Any = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "run_variant",
) -> CompiledStateGraph[Any, ScientistContext, RunVariantInput, RunVariantOutput]:
    """Return the compiled graph that runs one variant (see the module docstring).

    The graph is mounted by the variants graph as the target of ``Send`` and can be run alone
    with a :class:`RunVariantInput`. Its steps are ``variants.init_variant``, ``variants.edit``,
    ``variants.evaluate``, ``variants.scaffold_status``, ``variants.verify``,
    ``variants.after_verify`` and ``variants.finish_variant``, each with hook nodes.
    """
    from langgraph.graph import END, START

    graph = StepGraph(
        "variants",
        RunVariantState,
        input_schema=RunVariantInput,
        output_schema=RunVariantOutput,
        overrides=overrides,
    )
    retry = agent_node_retry_policy()
    graph.step("init_variant", init_variant)
    graph.step("edit", edit, retry_policy=retry)
    graph.add_node("after_edit", after_edit)
    graph.step("evaluate", evaluate)
    graph.step("scaffold_status", scaffold_status)
    graph.step("verify", verify, retry_policy=retry)
    graph.step("after_verify", after_verify)
    graph.add_node("next_program", next_program)
    graph.step("finish_variant", finish_variant)

    graph.graph.add_edge(START, graph.entry("init_variant"))
    graph.add_conditional_edges(
        "init_variant", _route, {"edit": "edit", "evaluate": "evaluate", "finish": "finish_variant"}
    )
    graph.add_edge("edit", "after_edit")
    graph.add_conditional_edges(
        "after_edit", _route, {"edit": "edit", "evaluate": "evaluate", "finish": "finish_variant"}
    )
    graph.add_conditional_edges(
        "evaluate", _route, {"scaffold": "scaffold_status", "finish": "finish_variant"}
    )
    graph.add_edge("scaffold_status", "verify")
    graph.add_edge("verify", "after_verify")
    graph.add_conditional_edges(
        "after_verify", _route, {"scaffold": "scaffold_status", "next": "next_program"}
    )
    graph.add_conditional_edges(
        "next_program", _route, {"edit": "edit", "finish": "finish_variant"}
    )
    graph.graph.add_edge(graph.exit("finish_variant"), END)
    return graph.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "RunVariantInput",
    "RunVariantOutput",
    "RunVariantState",
    "build_edit_prompt",
    "build_run_variant_graph",
    "build_verify_prompt",
    "check_analysis_files",
    "discard_evaluation_artifacts",
    "failed_rubric_status",
    "previous_rubric_summary",
    "publish_evaluation",
    "published_for",
    "run_variant_input",
    "update_manifest_status",
    "verification_error_context",
    "write_rubric_scaffold",
]
