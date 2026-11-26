"""Stage 4: materialise one workspace per accepted analysis (upstream ``materialize.py``).

Each analysis gets ``analyses/<id>/`` with its ``analysis_spec.json``, the six-rubric
``rubric_contract.json`` and ``target_rubrics.json``, a ``run_manifest.json`` and a
``task_bundle/``: the task-level files of the analysis-ready bundle (including the staged
evidence), clones of ``program_0`` and ``reference_program`` without evaluator files, and a
generated ``task.md`` (the original becomes ``original_task.md``). ``evaluate.py`` and the
extraction manifest are removed from the task bundle: the analysis is analysis-only and never
evaluates.

The generated ``task.md`` is assembled from upstream's texts (``prompts/workflow/posthoc/
analysis_task/``) with upstream's list formats.
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from ..context import ScientistContext
from ..core.state import Event, utc_now
from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .common import clone_program_bundle, copy_task_files, write_json
from .layout import Layout
from .models import AnalysisSelection, AnalysisSpec, PosthocState, is_valid_analysis_id
from .rubric_contract import build_rubric_contract, build_target_rubrics
from .settings import posthoc_config

PHASE = "posthoc"
_TEXT = "workflow/posthoc/analysis_task/"


def render_analysis_task_md(
    prompts: PromptRegistry,
    spec: AnalysisSpec,
    *,
    max_attempts_per_analysis: int,
    has_original: bool,
) -> str:
    """Return the generated analysis ``task.md`` (upstream ``_write_analysis_task_md``)."""

    def text(name: str, **values: object) -> str:
        return prompts.render(
            _TEXT + name, {key: str(value) for key, value in values.items()}, fragments=False
        )

    lines: list[str] = [
        text("title", TITLE=spec.title),
        text("analysis_id", ANALYSIS_ID=spec.analysis_id),
        text("family", FAMILY=spec.family),
        text("max_attempts", MAX_ATTEMPTS=max_attempts_per_analysis),
        text("requires_retrain", REQUIRES_RETRAIN=spec.requires_retrain),
    ]
    if spec.hypothesis:
        lines.append(text("hypothesis", HYPOTHESIS=spec.hypothesis))
    if spec.expected_claim:
        lines.append(text("expected_claim", EXPECTED_CLAIM=spec.expected_claim))
    if spec.required_artifacts:
        lines.append(
            text("required_artifacts_heading")
            + "\n".join(f"- `{item}`" for item in spec.required_artifacts)
            + "\n"
        )
    if spec.expected_outputs:
        lines.append(
            text("expected_outputs_heading")
            + "\n".join(f"- `{item}`" for item in spec.expected_outputs)
            + "\n"
        )
    if spec.implementation_plan:
        lines.append(
            text("implementation_plan_heading")
            + "\n".join(
                f"{index + 1}. {step}" for index, step in enumerate(spec.implementation_plan)
            )
            + "\n"
        )
    lines.append(text("execution_rules"))
    if has_original:
        lines.append(text("original_task_context"))
    return "".join(lines)


def build_run_manifest(spec: AnalysisSpec, max_attempts_per_analysis: int) -> dict[str, Any]:
    """Return ``run_manifest.json`` (upstream ``_build_run_manifest``, without agent-CLI fields)."""
    return {
        "analysis_id": spec.analysis_id,
        "analysis_family": spec.family,
        "title": spec.title,
        "analysis_dir": spec.analysis_dir,
        "task_bundle_dir": join(spec.analysis_dir, "task_bundle"),
        "expected_outputs": spec.expected_outputs,
        "requires_retrain": spec.requires_retrain,
        "max_attempts_per_analysis": max_attempts_per_analysis,
        "status": "created",
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "attempt_count": 0,
        "selected_attempt": None,
    }


async def _write_analysis_task_md(
    workspace: Workspace,
    prompts: PromptRegistry,
    task_bundle_dir: str,
    spec: AnalysisSpec,
    *,
    max_attempts_per_analysis: int,
) -> None:
    task_md = join(task_bundle_dir, "task.md")
    original = join(task_bundle_dir, "original_task.md")
    entry = await workspace.stat(task_md)
    if entry is not None:
        await workspace.write_bytes(original, await workspace.read_bytes(task_md))
        await workspace.remove(task_md)
    content = render_analysis_task_md(
        prompts,
        spec,
        max_attempts_per_analysis=max_attempts_per_analysis,
        has_original=await workspace.exists(original),
    )
    await workspace.write_bytes(task_md, content.encode("utf-8"))


async def materialize_analysis_workspaces(
    workspace: Workspace,
    prompts: PromptRegistry,
    *,
    layout: Layout,
    selection: AnalysisSelection,
    all_candidate_ids: list[str],
    max_attempts_per_analysis: int,
    evaluator: str = "evaluate.py",
) -> list[AnalysisSpec]:
    """Create one analysis workspace per accepted candidate and return their specifications.

    Workspaces of candidates that were not accepted are removed, so a re-materialisation never
    leaves a stale analysis behind.

    Raises:
        ValueError: When an accepted analysis id is not safe as a directory name.
    """
    invalid_ids = [
        candidate.analysis_id
        for candidate in selection.accepted
        if not is_valid_analysis_id(candidate.analysis_id)
    ]
    if invalid_ids:
        raise ValueError(f"Invalid analysis_id values: {invalid_ids}")
    accepted_ids = {candidate.analysis_id for candidate in selection.accepted}
    for analysis_id in all_candidate_ids:
        if analysis_id not in accepted_ids and is_valid_analysis_id(analysis_id):
            await workspace.remove(layout.analysis(analysis_id))

    specs: list[AnalysisSpec] = []
    for candidate in selection.accepted:
        analysis_dir = layout.analysis(candidate.analysis_id)
        await workspace.remove(analysis_dir)
        task_bundle = join(analysis_dir, "task_bundle")
        await copy_task_files(workspace, layout.ready, task_bundle)
        await workspace.remove(join(task_bundle, evaluator))
        await workspace.remove(join(task_bundle, "extraction_manifest.json"))
        await clone_program_bundle(
            workspace, join(layout.ready, "program_0"), join(task_bundle, "program_0")
        )
        reference = join(layout.ready, "reference_program")
        reference_entry = await workspace.stat(reference)
        if reference_entry is not None and reference_entry.kind == "dir":
            await clone_program_bundle(workspace, reference, join(task_bundle, "reference_program"))

        spec = AnalysisSpec(
            analysis_id=candidate.analysis_id,
            title=candidate.title,
            family=candidate.family,
            analysis_dir=analysis_dir,
            hypothesis=candidate.hypothesis,
            relevance_reason=candidate.relevance_reason,
            feasibility_reason=candidate.feasibility_reason,
            required_artifacts=candidate.required_artifacts,
            requires_retrain=candidate.requires_retrain,
            implementation_plan=candidate.implementation_plan,
            expected_claim=candidate.expected_claim,
            expected_outputs=candidate.expected_outputs,
        )
        await write_json(workspace, join(analysis_dir, "analysis_spec.json"), spec.to_dict())
        contract = build_rubric_contract(candidate.analysis_id, candidate.family)
        await write_json(workspace, join(analysis_dir, "rubric_contract.json"), contract)
        await write_json(
            workspace, join(analysis_dir, "target_rubrics.json"), build_target_rubrics(contract)
        )
        await write_json(
            workspace,
            join(analysis_dir, "run_manifest.json"),
            build_run_manifest(spec, max_attempts_per_analysis),
        )
        await _write_analysis_task_md(
            workspace,
            prompts,
            task_bundle,
            spec,
            max_attempts_per_analysis=max_attempts_per_analysis,
        )
        for staged in ("analysis_spec.json", "rubric_contract.json", "target_rubrics.json"):
            await workspace.write_bytes(
                join(task_bundle, staged), await workspace.read_bytes(join(analysis_dir, staged))
            )
        specs.append(spec)
    return specs


async def materialize(state: PosthocState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Graph step: materialise the accepted analyses."""
    ctx = runtime.context
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    task = state["task"]
    selection = state.get("selection") or AnalysisSelection()
    specs = await materialize_analysis_workspaces(
        ctx.workspace,
        ctx.prompts,
        layout=layout,
        selection=selection,
        all_candidate_ids=[candidate.analysis_id for candidate in state.get("candidates") or []],
        max_attempts_per_analysis=config.max_attempts_per_analysis,
        evaluator=task.contract.evaluators.search if task.contract is not None else "evaluate.py",
    )
    return {
        "specs": specs,
        "events": [Event(kind="posthoc_materialized", phase=PHASE, data={"analyses": len(specs)})],
    }


__all__ = [
    "build_run_manifest",
    "materialize",
    "materialize_analysis_workspaces",
    "render_analysis_task_md",
]
