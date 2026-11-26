"""Stage 4: materialise one workspace per variant (upstream ``materialize.py``).

Creates ``variants/<variant_id>/`` directories, each holding a task bundle, the variant spec, the
rubric contract and the run manifest. Every variant is cloned from the *retrofitted* bundle, so
the CSV analysis infrastructure is already in place. ``control_best`` is the best program as it is;
``baseline_reference`` (only when the reference differs from the best program) is the seed;
every accepted candidate follows. The variant-specific ``task.md`` is assembled from the pieces
in ``workflow/variants/variant_task/`` in upstream's order, so that a caller can override a
piece.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from ..core.errors import ProgramSpecError
from ..core.results import VariantSpec
from ..core.state import ProgramRef
from ..evaluation.program_clone import clone_program
from ..evaluation.program_spec import load_program_spec, resolve_entrypoint
from ..prompts import PromptRegistry
from ..workspace.paths import join, parse_ignore_patterns
from ..workspace.protocol import Workspace, is_dir, is_file
from .constants import (
    BASELINE_REFERENCE_ID,
    CONTROL_BEST_ID,
    RESERVED_VARIANT_IDS,
    TASK_COPY_EXCLUDE_FILES,
    TASK_COPY_EXCLUDE_SUFFIXES,
    VARIANTS_SUBDIR,
)
from .models import (
    VariantCandidate,
    VariantSelection,
    make_spec,
    spec_detail,
    spec_to_dict,
)
from .preflight import validate_reference_workspace
from .rubrics import build_rubric_contract, build_target_rubrics
from .workspace_ops import (
    copy_tree_replacing,
    make_dir,
    read_text_or_none,
    variant_copy_patterns,
    write_json,
)

logger = logging.getLogger(__name__)

TASK_PIECES = "workflow/variants/variant_task/"

DEFAULT_PROGRAM_SPEC = """\
schema_version = 1
code_root = "src"

[entrypoint]
path = "program.py"
language = "python"
type = "script"

[copy]
exclude = [
    "**/__pycache__/**",
    "**/*.pyc",
    ".venv/**",
    "node_modules/**",
    ".git/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    ".ruff_cache/**",
]
include = []

[integration]
kind = "generic"
notes = "Auto-generated for folder-based reference implementation"
"""
"""Upstream's ``program_spec.toml`` for a reference folder that has none (``materialize.py``)."""

REFERENCE_COPY_IGNORE = (
    "baseline.md",
    "__pycache__",
    "*.pyc",
    ".git",
    ".venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
)
"""Names a reference folder is copied without (``shutil.ignore_patterns`` of upstream)."""


def piece(prompts: PromptRegistry, group: str, name: str, **values: str) -> str:
    """Render one piece of an assembled text from ``workflow/variants/<group>/<name>``."""
    return prompts.render(f"workflow/variants/{group}/{name}", values, fragments=False)


async def task_ignore_patterns(workspace: Workspace, source: str) -> tuple[str, ...]:
    """Return the ``.agentic_task_ignore`` patterns of a task directory."""
    text = await read_text_or_none(workspace, join(source, ".agentic_task_ignore"))
    return parse_ignore_patterns(text) if text is not None else ()


async def copy_task_files(workspace: Workspace, src_dir: str, dst_dir: str) -> None:
    """Copy task-level files (``evaluate.py``, ``capture.py``, ...) into ``dst_dir``.

    Program directories (``program_*``, ``reference_program``), cache and environment
    directories, files of a previous run, runtime markers and log files are left out, as are
    entries the task's ``.agentic_task_ignore`` names (upstream ``_copy_task_files``). Links are
    copied as links.
    """
    patterns = variant_copy_patterns(await task_ignore_patterns(workspace, src_dir))
    await copy_tree_replacing(workspace, src_dir, dst_dir, ignore=patterns)


def build_run_manifest(
    spec: VariantSpec,
    *,
    agent: str,
    agent_model: str | None,
    agent_profile: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build the initial ``run_manifest.json`` of a variant."""
    now = datetime.now(UTC).isoformat()
    return {
        "variant_id": spec.variant_id,
        "variant_kind": spec.kind,
        "base_program": spec_detail(spec, "base_program"),
        "title": spec.title,
        "agent": agent,
        "agent_model": agent_model,
        "agent_profile": agent_profile,
        "reasoning_effort": reasoning_effort,
        "variant_dir": spec.variant_dir,
        "task_bundle_dir": join(spec.variant_dir, "task_bundle"),
        "expected_analysis_files": [
            "analysis/run_summary.csv",
            "analysis/metric_values.csv",
            "analysis/timeseries.csv",
            "rubric_status.json",
        ],
        "status": "created",
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "eval_count": 0,
    }


async def write_variant_task_md(
    workspace: Workspace,
    prompts: PromptRegistry,
    task_bundle: str,
    spec: VariantSpec,
    *,
    max_evals: int = 2,
) -> None:
    """Keep the original ``task.md`` as ``original_task.md`` and write the variant's ``task.md``.

    The new ``task.md`` tells the agent exactly what change to make; the original stays readable
    as ``original_task.md`` (port of ``_write_variant_task_md``; the texts are the pieces of
    ``workflow/variants/variant_task/``).
    """
    task_md = join(task_bundle, "task.md")
    original = join(task_bundle, "original_task.md")
    kept = await workspace.stat(task_md) is not None
    if kept:
        await workspace.write_bytes(original, await workspace.read_bytes(task_md))
        await workspace.remove(task_md)

    def part(name: str, **values: str) -> str:
        return piece(prompts, "variant_task", name, **values)

    base_program = str(spec_detail(spec, "base_program"))
    attempts = str(max_evals)
    sections: list[str] = [
        part("title", TITLE=spec.title),
        part("variant_id", VARIANT_ID=spec.variant_id),
        part("kind", KIND=spec.kind),
        part("base_program", BASE_PROGRAM=base_program),
        part("max_attempts", MAX_EVALS=attempts),
    ]
    if spec.description:
        sections.append(part("description", DESCRIPTION_BODY=spec.description))
    hypothesis = str(spec_detail(spec, "hypothesis"))
    if hypothesis:
        sections.append(part("hypothesis", HYPOTHESIS=hypothesis))
    expected_effect = str(spec_detail(spec, "expected_effect"))
    if expected_effect:
        sections.append(part("expected_effect", EXPECTED_EFFECT=expected_effect))
    components = spec_detail(spec, "target_components")
    if components:
        rendered = ", ".join(f"`{c}`" for c in components)
        sections.append(part("target_components", COMPONENTS=rendered))
    plan = spec_detail(spec, "implementation_plan")
    if plan:
        steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
        sections.append(part("implementation_plan", STEPS=steps))

    if spec.kind == "control":
        sections.append(part("what_control"))
    elif spec.kind == "ablation":
        sections.append(part("what_ablation", MAX_EVALS=attempts))
    elif spec.variant_id == BASELINE_REFERENCE_ID:
        sections.append(part("what_baseline_reference"))
    elif spec.kind == "baseline" and base_program == "reference":
        sections.append(part("what_reference_baseline", MAX_EVALS=attempts))
    elif spec.kind == "baseline":
        sections.append(part("what_baseline", MAX_EVALS=attempts))

    if kept:
        sections.append(part("original_task_context"))
    await workspace.write_bytes(task_md, "".join(sections).encode("utf-8"))


async def copy_reference_as_program_0(
    workspace: Workspace, reference_dir: str, program_dst: str
) -> None:
    """Copy a reference implementation folder as ``program_0`` (upstream ``shutil.copytree``).

    Pipeline metadata (``baseline.md``) and clutter are left out. A default
    ``program_spec.toml`` (and a placeholder ``src/program.py``) is written when the folder has
    none, so that the trusted clone can later copy ``program_0`` to ``program_1``.
    """
    await copy_tree_replacing(workspace, reference_dir, program_dst, ignore=REFERENCE_COPY_IGNORE)
    spec_path = join(program_dst, "program_spec.toml")
    if not await is_file(workspace, spec_path):
        entrypoint = join(program_dst, "src/program.py")
        if await workspace.stat(entrypoint) is None:
            await workspace.write_bytes(
                entrypoint,
                b"# Placeholder entrypoint \xe2\x80\x94 see reference code in parent directory\n",
            )
        await workspace.write_bytes(spec_path, DEFAULT_PROGRAM_SPEC.encode("utf-8"))
        logger.info("Generated default program_spec.toml in %s", program_dst)


async def remove_stale_files(workspace: Workspace, program_dir: str) -> None:
    """Remove result and log files a clone may have carried (``_remove_stale_files``)."""
    for entry in await workspace.list_tree(program_dir):
        if entry.kind != "file":
            continue
        name = entry.path.rsplit("/", 1)[-1]
        if name in TASK_COPY_EXCLUDE_FILES or name.endswith(TASK_COPY_EXCLUDE_SUFFIXES):
            await workspace.remove(join(program_dir, entry.path))


async def reference_differs_from_best(workspace: Workspace, bundle: str) -> bool:
    """Whether ``reference_program/`` differs materially from ``program_0/`` (entrypoints compared).

    ``False`` when ``reference_program/`` or ``program_0/`` does not exist; ``True`` when the
    programs cannot be compared (upstream assumes they differ).
    """
    ref_dir = join(bundle, "reference_program")
    best_dir = join(bundle, "program_0")
    if not await is_dir(workspace, ref_dir) or not await is_dir(workspace, best_dir):
        return False
    try:
        ref_entry = await resolve_entrypoint(workspace, ref_dir)
        best_entry = await resolve_entrypoint(workspace, best_dir)
        await load_program_spec(workspace, ref_dir)
        await load_program_spec(workspace, best_dir)
        return (await workspace.read_bytes(join(ref_dir, ref_entry))) != (
            await workspace.read_bytes(join(best_dir, best_entry))
        )
    except (ProgramSpecError, OSError):
        return True


async def materialize_single_variant(
    workspace: Workspace,
    prompts: PromptRegistry,
    *,
    variant_id: str,
    title: str,
    kind: str,
    base_program: str,
    source_program_dir: str,
    retrofitted_bundle_dir: str,
    variant_dir: str,
    spec_data: Mapping[str, Any],
    agent: str,
    agent_model: str | None,
    max_evals_per_variant: int = 2,
    reference_dir: str | None = None,
) -> VariantSpec:
    """Create one variant directory with every required artifact and return its spec."""
    await make_dir(workspace, variant_dir)

    # 1. Task-level files into task_bundle/
    task_bundle = join(variant_dir, "task_bundle")
    await copy_task_files(workspace, retrofitted_bundle_dir, task_bundle)

    # 2. program_0 of the task bundle
    program_dst = join(task_bundle, "program_0")
    await workspace.remove(program_dst)
    if reference_dir is not None:
        # Folder-based baseline: the reference code is program_0; the control is control_program/.
        await copy_reference_as_program_0(workspace, reference_dir, program_dst)
        control_dst = join(task_bundle, "control_program")
        await workspace.remove(control_dst)
        await clone_program(workspace, src_dir=source_program_dir, dst_dir=control_dst)
        logger.info(
            "Folder baseline '%s': reference code -> program_0, control -> control_program",
            variant_id,
        )
    else:
        await clone_program(workspace, src_dir=source_program_dir, dst_dir=program_dst)
    await remove_stale_files(workspace, program_dst)
    await validate_reference_workspace(workspace, task_bundle, variant_id, base_program)

    # 3. variant_spec.json
    try:
        entrypoint = await resolve_entrypoint(workspace, program_dst)
    except ProgramSpecError:
        entrypoint = "src/program.py"
    spec = make_spec(
        variant_id=variant_id,
        title=title,
        kind=kind,
        base_program=base_program,
        variant_dir=variant_dir,
        hypothesis=spec_data.get("hypothesis", ""),
        relevance_reason=spec_data.get("relevance_reason", ""),
        feasibility_reason=spec_data.get("feasibility_reason", ""),
        target_components=spec_data.get("target_components", []),
        implementation_plan=spec_data.get("implementation_plan", []),
        expected_effect=spec_data.get("expected_effect", ""),
        description_body=spec_data.get("description_body", ""),
        task_dir=task_bundle,
        program=ProgramRef(path=program_dst, entrypoint=entrypoint),
    )
    await write_json(workspace, join(variant_dir, "variant_spec.json"), spec_to_dict(spec))

    # 4. rubric_contract.json and target_rubrics.json
    rubric_contract = build_rubric_contract(prompts, variant_id, kind, base_program=base_program)
    await write_json(workspace, join(variant_dir, "rubric_contract.json"), rubric_contract)
    await write_json(
        workspace, join(variant_dir, "target_rubrics.json"), build_target_rubrics(rubric_contract)
    )

    # 5. run_manifest.json
    await write_json(
        workspace,
        join(variant_dir, "run_manifest.json"),
        build_run_manifest(spec, agent=agent, agent_model=agent_model),
    )

    # 6. The variant's task.md, and 7. the rubric contract where the agent can see it
    await write_variant_task_md(
        workspace, prompts, task_bundle, spec, max_evals=max_evals_per_variant
    )
    await workspace.write_bytes(
        join(task_bundle, "rubric_contract.json"),
        await workspace.read_bytes(join(variant_dir, "rubric_contract.json")),
    )
    logger.info("Materialized variant: %s (%s)", variant_id, kind)
    return spec


async def materialize_variant_workspaces(
    workspace: Workspace,
    prompts: PromptRegistry,
    *,
    bundle_dir: str,
    retrofitted_bundle_dir: str,
    variants_run_dir: str,
    selection: VariantSelection | None = None,
    agent: str,
    agent_model: str | None,
    max_evals_per_variant: int = 2,
    custom_only: bool = False,
    required_controls: Sequence[str] = (),
) -> list[VariantSpec]:
    """Create every variant directory from the selection and return their specs.

    Args:
        workspace: The run's workspace.
        prompts: The prompt registry (the task.md pieces and rubric descriptions).
        bundle_dir: The extracted bundle (decides whether the reference differs from the best).
        retrofitted_bundle_dir: The retrofitted bundle every variant is cloned from.
        variants_run_dir: The phase's output directory.
        selection: The accepted candidates (generated and custom).
        agent: The agent name recorded in run manifests.
        agent_model: The model name recorded in run manifests.
        max_evals_per_variant: Programs each variant may produce.
        custom_only: Skip ``control_best`` and ``baseline_reference``.
        required_controls: Controls that must still be materialised in ``custom_only`` mode
            (only ``control_best`` is supported).

    Raises:
        ValueError: For a candidate that uses a reserved id or an unsupported required control.
    """
    accepted = selection.accepted if selection is not None else []
    reserved = sorted(c.variant_id for c in accepted if c.variant_id in RESERVED_VARIANT_IDS)
    if reserved:
        raise ValueError(f"Candidates use reserved variant IDs: {', '.join(reserved)}")
    unsupported = set(required_controls) - {CONTROL_BEST_ID}
    if unsupported:
        raise ValueError(f"Unsupported required controls: {sorted(unsupported)}")
    variants_dir = join(variants_run_dir, VARIANTS_SUBDIR)
    await make_dir(workspace, variants_dir)

    specs: list[VariantSpec] = []
    common: dict[str, Any] = {
        "retrofitted_bundle_dir": retrofitted_bundle_dir,
        "agent": agent,
        "agent_model": agent_model,
        "max_evals_per_variant": max_evals_per_variant,
    }
    if not custom_only or CONTROL_BEST_ID in required_controls:
        specs.append(
            await materialize_single_variant(
                workspace,
                prompts,
                variant_id=CONTROL_BEST_ID,
                title="Control (best program)",
                kind="control",
                base_program="best",
                source_program_dir=join(retrofitted_bundle_dir, "program_0"),
                variant_dir=join(variants_dir, CONTROL_BEST_ID),
                spec_data={
                    "hypothesis": "Baseline control using the best program as-is.",
                    "expected_effect": "Establishes the reference metrics.",
                },
                **common,
            )
        )
        if not custom_only and await reference_differs_from_best(workspace, bundle_dir):
            ref_program_dir = join(retrofitted_bundle_dir, "reference_program")
            if not await is_dir(workspace, ref_program_dir):
                ref_program_dir = join(bundle_dir, "reference_program")
            if await is_dir(workspace, ref_program_dir):
                specs.append(
                    await materialize_single_variant(
                        workspace,
                        prompts,
                        variant_id=BASELINE_REFERENCE_ID,
                        title="Baseline (reference / seed program)",
                        kind="baseline",
                        base_program="reference",
                        source_program_dir=ref_program_dir,
                        variant_dir=join(variants_dir, BASELINE_REFERENCE_ID),
                        spec_data={
                            "hypothesis": (
                                "Compare the improved best program against the original seed."
                            ),
                            "expected_effect": "Shows improvement over the starting point.",
                        },
                        **common,
                    )
                )
    else:
        logger.info("custom_only mode: skipping control_best")
    if custom_only:
        logger.info("custom_only mode: skipping baseline_reference")

    for candidate in accepted:
        specs.append(
            await _materialize_candidate(workspace, prompts, candidate, variants_dir, common)
        )
    logger.info("Materialized %d variant workspaces under %s", len(specs), variants_dir)
    return specs


async def _materialize_candidate(
    workspace: Workspace,
    prompts: PromptRegistry,
    candidate: VariantCandidate,
    variants_dir: str,
    common: Mapping[str, Any],
) -> VariantSpec:
    return await materialize_single_variant(
        workspace,
        prompts,
        variant_id=candidate.variant_id,
        title=candidate.title,
        kind=candidate.kind,
        base_program=candidate.base_program,
        source_program_dir=join(common["retrofitted_bundle_dir"], "program_0"),
        variant_dir=join(variants_dir, candidate.variant_id),
        spec_data={
            "hypothesis": candidate.hypothesis,
            "relevance_reason": candidate.relevance_reason,
            "feasibility_reason": candidate.feasibility_reason,
            "target_components": candidate.target_components,
            "implementation_plan": candidate.implementation_plan,
            "expected_effect": candidate.expected_effect,
            "description_body": candidate.description_body,
        },
        reference_dir=candidate.reference_dir,
        **common,
    )


__all__ = [
    "DEFAULT_PROGRAM_SPEC",
    "build_run_manifest",
    "copy_reference_as_program_0",
    "copy_task_files",
    "materialize_single_variant",
    "materialize_variant_workspaces",
    "piece",
    "reference_differs_from_best",
    "remove_stale_files",
    "write_variant_task_md",
]
