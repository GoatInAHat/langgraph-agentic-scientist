"""Stage 8: the variants overview README for the paper-writing agent (upstream ``readme.py``).

The README has a deterministic structure and agent-written per-variant summaries. The *README
summariser* (R22, a structured role, no skills) writes one sentence about the task and two or
three sentences per variant; when it fails, times out or omits a variant, the summary falls back
to the variant's hypothesis (or ``<title> (<kind> variant).``) and the task summary to the first
non-heading line of ``task.md``. The README is written as ``README.md`` in the output directory,
where the paper stage picks it up (upstream's docs call it ``variants_readme.md``; the code
writes ``README.md``).

Two extra-task layouts are described: *structured* (pipeline-generated, with ``variants/`` and
``task_bundle/program_*/analysis/``) and *flat* (added by hand: every variant a subdirectory with
CSVs and an optional ``variant_spec.json``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from ..core.results import VariantSpec
from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace, is_dir, is_file
from .constants import EXTRA_TASKS_SUBDIR, VARIANTS_SUBDIR
from .materialize import piece
from .models import spec_detail
from .workspace_ops import list_child_dirs, list_children, read_json_object, read_text_or_none

logger = logging.getLogger(__name__)

PROMPT_KEY = "variants/variant_readme"
KIND_SORT_KEY = {"control": 0, "baseline": 1, "ablation": 2}
SUMMARIES_FILENAME = "variant_summaries.json"


def _part(prompts: PromptRegistry, name: str, **values: str) -> str:
    return piece(prompts, "readme", name, **values)


# --- prompt --------------------------------------------------------------------------------------


async def read_task_md(workspace: Workspace, task_dir: str) -> str:
    """Read ``task.md`` of a task directory (empty when absent)."""
    return await read_text_or_none(workspace, join(task_dir, "task.md")) or ""


def build_readme_prompt(
    prompts: PromptRegistry,
    variant_specs: Sequence[VariantSpec],
    task_md: str,
    flat_tasks: Mapping[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Build the summariser's prompt (the task, the variant specifications, flat extra tasks)."""
    variants_json = json.dumps(
        [
            {
                "variant_id": s.variant_id,
                "title": s.title,
                "kind": s.kind,
                "base_program": spec_detail(s, "base_program"),
                "hypothesis": spec_detail(s, "hypothesis"),
                "target_components": spec_detail(s, "target_components"),
                "expected_effect": spec_detail(s, "expected_effect"),
            }
            for s in variant_specs
        ],
        indent=2,
    )
    flat_section = ""
    if flat_tasks:
        flat_section = _part(prompts, "flat_prompt_section") + json.dumps(flat_tasks, indent=2)
    return prompts.render(
        PROMPT_KEY,
        {"TASK_MD": task_md, "VARIANTS_JSON": variants_json + flat_section},
        fragments=False,
    )


# --- fallbacks -----------------------------------------------------------------------------------


def fallback_summary(prompts: PromptRegistry, spec: VariantSpec) -> str:
    """Return the variant's hypothesis, or a generic sentence."""
    hypothesis = str(spec_detail(spec, "hypothesis"))
    if hypothesis:
        return hypothesis
    return _part(prompts, "fallback_summary", TITLE=spec.title, KIND=spec.kind)


def fallback_task_summary(prompts: PromptRegistry, task_md: str) -> str:
    """Return the first non-empty, non-heading line of ``task.md``, or a generic sentence."""
    for line in task_md.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return _part(prompts, "fallback_task_summary")


# --- discovery -----------------------------------------------------------------------------------


def extract_task_goal(task_md: str) -> str:
    """Extract the first goal bullet of ``task.md``; past 200 characters cut to a sentence."""
    in_goal = False
    for line in task_md.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("# goal") or stripped.lower().startswith("## goal"):
            in_goal = True
            continue
        if in_goal and stripped.startswith("- "):
            goal = stripped.lstrip("- ").strip()
            if len(goal) > 200:
                dot = goal.find(". ", 0, 200)
                goal = goal[: dot + 1] if dot > 0 else goal[:200] + "..."
            return goal
        if in_goal and stripped.startswith("#"):
            break
    return ""


async def _is_flat_extra_task(workspace: Workspace, task_dir: str) -> bool:
    """A flat extra task has a subdirectory with a ``variant_spec.json`` or any CSV file."""
    for name in await list_child_dirs(workspace, task_dir):
        if name.startswith("_"):
            continue
        child = join(task_dir, name)
        if await is_file(workspace, join(child, "variant_spec.json")):
            return True
        if any(
            not directory and entry.endswith(".csv")
            for entry, directory in await list_children(workspace, child)
        ):
            return True
    return False


async def load_flat_variant_specs(workspace: Workspace, task_dir: str) -> list[dict[str, Any]]:
    """Load ``variant_spec.json`` from each variant subdirectory of a flat task."""
    specs: list[dict[str, Any]] = []
    for name in await list_child_dirs(workspace, task_dir):
        if name.startswith("_"):
            continue
        data = await read_json_object(workspace, join(task_dir, name, "variant_spec.json"))
        if data is not None:
            specs.append(data)
    return specs


async def discover_tasks(
    workspace: Workspace,
    output_dir: str,
    task_dir: str,
    primary_task_name: str | None = None,
) -> list[dict[str, str]]:
    """Discover the primary task and the extra tasks (``primary``, ``cross-task`` or ``flat``)."""
    primary_name = primary_task_name or task_dir.rstrip("/").rsplit("/", 1)[-1] or "original"
    tasks = [
        {
            "name": primary_name,
            "data_path": f"{VARIANTS_SUBDIR}/",
            "goal": extract_task_goal(await read_task_md(workspace, task_dir)),
            "role": "primary",
        }
    ]
    extra_root = join(output_dir, EXTRA_TASKS_SUBDIR)
    if await is_dir(workspace, extra_root):
        for name in await list_child_dirs(workspace, extra_root):
            sub = join(extra_root, name)
            if await is_dir(workspace, join(sub, VARIANTS_SUBDIR)):
                bundle = join(sub, "_retrofitted_bundle")
                goal_dir = bundle if await is_dir(workspace, bundle) else sub
                tasks.append(
                    {
                        "name": name,
                        "data_path": f"{EXTRA_TASKS_SUBDIR}/{name}/{VARIANTS_SUBDIR}/",
                        "goal": extract_task_goal(await read_task_md(workspace, goal_dir)),
                        "role": "cross-task",
                    }
                )
            elif await _is_flat_extra_task(workspace, sub):
                tasks.append(
                    {
                        "name": name,
                        "data_path": f"{EXTRA_TASKS_SUBDIR}/{name}/",
                        "goal": extract_task_goal(await read_task_md(workspace, sub)),
                        "role": "flat",
                    }
                )
    return tasks


def build_status_map(pipeline_summary: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Extract per-variant status from ``pipeline_summary.json``."""
    if not pipeline_summary:
        return {}
    entries = pipeline_summary.get("variants")
    if not isinstance(entries, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("variant_id"), str):
            result[entry["variant_id"]] = entry
    return result


# --- assembly ------------------------------------------------------------------------------------


def assemble_readme(
    prompts: PromptRegistry,
    task_summary: str,
    variant_specs: Sequence[VariantSpec],
    summaries: Mapping[str, str],
    status_map: Mapping[str, Mapping[str, Any]],
    task_info: Sequence[Mapping[str, str]] | None = None,
    flat_tasks: Mapping[str, list[dict[str, Any]]] | None = None,
) -> str:
    """Assemble the final Markdown README (pure; upstream ``_assemble_readme``)."""
    flat_tasks = flat_tasks or {}
    part = _part
    lines: list[str] = [part(prompts, "title"), "", task_summary, ""]

    if task_info:
        lines.append(part(prompts, "evaluated_tasks"))
        lines.append("")
        for task in task_info:
            role = task.get("role", "cross-task")
            if role == "primary":
                role_label = part(prompts, "role_primary")
            elif role == "flat":
                role_label = part(prompts, "role_extra")
            else:
                role_label = part(prompts, "role_cross_task")
            lines.append(part(prompts, "task_header", NAME=task["name"], ROLE_LABEL=role_label))
            if task.get("goal"):
                lines.append(task["goal"])
            lines.append("")
            lines.append(part(prompts, "task_results", DATA_PATH=task["data_path"]))
            lines.append("")

    lines.append(part(prompts, "variants_heading"))
    lines.append("")
    sorted_specs = sorted(variant_specs, key=lambda s: (KIND_SORT_KEY.get(s.kind, 9), s.variant_id))
    for i, spec in enumerate(sorted_specs):
        status_info = status_map.get(spec.variant_id, {})
        status = status_info.get("status", "unknown")
        rubric = "passed" if status_info.get("rubric_passed") else "failed"
        lines.append(part(prompts, "variant_header", TITLE=spec.title, VARIANT_ID=spec.variant_id))
        lines.append(part(prompts, "variant_meta", KIND=spec.kind, STATUS=status, RUBRIC=rubric))
        lines.append("")
        lines.append(summaries.get(spec.variant_id, fallback_summary(prompts, spec)))
        lines.append("")
        lines.append(part(prompts, "variant_workspace", VARIANT_ID=spec.variant_id))
        if i < len(sorted_specs) - 1:
            lines.append("")
            lines.append("---")
        lines.append("")

    if flat_tasks:
        for task_name, flat_specs in sorted(flat_tasks.items()):
            lines.append(part(prompts, "extra_task_heading", TASK_NAME=task_name))
            lines.append("")
            sorted_flat = sorted(
                flat_specs,
                key=lambda s: (KIND_SORT_KEY.get(s.get("kind", ""), 9), s.get("variant_id", "")),
            )
            for j, fspec in enumerate(sorted_flat):
                vid = fspec.get("variant_id", "unknown")
                title = fspec.get("title", vid)
                kind = fspec.get("kind", "unknown")
                hypothesis = fspec.get(
                    "hypothesis", part(prompts, "fallback_summary", TITLE=title, KIND=kind)
                )
                lines.append(part(prompts, "flat_header", TITLE=title, VARIANT_ID=vid))
                lines.append(part(prompts, "flat_kind", KIND=kind))
                lines.append("")
                lines.append(hypothesis)
                lines.append("")
                lines.append(part(prompts, "flat_workspace", TASK_NAME=task_name, VARIANT_ID=vid))
                if j < len(sorted_flat) - 1:
                    lines.append("")
                    lines.append("---")
                lines.append("")

    lines.append(part(prompts, "guide_heading"))
    lines.append("")
    lines.append(part(prompts, "guide_table_header"))
    lines.append(part(prompts, "guide_table_rule"))
    for name in (
        "guide_run_summary",
        "guide_metric_values",
        "guide_timeseries",
        "guide_selection",
        "guide_pipeline_summary",
        "guide_variant_spec",
        "guide_task_bundle",
    ):
        lines.append(part(prompts, name))
    has_structured_extra = bool(task_info) and any(
        t.get("role") == "cross-task" for t in task_info or ()
    )
    if has_structured_extra:
        lines.append(part(prompts, "guide_structured_extra"))
    if flat_tasks:
        lines.append(part(prompts, "guide_flat_extra"))
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "PROMPT_KEY",
    "SUMMARIES_FILENAME",
    "assemble_readme",
    "build_readme_prompt",
    "build_status_map",
    "discover_tasks",
    "extract_task_goal",
    "fallback_summary",
    "fallback_task_summary",
    "load_flat_variant_specs",
    "read_task_md",
]
