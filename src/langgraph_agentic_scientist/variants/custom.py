"""Custom baselines (upstream ``custom_baselines.py``): always accepted, bypassing stages 2 and 3.

A custom baseline is one of two things, both named by a workspace path:

* a **Markdown file** with YAML frontmatter (``variant_id``, ``title``, ``hypothesis``, ...) and a
  Markdown body; the agent implements the baseline from the description;
* a **folder** with a working reference implementation, cloned as ``program_0`` that the agent
  adapts to the evaluation harness. Its metadata is inferred from the folder name; a
  ``baseline.md`` inside the folder may override it, and its ``README.md`` is the description.

:func:`inject_custom_baselines` merges them into a selection the way upstream's pipeline does.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..workspace.paths import join, safe_relpath
from ..workspace.protocol import Workspace, is_dir, is_file
from .constants import VARIANT_ID_RE
from .models import VariantCandidate, VariantSelection
from .workspace_ops import read_text_or_none

logger = logging.getLogger(__name__)

_CUSTOM_KINDS = ("baseline", "ablation")


def _split_frontmatter(text: str, path: str) -> tuple[str, str]:
    """Split ``---``-delimited YAML frontmatter from the Markdown body."""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        raise ValueError(
            f"File {path} does not contain YAML frontmatter (expected '---' delimiter at the start)"
        )
    first_newline = stripped.index("\n")
    rest = stripped[first_newline + 1 :]
    closing = rest.find("\n---")
    if closing == -1:
        raise ValueError(f"File {path} has an opening '---' but no closing '---' delimiter")
    frontmatter = rest[:closing]
    body = rest[closing + 4 :]
    if body and body[0] == "\n":
        body = body[1:]
    return frontmatter, body


def _sanitize_folder_name(name: str) -> str:
    """Convert a folder name to a variant id component (lowercase, ``[a-z0-9_]`` only)."""
    result = name.lower().replace("-", "_").replace(" ", "_")
    result = re.sub(r"[^a-z0-9_]", "", result)
    return re.sub(r"_+", "_", result).strip("_")


def _prettify_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in (value or [])]


async def _parse_baseline_markdown(workspace: Workspace, path: str) -> VariantCandidate:
    text = await read_text_or_none(workspace, path)
    if text is None:
        raise FileNotFoundError(f"Custom baseline path not found: {path}")
    frontmatter, body = _split_frontmatter(text, path)
    meta = yaml.safe_load(frontmatter)
    if not isinstance(meta, dict):
        raise ValueError(f"YAML frontmatter in {path} is not a mapping")
    variant_id = meta.get("variant_id")
    if not variant_id:
        raise ValueError(f"Missing required 'variant_id' in frontmatter of {path}")
    variant_id = str(variant_id)
    if not VARIANT_ID_RE.match(variant_id):
        raise ValueError(
            f"Invalid variant_id '{variant_id}' in {path}: must match {VARIANT_ID_RE.pattern}"
        )
    if not variant_id.startswith("baseline_"):
        old_id = variant_id
        variant_id = f"baseline_{variant_id}"
        logger.warning(
            "variant_id '%s' in %s does not start with 'baseline_'; auto-renamed to '%s'",
            old_id,
            path,
            variant_id,
        )
    kind = str(meta.get("kind", "baseline"))
    if kind not in _CUSTOM_KINDS:
        raise ValueError(f"Invalid kind '{kind}' in {path}: must be one of {_CUSTOM_KINDS}")
    return VariantCandidate(
        variant_id=variant_id,
        title=str(meta.get("title", variant_id)),
        kind=kind,
        base_program=str(meta.get("base_program", "best")),
        hypothesis=str(meta.get("hypothesis", "")),
        relevance_reason=str(meta.get("relevance_reason", "")),
        feasibility_reason=str(meta.get("feasibility_reason", "")),
        target_components=_string_list(meta.get("target_components")),
        implementation_plan=_string_list(meta.get("implementation_plan")),
        expected_effect=str(meta.get("expected_effect", "")),
        description_body=body.strip(),
    )


async def _load_folder_baseline(workspace: Workspace, folder: str) -> VariantCandidate:
    name = folder.rstrip("/").rsplit("/", 1)[-1]
    raw_id = _sanitize_folder_name(name)
    if not raw_id:
        raise ValueError(f"Cannot derive variant_id from folder name '{name}'")
    if not raw_id.startswith("baseline_"):
        raw_id = f"baseline_{raw_id}"
    variant_id = raw_id
    title = _prettify_name(name)
    kind = "baseline"
    base_program = "reference"
    hypothesis = expected_effect = relevance_reason = feasibility_reason = ""
    target_components: list[str] = []
    implementation_plan: list[str] = []
    description_body = ""

    readme = await read_text_or_none(workspace, join(folder, "README.md"))
    if readme is not None:
        description_body = readme.strip()

    baseline_md = join(folder, "baseline.md")
    if await is_file(workspace, baseline_md):
        try:
            override = await _parse_baseline_markdown(workspace, baseline_md)
        except ValueError:
            logger.warning("baseline.md in %s could not be parsed; using inferred defaults", folder)
        else:
            variant_id = override.variant_id
            title = override.title
            kind = override.kind
            base_program = override.base_program if override.base_program != "best" else "reference"
            hypothesis = override.hypothesis or hypothesis
            expected_effect = override.expected_effect or expected_effect
            relevance_reason = override.relevance_reason or relevance_reason
            feasibility_reason = override.feasibility_reason or feasibility_reason
            target_components = override.target_components or target_components
            implementation_plan = override.implementation_plan or implementation_plan
            description_body = override.description_body or description_body
    return VariantCandidate(
        variant_id=variant_id,
        title=title,
        kind=kind,
        base_program=base_program,
        hypothesis=hypothesis,
        relevance_reason=relevance_reason,
        feasibility_reason=feasibility_reason,
        target_components=target_components,
        implementation_plan=implementation_plan,
        expected_effect=expected_effect,
        description_body=description_body,
        reference_dir=folder,
    )


async def load_custom_baselines(
    workspace: Workspace, paths: Sequence[str]
) -> list[VariantCandidate]:
    """Load and validate custom baseline paths (Markdown files or folders of the workspace).

    Raises:
        FileNotFoundError: When a path does not exist.
        ValueError: On parse errors, a missing or invalid ``variant_id``, or a duplicate id.
    """
    candidates: list[VariantCandidate] = []
    seen_ids: dict[str, str] = {}
    for raw in paths:
        path = safe_relpath(raw, allow_root=False)
        if await workspace.stat(path) is None:
            raise FileNotFoundError(f"Custom baseline path not found: {path}")
        if await is_dir(workspace, path):
            candidate = await _load_folder_baseline(workspace, path)
        elif await is_file(workspace, path):
            candidate = await _parse_baseline_markdown(workspace, path)
        else:
            raise ValueError(f"Custom baseline path must be a file or directory: {path}")
        if candidate.variant_id in seen_ids:
            raise ValueError(
                f"Duplicate variant_id '{candidate.variant_id}' in {path} and "
                f"{seen_ids[candidate.variant_id]}"
            )
        seen_ids[candidate.variant_id] = path
        candidates.append(candidate)
    folder_count = sum(1 for c in candidates if c.reference_dir is not None)
    logger.info(
        "Loaded %d custom baseline(s) (%d folder, %d markdown): %s",
        len(candidates),
        folder_count,
        len(candidates) - folder_count,
        [c.variant_id for c in candidates],
    )
    return candidates


def inject_custom_baselines(
    selection: VariantSelection | None,
    custom: Sequence[VariantCandidate],
    *,
    custom_only: bool,
) -> VariantSelection:
    """Merge custom baselines into a selection (between stages 3 and 4 of upstream's pipeline).

    With ``custom_only`` every generated candidate is discarded and no control is kept. Otherwise
    a custom baseline replaces a generated candidate of the same id and is appended after the
    generated ones.
    """
    if selection is None:
        selection = VariantSelection(accepted=[], rejected=[], controls=["control_best"])
    if custom_only:
        return VariantSelection(accepted=list(custom), rejected=[], controls=[])
    accepted = list(selection.accepted)
    for item in custom:
        if any(c.variant_id == item.variant_id for c in accepted):
            accepted = [c for c in accepted if c.variant_id != item.variant_id]
            logger.info("Custom baseline '%s' replaces LLM-generated candidate", item.variant_id)
        accepted.append(item)
    return selection.model_copy(update={"accepted": accepted})


__all__ = ["inject_custom_baselines", "load_custom_baselines"]
