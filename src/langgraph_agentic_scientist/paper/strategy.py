"""The paper strategist (R28): one session per repair round, upstream's validation, family choice.

Upstream (``paper.sh``) runs the strategy stage in a repair loop of at most
``paper_max_repair_rounds`` rounds. Each round renders the same prompt (the repair context comes
from ``strategy_validation_feedback.*`` in the workspace), runs the stage, and validates
``paper_strategy.md`` and ``paper_strategy.json`` with ``validate_strategy_with_feedback``.
A stage failure or a validation failure starts the next round; exhausting the rounds ends the
paper run with an error. ``status: fail`` is a validation error like any other: upstream raises
``paper strategy failed: <reason>`` inside the validator, which writes it as a hard error and
the loop repairs.

In the port the strategist answers through structured output (:class:`StrategyAnswer`); the host
writes both files under upstream's names before validating them, so the validator and every
later prompt read the same files.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..core.state import utc_now
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .contexts import read_text_or_none, write_json, write_text
from .models import StrategyAnswer

PREFERRED_GENERATORS = frozenset(
    {"generate-concept-figure", "matplotlib", "tikz", "native_latex", "manual"}
)
"""Accepted ``figure_specs[].preferred_generator`` values."""


class StrategyInvalid(Exception):
    """A strategy failed upstream's validation; the message is the hard error."""


def validate_strategy(payload: Any, paper_data: Any) -> None:
    """The checks of ``validate_strategy_with_feedback``, in upstream's order and wording.

    Args:
        payload: The parsed ``paper_strategy.json``.
        paper_data: The parsed ``paper_data.json``.

    Raises:
        StrategyInvalid: With upstream's message for the first failed check.
    """
    if not isinstance(payload, dict):
        raise StrategyInvalid("Expected JSON object in paper_strategy.json")
    status = str(payload.get("status", "")).strip()
    if status not in {"ready", "fail"}:
        raise StrategyInvalid("paper_strategy.json must set status to 'ready' or 'fail'")
    if not str(payload.get("selected_prompt_family", "")).strip():
        raise StrategyInvalid("paper_strategy.json missing selected_prompt_family")
    if status == "fail":
        reason = str(payload.get("failure_reason", "")).strip()
        if not reason:
            raise StrategyInvalid(
                "paper_strategy.json status=fail requires non-empty failure_reason"
            )
        raise StrategyInvalid(f"paper strategy failed: {reason}")
    if not isinstance(paper_data, dict):
        raise StrategyInvalid("Expected JSON object in paper_data.json")
    grounding = paper_data.get("grounding_sources")
    if not isinstance(grounding, dict):
        raise StrategyInvalid("paper_data.json grounding_sources must be an object")
    allowed = set(grounding)
    for name in ("recommended_title", "central_question", "central_thesis", "narrative_rationale"):
        if not str(payload.get(name, "")).strip():
            raise StrategyInvalid(f"paper_strategy.json missing non-empty field: {name}")
    plan = payload.get("supplementary_plan")
    if not isinstance(plan, dict):
        raise StrategyInvalid("paper_strategy.json must contain a supplementary_plan object")
    if not str(plan.get("purpose", "")).strip():
        raise StrategyInvalid("paper_strategy.json supplementary_plan.purpose must be non-empty")
    subsections = plan.get("planned_subsections")
    if not isinstance(subsections, list) or not subsections:
        raise StrategyInvalid(
            "paper_strategy.json supplementary_plan.planned_subsections must be a non-empty list"
        )
    prefix = "paper_strategy.json supplementary_plan.planned_subsections"
    for index, row in enumerate(subsections):
        if not isinstance(row, dict):
            raise StrategyInvalid(f"{prefix}[{index}] must be an object")
        for key in ("heading", "grounded_content"):
            if not str(row.get(key, "")).strip():
                raise StrategyInvalid(f"{prefix}[{index}].{key} must be non-empty")
        source_ids = row.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise StrategyInvalid(f"{prefix}[{index}].source_ids must be a non-empty list")
        if not all(isinstance(item, str) and item.strip() for item in source_ids):
            raise StrategyInvalid(f"{prefix}[{index}].source_ids entries must be non-empty strings")
        unknown = sorted(set(source_ids) - allowed)
        if unknown:
            raise StrategyInvalid(
                f"{prefix}[{index}].source_ids contains unknown grounding source ids: {unknown}"
            )
    groups = payload.get("method_groups")
    if not isinstance(groups, list) or not groups:
        raise StrategyInvalid("paper_strategy.json must contain non-empty method_groups list")
    for index, row in enumerate(groups):
        if not isinstance(row, dict):
            raise StrategyInvalid(f"paper_strategy.json method_groups[{index}] must be an object")
        for key in ("name", "role", "include_reason"):
            if not str(row.get(key, "")).strip():
                raise StrategyInvalid(
                    f"paper_strategy.json method_groups[{index}] missing non-empty field: {key}"
                )
    for name in (
        "results_focus",
        "figure_plan",
        "figure_specs",
        "forbidden_headings",
        "forbidden_phrases",
        "notes_for_writer",
    ):
        if not isinstance(payload.get(name), list):
            raise StrategyInvalid(f"paper_strategy.json field must be a list: {name}")
    for index, row in enumerate(payload.get("figure_specs", [])):
        where = f"paper_strategy.json figure_specs[{index}]"
        if not isinstance(row, dict):
            raise StrategyInvalid(f"{where} must be an object")
        generator = str(row.get("preferred_generator", "")).strip()
        for key in ("figure_id", "figure_type"):
            if not str(row.get(key, "")).strip():
                raise StrategyInvalid(f"{where} missing non-empty field: {key}")
        if generator and generator not in PREFERRED_GENERATORS:
            raise StrategyInvalid(f"{where}.preferred_generator has unsupported value: {generator}")
        if (
            bool(row.get("required", False))
            and str(row.get("figure_type", "")).strip() == "diagram"
            and not generator
        ):
            raise StrategyInvalid(f"{where} missing preferred_generator for required diagram")
        for key in ("max_panels", "max_categories_per_panel", "max_series_per_panel"):
            value = row.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise StrategyInvalid(f"{where}.{key} must be positive int")
        if not isinstance(row.get("allow_progression"), bool):
            raise StrategyInvalid(f"{where}.allow_progression must be bool")
        required = row.get("required")
        if required is not None and not isinstance(required, bool):
            raise StrategyInvalid(f"{where}.required must be bool when provided")


async def write_strategy_artifacts(
    workspace: Workspace, paper_dir: str, answer: StrategyAnswer
) -> None:
    """Write ``paper_strategy.md`` and ``paper_strategy.json`` from the structured answer."""
    await write_text(workspace, join(paper_dir, "paper_strategy.md"), answer.memo)
    await write_json(
        workspace,
        join(paper_dir, "paper_strategy.json"),
        answer.strategy.model_dump(mode="json", exclude_none=True),
    )


async def validate_strategy_files(workspace: Workspace, paper_dir: str) -> list[str]:
    """``validate_strategy_with_feedback``: validate the files and write the strategy feedback.

    Returns the hard errors (empty when the strategy is accepted).
    """
    memo_path = join(paper_dir, "paper_strategy.md")
    json_path = join(paper_dir, "paper_strategy.json")
    feedback: dict[str, Any] = {
        "generated_at": utc_now(),
        "paper_dir": paper_dir,
        "ok": False,
        "hard_errors": [],
    }
    try:
        memo = await read_text_or_none(workspace, memo_path)
        raw = await read_text_or_none(workspace, json_path)
        if memo is None:
            raise StrategyInvalid(f"Missing required paper strategy memo: {memo_path}")
        if raw is None:
            raise StrategyInvalid(f"Missing required paper strategy JSON: {json_path}")
        if not memo.strip():
            raise StrategyInvalid(f"Paper strategy memo is empty: {memo_path}")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StrategyInvalid(f"Invalid JSON in {json_path}: {exc}") from exc
        data_path = join(paper_dir, "paper_data.json")
        data_text = await read_text_or_none(workspace, data_path)
        try:
            paper_data = json.loads(data_text) if data_text is not None else None
        except json.JSONDecodeError as exc:
            raise StrategyInvalid(f"Invalid paper data in {data_path}: {exc}") from exc
        if data_text is None:
            raise StrategyInvalid(f"Invalid paper data in {data_path}: missing")
        validate_strategy(payload, paper_data)
    except StrategyInvalid as exc:
        feedback["hard_errors"] = [str(exc)]
    else:
        feedback["ok"] = True
    await write_json(workspace, join(paper_dir, "strategy_validation_feedback.json"), feedback)
    lines = [
        "# Paper strategy validation feedback",
        "",
        f"- generated_at: {feedback['generated_at']}",
        f"- paper_dir: `{paper_dir}`",
        f"- ok: `{feedback['ok']}`",
        "",
    ]
    if feedback["hard_errors"]:
        lines.append("## Hard errors")
        lines.extend(f"- {message}" for message in feedback["hard_errors"])
        lines.append("")
    await write_text(
        workspace,
        join(paper_dir, "strategy_validation_feedback.md"),
        "\n".join(lines).strip() + "\n",
    )
    return list(feedback["hard_errors"])


def strategy_resolution(
    family: str, prompt_key: str, writer_skill: str, skills: list[str]
) -> Mapping[str, Any]:
    """The content of ``paper_strategy_resolution.json`` (``FinalPaperStage.to_dict``)."""
    return {
        "prompt_family": family,
        "prompt_path": prompt_key,
        "writer_skill": writer_skill,
        "selected_skills": list(skills),
    }


__all__ = [
    "PREFERRED_GENERATORS",
    "StrategyInvalid",
    "strategy_resolution",
    "validate_strategy",
    "validate_strategy_files",
    "write_strategy_artifacts",
]
