"""Helpers of the variants phase for the roles it runs: model labels and skill values."""

from __future__ import annotations

from collections.abc import Mapping

from ..context import ScientistContext
from ..core.render import task_sections
from ..models import Role
from .constants import SAFE_LABEL


def model_label(ctx: ScientistContext, role: Role) -> str | None:
    """Return a name for the model that plays ``role`` (``model_name`` or ``model`` attribute)."""
    model = ctx.models.for_role(role)
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def agent_name(ctx: ScientistContext, role: Role) -> str:
    """Return the ``AGENT_NAME`` of the analysis CSVs: the role's model name, or ``unknown``.

    The CSV contract accepts labels of letters, digits and ``_.:/|-`` only; any other name is
    replaced by ``unknown``.
    """
    label = model_label(ctx, role)
    if label is None or not SAFE_LABEL.fullmatch(label):
        return "unknown"
    return label


def variant_skill_values(
    *,
    task_md: str | None,
    attempt_id: str,
    variant_id: str,
    max_evals: int,
    active_idea: str,
    active_rubrics: str,
) -> dict[str, str]:
    """Return the placeholder values upstream's ``variant_agent.sh`` gives staged skills."""
    goal, rubrics = task_sections(task_md)
    return {
        "TASK_DIR": ".",
        "RESULTS_DIR": ".",
        "TASK_MD_PATH": "task.md",
        "TASK_GOAL": goal,
        "TASK_RUBRICS": rubrics,
        "ACTIVE_IDEA_JSON": active_idea,
        "ACTIVE_RUBRICS_JSON": active_rubrics,
        "ACTIVE_IDEA_FILE": "./active_idea.json",
        "ACTIVE_RUBRICS_FILE": "./active_rubrics.json",
        "ATTEMPT_ID": attempt_id,
        "IDEA_ID": variant_id,
        "IDEA_DIR_NAME": variant_id,
        "ATTEMPT_DIR_NAME": "",
        "MAX_PROGRAM_EVALS_PER_ATTEMPT": str(max_evals),
        "GLOBAL_MEMORY_MD_FILE": "./memory.md",
        "GLOBAL_MEMORY_JSONL_FILE": "./memory.jsonl",
        "GLOBAL_MEMORY_RECENT_SUMMARY": "- none",
    }


def agent_environment(
    *,
    variant_id: str,
    variant_kind: str,
    base_program: str,
    attempt_id: str,
    max_evals: int,
    num_replications: int,
    evaluation_id: str | None = None,
) -> dict[str, str]:
    """Return the environment variables of the agent's commands (upstream's variant exports)."""
    env: dict[str, str] = {
        "VARIANT_ID": variant_id,
        "VARIANT_KIND": variant_kind,
        "BASE_PROGRAM": base_program,
        "MAX_PROGRAM_EVALS_PER_ATTEMPT": str(max_evals),
        "ATTEMPT_ID": attempt_id,
        "IDEA_ID": variant_id,
        "IDEA_DIR_NAME": variant_id,
        "ATTEMPT_DIR_NAME": "",
    }
    if num_replications > 1:
        env["NUM_REPLICATIONS"] = str(num_replications)
    if evaluation_id is not None:
        env["EVALUATION_ID"] = evaluation_id
    return env


def trusted_environment(
    *,
    variant_id: str,
    variant_kind: str,
    base_program: str,
    agent: str,
    evaluation_id: str,
    num_replications: int,
) -> Mapping[str, str]:
    """Return the identity environment of an evaluation (``host_env`` of upstream's executor)."""
    env = {
        "VARIANT_ID": variant_id,
        "VARIANT_KIND": variant_kind,
        "BASE_PROGRAM": base_program,
        "AGENT_NAME": agent,
        "EVALUATION_ID": evaluation_id,
    }
    if num_replications > 1:
        env["NUM_REPLICATIONS"] = str(num_replications)
    return env


__all__ = [
    "agent_environment",
    "agent_name",
    "model_label",
    "trusted_environment",
    "variant_skill_values",
]
