"""Staging upstream skills into a role's workspace (upstream ``skill_scope.py``).

Upstream stages each step's skill set as ``SKILL.md`` folders under the agent CLI's configuration
directory (``.claude/skills/<name>/``, ...). The port stages the same skills, as sibling
directories, under ``<root>/.agents/skills/<name>/`` (adaptation A4), so that relative links between
skills (``../write-paper/...``) and each skill's own ``scripts/...`` stay true. ``SKILL.md`` is
rendered with the role's placeholder values, as upstream's drivers render staged skills; scripts
and templates are copied byte for byte.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..core.errors import PromptNotFound, SkillNotFound
from ..prompts import PromptRegistry, render_text
from ..workspace.paths import join
from ..workspace.protocol import Workspace

SKILLS_DIR = ".agents/skills"

SKILLS_BY_STEP: dict[str, tuple[str, ...]] = {
    "idea": ("create-idea", "memory-log"),
    "run": ("edit-program", "evaluate-program", "verify-rubrics", "memory-log"),
    "run_harness_eval": ("edit-program", "memory-log"),
    "variant_edit": ("edit-program", "memory-log"),
    "variant_verify": ("verify-rubrics", "memory-log"),
    "task": ("draft-task", "memory-log"),
    "paper": (
        "retrieve-references",
        "generate-concept-figure",
        "scientific-evidence",
        "write-paper",
        "validate-paper",
        "memory-log",
    ),
}
"""Upstream ``SKILLS_BY_STEP``: the skill set of each workflow step."""

PAPER_SKILLS_BY_PROFILE: dict[str, tuple[str, ...]] = {
    "default": SKILLS_BY_STEP["paper"],
    "standalone": (
        "retrieve-references",
        "generate-concept-figure",
        "scientific-evidence",
        "write-paper-standalone",
        "validate-paper",
        "memory-log",
    ),
}
"""Upstream ``PAPER_SKILLS_BY_PROFILE``."""

PAPER_WRITER_SKILLS = frozenset({"write-paper", "write-paper-standalone"})
PAPER_TEMPLATES_SKILL = "write-paper"
"""The writer skill whose ``templates/`` tree every paper writer skill shares."""


def skills_for_step(step: str, *, paper_profile: str = "default") -> tuple[str, ...]:
    """Return the configured skill names of a workflow step (upstream ``skills_for_step``)."""
    if step == "paper":
        selected = PAPER_SKILLS_BY_PROFILE.get(paper_profile)
        if selected is None:
            available = ", ".join(sorted(PAPER_SKILLS_BY_PROFILE))
            raise ValueError(
                f"Unknown paper profile '{paper_profile}'. Available profiles: {available}"
            )
        return selected
    selected = SKILLS_BY_STEP.get(step)
    if selected is None:
        available = ", ".join(sorted(SKILLS_BY_STEP))
        raise ValueError(f"Unknown step '{step}'. Available steps: {available}")
    return selected


def skill_dir(root: str, name: str) -> str:
    """Return the workspace directory a skill is staged in for an agent rooted at ``root``."""
    return join(root, SKILLS_DIR, name)


async def stage_skills(
    workspace: Workspace,
    root: str,
    names: Sequence[str],
    *,
    prompts: PromptRegistry,
    values: Mapping[str, str],
) -> list[str]:
    """Stage skills under ``<root>/.agents/skills/`` and return the staged names.

    ``SKILL.md`` is rendered with ``values`` in one pass (unknown placeholders stay literal);
    every other file is copied unchanged. A paper writer skill other than ``write-paper`` also
    receives ``write-paper``'s ``templates/`` tree, as upstream ``stage_paper_templates`` does.

    Raises:
        SkillNotFound: When a skill is not shipped with the registry.
    """
    staged: list[str] = []
    for name in names:
        try:
            files = dict(prompts.skill_files(name))
        except PromptNotFound as exc:
            raise SkillNotFound(name) from exc
        if name in PAPER_WRITER_SKILLS and name != PAPER_TEMPLATES_SKILL:
            try:
                template_files = prompts.skill_files(PAPER_TEMPLATES_SKILL)
            except PromptNotFound:
                template_files = {}
            for path, data in template_files.items():
                if path.startswith("templates/") and path not in files:
                    files[path] = data
        target = skill_dir(root, name)
        for path, data in sorted(files.items()):
            content = data
            if path == "SKILL.md":
                content = render_text(data.decode("utf-8"), values).encode("utf-8")
            await workspace.write_bytes(f"{target}/{path}", content)
        staged.append(name)
    return staged


__all__ = [
    "PAPER_SKILLS_BY_PROFILE",
    "PAPER_TEMPLATES_SKILL",
    "PAPER_WRITER_SKILLS",
    "SKILLS_BY_STEP",
    "SKILLS_DIR",
    "skill_dir",
    "skills_for_step",
    "stage_skills",
]
