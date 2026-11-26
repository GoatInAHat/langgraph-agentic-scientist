"""``repair_task.md`` splits on ``# Role`` and ``# Repair policy`` as upstream splits it."""

from __future__ import annotations

from pathlib import Path

import pytest

from langgraph_agentic_scientist.core.render import split_repair_policy
from langgraph_agentic_scientist.prompts import PromptRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "third_party/ai-scientist-v3/src/prompts/repair_task.md"


def _upstream_parts(template: str) -> tuple[str, str]:
    # The exact partition of upstream task.sh (qualification repair prompt assembly).
    boundary_policy = template.partition("# Role\n")[0]
    _, heading, repair_policy = template.partition("# Repair policy\n")
    assert heading
    return boundary_policy, repair_policy


def test_upstream_headings_occur_exactly_once() -> None:
    text = UPSTREAM.read_text(encoding="utf-8")
    assert text.count("# Role\n") == 1
    assert text.count("# Repair policy\n") == 1


def test_split_yields_upstreams_parts() -> None:
    registry = PromptRegistry.default()
    expanded = registry.expand_fragments(UPSTREAM.read_text(encoding="utf-8"))
    assert split_repair_policy(expanded) == _upstream_parts(expanded)
    boundary, policy = split_repair_policy(expanded)
    assert boundary and policy
    assert "# Role" not in boundary


def test_package_prompt_keeps_the_headings_when_present() -> None:
    registry = PromptRegistry.default()
    if "drafting/repair_task" not in registry:
        pytest.skip("the drafting package's repair_task prompt is not in this tree")
    text = registry.text("drafting/repair_task")
    assert text.count("# Role\n") == 1
    assert text.count("# Repair policy\n") == 1


def test_missing_policy_is_an_error() -> None:
    with pytest.raises(ValueError, match="repair policy is missing"):
        split_repair_policy("# Role\nonly a role\n")
