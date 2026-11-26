"""``repair_task.md`` keeps its two headings byte-identical and splits as upstream does."""

from __future__ import annotations

from pathlib import Path

from langgraph_agentic_scientist.core.render import split_repair_policy
from langgraph_agentic_scientist.drafting.qualification import render_qualification_repair_prompt
from langgraph_agentic_scientist.prompts import PromptRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "third_party/ai-scientist-v3/src/prompts/repair_task.md"


def upstream_parts(template: str) -> tuple[str, str]:
    """The exact partition of upstream's ``task.sh`` (qualification repair prompt assembly)."""
    boundary_policy = template.partition("# Role\n")[0]
    _, heading, repair_policy = template.partition("# Repair policy\n")
    assert heading
    return boundary_policy, repair_policy


def test_the_package_prompt_has_each_heading_exactly_once() -> None:
    text = PromptRegistry.default().text("drafting/repair_task")
    assert text.count("# Role\n") == 1
    assert text.count("# Repair policy\n") == 1
    assert text.index("# Role\n") < text.index("# Inputs") < text.index("# Repair policy\n")


def test_the_package_prompt_splits_into_upstreams_parts() -> None:
    registry = PromptRegistry.default()
    package = registry.expand_fragments(registry.text("drafting/repair_task"))
    upstream = registry.expand_fragments(UPSTREAM.read_text(encoding="utf-8"))
    boundary, policy = split_repair_policy(package)
    expected_boundary, expected_policy = upstream_parts(package)
    assert (boundary, policy) == (expected_boundary, expected_policy)
    # the only declared change (A2) lies in the policy part; the boundary part is upstream's
    assert boundary == upstream_parts(upstream)[0]
    assert "report_blocker" in policy and "Finish non-zero" not in policy
    assert "# Role" not in boundary and "# Inputs" not in policy


def test_the_qualification_prompt_embeds_both_parts_under_the_policy_heading() -> None:
    registry = PromptRegistry.default()
    expanded = registry.expand_fragments(registry.text("drafting/repair_task"))
    boundary, policy = split_repair_policy(expanded)
    prompt = render_qualification_repair_prompt(
        registry,
        diagnostic="d",
        design="{}",
        alignment_report="",
        calibration="",
        post_validation="",
        alignment_history="",
        repair_history="",
    )
    assert prompt.endswith(
        "# Shared scaffold repair policy\n" + boundary + "\n" + policy.rstrip("\n")
    )
    assert "# Inputs" not in prompt.split("# Shared scaffold repair policy")[0]
    assert "{{" not in prompt.split("# Shared scaffold repair policy")[0]


def test_an_override_of_the_prompt_is_what_the_qualification_repair_splits() -> None:
    registry = PromptRegistry.default().with_overrides(
        {"drafting/repair_task": "BOUNDARY\n# Role\nrole text\n# Repair policy\nMY POLICY\n"}
    )
    prompt = render_qualification_repair_prompt(
        registry,
        diagnostic="",
        design="",
        alignment_report="",
        calibration="",
        post_validation="",
        alignment_history="",
        repair_history="",
    )
    assert prompt.endswith("# Shared scaffold repair policy\nBOUNDARY\n\nMY POLICY")
