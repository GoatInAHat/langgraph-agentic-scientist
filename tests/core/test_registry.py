"""The prompt registry renders as upstream does, in one pass, with overrides by key or path."""

from __future__ import annotations

from pathlib import Path

import pytest

from langgraph_agentic_scientist.core.errors import PromptNotFound
from langgraph_agentic_scientist.evaluation.reset_sequence import (
    declared_reset_sequence_matches,
    reduced_draft_reset_sequence_matches,
    reset_sequence_diagnostic,
)
from langgraph_agentic_scientist.prompts import FRAGMENT_MARKERS, PromptRegistry, render_text


@pytest.fixture(scope="module")
def registry() -> PromptRegistry:
    return PromptRegistry.default()


def test_default_registry_has_fragments_harness_and_skills(registry: PromptRegistry) -> None:
    keys = set(registry.keys())
    for _, key in FRAGMENT_MARKERS:
        assert key in keys
    assert "harness/agent_system" in keys
    assert "workflow/evaluation/reset_sequence_diagnostic" in keys
    assert "skills/memory-log/SKILL.md" in keys
    assert {"edit-program", "evaluate-program", "memory-log", "verify-rubrics"} <= set(
        registry.skill_names()
    )


def test_fragments_expand_in_upstream_order(registry: PromptRegistry) -> None:
    custom = registry.with_overrides(
        {
            "fragments/evaluator_evidence": "EVIDENCE[{{CANDIDATE_ENVIRONMENT_RECIPE}}]",
            "fragments/candidate_environment": "ENV[{{EVALUATOR_EVIDENCE_CONTRACT}}]",
        }
    )
    template = "a {{EVALUATOR_EVIDENCE_CONTRACT}} b {{CANDIDATE_ENVIRONMENT_RECIPE}}"
    expanded = custom.expand_fragments(template)
    # Evidence is replaced first, so the environment marker it introduced is expanded next; the
    # evidence marker introduced by the environment text is not expanded again.
    assert expanded == (
        "a EVIDENCE[ENV[{{EVALUATOR_EVIDENCE_CONTRACT}}]] b ENV[{{EVALUATOR_EVIDENCE_CONTRACT}}]"
    )


def test_one_pass_substitution_does_not_rescan_values() -> None:
    rendered = render_text("x={{X}} y={{Y}}", {"X": "{{Y}}", "Y": "why"})
    assert rendered == "x={{Y}} y=why"


def test_unknown_placeholders_stay_literal() -> None:
    assert render_text("{{KNOWN}} {{UNKNOWN}}", {"KNOWN": "k"}) == "k {{UNKNOWN}}"


def test_override_by_key_and_by_path(registry: PromptRegistry, tmp_path: Path) -> None:
    path = tmp_path / "agent.md"
    path.write_text("from a file", encoding="utf-8")
    custom = registry.with_overrides({"harness/agent_system": path, "harness/nl": "by key"})
    assert custom.text("harness/agent_system") == "from a file"
    assert custom.text("harness/nl") == "by key"
    assert registry.text("harness/nl") != "by key", "the original registry is unchanged"
    assert custom.fingerprint() != registry.fingerprint()


def test_unknown_override_key_is_refused(registry: PromptRegistry) -> None:
    with pytest.raises(PromptNotFound):
        registry.with_overrides({"drafting/no_such_prompt": "x"})


def test_design_prompts_skip_fragments(registry: PromptRegistry) -> None:
    custom = registry.with_overrides(
        {"harness/nl": "keep {{EVALUATOR_EVIDENCE_CONTRACT}} and {{V}}"}
    )
    assert custom.render("harness/nl", {"V": "v"}, fragments=False) == (
        "keep {{EVALUATOR_EVIDENCE_CONTRACT}} and v"
    )
    expanded = custom.render("harness/nl", {"V": "v"})
    assert "{{EVALUATOR_EVIDENCE_CONTRACT}}" not in expanded


def test_skill_override_reaches_staged_files(registry: PromptRegistry) -> None:
    custom = registry.with_overrides({"skills/memory-log/SKILL.md": "overridden"})
    assert custom.skill_files("memory-log")["SKILL.md"] == b"overridden"
    assert "scripts/append_memory.py" in custom.skill_files("memory-log")


def test_reset_sequence_rules_and_diagnostic(registry: PromptRegistry) -> None:
    assert declared_reset_sequence_matches((1, 2, 3), (1, 2, 3))
    assert declared_reset_sequence_matches((1, 2, 3, 1), (1, 2, 3))
    assert not declared_reset_sequence_matches((1, 2), (1, 2, 3))
    assert not declared_reset_sequence_matches((), ())
    assert reduced_draft_reset_sequence_matches((1,), (1, 2, 3))
    text = reset_sequence_diagnostic(
        (1, True, 2**70), (1, 2, 3), context="search smoke", prompts=registry
    )
    assert text == (
        "search smoke expected primary [1, 2, 3]; observed [1, '<bool>', '<int bits=71>'] "
        "(count=3). The primary sequence must run exactly once; one trailing deterministic "
        "replay of seed 1 is allowed."
    )
