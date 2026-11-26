"""The design prompts are upstream's text: derived from vendored files by declared changes only."""

from __future__ import annotations

import difflib
import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from langgraph_agentic_scientist import PromptRegistry
from langgraph_agentic_scientist.prompts import FRAGMENT_MARKERS, placeholders

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party" / "ai-scientist-v3"
PACKAGE = REPO_ROOT / "src" / "langgraph_agentic_scientist"
CHANGES = PACKAGE / "prompts" / "changes" / "drafting_design.toml"
DRIVER = THIRD_PARTY / "src" / "scripts" / "lib" / "task_design.sh"

ROLE_PROMPTS = {
    "design_task": "design_task",
    "validate_task_design": "validate_task_design",
    "repair_task_design": "repair_task_design",
    "repair_task_design_structure": "repair_task_design_structure",
    "audit_task_design_delta": "audit_task_design_delta",
}
# the placeholders the graph fills for each role
EXPECTED_PLACEHOLDERS = {
    "design_task": {
        "TASK_MD_PATH",
        "TASK_DESIGN_PATH",
        "MAX_IDEA_CYCLES",
        "MAX_PROGRAM_EVALS_PER_ATTEMPT",
    },
    "validate_task_design": {
        "TASK_MD_PATH",
        "TASK_DESIGN_PATH",
        "TASK_DESIGN_FINGERPRINT",
        "TASK_DESIGN_REVIEW_PATH",
    },
    "repair_task_design": {
        "TASK_DESIGN_PATCH_PATH",
        "TASK_MD_PATH",
        "TASK_DESIGN_PATH",
        "TASK_DESIGN_REVIEW",
        "TASK_DESIGN_FINGERPRINT",
    },
    "repair_task_design_structure": {"TASK_DESIGN_PATH", "TASK_MD_PATH", "TASK_DESIGN_REVIEW"},
    "audit_task_design_delta": {
        "TASK_MD_PATH",
        "TASK_PARENT_DESIGN_PATH",
        "TASK_DESIGN_PATH",
        "TASK_PARENT_REVIEW_PATH",
        "TASK_CANDIDATE_REVIEW_PATH",
        "TASK_DESIGN_FINGERPRINT",
        "TASK_DESIGN_REVIEW_PATH",
    },
}
# the lines of each upstream prompt that the declared changes replace (everything else is verbatim)
CHANGED_LINES = {
    "design_task": 1,
    "validate_task_design": 1,
    "repair_task_design": 1,
    "repair_task_design_structure": 2,
    "audit_task_design_delta": 1,
}


@pytest.fixture(scope="module")
def derive() -> ModuleType:
    path = REPO_ROOT / "tools" / "derive_prompts.py"
    spec = importlib.util.spec_from_file_location("_tool_derive_prompts_design", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _entries(derive: ModuleType) -> dict[str, object]:
    original = derive.CHANGES_DIR
    try:
        _, files = derive.load_changes(CHANGES.parent)
    finally:
        derive.CHANGES_DIR = original
    return {entry.target: entry for entry in files if entry.source == CHANGES.name}


def test_every_design_file_equals_its_declared_derivation(derive: ModuleType) -> None:
    entries = _entries(derive)
    assert len(entries) == 9
    for target, entry in entries.items():
        assert (PACKAGE / target).read_bytes() == derive.derive(entry), target


def test_the_whole_derivation_check_passes_for_the_design_files(derive: ModuleType) -> None:
    problems = [
        problem
        for problem in derive.run(check=True)
        if "drafting/" in problem and ("design" in problem or "task_design" in problem)
    ]
    assert problems == []


def test_the_role_prompts_differ_from_upstream_only_by_the_declared_lines() -> None:
    for name, changed in CHANGED_LINES.items():
        upstream = (THIRD_PARTY / "src" / "prompts" / f"{name}.md").read_text().splitlines()
        package = (PACKAGE / "prompts" / "drafting" / f"{name}.md").read_text().splitlines()
        assert len(upstream) == len(package)
        removed = [line for line in difflib.ndiff(upstream, package) if line.startswith("- ")]
        assert len(removed) == changed, (name, removed)
        # the sentence that told the agent to write a file now says the harness writes it
        added = [line for line in difflib.ndiff(upstream, package) if line.startswith("+ ")]
        assert all("structured response" in line or "before you answer" in line for line in added)


def test_the_declared_changes_are_a3_and_one_wording_change() -> None:
    import tomllib

    data = tomllib.loads(CHANGES.read_text(encoding="utf-8"))
    kinds = [change["kind"] for entry in data["file"] for change in entry.get("change", [])]
    assert sorted(kinds) == ["A3", "A3", "A3", "A3", "A3", "other"]
    for entry in data["file"]:
        assert entry["reason"].strip()
        for change in entry.get("change", []):
            assert change["reason"].strip() and change["old"] != change["new"]


def test_the_prompt_placeholders_are_the_ones_the_graph_fills() -> None:
    registry = PromptRegistry.default()
    for name, expected in EXPECTED_PLACEHOLDERS.items():
        assert placeholders(registry.text(f"drafting/{name}")) == expected, name
        for marker, _ in FRAGMENT_MARKERS:
            assert marker not in registry.text(f"drafting/{name}")


def test_the_dormant_adjudicator_is_vendored_and_carried_nowhere() -> None:
    assert (THIRD_PARTY / "src" / "prompts" / "adjudicate_task_design.md").is_file()
    assert "drafting/adjudicate_task_design" not in PromptRegistry.default()
    assert not (PACKAGE / "prompts" / "drafting" / "adjudicate_task_design.md").exists()


# --- the texts that live in the shell driver ----------------------------------------------------


def _shell_string(line: str) -> str:
    """Decode the ANSI-C quoted string ``$'...'`` that ends a line of the driver."""
    match = re.search(r"\$'((?:[^'\\]|\\.)*)'\s*$", line)
    assert match, line
    text = match.group(1)
    out: list[str] = []
    index = 0
    escapes = {"n": "\n", "\\": "\\", "'": "'", "t": "\t"}
    while index < len(text):
        if text[index] == "\\":
            out.append(escapes[text[index + 1]])
            index += 2
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def _driver_lines() -> list[str]:
    return DRIVER.read_text(encoding="utf-8").split("\n")


DRIVER_TEXTS = {
    "focused_repair_verification": (
        846,
        848,
        "TARGET_DIMENSIONS",
        'review_prompt+="$target_dimensions"',
    ),
    "review_retry_feedback": (856, 858, "DIAGNOSTIC", 'review_prompt+="$diagnostic"'),
    "delta_retry_feedback": (1225, 1227, "DIAGNOSTIC", 'attempt_prompt+="$diagnostic"'),
    "patch_format_retry_feedback": (
        1090,
        1092,
        "DIAGNOSTIC",
        'repair_prompt+="${previous_diagnostic:0:8192}"',
    ),
}


@pytest.mark.parametrize("name", sorted(DRIVER_TEXTS))
def test_the_driver_texts_are_the_shell_strings_of_the_vendored_driver(name: str) -> None:
    first, last, variable, middle = DRIVER_TEXTS[name]
    lines = _driver_lines()
    assert lines[first].strip() == middle, "the variable between the two strings moved"
    expected = (
        _shell_string(lines[first - 1]) + "{{" + variable + "}}" + _shell_string(lines[last - 1])
    )
    extracted = (THIRD_PARTY / "extracted" / "drafting_design" / f"{name}.md").read_text(
        encoding="utf-8"
    )
    header, _, body = extracted.partition("\n")
    assert header.startswith("<!-- Extracted from src/scripts/lib/task_design.sh")
    assert f"lines {first}-{last}" in header
    assert body == expected
    package = (PACKAGE / "prompts" / "workflow" / "drafting" / f"{name}.md").read_text(
        encoding="utf-8"
    )
    assert package == expected


def test_the_retry_texts_render_with_their_diagnostics() -> None:
    registry = PromptRegistry.default()
    review = registry.render(
        "workflow/drafting/review_retry_feedback", {"DIAGNOSTIC": "bad {{X}}"}, fragments=False
    )
    assert review.startswith("\n\n# Retry feedback\nYour previous review artifact failed")
    assert review.endswith("```text\nbad {{X}}\n```\n"), "a diagnostic is never rescanned"
    delta = registry.render(
        "workflow/drafting/delta_retry_feedback", {"DIAGNOSTIC": "d"}, fragments=False
    )
    assert (
        "Your previous delta artifact failed its contract. Correct the artifact and repeat" in delta
    )
    patch = registry.render(
        "workflow/drafting/patch_format_retry_feedback", {"DIAGNOSTIC": "p"}, fragments=False
    )
    assert "# Format retry feedback\nThe prior patch artifact was invalid." in patch
    focused = registry.render(
        "workflow/drafting/focused_repair_verification",
        {"TARGET_DIMENSIONS": '["compute_realism"]'},
        fragments=False,
    )
    assert focused.endswith('Target dimension IDs: `["compute_realism"]`\n')
    assert "# Focused repair verification\nThis is a delta review after a repair." in focused


def test_a_caller_can_override_a_design_prompt_entry_by_entry() -> None:
    registry = PromptRegistry.default().with_overrides(
        {"workflow/drafting/review_retry_feedback": "RETRY {{DIAGNOSTIC}}"}
    )
    assert (
        registry.render(
            "workflow/drafting/review_retry_feedback", {"DIAGNOSTIC": "x"}, fragments=False
        )
        == "RETRY x"
    )
    assert registry.fingerprint() != PromptRegistry.default().fingerprint()
