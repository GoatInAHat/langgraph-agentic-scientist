"""The investigation prompts and workflow texts derive from upstream bytes."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import Any

from langgraph_agentic_scientist.prompts import PromptRegistry

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/ai-scientist-v3"
sys.path.insert(0, str(ROOT / "tools"))

import derive_prompts  # noqa: E402


def test_every_investigation_text_derives_from_upstream() -> None:
    assert derive_prompts.run(check=True) == []


def test_the_candidate_author_prompt_is_upstream_byte_for_byte() -> None:
    registry = PromptRegistry.default()
    upstream = (UPSTREAM / "src/prompts/harness_edit.md").read_bytes()
    assert registry.text("investigation/harness_edit").encode() == upstream


def test_the_search_prompt_names_the_tools_and_keeps_every_budget_rule() -> None:
    text = PromptRegistry.default().text("investigation/general")
    original = (UPSTREAM / "src/prompts/general.md").read_text()
    assert (
        "src.investigation" not in text
        and "WRAPPER_PYTHON" not in text
        and "timeout` utility" not in text
    )
    assert "`clone_program` tool" in text and "evaluate_program(program_dir=" in text
    for rule in (
        "Never start editing `program_<idx+1>` before writing rubric artifacts for "
        "`program_<idx>`.",
        "Never use `cp`, `cp -r`, `cp -a`, `rsync`, or another recursive copy to create a "
        "candidate.",
        "Stop the attempt immediately when latest `rubric_status.json` has `overall_pass: true`.",
        "- Never ask user questions.",
        "At least one structured event for this attempt must include `attempt_id` exactly equal to",
    ):
        assert rule in original and rule in text
    # every line of the upstream prompt that does not name the helpers survives unchanged
    kept = set(text.splitlines())
    missing = [
        line
        for line in original.splitlines()
        if "evaluate.py" not in line and "program_clone" not in line and line not in kept
    ]
    wrapper_block = {
        "- Use the task runtime for every official evaluator call so the evaluator and candidate "
        "retain the task's installed dependencies:",
        "```bash",
        '"${TASK_PYTHON:-$WRAPPER_PYTHON}" -I -m src.investigation.evaluate_program \\',
        '    --program-dir "{{RESULTS_DIR}}/program_<idx>" \\',
        '    --timeout-seconds "${EVALUATION_TIMEOUT_SECONDS:-21600}"',
    }
    assert set(missing) - wrapper_block == {
        "- Do not wrap the command with the shell `timeout` utility. If the shell tool supports an "
        "unlimited/omitted timeout, omit its optional timeout. Otherwise, configure it longer than "
        "the effective evaluator timeout plus its cleanup margin. Never choose a shell-tool "
        "timeout shorter than the evaluator timeout."
    }


def test_the_idea_prompt_answers_through_structured_output() -> None:
    text = PromptRegistry.default().text("investigation/idea")
    assert "Write exactly these files" not in text
    assert "Return the idea bundle as your structured response" in text
    assert "{{IDEA_OUTPUT_DIR}}/memory.md" in text
    assert "No Execution During Idea Generation" in text
    skill = PromptRegistry.default().text("skills/create-idea/SKILL.md")
    assert "create_idea_placeholder" not in skill
    assert "source_contains" in skill, "the rubric and verification rules stay"
    assert "skills/create-idea/scripts/create_idea_placeholder.py" not in PromptRegistry.default()


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def test_the_hand_extracted_formats_occur_in_the_upstream_source() -> None:
    sources = "".join(
        (UPSTREAM / name).read_text()
        for name in (
            "src/scripts/idea.sh",
            "src/scripts/run.sh",
            "src/investigation/harness_eval.py",
            "src/investigation/harness_feedback.py",
        )
    )
    sources = re.sub(r"\{[^{}]*\}", "", re.sub(r'"\s*\n\s*f?"', "", sources))
    registry = PromptRegistry.default()
    for key in ("context_blocks", "harness_feedback"):
        data = tomllib.loads(registry.text(f"workflow/investigation/{key}.toml"))
        for string in _strings(data):
            for fragment in re.split(r"\{\{[A-Za-z0-9_]+\}\}", string):
                fragment = fragment.strip()
                if len(fragment) >= 8:
                    assert fragment in sources, f"{key}: {fragment!r} is not in the upstream source"


def test_the_prune_banner_is_upstreams() -> None:
    banner = PromptRegistry.default().text("workflow/investigation/prune_warning")
    assert "Pruning may be ineffective without containers!" in banner
    assert "Use container.enabled=true for full round isolation." in banner
