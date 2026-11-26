"""The workflow texts that live in upstream shell code equal their extraction from upstream.

Upstream keeps these texts inside ``src/scripts/task.sh`` and ``src/scripts/lib/task_gate.sh``
(shell strings, heredocs, and Python ``print`` statements embedded in heredocs). They are written
under ``third_party/ai-scientist-v3/extracted/drafting_scaffold/`` with a header naming the source
and the lines; this module re-extracts each one from the vendored upstream file and compares.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party" / "ai-scientist-v3"
EXTRACTED = THIRD_PARTY / "extracted" / "drafting_scaffold"
HEADER_PREFIX = "<!-- Extracted from "

# (output file, upstream file, first line, last line, kind, names of f-string fields)
SPECS: list[tuple[str, str, int, int, str, dict[str, str]]] = [
    (
        "draft_retry_feedback.md",
        "src/scripts/task.sh",
        1221,
        1229,
        "shell_suffix_after_PROMPT",
        {},
    ),
    (
        "continue_provisional_repair.md",
        "src/scripts/task.sh",
        2202,
        2209,
        "python_print",
        {"encoded": "DIAGNOSTIC_BASE64"},
    ),
    (
        "alignment_report_correction.md",
        "src/scripts/task.sh",
        2472,
        2479,
        "python_print",
        {"diagnostic_base64": "DIAGNOSTIC_BASE64"},
    ),
    (
        "candidate_boundary_repair.md",
        "src/scripts/task.sh",
        2595,
        2602,
        "python_print",
        {"diagnostic_base64": "DIAGNOSTIC_BASE64"},
    ),
    (
        "qualification_repair.md",
        "src/scripts/task.sh",
        2856,
        2874,
        "python_print",
        {
            "diagnostic": "DIAGNOSTIC_BASE64",
            "design": "DESIGN_BASE64",
            "alignment": "ALIGNMENT_REPORT_BASE64",
            "calibration": "CALIBRATION_RECORD_BASE64",
            "post_validation": "POST_VALIDATION_BASE64",
            "boundary_policy": "BOUNDARY_POLICY",
            "repair_policy": "REPAIR_POLICY",
        },
    ),
    (
        "task_gate.md",
        "src/scripts/lib/task_gate.sh",
        5,
        17,
        "heredoc_body",
        {},
    ),
]


def _render(node: ast.expr, names: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name):
                name = value.value.id
                parts.append("{{" + names.get(name, name.upper()) + "}}")
            else:
                raise ValueError(f"unsupported f-string part: {ast.dump(value)}")
        return "".join(parts)
    raise ValueError(f"not a string expression: {ast.dump(node)}")


def extract_shell_text(source: str, first: int, last: int, kind: str, names: dict[str, str]) -> str:
    """Return the text value that upstream's shell code builds in lines ``first``-``last``."""
    lines = source.splitlines()[first - 1 : last]
    if kind == "python_print":
        tree = ast.parse("\n".join(lines))
        (statement,) = tree.body
        assert isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
        (argument,) = statement.value.args
        return _render(argument, names)
    if kind == "heredoc_body":
        return "".join(line + "\n" for line in lines)
    if kind == "shell_suffix_after_PROMPT":
        text = "\n".join([lines[0].lstrip(), *lines[1:]]) + "\n"
        prefix = 'retry_prompt="$PROMPT'
        assert text.startswith(prefix) and text.endswith('"\n')
        body = text[len(prefix) : -2]
        body = body.replace("\\`", "`")
        return body.replace("${diagnostic#MISSING:}", "{{DIAGNOSTIC}}")
    raise ValueError(kind)


def header(upstream: str, first: int, last: int, commit: str) -> str:
    """Return the first line of an extracted file."""
    return (
        f"{HEADER_PREFIX}{upstream} lines {first}-{last} of AI-Scientist-v3 {commit[:12]} by "
        "tests/drafting/test_scaffold_texts.py; deriving a package file drops this line. -->"
    )


def expected_files() -> dict[str, str]:
    """Return every extracted file's expected content."""
    commit = (THIRD_PARTY / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()
    files: dict[str, str] = {}
    for output, upstream, first, last, kind, names in SPECS:
        source = (THIRD_PARTY / upstream).read_text(encoding="utf-8")
        value = extract_shell_text(source, first, last, kind, names)
        files[output] = header(upstream, first, last, commit) + "\n" + value
    return files


@pytest.mark.parametrize("name", [spec[0] for spec in SPECS])
def test_extracted_file_equals_its_extraction(name: str) -> None:
    path = EXTRACTED / name
    assert path.is_file(), f"missing extracted file {name}"
    assert path.read_text(encoding="utf-8") == expected_files()[name]


def test_no_unlisted_extracted_file() -> None:
    present = {path.name for path in EXTRACTED.iterdir() if path.is_file()}
    python_extractions = {"repair_history_heading.md", "alignment_history_heading.md"}
    assert present == {spec[0] for spec in SPECS} | python_extractions


def test_the_placeholders_of_the_texts_are_the_documented_ones() -> None:
    expected = {
        "draft_retry_feedback.md": {"DIAGNOSTIC"},
        "continue_provisional_repair.md": {"DIAGNOSTIC_BASE64"},
        "alignment_report_correction.md": {"DIAGNOSTIC_BASE64"},
        "candidate_boundary_repair.md": {"DIAGNOSTIC_BASE64"},
        "qualification_repair.md": {
            "DIAGNOSTIC_BASE64",
            "DESIGN_BASE64",
            "ALIGNMENT_REPORT_BASE64",
            "CALIBRATION_RECORD_BASE64",
            "POST_VALIDATION_BASE64",
            "BOUNDARY_POLICY",
            "REPAIR_POLICY",
        },
        "task_gate.md": set(),
    }
    for name, names in expected.items():
        text = (EXTRACTED / name).read_text(encoding="utf-8")
        assert set(re.findall(r"\{\{([A-Z_0-9]+)\}\}", text)) == names, name
