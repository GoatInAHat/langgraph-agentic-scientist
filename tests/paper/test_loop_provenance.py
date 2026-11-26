"""The paper-loop texts that live in upstream code equal their extraction from the vendored files.

Upstream keeps the context strings of the paper stages inside ``src/scripts/paper.sh`` and
``src/scripts/paper_edit.sh`` (Python heredocs and shell strings) and in the packet builders of
``src/orchestrator/paper_evidence_audit.py`` and ``src/orchestrator/paper_review.py`` (lists of
lines). ``tools/derive_prompts.py`` extracts single string expressions of Python files; the texts
below are of other shapes, so they are written under
``third_party/ai-scientist-v3/extracted/paper_loop/`` with a header naming the source and the lines,
and this module re-extracts each one from the vendored upstream file and compares. The package
files derived from them are checked by ``tools/derive_prompts.py --check``.
"""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party" / "ai-scientist-v3"
EXTRACTED = THIRD_PARTY / "extracted" / "paper_loop"
HEADER_PREFIX = "<!-- Extracted from "


@dataclass(frozen=True)
class Spec:
    """One text extracted from upstream code.

    ``kind`` is ``python`` (one string expression of Python code inside a heredoc, after
    ``strip_prefix`` is removed), ``python_lines`` (a run of string literals that are elements of
    a Python list of lines, joined with newlines) or ``shell`` (a double-quoted shell string whose
    ``strip_prefix`` and ``strip_suffix`` are removed).
    """

    output: str
    upstream: str
    first: int
    last: int
    kind: str
    names: dict[str, str] = field(default_factory=dict)
    strip_prefix: str = ""
    strip_suffix: str = ""


PAPER_SH = "src/scripts/paper.sh"
PAPER_EDIT_SH = "src/scripts/paper_edit.sh"
AUDIT_PY = "src/orchestrator/paper_evidence_audit.py"
REVIEW_PY = "src/orchestrator/paper_review.py"

SPECS: list[Spec] = [
    Spec("strategy_context_ready.md", PAPER_SH, 1368, 1372, "python"),
    Spec("strategy_context_pending.md", PAPER_SH, 1376, 1378, "python"),
    Spec("strategy_context_none.md", PAPER_SH, 1382, 1383, "python"),
    Spec(
        "validation_feedback_context_present.md",
        PAPER_SH,
        1395,
        1400,
        "python",
        names={
            "validation_feedback_json.name": "FEEDBACK_JSON_NAME",
            "validation_feedback_md.name": "FEEDBACK_MD_NAME",
        },
    ),
    Spec("validation_feedback_context_absent.md", PAPER_SH, 1404, 1404, "python"),
    Spec("review_feedback_context_present.md", PAPER_SH, 1411, 1417, "python"),
    Spec("review_feedback_context_pending.md", PAPER_SH, 1421, 1421, "python"),
    Spec(
        "review_feedback_context_disabled.md",
        PAPER_SH,
        1424,
        1424,
        "python",
        strip_prefix="review_feedback_context = ",
    ),
    Spec(
        "openai_key_available.md",
        PAPER_SH,
        536,
        536,
        "shell",
        strip_prefix='PAPER_OPENAI_API_KEY_CONTEXT="',
        strip_suffix='"',
    ),
    Spec(
        "openai_key_missing.md",
        PAPER_SH,
        539,
        539,
        "shell",
        strip_prefix='PAPER_OPENAI_API_KEY_CONTEXT="',
        strip_suffix='"',
    ),
    Spec(
        "gemini_key_available.md",
        PAPER_SH,
        548,
        548,
        "shell",
        strip_prefix='PAPER_GEMINI_API_KEY_CONTEXT="',
        strip_suffix='"',
    ),
    Spec(
        "gemini_key_missing.md",
        PAPER_SH,
        551,
        551,
        "shell",
        strip_prefix='PAPER_GEMINI_API_KEY_CONTEXT="',
        strip_suffix='"',
    ),
    Spec("ranking_repair.md", PAPER_SH, 3077, 3078, "shell", strip_suffix='"'),
    Spec("global_memory_summary_header.md", PAPER_SH, 862, 865, "python_lines"),
    Spec("edit_strategy_context_edit.md", PAPER_EDIT_SH, 1929, 1932, "python"),
    Spec(
        "edit_strategy_context_fresh.md",
        PAPER_EDIT_SH,
        1934,
        1934,
        "python",
        strip_prefix="else ",
    ),
    Spec("edit_context_edit.md", PAPER_EDIT_SH, 1937, 1941, "python"),
    Spec("edit_context_fresh.md", PAPER_EDIT_SH, 1943, 1943, "python", strip_prefix="else "),
    Spec("paper_edit_contract.md", PAPER_EDIT_SH, 1353, 1365, "python_lines"),
    Spec("audit_packet_intro.md", AUDIT_PY, 1413, 1423, "python_lines"),
    Spec("review_packet_intro.md", REVIEW_PY, 189, 197, "python_lines"),
    Spec("edit_interpretation_guidance.md", PAPER_EDIT_SH, 1334, 1336, "python_lines"),
    Spec(
        "edit_results_linkage_intro.md",
        PAPER_EDIT_SH,
        1390,
        1390,
        "python",
        strip_prefix="lines.append(",
        strip_suffix=")",
    ),
    Spec("edit_results_linkage_outro.md", PAPER_EDIT_SH, 1422, 1423, "python_lines"),
]


def _commit() -> str:
    return (THIRD_PARTY / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()


def _render(node: ast.expr, names: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                expression = ast.unparse(value.value)
                if expression not in names:
                    raise AssertionError(f"name the f-string field {expression!r}")
                parts.append("{{" + names[expression] + "}}")
            else:
                raise AssertionError(f"unsupported f-string part: {ast.dump(value)}")
        return "".join(parts)
    raise AssertionError(f"not a string expression: {ast.dump(node)}")


def extract(spec: Spec) -> str:
    """Return the value of the text that ``spec`` names, from the vendored upstream file."""
    lines = (THIRD_PARTY / spec.upstream).read_text(encoding="utf-8").split("\n")
    selected = "\n".join(lines[spec.first - 1 : spec.last])
    if spec.kind == "shell":
        text = textwrap.dedent(selected) if spec.strip_prefix else selected
        if spec.strip_prefix:
            assert text.startswith(spec.strip_prefix), spec
            text = text[len(spec.strip_prefix) :]
        if spec.strip_suffix:
            assert text.endswith(spec.strip_suffix), spec
            text = text[: -len(spec.strip_suffix)]
        return text
    source = textwrap.dedent(selected).strip()
    if spec.strip_prefix:
        assert source.startswith(spec.strip_prefix), spec
        source = source[len(spec.strip_prefix) :]
    if spec.strip_suffix:
        assert source.endswith(spec.strip_suffix), spec
        source = source[: -len(spec.strip_suffix)]
    if spec.kind == "python":
        expression = ast.parse(f"(\n{source}\n)", mode="eval").body
        return _render(expression, spec.names)
    if spec.kind == "python_lines":
        listing = ast.parse(f"[\n{source}\n]", mode="eval").body
        assert isinstance(listing, ast.List), spec
        return "\n".join(_render(element, spec.names) for element in listing.elts)
    raise AssertionError(f"unknown kind {spec.kind}")


def extracted_content(spec: Spec) -> str:
    """Return the full content of the extracted file of ``spec`` (header line and value)."""
    header = (
        f"{HEADER_PREFIX}{spec.upstream} lines {spec.first}-{spec.last} of AI-Scientist-v3 "
        f"{_commit()[:12]} by tests/paper/test_loop_provenance.py; deriving a package file "
        "drops this line. -->"
    )
    return f"{header}\n{extract(spec)}"


@pytest.mark.parametrize("spec", SPECS, ids=[spec.output for spec in SPECS])
def test_extracted_text_equals_upstream(spec: Spec) -> None:
    path = EXTRACTED / spec.output
    assert path.is_file(), f"missing {path}"
    assert path.read_text(encoding="utf-8") == extracted_content(spec)


def test_every_extracted_file_is_specified() -> None:
    import tomllib

    changes = REPO_ROOT / "src" / "langgraph_agentic_scientist" / "prompts" / "changes"
    data = tomllib.loads((changes / "paper_loop.toml").read_text(encoding="utf-8"))
    by_tool = {Path(item["output"]).name for item in data.get("extract", [])}
    declared = {spec.output for spec in SPECS}
    assert not declared & by_tool
    present = {path.name for path in EXTRACTED.glob("*.md")}
    assert present == declared | by_tool


def test_package_prompts_derive_from_upstream() -> None:
    import importlib.util
    import sys

    name = "_paper_loop_derive_prompts"
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / "derive_prompts.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _, files = module.load_changes()
    mine = [entry for entry in files if entry.source == "paper_loop.toml"]
    assert mine, "paper_loop.toml declares no files"
    for entry in mine:
        target = REPO_ROOT / "src" / "langgraph_agentic_scientist" / entry.target
        assert target.read_bytes() == module.derive(entry), entry.target
