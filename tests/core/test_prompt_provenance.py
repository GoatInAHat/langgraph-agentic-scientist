"""Every package prompt and skill is its declared derivation from the vendored upstream files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_tool(name: str) -> ModuleType:
    path = REPO_ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_tool_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def derive() -> ModuleType:
    return _load_tool("derive_prompts")


def test_every_package_file_equals_its_derivation(derive: ModuleType) -> None:
    assert derive.run(check=True) == []


def test_vendored_files_match_the_manifest() -> None:
    vendor = _load_tool("vendor_upstream")
    _, files = vendor.load_manifest()
    assert files, "the manifest lists no files"
    assert vendor.check_files(files) == 0
    assert vendor.pinned_commit() == "df8313889a004f8a44a472e19a833cd547bb828a"


def test_every_change_names_a_known_kind_and_a_reason(derive: ModuleType) -> None:
    _, files = derive.load_changes()
    assert files
    for entry in files:
        assert entry.reason.strip()
        for change in entry.changes:
            assert change.kind in derive.KINDS
            assert change.reason.strip()
            assert change.old != change.new


def test_an_old_string_that_does_not_occur_is_refused(derive: ModuleType) -> None:
    entry = derive.FileEntry(
        upstream="src/agents/skills/verify-rubrics/SKILL.md",
        target="skills/verify-rubrics/SKILL.md",
        reason="test",
        source="test.toml",
        changes=[derive.Change(kind="other", old="no such text", new="x", count=1, reason="t")],
    )
    with pytest.raises(derive.DerivationError, match="occurs 0 time"):
        derive.derive(entry)


def test_the_core_skills_differ_from_upstream_only_by_declared_changes(derive: ModuleType) -> None:
    _, files = derive.load_changes()
    by_target = {entry.target: entry for entry in files}
    memory = by_target["skills/memory-log/SKILL.md"]
    assert {change.kind for change in memory.changes} == {"A4"}
    assert not by_target["skills/verify-rubrics/SKILL.md"].changes
    for script in (
        "skills/memory-log/scripts/append_memory.py",
        "skills/verify-rubrics/scripts/create_rubric_status_placeholder.py",
    ):
        entry = by_target[script]
        assert not entry.changes
        upstream = (REPO_ROOT / "third_party/ai-scientist-v3" / entry.upstream).read_bytes()
        package = (REPO_ROOT / "src/langgraph_agentic_scientist" / script).read_bytes()
        assert upstream == package
    for name in ("evaluator_evidence", "candidate_environment", "baseline_failures"):
        entry = by_target[f"prompts/fragments/{name}.md"]
        assert not entry.changes


def test_the_extraction_of_the_reset_diagnostic(derive: ModuleType) -> None:
    extractions, _ = derive.load_changes()
    (reset,) = [
        item for item in extractions if item.output.endswith("reset_sequence_diagnostic.md")
    ]
    text = derive.extract(reset)
    header, _, body = text.partition("\n")
    assert header.startswith(derive.EXTRACT_HEADER_PREFIX)
    assert body.startswith("{{CONTEXT}} expected primary {{EXPECTED}}; observed {{OBSERVED}}")
    assert "(count={{COUNT}})" in body and "seed {{REPLAY_SEED}} is allowed." in body
