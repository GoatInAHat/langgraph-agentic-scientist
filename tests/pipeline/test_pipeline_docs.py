"""The documentation matches the repository: links resolve, generated files are current."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "src" / "langgraph_agentic_scientist"
DOCS = REPO_ROOT / "docs"
LINK = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
# The notes docs/adaptations/<area>.md, one per area of the package, that ADAPTATIONS.md links.
AREA_NOTES = (
    "core",
    "drafting_design",
    "drafting_scaffold",
    "investigation",
    "variants",
    "final_and_posthoc",
    "paper_host",
    "paper_loop",
    "review_revision",
)


def _load_tool(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _links(path: Path) -> list[str]:
    return [
        target
        for target in LINK.findall(path.read_text())
        if not target.startswith(("http://", "https://", "mailto:"))
    ]


def test_adaptations_links_every_area_note_and_every_declared_change() -> None:
    text = (DOCS / "ADAPTATIONS.md").read_text()
    for note in sorted((DOCS / "adaptations").glob("*.md")):
        assert f"adaptations/{note.name}" in text, f"{note.name} is not linked"
    for changes in sorted((PACKAGE / "prompts" / "changes").glob("*.toml")):
        assert f"prompts/changes/{changes.name}" in text, f"{changes.name} is not linked"


@pytest.mark.parametrize("name", ["README.md", "docs/ADAPTATIONS.md", "docs/ARCHITECTURE.md"])
def test_relative_links_resolve(name: str) -> None:
    path = REPO_ROOT / name
    for target in _links(path):
        resolved = (path.parent / target).resolve()
        assert resolved.exists(), f"{name} links {target}, which does not exist"


def test_every_area_has_its_note_and_adaptations_links_it() -> None:
    text = (DOCS / "ADAPTATIONS.md").read_text()
    for note in AREA_NOTES:
        assert (DOCS / "adaptations" / f"{note}.md").is_file(), note
        assert f"adaptations/{note}.md" in text, note


def test_adaptations_covers_every_numbered_adaptation_and_omission() -> None:
    text = (DOCS / "ADAPTATIONS.md").read_text()
    adaptations = re.findall(r"^(\d+)\. \*\*", text.split("## Omissions")[0], flags=re.M)
    assert [int(n) for n in adaptations] == list(range(1, 31))
    omissions = re.findall(r"^(\d+)\. ", text.split("## Omissions")[1], flags=re.M)
    assert [int(n) for n in omissions] == list(range(1, 16))


def test_generated_files_are_current() -> None:
    tool = _load_tool("list_steps")
    warnings: list[str] = []
    built = [(entry, tool._build(entry, warnings)) for entry in tool.GRAPHS]
    assert not warnings, warnings
    assert (DOCS / "STEPS.md").read_text() == tool.render_steps(built)
    architecture = (DOCS / "ARCHITECTURE.md").read_text()
    assert architecture == tool.splice(architecture, tool.render_diagrams(built))


def test_the_generated_listing_names_the_pipelines_steps_and_hook_writable_keys() -> None:
    from langgraph_agentic_scientist import GraphOverrides, build_pipeline_graph

    tool = _load_tool("list_steps")
    graph = build_pipeline_graph(overrides=GraphOverrides(phases=_stubs()))
    keys = tool.hook_keys(graph)
    assert {"pipeline.snapshot_task", "pipeline.paper", "pipeline.finish"} <= set(keys)
    assert keys["pipeline.paper"] == ()
    assert "pipeline.route_paper" not in keys  # routers carry no hooks


def _stubs() -> dict[str, object]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pipeline_stubs import Ledger, default_results, stub_phase

    ledger = Ledger()
    return {name: stub_phase(name, ledger, value) for name, value in default_results().items()}


def test_every_documented_slot_is_read_by_the_package() -> None:
    tool = _load_tool("list_steps")
    source = "\n".join(path.read_text() for path in PACKAGE.rglob("*.py"))
    for key in tool.SLOTS:
        assert f'"{key}"' in source, f"no factory reads the slot {key}"


def test_notice_credits_the_original_and_says_the_prompts_are_adapted() -> None:
    notice = (REPO_ROOT / "NOTICE").read_text()
    assert "AI-Scientist-v3" in notice and "df8313889a004f8a44a472e19a833cd547bb828a" in notice
    assert "Apache License" in notice and "adapted" in notice
    assert (
        (REPO_ROOT / "LICENSE")
        .read_text()
        .startswith(("\n", " ", "Apache", "                                 Apache"))
    )


def test_the_repository_names_no_local_paths() -> None:
    check = _load_tool("check_public")
    assert check.scan(check.candidate_files()) == []
