"""The figure gate: manifest schema, strategy specs, readability budgets and figure files."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest
from test_validation_support import FakeLatexWorkspace, paper_files, valid_tex, write_files

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.paper.validation import (
    DirectoryTree,
    SnapshotTree,
    WorkspacePaperToolchain,
    classify_figure_validation_errors,
    has_actionable_figure_quality_warnings,
    validate_figure_manifest,
)
from langgraph_agentic_scientist.paper.validation.figure_manifest import (
    build_figure_manifest_json_schema,
    normalize_figure_manifest,
)
from langgraph_agentic_scientist.paper.validation.figures import (
    read_concept_figure_prompt_log,
    read_png_size,
)
from langgraph_agentic_scientist.paper.validation.graphics import resolve_graphics_references
from langgraph_agentic_scientist.prompts import PromptRegistry

SOURCES = {"result.best_correct_score"}


def png(width: int, height: int) -> bytes:
    """A PNG header with the given size (the gate reads only the header)."""
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I4sII", 13, b"IHDR", width, height) + b"\x00" * 16


def pdf() -> bytes:
    return b"%PDF-1.4\n%%EOF\n"


def row(**updates: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "figure_id": "fig1",
        "path_pdf": "figures/fig1.pdf",
        "path_png": "figures/fig1.png",
        "figure_type": "bar",
        "n_panels": 1,
        "panels_per_row": 1,
        "categories_per_panel": [3],
        "series_per_panel": [2],
        "max_label_chars": 10,
        "font_sizes": {"title": 10, "axis": 9, "ticks": 8, "legend": 8},
        "uses_abbrev_labels": True,
        "uses_progression": False,
        "source_ids": ["result.best_correct_score"],
        "placement": "column",
    }
    base.update(updates)
    return base


async def figure_tree(
    workspace: LocalWorkspace,
    *,
    manifest: Any,
    strategy: Any = None,
    files: dict[str, bytes] | None = None,
) -> SnapshotTree:
    await workspace.write_bytes("p/figure_manifest.json", json.dumps(manifest).encode())
    if strategy is not None:
        await workspace.write_bytes("p/paper_strategy.json", json.dumps(strategy).encode())
    for name, data in (
        files or {"figures/fig1.pdf": pdf(), "figures/fig1.png": png(1300, 800)}
    ).items():
        await workspace.write_bytes(f"p/{name}", data)
    return await SnapshotTree.load(workspace, "p", include=lambda rel, entry: 1 << 20)


TEX = valid_tex(include_graphics="figures/fig1")


async def test_a_valid_manifest_passes(workspace: LocalWorkspace) -> None:
    tree = await figure_tree(workspace, manifest={"figures": [row()]})
    report = validate_figure_manifest(tree, tex_text=TEX, allowed_source_ids=SOURCES)
    assert report["checked"] and report["required"]
    assert report["errors"] == [] and report["figure_count"] == 1
    assert report["referenced_paths"] == ["figures/fig1.pdf", "figures/fig1.png"]


async def test_a_paper_without_figures_does_not_need_a_manifest(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("p/paper.tex", b"x")
    tree = await SnapshotTree.load(workspace, "p", include=lambda rel, entry: 1000)
    report = validate_figure_manifest(tree, tex_text=valid_tex(), allowed_source_ids=SOURCES)
    assert report["checked"] is False and report["required"] is False


async def test_a_referenced_figure_must_be_in_the_manifest(workspace: LocalWorkspace) -> None:
    tree = await figure_tree(
        workspace,
        manifest={"figures": [row()]},
        files={
            "figures/fig1.pdf": pdf(),
            "figures/fig1.png": png(1300, 800),
            "figures/other.pdf": pdf(),
        },
    )
    tex = valid_tex(include_graphics="figures/other")
    report = validate_figure_manifest(tree, tex_text=tex, allowed_source_ids=SOURCES)
    assert (
        "paper.tex references figure missing from figure_manifest.json: figures/other"
        in report["errors"]
    )
    assert "figure_manifest entry not referenced by paper.tex: figures/fig1.png" in report["errors"]


@pytest.mark.parametrize(
    ("updates", "error", "hard"),
    [
        ({"source_ids": ["result.nope"]}, "figures[0] unknown source id: 'result.nope'", True),
        ({"source_ids": []}, "figures[0].source_ids must be non-empty list", True),
        ({"figure_type": "bubble"}, "failed schema validation", True),
        (
            {
                "n_panels": 5,
                "panels_per_row": 1,
                "categories_per_panel": [1] * 5,
                "series_per_panel": [1] * 5,
            },
            "figures[0] exceeds max total panels: 5 > 4",
            False,
        ),
        (
            {"categories_per_panel": [20]},
            "figures[0] exceeds category budget in panel 0: 20 > 8",
            False,
        ),
        (
            {"max_label_chars": 30, "uses_abbrev_labels": False},
            "figures[0] labels too long without abbreviation",
            False,
        ),
        (
            {"font_sizes": {"title": 10, "axis": 9, "ticks": 5, "legend": 8}},
            "figures[0].font_sizes.ticks below minimum: 5.0 < 8.0",
            False,
        ),
    ],
)
async def test_readability_budgets_are_soft_and_structure_is_hard(
    workspace: LocalWorkspace, updates: dict[str, Any], error: str, hard: bool
) -> None:
    tree = await figure_tree(workspace, manifest={"figures": [row(**updates)]})
    report = validate_figure_manifest(tree, tex_text=TEX, allowed_source_ids=SOURCES)
    assert any(error in message for message in report["errors"]), report["errors"]
    assert bool(any(error in message for message in report["hard_errors"])) is hard
    assert bool(any(error in message for message in report["soft_warnings"])) is not hard


async def test_png_size_and_raster_references_are_soft_quality_warnings(
    workspace: LocalWorkspace,
) -> None:
    tree = await figure_tree(
        workspace,
        manifest={"figures": [row()]},
        files={"figures/fig1.pdf": pdf(), "figures/fig1.png": png(800, 500)},
    )
    tex = valid_tex(include_graphics="figures/fig1.png")
    report = validate_figure_manifest(tree, tex_text=tex, allowed_source_ids=SOURCES)
    assert report["hard_errors"] == []
    assert "figures[0] PNG width too small: 800 < 1200" in report["soft_warnings"]
    assert "figures[0] PNG height too small: 500 < 700" in report["soft_warnings"]
    assert (
        "figures[0] paper.tex references raster PNG despite available vector PDF"
        in report["soft_warnings"]
    )
    assert has_actionable_figure_quality_warnings(report["soft_warnings"])
    assert not has_actionable_figure_quality_warnings(
        ["figures[0] exceeds max total panels: 5 > 4"]
    )


async def test_missing_and_symlinked_figure_files_are_refused(workspace: LocalWorkspace) -> None:
    tree = await figure_tree(
        workspace, manifest={"figures": [row()]}, files={"figures/fig1.png": png(1300, 800)}
    )
    report = validate_figure_manifest(tree, tex_text=TEX, allowed_source_ids=SOURCES)
    assert "figures[0] missing PDF artifact: figures/fig1.pdf" in report["hard_errors"]
    escaping = row(path_pdf="figures/../fig1.pdf")
    tree = await figure_tree(workspace, manifest={"figures": [escaping]})
    report = validate_figure_manifest(tree, tex_text=TEX, allowed_source_ids=SOURCES)
    assert any("missing PDF artifact" in message for message in report["hard_errors"])


async def test_the_strategy_binds_generators_budgets_and_required_figures(
    workspace: LocalWorkspace,
) -> None:
    strategy = {
        "figure_specs": [
            {
                "figure_id": "fig1",
                "figure_type": "line",
                "max_panels": 1,
                "allow_progression": False,
            },
            {"figure_id": "need", "figure_type": "bar", "required": True},
        ]
    }
    tree = await figure_tree(workspace, manifest={"figures": [row()]}, strategy=strategy)
    report = validate_figure_manifest(tree, tex_text=TEX, allowed_source_ids=SOURCES)
    assert "figures[0] figure_type does not match strategy for fig1" in report["hard_errors"]
    assert "Missing required figure from strategy: need" in report["hard_errors"]


async def test_a_concept_figure_needs_its_prompt_log(workspace: LocalWorkspace) -> None:
    strategy = {
        "figure_specs": [
            {
                "figure_id": "concept",
                "figure_type": "diagram",
                "preferred_generator": "generate-concept-figure",
            }
        ]
    }
    concept = row(
        figure_id="concept",
        figure_type="diagram",
        path_pdf="",
        path_png="figures/concept.png",
        source_ids=[],
        generator="generate-concept-figure",
        prompt_log_path="figures/concept.prompts.json",
        categories_per_panel=[1],
        series_per_panel=[1],
    )
    files = {"figures/concept.png": png(2400, 1000)}
    tex = valid_tex(include_graphics="figures/concept.png")
    tree = await figure_tree(
        workspace, manifest={"figures": [concept]}, strategy=strategy, files=files
    )
    report = validate_figure_manifest(tree, tex_text=tex, allowed_source_ids=SOURCES)
    assert any("invalid prompt log" in message for message in report["hard_errors"])
    files["figures/concept.prompts.json"] = json.dumps({"status": "ok", "panels": []}).encode()
    tree = await figure_tree(
        workspace, manifest={"figures": [concept]}, strategy=strategy, files=files
    )
    report = validate_figure_manifest(tree, tex_text=tex, allowed_source_ids=SOURCES)
    assert [m for m in report["hard_errors"] if "prompt log" in m] == []
    with pytest.raises(RuntimeError, match="must use prompt_log_path"):
        read_concept_figure_prompt_log(tree, "concept", "figures/other.json")
    with pytest.raises(RuntimeError, match="Invalid concept figure_id"):
        read_concept_figure_prompt_log(tree, "../x", "figures/x.prompts.json")


async def test_a_directory_path_works_like_a_tree(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "fig1.pdf").write_bytes(pdf())
    (tmp_path / "figures" / "fig1.png").write_bytes(png(1300, 800))
    (tmp_path / "figure_manifest.json").write_text(json.dumps({"figures": [row()]}))
    by_path = validate_figure_manifest(tmp_path, tex_text=TEX, allowed_source_ids=SOURCES)
    by_tree = validate_figure_manifest(
        DirectoryTree(tmp_path), tex_text=TEX, allowed_source_ids=SOURCES
    )
    assert by_path == by_tree and by_path["errors"] == []
    assert by_path["manifest_path"] == str((tmp_path / "figure_manifest.json").resolve())
    assert read_png_size(tmp_path / "figures" / "fig1.png") == (1300, 800)


def test_graphics_resolution_follows_pdflatex_extension_order(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "a.pdf").write_bytes(pdf())
    (tmp_path / "figures" / "a.png").write_bytes(png(10, 10))
    (tmp_path / "figures" / "b.png").write_bytes(png(10, 10))
    tex = "\\includegraphics{figures/a}\\includegraphics{figures/b}\\includegraphics{figures/a.png}"
    resolved = resolve_graphics_references(tex, tmp_path)
    assert [(r.raw_path, r.resolved_path, r.explicitly_png) for r in resolved] == [
        ("figures/a", "figures/a.pdf", False),
        ("figures/b", "figures/b.png", False),
        ("figures/a.png", "figures/a.png", True),
    ]
    with pytest.raises(ValueError, match="Unsafe figure reference"):
        resolve_graphics_references("\\includegraphics{../x.png}", tmp_path)
    (tmp_path / "figures" / "c.png").symlink_to(tmp_path / "figures" / "a.png")
    with pytest.raises(ValueError, match="Unsafe figure asset"):
        resolve_graphics_references("\\includegraphics{figures/c}", tmp_path)


def test_hard_and_soft_classification() -> None:
    hard, soft = classify_figure_validation_errors(
        ["figures[0] missing PNG artifact: x", "figures[1] PNG width too small: 1 < 2"]
    )
    assert hard == ["figures[0] missing PNG artifact: x"]
    assert soft == ["figures[1] PNG width too small: 1 < 2"]


def test_the_generated_schema_is_the_schema_agents_read() -> None:
    shipped = PromptRegistry.default().skill_files("validate-paper")[
        "templates/figure_manifest.schema.json"
    ]
    generated = json.dumps(build_figure_manifest_json_schema(), indent=2, sort_keys=True) + "\n"
    assert json.loads(shipped) == json.loads(generated)
    rows = normalize_figure_manifest({"figures": [row()]})
    assert rows[0].figure_id == "fig1" and rows[0].font_sizes == {
        "title": 10.0,
        "axis": 9.0,
        "ticks": 8.0,
        "legend": 8.0,
    }


async def test_the_gate_reports_figure_errors_through_the_toolchain(tmp_path: Path) -> None:
    workspace = FakeLatexWorkspace(tmp_path / "ws")
    files = paper_files(
        tex=TEX,
        extra={
            "figure_manifest.json": json.dumps({"figures": [row(source_ids=["nope"])]}),
            "figures/fig1.pdf": pdf(),
            "figures/fig1.png": png(1300, 800),
        },
    )
    await write_files(workspace, "paper", files)
    report = await WorkspacePaperToolchain().validate("paper", workspace=workspace, options={})
    assert not report.passed and "figure validation failed" in report.errors[0]
    assert report.checks["feedback"]["figure_validation_errors"]
    assert workspace.commands == []
    soft = row(figure_id="fig1", categories_per_panel=[20])
    files["figure_manifest.json"] = json.dumps({"figures": [soft]})
    await write_files(workspace, "paper", files)
    report = await WorkspacePaperToolchain().validate("paper", workspace=workspace, options={})
    assert report.passed and any("category budget" in m for m in report.figure_warnings)
