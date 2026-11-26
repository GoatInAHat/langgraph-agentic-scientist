"""The gates of the paper validation, in upstream's order, on fixture manuscripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_validation_support import (
    ALIFE_AUTHORS,
    BIB,
    BODY_PAGE,
    DEFAULT_PAGES,
    REFS_PAGE,
    SUPPLEMENT_PAGE,
    FakeLatexWorkspace,
    paper_files,
    valid_alife_tex,
    valid_tex,
    write_files,
)

from langgraph_agentic_scientist.paper.validation import WorkspacePaperToolchain
from langgraph_agentic_scientist.paper.watermark import (
    WATERMARK_BLOCK_START,
    has_managed_watermark,
    strip_managed_block,
)
from langgraph_agentic_scientist.venues import VenueCatalog

PAPER = "paper"


def alife_without_budget(directory: Path) -> VenueCatalog:
    """The shipped ALife venue with its page budget removed (the fixtures are short)."""
    source = VenueCatalog.default().get("alife")
    (directory / "authors").mkdir(parents=True)
    (directory / "alife.md").write_text(source.brief_text())
    (directory / "authors" / "alife.tex").write_text(source.author_block_text() or "")
    (directory / "alife.yaml").write_text(
        "name: alife\nbrief: alife.md\nauthor_block: authors/alife.tex\n"
        "template_dir: templates/alife\n"
        "latex_markers: ['\\\\usepackage\\{[^}]*alifeconf[^}]*\\}']\n"
        "page_budget: null\nchecks: [alife_framing]\n"
    )
    return VenueCatalog.from_directory(directory)


async def validate(
    tmp_path: Path,
    files: dict[str, Any],
    *,
    pages: list[str] | None = None,
    toolchain: WorkspacePaperToolchain | None = None,
    options: dict[str, Any] | None = None,
    **workspace_options: Any,
) -> tuple[Any, FakeLatexWorkspace]:
    workspace = FakeLatexWorkspace(
        tmp_path / "ws", pages=pages or DEFAULT_PAGES, **workspace_options
    )
    await write_files(workspace, PAPER, files)
    report = await (toolchain or WorkspacePaperToolchain()).validate(
        PAPER, workspace=workspace, options=options or {}
    )
    return report, workspace


async def test_a_valid_paper_passes_every_gate(tmp_path: Path) -> None:
    report, workspace = await validate(tmp_path, paper_files())
    assert report.passed, report.errors
    assert report.errors == [] and report.fingerprint is None
    assert report.page_count == 3
    assert report.pdf_path == workspace.host_path("paper/paper.pdf")
    # the commands upstream runs, in order, with the compile flags
    assert [command[0] for command in workspace.commands] == [
        "pdflatex",
        "bibtex",
        "pdflatex",
        "pdflatex",
    ]
    assert workspace.commands[0] == [
        "pdflatex",
        "-recorder",
        "-no-shell-escape",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "paper.tex",
    ]
    # the report file has upstream's keys and says ok
    on_disk = json.loads((await workspace.read_bytes("paper/validation_report.json")).decode())
    assert on_disk["ok"] is True and on_disk["error"] is None
    assert on_disk["watermark_validation"]["applied"] is True
    assert report.checks["bibtex_key_count"] == 1 and report.checks["citation_count"] == 1
    assert report.checks["claim_count"] == 1


async def test_the_watermark_is_inserted_before_the_manuscript_is_read(tmp_path: Path) -> None:
    _, workspace = await validate(tmp_path, paper_files())
    tex = (await workspace.read_bytes("paper/paper.tex")).decode()
    assert WATERMARK_BLOCK_START in tex and has_managed_watermark(tex)
    assert strip_managed_block(tex) == valid_tex()


async def test_the_watermark_can_be_switched_off(tmp_path: Path) -> None:
    toolchain = WorkspacePaperToolchain(watermark=False)
    report, workspace = await validate(tmp_path, paper_files(), toolchain=toolchain)
    assert report.passed
    assert (await workspace.read_bytes("paper/paper.tex")).decode() == valid_tex()
    assert report.checks["watermark_validation"]["checked"] is False
    # a call can override the switch
    report, _ = await validate(
        tmp_path / "again", paper_files(), toolchain=toolchain, options={"watermark": True}
    )
    assert report.checks["watermark_validation"]["applied"] is True


@pytest.mark.parametrize(
    ("files", "needle"),
    [
        ({"paper.tex": None}, "Missing required artifact: paper.tex"),
        ({"references.bib": None}, "Missing required artifact: references.bib"),
        (
            {"paper.tex": valid_tex().replace("\\section{Method}\nMethod.\n", "")},
            "paper.tex missing required sections: ['method']",
        ),
        (
            {"paper.tex": valid_tex().replace("\\clearpage\n\\appendix\n", "\\appendix\n")},
            "Supplementary Information validation failed",
        ),
        (
            {"paper.tex": valid_tex().replace("RefKey2024", "Nope")},
            "paper.tex cites unknown BibTeX keys: ['Nope']",
        ),
        ({"references.bib": "no entries"}, "references.bib did not contain any BibTeX keys"),
        (
            {
                "paper.tex": valid_tex().replace(
                    "\\bibliography{references}", "\\bibliography{other}"
                )
            },
            "paper.tex must use the canonical references.bib bibliography",
        ),
        (
            {
                "paper.tex": valid_tex().replace(
                    "\\bibliography{references}", "\\bibliography{\\jobname}"
                )
            },
            "paper.tex uses a dynamic bibliography target",
        ),
        ({"claims_grounding.json": "{}"}, "claims_grounding.json validation failed"),
        (
            {"references_provenance.json": json.dumps({"references": []})},
            "references provenance validation failed",
        ),
        ({"paper_data.json": json.dumps({"x": 1})}, "paper_data.json missing object field"),
        ({"paper_data.json": None}, "Missing required artifact: paper_data.json"),
    ],
)
async def test_each_source_gate_fails_with_its_message(
    tmp_path: Path, files: dict[str, str | None], needle: str
) -> None:
    base = paper_files()
    for name, content in files.items():
        if content is None:
            del base[name]
        else:
            base[name] = content
    report, workspace = await validate(tmp_path, base)
    assert not report.passed
    assert needle in report.errors[0], report.errors
    assert report.fingerprint is not None and len(report.fingerprint) == 64
    # a failing gate stops the run: nothing was compiled
    assert workspace.commands == []
    on_disk = json.loads((await workspace.read_bytes("paper/validation_report.json")).decode())
    assert on_disk["ok"] is False and needle in on_disk["error"]


async def test_the_claims_must_cite_known_sources_and_keys(tmp_path: Path) -> None:
    claims = [{"claim": "x", "source_ids": ["result.nope"], "citation_keys": ["Missing"]}]
    report, _ = await validate(tmp_path, paper_files(claims=claims))
    assert "contains unknown id: 'result.nope'" in report.errors[0]


async def test_reference_ids_of_the_provenance_are_valid_grounding_sources(tmp_path: Path) -> None:
    claims = [{"claim": "x", "source_ids": ["ref.s2.abc"], "citation_keys": ["RefKey2024"]}]
    report, _ = await validate(tmp_path, paper_files(claims=claims))
    assert report.passed, report.errors


async def test_an_evidence_token_without_a_bundle_fails(tmp_path: Path) -> None:
    files = paper_files(extra={"notes.md": "see @evidence[[results/x.json#/a]]\n"})
    report, _ = await validate(tmp_path, files)
    assert "scientific-evidence/ bundle missing" in report.errors[0]
    assert "notes.md:1" in report.errors[0]


async def test_staged_skills_and_aliases_are_not_scanned_for_evidence_tokens(
    tmp_path: Path,
) -> None:
    files = paper_files(
        extra={
            ".agents/skills/scientific-evidence/SKILL.md": "example @evidence[[a/b.json#/x]]\n",
            "results/notes.md": "copied @evidence[[a/b.json#/x]]\n",
        }
    )
    report, _ = await validate(tmp_path, files)
    assert report.passed, report.errors


async def test_standalone_language_hard_error_and_soft_warning(tmp_path: Path) -> None:
    tex = valid_tex().replace("Method.", "The combined_score of eligible implementations.")
    contract = {"forbidden_visible_terms": ["eligible", "combined_score"]}
    report, _ = await validate(
        tmp_path, paper_files(tex=tex, profile="standalone", contract=contract)
    )
    assert "standalone paper language validation failed" in report.errors[0]
    assert "'eligible'" in report.errors[0] or "combined_score" in report.errors[0]
    soft = valid_tex().replace("Method.", "Some implementations of the idea.")
    report, _ = await validate(tmp_path / "soft", paper_files(tex=soft, profile="standalone"))
    assert report.passed
    assert any("implementations" in message for message in report.warnings)


async def test_the_default_profile_does_not_check_language(tmp_path: Path) -> None:
    tex = valid_tex().replace("Method.", "The combined_score of eligible implementations.")
    report, _ = await validate(tmp_path, paper_files(tex=tex))
    assert report.passed
    assert report.checks["standalone_language_validation"]["checked"] is False


async def test_the_alife_author_block_must_be_the_canonical_one(tmp_path: Path) -> None:
    canonical = VenueCatalog.default().get("alife").author_block_text()
    assert canonical
    toolchain = WorkspacePaperToolchain(venues=alife_without_budget(tmp_path / "venues"))
    files = paper_files(tex=valid_alife_tex(), extra={"paper_authors.tex": ALIFE_AUTHORS})
    report, _ = await validate(tmp_path, files, toolchain=toolchain)
    assert report.passed, report.errors
    assert report.checks["venue_validation"]["paper_format"] == "alife"
    other = paper_files(tex=valid_alife_tex(), extra={"paper_authors.tex": "Someone Else"})
    report, _ = await validate(tmp_path / "other", other, toolchain=toolchain)
    assert "ALIFE author block does not match canonical `./paper_authors.tex`" in report.errors[0]


async def test_alife_placeholders_and_missing_author_blocks_fail(tmp_path: Path) -> None:
    tex = valid_alife_tex(author="Anonymous Authors")
    toolchain = WorkspacePaperToolchain(venues=alife_without_budget(tmp_path / "venues"))
    report, _ = await validate(tmp_path, paper_files(tex=tex), toolchain=toolchain)
    assert "forbidden anonymous/generic placeholder 'Anonymous Authors'" in report.errors[0]


async def test_alife_framing_gaps_are_soft_warnings(tmp_path: Path) -> None:
    tex = valid_alife_tex().replace(
        "Artificial life summary about adaptive computation and biologically plausible learning.",
        "A summary.",
    )
    toolchain = WorkspacePaperToolchain(venues=alife_without_budget(tmp_path / "venues"))
    report, _ = await validate(
        tmp_path,
        paper_files(tex=tex, extra={"paper_authors.tex": ALIFE_AUTHORS}),
        toolchain=toolchain,
    )
    assert report.passed
    assert any(
        "abstract lacks explicit ALIFE-facing framing language" in m for m in report.warnings
    )


async def test_a_caller_venue_is_used_for_the_markers_and_the_budget(tmp_path: Path) -> None:
    custom = tmp_path / "venues"
    custom.mkdir()
    (custom / "tiny.yaml").write_text(
        "name: tiny\nbrief: tiny.md\nauthor_block: null\ntemplate_dir: templates/tiny\n"
        "latex_markers: ['\\\\usepackage\\{tinyconf\\}']\n"
        "page_budget: {max_body_pages: 1, exclude_references: true}\nchecks: []\n"
    )
    (custom / "tiny.md").write_text("A tiny venue brief.\n")
    catalog = VenueCatalog.default().with_venues(VenueCatalog.from_directory(custom))
    tex = valid_tex(preamble="\\usepackage{tinyconf}\n")
    pages = ["page one", "page two", REFS_PAGE, SUPPLEMENT_PAGE]
    report, _ = await validate(
        tmp_path / "run",
        paper_files(tex=tex),
        pages=pages,
        toolchain=WorkspacePaperToolchain(venues=catalog),
    )
    assert (
        "tiny page limit exceeded: body uses 2 pages (max 1 excluding references)"
        in report.errors[0]
    )


async def test_the_iclr_page_budget_counts_body_pages_before_the_references(tmp_path: Path) -> None:
    tex = valid_tex(preamble="\\iclrfinalcopy\n")
    short = [BODY_PAGE, REFS_PAGE, SUPPLEMENT_PAGE]
    report, _ = await validate(tmp_path, paper_files(tex=tex), pages=short)
    assert (
        "iclr page minimum not met: body uses 1 pages (min 8 excluding references"
        in (report.errors[0])
    )
    assert report.checks["page_validation"]["reference_start_page"] == 2
    assert report.checks["page_validation"]["supplement_start_page"] == 3
    long = [BODY_PAGE] * 10 + [REFS_PAGE, SUPPLEMENT_PAGE]
    report, _ = await validate(tmp_path / "long", paper_files(tex=tex), pages=long)
    assert (
        "iclr page limit exceeded: body uses 10 pages (max 9 excluding references)"
        in (report.errors[0])
    )


async def test_a_compile_failure_is_a_failed_report(tmp_path: Path) -> None:
    report, workspace = await validate(tmp_path, paper_files(), pdflatex_returncode=1)
    assert "LaTeX command failed: pdflatex" in report.errors[0]
    assert len(workspace.commands) == 1


async def test_a_missing_compiler_is_an_infrastructure_failure(
    tmp_path: Path, workspace: Any
) -> None:
    toolchain = WorkspacePaperToolchain(pdflatex="pdflatex-that-does-not-exist")
    await write_files(workspace, PAPER, paper_files())
    report = await toolchain.validate(PAPER, workspace=workspace, options={})
    assert not report.passed
    assert "Missing required command(s): pdflatex-that-does-not-exist" in report.errors[0]
    assert report.checks["infrastructure_error"] is True


async def test_compile_inputs_from_staged_skills_are_refused(tmp_path: Path) -> None:
    recorded = ("paper.tex", ".agents/skills/write-paper/templates/iclr/paper.sty")
    report, _ = await validate(tmp_path, paper_files(), recorded_inputs=recorded)
    assert "compiled paper must not read staged agent skill files" in report.errors[0]
    outside = ("paper.tex", "/etc/passwd")
    report, _ = await validate(tmp_path / "outside", paper_files(), recorded_inputs=outside)
    assert "outside approved TeX roots" in report.errors[0]
    system = ("paper.tex", "/usr/share/texlive/texmf-dist/tex/latex/base/article.cls")
    report, _ = await validate(tmp_path / "system", paper_files(), recorded_inputs=system)
    assert report.passed, report.errors


async def test_bibtex_must_read_the_canonical_bibliography(tmp_path: Path) -> None:
    report, _ = await validate(
        tmp_path, paper_files(), aux="\\bibstyle{plain}\n\\bibdata{elsewhere}\n"
    )
    assert "canonical references.bib" in report.errors[0]
    report, _ = await validate(tmp_path / "two", paper_files(), aux="\\bibstyle{plain}\n")
    assert "paper.aux must declare exactly one bibliography database and style" in report.errors[0]


async def test_the_rendered_supplement_must_follow_the_references(tmp_path: Path) -> None:
    report, _ = await validate(tmp_path, paper_files(), pages=[BODY_PAGE, REFS_PAGE])
    assert "missing the Supplementary Information heading" in report.errors[0]
    same_page = [BODY_PAGE, REFS_PAGE + "\nA Supplementary Information\n" + "text " * 40]
    report, _ = await validate(tmp_path / "same", paper_files(), pages=same_page)
    assert "must start on a separate page after References" in report.errors[0]


async def test_the_failure_fingerprint_follows_errors_and_inputs(tmp_path: Path) -> None:
    bad = paper_files(tex=valid_tex().replace("RefKey2024", "Nope"))
    first, workspace = await validate(tmp_path, bad)
    again = await WorkspacePaperToolchain().validate(PAPER, workspace=workspace, options={})
    assert first.fingerprint == again.fingerprint and first.fingerprint
    await workspace.write_bytes(
        "paper/paper.tex",
        valid_tex().replace("RefKey2024", "Nope").replace("Done.", "Done!").encode(),
    )
    changed = await WorkspacePaperToolchain().validate(PAPER, workspace=workspace, options={})
    assert changed.errors == first.errors and changed.fingerprint != first.fingerprint
    # the managed watermark block is not manuscript content
    assert BIB  # keep the import used


async def test_pdf_text_extracts_the_pages(tmp_path: Path) -> None:
    report, workspace = await validate(tmp_path, paper_files())
    assert report.passed
    text = await WorkspacePaperToolchain().pdf_text("paper/paper.pdf", workspace=workspace)
    assert "Supplementary Information" in text and "References" in text


async def test_compile_alone_returns_the_pdf(tmp_path: Path) -> None:
    workspace = FakeLatexWorkspace(tmp_path / "ws")
    await write_files(workspace, PAPER, paper_files())
    result = await WorkspacePaperToolchain().compile(PAPER, workspace=workspace)
    assert result.success and result.pdf_path == "paper/paper.pdf" and result.page_count == 3
    failing = FakeLatexWorkspace(tmp_path / "bad", pdflatex_returncode=1)
    await write_files(failing, PAPER, paper_files())
    result = await WorkspacePaperToolchain().compile(PAPER, workspace=failing)
    assert not result.success and "LaTeX command failed" in result.log
