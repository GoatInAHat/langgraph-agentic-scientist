"""Edit mode on a paper that is not the package's output: a named main file, the venue file."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError
from test_loop_support import (
    audit_pass,
    build,
    edit_outputs,
    fast_config,
    models,
    no_gaps,
    paper_config,
    paper_data,
    prompts_of,
    reports,
    run,
    setup_task,
)

from langgraph_agentic_scientist.core.results import EditOptions, GroundingSource
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper.validation import WorkspacePaperToolchain
from langgraph_agentic_scientist.paper.watermark import add_watermark, has_managed_watermark
from langgraph_agentic_scientist.paper.workspace import (
    PaperPreparationError,
    seed_edit_workspace,
    stage_documents,
    stage_writer_skills,
)

EXT = "ext/paper"
MAIN = (
    "\\documentclass{mycls}\n\\usepackage{local}\n\\begin{document}\n"
    "\\input{sections/intro}\n\\include{sections/method}\n\\end{document}\n"
)


async def _external_paper(workspace: Any) -> None:
    await workspace.write_bytes(f"{EXT}/main.tex", MAIN.encode())
    await workspace.write_bytes(f"{EXT}/mycls.cls", b"% class\n")
    await workspace.write_bytes(f"{EXT}/local.sty", b"% package\n")
    await workspace.write_bytes(
        f"{EXT}/sections/intro.tex", b"\\section{Introduction}\n\\input{sections/sub/detail}\n"
    )
    await workspace.write_bytes(f"{EXT}/sections/method.tex", b"\\section{Method}\n")
    await workspace.write_bytes(f"{EXT}/sections/sub/detail.tex", b"Detail.\n")
    await workspace.write_bytes(f"{EXT}/sections/unused.tex", b"\\section{Unused}\n")
    await workspace.write_bytes(f"{EXT}/refs.bib", b"@article{a,\n title={A}\n}\n")


async def test_a_named_main_file_is_seeded_as_paper_tex_with_its_inputs(
    make_context: Any, workspace: Any
) -> None:
    await _external_paper(workspace)
    seeded = await seed_edit_workspace(
        make_context(),
        paper_dir="out",
        previous_paper_dir=EXT,
        options=EditOptions(main_file="main.tex"),
        reference_mode="refresh",
        paper_data=None,
    )
    assert (await workspace.read_bytes("out/paper.tex")).decode() == MAIN
    inputs = [
        "local.sty",
        "mycls.cls",
        "sections/intro.tex",
        "sections/method.tex",
        "sections/sub/detail.tex",
    ]
    for rel in inputs:
        assert await workspace.read_bytes(f"out/{rel}") == await workspace.read_bytes(
            f"{EXT}/{rel}"
        )
        assert await workspace.exists(f"out/previous_paper/{rel}"), rel
    assert await workspace.exists("out/previous_paper/main.tex")
    assert not await workspace.exists("out/main.tex")
    assert not await workspace.exists("out/sections/unused.tex")
    assert seeded["seeded"][: len(inputs) + 1] == ["paper.tex", *inputs]
    context = json.loads((await workspace.read_bytes("out/paper_edit_context.json")).decode())
    assert context["existing_paper_main_file"] == "main.tex"


async def test_without_a_main_file_the_seeding_is_upstreams(
    make_context: Any, workspace: Any
) -> None:
    await _external_paper(workspace)
    await workspace.write_bytes(f"{EXT}/paper.tex", MAIN.encode())
    await seed_edit_workspace(
        make_context(),
        paper_dir="out",
        previous_paper_dir=EXT,
        options=EditOptions(),
        reference_mode="refresh",
        paper_data=None,
    )
    assert await workspace.exists("out/paper.tex")
    assert not await workspace.exists("out/sections/intro.tex")
    context = json.loads((await workspace.read_bytes("out/paper_edit_context.json")).decode())
    assert context["existing_paper_main_file"] == "paper.tex"


async def test_a_missing_main_file_is_refused(make_context: Any, workspace: Any) -> None:
    await _external_paper(workspace)
    with pytest.raises(PaperPreparationError, match="Missing main file"):
        await seed_edit_workspace(
            make_context(),
            paper_dir="out",
            previous_paper_dir=EXT,
            options=EditOptions(main_file="article.tex"),
            reference_mode="refresh",
            paper_data=None,
        )


async def test_an_input_outside_the_paper_is_refused(make_context: Any, workspace: Any) -> None:
    await workspace.write_bytes(f"{EXT}/main.tex", b"\\input{../../secret}\n")
    with pytest.raises(PaperPreparationError, match="escapes the paper workspace"):
        await seed_edit_workspace(
            make_context(),
            paper_dir="out",
            previous_paper_dir=EXT,
            options=EditOptions(main_file="main.tex"),
            reference_mode="refresh",
            paper_data=None,
        )


@pytest.mark.parametrize("name", ["../main.tex", "/abs/main.tex", "main.txt", ""])
def test_the_main_file_is_a_relative_tex_path(name: str) -> None:
    with pytest.raises(ValidationError):
        EditOptions(main_file=name)


async def test_an_empty_venue_file_stages_no_brief(make_context: Any, workspace: Any) -> None:
    await workspace.write_bytes("out/VENUE.md", b"seeded from the edited paper")
    await stage_documents(
        make_context(),
        paper_dir="out",
        paper_format="iclr",
        min_refs=1,
        max_refs=2,
        contributions_source=None,
        task_md=None,
        venue_file="",
    )
    assert not await workspace.exists("out/VENUE.md")


async def test_a_venue_file_is_rendered_as_the_brief(make_context: Any, workspace: Any) -> None:
    await workspace.write_bytes("out/VENUE.md", b"seeded from the edited paper")
    await workspace.write_bytes(
        "briefs/journal.md", b"Write for {{PAPER_FORMAT}} with {{PAPER_MIN_REFS}} references."
    )
    await stage_documents(
        make_context(),
        paper_dir="out",
        paper_format="iclr",
        min_refs=3,
        max_refs=5,
        contributions_source=None,
        task_md=None,
        venue_file="briefs/journal.md",
    )
    text = (await workspace.read_bytes("out/VENUE.md")).decode()
    assert text.strip() == "Write for iclr with 3 references."
    with pytest.raises(PaperPreparationError, match="Missing venue file"):
        await stage_documents(
            make_context(),
            paper_dir="out",
            paper_format="iclr",
            min_refs=3,
            max_refs=5,
            contributions_source=None,
            task_md=None,
            venue_file="briefs/missing.md",
        )


@pytest.mark.parametrize("watermark", [True, False])
async def test_the_staged_templates_follow_the_watermark_switch(
    make_context: Any, workspace: Any, watermark: bool
) -> None:
    ctx = make_context()
    await stage_writer_skills(
        ctx,
        paper_dir="out",
        skills=["write-paper"],
        writer_skill="write-paper",
        include_review_skill=False,
        values={},
        paper_format="iclr",
        watermark=watermark,
    )
    templates = [
        entry.path
        for entry in await workspace.list_tree("out/.agents/skills/write-paper")
        if entry.path.endswith("/paper.tex")
    ]
    assert templates
    for path in templates:
        text = (await workspace.read_bytes(f"out/.agents/skills/write-paper/{path}")).decode()
        assert has_managed_watermark(text) is watermark


async def test_validation_without_the_watermark_removes_a_managed_block(workspace: Any) -> None:
    tex = "\\documentclass{article}\n\\begin{document}\nText.\n\\end{document}\n"
    await workspace.write_bytes("p/paper.tex", add_watermark(tex).encode())
    await workspace.write_bytes("p/references.bib", b"")
    toolchain = WorkspacePaperToolchain(pdflatex="no-such-pdflatex", bibtex="no-such-bibtex")
    report = await toolchain.validate("p", workspace=workspace, options={"watermark": False})
    assert not report.passed
    text = (await workspace.read_bytes("p/paper.tex")).decode()
    assert not has_managed_watermark(text)
    assert text == tex
    stored = json.loads((await workspace.read_bytes("p/validation_report.json")).decode())
    assert stored["watermark_validation"]["removed"] is True


async def test_edit_mode_on_an_external_multi_file_paper(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    await _external_paper(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: edit_outputs(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(watermark=False)),
    )
    options = EditOptions(main_file="main.tex", venue_file="")
    out = await run(build(), ctx, task, mode="edit", previous_paper_dir=EXT, edit_options=options)
    assert out["paper"].status == "complete", out["paper"]
    paper_dir = out["paper"].paper_dir
    assert not await workspace.exists(f"{paper_dir}/VENUE.md")
    assert await workspace.exists(f"{paper_dir}/sections/sub/detail.tex")
    assert await workspace.exists(f"{paper_dir}/method_scope_audit.json")
    prompt = prompts_of(scripted, Role.PAPER_WRITER)[0]
    assert "No venue-specific brief is provided for this run." in prompt
    assert "Edit mode is enabled for this run." in prompt


async def test_a_paper_grounded_in_several_tasks_uses_the_cross_task_route(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    await _external_paper(workspace)
    for name in ("sorting", "routing"):
        await workspace.write_bytes(f"studies/extra/{name}/task.md", f"# {name}\n".encode())
        await workspace.write_bytes(f"studies/aggregated/{name}/run_summary.csv", b"variant_id\n")
    data = paper_data(
        task_names=["packing", "sorting", "routing"],
        cross_task_results={
            "sorting": {"control_best": {"metric_name": "time", "mean": 1.5}},
            "routing": {"control_best": {"metric_name": "cost", "mean": 7.0}},
        },
    )
    data.grounding_sources["result.sorting.control_best.time.mean"] = GroundingSource(
        type="metric", value=1.5
    )
    data.grounding_sources["result.routing.control_best.cost.mean"] = GroundingSource(
        type="metric", value=7.0
    )
    scripted = models(
        {
            Role.PAPER_WRITER: edit_outputs(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(
        build(),
        ctx,
        task,
        mode="edit",
        previous_paper_dir=EXT,
        edit_options=EditOptions(main_file="main.tex"),
        paper_data=data,
        artifact_paths={"extra_tasks": "studies/extra", "aggregated": "studies/aggregated"},
    )
    assert out["paper"].status == "complete", out["paper"]
    paper_dir = out["paper"].paper_dir
    written = json.loads((await workspace.read_bytes(f"{paper_dir}/paper_data.json")).decode())
    assert written["task_names"] == ["packing", "sorting", "routing"]
    assert "result.routing.control_best.cost.mean" in written["grounding_sources"]
    assert await workspace.exists(f"{paper_dir}/extra_tasks/sorting/task.md")
    context = json.loads(
        (await workspace.read_bytes(f"{paper_dir}/paper_edit_context.json")).decode()
    )
    rows = context["results_task_linkage"]["linked_results_tasks"]
    linked = {row["task_name"]: row for row in rows}
    assert linked["routing"]["extra_task_workspace_path"] == "./extra_tasks/routing"
    assert linked["sorting"]["aggregated_workspace_path"] == "./aggregated/sorting"
    markdown = (await workspace.read_bytes(f"{paper_dir}/paper_edit_context.md")).decode()
    assert "- Task `routing` -> `./aggregated/routing` and `./extra_tasks/routing`" in markdown
