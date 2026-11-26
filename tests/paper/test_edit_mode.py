"""Edit and revision modes (``paper_edit.sh``): seeding, options, scope audit, references."""

from __future__ import annotations

import json
from typing import Any

from test_loop_support import (
    audit_pass,
    build,
    edit_outputs,
    fast_config,
    models,
    no_gaps,
    paper_config,
    prompts_of,
    reports,
    run,
    setup_task,
    write_paper,
)

from langgraph_agentic_scientist.core.results import EditOptions
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import final, tool_call

OLD = "old/paper"


async def _previous_paper(workspace: Any) -> None:
    await workspace.write_bytes(f"{OLD}/paper.tex", b"old manuscript")
    await workspace.write_bytes(f"{OLD}/claims_grounding.json", b'{"claims": ["old"]}')
    await workspace.write_bytes(f"{OLD}/references.bib", b"@article{old,\n title={Old}\n}\n")
    await workspace.write_bytes(f"{OLD}/references_provenance.json", b'{"references": []}')
    await workspace.write_bytes(f"{OLD}/figures/a.png", b"png")
    await workspace.write_bytes(f"{OLD}/notes.txt", b"not part of the snapshot")
    await workspace.write_bytes(
        f"{OLD}/paper_data.json",
        json.dumps(
            {
                "posthoc_analyses": [{"analysis_id": "old1", "rank": 1}],
                "grounding_sources": {"posthoc.old1.summary": {"type": "posthoc_analysis"}},
            }
        ).encode(),
    )
    await workspace.write_bytes("programs/new/src/program.py", b"NEW = 1\n")
    await workspace.write_bytes("programs/old/src/program.py", b"OLD = 1\n")


async def test_edit_mode_seeds_the_workspace_and_completes(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    await _previous_paper(workspace)
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
    options = EditOptions(edit_programs=["programs/new"], old_programs=["programs/old"])
    out = await run(build(), ctx, task, mode="edit", previous_paper_dir=OLD, edit_options=options)
    assert out["paper"].status == "complete", out["paper"]
    paper_dir = out["paper"].paper_dir
    for name in (
        "previous_paper/paper.tex",
        "previous_paper/figures/a.png",
        "edit_programs/01_new/src/program.py",
        "old_programs/01_old/src/program.py",
        "figures/a.png",
        "method_scope_audit.json",
        "paper_edit_context.json",
    ):
        assert await workspace.exists(f"{paper_dir}/{name}"), name
    assert not await workspace.exists(f"{paper_dir}/previous_paper/notes.txt")
    assert await workspace.read_bytes(f"{paper_dir}/references.bib") == (
        await workspace.read_bytes(f"{OLD}/references.bib")
    )
    context = json.loads(
        (await workspace.read_bytes(f"{paper_dir}/paper_edit_context.json")).decode()
    )
    assert context["reference_mode"] == "preserve" and context["figure_refresh_mode"] == "auto"
    assert context["edit_programs"][0]["workspace_path"] == "./edit_programs/01_new"
    markdown = (await workspace.read_bytes(f"{paper_dir}/paper_edit_context.md")).decode()
    assert "## Editing Contract" in markdown and "`./edit_programs/01_new` sourced from" in markdown
    data = out["paper_data"].model_dump(mode="json", by_alias=True)
    assert data["posthoc_analyses"][0]["analysis_id"] == "old1"
    assert "posthoc.old1.summary" in data["grounding_sources"]
    prompt = prompts_of(scripted, Role.PAPER_WRITER)[0]
    assert "Edit mode is enabled for this run." in prompt
    assert "./method_scope_audit.json" in prompt


async def test_edit_mode_requires_the_scope_audit_outputs(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    await _previous_paper(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: [*write_paper(), *edit_outputs()],
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task, mode="edit", previous_paper_dir=OLD)
    assert out["paper"].status == "complete"
    feedback_prompt = prompts_of(scripted, Role.PAPER_WRITER)[1]
    assert "Validation feedback from a previous draft is present." in feedback_prompt


async def test_edit_mode_skip_and_refresh_options(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    await _previous_paper(workspace)
    scripted = models({Role.PAPER_WRITER: edit_outputs(), Role.EVIDENCE_AUDITOR: [audit_pass()]})
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    options = EditOptions(
        reference_mode="refresh", figure_refresh_mode="refresh", skip_postdraft_enrichment=True
    )
    out = await run(build(), ctx, task, mode="edit", previous_paper_dir=OLD, edit_options=options)
    assert out["paper"].status == "complete"
    paper_dir = out["paper"].paper_dir
    assert not await workspace.exists(f"{paper_dir}/figures/a.png")
    assert b"title={A}" in await workspace.read_bytes(f"{paper_dir}/references.bib")
    assert not await workspace.exists(f"{paper_dir}/.postdraft_reference_enrichment_complete")


async def test_edit_mode_operational_audit_failure_fails(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    await _previous_paper(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: edit_outputs(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [RuntimeError("down")],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task, mode="edit", previous_paper_dir=OLD)
    assert out["paper"].status == "failed"
    assert out["paper"].failure.detail == "required evidence audit failed operationally"


async def test_missing_previous_paper_fails(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    ctx = make_context(
        models=models({}), paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task, mode="edit", previous_paper_dir="nowhere")
    assert out["paper"].status == "failed"
    assert "Missing edit paper directory" in out["paper"].failure.detail


async def test_revision_mode_fixes_the_options_and_restores_references(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    await _previous_paper(workspace)
    scripted = models(
        {
            Role.REVISION_WRITER: [
                tool_call("write_file", {"file_path": "/references.bib", "content": "@misc{new}"}),
                *edit_outputs(),
            ],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(two_stage=True)),
    )
    external = {
        "Summary": "Solid but overclaims.",
        "Strengths": ["clear"],
        "Weaknesses": ["overclaim"],
        "Questions": [],
        "Limitations": [],
        "Overall": 5,
        "Ignored": "dropped",
    }
    out = await run(
        build(),
        ctx,
        task,
        mode="revision",
        previous_paper_dir=OLD,
        external_review=external,
        edit_options=EditOptions(reference_mode="auto", skip_postdraft_enrichment=False),
    )
    assert out["paper"].status == "complete", out["paper"]
    paper_dir = out["paper"].paper_dir
    assert await workspace.read_bytes(f"{paper_dir}/references.bib") == (
        await workspace.read_bytes(f"{OLD}/references.bib")
    )
    feedback = json.loads(
        (await workspace.read_bytes(f"{paper_dir}/revision_feedback.json")).decode()
    )
    assert feedback["untrusted_reviewer_data"] is True and "Ignored" not in feedback
    assert feedback["Overall"] == 5
    md = (await workspace.read_bytes(f"{paper_dir}/revision_feedback.md")).decode()
    assert md.startswith("# External AI Review Feedback")
    assert not await workspace.exists(f"{paper_dir}/.paper_two_stage_enabled")
    assert scripted.scripted(Role.REVISION_WRITER).call_count == 6


async def test_revision_mode_needs_a_review(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    await _previous_paper(workspace)
    ctx = make_context(
        models=models({}), paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task, mode="revision", previous_paper_dir=OLD)
    assert out["paper"].status == "failed"


async def test_write_failure_after_last_round_fails_in_edit_mode(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    await _previous_paper(workspace)
    scripted = models({Role.PAPER_WRITER: [final("nothing"), final("nothing")]})
    ctx = make_context(
        models=scripted,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(max_repair_rounds=1)),
    )
    out = await run(build(), ctx, task, mode="edit", previous_paper_dir=OLD)
    assert out["paper"].status == "failed"
    assert out["paper"].failure.detail.startswith("exhausted repair rounds after 1 attempts")
