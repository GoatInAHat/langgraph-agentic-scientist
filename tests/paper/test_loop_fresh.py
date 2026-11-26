"""The fresh paper loop (``paper.sh``) end to end, with scripted roles and a fake toolchain."""

from __future__ import annotations

import json
from typing import Any

from test_loop_support import (
    audit_fail,
    audit_pass,
    build,
    fast_config,
    models,
    no_gaps,
    paper_config,
    prompts_of,
    reports,
    review,
    run,
    setup_task,
    strategy,
    write_paper,
)

from langgraph_agentic_scientist.core.results import AnalysisResult, PosthocResult, ValidationReport
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import final


async def test_fresh_single_stage_run_completes(make_context: Any, workspace: Any) -> None:
    scripted = models(
        {
            Role.PAPER_WRITER: write_paper(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    task = await setup_task(workspace)
    out = await run(build(), ctx, task)
    result = out["paper"]
    assert result.status == "complete", result
    paper_dir = result.paper_dir
    assert paper_dir == "run/paper/final_r1"
    for name in (
        "paper.tex",
        "paper_prompt.md",
        "paper_data.json",
        "VENUE.md",
        "global_memory_summary.md",
        "validation_report.json",
        "validation_feedback.json",
        "paper_evidence_audit.json",
        "paper_evidence_audit_packet.md",
        ".postdraft_reference_enrichment_complete",
        "task/task.md",
    ):
        assert await workspace.exists(f"{paper_dir}/{name}"), name
    manifest = json.loads((await workspace.read_bytes("run/paper/latest_manifest.json")).decode())
    assert manifest["paper_tex"] == f"{paper_dir}/paper.tex"
    memory = (await workspace.read_bytes(f"{paper_dir}/global_memory/memory.jsonl")).decode()
    assert '"step": "paper"' in memory
    writer_prompt = prompts_of(scripted, Role.PAPER_WRITER)[0]
    assert "{{" not in writer_prompt.split("PAPER_OPENAI_API_KEY")[0][:200]
    assert "No concept-figure renderer is configured for this run." in writer_prompt
    assert "Pack circles densely." in writer_prompt


async def test_two_stage_strategy_selects_the_family_and_ranking_is_merged(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    analysis_dir = "run/posthoc/analyses/a1/attempt_01"
    await workspace.write_bytes(f"{analysis_dir}/analysis_summary.md", b"Mechanism found.")
    await workspace.write_bytes(
        f"{analysis_dir}/analysis_manifest.json",
        json.dumps({"outputs": {"data_files": ["data.json"]}}).encode(),
    )
    await workspace.write_bytes(f"{analysis_dir}/data.json", b'{"x": 1}')
    posthoc = PosthocResult(
        status="completed",
        analyses=[
            AnalysisResult(
                analysis_id="a1",
                family="mechanism",
                title="Mechanism",
                status="passed",
                analysis_dir=analysis_dir,
            )
        ],
    )
    from langgraph_agentic_scientist.paper.models import PosthocRanking
    from langgraph_agentic_scientist.testing import structured_call

    ranking = structured_call(
        PosthocRanking,
        {
            "ranked_analyses": [
                {
                    "analysis_id": "a1",
                    "rank": 1,
                    "relevance_to_thesis": "high",
                    "relevance_justification": "It explains the result.",
                    "recommended_usage": "main_results",
                    "key_finding_for_paper": "The mechanism.",
                }
            ],
            "ranking_rationale": "Only one.",
        },
    )
    scripted = models(
        {
            Role.PAPER_STRATEGIST: [strategy("paper_standalone")],
            Role.POSTHOC_RANKER: [ranking],
            Role.PAPER_WRITER: write_paper(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(two_stage=True)),
    )
    out = await run(build(), ctx, task, posthoc=posthoc)
    assert out["paper"].status == "complete", out["paper"]
    paper_dir = out["paper"].paper_dir
    resolution = json.loads(
        (await workspace.read_bytes(f"{paper_dir}/paper_strategy_resolution.json")).decode()
    )
    assert resolution["prompt_family"] == "paper_standalone"
    assert resolution["writer_skill"] == "write-paper-standalone"
    writer_prompt = prompts_of(scripted, Role.PAPER_WRITER)[0]
    assert "Two-stage writing is enabled for this run." in writer_prompt
    data = out["paper_data"].model_dump(mode="json", by_alias=True)
    assert "posthoc.a1.summary" in data["grounding_sources"]
    assert "posthoc.a1.data.data.json" in data["grounding_sources"]
    assert await workspace.exists(f"{paper_dir}/posthoc_ranking.json")
    assert await workspace.exists(f"{paper_dir}/.agents/skills/write-paper-standalone/SKILL.md")


async def test_strategy_status_fail_is_repaired_then_exhausts(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    scripted = models(
        {
            Role.PAPER_STRATEGIST: [
                strategy(status="fail", failure_reason="no story"),
                strategy(status="fail", failure_reason="still none"),
            ],
        }
    )
    ctx = make_context(
        models=scripted,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(two_stage=True, max_repair_rounds=1)),
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "failed"
    assert "exhausted strategy repair rounds after 1 attempts" in out["paper"].failure.detail
    feedback = json.loads(
        (
            await workspace.read_bytes("run/paper/final_r1/strategy_validation_feedback.json")
        ).decode()
    )
    assert feedback["hard_errors"] == ["paper strategy failed: still none"]
    second = prompts_of(scripted, Role.PAPER_STRATEGIST)[1]
    assert "./strategy_validation_feedback.json" in second


async def test_writer_failure_writes_feedback_and_repairs(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: [final("I could not write it"), *write_paper()],
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "complete"
    paper_dir = out["paper"].paper_dir
    assert await workspace.exists(f"{paper_dir}/paper_repair_prompt_round_1.md")
    second = prompts_of(scripted, Role.PAPER_WRITER)[1]
    assert "Validation feedback from a previous draft is present." in second


async def test_audit_findings_start_a_repair_round(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: [*write_paper(), final("repaired")],
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_fail(), audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "complete"
    paper_dir = out["paper"].paper_dir
    assert await workspace.exists(
        f"{paper_dir}/revisions/evidence_audit_before_repair_round_01.pdf"
    )
    repair_prompt = prompts_of(scripted, Role.PAPER_WRITER)[1]
    assert "Validation feedback from a previous draft is present." in repair_prompt
    assert scripted.scripted(Role.EVIDENCE_AUDITOR).call_count == 2


async def test_audit_operational_failure_is_incomplete(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: write_paper(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [RuntimeError("auditor crashed")],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=reports(True), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "incomplete"
    assert "paper evidence audit failed operationally" in out["paper"].incomplete_reason
    assert not await workspace.exists("run/paper/latest_manifest.json")


async def test_review_round_then_repair(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: [*write_paper(), final("addressed review")],
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass(), audit_pass()],
            Role.PAPER_REVIEWER: [review("revise")],
        }
    )
    ctx = make_context(
        models=scripted,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(review_rounds=1)),
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "complete"
    paper_dir = out["paper"].paper_dir
    feedback = json.loads(
        (await workspace.read_bytes(f"{paper_dir}/paper_review_feedback.json")).decode()
    )
    assert feedback["major_comment_count"] == 1
    assert await workspace.exists(
        f"{paper_dir}/revisions/review_round_01_before_repair_round_01.pdf"
    )
    repair_prompt = prompts_of(scripted, Role.PAPER_WRITER)[1]
    assert "Reviewer feedback from a prior draft is present." in repair_prompt


async def test_review_stage_failure_is_incomplete(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: write_paper(),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass()],
            Role.PAPER_REVIEWER: [RuntimeError("reviewer crashed")],
        }
    )
    ctx = make_context(
        models=scripted,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(review_rounds=1)),
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "incomplete"
    assert "paper review stage failed" in out["paper"].incomplete_reason


async def test_figure_quality_repair_rolls_back_a_validation_regression(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    warning = "figures[0] PNG width too small: 600 < 1200"
    toolchain = reports(
        ValidationReport(passed=True, figure_warnings=[warning]),
        ValidationReport(passed=False, errors=["figure broke the build"]),
        ValidationReport(passed=True, figure_warnings=[warning]),
    )
    scripted = models(
        {
            Role.PAPER_WRITER: write_paper(),
            Role.FIGURE_QUALITY_REPAIRER: [
                *write_paper("\\documentclass{article}\\begin{document}broken\\end{document}")
            ],
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass(), audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=toolchain, config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "complete", out["paper"]
    tex = (await workspace.read_bytes(f"{out['paper'].paper_dir}/paper.tex")).decode()
    assert "broken" not in tex
    assert len(toolchain.validations) == 3


async def test_figure_quality_repair_rolls_back_an_audit_failure(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    warning = "figures[0] paper.tex references raster PNG despite available vector PDF"
    toolchain = reports(
        ValidationReport(passed=True, figure_warnings=[warning]),
        ValidationReport(passed=True, figure_warnings=[warning]),
    )
    scripted = models(
        {
            Role.PAPER_WRITER: write_paper(),
            Role.FIGURE_QUALITY_REPAIRER: write_paper("relinked figure version"),
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass(), audit_fail()],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=toolchain, config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "complete", out["paper"]
    tex = (await workspace.read_bytes(f"{out['paper'].paper_dir}/paper.tex")).decode()
    assert "relinked" not in tex


async def test_repeated_validation_failure_is_incomplete(make_context: Any, workspace: Any) -> None:
    task = await setup_task(workspace)
    scripted = models(
        {
            Role.PAPER_WRITER: [*write_paper(), final("no change")],
            Role.DRAFT_GAP_ANALYST: [no_gaps()],
            Role.EVIDENCE_AUDITOR: [audit_pass(), audit_pass()],
        }
    )
    ctx = make_context(
        models=scripted, paper_toolchain=reports(False), config=fast_config(paper=paper_config())
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "incomplete"
    assert "inputs were unchanged after repair" in out["paper"].incomplete_reason


async def test_postdraft_enrichment_failure_fails_the_phase(
    make_context: Any, workspace: Any
) -> None:
    task = await setup_task(workspace)
    from langgraph_agentic_scientist.paper.models import DraftGapsAnswer
    from langgraph_agentic_scientist.testing import structured_call

    bad = structured_call(DraftGapsAnswer, {"memo": "", "gaps": {"status": "maybe"}})
    scripted = models({Role.PAPER_WRITER: write_paper(), Role.DRAFT_GAP_ANALYST: [bad, bad]})
    ctx = make_context(
        models=scripted,
        paper_toolchain=reports(True),
        config=fast_config(paper=paper_config(max_repair_rounds=1)),
    )
    out = await run(build(), ctx, task)
    assert out["paper"].status == "failed"
    assert "exhausted draft reference gap repair rounds" in out["paper"].failure.detail
