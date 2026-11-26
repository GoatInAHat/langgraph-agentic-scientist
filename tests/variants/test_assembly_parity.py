"""Behaviours of the variants assembly that equal the vendored upstream's."""

from __future__ import annotations

from typing import Any

from langgraph_agentic_scientist.variants.cross_task import find_best_program_dir
from langgraph_agentic_scientist.variants.execute import check_analysis_files
from langgraph_agentic_scientist.workspace import LocalWorkspace


async def test_program_directories_are_listed_not_probed(workspace: LocalWorkspace) -> None:
    for name in ("program_0", "program_9", "program_007"):
        await workspace.write_bytes(f"b/{name}/metrics.json", b"{}")
    await workspace.write_bytes("b/program_9/analysis/run_summary.csv", b"x")
    assert await find_best_program_dir(workspace, "b") == "b/program_9"
    files = await check_analysis_files(workspace, "b")
    assert files["run_summary.csv"] is True


async def test_best_program_with_a_gap_beyond_the_attempts(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("g/program_0/x", b"x")
    await workspace.write_bytes("g/program_007/metrics.json", b"{}")
    assert await find_best_program_dir(workspace, "g") == "g/program_007"


async def test_a_directory_named_like_a_csv_makes_a_flat_task(workspace: LocalWorkspace) -> None:
    from langgraph_agentic_scientist.variants import aggregate, readme

    await workspace.write_bytes("t/v/results.csv/keep", b"x")
    assert await aggregate._is_flat_extra_task(workspace, "t")
    assert await readme._is_flat_extra_task(workspace, "t")


def test_a_json_null_in_a_flat_task_reads_none() -> None:
    from langgraph_agentic_scientist import PromptRegistry
    from langgraph_agentic_scientist.variants.readme import assemble_readme

    text = assemble_readme(
        PromptRegistry.default(),
        "summary",
        [],
        {},
        {},
        None,
        {"t": [{"variant_id": "a", "title": None, "kind": None}]},
    )
    assert "### None (`a`)" in text
    assert "**Kind**: None" in text
    assert "None (None variant)." in text
    assert "{{" not in text


async def test_a_bare_carriage_return_summary_is_unknown(workspace: LocalWorkspace) -> None:
    from langgraph_agentic_scientist import PromptRegistry
    from langgraph_agentic_scientist.variants.analysis import summary_qualification

    from .test_analysis_and_aggregate import _qualified_program

    program = await _qualified_program(workspace, overall=True)
    path = workspace.root / program / "analysis" / "run_summary.csv"
    path.write_bytes(path.read_bytes().replace(b"\r\n", b"\r"))
    result = await summary_qualification(workspace, program, prompts=PromptRegistry.default())
    assert result == {"overall_pass": ""}


async def _cross_task_with(
    workspace: LocalWorkspace, study: Any, phase_context: Any, broken: str
) -> Any:
    from langgraph_agentic_scientist.evaluation.protocol import EvaluationResult
    from langgraph_agentic_scientist.testing import FakeEvaluator, write_minimal_task

    from .test_cross_task import _run, _scripts

    task, best = study
    await write_minimal_task(workspace, "extra1")
    down = EvaluationResult(evaluation_id="x", status="infrastructure_error", summary="no docker")
    evaluator = FakeEvaluator(results={f"r1/extra/extra1/{broken}/program_1": down})
    ctx = phase_context(_scripts(), evaluator=evaluator, extra_tasks=("extra1",))
    return await _run(ctx, task, best)


async def test_an_infrastructure_failure_of_the_adapted_control_fails_the_phase(
    workspace: LocalWorkspace, study: Any, phase_context: Any
) -> None:
    result = await _cross_task_with(workspace, study, phase_context, "control_best")
    assert result.status == "failed" and result.failure is not None
    assert result.failure.scope == "infrastructure"
    assert "no docker" in result.failure.detail


async def test_an_infrastructure_failure_of_an_adapted_variant_fails_the_phase(
    workspace: LocalWorkspace, study: Any, phase_context: Any
) -> None:
    result = await _cross_task_with(workspace, study, phase_context, "ablation_no_reheat")
    assert result.status == "failed" and result.failure is not None
    assert result.failure.scope == "infrastructure"
    assert result.readme_path is None


async def test_ignore_patterns_apply_to_top_level_names_as_written(
    workspace: LocalWorkspace,
) -> None:
    from langgraph_agentic_scientist.variants.materialize import copy_task_files

    await workspace.write_bytes("s/.agentic_task_ignore", b"data/cache/\n*.tmp\nbig\n")
    for path in ("data/cache/x", "data/keep", "a[1].tmp", "a1.tmp", "sub/b.tmp", "big/y", "k"):
        await workspace.write_bytes(f"s/{path}", b"x")
    await copy_task_files(workspace, "s", "d")
    root = workspace.root / "d"
    copied = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    assert copied == {
        ".agentic_task_ignore",
        "data/cache/x",
        "data/keep",
        "sub/b.tmp",
        "k",
    }
