"""Behaviours of the variants assembly that equal the vendored upstream's."""

from __future__ import annotations

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
