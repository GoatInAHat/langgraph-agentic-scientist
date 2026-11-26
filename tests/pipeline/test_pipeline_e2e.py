"""The whole pipeline over the vendored circle-packing example, phases stubbed at their boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pipeline_stubs import Ledger
from testkit import thread

from langgraph_agentic_scientist import LocalWorkspace, ScientistConfig
from langgraph_agentic_scientist.config import ALL_PHASES


async def test_every_phase_runs_in_upstreams_order(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    ctx = make_context(config=ScientistConfig(phases=ALL_PHASES))
    out = await make_app().ainvoke({"task_dir": circpack}, thread("e2e"), context=ctx)
    assert out["status"] == "completed", out["phase_status"]
    assert ledger.order == [
        "investigation",
        "ablations",
        "final_evaluation",
        "posthoc",
        "paper",
        "review",
        "revision",
    ]
    statuses = {name: status.status for name, status in out["phase_status"].items()}
    assert statuses["task_drafting"] == "completed"  # nothing to draft: the scaffold is present
    assert statuses["downstream_selection"] == "skipped"  # the investigation has a winner
    assert {statuses[name] for name in ALL_PHASES[1:]} == {"completed"}
    assert out["outcome"] is None and out["stop_reason"] is None and not out["failures"]
    assert out["paper"].status == "complete" and out["revision"].status == "promoted"


async def test_the_task_is_snapshotted_once_and_phases_read_the_snapshot(
    make_app: Callable[..., Any],
    make_context: Callable[..., Any],
    workspace: LocalWorkspace,
    circpack: str,
    ledger: Ledger,
) -> None:
    out = await make_app().ainvoke({"task_dir": circpack}, thread("snap"), context=make_context())
    task = out["task"]
    assert task.root == "task" and task.seed_program is not None
    assert (
        task.seed_program.path == "task/program_0"
        and task.seed_program.entrypoint == "src/program.py"
    )
    assert not task.harness_evaluated and task.contract is None
    assert "circle packing" in task.goal
    assert (workspace.root / "task" / "evaluate.py").is_file()
    assert (workspace.root / "task" / ".agentic_task_snapshot.json").is_file()
    assert ledger.last_input("investigation")["task"] == task
    assert "capacity" not in ledger.last_input("investigation")


async def test_phases_receive_the_keys_of_their_input_schema_and_the_recorded_config(
    make_app: Callable[..., Any], make_context: Callable[..., Any], circpack: str, ledger: Ledger
) -> None:
    ctx = make_context(config=ScientistConfig(phases=ALL_PHASES))
    out = await make_app().ainvoke(
        {"task_dir": circpack, "run_id": "r7"}, thread("keys"), context=ctx
    )
    assert out["run_id"] == "r7"
    paper = ledger.last_input("paper")
    assert paper["mode"] == "fresh" and paper["run_id"] == "r7"
    assert paper["final"].outcome == "qualified" and paper["posthoc"].status == "completed"
    assert paper["variants"].status == "completed"
    assert paper["run_config"].config_sha256
    review = ledger.last_input("review")
    assert review["pdf_path"] == "paper/paper.pdf" and review["paper_dir"] == "paper"
    assert review["artifact_stem"] == "ai_review.before_revision"
    assert review["manifest_key"] == "before_revision"
    assert ledger.last_input("revision")["review"].status == "completed"
