"""``stage_study_sources`` after the paper data (``paper.sh`` 857, ``paper_edit.sh`` 1447) and
``verify_study_sources`` before each audit (``paper.sh`` 3366, ``paper_edit.sh`` 3819)."""

from __future__ import annotations

import json
from typing import Any

from test_data_build import selection_file
from test_loop_support import setup_task

from langgraph_agentic_scientist.core.results import DownstreamSelection
from langgraph_agentic_scientist.core.runconfig import build_run_config
from langgraph_agentic_scientist.paper.audit import build_audit_graph
from langgraph_agentic_scientist.paper.data import build_paper_data
from langgraph_agentic_scientist.paper.graph import _seed_edit

PAPER_DIR = "run/paper/p"
DOWNSTREAM = DownstreamSelection(
    status="selected", selection_path="downstream/.downstream_selection.json"
)


class _Runtime:
    def __init__(self, context: Any) -> None:
        self.context = context


async def test_a_fresh_paper_stages_the_study_sources_into_its_data(
    make_context: Any, workspace: Any
) -> None:
    await selection_file(workspace)
    data = await build_paper_data(workspace, downstream=DOWNSTREAM)
    state = {"mode": "fresh", "paper_dir": PAPER_DIR, "paper_data": data, "downstream": DOWNSTREAM}
    update = await _seed_edit(state, _Runtime(make_context()))  # type: ignore[arg-type]
    study = update["paper_data"].model_extra["downstream_study"]
    assert study["source_evidence"]["program"] == "source_evidence/downstream/program"
    assert await workspace.exists(f"{PAPER_DIR}/source_evidence/downstream/program/src/program.py")
    written = json.loads((await workspace.read_bytes(f"{PAPER_DIR}/paper_data.json")).decode())
    assert written["downstream_study"]["source_evidence"] == study["source_evidence"]


async def test_the_audit_refuses_a_paper_whose_study_sources_were_not_staged(
    make_context: Any, workspace: Any
) -> None:
    await selection_file(workspace)
    data = await build_paper_data(workspace, downstream=DOWNSTREAM)
    ctx = make_context()
    task = await setup_task(workspace)
    await workspace.write_bytes(f"{PAPER_DIR}/paper.tex", b"\\documentclass{article}")
    await workspace.write_bytes(
        f"{PAPER_DIR}/paper_data.json", json.dumps(data.model_dump(mode="json")).encode()
    )
    out = await build_audit_graph().ainvoke(
        {
            "run_config": build_run_config(ctx),
            "task": task,
            "paper_dir": PAPER_DIR,
            "downstream": DOWNSTREAM,
        },
        context=ctx,
    )
    assert out["audit"].outcome == "operational_failure", out
    assert "source" in str(out["audit"]), out["audit"]
