"""The evidence-audit gate against upstream ``paper_evidence_audit.py`` and its shell callers."""

from __future__ import annotations

import json
from typing import Any

from test_loop_support import paper_config
from testkit import fast_config

from langgraph_agentic_scientist.core.results import ValidationReport
from langgraph_agentic_scientist.core.runconfig import build_run_config
from langgraph_agentic_scientist.paper.audit import build_audit_packet
from langgraph_agentic_scientist.paper.graph import _default_validation
from langgraph_agentic_scientist.testing import FakePaperToolchain

PAPER_DIR = "run/paper/p"
TEX = "\\documentclass{article}\n\\begin{document}\nResult 2.6.\n\\end{document}\n"


async def test_packet_validation_summary_reads_ok_only(make_context: Any, workspace: Any) -> None:
    await workspace.write_bytes(f"{PAPER_DIR}/paper.tex", TEX.encode())
    await workspace.write_bytes(f"{PAPER_DIR}/validation_report.json", b'{"passed": true}')
    await build_audit_packet(make_context(), PAPER_DIR, f"{PAPER_DIR}/packet.md")
    packet = (await workspace.read_bytes(f"{PAPER_DIR}/packet.md")).decode()
    summary = packet.split("## validation report summary\n\n```json\n", 1)[1].split("```", 1)[0]
    assert json.loads(summary)["ok"] is None


class _Runtime:
    def __init__(self, context: Any) -> None:
        self.context = context


async def test_failure_fingerprint_covers_imported_tex_sources(
    make_context: Any, workspace: Any
) -> None:
    toolchain = FakePaperToolchain([ValidationReport(passed=False, errors=["missing section"])])
    ctx = make_context(paper_toolchain=toolchain, config=fast_config(paper=paper_config()))
    tex = TEX.replace("Result 2.6.", "\\input{sections/results}")
    await workspace.write_bytes(f"{PAPER_DIR}/paper.tex", tex.encode())
    await workspace.write_bytes(f"{PAPER_DIR}/sections/results.tex", b"first")
    state = {"paper_dir": PAPER_DIR, "run_config": build_run_config(ctx), "repair_round": 1}
    first = (await _default_validation(state, _Runtime(ctx)))["validation_fingerprint"]
    await workspace.write_bytes(f"{PAPER_DIR}/sections/results.tex", b"second")
    second = (await _default_validation(state, _Runtime(ctx)))["validation_fingerprint"]
    assert first and second and first != second
