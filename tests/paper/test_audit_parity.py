"""The evidence-audit gate against upstream ``paper_evidence_audit.py`` and its shell callers."""

from __future__ import annotations

import json
from typing import Any

from langgraph_agentic_scientist.paper.audit import build_audit_packet

PAPER_DIR = "run/paper/p"
TEX = "\\documentclass{article}\n\\begin{document}\nResult 2.6.\n\\end{document}\n"


async def test_packet_validation_summary_reads_ok_only(make_context: Any, workspace: Any) -> None:
    await workspace.write_bytes(f"{PAPER_DIR}/paper.tex", TEX.encode())
    await workspace.write_bytes(f"{PAPER_DIR}/validation_report.json", b'{"passed": true}')
    await build_audit_packet(make_context(), PAPER_DIR, f"{PAPER_DIR}/packet.md")
    packet = (await workspace.read_bytes(f"{PAPER_DIR}/packet.md")).decode()
    summary = packet.split("## validation report summary\n\n```json\n", 1)[1].split("```", 1)[0]
    assert json.loads(summary)["ok"] is None
