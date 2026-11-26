"""The evidence-audit gate against upstream ``paper_evidence_audit.py`` and its shell callers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_loop_support import audit_pass, models, paper_config, paper_data, setup_task
from testkit import fast_config

from langgraph_agentic_scientist.core.results import ValidationReport
from langgraph_agentic_scientist.core.runconfig import build_run_config
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper.audit import (
    audit_input_sha256,
    build_audit_graph,
    build_audit_packet,
)
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


async def _paper(workspace: Any, tex: str = TEX) -> None:
    await workspace.write_bytes(f"{PAPER_DIR}/paper.tex", tex.encode())
    await workspace.write_bytes(f"{PAPER_DIR}/claims_grounding.json", b'{"claims": []}')
    await workspace.write_bytes(
        f"{PAPER_DIR}/paper_data.json",
        json.dumps(paper_data().model_dump(mode="json")).encode(),
    )
    await workspace.write_bytes(f"{PAPER_DIR}/validation_report.json", b'{"ok": true}')


async def _run(ctx: Any, task: Any, **extra: Any) -> dict[str, Any]:
    payload = {"run_config": build_run_config(ctx), "task": task, "paper_dir": PAPER_DIR, **extra}
    return dict(await build_audit_graph().ainvoke(payload, context=ctx))


async def test_hashes_cover_referenced_figures_with_upstream_suffixes(workspace: Any) -> None:
    await _paper(workspace, TEX.replace("Result 2.6.", "\\includegraphics{figures/plot.svg}"))
    await workspace.write_bytes(f"{PAPER_DIR}/figures/plot.svg", b"<svg/>")
    await workspace.write_bytes(f"{PAPER_DIR}/figures/unused.pdf", b"pdf")
    hashes = await audit_input_sha256(workspace, PAPER_DIR)
    assert "figure-asset/figures/plot.svg" in hashes
    assert "figure-asset/figures/unused.pdf" not in hashes


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("paper.tex", b"\xff\xfe", "utf-8"),
        ("scientific-evidence/verification-report.json", b"{bad", "Expecting"),
        ("scientific-evidence/verification-report.json", b"[1]", "Expected JSON object"),
        ("figure_manifest.json", b"{bad", "figure_manifest.json is invalid JSON"),
        (
            "figure_manifest.json",
            json.dumps({"figures": [{}] * 257}).encode(),
            "more than 256 figures",
        ),
        ("paper.tex", b"\\includegraphics{a.png}", "Unsafe figure reference"),
        ("paper.tex", b"x" * (16 * 1024 * 1024 + 1), "exceeds 16777216 bytes"),
    ],
)
async def test_hash_inputs_raise_where_upstream_raises(
    workspace: Any, name: str, content: bytes, message: str
) -> None:
    await _paper(workspace)
    await workspace.write_bytes(f"{PAPER_DIR}/{name}", content)
    with pytest.raises(ValueError, match=message):
        await audit_input_sha256(workspace, PAPER_DIR)


async def test_an_unreadable_input_is_an_operational_failure(
    make_context: Any, workspace: Any
) -> None:
    ctx = make_context(
        models=models({Role.EVIDENCE_AUDITOR: [audit_pass()]}),
        config=fast_config(paper=paper_config()),
    )
    task = await setup_task(workspace)
    await _paper(workspace)
    await workspace.write_bytes(f"{PAPER_DIR}/figure_manifest.json", b"{bad")
    out = await _run(ctx, task)
    assert out["audit"].outcome == "operational_failure"
    assert out["audit"].detail == "figure_manifest.json is invalid JSON"


async def test_hash_exclusions_follow_upstream(workspace: Any) -> None:
    await _paper(workspace)
    for name in (
        "revisions/notes.md",
        "sub/.agents/k.md",
        ".agents/notes.md",
        "paper_events_1.jsonl",
        ".claude/skills/s/SKILL.md",
        ".agents/skills/s/SKILL.md",
        "results/run.json",
    ):
        await workspace.write_bytes(f"{PAPER_DIR}/{name}", b"x")
    hashes = await audit_input_sha256(workspace, PAPER_DIR)
    sources = {key.removeprefix("evidence-source/") for key in hashes if "evidence-source/" in key}
    assert {"revisions/notes.md", "sub/.agents/k.md", ".agents/notes.md"} <= sources
    assert not sources & {
        "paper_events_1.jsonl",
        ".claude/skills/s/SKILL.md",
        ".agents/skills/s/SKILL.md",
        "results/run.json",
    }


async def test_the_audit_copy_excludes_only_skill_directories(
    make_context: Any, workspace: Any
) -> None:
    seen: list[bool] = []

    def look(messages: Any) -> Any:
        root = workspace.host_path("run/paper/.paper_evidence_audit_workspaces/p-round1-attempt1")
        seen.append(Path(root, ".agents/notes.md").exists())
        seen.append(Path(root, ".agents/skills/s/SKILL.md").exists())
        return audit_pass()

    ctx = make_context(
        models=models({Role.EVIDENCE_AUDITOR: [look]}), config=fast_config(paper=paper_config())
    )
    task = await setup_task(workspace)
    await _paper(workspace)
    await workspace.write_bytes(f"{PAPER_DIR}/.agents/notes.md", b"x")
    await workspace.write_bytes(f"{PAPER_DIR}/.agents/skills/s/SKILL.md", b"x")
    out = await _run(ctx, task)
    assert out["audit"].outcome == "pass"
    assert seen == [True, False]


async def test_hashes_cover_the_targets_the_verifier_resolved(workspace: Any) -> None:
    await _paper(workspace)
    results = [
        {"status": "resolved", "resolved_path": "data/model.pt"},
        {"status": "resolved", "resolved_path": "data/run"},
        {"status": "resolved", "resolved_path": "results/aliased.bin"},
        {"status": "resolved", "resolved_path": "/abs/model.pt"},
    ]
    report = json.dumps({"summary": {}, "results": results}).encode()
    await workspace.write_bytes(f"{PAPER_DIR}/scientific-evidence/verification-report.json", report)
    await workspace.write_bytes(f"{PAPER_DIR}/data/model.pt", b"weights")
    await workspace.write_bytes(f"{PAPER_DIR}/data/run/metrics.bin", b"1")
    await workspace.write_bytes(f"{PAPER_DIR}/results/aliased.bin", b"1")
    before = await audit_input_sha256(workspace, PAPER_DIR)
    assert "evidence-source/data/model.pt" in before
    assert "evidence-tree/data/run" in before
    assert not any("aliased" in key or "abs" in key for key in before)
    await workspace.write_bytes(f"{PAPER_DIR}/data/run/metrics.bin", b"2")
    after = await audit_input_sha256(workspace, PAPER_DIR)
    assert after["evidence-tree/data/run"] != before["evidence-tree/data/run"]
