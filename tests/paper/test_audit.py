"""The evidence-audit gate: disposable read-only copy, input-hash check, contract attempts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from test_data_support import final_result
from test_loop_support import (
    audit_fail,
    audit_pass,
    fast_config,
    models,
    paper_config,
    paper_data,
    setup_task,
)

from langgraph_agentic_scientist.core.results import GroundingSource
from langgraph_agentic_scientist.core.runconfig import build_run_config
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper.audit import (
    audit_input_sha256,
    build_audit_graph,
    build_audit_packet,
    is_generated_audit_artifact,
    verify_final_outcome,
)
from langgraph_agentic_scientist.testing import final, tool_call

PAPER_DIR = "run/paper/p"
TEX = "\\documentclass{article}\n\\begin{document}\nResult 2.6.\n\\end{document}\n"


async def _paper(workspace: Any) -> None:
    await workspace.write_bytes(f"{PAPER_DIR}/paper.tex", TEX.encode())
    await workspace.write_bytes(f"{PAPER_DIR}/claims_grounding.json", b'{"claims": []}')
    await workspace.write_bytes(
        f"{PAPER_DIR}/paper_data.json",
        json.dumps(paper_data().model_dump(mode="json")).encode(),
    )
    await workspace.write_bytes(f"{PAPER_DIR}/validation_report.json", b'{"ok": true}')


async def _run(ctx: Any, task: Any, **extra: Any) -> dict[str, Any]:
    payload = {"run_config": build_run_config(ctx), "task": task, "paper_dir": PAPER_DIR, **extra}
    return dict(await build_audit_graph().ainvoke(payload, context=ctx))


async def test_pass_writes_artifacts_and_merges_the_report(
    make_context: Any, workspace: Any
) -> None:
    ctx = make_context(
        models=models({Role.EVIDENCE_AUDITOR: [audit_pass()]}),
        config=fast_config(paper=paper_config()),
    )
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task)
    assert out["audit"].outcome == "pass"
    audit = json.loads(
        (await workspace.read_bytes(f"{PAPER_DIR}/paper_evidence_audit.json")).decode()
    )
    assert audit["verdict"] == "pass" and audit["finding_count"] == 0
    assert audit["input_sha256"]["paper.tex"]
    report = json.loads(
        (await workspace.read_bytes(f"{PAPER_DIR}/validation_report.json")).decode()
    )
    assert report["agentic_evidence_audit"]["checked"] is True
    packet = (await workspace.read_bytes(f"{PAPER_DIR}/paper_evidence_audit_packet.md")).decode()
    assert packet.startswith("# Agentic evidence audit packet")
    assert "Use this packet as the default entrypoint." in packet
    assert "Result 2.6." in packet
    assert await workspace.exists(f"{PAPER_DIR}/paper_evidence_audit_prompt.md")


async def test_findings_write_feedback_for_the_repair(make_context: Any, workspace: Any) -> None:
    ctx = make_context(
        models=models({Role.EVIDENCE_AUDITOR: [audit_fail()]}),
        config=fast_config(paper=paper_config()),
    )
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task)
    assert out["audit"].outcome == "findings"
    feedback = json.loads(
        (await workspace.read_bytes(f"{PAPER_DIR}/paper_evidence_audit_feedback.json")).decode()
    )
    assert feedback["finding_count"] == 1 and feedback["ok"] is False
    validation = json.loads(
        (await workspace.read_bytes(f"{PAPER_DIR}/validation_feedback.json")).decode()
    )
    assert validation["hard_errors"][0].startswith("agentic evidence audit failed")
    assert validation["agentic_evidence_audit"]["findings"][0]["id"] == "results-overclaim-1"


async def test_the_auditor_cannot_write(make_context: Any, workspace: Any) -> None:
    scripted = models(
        {
            Role.EVIDENCE_AUDITOR: [
                tool_call("write_file", {"file_path": "/paper.tex", "content": "tampered"}),
                tool_call("execute", {"command": "touch x"}),
                audit_pass(),
            ]
        }
    )
    ctx = make_context(models=scripted, config=fast_config(paper=paper_config()))
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task)
    assert out["audit"].outcome == "pass"
    assert (await workspace.read_bytes(f"{PAPER_DIR}/paper.tex")).decode() == TEX
    assert not await workspace.exists(f"{PAPER_DIR}/x")


async def test_a_change_behind_the_auditors_back_is_refused(
    make_context: Any, workspace: Any
) -> None:
    def change_then_pass(messages: list[BaseMessage]) -> AIMessage:
        Path(workspace.host_path(f"{PAPER_DIR}/claims_grounding.json")).write_text(
            '{"claims": [1]}'
        )
        return audit_pass()

    ctx = make_context(
        models=models({Role.EVIDENCE_AUDITOR: [change_then_pass]}),
        config=fast_config(paper=paper_config()),
    )
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task)
    assert out["audit"].outcome == "operational_failure"
    assert (
        out["audit"].detail
        == "Paper inputs changed after the read-only audit snapshot was prepared"
    )


async def test_contract_retry_then_exhaustion(make_context: Any, workspace: Any) -> None:
    scripted = models({Role.EVIDENCE_AUDITOR: [final("AUDIT_VERDICT: PASS"), final("still text")]})
    ctx = make_context(models=scripted, config=fast_config(paper=paper_config()))
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task)
    assert out["audit"].outcome == "contract_exhausted" and out["audit"].attempts == 2
    feedback = json.loads(
        (
            await workspace.read_bytes(f"{PAPER_DIR}/paper_evidence_audit_stage_feedback.json")
        ).decode()
    )
    assert feedback["failure_kind"] == "stage_contract"
    assert await workspace.exists(
        f"{PAPER_DIR}/paper_evidence_audit_prompt_round_1_contract_retry_2.md"
    )


async def test_contract_retry_then_pass(make_context: Any, workspace: Any) -> None:
    scripted = models({Role.EVIDENCE_AUDITOR: [final("plain text"), audit_pass()]})
    ctx = make_context(models=scripted, config=fast_config(paper=paper_config()))
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task)
    assert out["audit"].outcome == "pass" and out["audit"].attempts == 2
    assert not await workspace.exists(f"{PAPER_DIR}/paper_evidence_audit_stage_feedback.json")


async def test_tampered_final_evidence_fails_before_the_audit(
    make_context: Any, workspace: Any
) -> None:
    verified = paper_data(
        final_evaluation={"status": "completed", "correct": True, "outcome": "qualified"}
    )
    verified.grounding_sources["final.metric.score"] = GroundingSource(
        type="final_metric", value=2.0
    )
    ctx = make_context(
        models=models({}),
        config=fast_config(paper=paper_config(), final_outcome_policy="allow_not_qualified"),
    )
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task, paper_data=verified, final=final_result())
    assert out["audit"].outcome == "operational_failure"
    assert out["audit"].detail == "Paper final evidence differs from verified final evaluation"


def test_verify_final_outcome_checks_final_grounding() -> None:
    expected = paper_data(final_evaluation={"correct": True})
    expected.grounding_sources["final.metric.score"] = GroundingSource(
        type="final_metric", value=2.0
    )
    actual = expected.model_dump(mode="json", by_alias=True)
    assert verify_final_outcome(actual, expected) is None
    actual["grounding_sources"]["final.metric.score"]["value"] = 3.0
    assert verify_final_outcome(actual, expected) == (
        "Paper final evidence grounding differs from verified final evaluation"
    )
    assert verify_final_outcome({}, paper_data()) == "Required paper final evidence is missing"


async def test_hashes_ignore_generated_artifacts_and_skills(workspace: Any) -> None:
    await _paper(workspace)
    before = await audit_input_sha256(workspace, PAPER_DIR)
    await workspace.write_bytes(f"{PAPER_DIR}/paper_evidence_audit_packet.md", b"x")
    await workspace.write_bytes(f"{PAPER_DIR}/paper_prompt.md", b"x")
    await workspace.write_bytes(f"{PAPER_DIR}/.agents/skills/a/SKILL.md", b"x")
    await workspace.write_bytes(f"{PAPER_DIR}/validation_feedback.json", b"{}")
    assert await audit_input_sha256(workspace, PAPER_DIR) == before
    await workspace.write_bytes(f"{PAPER_DIR}/notes.md", b"a new evidence source")
    assert "evidence-source/notes.md" in await audit_input_sha256(workspace, PAPER_DIR)
    assert is_generated_audit_artifact("paper_repair_prompt_round_2.md")
    assert not is_generated_audit_artifact("sections/paper_prompt.md")


async def test_packet_requires_the_manuscript(make_context: Any, workspace: Any) -> None:
    ctx = make_context()
    try:
        await build_audit_packet(ctx, "run/empty", "run/empty/packet.md")
    except FileNotFoundError as exc:
        assert "Missing required manuscript artifact" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected FileNotFoundError")
