"""The in-loop review gate: packet, structured review, feedback, contract attempts."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError
from test_loop_support import fast_config, models, paper_config, review, setup_task

from langgraph_agentic_scientist.core.runconfig import build_run_config
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper.models import ReviewPayload
from langgraph_agentic_scientist.paper.review_gate import build_review_gate_graph
from langgraph_agentic_scientist.testing import final

PAPER_DIR = "run/paper/p"


async def _paper(workspace: Any) -> None:
    await workspace.write_bytes(f"{PAPER_DIR}/paper.tex", b"\\documentclass{article}\nBody.\n")
    await workspace.write_bytes(f"{PAPER_DIR}/paper_strategy.json", b'{"status": "ready"}')


async def _run(ctx: Any, task: Any, review_round: int = 0) -> dict[str, Any]:
    payload = {
        "run_config": build_run_config(ctx),
        "task": task,
        "paper_dir": PAPER_DIR,
        "review_round": review_round,
    }
    return dict(await build_review_gate_graph().ainvoke(payload, context=ctx))


async def test_review_writes_json_markdown_and_feedback(make_context: Any, workspace: Any) -> None:
    scripted = models({Role.PAPER_REVIEWER: [review("revise")]})
    ctx = make_context(models=scripted, config=fast_config(paper=paper_config(review_rounds=1)))
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task)
    assert out["in_loop_review"].outcome == "completed"
    assert out["in_loop_review"].major_comment_count == 1
    stored = json.loads((await workspace.read_bytes(f"{PAPER_DIR}/paper_review.json")).decode())
    assert stored["status"] == "revise"
    markdown = (await workspace.read_bytes(f"{PAPER_DIR}/paper_review.md")).decode()
    assert markdown.startswith("# Paper review\n\n- status: `revise`")
    feedback = json.loads(
        (await workspace.read_bytes(f"{PAPER_DIR}/paper_review_feedback.json")).decode()
    )
    assert feedback["major_comment_count"] == 1 and feedback["ok"] is False
    packet = (await workspace.read_bytes(f"{PAPER_DIR}/paper_review_packet.md")).decode()
    assert "Use this packet as the default review entrypoint." in packet
    assert "## paper_strategy excerpt" in packet
    prompt = scripted.scripted(Role.PAPER_REVIEWER).calls[0]
    text = next(m.content for m in prompt if m.type == "human")
    assert "Return the review as your structured response" in text
    assert await workspace.exists(f"{PAPER_DIR}/.agents/skills/review-paper/SKILL.md")


async def test_contract_exhaustion_clears_feedback_and_continues(
    make_context: Any, workspace: Any
) -> None:
    scripted = models({Role.PAPER_REVIEWER: [final("not json"), final("again")]})
    ctx = make_context(models=scripted, config=fast_config(paper=paper_config(review_rounds=1)))
    task = await setup_task(workspace)
    await _paper(workspace)
    await workspace.write_bytes(f"{PAPER_DIR}/paper_review_feedback.json", b"{}")
    out = await _run(ctx, task, review_round=1)
    assert out["in_loop_review"].outcome == "contract_exhausted"
    assert not await workspace.exists(f"{PAPER_DIR}/paper_review_feedback.json")
    assert await workspace.exists(f"{PAPER_DIR}/paper_review_prompt_round_2_contract_retry_2.md")
    stage = json.loads(
        (await workspace.read_bytes(f"{PAPER_DIR}/paper_review_stage_feedback.json")).decode()
    )
    assert stage["failure_kind"] == "stage_contract"


async def test_stage_failure_restores_the_previous_review(
    make_context: Any, workspace: Any
) -> None:
    scripted = models({Role.PAPER_REVIEWER: [RuntimeError("crash")]})
    ctx = make_context(models=scripted, config=fast_config(paper=paper_config(review_rounds=1)))
    task = await setup_task(workspace)
    await _paper(workspace)
    out = await _run(ctx, task)
    assert out["in_loop_review"].outcome == "stage_failed"


def test_review_payload_rules() -> None:
    issue = {"id": "a", "section": "S", "issue": "i", "evidence": ["e"], "requested_change": "c"}
    score = {"value": 4, "label": "weak"}
    with pytest.raises(ValidationError, match="cannot include major_comments"):
        ReviewPayload.model_validate(
            {"status": "pass", "summary": "s", "major_comments": [issue], "score": score}
        )
    with pytest.raises(ValidationError, match="must include at least one major_comment"):
        ReviewPayload.model_validate({"status": "revise", "summary": "s", "score": score})
    with pytest.raises(ValidationError, match="must include evidence"):
        ReviewPayload.model_validate(
            {
                "status": "revise",
                "summary": "s",
                "major_comments": [{**issue, "evidence": []}],
                "score": score,
            }
        )
    with pytest.raises(ValidationError, match="duplicate review issue id"):
        ReviewPayload.model_validate(
            {
                "status": "revise",
                "summary": "s",
                "major_comments": [issue],
                "minor_comments": [issue],
                "score": score,
            }
        )
    with pytest.raises(ValidationError):
        ReviewPayload.model_validate(
            {"status": "pass", "summary": "s", "score": {"value": 11, "label": "x"}}
        )
