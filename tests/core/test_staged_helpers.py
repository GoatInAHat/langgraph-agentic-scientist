"""The vendored helper scripts run from their staged paths through the agent's execute."""

from __future__ import annotations

import json
import shlex
import sys

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.agents.staging import skills_for_step, stage_skills
from langgraph_agentic_scientist.prompts import PromptRegistry

VALUES = {
    "GLOBAL_MEMORY_MD_FILE": "../memory.md",
    "GLOBAL_MEMORY_JSONL_FILE": "../memory.jsonl",
    "CONCLUSIONS_MD_FILE": "../conclusions.md",
    "CONCLUSIONS_HISTORY_FILE": "../conclusions_history.jsonl",
    "ATTEMPT_ID": "idea_1_attempt_01",
    "IDEA_ID": "idea_1",
    "CYCLE_INDEX": "3",
    "WORKER_INDEX": "0",
    "RESULTS_DIR": ".",
    "TASK_DIR": ".",
    "MAX_PROGRAM_EVALS_PER_ATTEMPT": "4",
}


def _execute(backend: object, command: str) -> tuple[int | None, str]:
    response = backend.execute(command)  # type: ignore[attr-defined]
    return response.exit_code, response.output


async def test_staged_skills_are_rendered_and_complete(workspace: LocalWorkspace) -> None:
    staged = await stage_skills(
        workspace,
        "run/attempt",
        skills_for_step("run"),
        prompts=PromptRegistry.default(),
        values=VALUES,
    )
    assert staged == ["edit-program", "evaluate-program", "verify-rubrics", "memory-log"]
    memory = (await workspace.read_bytes("run/attempt/.agents/skills/memory-log/SKILL.md")).decode()
    assert "../memory.jsonl" in memory and "{{GLOBAL_MEMORY_JSONL_FILE}}" not in memory
    assert "find ./.agents -path" in memory
    assert await workspace.exists("run/attempt/.agents/skills/memory-log/scripts/append_memory.py")
    assert await workspace.exists(
        "run/attempt/.agents/skills/verify-rubrics/scripts/create_rubric_status_placeholder.py"
    )


async def test_memory_helper_runs_from_its_staged_path(workspace: LocalWorkspace) -> None:
    await stage_skills(
        workspace, "run/attempt", ("memory-log",), prompts=PromptRegistry.default(), values=VALUES
    )
    backend = workspace.agent_backend("run/attempt")
    python = shlex.quote(sys.executable)
    code, out = _execute(
        backend,
        "find ./.agents -path '*/skills/memory-log/scripts/append_memory.py' "
        "-print -quit 2>/dev/null",
    )
    assert code == 0 and out.strip() == "./.agents/skills/memory-log/scripts/append_memory.py"
    event = {
        "step": "evaluate",
        "scope": "attempt",
        "summary": "s",
        "evidence": "e",
        "next_action": "n",
        "attempt_id": "idea_1_attempt_01",
        "program_id": "program_1",
    }
    helper = "./.agents/skills/memory-log/scripts/append_memory.py"
    code, out = _execute(
        backend,
        f"{python} {helper} --global-memory-md ../memory.md --global-memory-jsonl ../memory.jsonl "
        f"--event-json {shlex.quote(json.dumps(event))}",
    )
    assert code == 0, out
    lines = (await workspace.read_bytes("run/memory.jsonl")).decode().splitlines()
    assert json.loads(lines[-1])["program_id"] == "program_1"
    assert await workspace.exists("run/attempt/memory.md")
    conclusion = "Prefer restarts over long annealing. " * 3
    base = (
        f"{python} {helper} --write-conclusions --conclusions-md ../conclusions.md "
        "--conclusions-history-jsonl ../conclusions_history.jsonl --cycle-index 3 --worker-index 0 "
        "--gate-trigger b --gate-reason 'new category' --conclusions-text "
    )
    code, out = _execute(backend, base + shlex.quote(conclusion))
    assert code == 0, out
    assert (await workspace.read_bytes("run/conclusions.md")).decode().strip() == conclusion.strip()
    code, out = _execute(backend, base + shlex.quote("word " * 301))
    assert code == 1
    assert "word count > 300 (got 301); shorten the paragraph" in out


async def test_rubric_placeholder_helper_runs_from_its_staged_path(
    workspace: LocalWorkspace,
) -> None:
    await stage_skills(
        workspace, "attempt", ("verify-rubrics",), prompts=PromptRegistry.default(), values=VALUES
    )
    rubrics = [{"id": "uses_restarts", "description": "Uses restarts", "type": "implementation"}]
    await workspace.write_bytes("attempt/active_rubrics.json", json.dumps(rubrics).encode())
    await workspace.write_bytes("attempt/active_idea.json", b'{"idea_id": "idea_1"}')
    backend = workspace.agent_backend("attempt")
    code, out = _execute(
        backend,
        f"{shlex.quote(sys.executable)} "
        ".agents/skills/verify-rubrics/scripts/create_rubric_status_placeholder.py "
        ". --program-index 1 --attempt-id idea_1_attempt_01",
    )
    assert code == 0, out
    status = json.loads(await workspace.read_bytes("attempt/program_1/rubric_status.json"))
    assert status["idea_id"] == "idea_1" and status["attempt_id"] == "idea_1_attempt_01"
    assert status["rubrics"][0] == {"id": "uses_restarts", "pass": False, "evidence": ""}
    assert await workspace.exists("attempt/rubric_status.json")
