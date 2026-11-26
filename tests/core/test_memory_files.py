"""Reading the memory files agents write: deltas against a pre-state and record checks."""

from __future__ import annotations

import json

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.agents.memory import (
    has_record,
    memory_delta,
    memory_pre_state,
    read_memory_records,
)


async def test_delta_against_a_pre_state(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("w/memory.jsonl", b'{"step": "idea"}\n')
    await workspace.write_bytes("w/memory.md", b"# Memory Log\nfirst\n")
    pre = await memory_pre_state(
        workspace,
        jsonl_path="w/memory.jsonl",
        md_path="w/memory.md",
        conclusions_path="w/conclusions.md",
    )
    assert (pre.jsonl_lines, pre.md_bytes, pre.conclusions_bytes) == (
        1,
        len(b"# Memory Log\nfirst\n"),
        0,
    )
    await workspace.write_bytes("w/memory.jsonl", b'{"step": "idea"}\n{"step": "edit"}\nnot json\n')
    await workspace.write_bytes("w/memory.md", "# Memory Log\nfirst\nsecond — ü\n".encode())
    delta = await memory_delta(
        workspace, jsonl_path="w/memory.jsonl", md_path="w/memory.md", pre=pre
    )
    assert delta.jsonl_lines == ['{"step": "edit"}', "not json"]
    assert delta.md_text == "second — ü\n"


async def test_record_checks_need_ids_on_the_same_record(workspace: LocalWorkspace) -> None:
    records = [
        {"attempt_id": "a1", "step": "edit"},
        {"attempt_id": "a1", "best_program": "program_2"},
        {"attempt": "a1", "program_id": "program_3"},
        {"attempt_id": "a1", "program_id": "program_1"},
    ]
    payload = "\n".join(json.dumps(record) for record in records) + "\n[1, 2]\n"
    await workspace.write_bytes("memory.jsonl", payload.encode())
    loaded = await read_memory_records(workspace, "memory.jsonl")
    assert len(loaded) == 4
    assert has_record(loaded, attempt_id="a1")
    assert has_record(loaded, attempt_id="a1", program_id="program_1")
    assert not has_record(loaded, attempt_id="a1", program_id="program_2")
    assert not has_record(loaded, attempt_id="a1", program_id="program_3")
    assert not has_record(loaded, attempt_id="a2")
