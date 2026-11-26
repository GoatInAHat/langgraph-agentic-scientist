"""Memory files and their sharing: pull, push, the attempt's view, the host's own events."""

from __future__ import annotations

import json
import sys
from typing import Any

from langgraph.store.memory import InMemoryStore

from langgraph_agentic_scientist.agents.staging import stage_skills
from langgraph_agentic_scientist.investigation.context import MEMORY_HEADER, load_blocks
from langgraph_agentic_scientist.investigation.memory_sync import (
    append_memory_event_once,
    conclusions_key,
    delta_key,
    ensure_memory_files,
    format_memory_markdown_entry,
    memory_namespace,
    pull_shared_memory,
    push_shared_memory,
    stage_shared_view,
    sync_shared_back,
)
from langgraph_agentic_scientist.prompts import PromptRegistry

from .helpers import memory_line

BLOCKS = load_blocks(PromptRegistry.default())
NS = memory_namespace("run.1", 0)


async def test_ensure_creates_the_files_except_the_empty_jsonl(workspace: Any) -> None:
    await ensure_memory_files(workspace, BLOCKS, "w")
    assert (await workspace.read_bytes("w/memory.md")).decode() == MEMORY_HEADER
    assert (await workspace.read_bytes("w/conclusions.md")).decode() == "_No conclusions yet._\n"
    assert not await workspace.exists("w/memory.jsonl"), "so that write_file can create it"
    assert not await workspace.exists("w/conclusions_history.jsonl")
    assert NS == ("agentic_scientist", "run_1", "memory", "round_0")


async def test_pull_replaces_local_files_by_the_shared_copy_in_cycle_worker_order(
    workspace: Any,
) -> None:
    store = InMemoryStore()
    await store.aput(NS, delta_key(2, 1), {"jsonl_lines": ["c2w1"], "md_text": "### c2w1\n"})
    await store.aput(NS, delta_key(1, 2), {"jsonl_lines": ["c1w2\n"], "md_text": "### c1w2\n"})
    await store.aput(NS, delta_key(1, 1), {"jsonl_lines": ["c1w1"], "md_text": "### c1w1\n"})
    await store.aput(NS, conclusions_key(2), {"text": "other worker says: go low\n"})
    await store.aput(NS, conclusions_key(1), {"text": "my own, not mine to read back"})
    await ensure_memory_files(workspace, BLOCKS, "w")
    await workspace.write_bytes("w/memory.jsonl", b"stale local line\n")
    pre, sha = await pull_shared_memory(
        store, workspace, BLOCKS, run_id="run.1", round_index=0, root="w", worker_index=1
    )
    assert (await workspace.read_bytes("w/memory.jsonl")).decode() == "c1w1\nc1w2\nc2w1\n"
    assert (
        await workspace.read_bytes("w/memory.md")
    ).decode() == MEMORY_HEADER + "### c1w1\n### c1w2\n### c2w1\n"
    assert pre.jsonl_lines == 3 and pre.md_bytes == len(
        (MEMORY_HEADER + "### c1w1\n### c1w2\n### c2w1\n").encode()
    )
    others = (await workspace.read_bytes("w/conclusions_other_workers.md")).decode()
    assert others == "### From worker_02\n\nother worker says: go low\n"
    assert len(sha) == 64
    # pulling again changes nothing
    again, _ = await pull_shared_memory(
        store, workspace, BLOCKS, run_id="run.1", round_index=0, root="w", worker_index=1
    )
    assert again == pre


async def test_an_empty_store_leaves_no_jsonl_and_the_placeholder_for_other_workers(
    workspace: Any,
) -> None:
    await ensure_memory_files(workspace, BLOCKS, "w")
    pre, _ = await pull_shared_memory(
        InMemoryStore(), workspace, BLOCKS, run_id="r", round_index=0, root="w", worker_index=1
    )
    assert pre.jsonl_lines == 0 and not await workspace.exists("w/memory.jsonl")
    assert (await workspace.read_bytes("w/conclusions_other_workers.md")).decode() == (
        "_No conclusions from other workers yet._\n"
    )


async def test_push_sends_only_what_was_appended_and_twice_is_the_same(workspace: Any) -> None:
    store = InMemoryStore()
    await ensure_memory_files(workspace, BLOCKS, "w")
    await workspace.write_bytes("w/memory.jsonl", b'{"a": 1}\n')
    pre, sha = await pull_shared_memory(
        store, workspace, BLOCKS, run_id="r", round_index=0, root="w", worker_index=1
    )
    # pull replaced the stale line (the store is empty); the worker now appends during its cycle
    await workspace.write_bytes("w/memory.jsonl", b'{"b": 2}\n{"c": 3}\n')
    md = (await workspace.read_bytes("w/memory.md")) + b"### new entry\n"
    await workspace.write_bytes("w/memory.md", md)
    await workspace.write_bytes("w/conclusions.md", b"Prefer low rates.")
    for _ in range(2):
        wrote = await push_shared_memory(
            store,
            workspace,
            run_id="r",
            round_index=0,
            root="w",
            worker_index=1,
            global_cycle=4,
            pre=pre,
            conclusions_sha256=sha,
        )
        assert wrote
    items = {i.key: i.value for i in await store.asearch(memory_namespace("r", 0), limit=10)}
    assert sorted(items) == ["conclusions/01", "delta/0004/01"]
    assert items["delta/0004/01"]["jsonl_lines"] == ['{"b": 2}', '{"c": 3}']
    assert items["delta/0004/01"]["md_text"] == "### new entry\n"
    assert items["conclusions/01"]["text"] == "Prefer low rates."


async def test_nothing_appended_pushes_nothing(workspace: Any) -> None:
    store = InMemoryStore()
    await ensure_memory_files(workspace, BLOCKS, "w")
    pre, sha = await pull_shared_memory(
        store, workspace, BLOCKS, run_id="r", round_index=0, root="w", worker_index=1
    )
    assert not await push_shared_memory(
        store,
        workspace,
        run_id="r",
        round_index=0,
        root="w",
        worker_index=1,
        global_cycle=1,
        pre=pre,
        conclusions_sha256=sha,
    )
    assert await store.asearch(memory_namespace("r", 0), limit=10) == []


async def test_the_attempts_view_is_merged_back_once_however_often_it_runs(workspace: Any) -> None:
    await ensure_memory_files(workspace, BLOCKS, "w")
    await workspace.write_bytes("w/memory.jsonl", b'{"old": 1}\n')
    sizes = await stage_shared_view(workspace, "w", "w/ideas/x/_shared_memory")
    view = "w/ideas/x/_shared_memory/memory.jsonl"
    await workspace.write_bytes(view, b'{"old": 1}\n{"new": 2}\n')
    await workspace.write_bytes("w/ideas/x/_shared_memory/conclusions.md", b"Changed.")
    for _ in range(2):
        await sync_shared_back(workspace, "w", "w/ideas/x/_shared_memory", sizes)
    assert (await workspace.read_bytes("w/memory.jsonl")).decode() == '{"old": 1}\n{"new": 2}\n'
    assert (await workspace.read_bytes("w/conclusions.md")).decode() == "Changed."


async def test_a_view_the_agent_rewrote_keeps_its_lines_without_duplicating_old_ones(
    workspace: Any,
) -> None:
    await ensure_memory_files(workspace, BLOCKS, "w")
    await workspace.write_bytes("w/memory.jsonl", b'{"old": 1}\n')
    sizes = await stage_shared_view(workspace, "w", "w/v")
    await workspace.write_bytes(
        "w/v/memory.jsonl", b'{"new": 2}\n{"old": 1}\n'
    )  # write_file overwrote
    await workspace.write_bytes("w/v/memory.md", b"# Memory Log\n- fresh md\n")
    for _ in range(2):
        await sync_shared_back(workspace, "w", "w/v", sizes)
    assert (await workspace.read_bytes("w/memory.jsonl")).decode() == '{"old": 1}\n{"new": 2}\n'
    md = (await workspace.read_bytes("w/memory.md")).decode()
    assert md.startswith(MEMORY_HEADER) and md.count("- fresh md") == 1


async def test_the_hosts_own_event_is_appended_once(workspace: Any) -> None:
    await ensure_memory_files(workspace, BLOCKS, "w")
    await workspace.write_bytes("w/memory.jsonl", memory_line("a1", None, step="idea").encode())
    event = {
        "step": "evaluate",
        "scope": "attempt",
        "summary": "s",
        "evidence": "e",
        "next_action": "n",
        "idea_id": "i",
        "attempt_id": "a1",
        "program_id": "program_1",
        "timestamp": "t",
    }
    markdown = format_memory_markdown_entry(event)
    marker = "- attempt_id: a1\n- program_id: program_1\n"
    assert await append_memory_event_once(workspace, "w", event, markdown=markdown, marker=marker)
    assert not await append_memory_event_once(
        workspace, "w", event, markdown=markdown, marker=marker
    )
    lines = (await workspace.read_bytes("w/memory.jsonl")).decode().splitlines()
    assert [json.loads(x).get("program_id") for x in lines] == [None, "program_1"]
    assert (await workspace.read_bytes("w/memory.md")).decode().count("program_id: program_1") == 1


async def test_the_staged_helper_creates_and_appends_the_memory_files(workspace: Any) -> None:
    await stage_skills(
        workspace, "root", ["memory-log"], prompts=PromptRegistry.default(), values={}
    )
    event = json.dumps(
        {
            "step": "edit",
            "scope": "attempt",
            "summary": "s",
            "evidence": "e",
            "next_action": "n",
            "attempt_id": "a1",
            "program_id": "program_1",
        }
    )
    result = await workspace.run(
        [
            sys.executable,
            ".agents/skills/memory-log/scripts/append_memory.py",
            "--global-memory-md",
            "memory.md",
            "--global-memory-jsonl",
            "memory.jsonl",
            "--event-json",
            event,
        ],
        cwd="root",
    )
    assert result.returncode == 0, result.stderr
    record = json.loads((await workspace.read_bytes("root/memory.jsonl")).decode().splitlines()[0])
    assert record["attempt_id"] == "a1" and "timestamp" in record
    assert "program_id: program_1" in (await workspace.read_bytes("root/memory.md")).decode()
