"""Reading the memory files agents write (design decision D4).

Agents append to their memory files themselves, directly with their file tools or through the
staged ``append_memory.py`` helper, exactly as upstream's prompts and the ``memory-log`` skill
say. The host never rewrites those files from state; it reads them: to check that an attempt left
the records upstream requires (``run.sh`` post-checks), and to compute what a worker appended
during a cycle, which is pushed to the other workers (upstream ``sync_memory_after_cycle``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..core.state import StateModel
from ..workspace.protocol import Workspace


class MemoryPreState(StateModel):
    """Sizes of the memory files before a cycle (upstream's pre-cycle sync state).

    Attributes:
        jsonl_lines: Line count of the global ``memory.jsonl``.
        md_bytes: Byte size of the global ``memory.md``.
        conclusions_bytes: Byte size of the worker's own ``conclusions.md``.
    """

    jsonl_lines: int = 0
    md_bytes: int = 0
    conclusions_bytes: int = 0


class MemoryDelta(StateModel):
    """What a worker appended to the memory files beyond a :class:`MemoryPreState`."""

    jsonl_lines: list[str]
    md_text: str


async def _read_or_empty(workspace: Workspace, path: str) -> bytes:
    try:
        return await workspace.read_bytes(path)
    except FileNotFoundError:
        return b""


def _lines(data: bytes) -> list[str]:
    return [line for line in data.decode("utf-8", errors="replace").splitlines() if line.strip()]


async def memory_pre_state(
    workspace: Workspace,
    *,
    jsonl_path: str,
    md_path: str,
    conclusions_path: str | None = None,
) -> MemoryPreState:
    """Record the line count and byte sizes of the memory files now."""
    jsonl = await _read_or_empty(workspace, jsonl_path)
    md = await _read_or_empty(workspace, md_path)
    conclusions = await _read_or_empty(workspace, conclusions_path) if conclusions_path else b""
    return MemoryPreState(
        jsonl_lines=len(_lines(jsonl)), md_bytes=len(md), conclusions_bytes=len(conclusions)
    )


async def memory_delta(
    workspace: Workspace, *, jsonl_path: str, md_path: str, pre: MemoryPreState
) -> MemoryDelta:
    """Return the JSONL lines and Markdown text appended beyond ``pre``."""
    jsonl = await _read_or_empty(workspace, jsonl_path)
    md = await _read_or_empty(workspace, md_path)
    return MemoryDelta(
        jsonl_lines=_lines(jsonl)[pre.jsonl_lines :],
        md_text=md[pre.md_bytes :].decode("utf-8", errors="replace"),
    )


def parse_memory_records(lines: Iterable[str]) -> list[dict[str, Any]]:
    """Parse JSONL memory lines; lines that are not JSON objects are skipped."""
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


async def read_memory_records(workspace: Workspace, jsonl_path: str) -> list[dict[str, Any]]:
    """Read every JSON-object record of a ``memory.jsonl``."""
    return parse_memory_records(_lines(await _read_or_empty(workspace, jsonl_path)))


def has_record(
    records: Iterable[Mapping[str, Any]],
    *,
    attempt_id: str,
    program_id: str | None = None,
) -> bool:
    """Return whether a record carries ``attempt_id`` (and ``program_id`` on the same record).

    Upstream's post-check requires the exact ``attempt_id`` and, for each evaluated program, the
    exact ``program_id`` on the same record; substitutes such as ``attempt`` or ``best_program``
    do not count.
    """
    for record in records:
        if record.get("attempt_id") != attempt_id:
            continue
        if program_id is None or record.get("program_id") == program_id:
            return True
    return False


__all__ = [
    "MemoryDelta",
    "MemoryPreState",
    "has_record",
    "memory_delta",
    "memory_pre_state",
    "parse_memory_records",
    "read_memory_records",
]
