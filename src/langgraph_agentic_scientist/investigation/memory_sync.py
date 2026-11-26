"""Memory files and their sharing between workers.

Memory is the agents' files (design decision D4): ``memory.md``, ``memory.jsonl``,
``conclusions.md`` and ``conclusions_history.jsonl`` in a worker's results root are appended to
by the agents themselves, directly or through the staged ``append_memory.py`` helper, and the
host reads them; it never rewrites them from state.

Parallel workers share memory as upstream's ``sync_memory_*`` and ``sync_conclusions_*`` do
(``orchestrator/state.py``): before a cycle the worker's memory files are replaced by the shared
copy, after the cycle what the worker appended beyond the pre-cycle state is appended to the
shared copy. The shared copy lives in the caller's :class:`~langgraph.store.base.BaseStore`, in
the namespace ``("agentic_scientist", <run id>, "memory", "round_<n>")``, under keys that make
every write idempotent: ``delta/<cycle:04d>/<worker:02d>`` holds one worker's appended lines and
bytes of one cycle, ``conclusions/<worker:02d>`` the worker's current conclusions. A push that
runs twice (after a crash) overwrites the same key. The shared order is (cycle, worker).

An attempt's agent works inside its experiment directory, so the host stages a view of the
worker's memory files there (:func:`stage_shared_view`) and merges what the agent appended back
(:func:`sync_shared_back`), idempotently.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langgraph.store.base import BaseStore

from ..agents.memory import MemoryPreState, memory_delta, memory_pre_state
from ..core.errors import ConfigurationMismatch
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .context import (
    CONCLUSIONS_FILE,
    CONCLUSIONS_HISTORY_FILE,
    GLOBAL_MEMORY_JSONL_FILE,
    GLOBAL_MEMORY_MD_FILE,
    MEMORY_HEADER,
    OTHER_WORKERS_CONCLUSIONS_FILE,
    load_blocks,
)

NAMESPACE_ROOT = "agentic_scientist"
PAGE = 500


class StoreRequired(ConfigurationMismatch):
    """Parallel workers share memory through a store, and the run was given none."""

    def __init__(self) -> None:
        super().__init__(
            [
                "runtime.store: workers > 1 with shared memory needs a BaseStore "
                "(compile the graph with store=..., or set isolate_worker_memory)"
            ]
        )


def _label(value: object) -> str:
    return str(value).replace(".", "_").replace("/", "_") or "_"


def memory_namespace(run_id: str, round_index: int) -> tuple[str, ...]:
    """Return the store namespace of the shared memory of one round of a run."""
    return (NAMESPACE_ROOT, _label(run_id), "memory", f"round_{round_index}")


def delta_key(cycle: int, worker: int) -> str:
    """Return the key of a worker's memory delta of a cycle."""
    return f"delta/{cycle:04d}/{worker:02d}"


def conclusions_key(worker: int) -> str:
    """Return the key of a worker's conclusions."""
    return f"conclusions/{worker:02d}"


async def _read(workspace: Workspace, path: str) -> bytes:
    try:
        return await workspace.read_bytes(path)
    except FileNotFoundError:
        return b""


async def ensure_memory_files(
    workspace: Workspace, prompts_blocks: dict[str, Any], root: str
) -> None:
    """Create the memory files of a results root when missing (upstream ``ensure_*_files``).

    ``memory.md`` gets its header and ``conclusions.md`` and ``conclusions_other_workers.md``
    their placeholders. ``memory.jsonl`` and the conclusions history stay absent until the first
    write: upstream creates ``memory.jsonl`` empty, but Deep Agents' ``write_file`` refuses to
    overwrite a file and ``edit_file`` cannot edit an empty one, so an empty file would leave
    an agent without file tools no way to start it (the staged helper creates it as well).
    """
    md_path = join(root, GLOBAL_MEMORY_MD_FILE)
    if not await workspace.exists(md_path):
        await workspace.write_bytes(md_path, MEMORY_HEADER.encode("utf-8"))
    conclusions = prompts_blocks["conclusions"]
    own = join(root, CONCLUSIONS_FILE)
    other = join(root, OTHER_WORKERS_CONCLUSIONS_FILE)
    if not await workspace.exists(own):
        await workspace.write_bytes(own, (str(conclusions["own_fallback"]) + "\n").encode("utf-8"))
    if not await workspace.exists(other):
        await workspace.write_bytes(
            other, (str(conclusions["other_workers_placeholder"]) + "\n").encode("utf-8")
        )


async def _all_items(store: BaseStore, namespace: tuple[str, ...]) -> list[Any]:
    items: list[Any] = []
    offset = 0
    while True:
        page = await store.asearch(namespace, limit=PAGE, offset=offset)
        items.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    return sorted(items, key=lambda item: item.key)


async def pull_shared_memory(
    store: BaseStore | None,
    workspace: Workspace,
    blocks: dict[str, Any],
    *,
    run_id: str,
    round_index: int,
    root: str,
    worker_index: int,
) -> tuple[MemoryPreState, str]:
    """Replace the worker's memory files by the shared copy; return the pre-cycle state.

    The returned state holds the JSONL line count, the byte size of ``memory.md`` and of the
    worker's conclusions, and the SHA-256 of those conclusions. The node that calls this writes
    only files it fully replaces, so running it again after a crash changes nothing.

    Raises:
        StoreRequired: When there is no store.
    """
    if store is None:
        raise StoreRequired()
    items = await _all_items(store, memory_namespace(run_id, round_index))
    lines: list[str] = []
    md_parts: list[str] = []
    sections: list[str] = []
    own_name = conclusions_key(worker_index)
    for item in items:
        value = item.value
        if item.key.startswith("delta/"):
            lines.extend(str(line) for line in value.get("jsonl_lines", []))
            md_parts.append(str(value.get("md_text", "")))
        elif item.key.startswith("conclusions/") and item.key != own_name:
            body = str(value.get("text", "")).strip()
            if body:
                name = "worker_" + item.key.rsplit("/", 1)[-1]
                sections.append(f"### From {name}\n\n{body}\n")
    jsonl = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    md = MEMORY_HEADER + "".join(md_parts)
    if jsonl:
        await workspace.write_bytes(join(root, GLOBAL_MEMORY_JSONL_FILE), jsonl.encode("utf-8"))
    else:
        await workspace.remove(join(root, GLOBAL_MEMORY_JSONL_FILE))
    await workspace.write_bytes(join(root, GLOBAL_MEMORY_MD_FILE), md.encode("utf-8"))
    other = (
        "\n".join(sections)
        if sections
        else str(blocks["conclusions"]["other_workers_placeholder"]) + "\n"
    )
    await workspace.write_bytes(join(root, OTHER_WORKERS_CONCLUSIONS_FILE), other.encode("utf-8"))
    pre = await memory_pre_state(
        workspace,
        jsonl_path=join(root, GLOBAL_MEMORY_JSONL_FILE),
        md_path=join(root, GLOBAL_MEMORY_MD_FILE),
        conclusions_path=join(root, CONCLUSIONS_FILE),
    )
    own = await _read(workspace, join(root, CONCLUSIONS_FILE))
    return pre, hashlib.sha256(own).hexdigest()


async def push_shared_memory(
    store: BaseStore | None,
    workspace: Workspace,
    *,
    run_id: str,
    round_index: int,
    root: str,
    worker_index: int,
    global_cycle: int,
    pre: MemoryPreState,
    conclusions_sha256: str,
) -> bool:
    """Push what the worker appended during a cycle; return whether anything was written.

    The lines and bytes appended beyond ``pre`` go to one item ``delta/<cycle>/<worker>`` and the
    worker's conclusions, when they changed, to ``conclusions/<worker>``. Both keys are
    deterministic, so a push that runs again overwrites them and nothing is duplicated.

    Raises:
        StoreRequired: When there is no store.
    """
    if store is None:
        raise StoreRequired()
    namespace = memory_namespace(run_id, round_index)
    delta = await memory_delta(
        workspace,
        jsonl_path=join(root, GLOBAL_MEMORY_JSONL_FILE),
        md_path=join(root, GLOBAL_MEMORY_MD_FILE),
        pre=pre,
    )
    wrote = False
    if delta.jsonl_lines or delta.md_text:
        await store.aput(
            namespace,
            delta_key(global_cycle, worker_index),
            {
                "worker": worker_index,
                "cycle": global_cycle,
                "jsonl_lines": delta.jsonl_lines,
                "md_text": delta.md_text,
            },
        )
        wrote = True
    own = await _read(workspace, join(root, CONCLUSIONS_FILE))
    if own and hashlib.sha256(own).hexdigest() != conclusions_sha256:
        await store.aput(
            namespace,
            conclusions_key(worker_index),
            {"worker": worker_index, "text": own.decode("utf-8", errors="replace")},
        )
        wrote = True
    return wrote


# --- the attempt's view of the memory files ----------------------------------------------------

_APPEND_ONLY = (GLOBAL_MEMORY_MD_FILE, GLOBAL_MEMORY_JSONL_FILE, CONCLUSIONS_HISTORY_FILE)
_COPIED = (CONCLUSIONS_FILE, OTHER_WORKERS_CONCLUSIONS_FILE)


async def stage_shared_view(workspace: Workspace, root: str, shared_dir: str) -> dict[str, int]:
    """Copy the worker's memory files into an attempt's ``shared_dir``; return their sizes.

    The sizes (and the digest of ``conclusions.md``, under ``conclusions.sha``) are what
    :func:`sync_shared_back` needs to merge the agent's additions back.
    """
    sizes: dict[str, int] = {}
    for name in (*_APPEND_ONLY, *_COPIED):
        data = await _read(workspace, join(root, name))
        if data or await workspace.exists(join(root, name)):
            await workspace.write_bytes(join(shared_dir, name), data)
        sizes[name] = len(data)
        if name == CONCLUSIONS_FILE:
            sizes["conclusions.sha"] = int(hashlib.sha256(data).hexdigest()[:15], 16)
    return sizes


def _merge_added(name: str, prefix: bytes, view: bytes) -> bytes:
    """Return what the agent added to its view of an append-only file beyond the staged prefix.

    Normally the view extends the prefix. An agent whose file tool rewrote the whole file
    (Deep Agents' ``write_file`` overwrites) leaves a view that does not start with the prefix;
    then the lines (JSONL) or the text (Markdown) that the prefix does not already hold count as
    added, so what the agent wrote is kept and nothing already recorded is duplicated.
    """
    if view.startswith(prefix):
        return view[len(prefix) :]
    if name.endswith(".jsonl"):
        known = set(prefix.decode("utf-8", errors="replace").splitlines())
        fresh = [
            line
            for line in view.decode("utf-8", errors="replace").splitlines()
            if line.strip() and line not in known
        ]
        return "".join(line + "\n" for line in fresh).encode("utf-8")
    text = view.decode("utf-8", errors="replace")
    if not text.strip() or text in prefix.decode("utf-8", errors="replace"):
        return b""
    return (text if text.endswith("\n") else text + "\n").encode("utf-8")


async def sync_shared_back(
    workspace: Workspace, root: str, shared_dir: str, sizes: dict[str, int]
) -> None:
    """Merge what an attempt's agent added to its view of the memory files into the worker's.

    Append-only files become ``<worker file up to the staged size> + <what the agent added to
    its view>``, which gives the same result however often it runs (see :func:`_merge_added`
    for a view the agent rewrote). ``conclusions.md`` is copied back when the agent changed it.
    """
    for name in _APPEND_ONLY:
        view = await _read(workspace, join(shared_dir, name))
        if not view:
            continue
        staged = sizes.get(name, 0)
        current = await _read(workspace, join(root, name))
        prefix = current[:staged]
        added = _merge_added(name, prefix, view)
        if added:
            await workspace.write_bytes(join(root, name), prefix + added)
    view = await _read(workspace, join(shared_dir, CONCLUSIONS_FILE))
    if view and int(hashlib.sha256(view).hexdigest()[:15], 16) != sizes.get("conclusions.sha"):
        await workspace.write_bytes(join(root, CONCLUSIONS_FILE), view)


def format_memory_markdown_entry(event: dict[str, Any]) -> str:
    """Render one Markdown memory entry from a structured event (upstream's format)."""
    timestamp = str(event.get("timestamp", "")).strip() or "unknown_ts"
    step = str(event.get("step", "")).strip() or "unknown_step"
    scope = str(event.get("scope", "")).strip() or "unknown_scope"
    summary = str(event.get("summary", "")).strip() or "(missing summary)"
    evidence = str(event.get("evidence", "")).strip() or "(missing evidence)"
    next_action = str(event.get("next_action", "")).strip() or "(missing next_action)"
    lines = [
        f"### {timestamp} | step={step} | scope={scope}",
        f"- summary: {summary}",
        f"- evidence: {evidence}",
        f"- next_action: {next_action}",
    ]
    for key in ("idea_id", "attempt_id", "program_id"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"- {key}: {value.strip()}")
    metrics = event.get("metrics")
    if isinstance(metrics, dict) and metrics:
        lines.append(f"- metrics: {json.dumps(metrics, sort_keys=True)}")
    rubrics = event.get("rubrics")
    if isinstance(rubrics, list) and rubrics:
        lines.append(f"- rubrics: {json.dumps(rubrics, sort_keys=True)}")
    tags = event.get("tags")
    if isinstance(tags, list) and tags:
        lines.append(f"- tags: {', '.join(str(item) for item in tags)}")
    return "\n".join(lines) + "\n\n"


async def append_memory_event_once(
    workspace: Workspace,
    root: str,
    event: dict[str, Any],
    *,
    markdown: str,
    match_keys: tuple[str, ...] = ("step", "attempt_id", "program_id"),
    marker: str,
) -> bool:
    """Append one host memory event to ``memory.jsonl`` and ``memory.md`` exactly once.

    The event is identified by the values of ``match_keys``; when the JSONL already holds a
    record with those, nothing is appended to it, and the Markdown entry is appended only when
    ``marker`` (a line of the entry) is not in ``memory.md`` yet. The call is therefore safe to
    repeat after a crash. Returns whether anything was appended.
    """
    jsonl_path = join(root, GLOBAL_MEMORY_JSONL_FILE)
    md_path = join(root, GLOBAL_MEMORY_MD_FILE)
    jsonl = (await _read(workspace, jsonl_path)).decode("utf-8", errors="replace")
    wrote = False
    if not any(_matches(line, event, match_keys) for line in jsonl.splitlines() if line.strip()):
        prefix = jsonl if not jsonl or jsonl.endswith("\n") else jsonl + "\n"
        await workspace.write_bytes(
            jsonl_path, (prefix + json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
        )
        wrote = True
    md = (await _read(workspace, md_path)).decode("utf-8", errors="replace")
    if marker not in md:
        await workspace.write_bytes(md_path, (md + markdown).encode("utf-8"))
        wrote = True
    return wrote


def _matches(line: str, event: dict[str, Any], keys: tuple[str, ...]) -> bool:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return False
    return isinstance(record, dict) and all(record.get(key) == event.get(key) for key in keys)


__all__ = [
    "NAMESPACE_ROOT",
    "StoreRequired",
    "append_memory_event_once",
    "conclusions_key",
    "delta_key",
    "ensure_memory_files",
    "format_memory_markdown_entry",
    "load_blocks",
    "memory_namespace",
    "pull_shared_memory",
    "push_shared_memory",
    "stage_shared_view",
    "sync_shared_back",
]
