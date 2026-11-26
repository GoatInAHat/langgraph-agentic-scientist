"""The whole method stopped in the middle and resumed from its checkpoint, with no step done twice.

The run of ``test_whole_flow`` is made to crash once at a step deep inside a phase (a ``before``
hook that raises the first time it runs, standing for a process that dies there). A second
process (a new context: new scripted models, new evaluators, and for SQLite a new connection to
the same file) resumes the thread with ``ainvoke(None, ...)``. The model calls of every role and
the evaluations of both processes together must equal those of a run that never crashed: nothing
that completed before the crash runs again.
"""

from __future__ import annotations

import collections
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.store.memory import InMemoryStore
from testkit import thread
from whole_flow import context, policies, write_task

from langgraph_agentic_scientist import (
    Hooks,
    LocalWorkspace,
    ScientistContext,
    build_pipeline_graph,
    serde_allowlist,
)

CRASH_POINTS = (
    "worker.record_attempt",  # inside a worker, inside the investigation
    "variants.aggregate",
    "final_evaluation.evaluate_sealed",
    "posthoc.publish",
    "paper.audit",
    "review.aggregate",
    "revision.promote",
)

Ledger = collections.Counter[str]


class Crash(RuntimeError):
    """The process dies here."""


def crash_once(key: str) -> Hooks:
    """Hooks that raise :class:`Crash` the first time the step ``key`` is about to run."""
    fired: list[bool] = []

    async def hook(state: Any, runtime: Any) -> None:
        if not fired:
            fired.append(True)
            raise Crash(key)

    return Hooks(before={key: [hook]})


def ledger(ctx: ScientistContext) -> Ledger:
    """What a process did: model calls by role, evaluations, sealed evaluations."""
    counts: Ledger = collections.Counter()
    for role in policies():
        counts[f"model:{role}"] = len(ctx.models.scripted(role).calls)  # type: ignore[attr-defined]
    counts["evaluations"] = len(ctx.evaluator.calls)  # type: ignore[union-attr]
    counts["sealed_evaluations"] = ctx.final_evaluator.runs  # type: ignore[union-attr]
    return counts


Saver = Callable[[Path], AbstractAsyncContextManager[Any]]


_MEMORY: dict[Path, InMemorySaver] = {}


@asynccontextmanager
async def memory_saver(path: Path) -> AsyncIterator[Any]:
    """One in-memory saver per path, kept across 'processes' as a database would be."""
    yield _MEMORY.setdefault(path, InMemorySaver())


@asynccontextmanager
async def sqlite_saver(path: Path) -> AsyncIterator[Any]:
    """A SQLite saver on a file, with the package's serializer allowlist (a new connection each
    time, as a restarted process would open)."""
    aiosqlite = pytest.importorskip("aiosqlite")
    sqlite = pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    async with aiosqlite.connect(str(path)) as conn:
        saver = sqlite.AsyncSqliteSaver(
            conn, serde=JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist())
        )
        await saver.setup()
        yield saver


async def clean_run(tmp: Path) -> Ledger:
    workspace = LocalWorkspace(tmp / "clean")
    task_dir = await write_task(workspace)
    ctx = context(workspace)
    app = build_pipeline_graph(checkpointer=InMemorySaver(), store=InMemoryStore())
    out = await app.ainvoke({"task_dir": task_dir}, thread("clean"), context=ctx)
    assert out["status"] == "completed", out["phase_status"]
    return ledger(ctx)


async def crashed_and_resumed(tmp: Path, saver: Saver, key: str) -> tuple[Ledger, dict[str, Any]]:
    workspace = LocalWorkspace(tmp / "crash")
    task_dir = await write_task(workspace)
    database = tmp / "checkpoints.sqlite"
    store = InMemoryStore()

    first = context(workspace, hooks=crash_once(key))
    async with saver(database) as checkpointer:
        app = build_pipeline_graph(checkpointer=checkpointer, store=store)
        with pytest.raises(Crash):
            await app.ainvoke({"task_dir": task_dir}, thread("crash"), context=first)

    second = context(workspace)
    async with saver(database) as checkpointer:
        app = build_pipeline_graph(checkpointer=checkpointer, store=store)
        out: dict[str, Any] = await app.ainvoke(None, thread("crash"), context=second)
    return ledger(first) + ledger(second), out


@pytest.mark.parametrize("key", CRASH_POINTS)
async def test_a_run_resumed_after_a_crash_repeats_nothing(tmp_path: Path, key: str) -> None:
    expected = await clean_run(tmp_path)
    done, out = await crashed_and_resumed(tmp_path, memory_saver, key)
    assert out["status"] == "completed", out["phase_status"]
    assert done == expected


async def test_a_run_resumed_from_sqlite_in_a_new_process_repeats_nothing(tmp_path: Path) -> None:
    expected = await clean_run(tmp_path)
    done, out = await crashed_and_resumed(tmp_path, sqlite_saver, "posthoc.publish")
    assert out["status"] == "completed", out["phase_status"]
    assert done == expected
