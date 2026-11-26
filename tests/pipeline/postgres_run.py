"""One process of the Postgres test: run or resume a stubbed pipeline on a Postgres checkpointer.

Usage: ``python postgres_run.py <first|resume> <postgres url> <thread id> <workspace> <task dir>``.
Every phase stub appends its name to ``calls.log`` in the workspace's parent directory; in the
first process the stub of ``posthoc`` kills the process (``os._exit``) instead of returning, which
is a crash with nothing flushed. Run it with ``LANGGRAPH_STRICT_MSGPACK=true``.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde import _msgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pipeline_stubs import Ledger, default_results, stub_phase

from langgraph_agentic_scientist import (
    GraphOverrides,
    LocalWorkspace,
    ModelRegistry,
    ScientistConfig,
    ScientistContext,
    build_pipeline_graph,
    serde_allowlist,
)
from langgraph_agentic_scientist.config import ALL_PHASES
from langgraph_agentic_scientist.testing import ScriptedModels


def _logged(name: str, mode: str, log: Path, value: Any) -> Any:
    def run(given: Any) -> Any:
        with log.open("a") as handle:
            handle.write(name + "\n")
        if mode == "first" and name == "posthoc":
            os._exit(9)
        return value(given) if callable(value) else value

    return run


async def main(mode: str, url: str, thread_id: str, workspace: str, task_dir: str) -> None:
    assert _msgpack.STRICT_MSGPACK_ENABLED, "run with LANGGRAPH_STRICT_MSGPACK=true"
    log = Path(workspace).parent / "calls.log"
    ledger = Ledger()
    phases = {
        name: stub_phase(name, ledger, _logged(name, mode, log, value))
        for name, value in default_results().items()
    }
    serde = JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist())
    models: ModelRegistry = ScriptedModels.from_scripts({})
    context = ScientistContext(
        models=models,
        workspace=LocalWorkspace(workspace),
        config=ScientistConfig(phases=ALL_PHASES),
    )
    config = {"configurable": {"thread_id": thread_id}}
    async with AsyncPostgresSaver.from_conn_string(url, serde=serde) as saver:
        await saver.setup()
        graph = build_pipeline_graph(checkpointer=saver, overrides=GraphOverrides(phases=phases))
        if mode == "first":
            await graph.ainvoke({"task_dir": task_dir}, config, context=context, durability="sync")
            raise SystemExit("the first process was meant to be killed")
        out = await graph.ainvoke(None, config, context=context, durability="sync")
        summary = {
            "status": out["status"],
            "phases": {name: status.status for name, status in out["phase_status"].items()},
        }
        print(json.dumps(summary))


if __name__ == "__main__":
    mode, url, thread_id, workspace, task_dir = sys.argv[1:6]
    asyncio.run(main(mode, url, thread_id, workspace, task_dir))
