"""A pipeline run, interrupted and resumed, on a checkpointer that applies strict msgpack.

Usage: ``LANGGRAPH_STRICT_MSGPACK=true python strict_msgpack_run.py <workspace> <task dir>``. Prints
the run's status and the status of each phase as JSON.
"""

from __future__ import annotations

import asyncio
import json
import sys

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde import _msgpack
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from pipeline_stubs import Ledger, stub_overrides

from langgraph_agentic_scientist import (
    LocalWorkspace,
    ScientistConfig,
    ScientistContext,
    build_pipeline_graph,
    serde_allowlist,
)
from langgraph_agentic_scientist.config import ALL_PHASES
from langgraph_agentic_scientist.testing import ScriptedModels


async def main(workspace: str, task_dir: str) -> None:
    assert _msgpack.STRICT_MSGPACK_ENABLED, "run with LANGGRAPH_STRICT_MSGPACK=true"
    ledger = Ledger()
    ledger.pause.add("paper")
    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist()))
    graph = build_pipeline_graph(checkpointer=saver, overrides=stub_overrides(ledger))
    context = ScientistContext(
        models=ScriptedModels.from_scripts({}),
        workspace=LocalWorkspace(workspace),
        config=ScientistConfig(phases=ALL_PHASES),
    )
    config = {"configurable": {"thread_id": "strict"}}
    out = await graph.ainvoke({"task_dir": task_dir}, config, context=context)
    assert "__interrupt__" in out, "the run should stop inside the paper phase"
    state = await graph.aget_state(
        config
    )  # reads the checkpoint back through the strict serializer
    assert state.values["task"].root == "task" and state.values["final"].outcome == "qualified"
    out = await graph.ainvoke(Command(resume="go"), config, context=context)
    print(
        json.dumps(
            {
                "status": out["status"],
                "phases": {name: status.status for name, status in out["phase_status"].items()},
                "calls": dict(ledger.calls),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
