"""Run the whole pipeline on the vendored circle-packing example.

``python examples/quickstart.py`` needs a chat model: set ``LAS_MODEL`` to any model name that
LangChain's ``init_chat_model`` accepts (for example ``provider:model``) and the provider's own
credentials. The snippet between the markers below is the one the README shows; ``run`` takes any
chat model, which is how the tests run it with a scripted one.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langgraph_agentic_scientist import (
    GraphOverrides,
    LocalCommandEvaluator,
    LocalFinalEvaluator,
    LocalWorkspace,
    ModelRegistry,
    ScientistConfig,
    ScientistContext,
    build_pipeline_graph,
)

EXAMPLE = Path(__file__).resolve().parent / "circpack"


# --8<-- [start:example]
async def run(
    model: BaseChatModel,
    workdir: Path,
    *,
    config: ScientistConfig | None = None,
    overrides: GraphOverrides | None = None,
) -> dict[str, Any]:
    """Run the pipeline on the circle-packing task and return its output."""
    # The workspace is the run's directory; the task to study is copied into it.
    workspace = LocalWorkspace(workdir / "run")
    shutil.copytree(EXAMPLE, workspace.root / "inputs" / "circpack")

    context = ScientistContext(
        models=ModelRegistry(default=model),  # any LangChain chat model
        workspace=workspace,
        evaluator=LocalCommandEvaluator(workspace),
        final_evaluator=LocalFinalEvaluator(workspace),
        config=config or ScientistConfig.preset("paper"),  # upstream's campaign preset
    )
    graph = build_pipeline_graph(checkpointer=InMemorySaver(), overrides=overrides)
    run_config: RunnableConfig = {"configurable": {"thread_id": "circpack-1"}}

    out: dict[str, Any] = await graph.ainvoke(
        {"task_dir": "inputs/circpack"}, run_config, context=context
    )
    # The graph stops with an interrupt wherever upstream would ask a person (its only question
    # is the prune warning). Resume with the same thread id; Command(resume=...) is the answer.
    while out.get("__interrupt__"):
        out = await graph.ainvoke(Command(resume="proceed"), run_config, context=context)
    return out


# --8<-- [end:example]


def main() -> None:
    """Run the example with the model named by ``LAS_MODEL``."""
    from langchain.chat_models import init_chat_model

    name = os.environ.get("LAS_MODEL")
    if not name:
        sys.exit("set LAS_MODEL to a model name for langchain's init_chat_model")
    out = asyncio.run(run(init_chat_model(name), Path("runs")))
    print("status:", out["status"])
    for phase, status in out["phase_status"].items():
        print(f"  {phase}: {status.status}" + (f" ({status.reason})" if status.reason else ""))


if __name__ == "__main__":
    main()
