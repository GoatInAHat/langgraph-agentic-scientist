"""Run the whole pipeline on the circle-packing task against a served model, durably.

The model is any OpenAI-compatible endpoint; the checkpointer is Postgres; the run resumes from
its checkpoint when it is started again with the same thread id. Everything comes from the
environment:

``LAS_BASE_URL``, ``LAS_MODEL`` and ``LAS_API_KEY_FILE`` (or ``OPENAI_API_KEY``) name the model;
``LAS_CONTEXT_WINDOW`` and ``LAS_MAX_TOKENS`` its window and output limit;
``LAS_REASONING_EFFORT`` is sent as ``chat_template_kwargs.reasoning_effort`` when set;
``LAS_POSTGRES_URI`` is the checkpointer's database; ``LAS_WORKDIR`` the run directory;
``LAS_THREAD_ID`` the thread; ``LAS_MAX_CONCURRENCY`` bounds parallel branches (default 2);
``LAS_SCALE`` is ``small`` (the scaled-down proof configuration) or ``paper`` (upstream's campaign
preset). With ``LANGFUSE_HOST``, ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` set, the run
is traced to Langfuse under the session ``LAS_TRACE_SESSION`` (default: the thread id). Every
model call's usage is appended to ``usage.jsonl`` in the run directory.

The process runs agent commands on its own host through ``LocalWorkspace``: run it inside a
container whose only writable mount is the run directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from langgraph_agentic_scientist import (
    AutoApprovePolicy,
    LocalCommandEvaluator,
    LocalFinalEvaluator,
    LocalWorkspace,
    ModelRegistry,
    ScientistConfig,
    ScientistContext,
    build_pipeline_graph,
    serde_allowlist,
)

EXAMPLE = Path(__file__).resolve().parent / "circpack"


def small_config() -> ScientistConfig:
    """Upstream's campaign preset with every count at the smallest value it admits."""
    paper = ScientistConfig.preset("paper")
    return paper.model_copy(
        update={
            "allow_unisolated_contracted": True,
            "investigation": paper.investigation.model_copy(
                update={"max_idea_cycles": 1, "workers": 1, "max_program_evals_per_attempt": 1}
            ),
            "variants": paper.variants.model_copy(
                update={"max_ablations": 1, "max_extra_baselines": 1}
            ),
            "posthoc": paper.posthoc.model_copy(update={"max_selected_analyses": 1}),
            "paper": paper.paper.model_copy(update={"min_refs": 5, "max_refs": 10}),
            "review": paper.review.model_copy(update={"num_reviewers": 2}),
        }
    )


class UsageLog(AsyncCallbackHandler):
    """Append one line per model call: its graph path, duration and token usage."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.started: dict[UUID, tuple[float, str]] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        namespace = str((metadata or {}).get("langgraph_checkpoint_ns", ""))
        self.started[run_id] = (time.time(), namespace)

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        start, namespace = self.started.pop(run_id, (time.time(), ""))
        usage: dict[str, Any] = {}
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                usage = dict(getattr(message, "usage_metadata", None) or {}) or usage
        record = {"t": time.time(), "seconds": time.time() - start, "ns": namespace, **usage}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        start, namespace = self.started.pop(run_id, (time.time(), ""))
        record = {"t": time.time(), "seconds": time.time() - start, "ns": namespace}
        record["error"] = f"{type(error).__name__}: {error}"[:500]
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def chat_model() -> Any:
    """The served model, from the environment."""
    from langchain_openai import ChatOpenAI

    key_file = os.environ.get("LAS_API_KEY_FILE")
    api_key = Path(key_file).read_text().strip() if key_file else os.environ["OPENAI_API_KEY"]
    extra_body: dict[str, Any] = {}
    effort = os.environ.get("LAS_REASONING_EFFORT")
    if effort:
        extra_body["chat_template_kwargs"] = {"reasoning_effort": effort}
    return ChatOpenAI(
        base_url=os.environ["LAS_BASE_URL"],
        model=os.environ["LAS_MODEL"],
        api_key=api_key,
        max_tokens=int(os.environ.get("LAS_MAX_TOKENS", "32768")),
        timeout=float(os.environ.get("LAS_REQUEST_TIMEOUT", "1800")),
        max_retries=2,
        stream_usage=True,
        extra_body=extra_body or None,
    )


def callbacks(workdir: Path, session: str) -> list[Any]:
    """The usage log, and Langfuse's handler when its keys are set."""
    handlers: list[Any] = [UsageLog(workdir / "usage.jsonl")]
    if os.environ.get("LANGFUSE_HOST") and os.environ.get("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler

        handlers.append(CallbackHandler())
    return handlers


async def main() -> int:
    """Start the run, or resume it when its thread has a checkpoint."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    workdir = Path(os.environ.get("LAS_WORKDIR", "runs/served"))
    thread_id = os.environ.get("LAS_THREAD_ID", "circpack-served")
    session = os.environ.get("LAS_TRACE_SESSION", thread_id)
    workspace = LocalWorkspace(workdir / "run")
    if not (workspace.root / "inputs" / "circpack").exists():
        shutil.copytree(EXAMPLE, workspace.root / "inputs" / "circpack")
    scale = os.environ.get("LAS_SCALE", "small")
    config = small_config() if scale == "small" else ScientistConfig.preset("paper")
    window = os.environ.get("LAS_CONTEXT_WINDOW")
    context = ScientistContext(
        models=ModelRegistry(default=chat_model(), context_window=int(window) if window else None),
        workspace=workspace,
        evaluator=LocalCommandEvaluator(workspace),
        final_evaluator=LocalFinalEvaluator(workspace),
        config=config,
        policy=AutoApprovePolicy(),
    )
    run_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": int(os.environ.get("LAS_MAX_CONCURRENCY", "2")),
        "callbacks": callbacks(workdir, session),
        "metadata": {"langfuse_session_id": session},
    }
    serde = JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist())
    async with AsyncPostgresSaver.from_conn_string(
        os.environ["LAS_POSTGRES_URI"], serde=serde
    ) as saver:
        await saver.setup()
        graph = build_pipeline_graph(checkpointer=saver)
        resuming = (await saver.aget_tuple(run_config)) is not None
        print("RUN RESUME" if resuming else "RUN START", thread_id, flush=True)
        payload: Any = None if resuming else {"task_dir": "inputs/circpack"}
        out: dict[str, Any] = await graph.ainvoke(
            payload, run_config, context=context, durability="sync"
        )
        while out.get("__interrupt__"):
            out = await graph.ainvoke(
                Command(resume="proceed"), run_config, context=context, durability="sync"
            )
    print("status:", out["status"])
    for phase, status in out["phase_status"].items():
        print(f"  {phase}: {status.status}" + (f" ({status.reason})" if status.reason else ""))
    print("RUN DONE", out["status"], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
