"""The final evaluation and post-hoc graphs checkpoint and resume under a strict allowlist."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command

from langgraph_agentic_scientist import (
    DeferredFinalEvaluator,
    LocalWorkspace,
    ScientistConfig,
    serde_allowlist,
)
from langgraph_agentic_scientist.core.state import ProgramRef
from langgraph_agentic_scientist.evaluation import FinalEvaluationRequest, SealedInputs
from langgraph_agentic_scientist.final_evaluation import build_final_evaluation_graph
from langgraph_agentic_scientist.testing import FakeFinalEvaluator, write_minimal_task


def _strict_saver() -> InMemorySaver:
    return InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist()))


async def test_final_evaluation_resumes_from_a_strict_checkpoint(
    make_context: Any, workspace: LocalWorkspace, tmp_path: Path
) -> None:
    task = await write_minimal_task(workspace)
    await workspace.write_bytes("task/evaluate_final.py", b"# sealed\n")

    async def draw(value: Any, runtime: Any) -> SealedInputs:
        return SealedInputs(
            ref="drawn", drawn_at="2026-01-02T00:00:00Z", metadata={"seeds": [1, 2]}
        )

    from langgraph_agentic_scientist import GraphOverrides

    ctx = make_context(final_evaluator=DeferredFinalEvaluator(), config=ScientistConfig())
    app = build_final_evaluation_graph(
        checkpointer=_strict_saver(),
        overrides=GraphOverrides(slots={"final_evaluation.draw_inputs": draw}),
    )
    thread = {"configurable": {"thread_id": "strict"}}
    out = await app.ainvoke(
        {"task": task, "run_id": "r1", "incumbent": ProgramRef(path="task/program_0")},
        thread,
        context=ctx,
    )
    (pending,) = out["__interrupt__"]
    request = FinalEvaluationRequest.model_validate(pending.value["request"])
    answer = FakeFinalEvaluator()._answer(request).model_dump(mode="json")
    final = await app.ainvoke(Command(resume=answer), thread, context=ctx)
    assert final["final_status"] == "completed"
    assert final["frozen"].sealed_inputs.metadata == {"seeds": [1, 2]}
