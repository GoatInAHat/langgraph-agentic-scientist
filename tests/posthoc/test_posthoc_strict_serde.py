"""The post-hoc graph checkpoints reports, candidates and snapshots under a strict allowlist."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
from posthoc_support import (
    CHECKPOINT,
    config,
    executor_script,
    finished_run,
    generator,
    proposal,
    selector,
    valid_outputs,
)

from langgraph_agentic_scientist import serde_allowlist
from langgraph_agentic_scientist.evaluation import DeferredEvaluator, EvaluationResult
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.posthoc import build_posthoc_graph
from langgraph_agentic_scientist.testing import ScriptedModels, final, tool_call


async def test_a_run_interrupted_in_the_bootstrap_resumes_from_a_strict_checkpoint(
    make_context: Any, workspace: Any
) -> None:
    task, investigation = await finished_run(workspace, checkpoint=False)
    models = ScriptedModels.from_scripts(
        {
            Role.BOOTSTRAP_AGENT: [
                tool_call(
                    "write_file", {"file_path": "/program_0/model_state.pkl", "content": CHECKPOINT}
                ),
                final("done"),
            ],
            Role.ANALYSIS_GENERATOR: generator(
                proposal("weights", required_artifacts=["model_state.pkl"])
            ),
            Role.ANALYSIS_SELECTOR: selector(("weights", True, "good")),
            Role.ANALYSIS_EXECUTOR: executor_script(
                "weights",
                valid_outputs(
                    "weights", 1, evidence_paths=["task_bundle/program_0/model_state.pkl"]
                ),
            ),
        },
        context_window=200_000,
    )
    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist()))
    ctx = make_context(models=models, evaluator=DeferredEvaluator(), config=config())
    app = build_posthoc_graph(checkpointer=saver)
    thread = {"configurable": {"thread_id": "strict"}}
    out = await app.ainvoke(
        {"task": task, "run_id": "r1", "investigation": investigation}, thread, context=ctx
    )
    (pending,) = out["__interrupt__"]
    answer = EvaluationResult(
        evaluation_id=pending.value["request"]["evaluation_id"],
        status="completed",
        metrics={"combined_score": 2.0},
        correct=True,
    ).model_dump(mode="json")
    done = await app.ainvoke(Command(resume=answer), thread, context=ctx)
    assert done["posthoc"].status == "completed"
    snapshot = await app.aget_state(thread)
    assert snapshot.values["selection"].accepted[0].analysis_id == "weights"
