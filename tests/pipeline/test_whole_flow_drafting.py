"""Task drafting inside the pipeline, on the package's own graphs: a task with only ``task.md``.

The design loop and the scaffold loop draft the toy contracted task (the drafter writes the
scaffold of ``tests/fixtures/core/contract_task``), the pipeline reseals its snapshot and reloads
the task with its contract, and the investigation then searches the drafted scaffold in the
harness-evaluated regime that a contract forces.
"""

from __future__ import annotations

from typing import Any

from drafting.test_design_support import design_payload
from drafting.test_design_support import judge as design_judge
from drafting.test_scaffold_support import (
    FIXTURE,
    drafter_script,
    judge_step,
    make_evaluator,
)
from langgraph.checkpoint.memory import InMemorySaver
from testkit import thread
from whole_flow import IDEA, answer, memory_line, steps

from langgraph_agentic_scientist import (
    LocalWorkspace,
    ScientistConfig,
    ScientistContext,
    build_pipeline_graph,
)
from langgraph_agentic_scientist.config import AgentsConfig, InvestigationConfig, ModelRetryConfig
from langgraph_agentic_scientist.drafting.design_models import TaskDesign
from langgraph_agentic_scientist.investigation.ideas import IdeaDraft
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import ScriptedModels, structured_call, tool_call

TASK_MD = (FIXTURE / "task.md").read_text(encoding="utf-8")
CONTRACT_IDEA = {
    **IDEA,
    "target_rubrics": [
        {
            "id": "returns_three",
            "description": "handle returns 3.0.",
            "type": "implementation",
            "verification": {"type": "source_contains", "patterns": ["return 3.0"]},
        }
    ],
}

AUTHOR = steps(
    tool_call("execute", {"command": "cp -r program_0 program_1"}),
    tool_call(
        "write_file",
        {
            "file_path": "/program_1/src/program.py",
            "content": "def handle(request):\n    return 3.0\n",
        },
    ),
    memory_line,
)


async def test_a_task_drafted_in_the_pipeline_is_then_searched(workspace: LocalWorkspace) -> None:
    await workspace.write_bytes("inputs/toy/task.md", TASK_MD.encode("utf-8"))
    models = ScriptedModels.from_scripts(
        {
            Role.TASK_DESIGNER: [structured_call(TaskDesign, design_payload(TASK_MD))],
            Role.DESIGN_JUDGE: [design_judge()],
            Role.TASK_DRAFTER: drafter_script(),
            Role.ALIGNMENT_JUDGE: [judge_step()] * 4,
            Role.IDEATOR: [answer(IdeaDraft, CONTRACT_IDEA)] * 4,
            Role.CANDIDATE_AUTHOR: [AUTHOR] * 20,
        },
        context_window=200_000,
    )
    evaluator = make_evaluator()
    ctx = ScientistContext(
        models=models,
        workspace=workspace,
        evaluator=evaluator,
        config=ScientistConfig(
            phases=("task_drafting", "investigation"),
            allow_unisolated_contracted=True,
            agents=AgentsConfig(
                model_retry=ModelRetryConfig(
                    max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0
                )
            ),
            investigation=InvestigationConfig(max_idea_cycles=1),
        ),
    )
    app = build_pipeline_graph(checkpointer=InMemorySaver())

    out: dict[str, Any] = await app.ainvoke(
        {"task_dir": "inputs/toy"}, thread("drafted"), context=ctx
    )

    statuses = {name: status.status for name, status in out["phase_status"].items()}
    assert statuses["task_drafting"] == "completed", out["phase_status"]
    assert statuses["investigation"] == "completed", out["phase_status"]
    assert out["drafting"].status == "published"
    task = out["task"]
    assert task.contract is not None and task.harness_evaluated
    assert (workspace.root / "task" / "evaluate_final.py").is_file()
    assert models.scripted(Role.CANDIDATE_AUTHOR).calls, "the drafted task was searched"
    # every evaluation of the search is the host's (the harness-evaluated regime)
    assert any(request.plane == "search" for request in evaluator.calls)
