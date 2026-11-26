"""Shared builders of the post-hoc tests: a finished run, scripted roles, a valid analysis."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage

from langgraph_agentic_scientist import ScientistConfig
from langgraph_agentic_scientist.config import AgentsConfig, ModelRetryConfig, PosthocConfig
from langgraph_agentic_scientist.core.results import InvestigationResult
from langgraph_agentic_scientist.core.state import BestManifest, TaskRef
from langgraph_agentic_scientist.posthoc.models import PosthocBundle, SelectionAnswer
from langgraph_agentic_scientist.testing import (
    FakeEvaluator,
    completed_result,
    final,
    structured_call,
    tool_call,
    tool_calls,
    write_minimal_task,
    write_program,
)

CHECKPOINT = pickle.dumps({"weights": [1.0, 2.0, 3.0]}, protocol=0).decode("ascii")
BEST_DIR = "results/w00/program_1"


def config(**posthoc: Any) -> ScientistConfig:
    """A configuration whose in-session model retries do not sleep, with post-hoc overrides."""
    agents = AgentsConfig(
        model_retry=ModelRetryConfig(max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0)
    )
    return ScientistConfig(agents=agents, posthoc=PosthocConfig(**posthoc))


async def finished_run(
    workspace: Any, *, checkpoint: bool = True, score: float = 2.0
) -> tuple[TaskRef, InvestigationResult]:
    """Write a task and a finished investigation whose best program may hold a checkpoint."""
    task = await write_minimal_task(workspace)
    await write_program(workspace, BEST_DIR, f"SCORE = {score}\n")
    await workspace.write_bytes(
        f"{BEST_DIR}/metrics.json", json.dumps({"combined_score": score}).encode()
    )
    await workspace.write_bytes(f"{BEST_DIR}/correct.json", b'{"correct": true}')
    if checkpoint:
        await workspace.write_bytes(f"{BEST_DIR}/best_model.pkl", CHECKPOINT.encode("ascii"))
        await workspace.write_bytes(
            f"{BEST_DIR}/_results.json", b'{"checkpoint_path": "best_model.pkl"}'
        )
    investigation = InvestigationResult(
        status="completed",
        best=BestManifest(program_dir=BEST_DIR, combined_score=score, correct=True),
        results_dir="results",
    )
    return task, investigation


def proposal(
    analysis_id: str,
    *,
    family: str = "weight_introspection",
    title: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    """One analysis proposal of the generator."""
    return {
        "id": analysis_id,
        "title": title or f"Analysis {analysis_id}",
        "family": family,
        "hypothesis": "weights are small",
        "relevance_reason": "supports the story",
        "feasibility_reason": "the checkpoint exists",
        "required_artifacts": ["best_model.pkl"],
        "requires_retrain": False,
        "implementation_plan": ["load the checkpoint", "plot the weights"],
        "expected_claim": "weights are non-negative",
        "expected_outputs": ["weights.csv"],
        **fields,
    }


def generator(*proposals: dict[str, Any]) -> list[AIMessage]:
    """The generator's script: one structured answer."""
    decision = "propose_analyses" if proposals else "no_relevant_analyses"
    bundle = PosthocBundle.model_validate(
        {"decision": decision, "summary": "s", "analyses": list(proposals)}
    )
    return [structured_call(PosthocBundle, bundle)]


def selector(*decisions: tuple[str, bool, str]) -> list[AIMessage]:
    """The selector's script: one structured answer."""
    answer = SelectionAnswer.model_validate(
        {"decisions": [{"id": i, "accepted": a, "reason": r} for i, a, r in decisions]}
    )
    return [structured_call(SelectionAnswer, answer)]


def valid_outputs(analysis_id: str, attempt: int, **manifest: Any) -> dict[str, str]:
    """The files of a valid attempt, by path relative to the attempt root."""
    rubric_ids = [
        "analysis_summary_present",
        "analysis_manifest_present",
        "interestingness_rubric_present",
        "required_outputs_present",
        "claim_grounded",
        "checkpoint_guardrail_respected",
    ]
    manifest_data = {
        "analysis_id": analysis_id,
        "attempt_id": f"attempt_{attempt}",
        "summary": "weights are non-negative",
        "claim": "weights are non-negative",
        "grounded": True,
        "reran_evaluator": False,
        "primary_metric_tolerance": 0.01,
        "outputs": {"tables": ["weights.csv"]},
        "evidence_paths": ["task_bundle/program_0/best_model.pkl"],
        "checkpoint_paths": [],
        **manifest,
    }
    return {
        "analysis_summary.md": "# Summary\nThe weights are non-negative.\n",
        "interestingness_rubric.json": json.dumps({"novelty": "high"}),
        "analysis_manifest.json": json.dumps(manifest_data),
        "rubric_status.json": json.dumps(
            {
                "analysis_id": analysis_id,
                "overall_pass": True,
                "rubrics": [{"id": rubric, "pass": True} for rubric in rubric_ids],
            }
        ),
        "weights.csv": "weight\n1.0\n2.0\n3.0\n",
    }


def write_all(files: dict[str, str]) -> AIMessage:
    """One model turn that writes every file with ``write_file``."""
    return tool_calls(
        *(
            ("write_file", {"file_path": f"/{path}", "content": text})
            for path, text in files.items()
        )
    )


def executor_script(analysis_id: str, *attempts: dict[str, str] | Sequence[AIMessage]) -> list[Any]:
    """The executor's script: per attempt, write the files (or run the given turns), then stop."""
    steps: list[Any] = []
    for number, attempt in enumerate(attempts, start=1):
        if isinstance(attempt, dict):
            steps.extend([write_all(attempt), final(f"attempt {number} done")])
        else:
            steps.extend([*attempt, final(f"attempt {number} done")])
    return steps


def shell(command: str) -> AIMessage:
    """One model turn that runs a shell command."""
    return tool_call("execute", {"command": command})


def evaluator_with_score(score: float = 2.0) -> FakeEvaluator:
    """A fake evaluator whose every evaluation completes with the given score."""

    def answer(request: Any) -> Any:
        return completed_result(request, combined_score=score)

    return FakeEvaluator(default=answer)


def last_human_text(messages: Sequence[Any]) -> str:
    """The text of the first human message of a model call (the session's prompt)."""
    for message in messages:
        if getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else json.dumps(content)
    return ""


Script = Callable[[], list[Any]]


EVIDENCE = json.dumps(
    {
        "schema_version": 1,
        "episodes": [{"seed": 1, "score": 1.5, "correct": True, "duration_seconds": 0.1}],
    }
)


async def ablations(workspace: Any, root: str = "ablations") -> None:
    summary = {
        "schema_version": 1,
        "total_variants": 2,
        "total_completed": 1,
        "variants": [
            {"variant_id": "control_best", "status": "completed"},
            {"variant_id": "no_trick", "status": "failed"},
        ],
    }
    await workspace.write_bytes(f"{root}/pipeline_summary.json", json.dumps(summary).encode())
    await workspace.write_bytes(f"{root}/README.md", b"# ablations\n")
    await workspace.write_bytes(f"{root}/generation/variant_bundle.json", b"{}")
    await workspace.write_bytes(f"{root}/aggregated/task/run_summary.csv", b"a,b\n1,2\n")
    await workspace.write_bytes(f"{root}/aggregated/task/ignored.csv", b"x\n")
    await workspace.write_bytes(
        f"{root}/variants/control_best/task_bundle/program_0/evaluation_evidence.json",
        EVIDENCE.encode(),
    )
    await workspace.write_bytes(
        f"{root}/variants/control_best/task_bundle/program_0/stdout.log", b"not staged"
    )
