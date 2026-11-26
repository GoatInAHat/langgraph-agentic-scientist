"""Fixtures of the paper-data tests: an investigation, an ablation bundle and a sealed final."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from langgraph_agentic_scientist import LocalWorkspace, ScientistConfig
from langgraph_agentic_scientist.config import AgentsConfig, ModelRetryConfig
from langgraph_agentic_scientist.core.results import (
    AttemptRecord,
    IdeaRecord,
    InvestigationResult,
    VariantSpec,
    VariantsResult,
)
from langgraph_agentic_scientist.core.state import EvaluationRecord, IdeaBundle, ProgramRef
from langgraph_agentic_scientist.evaluation.protocol import FinalEvaluationResult


def fast_config(**updates: Any) -> ScientistConfig:
    """A configuration whose in-session model retries do not sleep."""
    agents = AgentsConfig(
        model_retry=ModelRetryConfig(max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0)
    )
    return ScientistConfig(agents=agents, **updates)


EVIDENCE = {
    "schema_version": 1,
    "episodes": [
        {"seed": 0, "score": 0.5, "correct": True, "duration_seconds": 1.5},
        {"seed": 1, "score": 0.7, "correct": True, "duration_seconds": 2.0},
    ],
}


async def write_json(workspace: LocalWorkspace, path: str, payload: Any) -> None:
    await workspace.write_bytes(path, json.dumps(payload).encode("utf-8"))


async def write_text(workspace: LocalWorkspace, path: str, text: str) -> None:
    await workspace.write_bytes(path, text.encode("utf-8"))


def idea_record(directory: str, idea_id: str, title: str, cycle: int) -> IdeaRecord:
    return IdeaRecord(
        idea_id=idea_id,
        idea_dir=f"results/ideas/{directory}",
        cycle=cycle,
        bundle=IdeaBundle(
            idea_id=idea_id,
            title=title,
            hypothesis="h",
            implementation_plan=["p"],
            target_rubrics=[],
        ),
    )


def evaluation(
    directory: str, idea_id: str, number: int, score: float | None, correct: bool | None
) -> EvaluationRecord:
    metrics = {} if score is None else {"combined_score": score}
    return EvaluationRecord(
        evaluation_id=f"run/{directory}/program_{number}",
        program=ProgramRef(path=f"results/ideas/{directory}/program_{number}"),
        plane="search",
        status="completed",
        metrics=metrics,
        correct=correct,
        idea_id=idea_id,
        attempt_id=f"{idea_id}_attempt_01",
    )


async def investigation(
    workspace: LocalWorkspace, *, with_evidence: bool = False, pending: bool = True
) -> InvestigationResult:
    """Two ideas; four complete evaluations and one still pending.

    Evaluations (idea, program, score, correct): alpha 1 0.3 yes, alpha 2 0.5 yes, beta 1 0.4 no,
    beta 2 0.9 yes; beta 3 is pending (its evaluator has not completed).
    """
    rows = [
        ("01_alpha", "alpha", 1, 0.3, True),
        ("01_alpha", "alpha", 2, 0.5, True),
        ("02_beta", "beta", 1, 0.4, False),
        ("02_beta", "beta", 2, 0.9, True),
    ]
    evaluations: list[EvaluationRecord] = []
    for directory, idea_id, number, score, correct in rows:
        program = f"results/ideas/{directory}/program_{number}"
        await write_json(workspace, f"{program}/metrics.json", {"combined_score": score})
        await write_json(workspace, f"{program}/correct.json", {"correct": correct})
        evaluations.append(evaluation(directory, idea_id, number, score, correct))
    if with_evidence:
        await write_json(
            workspace, "results/ideas/02_beta/program_2/evaluation_evidence.json", EVIDENCE
        )
    if pending:
        await write_json(
            workspace, "results/ideas/02_beta/program_3/metrics.json", {"combined_score": 5.0}
        )
        await write_text(workspace, "results/ideas/02_beta/program_3/.evaluation_pending", "")
        evaluations.append(evaluation("02_beta", "beta", 3, 5.0, True))
    return InvestigationResult(
        status="completed",
        ideas=[
            idea_record("02_beta", "beta", "Beta idea", 2),
            idea_record("01_alpha", "alpha", "Alpha idea", 1),
        ],
        attempts=[
            AttemptRecord(
                attempt_id=f"{idea}_attempt_01",
                idea_id=idea,
                cycle=cycle,
                attempt_dir=f"results/ideas/{directory}",
                status="completed",
            )
            for directory, idea, cycle in (("02_beta", "beta", 2), ("01_alpha", "alpha", 1))
        ],
        evaluations=list(reversed(evaluations)),
        results_dir="results",
    )


CSV_COLUMNS = (
    "run_id,evaluation_id,variant_id,variant_kind,base_program,agent_name,correct,combined_score,"
    "overall_pass,primary_metric_name,primary_metric_value,program_dir,results_dir"
)


def run_rows(entries: Iterable[tuple[str, str, float, str]]) -> str:
    lines = [CSV_COLUMNS]
    for index, (variant, kind, score, correct) in enumerate(entries, start=1):
        lines.append(
            f"run_{index},ev_{index},{variant},{kind},best,codex,{correct},{score},True,"
            f"combined_score,{score},variants/{variant}/task_bundle/program_1,out/{variant}/run_{index}"
        )
    return "\n".join(lines) + "\n"


def metric_rows(entries: Iterable[tuple[str, str, float]]) -> str:
    lines = ["run_id,evaluation_id,variant_id,metric_name,metric_value,is_primary_metric"]
    for index, (variant, metric, value) in enumerate(entries, start=1):
        primary = "True" if metric == "combined_score" else "False"
        lines.append(f"run_{index},ev_{index},{variant},{metric},{value},{primary}")
    return "\n".join(lines) + "\n"


async def ablation_bundle(workspace: LocalWorkspace, root: str = "abl") -> VariantsResult:
    """A bundle with a control (0.8, 0.9) and a variant (0.6, 0.7 and one incorrect run)."""
    await write_json(
        workspace,
        f"{root}/pipeline_summary.json",
        {
            "title": "Ablation of X",
            "variants": [
                {
                    "variant_id": "control_best",
                    "title": "Control",
                    "kind": "control",
                    "status": "completed",
                },
                {
                    "variant_id": "no_x",
                    "title": "Without X",
                    "kind": "ablation",
                    "status": "completed",
                },
            ],
        },
    )
    entries = [
        ("control_best", "control", 0.8, "True"),
        ("control_best", "control", 0.9, "True"),
        ("no_x", "ablation", 0.6, "True"),
        ("no_x", "ablation", 0.7, "False"),
    ]
    await write_text(workspace, f"{root}/aggregated/run_summary.csv", run_rows(entries))
    await write_text(
        workspace,
        f"{root}/aggregated/metric_values.csv",
        metric_rows(
            [
                ("control_best", "combined_score", 0.8),
                ("control_best", "combined_score", 0.9),
                ("no_x", "combined_score", 0.6),
                ("no_x", "combined_score", 0.7),
                ("no_x", "latency", 3.0),
            ]
        ),
    )
    for variant, kind, title in (
        ("control_best", "control", "Control"),
        ("no_x", "ablation", "Without X"),
    ):
        directory = f"{root}/variants/{variant}"
        await write_json(
            workspace,
            f"{directory}/variant_spec.json",
            {"variant_id": variant, "kind": kind, "title": title},
        )
        await write_json(
            workspace, f"{directory}/run_manifest.json", {"status": "completed", "eval_count": 2}
        )
        await write_json(workspace, f"{directory}/target_rubrics.json", {"rubrics": []})
        await write_text(workspace, f"{directory}/task_bundle/README.md", "x")
    return VariantsResult(
        status="completed",
        output_dir=root,
        specs=[
            VariantSpec(variant_id=name, kind=kind, variant_dir=f"{root}/variants/{name}")
            for name, kind in (("control_best", "control"), ("no_x", "ablation"))
        ],
    )


def final_result(
    *,
    correct: bool = True,
    metrics: dict[str, float] | None = None,
    output_dir: str | None = "final",
) -> FinalEvaluationResult:
    return FinalEvaluationResult(
        correct=correct,
        outcome="qualified" if correct else "not_qualified",
        metrics=metrics or {"score": 0.9, "Edge Case!": 1.0},
        declared_final_metrics=["score"],
        task_tree_sha256="sha256:task",
        evaluator_sha256="sha256:eval",
        program_tree_sha256="sha256:program",
        results_tree_sha256="sha256:results",
        started_at="2026-01-01T00:00:00+00:00",
        completed_at="2026-01-01T01:00:00+00:00",
        output_dir=output_dir,
    )


async def build_paper_data_for_tests(workspace: LocalWorkspace) -> dict[str, Any]:
    """The paper data of the fixture investigation, as the dict that is written to JSON."""
    from langgraph_agentic_scientist.paper.data import build_paper_data

    data = await build_paper_data(workspace, investigation=await investigation(workspace))
    return data.model_dump(mode="json")


def source_ids(paper_data: Any) -> Sequence[str]:
    return sorted(paper_data.grounding_sources)


def test_support_builds_fixtures() -> None:
    assert final_result().outcome == "qualified" and final_result(correct=False).correct is False
