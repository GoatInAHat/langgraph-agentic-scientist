"""The host's canonical analysis (stage 5) and the aggregation (stage 7)."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import pytest

from langgraph_agentic_scientist import PromptRegistry
from langgraph_agentic_scientist.evaluation.protocol import (
    EvaluationEvidence,
    EvaluationResult,
    EvidenceEpisode,
)
from langgraph_agentic_scientist.variants import analysis
from langgraph_agentic_scientist.variants.aggregate import aggregate_variant_csvs
from langgraph_agentic_scientist.variants.analysis import (
    numeric_metrics,
    publish_summary_qualification,
    summary_qualification,
    write_canonical_analysis,
)
from langgraph_agentic_scientist.variants.rubrics import build_rubric_contract
from langgraph_agentic_scientist.workspace import LocalWorkspace

PROMPTS = PromptRegistry.default()
ENV = {
    "EVALUATION_ID": "v1_eval_1",
    "VARIANT_ID": "v1",
    "VARIANT_KIND": "ablation",
    "BASE_PROGRAM": "best",
    "AGENT_NAME": "model-x",
}


def _read(workspace: LocalWorkspace, path: str) -> str:
    """Read a file without newline translation (the CSV dialect is part of the contract)."""
    return (workspace.root / path).read_bytes().decode("utf-8")


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def _result(**updates: Any) -> EvaluationResult:
    fields: dict[str, Any] = {
        "evaluation_id": "run/v1/program_1",
        "status": "completed",
        "metrics": {"combined_score": 2.5, "detail": {"loss": 0.5, "flag": True, "name": "x"}},
        "correct": True,
        "output_dir": ".evaluations/run/v1/program_1/results",
    }
    fields.update(updates)
    return EvaluationResult(**fields)


def test_numeric_metrics_flatten_nested_and_sort_the_primary_first() -> None:
    metrics = numeric_metrics(
        {
            "b": 1,
            "a": {"y": 2.0, "x": {"z": 3}},
            "combined_score": 9,
            "s": "text",
            "l": [1],
            "t": True,
        }
    )
    assert metrics == [("combined_score", 9.0), ("a.x.z", 3.0), ("a.y", 2.0), ("b", 1.0)]
    with pytest.raises(RuntimeError, match="non-finite"):
        numeric_metrics({"x": float("inf")})
    with pytest.raises(RuntimeError, match="invalid metric name"):
        numeric_metrics({"bad name": 1})


async def test_canonical_analysis_files(workspace: LocalWorkspace) -> None:
    out = ".evaluations/run/v1/program_1/results"
    scratch = (
        "evaluation_id,variant_id,series_name,step,value\r\n"
        "x,y,train_loss,0,1.5\r\n"
        "x,y,train_loss,1,nan\r\n"
        "x,y,flag,2,True\r\n"
    )
    await workspace.write_bytes(f"{out}/runs/rep_1/analysis/timeseries.csv", scratch.encode())
    evidence = EvaluationEvidence(
        episodes=[
            EvidenceEpisode(
                seed=7,
                score=1.0,
                correct=True,
                duration_seconds=0.1,
                candidate_timeseries=[{"series_name": "reward", "step": 0, "value": 0.25}],
            )
        ]
    )
    await write_canonical_analysis(
        workspace,
        program_dir="b/program_1",
        result=_result(evidence=evidence),
        environment=ENV,
    )
    text = _read(workspace, "b/program_1/analysis/run_summary.csv")
    assert text.splitlines()[0] == ",".join(
        [
            "evaluation_id",
            "variant_id",
            "variant_kind",
            "base_program",
            "agent_name",
            "correct",
            "combined_score",
            "overall_pass",
            "primary_metric_name",
            "primary_metric_value",
            "program_dir",
            "results_dir",
        ]
    )
    (row,) = _rows(text)
    assert (
        row["correct"] == "True" and row["overall_pass"] == "" and row["program_dir"] == "program_1"
    )
    assert row["agent_name"] == "model-x" and row["combined_score"] == "2.5"
    metrics = _rows(_read(workspace, "b/program_1/analysis/metric_values.csv"))
    assert [(m["metric_name"], m["is_primary_metric"]) for m in metrics] == [
        ("combined_score", "True"),
        ("detail.loss", "False"),
    ]
    series = _rows(_read(workspace, "b/program_1/analysis/timeseries.csv"))
    assert [(s["series_name"], s["step"], s["value"]) for s in series] == [
        ("candidate/seed_7/reward", "0", "0.25"),
        ("runs/rep_1/train_loss", "0", "1.5"),  # the NaN row is omitted, never written
        ("runs/rep_1/flag", "2", "1.0"),  # a boolean observation is numeric
    ]
    assert "\r\n" in text  # upstream's CSV dialect


@pytest.mark.parametrize(
    ("scratch", "message"),
    [
        ("a,b\r\n1,2\r\n", "invalid analysis CSV"),
        ("evaluation_id,variant_id,series_name,step,value\r\nx,y,s,-1,1\r\n", "invalid step"),
        ("evaluation_id,variant_id,series_name,step,value\r\nx,y,s,0.5,1\r\n", "invalid step"),
        ("evaluation_id,variant_id,series_name,step,value\r\nx,y,s,0,inf\r\n", "non-finite"),
        ("evaluation_id,variant_id,series_name,step,value\r\nx,y,bad name,0,1\r\n", "series name"),
        (
            "evaluation_id,variant_id,series_name,step,value\r\nx,y,s,0,1,extra\r\n",
            "invalid analysis CSV",
        ),
    ],
)
async def test_canonical_analysis_refuses_bad_scratch_csv(
    workspace: LocalWorkspace, scratch: str, message: str
) -> None:
    out = ".evaluations/run/v1/program_1/results"
    await workspace.write_bytes(f"{out}/analysis/timeseries.csv", scratch.encode())
    with pytest.raises(RuntimeError, match=message):
        await write_canonical_analysis(
            workspace, program_dir="b/program_1", result=_result(), environment=ENV
        )


async def test_canonical_analysis_bounds_and_contracts(
    workspace: LocalWorkspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="combined_score"):
        await write_canonical_analysis(
            workspace, program_dir="b/program_1", result=_result(metrics={"x": 1}), environment=ENV
        )
    with pytest.raises(RuntimeError, match="boolean 'correct'"):
        await write_canonical_analysis(
            workspace, program_dir="b/program_1", result=_result(correct=None), environment=ENV
        )
    with pytest.raises(RuntimeError, match="invalid evaluation_id"):
        await write_canonical_analysis(
            workspace,
            program_dir="b/program_1",
            result=_result(),
            environment={**ENV, "EVALUATION_ID": "bad id"},
        )
    out = ".evaluations/run/v1/program_1/results"
    rows = "".join(f"x,y,s,{i},1\r\n" for i in range(4))
    await workspace.write_bytes(
        f"{out}/analysis/timeseries.csv",
        ("evaluation_id,variant_id,series_name,step,value\r\n" + rows).encode(),
    )
    monkeypatch.setattr(analysis, "MAX_ANALYSIS_ROWS", 3)
    with pytest.raises(RuntimeError, match=r"invalid analysis CSV|row limit"):
        await write_canonical_analysis(
            workspace, program_dir="b/program_1", result=_result(), environment=ENV
        )


async def _qualified_program(workspace: LocalWorkspace, *, overall: bool) -> str:
    program = "b/program_1"
    await write_canonical_analysis(
        workspace, program_dir=program, result=_result(), environment=ENV
    )
    await workspace.write_bytes(
        f"{program}/metrics.json", json.dumps({"combined_score": 2.5}).encode()
    )
    await workspace.write_bytes(f"{program}/correct.json", json.dumps({"correct": True}).encode())
    contract = build_rubric_contract(PROMPTS, "v1", "ablation", "best")
    status = {
        "variant_id": "v1",
        "attempt_id": "v1_attempt_01",
        "idea_id": "v1",
        "overall_pass": overall,
        "rubrics": [
            {"id": r["id"], "pass": overall or r["id"] != "single_change_scope", "evidence": "e"}
            for r in contract["rubrics"]
        ],
        "metrics": {"combined_score": 2.5, "correct": True},
        "next_recommendation": "",
    }
    await workspace.write_bytes(f"{program}/rubric_status.json", json.dumps(status).encode())
    return program


async def test_overall_pass_is_published_only_when_the_status_is_consistent(
    workspace: LocalWorkspace,
) -> None:
    program = await _qualified_program(workspace, overall=True)
    assert await summary_qualification(workspace, program, prompts=PROMPTS) == {
        "overall_pass": "True"
    }
    await publish_summary_qualification(workspace, program, prompts=PROMPTS)
    (row,) = _rows(_read(workspace, f"{program}/analysis/run_summary.csv"))
    assert row["overall_pass"] == "True"
    # An overall_pass that disagrees with the rubrics is not published.
    path = workspace.root / program / "rubric_status.json"
    status = json.loads(path.read_text())
    status["overall_pass"] = False
    path.write_text(json.dumps(status))
    assert await summary_qualification(workspace, program, prompts=PROMPTS) == {"overall_pass": ""}


# --- aggregation ---------------------------------------------------------------------------------


async def _write_csv(
    workspace: LocalWorkspace, path: str, header: list[str], *rows: list[str]
) -> None:
    text = ",".join(header) + "\r\n" + "".join(",".join(r) + "\r\n" for r in rows)
    await workspace.write_bytes(path, text.encode())


async def test_aggregation_takes_the_union_of_columns_and_the_latest_program(
    workspace: LocalWorkspace,
) -> None:
    base = "out/variants"
    await _write_csv(
        workspace,
        f"{base}/control_best/task_bundle/program_0/analysis/run_summary.csv",
        ["variant_id", "score"],
        ["control_best", "1"],
    )
    # an early attempt is not aggregated, the latest is
    await _write_csv(
        workspace,
        f"{base}/abl/task_bundle/program_1/analysis/run_summary.csv",
        ["variant_id", "score"],
        ["abl", "0"],
    )
    await _write_csv(
        workspace,
        f"{base}/abl/task_bundle/program_2/analysis/run_summary.csv",
        ["variant_id", "score", "extra"],
        ["abl", "5", "e"],
    )
    await _write_csv(
        workspace,
        f"{base}/abl/task_bundle/program_1/analysis/metric_values.csv",
        ["variant_id", "metric_name"],
        ["abl", "m"],
    )
    results = await aggregate_variant_csvs(workspace, PROMPTS, "out", original_task_name="orig")
    assert results["orig"]["timeseries.csv"] is None
    summary = _rows(_read(workspace, "out/aggregated/orig/run_summary.csv"))
    assert [(row["variant_id"], row["score"], row["extra"]) for row in summary] == [
        ("abl", "5", "e"),
        ("control_best", "1", ""),
    ]
    # the primary task's files are mirrored for existing consumers; a missing file removes the copy
    assert _read(workspace, "out/aggregated/run_summary.csv") == _read(
        workspace, "out/aggregated/orig/run_summary.csv"
    )
    assert not (workspace.root / "out/aggregated/timeseries.csv").exists()


async def test_aggregation_flat_nested_and_structured_extra_tasks(
    workspace: LocalWorkspace,
) -> None:
    await _write_csv(
        workspace,
        "out/variants/control_best/task_bundle/program_0/analysis/run_summary.csv",
        ["variant_id"],
        ["control_best"],
    )
    await _write_csv(workspace, "out/extra_tasks/flat/algo_a/run_summary.csv", ["score"], ["1"])
    await _write_csv(workspace, "out/extra_tasks/flat/algo_b/run_summary.csv", ["score"], ["2"])
    await _write_csv(
        workspace, "out/extra_tasks/nested/algo/rep_1/run_timeseries.csv", ["step"], ["0"]
    )
    await _write_csv(
        workspace, "out/extra_tasks/nested/algo/rep_2/run_timeseries.csv", ["step"], ["1"]
    )
    await _write_csv(
        workspace,
        "out/extra_tasks/struct/variants/v/task_bundle/program_1/analysis/timeseries.csv",
        ["variant_id"],
        ["v"],
    )
    results = await aggregate_variant_csvs(workspace, PROMPTS, "out", original_task_name="orig")
    assert set(results) == {"orig", "flat", "nested", "struct"}
    flat = _rows(_read(workspace, "out/aggregated/flat/run_summary.csv"))
    assert [(r["evaluation_id"], r["variant_id"], r["score"]) for r in flat] == [
        ("algo_a", "algo_a", "1"),
        ("algo_b", "algo_b", "2"),
    ]
    nested = _rows(_read(workspace, "out/aggregated/nested/run_timeseries.csv"))
    assert [(r["variant_id"], r["algorithm"], r["replication"]) for r in nested] == [
        ("algo__rep_1", "algo", "1"),
        ("algo__rep_2", "algo", "2"),
    ]
    assert results["struct"]["timeseries.csv"] == "out/aggregated/struct/timeseries.csv"


async def test_aggregation_without_variants_does_nothing(workspace: LocalWorkspace) -> None:
    assert await aggregate_variant_csvs(workspace, PROMPTS, "empty") == {}
