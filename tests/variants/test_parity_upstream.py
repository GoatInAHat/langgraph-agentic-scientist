"""Aggregation, canonical analysis and selection equal the vendored upstream functions.

The upstream modules under ``third_party/`` are imported and run on the same inputs as the port;
differences the adaptations declare are asserted explicitly.
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from langgraph_agentic_scientist import PromptRegistry
from langgraph_agentic_scientist.evaluation.protocol import EvaluationEvidence, EvaluationResult
from langgraph_agentic_scientist.variants.aggregate import aggregate_variant_csvs
from langgraph_agentic_scientist.variants.analysis import (
    numeric_metrics,
    publish_summary_qualification,
    summary_qualification,
    write_canonical_analysis,
)
from langgraph_agentic_scientist.variants.generate import screen_candidates
from langgraph_agentic_scientist.variants.models import (
    VariantBundleAnswer,
    VariantCandidate,
    VariantDecisionsAnswer,
    VariantSelection,
)
from langgraph_agentic_scientist.variants.select import (
    apply_hard_caps,
    decisions_from_answer,
    selection_json,
)
from langgraph_agentic_scientist.workspace import LocalWorkspace

PROMPTS = PromptRegistry.default()
Loader = Callable[..., ModuleType]
AGENT_PROMPT = {"src.variants_pipeline.agent_prompt": {"stage_agent_prompt": None}}


@pytest.fixture
def upstream_aggregate(upstream_modules: Loader) -> ModuleType:
    return upstream_modules("src.variants_pipeline.aggregate")


@pytest.fixture
def upstream_outputs(upstream_modules: Loader) -> ModuleType:
    return upstream_modules("src.variants_pipeline.sandbox_outputs")


@pytest.fixture
def upstream_select(upstream_modules: Loader) -> ModuleType:
    return upstream_modules("src.variants_pipeline.select", AGENT_PROMPT)


@pytest.fixture
def upstream_generate(upstream_modules: Loader) -> ModuleType:
    return upstream_modules("src.variants_pipeline.generate", AGENT_PROMPT)


def _csv(header: list[str], *rows: list[str], newline: str = "\r\n") -> bytes:
    return (newline.join([",".join(header), *(",".join(row) for row in rows)]) + newline).encode()


def _put(root: Path, relative: str, data: bytes | str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# --- summary qualification and rubric-bound overall_pass ----------------------------------------

SUMMARY_COLUMNS = [
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


def _write_qualified_program(
    root: Path,
    program: str,
    contract_module: ModuleType,
    *,
    variant_id: str = "v1",
    kind: str = "ablation",
    base: str = "best",
    overall: bool = True,
    flaw: str = "",
) -> None:
    suffix = program.rsplit("_", 1)[1]
    index = int(suffix) if suffix.isdigit() else 1
    row = [
        f"{variant_id}_eval",
        variant_id,
        kind,
        base,
        "model-x",
        "True",
        "2.5",
        "",
        "combined_score",
        "2.5",
        program.rsplit("/", 1)[-1],
        "analysis",
    ]
    rows = [row, row] if flaw == "two_rows" else [row]
    _put(root, f"{program}/analysis/run_summary.csv", _csv(SUMMARY_COLUMNS, *rows))
    contract = contract_module.build_rubric_contract(variant_id, kind, base)
    status: dict[str, Any] = {
        "variant_id": variant_id,
        "attempt_id": f"{variant_id}_attempt_{index:02d}",
        "idea_id": variant_id,
        "overall_pass": overall,
        "rubrics": [
            {"id": r["id"], "pass": overall or r["id"] != "single_change_scope", "evidence": "e"}
            for r in contract["rubrics"]
        ],
        "metrics": {"combined_score": 2.5, "correct": True},
        "next_recommendation": "",
    }
    metrics: Any = {"combined_score": 2.5}
    correct: Any = {"correct": True}
    if flaw == "attempt":
        status["attempt_id"] = f"{variant_id}_attempt_99"
    elif flaw == "idea":
        status["idea_id"] = "other"
    elif flaw == "wrong_overall":
        status["overall_pass"] = not overall
    elif flaw == "status_metrics":
        status["metrics"] = {"combined_score": 9.0, "correct": True}
    elif flaw == "metrics_mismatch":
        metrics = {"combined_score": 3.5}
    elif flaw == "correct_string":
        correct = {"correct": "True"}
    elif flaw == "correct_false":
        correct = {"correct": False}
    elif flaw == "missing_rubric":
        status["rubrics"] = status["rubrics"][:-1]
    elif flaw == "nan_score":
        metrics = {"combined_score": float("nan")}
    _put(root, f"{program}/metrics.json", json.dumps(metrics))
    _put(root, f"{program}/correct.json", json.dumps(correct))
    if flaw == "bad_status_json":
        _put(root, f"{program}/rubric_status.json", "{not json")
    elif flaw == "status_list":
        _put(root, f"{program}/rubric_status.json", "[]")
    elif flaw != "no_status":
        _put(root, f"{program}/rubric_status.json", json.dumps(status))
    if flaw == "no_correct":
        (root / program / "correct.json").unlink()
    if flaw == "no_summary":
        (root / program / "analysis/run_summary.csv").unlink()


QUALIFICATION_CASES = [
    {},
    {"overall": False},
    {"flaw": "attempt"},
    {"flaw": "idea"},
    {"flaw": "wrong_overall"},
    {"flaw": "status_metrics"},
    {"flaw": "metrics_mismatch"},
    {"flaw": "correct_string"},
    {"flaw": "correct_false"},
    {"flaw": "missing_rubric"},
    {"flaw": "nan_score"},
    {"flaw": "bad_status_json"},
    {"flaw": "status_list"},
    {"flaw": "no_status"},
    {"flaw": "no_correct"},
    {"flaw": "no_summary"},
    {"flaw": "two_rows"},
    {"kind": "baseline", "base": "reference"},
    {"variant_id": "baseline_reference", "kind": "baseline", "base": "reference"},
    {"kind": "control", "variant_id": "control_best", "overall": False},
]


@pytest.mark.parametrize("case", QUALIFICATION_CASES, ids=lambda c: json.dumps(c, sort_keys=True))
async def test_summary_qualification_equals_upstream(
    case: dict[str, Any], upstream_modules: Loader, tmp_path: Path
) -> None:
    upstream = upstream_modules("src.variants_pipeline.qualification_summary")
    contract = upstream_modules("src.variants_pipeline.rubric_contract")
    _write_qualified_program(tmp_path / "up", "b/program_1", contract, **case)
    _write_qualified_program(tmp_path / "port", "b/program_1", contract, **case)
    workspace = LocalWorkspace(tmp_path / "port")
    expected = upstream.summary_qualification(tmp_path / "up/b/program_1/analysis/run_summary.csv")
    assert await summary_qualification(workspace, "b/program_1", prompts=PROMPTS) == expected
    upstream.publish_summary_qualification(tmp_path / "up/b/program_1")
    await publish_summary_qualification(workspace, "b/program_1", prompts=PROMPTS)
    assert _snapshot(tmp_path / "port") == _snapshot(tmp_path / "up")


async def test_qualification_agrees_on_a_program_directory_that_is_not_numbered(
    upstream_modules: Loader, tmp_path: Path
) -> None:
    upstream = upstream_modules("src.variants_pipeline.qualification_summary")
    contract = upstream_modules("src.variants_pipeline.rubric_contract")
    for name in ("up", "port"):
        _write_qualified_program(tmp_path / name, "b/program_x1", contract)
    workspace = LocalWorkspace(tmp_path / "port")
    expected = upstream.summary_qualification(tmp_path / "up/b/program_x1/analysis/run_summary.csv")
    assert expected == {"overall_pass": ""}
    assert await summary_qualification(workspace, "b/program_x1", prompts=PROMPTS) == expected


# --- aggregation --------------------------------------------------------------------------------


def _build_tree(root: Path, contract: ModuleType) -> None:
    summary = ["variant_id", "score"]
    _put(
        root,
        "out/variants/control_best/task_bundle/program_0/analysis/run_summary.csv",
        _csv(summary, ["control_best", "1"]),
    )
    _put(
        root,
        "out/variants/control_best/task_bundle/program_0/analysis/metric_values.csv",
        _csv(
            ["variant_id", "metric_name", "metric_value"], ["control_best", "combined_score", "1"]
        ),
    )
    _put(
        root,
        "out/variants/abl/task_bundle/program_1/analysis/run_summary.csv",
        _csv(summary, ["abl", "0"]),
    )
    _write_qualified_program(
        root / "out/variants/abl/task_bundle", "program_2", contract, variant_id="abl"
    )
    _put(
        root,
        "out/variants/abl/task_bundle/program_2/analysis/metric_values.csv",
        _csv(["variant_id", "metric_name", "extra_metric"], ["abl", "loss", "x"]),
    )
    _put(
        root,
        "out/variants/abl/task_bundle/program_10/analysis/timeseries.csv",
        _csv(
            ["evaluation_id", "variant_id", "series_name", "step", "value"],
            ["e", "abl", "s", "0", "1.5"],
        ),
    )
    _put(
        root,
        "out/variants/abl/task_bundle/program_9/analysis/timeseries.csv",
        _csv(
            ["evaluation_id", "variant_id", "series_name", "step", "value"],
            ["e", "abl", "s", "0", "9"],
        ),
    )
    _put(
        root,
        "out/variants/abl/task_bundle/program_x/analysis/run_summary.csv",
        _csv(summary, ["ignored", "7"]),
    )
    _write_qualified_program(
        root / "out/variants/other/task_bundle",
        "program_1",
        contract,
        variant_id="other",
        flaw="attempt",
    )
    _put(root, "out/variants/nobundle/readme.txt", "x")
    _put(root, "out/variants/loose_file.csv", "a,b\r\n1,2\r\n")
    _put(
        root,
        "out/variants/_scratch/task_bundle/program_1/analysis/run_summary.csv",
        _csv(summary, ["_scratch", "4"]),
    )
    _put(
        root,
        "out/variants/spaced/task_bundle/program_1/analysis/run_summary.csv",
        _csv([" variant_id ", " score "], ["spaced", "3"]),
    )
    _put(
        root,
        "out/variants/ragged/task_bundle/program_1/analysis/run_summary.csv",
        "variant_id,score,note\r\nragged\r\nr2,5,n,extra\r\n",
    )
    _put(
        root,
        "out/variants/quoted/task_bundle/program_1/analysis/run_summary.csv",
        'variant_id,score\r\n"q,1","line1\r\nline2"\r\n',
    )
    _put(
        root,
        "out/variants/bom/task_bundle/program_1/analysis/run_summary.csv",
        b"\xef\xbb\xbfvariant_id,score\r\nbom,6\r\n",
    )
    _put(
        root,
        "out/variants/latin/task_bundle/program_1/analysis/run_summary.csv",
        b"variant_id,score\r\nlat\xe9,1\r\n",
    )
    _put(root, "out/variants/empty/task_bundle/program_1/analysis/run_summary.csv", b"")
    _put(
        root,
        "out/variants/headeronly/task_bundle/program_1/analysis/run_summary.csv",
        _csv(["variant_id", "score", "zeta"]),
    )
    _put(
        root,
        "out/variants/lf/task_bundle/program_1/analysis/run_summary.csv",
        _csv(summary, ["lf", "8"], newline="\n"),
    )

    _put(root, "out/extra_tasks/flat/algo_b/run_summary.csv", _csv(["score"], ["2"]))
    _put(root, "out/extra_tasks/flat/algo_a/run_summary.csv", _csv(["score", "w"], ["1", "z"]))
    _put(root, "out/extra_tasks/flat/_hidden/run_summary.csv", _csv(["score"], ["99"]))
    _put(root, "out/extra_tasks/flat/notes.txt", "n")
    _put(root, "out/extra_tasks/flat/algo_a/metric_values.csv", _csv(["metric_name"], ["m"]))
    _put(
        root,
        "out/extra_tasks/nested/algo/rep_2/run_timeseries.csv",
        _csv(["step", "v"], ["1", "b"]),
    )
    _put(root, "out/extra_tasks/nested/algo/rep_1/run_timeseries.csv", _csv(["step"], ["0"]))
    _put(root, "out/extra_tasks/nested/algo/rep_3/other.csv", _csv(["step"], ["3"]))
    _put(root, "out/extra_tasks/nested/algo2/rep_a.b/run_timeseries.csv", _csv(["step"], ["5"]))
    _put(root, "out/extra_tasks/nested/algo2/repx/run_timeseries.csv", _csv(["step"], ["6"]))
    _put(root, "out/extra_tasks/nested/_skip/rep_1/run_timeseries.csv", _csv(["step"], ["7"]))
    _put(root, "out/extra_tasks/both/algo/rep_1/run_timeseries.csv", _csv(["step"], ["0"]))
    _put(root, "out/extra_tasks/both/algo/run_summary.csv", _csv(["score"], ["1"]))
    _put(
        root,
        "out/extra_tasks/struct/variants/v/task_bundle/program_1/analysis/timeseries.csv",
        _csv(["variant_id"], ["v"]),
    )
    _put(
        root,
        "out/extra_tasks/struct/variants/v/task_bundle/program_1/analysis/run_summary.csv",
        _csv(["variant_id", "score"], ["v", "3"]),
    )
    _put(root, "out/extra_tasks/emptytask/readme.md", "nothing")
    _put(root, "out/extra_tasks/onlyunderscore/_x/run_summary.csv", _csv(["score"], ["1"]))
    _put(root, "out/aggregated/stale/old.csv", "old")


def _relative_results(results: dict[str, dict[str, Any]], root: Path) -> dict[str, Any]:
    def relative(value: Any) -> Any:
        if value is None:
            return None
        return Path(value).relative_to(root).as_posix() if Path(value).is_absolute() else str(value)

    return {
        task: {name: relative(path) for name, path in files.items()}
        for task, files in results.items()
    }


async def test_aggregation_equals_upstream_on_a_tree_of_every_layout(
    upstream_aggregate: ModuleType, upstream_modules: Loader, tmp_path: Path
) -> None:
    contract = upstream_modules("src.variants_pipeline.rubric_contract")
    _build_tree(tmp_path / "up", contract)
    _build_tree(tmp_path / "port", contract)
    expected = upstream_aggregate.aggregate_variant_csvs(tmp_path / "up/out", "orig")
    actual = await aggregate_variant_csvs(
        LocalWorkspace(tmp_path / "port"), PROMPTS, "out", original_task_name="orig"
    )
    assert _relative_results(actual, Path("")) == _relative_results(expected, tmp_path / "up")
    assert _snapshot(tmp_path / "port") == _snapshot(tmp_path / "up")
    aggregated = _snapshot(tmp_path / "port/out/aggregated")
    assert "orig/run_summary.csv" in aggregated and "run_summary.csv" in aggregated
    summary = list(csv.DictReader(io.StringIO(aggregated["orig/run_summary.csv"].decode())))
    assert {row["overall_pass"] for row in summary} == {"True", ""}


@pytest.mark.parametrize("layout", ["empty_dir", "no_variants", "only_extra_nothing"])
async def test_aggregation_without_sources_equals_upstream(
    layout: str, upstream_aggregate: ModuleType, tmp_path: Path
) -> None:
    for name in ("up", "port"):
        (tmp_path / name / "out").mkdir(parents=True)
        if layout == "no_variants":
            _put(tmp_path / name, "out/other/file.csv", "a\r\n1\r\n")
        if layout == "only_extra_nothing":
            _put(tmp_path / name, "out/extra_tasks/x/readme.txt", "r")
    expected = upstream_aggregate.aggregate_variant_csvs(tmp_path / "up/out")
    actual = await aggregate_variant_csvs(LocalWorkspace(tmp_path / "port"), PROMPTS, "out")
    assert actual == expected == {}
    assert _snapshot(tmp_path / "port") == _snapshot(tmp_path / "up")


async def test_aggregation_of_a_primary_task_with_no_csv_removes_the_stale_mirror(
    upstream_aggregate: ModuleType, tmp_path: Path
) -> None:
    for name in ("up", "port"):
        _put(tmp_path / name, "out/aggregated/run_summary.csv", "stale")
        _put(tmp_path / name, "out/variants/v/task_bundle/program_0/readme.txt", "x")
    upstream_aggregate.aggregate_variant_csvs(tmp_path / "up/out")
    await aggregate_variant_csvs(LocalWorkspace(tmp_path / "port"), PROMPTS, "out")
    assert _snapshot(tmp_path / "port") == _snapshot(tmp_path / "up")
    assert not (tmp_path / "port/out/aggregated/run_summary.csv").exists()


# --- canonical analysis -------------------------------------------------------------------------

ENV = {
    "EVALUATION_ID": "v1_eval_1",
    "VARIANT_ID": "v1",
    "VARIANT_KIND": "ablation",
    "BASE_PROGRAM": "best",
    "AGENT_NAME": "model-x",
}
TS_HEADER = ["evaluation_id", "variant_id", "series_name", "step", "value"]

EVIDENCE = {
    "schema_version": 1,
    "episodes": [
        {
            "seed": 7,
            "score": 1.5,
            "correct": True,
            "duration_seconds": 2.0,
            "candidate_timeseries": [
                {"series_name": "reward", "step": 0, "value": 0.25},
                {"series_name": "reward", "step": 1, "value": 3},
            ],
        },
        {"seed": 8, "score": 2.0, "correct": False, "duration_seconds": 1.0},
    ],
}

METRICS_CASES: dict[str, Any] = {
    "plain": {"combined_score": 2.5},
    "nested": {"combined_score": 1, "a": {"y": 2.0, "x": {"z": 3}}, "b": 4},
    "sorted_primary_first": {"zz": 1, "aa": 2, "combined_score": 0.5},
    "skipped_kinds": {"combined_score": 1, "s": "t", "n": None, "b": True, "l": [1, 2]},
    "integers_and_floats": {"combined_score": 3, "i": 10**20, "f": 1e-7, "neg": -0.0},
    "no_combined_score": {"loss": 1.0},
    "nan_value": {"combined_score": float("nan")},
    "inf_value": {"combined_score": 1, "x": float("inf")},
    "huge_int": {"combined_score": 10**400},
    "nested_nan": {"combined_score": 1, "d": {"e": float("-inf")}},
    "unsafe_name": {"combined_score": 1, "bad name": 2},
    "unsafe_nested": {"combined_score": 1, "d": {"bad/name!": 2}},
    "empty_name": {"combined_score": 1, "": 2},
    "long_name": {"combined_score": 1, "n" * 161: 2},
    "unicode_name": {"combined_score": 1, "é": 2},
    "deep": {"combined_score": 1, "a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": {"i": 1}}}}}}}}},
    "depth_eight": {"combined_score": 1, "a": {"b": {"c": {"d": {"e": {"f": {"g": {"h": 1}}}}}}}},
}


def _results_dir(
    root: Path, *, files: dict[str, bytes | str], metrics: Any, correct: Any, evidence: Any = None
) -> None:
    results = root / "res"
    results.mkdir(parents=True, exist_ok=True)
    _put(root, "res/metrics.json", json.dumps(metrics, allow_nan=True))
    _put(root, "res/correct.json", json.dumps(correct))
    if evidence is not None:
        _put(root, "res/evaluation_evidence.json", json.dumps(evidence))
    for name, data in files.items():
        _put(root, f"res/{name}", data)


async def _canonical_pair(
    upstream_outputs: ModuleType,
    tmp_path: Path,
    *,
    files: dict[str, bytes | str] | None = None,
    metrics: Any = None,
    correct: Any = None,
    evidence: Any = None,
    environment: dict[str, str] | None = None,
) -> tuple[dict[str, bytes] | str, dict[str, bytes] | str]:
    """Run upstream's and the port's canonical analysis; return the CSVs or the error message."""
    metrics = {"combined_score": 2.5} if metrics is None else metrics
    correct = {"correct": True} if correct is None else correct
    for name in ("up", "port"):
        _results_dir(
            tmp_path / name, files=files or {}, metrics=metrics, correct=correct, evidence=evidence
        )
    environment = environment or ENV
    try:
        upstream_outputs.canonicalize_results(tmp_path / "up/res", "program_1", environment)
        expected: dict[str, bytes] | str = _snapshot(tmp_path / "up/res/analysis")
    except RuntimeError as error:
        expected = str(error)
    workspace = LocalWorkspace(tmp_path / "port")
    result = EvaluationResult.model_construct(
        evaluation_id="e",
        status="completed",
        metrics=metrics,
        correct=correct.get("correct"),
        output_dir="res",
        evidence=None if evidence is None else EvaluationEvidence.model_validate(evidence),
    )
    try:
        await write_canonical_analysis(
            workspace, program_dir="b/program_1", result=result, environment=environment
        )
        actual: dict[str, bytes] | str = _snapshot(tmp_path / "port/b/program_1/analysis")
    except RuntimeError as error:
        actual = str(error)
    return expected, actual


@pytest.mark.parametrize("name", sorted(METRICS_CASES))
async def test_numeric_metrics_and_the_canonical_files_equal_upstream(
    name: str, upstream_outputs: ModuleType, tmp_path: Path
) -> None:
    metrics = METRICS_CASES[name]
    try:
        expected: Any = upstream_outputs._numeric_metrics(metrics)
    except (RuntimeError, OverflowError) as error:
        expected = (type(error), str(error))
    try:
        actual: Any = numeric_metrics(metrics)
    except (RuntimeError, OverflowError) as error:
        actual = (type(error), str(error))
    assert actual == expected
    if name == "huge_int":
        assert expected[0] is OverflowError
        return
    expected_files, actual_files = await _canonical_pair(
        upstream_outputs, tmp_path, metrics=metrics
    )
    assert actual_files == expected_files


async def test_the_canonical_timeseries_equal_upstream(
    upstream_outputs: ModuleType, tmp_path: Path
) -> None:
    files: dict[str, bytes | str] = {
        "analysis/timeseries.csv": _csv(
            TS_HEADER,
            ["x", "y", "loss", "0", "1.5"],
            ["x", "y", "loss", "1", "nan"],
            ["x", "y", "flag", "2", "TRUE"],
            ["x", "y", "flag", "3", " false "],
            ["x", "y", "candidate/seed_7/own", "0", "5"],
            ["x", "y", "cnt", "4.0", "7"],
        ),
        "runs/rep_1/analysis/timeseries.csv": _csv(TS_HEADER, ["x", "y", "train", "5", "0.5"]),
        "deep/er/analysis/timeseries.csv": _csv(
            TS_HEADER, ["x", "y", "candidate/seed_9/z", "0", "1"]
        ),
        "ignored/timeseries.csv": _csv(TS_HEADER, ["x", "y", "nope", "0", "1"]),
        "other.csv": "a\r\n1\r\n",
    }
    expected, actual = await _canonical_pair(
        upstream_outputs, tmp_path, files=files, evidence=EVIDENCE
    )
    assert isinstance(expected, dict) and actual == expected
    rows = list(csv.DictReader(io.StringIO(expected["timeseries.csv"].decode())))
    names = [row["series_name"] for row in rows]
    assert names.count("candidate/seed_7/reward") == 2 and "candidate/seed_7/own" not in names
    assert "deep/er/candidate/seed_9/z" in names and "runs/rep_1/train" in names


async def test_the_canonical_timeseries_without_evidence_keep_the_reserved_namespace(
    upstream_outputs: ModuleType, tmp_path: Path
) -> None:
    files: dict[str, bytes | str] = {
        "analysis/timeseries.csv": _csv(TS_HEADER, ["x", "y", "candidate/seed_7/own", "0", "5"]),
    }
    expected, actual = await _canonical_pair(upstream_outputs, tmp_path, files=files)
    assert isinstance(expected, dict) and actual == expected
    assert b"candidate/seed_7/own" in expected["timeseries.csv"]


TS_FAILURES: dict[str, bytes | str] = {
    "bad_header": _csv(["a", "b"], ["1", "2"]),
    "reordered_header": _csv(
        ["variant_id", "evaluation_id", "series_name", "step", "value"], ["a", "b", "c", "0", "1"]
    ),
    "extra_field": "evaluation_id,variant_id,series_name,step,value\r\nx,y,s,0,1,extra\r\n",
    "long_field": _csv(TS_HEADER, ["x", "y", "s", "0", "1" + "0" * 512]),
    "unsafe_series": _csv(TS_HEADER, ["x", "y", "bad series", "0", "1"]),
    "negative_step": _csv(TS_HEADER, ["x", "y", "s", "-1", "1"]),
    "fractional_step": _csv(TS_HEADER, ["x", "y", "s", "1.5", "1"]),
    "nan_step": _csv(TS_HEADER, ["x", "y", "s", "nan", "1"]),
    "text_step": _csv(TS_HEADER, ["x", "y", "s", "one", "1"]),
    "text_value": _csv(TS_HEADER, ["x", "y", "s", "0", "many"]),
    "empty_value": _csv(TS_HEADER, ["x", "y", "s", "0", ""]),
    "inf_value": _csv(TS_HEADER, ["x", "y", "s", "0", "inf"]),
    "nan_value_only": _csv(TS_HEADER, ["x", "y", "s", "0", "NaN"]),
    "short_row": "evaluation_id,variant_id,series_name,step,value\r\nx,y,s\r\n",
    "empty_file": b"",
    "not_utf8": b"evaluation_id,variant_id,series_name,step,value\r\nx,y,s\xff,0,1\r\n",
    "header_only": _csv(TS_HEADER),
    "bom": b"\xef\xbb\xbf" + _csv(TS_HEADER, ["x", "y", "s", "0", "1"]),
}


@pytest.mark.parametrize("name", sorted(TS_FAILURES))
async def test_a_program_timeseries_is_validated_as_upstream_validates_it(
    name: str, upstream_outputs: ModuleType, tmp_path: Path
) -> None:
    expected, actual = await _canonical_pair(
        upstream_outputs, tmp_path, files={"analysis/timeseries.csv": TS_FAILURES[name]}
    )
    assert actual == expected


@pytest.mark.parametrize(
    "case",
    [
        {"correct": {"correct": "yes"}},
        {"correct": {"correct": 1}},
        {"correct": {}},
        {"correct": {"correct": False}},
        {"metrics": {"x": 1}},
        {"environment": {**ENV, "EVALUATION_ID": "bad id"}},
        {"environment": {**ENV, "VARIANT_KIND": ""}},
        {"environment": {k: v for k, v in ENV.items() if k != "AGENT_NAME"}},
        {"environment": {**ENV, "BASE_PROGRAM": "a" * 161}},
        {"environment": {**ENV, "VARIANT_ID": "ok:/|-._9"}},
    ],
    ids=lambda case: json.dumps(case, sort_keys=True),
)
async def test_canonical_identity_and_contract_errors_equal_upstream(
    case: dict[str, Any], upstream_outputs: ModuleType, tmp_path: Path
) -> None:
    expected, actual = await _canonical_pair(upstream_outputs, tmp_path, **case)
    assert actual == expected


async def test_the_row_limit_and_the_bound_on_entries_equal_upstream(
    upstream_outputs: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import langgraph_agentic_scientist.variants.analysis as port

    monkeypatch.setattr(upstream_outputs, "MAX_ANALYSIS_ROWS", 3)
    monkeypatch.setattr(port, "MAX_ANALYSIS_ROWS", 3)
    rows = [["x", "y", "s", str(i), "1"] for i in range(4)]
    expected, actual = await _canonical_pair(
        upstream_outputs, tmp_path, files={"analysis/timeseries.csv": _csv(TS_HEADER, *rows)}
    )
    assert actual == expected and isinstance(expected, str)
    monkeypatch.setattr(upstream_outputs, "MAX_SANDBOX_ENTRIES", 4)
    monkeypatch.setattr(port, "MAX_SANDBOX_ENTRIES", 4)
    files: dict[str, bytes | str] = {f"f{i}.txt": "x" for i in range(5)}
    other = tmp_path / "second"
    other.mkdir()
    expected, actual = await _canonical_pair(upstream_outputs, other, files=files)
    assert actual == expected and isinstance(expected, str)


def test_metric_sorting_places_the_primary_first_and_keeps_names_unique_per_prefix(
    upstream_outputs: ModuleType,
) -> None:
    metrics = {"combined_score": 1, "combined_score_b": 2, "a": {"combined_score": 3}}
    assert numeric_metrics(metrics) == upstream_outputs._numeric_metrics(metrics)


# --- selection: hard caps, decisions, the selection file ----------------------------------------


def _port_candidate(variant_id: str, kind: str, **fields: Any) -> VariantCandidate:
    return VariantCandidate(variant_id=variant_id, title=f"t {variant_id}", kind=kind, **fields)


def _upstream_candidate(models: ModuleType, port: VariantCandidate) -> Any:
    return models.VariantCandidate(
        variant_id=port.variant_id,
        title=port.title,
        kind=port.kind,
        base_program=port.base_program,
        hypothesis=port.hypothesis,
        relevance_reason=port.relevance_reason,
        feasibility_reason=port.feasibility_reason,
        target_components=list(port.target_components),
        implementation_plan=list(port.implementation_plan),
        expected_effect=port.expected_effect,
        description_body=port.description_body,
    )


def _ids(selection: VariantSelection | Any) -> tuple[Any, ...]:
    accepted = [c.variant_id for c in selection.accepted]
    rejected = [
        (item.candidate.variant_id, item.reason)
        if isinstance(item, tuple | list) is False and hasattr(item, "candidate")
        else (item[0].variant_id, item[1])
        for item in selection.rejected
    ]
    return accepted, rejected, list(selection.controls)


@pytest.mark.parametrize("seed", range(60))
def test_hard_caps_equal_upstream_on_random_candidates(
    seed: int, upstream_select: ModuleType, upstream_modules: Loader
) -> None:
    models = upstream_modules("src.variants_pipeline.models")
    rng = random.Random(seed)
    kinds = ["ablation", "baseline", "ablation", "baseline", "control", "other"]
    port_candidates = [
        _port_candidate(f"c{i}", rng.choice(kinds), description_body=rng.choice(["", "body"]))
        for i in range(rng.randint(0, 12))
    ]
    decisions: dict[str, tuple[bool, str]] = {}
    for candidate in port_candidates:
        draw = rng.random()
        if draw < 0.2:
            continue
        decisions[candidate.variant_id] = (draw > 0.45, f"reason {candidate.variant_id}")
    decisions["unknown_id"] = (False, "no candidate")
    caps = {"max_ablations": rng.randint(0, 4), "max_extra_baselines": rng.randint(0, 4)}
    expected = upstream_select._apply_hard_caps(
        candidates=[_upstream_candidate(models, c) for c in port_candidates],
        llm_decisions=decisions,
        **caps,
    )
    actual = apply_hard_caps(port_candidates, decisions, **caps)
    assert _ids(actual) == _ids(expected)
    assert not isinstance(actual.controls, tuple)


def test_hard_caps_without_a_decision_accept_and_a_cap_of_zero_rejects(
    upstream_select: ModuleType, upstream_modules: Loader
) -> None:
    models = upstream_modules("src.variants_pipeline.models")
    port = [_port_candidate("a", "ablation"), _port_candidate("b", "baseline")]
    expected = upstream_select._apply_hard_caps(
        candidates=[_upstream_candidate(models, c) for c in port],
        llm_decisions={},
        max_ablations=0,
        max_extra_baselines=0,
    )
    actual = apply_hard_caps(port, {}, max_ablations=0, max_extra_baselines=0)
    assert (
        _ids(actual)
        == _ids(expected)
        == (
            [],
            [("a", "ablation cap reached (0)"), ("b", "baseline cap reached (0)")],
            ["control_best"],
        )
    )


def test_the_selection_file_equals_upstream(
    upstream_select: ModuleType, upstream_modules: Loader, tmp_path: Path
) -> None:
    models = upstream_modules("src.variants_pipeline.models")
    port = [
        _port_candidate("gen", "ablation", hypothesis="h"),
        _port_candidate("custom", "baseline", description_body="body"),
        _port_candidate("rej", "ablation"),
    ]
    selection = apply_hard_caps(
        port,
        {"rej": (False, "not useful: ünï")},
        max_ablations=3,
        max_extra_baselines=3,
    )
    upstream_selection = upstream_select._apply_hard_caps(
        candidates=[_upstream_candidate(models, c) for c in port],
        llm_decisions={"rej": (False, "not useful: ünï")},
        max_ablations=3,
        max_extra_baselines=3,
    )
    upstream_select._write_selection_json(tmp_path, upstream_selection)
    written = (tmp_path / "variant_selection.json").read_text(encoding="utf-8")
    assert json.dumps(selection_json(selection), indent=2) == written
    empty = VariantSelection()
    upstream_select._write_selection_json(
        tmp_path, models.VariantSelection(accepted=[], rejected=[], controls=["control_best"])
    )
    assert json.dumps(selection_json(empty), indent=2) == (
        tmp_path / "variant_selection.json"
    ).read_text(encoding="utf-8")


def test_the_decisions_the_reviewer_leaves_out_are_accepted_as_upstream_accepts_them() -> None:
    answer = VariantDecisionsAnswer.model_validate(
        {"decisions": [{"variant_id": "a", "accepted": False, "reason": "no"}, {"variant_id": "b"}]}
    )
    candidates = [_port_candidate(name, "ablation") for name in ("a", "b", "c")]
    assert decisions_from_answer(candidates, answer) == {
        "a": (False, "no"),
        "b": (True, ""),
        "c": (True, "accepted (not evaluated by agent)"),
    }


# --- generation: the bundle's candidates --------------------------------------------------------


def _bundle(**overrides: Any) -> dict[str, Any]:
    entry = {
        "variant_id": "ablate_x",
        "title": "Ablate x",
        "kind": "ablation",
        "base_program": "best",
        "hypothesis": "h",
        "relevance_reason": "r",
        "feasibility_reason": "f",
        "target_components": ["x", 3],
        "implementation_plan": ["a", "b"],
        "expected_effect": "e",
    }
    return {"decision": "propose_variants", "summary": "s", "candidates": [entry], **overrides}


def _upstream_parse(upstream_generate: ModuleType, raw: dict[str, Any]) -> list[dict[str, Any]]:
    _, candidates = upstream_generate._parse_variant_bundle(raw)
    return [
        {key: value for key, value in vars(candidate).items() if key != "reference_dir"}
        for candidate in candidates
    ]


def _port_parse(raw: dict[str, Any]) -> list[dict[str, Any]]:
    screened = screen_candidates(VariantBundleAnswer.model_validate(raw))
    return [
        {key: value for key, value in candidate.model_dump().items() if key != "reference_dir"}
        for candidate in screened.candidates
    ]


def test_a_wellformed_bundle_is_parsed_as_upstream_parses_it(upstream_generate: ModuleType) -> None:
    raw = _bundle()
    raw["candidates"].append({"variant_id": "base_y", "title": "Y", "kind": "baseline"})
    raw["candidates"][0]["target_components"] = ["x", "3"]
    assert _port_parse(raw) == _upstream_parse(upstream_generate, raw)
    assert [c["variant_id"] for c in _port_parse(raw)] == ["ablate_x", "base_y"]


@pytest.mark.parametrize(
    "candidates", [[], [{"variant_id": "a", "title": "A", "kind": "ablation"}]]
)
def test_no_relevant_variants_give_no_candidates_whatever_the_list_holds(
    candidates: list[dict[str, str]], upstream_generate: ModuleType
) -> None:
    raw = _bundle(decision="no_relevant_variants", candidates=candidates)
    assert _port_parse(raw) == _upstream_parse(upstream_generate, raw) == []


def test_the_screening_of_ids_is_the_declared_difference(upstream_generate: ModuleType) -> None:
    entries = [
        {"variant_id": "ok", "title": "t", "kind": "ablation"},
        {"variant_id": "Bad-Id", "title": "t", "kind": "ablation"},
        {"variant_id": "control_best", "title": "t", "kind": "baseline"},
        {"variant_id": "ok", "title": "again", "kind": "baseline"},
    ]
    raw = _bundle(candidates=entries)
    assert [c["variant_id"] for c in _upstream_parse(upstream_generate, raw)] == [
        "ok",
        "Bad-Id",
        "control_best",
        "ok",
    ]
    screened = screen_candidates(VariantBundleAnswer.model_validate(raw))
    assert [c.variant_id for c in screened.candidates] == ["ok"]
    assert screened.dropped == [
        ("Bad-Id", "malformed variant id"),
        ("control_best", "reserved variant id"),
        ("ok", "duplicate variant id"),
    ]


def test_an_entry_the_structured_answer_cannot_hold_is_the_declared_difference(
    upstream_generate: ModuleType,
) -> None:
    raw = _bundle(candidates=[{"variant_id": "a", "title": "t", "kind": "tuning"}])
    assert [c["kind"] for c in _upstream_parse(upstream_generate, raw)] == ["tuning"]
    with pytest.raises(ValueError, match="kind"):
        VariantBundleAnswer.model_validate(raw)
    raw = _bundle(
        candidates=[
            {"title": "no id", "kind": "ablation"},
            {"variant_id": "b", "title": "t", "kind": "ablation"},
        ]
    )
    assert [c["variant_id"] for c in _upstream_parse(upstream_generate, raw)] == ["b"]
    with pytest.raises(ValueError, match="variant_id"):
        VariantBundleAnswer.model_validate(raw)


def test_an_absent_decision_is_no_relevant_variants(upstream_generate: ModuleType) -> None:
    raw: dict[str, Any] = {"candidates": [{"variant_id": "a", "title": "t", "kind": "ablation"}]}
    assert _upstream_parse(upstream_generate, raw) == []
    assert _port_parse(raw) == []
