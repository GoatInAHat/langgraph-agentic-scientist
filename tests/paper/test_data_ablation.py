"""Ablation bundles, cross-task results and multi-task bundles as paper data."""

from __future__ import annotations

import pytest
from test_data_support import (
    CSV_COLUMNS,
    ablation_bundle,
    final_result,
    metric_rows,
    run_rows,
    source_ids,
    write_json,
    write_text,
)

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.core.results import VariantsResult
from langgraph_agentic_scientist.paper.data import (
    ResultsReader,
    build_ablation_paper_data,
    build_multitask_ablation_paper_data,
    build_paper_data,
    collect_cross_task_results,
)


async def test_a_bundle_alone_is_the_paper_data(workspace: LocalWorkspace) -> None:
    variants = await ablation_bundle(workspace)
    data = await build_paper_data(workspace, variants=variants)
    assert data.totals == {
        "ideas": 2,
        "attempts": 2,
        "evaluated_programs": 4,
        "correct_programs": 3,
        "ablation_studies": 1,
        "ablation_variants": 2,
        "ablation_runs": 4,
        "completed_ablation_variants": 2,
        "failed_ablation_variants": 0,
        "timed_out_ablation_variants": 0,
    }
    assert data.results_dir == workspace.host_path("abl")
    assert data.idea_titles == ["Control", "Without X"]
    assert data.best_correct is not None and data.best_correct["combined_score"] == 0.9
    assert data.best_correct["idea_id"] == "control_best"
    assert data.improvement_trace == []
    assert [row["variant_id"] for row in data.all_programs] == [
        "control_best",
        "control_best",
        "no_x",
        "no_x",
    ]
    assert data.all_programs[0]["attempt_id"] == "control_best_attempt_01"
    assert data.all_programs[0]["source_layout"] == "ablation_bundle"
    assert data.all_programs[0]["evaluation_evidence"] == {"status": "unavailable"}
    assert source_ids(data) == [
        "ablation.study_abl.compare_control_best_vs_no_x.combined_score.mean_delta",
        "ablation.study_abl.variant_control_best.combined_score.max",
        "ablation.study_abl.variant_control_best.combined_score.mean",
        "ablation.study_abl.variant_control_best.combined_score.min",
        "ablation.study_abl.variant_control_best.combined_score.n",
        "ablation.study_abl.variant_control_best.combined_score.std",
        "ablation.study_abl.variant_no_x.combined_score.max",
        "ablation.study_abl.variant_no_x.combined_score.mean",
        "ablation.study_abl.variant_no_x.combined_score.min",
        "ablation.study_abl.variant_no_x.combined_score.n",
        "ablation.study_abl.variant_no_x.combined_score.std",
        "result.best_any_score",
        "result.best_correct_score",
        "result.top_program_01",
        "result.top_program_02",
        "result.top_program_03",
        "result.total_ablation_runs",
        "result.total_ablation_variants",
        "result.total_attempts",
        "result.total_correct_programs",
        "result.total_evaluated_programs",
        "result.total_ideas",
    ]
    sources = data.grounding_sources
    mean = sources["ablation.study_abl.variant_control_best.combined_score.mean"]
    assert mean.value == pytest.approx(0.85) and mean.model_extra["summary_field"] == "mean"
    assert sources[
        "ablation.study_abl.variant_control_best.combined_score.std"
    ].value == pytest.approx(0.05)
    assert sources["ablation.study_abl.variant_no_x.combined_score.n"].type == "count"
    delta = sources["ablation.study_abl.compare_control_best_vs_no_x.combined_score.mean_delta"]
    assert delta.value == pytest.approx(-0.2)
    study = data.ablation_studies[0]
    assert study["study_id"] == "abl" and study["title"] == "Ablation of X"
    assert study["anchor_variant_id"] == "control_best" and study["variant_count"] == 2
    variant = study["variants"][1]
    assert variant["variant_id"] == "no_x" and variant["status"] == "completed"
    assert variant["correct_run_count"] == 1 and variant["run_count"] == 2
    assert variant["metric_summaries"]["latency"]["is_primary_metric"] is False
    assert variant["primary_metric_summary"]["mean"] == pytest.approx(0.65)
    assert variant["artifacts"]["aggregated_timeseries"] is None
    assert variant["task_bundle_dir"] == workspace.host_path("abl/variants/no_x/task_bundle")
    assert study["comparisons"][0]["mean_delta"] == pytest.approx(-0.2)


async def test_variants_named_only_by_the_tables_are_found(workspace: LocalWorkspace) -> None:
    await ablation_bundle(workspace)
    reader = ResultsReader(workspace)
    data = await build_ablation_paper_data(reader, "abl")  # no spec names: the tables name them
    assert [v["variant_id"] for v in data["ablation_studies"][0]["variants"]] == [
        "control_best",
        "no_x",
    ]
    assert (await reader.read_csv_rows("abl/aggregated/run_summary.csv"))[0][
        "variant_id"
    ] == "control_best"
    assert await reader.read_json("abl/aggregated/run_summary.csv") is None
    assert await reader.list_directories("abl/variants") == ["control_best", "no_x"]


async def test_cross_task_tables_become_grounding(workspace: LocalWorkspace) -> None:
    variants = await ablation_bundle(workspace)
    summary = run_rows(
        [("control_best", "control", 1.0, "True"), ("control_best", "control", 2.0, "True")]
    )
    await write_text(workspace, "abl/aggregated/other/run_summary.csv", summary)
    await write_text(
        workspace,
        "abl/aggregated/other/metric_values.csv",
        metric_rows([("control_best", "combined_score", 1.0), ("control_best", "Aux Metric", 4.0)]),
    )
    await write_json(
        workspace,
        "abl/extra_tasks/other/control_best/variant_spec.json",
        {"variant_id": "control_best", "kind": "control"},
    )
    data = await build_paper_data(workspace, variants=variants)
    assert data.model_extra["task_names"] == ["other"]
    results = data.model_extra["cross_task_results"]["other"]["control_best"]
    assert results["variant_kind"] == "control" and results["metric_name"] == "combined_score"
    assert (
        results["n"] == 2
        and results["mean"] == pytest.approx(1.5)
        and results["correct_count"] == 2
    )
    sources = data.grounding_sources
    assert sources["result.other.control_best.combined_score.mean"].value == pytest.approx(1.5)
    assert sources["result.other.control_best.combined_score.n"].value == 2
    assert (
        sources["result.other.control_best.aux_metric.mean"].model_extra["metric_name"]
        == "Aux Metric"
    )
    assert sources["result.other.control_best.combined_score.mean"].model_extra["source_csv"] == (
        workspace.host_path("abl/aggregated/other/run_summary.csv")
    )


async def test_a_timeseries_only_task_uses_the_final_step_of_each_run(
    workspace: LocalWorkspace,
) -> None:
    await ablation_bundle(workspace)
    series = [
        "algorithm,replication,metric,step,value",
        "ppo,0,episode_return,1,1.0",
        "ppo,0,episode_return,5,3.0",
        "ppo,1,episode_return,5,5.0",
        "ppo,1,loss,5,9.0",
    ]
    await write_text(workspace, "abl/aggregated/rl/run_timeseries.csv", "\n".join(series) + "\n")
    cross = await collect_cross_task_results(ResultsReader(workspace), "abl")
    assert cross["task_names"] == ["rl"]
    ppo = cross["cross_task_results"]["rl"]["ppo"]
    assert ppo["metric_name"] == "episode_return" and ppo["mean"] == 4.0 and ppo["n"] == 2
    assert ppo["source_layout"] == "run_timeseries_terminal"
    assert cross["grounding_sources"]["result.rl.ppo.episode_return.max"]["value"] == 5.0


async def test_colliding_metric_ids_across_variants_are_refused(workspace: LocalWorkspace) -> None:
    await write_text(
        workspace,
        "abl/aggregated/t/metric_values.csv",
        metric_rows([("v", "A-b", 1.0), ("v", "a_b", 2.0)]),
    )
    with pytest.raises(ValueError, match="metric grounding source ID collision"):
        await collect_cross_task_results(ResultsReader(workspace), "abl")


async def test_a_multitask_bundle_namespaces_its_studies(workspace: LocalWorkspace) -> None:
    await ablation_bundle(workspace)
    # the bundle root is the primary task; extra_tasks/t2 is a second canonical bundle
    await ablation_bundle(workspace, "abl/extra_tasks/t2")
    await write_json(
        workspace,
        "abl/paper_bundle_manifest.json",
        {
            "tasks": {
                "primary": {"role": "primary_root_bundle", "task_dir": "task/"},
                "t2": {"role": "supporting_task", "task_dir": "extra_tasks/t2/"},
                "ghost": {"role": "supporting_task", "task_dir": "extra_tasks/ghost/"},
            }
        },
    )
    reader = ResultsReader(workspace)
    data = await build_multitask_ablation_paper_data(reader, "abl")
    assert [study["study_id"] for study in data["ablation_studies"]] == ["primary", "t2"]
    assert [study["task_role"] for study in data["ablation_studies"]] == [
        "primary_root_bundle",
        "supporting_task",
    ]
    assert data["totals"]["ablation_studies"] == 2 and data["totals"]["ablation_runs"] == 8
    assert data["best_correct"] is None and data["top_programs"] == []
    row = data["all_programs"][0]
    assert row["study_id"] == "primary" and row["idea_id"] == "primary:control_best"
    assert (
        row["program_id"] == "primary:control_best:run_1"
        and row["task_role"] == "primary_root_bundle"
    )
    assert [row["eval_index"] for row in data["all_programs"]] == list(range(1, 9))
    sources = data["grounding_sources"]
    assert "ablation.study_t2.variant_no_x.combined_score.mean" in sources
    assert sources["ablation.study_primary.run_count"]["value"] == 4
    assert sources["result.total_ablation_studies"]["value"] == 2
    assert data["paper_bundle_manifest"]["tasks"]["t2"]["role"] == "supporting_task"
    # through the top-level builder, with the variants result of the root bundle
    top = await build_paper_data(
        workspace,
        variants=VariantsResult(status="completed", output_dir="abl"),
        final=final_result(),
    )
    assert top.totals["ablation_studies"] == 2 and "final.metric.score" in top.grounding_sources


async def test_a_manifest_without_resolvable_bundles_falls_back_to_the_plain_bundle(
    workspace: LocalWorkspace,
) -> None:
    await ablation_bundle(workspace)
    await write_json(workspace, "abl/paper_bundle_manifest.json", {"tasks": {"x": {"role": "r"}}})
    data = await build_multitask_ablation_paper_data(ResultsReader(workspace), "abl")
    assert data["totals"]["ablation_studies"] == 1 and CSV_COLUMNS
