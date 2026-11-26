"""Paper data from the results of the phases: grounding ids, rows, evidence, final, study."""

from __future__ import annotations

import json
from typing import Any, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from test_data_support import (
    EVIDENCE,
    ablation_bundle,
    fast_config,
    final_result,
    investigation,
    source_ids,
    write_json,
    write_text,
)

from langgraph_agentic_scientist import LocalWorkspace, ScientistContext
from langgraph_agentic_scientist.core.results import DownstreamSelection, PaperData
from langgraph_agentic_scientist.core.runconfig import record_run_config
from langgraph_agentic_scientist.core.state import ProgramRef
from langgraph_agentic_scientist.paper.data import (
    PaperDataError,
    build_paper_data,
    check_final_outcome_policy,
    paper_data_node,
    stage_study_sources,
    verify_study_sources,
    write_paper_data,
)
from langgraph_agentic_scientist.testing import ScriptedModels


async def test_an_investigation_becomes_the_upstream_rows_and_grounding_ids(
    workspace: LocalWorkspace,
) -> None:
    result = await investigation(workspace)
    data = await build_paper_data(workspace, investigation=result)
    assert isinstance(data, PaperData) and data.paper_profile == "default"
    assert data.totals == {
        "ideas": 2,
        "attempts": 2,
        "evaluated_programs": 4,
        "correct_programs": 3,
    }
    # rows are ordered by idea directory, then program index, whatever the records' order
    assert [(row["idea_dir_name"], row["program_id"]) for row in data.all_programs] == [
        ("01_alpha", "program_1"),
        ("01_alpha", "program_2"),
        ("02_beta", "program_1"),
        ("02_beta", "program_2"),
    ]
    first = data.all_programs[0]
    assert first == {
        "idea_id": "alpha",
        "idea_dir_name": "01_alpha",
        "attempt_id": "alpha_attempt_01",
        "attempt_dir_name": "idea_root",
        "program_id": "program_1",
        "program_idx": 1,
        "combined_score": 0.3,
        "correct": True,
        "eligible": True,
        "program_dir": workspace.host_path("results/ideas/01_alpha/program_1"),
        "evaluation_evidence": {"status": "unavailable"},
        "eval_index": 1,
    }
    assert [row["eval_index"] for row in data.all_programs] == [1, 2, 3, 4]
    assert data.all_programs[2]["correct"] is False and data.all_programs[2]["eligible"] is False
    # the pending evaluation (program_3) is not an evaluated program
    assert data.best_correct is not None and data.best_correct["combined_score"] == 0.9
    assert data.best_any is not None and data.best_any["program_id"] == "program_2"
    assert [row["combined_score"] for row in data.top_programs] == [0.9, 0.5, 0.3]
    assert [row["id"] for row in data.improvement_trace] == [
        "result.improvement_001",
        "result.improvement_002",
        "result.improvement_003",
    ]
    assert [row["step_index"] for row in data.improvement_trace] == [1, 2, 4]
    assert [row["x_index"] for row in data.improvement_trace] == [0, 1, 3]
    assert data.idea_titles == ["Alpha idea", "Beta idea"]
    assert source_ids(data) == [
        "result.best_any_score",
        "result.best_correct_score",
        "result.improvement_001",
        "result.improvement_002",
        "result.improvement_003",
        "result.top_program_01",
        "result.top_program_02",
        "result.top_program_03",
        "result.total_attempts",
        "result.total_correct_programs",
        "result.total_evaluated_programs",
        "result.total_ideas",
    ]
    sources = data.grounding_sources
    assert sources["result.best_correct_score"].value == 0.9
    assert sources["result.best_correct_score"].model_extra["program_id"] == "program_2"
    assert sources["result.total_evaluated_programs"].type == "count"
    assert sources["result.total_evaluated_programs"].value == 4
    assert data.model_extra["evidence_sources"]["investigation"] == {
        "status": "included",
        "path": workspace.host_path("results"),
    }
    assert data.ablation_studies == []
    assert data.presentation_contract == {
        "variant_unit": "idea",
        "run_unit": "program",
        "internal_fields": ["correct", "eligible"],
    }


async def test_the_standalone_profile_adds_the_presentation_contract(
    workspace: LocalWorkspace,
) -> None:
    result = await investigation(workspace)
    data = await build_paper_data(workspace, investigation=result, paper_profile="standalone")
    contract = data.presentation_contract
    assert contract is not None and contract["variant_unit"] == "model_variant"
    assert "combined_score" in contract["forbidden_visible_terms"]
    assert contract["preferred_terms"]["program"] == "seeded run"
    with pytest.raises(PaperDataError, match="paper_profile must be one of"):
        await build_paper_data(workspace, investigation=result, paper_profile="other")  # type: ignore[arg-type]


async def test_evaluation_evidence_is_attached_and_validated(workspace: LocalWorkspace) -> None:
    result = await investigation(workspace, with_evidence=True)
    data = await build_paper_data(workspace, investigation=result)
    evidence = data.all_programs[3]["evaluation_evidence"]
    assert evidence["status"] == "available" and evidence["episode_count"] == 2
    assert evidence["episodes"] == EVIDENCE["episodes"]
    assert evidence["candidate_diagnostics_trust"] == "unverified"
    assert evidence["path"] == workspace.host_path(
        "results/ideas/02_beta/program_2/evaluation_evidence.json"
    )
    assert evidence["sha256"].startswith("sha256:")
    bad = dict(EVIDENCE, episodes=[])
    await write_json(workspace, "results/ideas/02_beta/program_2/evaluation_evidence.json", bad)
    with pytest.raises(ValueError, match="empty"):
        await build_paper_data(workspace, investigation=result)


async def test_a_replaced_evidence_reader_is_used(workspace: LocalWorkspace) -> None:
    result = await investigation(workspace)
    calls: list[str] = []

    async def reader(_reader: Any, program_dir: str) -> bytes | None:
        calls.append(program_dir)
        return json.dumps(EVIDENCE).encode() if program_dir.endswith("program_1") else None

    data = await build_paper_data(workspace, investigation=result, evidence_reader=reader)
    assert sorted(set(calls)) == [
        "results/ideas/01_alpha/program_1",
        "results/ideas/01_alpha/program_2",
        "results/ideas/02_beta/program_1",
        "results/ideas/02_beta/program_2",
    ]
    assert [row["evaluation_evidence"]["status"] for row in data.all_programs] == [
        "available",
        "unavailable",
        "available",
        "unavailable",
    ]


async def test_an_evidence_directory_with_a_symlink_is_refused(workspace: LocalWorkspace) -> None:
    result = await investigation(workspace)
    import os

    root = workspace.root  # type: ignore[attr-defined]
    target = root / "results" / "elsewhere"
    target.mkdir()
    link = root / "results" / "ideas" / "01_alpha" / "program_1"
    for child in link.iterdir():
        child.unlink()
    link.rmdir()
    os.symlink(target, link)
    with pytest.raises(ValueError, match="contains symlink"):
        await build_paper_data(workspace, investigation=result)


async def test_no_evaluated_programs_is_an_error_unless_allowed(workspace: LocalWorkspace) -> None:
    with pytest.raises(PaperDataError, match="No evaluated programs found"):
        await build_paper_data(workspace)
    empty = await build_paper_data(workspace, allow_empty=True)
    assert empty.totals["evaluated_programs"] == 0 and empty.best_correct is None
    assert empty.grounding_sources["result.total_ideas"].value == 0


async def test_incomplete_or_unscored_evaluations_are_not_rows(workspace: LocalWorkspace) -> None:
    result = await investigation(workspace)
    broken = result.evaluations[0].model_copy(update={"metrics": {}, "evaluation_id": "x"})
    failed = result.evaluations[1].model_copy(update={"status": "failed", "evaluation_id": "y"})
    baseline = result.evaluations[2].model_copy(update={"idea_id": None, "evaluation_id": "z"})
    odd = result.evaluations[3].model_copy(
        update={
            "program": ProgramRef(path="results/ideas/01_alpha/other_dir"),
            "evaluation_id": "w",
        }
    )
    mixed = result.model_copy(
        update={"evaluations": [*result.evaluations, broken, failed, baseline, odd]}
    )
    data = await build_paper_data(workspace, investigation=mixed)
    assert data.totals["evaluated_programs"] == 4


# --- the sealed final evaluation ---


async def test_the_sealed_final_is_grounded(workspace: LocalWorkspace) -> None:
    await write_json(workspace, "final/results/evaluation_evidence.json", EVIDENCE)
    result = await investigation(workspace)
    data = await build_paper_data(workspace, investigation=result, final=final_result())
    final = data.model_extra["final_evaluation"]
    assert (
        final["status"] == "completed"
        and final["correct"] is True
        and final["outcome"] == "qualified"
    )
    assert final["metrics"] == {"score": 0.9, "Edge Case!": 1.0}
    assert final["manifest_path"] == workspace.host_path("final/final_manifest.json")
    assert final["program_tree_sha256"] == "sha256:program"
    assert final["evaluation_evidence"]["status"] == "available"
    assert "limitations" not in final
    sources = data.grounding_sources
    assert sources["final.metric.score"].value == 0.9
    assert sources["final.metric.edge_case"].value == 1.0  # the name is normalised
    assert sources["final.metric.score"].model_extra["evaluation_scope"] == "sealed_final"
    episodes = sources["final.evidence.episodes"].model_extra
    assert episodes["json_pointer"] == "/episodes" and episodes["episode_count"] == 2
    for name in ("status", "correct", "outcome"):
        assert sources[f"final.qualification.{name}"].type == "final_qualification"
    assert sources["final.qualification.outcome"].value == "qualified"


async def test_a_final_without_evidence_has_no_episode_source(workspace: LocalWorkspace) -> None:
    result = await investigation(workspace)
    data = await build_paper_data(workspace, investigation=result, final=final_result())
    assert "final.evidence.episodes" not in data.grounding_sources
    assert data.model_extra["final_evaluation"]["evaluation_evidence"] == {"status": "unavailable"}


async def test_a_not_qualified_final_needs_the_policy(workspace: LocalWorkspace) -> None:
    result = await investigation(workspace)
    not_qualified = final_result(correct=False)
    with pytest.raises(PaperDataError, match="Final evaluation did not qualify for publication"):
        await build_paper_data(workspace, investigation=result, final=not_qualified)
    data = await build_paper_data(
        workspace,
        investigation=result,
        final=not_qualified,
        final_outcome_policy="allow_not_qualified",
    )
    final = data.model_extra["final_evaluation"]
    assert final["outcome"] == "not_qualified" and len(final["limitations"]) == 3
    assert "did not qualify" in final["limitations"][0]
    assert data.grounding_sources["final.qualification.correct"].value is False
    with pytest.raises(PaperDataError):
        check_final_outcome_policy(data.model_dump(mode="json"), "qualified_only")
    check_final_outcome_policy(data.model_dump(mode="json"), "allow_not_qualified")
    check_final_outcome_policy({"x": 1}, "qualified_only")


async def test_final_metric_names_must_not_collide(workspace: LocalWorkspace) -> None:
    result = await investigation(workspace)
    colliding = final_result(metrics={"A-b": 1.0, "a_b": 2.0})
    with pytest.raises(PaperDataError, match="collide after grounding-ID normalization"):
        await build_paper_data(workspace, investigation=result, final=colliding)


# --- a downstream study without a winner ---


async def selection_file(workspace: LocalWorkspace) -> dict[str, Any]:
    await write_text(workspace, "downstream/program/src/program.py", "def handle(request): ...\n")
    await write_text(workspace, "downstream/program/__pycache__/x.pyc", "junk")
    await write_text(workspace, "downstream/program/metrics.json", "{}")
    await write_text(workspace, "baseline/program_0/src/program.py", "def handle(request): ...\n")
    selection: dict[str, Any] = {
        "selection_kind": "best_available",
        "investigation_outcome": "completed_no_winner",
        "combined_score": 0.6,
        "baseline_score": 0.5,
        "source_attempt_id": "alpha_attempt_01",
        "source_program_dir": "results/ideas/01_alpha/program_1",
        "program_dir": "downstream/program",
        "entrypoint_path": "src/program.py",
        "program_tree_sha256": await workspace.digest("downstream/program"),
        "source_tree_sha256": "sha256:source",
        "baseline_program_dir": "baseline/program_0",
        "baseline_tree_sha256": await workspace.digest("baseline/program_0"),
        "attempt_counts": {"total": 3, "scored": 2, "unscored": 1},
    }
    await write_json(workspace, "downstream/.downstream_selection.json", selection)
    return selection


async def test_a_downstream_study_adds_its_context_and_grounding(workspace: LocalWorkspace) -> None:
    selection = await selection_file(workspace)
    downstream = DownstreamSelection(
        status="selected",
        program=ProgramRef(path="downstream/program"),
        source="candidate",
        selection_path="downstream/.downstream_selection.json",
    )
    result = await investigation(workspace, pending=False)
    data = await build_paper_data(workspace, investigation=result, downstream=downstream)
    study = data.model_extra["downstream_study"]
    assert study["selection_kind"] == "best_available" and study[
        "search_score_delta"
    ] == pytest.approx(0.1)
    assert study["final_evaluation"] is None and len(study["limitations"]) == 4
    sources = data.grounding_sources
    assert sources["study.search_score"].value == 0.6
    assert sources["study.search_score_delta"].value == pytest.approx(0.1)
    assert sources["study.attempts.scored"].value == 2
    assert sources["study.selection_kind"].model_extra["evaluation_scope"] == "search"
    # with a final evaluation the study records its outcome, even when not qualified
    both = await build_paper_data(
        workspace, investigation=result, downstream=downstream, final=final_result(correct=False)
    )
    assert both.model_extra["downstream_study"]["final_evaluation"] == {
        "status": "completed",
        "correct": False,
        "outcome": "not_qualified",
    }
    assert both.grounding_sources["study.final.correct"].value is False
    # a study may be written without evaluated programs
    only = await build_paper_data(workspace, downstream=downstream)
    assert only.totals["evaluated_programs"] == 0 and "downstream_study" in only.model_extra
    assert selection["baseline_program_dir"]


async def test_a_downstream_study_without_a_baseline_has_no_baseline_facts(
    workspace: LocalWorkspace,
) -> None:
    selection = await selection_file(workspace)
    selection.update({"baseline_program_dir": None, "baseline_tree_sha256": None})
    await write_json(workspace, "downstream/.downstream_selection.json", selection)
    downstream = DownstreamSelection(
        status="selected", selection_path="downstream/.downstream_selection.json"
    )
    data = await build_paper_data(workspace, downstream=downstream)
    study = data.model_extra["downstream_study"]
    assert study["baseline_score"] is None and study["search_score_delta"] is None
    assert "study.baseline_score" not in data.grounding_sources
    assert "A valid scored baseline is unavailable" in study["limitations"][-1]
    missing = DownstreamSelection(status="selected", selection_path="downstream/missing.json")
    with pytest.raises(PaperDataError, match="invalid downstream selection manifest"):
        await build_paper_data(workspace, downstream=missing)


async def test_study_sources_are_staged_once_and_verified(workspace: LocalWorkspace) -> None:
    selection = await selection_file(workspace)
    downstream = DownstreamSelection(
        status="selected", selection_path="downstream/.downstream_selection.json"
    )
    data = await build_paper_data(workspace, downstream=downstream)
    paper_data = data.model_dump(mode="json")
    await stage_study_sources(paper_data, workspace, "paper", selection)
    assert paper_data["downstream_study"]["source_evidence"] == {
        "program": "source_evidence/downstream/program",
        "baseline": "source_evidence/downstream/baseline",
    }
    staged = "paper/source_evidence/downstream"
    assert await workspace.exists(f"{staged}/program/src/program.py")
    assert await workspace.exists(f"{staged}/baseline/src/program.py")
    assert not await workspace.exists(f"{staged}/program/__pycache__/x.pyc")  # code only
    assert not await workspace.exists(f"{staged}/program/metrics.json")  # evaluator output excluded
    await stage_study_sources(paper_data, workspace, "paper", selection)  # idempotent
    await write_json(workspace, "paper/paper_data.json", paper_data)
    await verify_study_sources(workspace, "paper", selection)
    await write_text(workspace, f"{staged}/program/src/program.py", "tampered")
    with pytest.raises(ValueError, match="differ from frozen selection"):
        await verify_study_sources(workspace, "paper", selection)
    with pytest.raises(ValueError, match="differ from frozen selection"):
        await stage_study_sources(paper_data, workspace, "paper", selection)
    changed = dict(paper_data["downstream_study"], combined_score=0.1)
    await write_json(workspace, "paper/paper_data.json", {"downstream_study": changed})
    await write_text(workspace, f"{staged}/program/src/program.py", "def handle(request): ...\n")
    with pytest.raises(
        ValueError, match="study context differs from frozen selection: combined_score"
    ):
        await verify_study_sources(workspace, "paper", selection)
    with pytest.raises(ValueError, match="without a frozen selection"):
        await verify_study_sources(workspace, "paper", None)


async def test_study_sources_are_refused_when_the_program_changed(
    workspace: LocalWorkspace,
) -> None:
    selection = await selection_file(workspace)
    data = await build_paper_data(
        workspace,
        downstream=DownstreamSelection(
            status="selected", selection_path="downstream/.downstream_selection.json"
        ),
    )
    await write_text(workspace, "downstream/program/src/program.py", "edited")
    with pytest.raises(ValueError, match="program source digest differs from frozen selection"):
        await stage_study_sources(data.model_dump(mode="json"), workspace, "paper", selection)


# --- the file and the slot node ---


async def test_paper_data_is_written_with_sorted_keys(workspace: LocalWorkspace) -> None:
    data = await build_paper_data(workspace, investigation=await investigation(workspace))
    path = await write_paper_data(workspace, "paper", data)
    assert path == "paper/paper_data.json"
    text = (await workspace.read_bytes(path)).decode()
    stored = json.loads(text)
    assert list(stored) == sorted(stored) and stored["totals"]["evaluated_programs"] == 4
    assert PaperData.model_validate(stored).totals == data.totals
    assert stored["grounding_sources"]["result.top_program_01"]["id"] == "result.top_program_01"


class NodeState(TypedDict, total=False):
    run_config: Any
    investigation: Any
    downstream: Any
    variants: Any
    final: Any
    paper_data: Any
    paper_dir: str
    failure: Any


def node_app() -> Any:
    graph: StateGraph[Any, Any, Any, Any] = StateGraph(NodeState, context_schema=ScientistContext)
    graph.add_node("record_run_config", record_run_config)
    graph.add_node("data", paper_data_node)
    graph.add_edge(START, "record_run_config")
    graph.add_edge("record_run_config", "data")
    graph.add_edge("data", END)
    return graph.compile()


async def test_the_slot_node_builds_the_paper_data_from_the_state(
    workspace: LocalWorkspace,
) -> None:
    ctx = ScientistContext(
        models=ScriptedModels.from_scripts({}), workspace=workspace, config=fast_config()
    )
    result = await investigation(workspace)
    out = await node_app().ainvoke({"investigation": result, "paper_dir": "paper"}, context=ctx)
    assert out["paper_data"].totals["evaluated_programs"] == 4
    assert await workspace.exists("paper/paper_data.json")
    # a prebuilt paper data is kept; the final-outcome gate applies to it either way
    given = out["paper_data"]
    kept = await node_app().ainvoke({"paper_data": given}, context=ctx)
    assert kept["paper_data"] is given
    failing = PaperData.model_validate(
        {
            **given.model_dump(mode="json"),
            "final_evaluation": {"correct": False, "manifest_path": "m"},
        }
    )
    refused = await node_app().ainvoke({"paper_data": failing}, context=ctx)
    assert refused["failure"].category == "paper_data_failed"
    assert "did not qualify" in refused["failure"].detail


async def test_the_slot_node_reports_a_paper_data_that_cannot_be_built(
    workspace: LocalWorkspace,
) -> None:
    ctx = ScientistContext(
        models=ScriptedModels.from_scripts({}), workspace=workspace, config=fast_config()
    )
    out = await node_app().ainvoke({}, context=ctx)
    assert "paper_data" not in out
    assert out["failure"].category == "paper_data_failed"
    assert "No evaluated programs found" in out["failure"].detail


async def test_a_nested_ablation_bundle_is_merged_into_the_investigation(
    workspace: LocalWorkspace,
) -> None:
    variants = await ablation_bundle(workspace)
    result = await investigation(workspace, pending=False)
    data = await build_paper_data(workspace, investigation=result, variants=variants)
    sources = data.model_extra["evidence_sources"]
    assert sources["ablations"] == {"status": "included", "path": workspace.host_path("abl")}
    assert data.totals["ablation_variants"] == 2 and data.totals["ablation_runs"] == 4
    assert data.totals["evaluated_programs"] == 8 and data.totals["correct_programs"] == 6
    assert [row["eval_index"] for row in data.all_programs] == list(range(1, 9))
    assert data.idea_titles[-2:] == ["Control", "Without X"]
    assert len(data.ablation_studies) == 1
    assert "result.total_ablation_variants" in data.grounding_sources
    # a bundle that is not one is reported as degraded
    degraded = variants.model_copy(update={"output_dir": "nowhere"})
    data = await build_paper_data(workspace, investigation=result, variants=degraded)
    assert data.model_extra["evidence_sources"]["ablations"]["status"] == "missing"
    await write_text(workspace, "plain/readme.md", "x")
    data = await build_paper_data(
        workspace,
        investigation=result,
        variants=variants.model_copy(update={"output_dir": "plain"}),
    )
    assert data.model_extra["evidence_sources"]["ablations"] == {
        "status": "degraded",
        "path": workspace.host_path("plain"),
        "reason": "no validated ablation bundle",
    }
