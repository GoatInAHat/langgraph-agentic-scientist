"""Post-hoc analyses as paper data: collection, ranking validation and the added grounding."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

import pytest
from test_data_support import build_paper_data_for_tests, write_json, write_text

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.paper.data import (
    ResultsReader,
    augment_paper_data_with_posthoc,
    collect_posthoc_analyses,
    copy_posthoc_figures,
    discover_posthoc_session,
    validate_posthoc_ranking,
)


async def session(workspace: LocalWorkspace, root: str = "results/posthoc") -> None:
    ranking = {
        "valid": [
            {"analysis_id": "a1", "selected_attempt": "attempt_1", "title": "First", "family": "f"},
            {"analysis_id": "a2", "selected_attempt": "attempt_2"},
            {"analysis_id": "../bad", "selected_attempt": "x"},
            {"analysis_id": "a3", "selected_attempt": "attempt_1"},
        ]
    }
    await write_json(workspace, f"{root}/ranking.json", ranking)
    first = f"{root}/analyses/a1/attempt_1"
    await write_text(workspace, f"{first}/analysis_summary.md", "# Summary of a1\n")
    await write_json(workspace, f"{first}/interestingness_rubric.json", {"score": 3})
    await write_json(
        workspace,
        f"{first}/analysis_manifest.json",
        {
            "outputs": {
                "figures": ["fig.png", "../escape.png", "notes.txt"],
                "data_files": ["table.csv", "stats.json", "missing.json", "../x.json"],
            }
        },
    )
    await workspace.write_bytes(f"{first}/fig.png", b"png")
    await write_text(workspace, f"{first}/table.csv", "a,b\n1,2\n3,4\n")
    await write_json(workspace, f"{first}/stats.json", {"mean": 1})
    await write_text(workspace, f"{first}/run_analysis.py", "print('analysis')\n")
    # a2 has no manifest: its PNGs and JSON files are collected by name
    second = f"{root}/analyses/a2/attempt_2"
    await write_text(workspace, f"{second}/analysis_summary.md", "Summary of a2")
    await workspace.write_bytes(f"{second}/plot.png", b"png")
    await write_json(workspace, f"{second}/curves.json", {"x": [1]})
    await write_json(workspace, f"{second}/rubric_status.json", {"control": True})
    # a3 has no summary and is skipped
    await write_text(workspace, f"{root}/analyses/a3/attempt_1/other.md", "x")


async def test_a_session_is_found_by_directory_manifest_or_results(
    workspace: LocalWorkspace,
) -> None:
    reader = ResultsReader(workspace)
    assert await discover_posthoc_session(reader, "results") is None
    await session(workspace)
    assert await discover_posthoc_session(reader, "results") == "results/posthoc"
    assert await discover_posthoc_session(reader, "results", "results/posthoc") == "results/posthoc"
    assert await discover_posthoc_session(reader, "results", "results/nope") is None
    await session(workspace, "results/posthoc/run_1")
    await write_json(workspace, "results/posthoc/latest_manifest.json", {"output_dir": "run_1"})
    assert await discover_posthoc_session(reader, "results") == "results/posthoc/run_1"
    await write_json(workspace, "results/posthoc/latest_manifest.json", {"output_dir": "../run_1"})
    assert await discover_posthoc_session(reader, "results") == "results/posthoc"


async def test_analyses_are_collected_from_the_manifest_or_by_name(
    workspace: LocalWorkspace,
) -> None:
    await session(workspace)
    analyses = await collect_posthoc_analyses(ResultsReader(workspace), "results")
    assert [analysis["analysis_id"] for analysis in analyses] == ["a1", "a2"]
    first, second = analyses
    assert first["title"] == "First" and first["family"] == "f" and first["rubric"] == {"score": 3}
    assert first["summary_text"] == "# Summary of a1\n"
    assert first["figure_paths"] == [
        workspace.host_path("results/posthoc/analyses/a1/attempt_1/fig.png")
    ]
    assert first["raw_data"]["table.csv"] == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]
    assert (
        first["raw_data"]["stats.json"] == {"mean": 1} and "missing.json" not in first["raw_data"]
    )
    assert first["run_analysis_code"] == "print('analysis')\n"
    assert first["attempt_dir"] == workspace.host_path("results/posthoc/analyses/a1/attempt_1")
    assert second["title"] == "a2" and second["family"] == ""
    assert [path.rsplit("/", 1)[-1] for path in second["figure_paths"]] == ["plot.png"]
    assert list(second["raw_data"]) == ["curves"]  # control files are not raw data
    assert copy_posthoc_figures(analyses) == ["a1", "a2"]


async def test_symlinked_attempts_and_big_csvs_are_not_collected(workspace: LocalWorkspace) -> None:
    import os

    await session(workspace)
    root = workspace.root  # type: ignore[attr-defined]
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "analysis_summary.md").write_text("linked")
    link = root / "results" / "posthoc" / "analyses" / "a2" / "attempt_2"
    for child in link.iterdir():
        child.unlink()
    link.rmdir()
    os.symlink(elsewhere, link)
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["x"])
    for index in range(10_001):
        writer.writerow([index])
    await write_text(
        workspace, "results/posthoc/analyses/a1/attempt_1/table.csv", buffer.getvalue()
    )
    analyses = await collect_posthoc_analyses(ResultsReader(workspace), "results")
    assert [analysis["analysis_id"] for analysis in analyses] == ["a1"]
    assert analyses[0]["raw_data"]["table.csv"] == []  # too many rows: dropped to nothing


def ranked(rank: int, analysis_id: str, **updates: Any) -> dict[str, Any]:
    entry = {
        "rank": rank,
        "analysis_id": analysis_id,
        "relevance_to_thesis": "high",
        "relevance_justification": "why",
        "recommended_usage": "main_results",
        "key_finding_for_paper": "finding",
    }
    entry.update(updates)
    return entry


ANALYSES = [{"analysis_id": "a1", "title": "First"}, {"analysis_id": "a2", "title": "Second"}]


def test_a_valid_ranking_orders_every_analysis_once() -> None:
    result = {"ranking_rationale": "because", "ranked_analyses": [ranked(2, "a2"), ranked(1, "a1")]}
    assert [entry["analysis_id"] for entry in validate_posthoc_ranking(result, ANALYSES)] == [
        "a1",
        "a2",
    ]


def rk(*entries: dict[str, Any], rationale: str = "r") -> dict[str, Any]:
    return {"ranking_rationale": rationale, "ranked_analyses": list(entries)}


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (rk(rationale=""), "ranking_rationale must be a non-empty string"),
        ({"ranking_rationale": "r", "ranked_analyses": "x"}, "ranked_analyses must be a list"),
        (rk(ranked(1, "a1")), "missing analysis IDs: ['a2']"),
        (rk(ranked(1, "a1"), ranked(2, "zz")), "unknown analysis IDs: ['zz']"),
        (rk(ranked(1, "a1"), ranked(1, "a2")), "ranks must be the contiguous integers 1..2"),
        (rk(ranked(1, "a1"), ranked(2, "a2", recommended_usage="x")), "recommended_usage must be"),
        (
            rk(ranked(1, "a1"), ranked(2, "a2", relevance_to_thesis="x")),
            "relevance_to_thesis must be",
        ),
        (
            rk(ranked(1, "a1"), ranked(2, "a2", key_finding_for_paper="")),
            "key_finding_for_paper must",
        ),
        (rk(ranked(1, "a1"), ranked(True, "a2")), "rank must be an integer"),
        (rk(ranked(1, "a1"), ranked(2, "a1")), "duplicate analysis IDs: ['a1']"),
    ],
)
def test_ranking_rules(result: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError) as error:
        validate_posthoc_ranking(result, ANALYSES)
    assert message in str(error.value) and str(error.value).startswith("Invalid posthoc ranking:")


def test_duplicate_collected_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="collected analyses contain duplicate IDs"):
        validate_posthoc_ranking(
            {"ranking_rationale": "r", "ranked_analyses": []}, ANALYSES + ANALYSES[:1]
        )


async def test_ranked_analyses_are_merged_without_changing_existing_evidence(
    workspace: LocalWorkspace,
) -> None:
    await session(workspace)
    analyses = await collect_posthoc_analyses(ResultsReader(workspace), "results")
    paper_data = await build_paper_data_for_tests(workspace)
    before = json.loads(json.dumps(paper_data))
    ranking = {"ranking_rationale": "r", "ranked_analyses": [ranked(2, "a1"), ranked(1, "a2")]}
    merged = augment_paper_data_with_posthoc(paper_data, ranking, analyses)
    assert [item["analysis_id"] for item in merged["posthoc_analyses"]] == ["a2", "a1"]
    assert [item["rank"] for item in merged["posthoc_analyses"]] == [1, 2]
    a1 = merged["posthoc_analyses"][1]
    assert a1["title"] == "First" and a1["summary_text"] == "# Summary of a1\n"
    assert a1["relevance_to_thesis"] == "high" and a1["recommended_usage"] == "main_results"
    assert set(a1) == {
        "rank", "analysis_id", "title", "family", "summary_text", "rubric",
        "relevance_to_thesis", "relevance_justification", "recommended_usage",
        "key_finding_for_paper", "figure_paths", "raw_data", "run_analysis_code",
    }  # fmt: skip
    assert merged["totals"]["posthoc_analyses_total"] == 2
    assert merged["totals"]["posthoc_analyses_ranked"] == 2
    sources = merged["grounding_sources"]
    assert sources["posthoc.a1.summary"] == {
        "id": "posthoc.a1.summary", "type": "posthoc_analysis", "analysis_id": "a1",
        "title": "First", "family": "f",
    }  # fmt: skip
    assert sources["posthoc.a1.data.stats.json"]["type"] == "posthoc_data"
    assert sources["posthoc.a2.data.curves"]["data_file"] == "curves"
    # everything that was there stays, and a second merge changes nothing
    for key, value in before["grounding_sources"].items():
        assert sources[key] == value
    again = augment_paper_data_with_posthoc(merged, ranking, analyses)
    assert again["posthoc_analyses"] == merged["posthoc_analyses"]
    with pytest.raises(ValueError, match="missing analysis IDs"):
        augment_paper_data_with_posthoc(
            paper_data, {"ranking_rationale": "r", "ranked_analyses": []}, analyses
        )
