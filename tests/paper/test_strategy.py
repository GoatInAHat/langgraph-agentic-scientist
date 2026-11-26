"""The strategist's validation (``validate_strategy_with_feedback``) and the family resolution."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from test_loop_support import paper_data, strategy

from langgraph_agentic_scientist.paper.models import StrategyAnswer
from langgraph_agentic_scientist.paper.strategy import (
    StrategyInvalid,
    validate_strategy,
    validate_strategy_files,
    write_strategy_artifacts,
)
from langgraph_agentic_scientist.paper.workspace import resolve_final_stage


def _valid() -> dict[str, Any]:
    args = strategy().tool_calls[0]["args"]
    return copy.deepcopy(args["strategy"])


DATA = paper_data().model_dump(mode="json")


def test_a_valid_strategy_passes() -> None:
    validate_strategy(_valid(), DATA)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.update(status="draft"), "must set status to 'ready' or 'fail'"),
        (lambda s: s.update(selected_prompt_family=""), "missing selected_prompt_family"),
        (
            lambda s: s.update(status="fail", failure_reason=""),
            "status=fail requires non-empty failure_reason",
        ),
        (lambda s: s.update(status="fail", failure_reason="weak"), "paper strategy failed: weak"),
        (lambda s: s.update(central_thesis=" "), "missing non-empty field: central_thesis"),
        (lambda s: s.update(supplementary_plan=[]), "must contain a supplementary_plan object"),
        (
            lambda s: s["supplementary_plan"].update(purpose=""),
            "supplementary_plan.purpose must be non-empty",
        ),
        (
            lambda s: s["supplementary_plan"]["planned_subsections"][0].update(source_ids=["nope"]),
            "contains unknown grounding source ids: ['nope']",
        ),
        (lambda s: s.update(method_groups=[]), "must contain non-empty method_groups list"),
        (lambda s: s.update(results_focus="x"), "field must be a list: results_focus"),
        (
            lambda s: s.update(
                figure_specs=[
                    {"figure_id": "f", "figure_type": "bar", "preferred_generator": "excel"}
                ]
            ),
            "preferred_generator has unsupported value: excel",
        ),
        (
            lambda s: s.update(
                figure_specs=[
                    {
                        "figure_id": "f",
                        "figure_type": "diagram",
                        "required": True,
                        "max_panels": 1,
                        "max_categories_per_panel": 1,
                        "max_series_per_panel": 1,
                        "allow_progression": False,
                    }
                ]
            ),
            "missing preferred_generator for required diagram",
        ),
        (
            lambda s: s.update(
                figure_specs=[
                    {
                        "figure_id": "f",
                        "figure_type": "bar",
                        "max_panels": 0,
                        "max_categories_per_panel": 1,
                        "max_series_per_panel": 1,
                        "allow_progression": False,
                    }
                ]
            ),
            "max_panels must be positive int",
        ),
        (
            lambda s: s.update(
                figure_specs=[
                    {
                        "figure_id": "f",
                        "figure_type": "bar",
                        "max_panels": 1,
                        "max_categories_per_panel": 1,
                        "max_series_per_panel": 1,
                        "allow_progression": "no",
                    }
                ]
            ),
            "allow_progression must be bool",
        ),
    ],
)
def test_upstream_checks(mutate: Any, message: str) -> None:
    payload = _valid()
    mutate(payload)
    with pytest.raises(StrategyInvalid, match=message.replace("[", r"\[").replace("]", r"\]")):
        validate_strategy(payload, DATA)


async def test_files_and_feedback(workspace: Any) -> None:
    answer = StrategyAnswer.model_validate(strategy().tool_calls[0]["args"])
    await workspace.write_bytes("p/paper_data.json", json.dumps(DATA).encode())
    await write_strategy_artifacts(workspace, "p", answer)
    assert await validate_strategy_files(workspace, "p") == []
    feedback = json.loads(
        (await workspace.read_bytes("p/strategy_validation_feedback.json")).decode()
    )
    assert feedback["ok"] is True
    await workspace.write_bytes("p/paper_strategy.md", b"  ")
    errors = await validate_strategy_files(workspace, "p")
    assert errors == ["Paper strategy memo is empty: p/paper_strategy.md"]
    markdown = (await workspace.read_bytes("p/strategy_validation_feedback.md")).decode()
    assert "## Hard errors" in markdown


def test_family_resolution_switches_the_prompt_and_retargets_the_writer_skill() -> None:
    skills = ["retrieve-references", "write-paper", "validate-paper", "memory-log"]
    family, key, writer, resolved = resolve_final_stage(
        current_prompt_key="paper/paper", current_skills=skills, selected_family="paper_standalone"
    )
    assert (family, key, writer) == (
        "paper_standalone",
        "paper/paper_standalone",
        "write-paper-standalone",
    )
    assert resolved == [
        "retrieve-references",
        "write-paper-standalone",
        "validate-paper",
        "memory-log",
    ]
    same = resolve_final_stage(
        current_prompt_key="paper/paper",
        current_skills=skills,
        selected_family="paper_detailed_grouped",
    )
    assert same[1] == "paper/paper_detailed_grouped" and same[3] == skills


def test_edit_prompt_is_kept_when_the_writer_skill_matches() -> None:
    skills = ["write-paper", "memory-log"]
    _, key, _, _ = resolve_final_stage(
        current_prompt_key="paper/paper_edit", current_skills=skills, selected_family="paper"
    )
    assert key == "paper/paper_edit"
    _, key, writer, _ = resolve_final_stage(
        current_prompt_key="paper/paper_edit",
        current_skills=skills,
        selected_family="paper_standalone",
    )
    assert key == "paper/paper_standalone" and writer == "write-paper-standalone"


def test_unknown_family_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown selected_prompt_family 'essay'"):
        resolve_final_stage(
            current_prompt_key="paper/paper", current_skills=[], selected_family="essay"
        )
