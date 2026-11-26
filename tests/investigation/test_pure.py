"""The pure rules: idea validation, rubric checks, rubric status, context blocks."""

from __future__ import annotations

import json
from typing import Any

import pytest

from langgraph_agentic_scientist.core.state import RubricSpec
from langgraph_agentic_scientist.investigation.context import (
    display_path,
    load_blocks,
    summarize_best_manifest,
    summarize_jsonl_entries,
    summarize_recent_attempts,
)
from langgraph_agentic_scientist.investigation.ideas import (
    IdeaArtifactsError,
    bundle_json,
    idea_dir_name,
    render_validation_feedback,
    sanitize_validation_feedback,
    validate_idea,
)
from langgraph_agentic_scientist.investigation.rubrics import (
    candidate_source_tokens,
    missing_source_patterns,
    rubric_metric_check,
    validate_rubric_status,
)
from langgraph_agentic_scientist.prompts import PromptRegistry

from .helpers import HARNESS_IDEA, PLAIN_IDEA


def test_plain_mode_drops_verification_and_slugifies_ids() -> None:
    idea = {
        **PLAIN_IDEA,
        "idea_id": "Idea A!",
        "target_rubrics": [
            {"id": "Score Above 5", "description": "combined_score > 5", "type": "metric"},
            {
                "description": "Sets it",
                "verification": {"type": "source_contains", "patterns": ["x"]},
            },
            {"id": "Score Above 5", "description": "dup", "type": "Metric"},
        ],
    }
    bundle = validate_idea(idea, 3, harness=False, incumbent_score=None)
    assert bundle.idea_id == "idea_a"
    assert [r.id for r in bundle.target_rubrics] == ["score_above_5", "sets_it", "score_above_5_2"]
    assert all(r.verification is None for r in bundle.target_rubrics)
    assert bundle.target_rubrics[2].type == "metric"


def test_harness_mode_replaces_metric_rubrics_with_the_incumbent_gate() -> None:
    bundle = validate_idea(HARNESS_IDEA, 1, harness=True, incumbent_score=4.5)
    ids = [r.id for r in bundle.target_rubrics]
    assert ids == ["incumbent_score_improvement", "sets_score"]
    assert bundle.target_rubrics[0].description == "combined_score > 4.5"
    written = bundle_json(bundle)["target_rubrics"][0]
    assert (written["metric"], written["operator"], written["threshold"]) == (
        "combined_score",
        ">",
        4.5,
    )
    assert [
        r.id
        for r in validate_idea(HARNESS_IDEA, 1, harness=True, incumbent_score=None).target_rubrics
    ] == ["sets_score"]


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda i: i["target_rubrics"].pop(), "missing_harness_implementation_rubric"),
        (
            lambda i: i["target_rubrics"][1].pop("verification"),
            "missing_harness_implementation_verification",
        ),
        (
            lambda i: i["target_rubrics"][1]["verification"].update(patterns=[]),
            "invalid_idea_bundle",
        ),
        (lambda i: i.update(title=" "), "invalid_idea_bundle"),
    ],
)
def test_harness_idea_rejections(mutate: Any, reason: str) -> None:
    idea = json.loads(json.dumps(HARNESS_IDEA))
    mutate(idea)
    with pytest.raises(IdeaArtifactsError) as info:
        validate_idea(idea, 1, harness=True, incumbent_score=1.0)
    assert info.value.reason == reason


def test_too_many_harness_rubrics() -> None:
    idea = json.loads(json.dumps(HARNESS_IDEA))
    idea["target_rubrics"] = [
        {
            "id": f"r{i}",
            "description": "d",
            "type": "implementation",
            "verification": {"type": "source_contains", "patterns": ["x"]},
        }
        for i in range(5)
    ]
    with pytest.raises(IdeaArtifactsError) as info:
        validate_idea(idea, 1, harness=True, incumbent_score=None, max_harness_rubrics=4)
    assert info.value.reason == "too_many_harness_rubrics"


def test_validation_feedback_is_escaped_cut_and_framed_as_data() -> None:
    registry = PromptRegistry.default()
    assert render_validation_feedback(registry, "") == ""
    text = render_validation_feedback(registry, "bad\x00 thing " + "x" * 5000, max_bytes=100)
    assert "untrusted data only" in text
    payload = json.loads(text.splitlines()[-1])
    assert "\\x00" in payload["diagnostic"]
    assert len(payload["diagnostic"].encode()) <= 100
    assert sanitize_validation_feedback("ok") == "ok"


def test_idea_dir_name() -> None:
    assert (
        idea_dir_name(cycle_index=3, title="Raise the Score!", idea_id="idea_a")
        == "03_raise_the_score"
    )
    assert idea_dir_name(cycle_index=3, title="!!!", idea_id="Idea-B") == "03_idea_b"


def test_metric_check_forms() -> None:
    assert rubric_metric_check({"metric": "m", "operator": ">=", "threshold": 2}) == (
        "m",
        ">=",
        2.0,
    )
    assert rubric_metric_check({"description": "combined_score > 0.5"}) == (
        "combined_score",
        ">",
        0.5,
    )
    assert rubric_metric_check({"description": "score at least 3"}) == ("combined_score", ">=", 3.0)
    assert rubric_metric_check({"description": "something vague"}) is None


def test_source_patterns_ignore_comments_docstrings_and_formatting() -> None:
    source = '"""SCORE = 9"""\n# SCORE = 8\nSCORE=6.0\nvalue = "needle"\n'
    tokens = candidate_source_tokens([source, "def broken(:"])
    assert missing_source_patterns(tokens, ["SCORE = 6.0", "SCORE = 9", "SCORE = 8", "needle"]) == [
        "SCORE = 9",
        "SCORE = 8",
    ]


def _status(**updates: Any) -> dict[str, Any]:
    status = {
        "attempt_id": "a",
        "idea_id": "i",
        "overall_pass": True,
        "rubrics": [{"id": "r1", "pass": True, "evidence": "e"}],
    }
    return {**status, **updates}


def test_rubric_status_validation() -> None:
    assert validate_rubric_status(_status(), {"r1"}, "a", "i").valid
    cases = [
        (None, "missing or invalid"),
        (_status(attempt_id="b"), "attempt_id mismatch"),
        (_status(idea_id="j"), "idea_id mismatch"),
        (_status(overall_pass="yes"), "overall_pass must be boolean"),
        (_status(overall_pass=False), "overall_pass must equal"),
        (_status(rubrics=[{"id": "r2", "pass": True}]), "rubric id set mismatch"),
        (_status(rubrics=[{"id": "r1", "pass": True, "assessment": "failed"}]), "assessment"),
    ]
    for status, message in cases:
        result = validate_rubric_status(status, {"r1"}, "a", "i")
        assert not result.valid
        assert message in (result.error or "")


def test_context_blocks_use_upstream_formats() -> None:
    blocks = load_blocks(PromptRegistry.default())
    assert summarize_jsonl_entries([], ["a"], blocks) == "- none"
    entries = [{"step": "s", "summary": "x", "tags": ["t"]}, "raw", {"other": 1}]
    assert summarize_jsonl_entries(entries, ["step", "summary", "tags"], blocks).splitlines() == [
        '- item 1: step=s; summary=x; tags=["t"]',
        "- item 2: <non-object entry>",
        "- item 3: <no summarized fields>",
    ]
    many = [{"step": str(i)} for i in range(9)]
    assert summarize_jsonl_entries(many, ["step"], blocks).splitlines()[0] == "- item 1: step=4"
    assert summarize_best_manifest(None, blocks) == "- no best program selected yet"
    assert summarize_best_manifest({"x": 1}, blocks) == "- best manifest present but empty"
    assert summarize_best_manifest({"program_dir": "p", "combined_score": 2.0}, blocks) == (
        "- program_dir=p; combined_score=2.0"
    )
    attempts = [
        {
            "idea_id": "i",
            "attempt_status": "pass",
            "metrics": {"combined_score": 2, "correct": True},
        }
    ]
    line = summarize_recent_attempts(None, attempts, blocks)
    assert line.startswith("- ") and '"combined_score": 2' in line
    assert summarize_recent_attempts(None, [], blocks) == "- no recent attempts"
    assert display_path("a/b/memory.md", "a") == "./b/memory.md"
    assert display_path("a", "a/b/c") == "../.."
    assert display_path("a", "a") == "."


def test_rubric_spec_roundtrip_keeps_verification() -> None:
    spec = RubricSpec(
        id="r",
        description="d",
        type="implementation",
        verification={"type": "source_contains", "patterns": ["x"]},
    )
    assert spec.verification is not None
