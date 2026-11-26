"""Concept figures: the request from the strategy, the Autoillustrate renderer, the authorisation
of a repair, and the A8 availability texts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

import pytest

from langgraph_agentic_scientist.core.interfaces import ConceptFigureRequest
from langgraph_agentic_scientist.paper.figures import (
    AutoillustrateRenderer,
    authorize_concept_figure_repair,
    resolve_concept_figure_request,
    select_mode,
)
from langgraph_agentic_scientist.paper.models import ConceptFigureState
from langgraph_agentic_scientist.testing import MemoryWorkspace
from langgraph_agentic_scientist.workspace import CommandResult

SPEC = {
    "figure_id": "overview",
    "figure_type": "diagram",
    "preferred_generator": "generate-concept-figure",
    "purpose": "Show the loop.",
}


async def test_request_comes_from_the_strategy() -> None:
    ws = MemoryWorkspace({"p/paper_strategy.json": json.dumps({"figure_specs": [SPEC]})})
    request = await resolve_concept_figure_request(ws, "p")
    assert request == {
        "figure_id": "overview",
        "output_rel": "figures/overview.png",
        "instructions": "Show the loop.",
        "placement": "half-width",
    }
    ws.files["p/paper_strategy.json"] = json.dumps({"figure_specs": [SPEC, SPEC]}).encode()
    with pytest.raises(ValueError, match="at most one generate-concept-figure spec"):
        await resolve_concept_figure_request(ws, "p")
    optional = {**SPEC, "required": False}
    ws.files["p/paper_strategy.json"] = json.dumps({"figure_specs": [optional]}).encode()
    assert await resolve_concept_figure_request(ws, "p") is None


def test_mode_steps_down_with_the_budget() -> None:
    assert select_mode("full", 4000, full=3600, medium=1800) == "full"
    assert select_mode("full", 2000, full=3600, medium=1800) == "medium"
    assert select_mode("full", 600, full=3600, medium=1800) == "light"
    assert select_mode("full", None, full=3600, medium=1800) == "full"


async def test_autoillustrate_command_and_availability() -> None:
    calls: list[tuple[Sequence[str], str]] = []

    def handler(argv: Sequence[str], cwd: str, env: Mapping[str, str] | None) -> CommandResult:
        calls.append((list(argv), cwd))
        ws.files["p/figures/overview.png"] = b"png"
        return CommandResult(argv=list(argv), returncode=0)

    ws = MemoryWorkspace(run=handler)
    renderer = AutoillustrateRenderer(agent="codex", environment={})
    availability = renderer.availability()
    assert availability.openai is False
    assert availability.openai_context.startswith("OPENAI_API_KEY was not detected")
    assert availability.gemini_context.startswith("GEMINI_API_KEY was not detected")
    result = await renderer.render(
        ConceptFigureRequest(
            paper_dir="p",
            instructions="Show the loop.",
            output_path="figures/overview.png",
            metadata={"placement": "full-width"},
        ),
        workspace=ws,
    )
    assert result.success and result.path == "p/figures/overview.png"
    argv, cwd = calls[0]
    assert cwd == "p"
    assert argv[:4] == ["autoillustrate", "./paper.tex", "--output", "./figures/overview.png"]
    assert "--skip-enhance" in argv
    prompt = argv[argv.index("--prompt") + 1]
    assert prompt == (
        "Methods diagram that will be the main figure of this scientific research paper. "
        "Show the loop."
    )
    keyed = AutoillustrateRenderer(environment={"OPENAI_API_KEY": "sk-test"})
    assert keyed.availability().openai_context.startswith("OPENAI_API_KEY is expected")


async def test_a_failed_render_is_reported_not_raised() -> None:
    ws = MemoryWorkspace()
    renderer = AutoillustrateRenderer(environment={})
    result = await renderer.render(
        ConceptFigureRequest(paper_dir="p", instructions="x", output_path="figures/a.png"),
        workspace=ws,
    )
    assert result.success is False and "exited 127" in result.detail


async def test_audit_findings_about_the_figure_authorise_its_repair() -> None:
    from langgraph_agentic_scientist.paper.figures import request_fingerprint

    ws = MemoryWorkspace({"p/paper_strategy.json": json.dumps({"figure_specs": [SPEC]})})
    request = await resolve_concept_figure_request(ws, "p")
    assert request is not None
    state = ConceptFigureState(
        available=True,
        attempted=True,
        request_fingerprint=request_fingerprint(request),
        rendered_path="stash/overview.png",
    )
    ws.files["p/paper_evidence_audit_feedback.json"] = json.dumps(
        {"findings": [{"id": "f1", "issue": "figure overview mislabels the loop"}]}
    ).encode()
    assert (await authorize_concept_figure_repair(ws, "p", state)).repair_authorized
    ws.files["p/paper_evidence_audit_feedback.json"] = json.dumps(
        {"findings": [{"id": "f1", "issue": "a table"}]}
    ).encode()
    assert not (await authorize_concept_figure_repair(ws, "p", state)).repair_authorized
