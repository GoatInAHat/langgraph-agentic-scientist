"""The names that callers implementing evaluators, slots and workspaces need are public."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

PUBLIC = {
    "langgraph_agentic_scientist": [
        "BestManifest",
        "CompileResult",
        "DRAW_INPUTS_SLOT",
        "DrawInputsInput",
        "DrawInputsOutput",
        "FinalEvaluationError",
        "GroundingSource",
        "ProgramRef",
        "ProgramSpecError",
        "ReferenceQuery",
        "ReferenceRecord",
        "TASK_SNAPSHOT_DIR",
        "TaskContract",
        "TaskContractError",
        "TaskRef",
        "load_task_contract",
        "reseal_task_snapshot",
    ],
    "langgraph_agentic_scientist.evaluation": ["campaign_baseline_id", "id_to_path"],
    "langgraph_agentic_scientist.final_evaluation": [
        "DRAW_INPUTS_SLOT",
        "DrawInputsInput",
        "DrawInputsOutput",
        "evaluator_digest",
        "task_digest",
    ],
    "langgraph_agentic_scientist.workspace": [
        "AGENT_ARTIFACTS_DIR",
        "AGENT_ARTIFACTS_ROUTE",
        "WorkspacePathError",
    ],
}

README = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("module", "name"), [(module, name) for module, names in PUBLIC.items() for name in names]
)
def test_name_is_exported_and_documented(module: str, name: str) -> None:
    imported = importlib.import_module(module)
    assert name in imported.__all__
    assert getattr(imported, name) is not None
    assert re.search(rf"`[^`]*\b{re.escape(name)}\b[^`]*`", README), name


def test_exported_values() -> None:
    from langgraph_agentic_scientist import DRAW_INPUTS_SLOT, TASK_SNAPSHOT_DIR
    from langgraph_agentic_scientist.workspace import AGENT_ARTIFACTS_DIR, AGENT_ARTIFACTS_ROUTE

    assert DRAW_INPUTS_SLOT == "final_evaluation.draw_inputs"
    assert TASK_SNAPSHOT_DIR == "task"
    assert AGENT_ARTIFACTS_ROUTE == f"/{AGENT_ARTIFACTS_DIR}/"


@pytest.mark.parametrize(
    "module",
    [
        "langgraph_agentic_scientist",
        "langgraph_agentic_scientist.evaluation",
        "langgraph_agentic_scientist.final_evaluation",
        "langgraph_agentic_scientist.workspace",
        "langgraph_agentic_scientist.testing",
    ],
)
def test_every_listed_name_resolves(module: str) -> None:
    imported = importlib.import_module(module)
    for name in imported.__all__:
        assert hasattr(imported, name), name
