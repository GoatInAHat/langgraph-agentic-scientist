"""Fixtures of the whole-flow tests: a task in the workspace, a ledger and stubbed pipelines."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from pipeline_stubs import CONTRACT_FIXTURE, Ledger, stub_overrides
from testkit import REPO_ROOT, thread

from langgraph_agentic_scientist import GraphOverrides, LocalWorkspace, build_pipeline_graph

__all__ = ["CONTRACT_FIXTURE", "thread"]


@pytest.fixture
def circpack(workspace: LocalWorkspace) -> str:
    """Copy the vendored circle-packing example into the workspace; return its path there."""
    target = workspace.root / "inputs" / "circpack"
    shutil.copytree(
        REPO_ROOT / "examples" / "circpack", target, ignore=shutil.ignore_patterns("__pycache__")
    )
    return "inputs/circpack"


@pytest.fixture
def contracted(workspace: LocalWorkspace) -> str:
    """A task with a contract that requires a final evaluation (and a scaffold)."""
    target = workspace.root / "inputs" / "contracted"
    shutil.copytree(CONTRACT_FIXTURE, target, ignore=shutil.ignore_patterns("__pycache__"))
    (target / "capture.py").write_text("# capture\n")
    return "inputs/contracted"


@pytest.fixture
def ledger() -> Ledger:
    """A fresh call ledger for the stub phases."""
    return Ledger()


@pytest.fixture
def make_app(
    ledger: Ledger,
) -> Callable[..., CompiledStateGraph[Any, Any, Any, Any]]:
    """Build a pipeline whose phases are stubs; ``results`` replaces the default results."""

    def build(
        results: Mapping[str, Any] | None = None,
        *,
        slots: Mapping[str, Any] | None = None,
        saver: Any = None,
        overrides: GraphOverrides | None = None,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        return build_pipeline_graph(
            checkpointer=saver or InMemorySaver(),
            overrides=overrides or stub_overrides(ledger, results, slots=slots),
        )

    return build
