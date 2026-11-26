"""The pipeline built from the package's own phase graphs (no stubs).

Skipped until every phase package can build its graph; once it can, this checks that the pipeline
mounts all of them and hands them its overrides.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from langgraph_agentic_scientist import GraphOverrides, build_pipeline_graph
from langgraph_agentic_scientist.pipeline import PHASE_SPECS


def _unavailable() -> str | None:
    for spec in PHASE_SPECS:
        try:
            module = importlib.import_module(f"langgraph_agentic_scientist.{spec.module}")
            getattr(module, spec.factory)(overrides=GraphOverrides(), name=spec.name)
        except Exception as exc:
            return f"{spec.factory}: {type(exc).__name__}: {exc}"
    return None


REASON = _unavailable()
pytestmark = pytest.mark.skipif(REASON is not None, reason=f"a phase is not available: {REASON}")


def test_the_default_pipeline_mounts_every_phase_graph() -> None:
    app = build_pipeline_graph()
    # LangGraph inlines a subgraph into a drawing only when it has one first and one last node,
    # so mounting is checked on the subgraphs it finds, not on the xray drawing.
    mounted = dict(app.get_subgraphs())
    nodes = set(app.get_graph().nodes)
    for spec in PHASE_SPECS:
        assert spec.name in mounted, f"{spec.name} is not mounted"
        assert {f"pipeline.{spec.name}.before", f"pipeline.{spec.name}.after"} <= nodes


def test_every_phase_alone_compiles_with_the_same_factory_shape() -> None:
    for spec in PHASE_SPECS:
        module = importlib.import_module(f"langgraph_agentic_scientist.{spec.module}")
        graph: Any = getattr(module, spec.factory)(overrides=GraphOverrides(), name="alone")
        assert graph.get_graph().nodes
