"""Fixtures of the variants tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from langgraph_agentic_scientist import ScientistContext
from langgraph_agentic_scientist.core.state import ProgramRef, TaskRef
from langgraph_agentic_scientist.testing import FakeEvaluator
from langgraph_agentic_scientist.workspace import LocalWorkspace

from .helpers import make_study, variants_config


@pytest.fixture
async def study(workspace: LocalWorkspace) -> tuple[TaskRef, ProgramRef]:
    """A task and a best program in the workspace."""
    return await make_study(workspace)


@pytest.fixture
def phase_context(make_context: Callable[..., ScientistContext]) -> Callable[..., ScientistContext]:
    """Build a context for the variants phase: scripted roles, a fake evaluator, fast retries."""

    def build(
        scripts: Any,
        *,
        evaluator: Any = None,
        **variants: Any,
    ) -> ScientistContext:
        return make_context(
            scripts,
            evaluator=evaluator if evaluator is not None else FakeEvaluator(),
            config=variants_config(**variants),
        )

    return build
