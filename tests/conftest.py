"""Shared test fixtures: workspaces, fast configurations and contexts with scripted models."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from testkit import fast_config
from upstream_modules import upstream_modules as upstream_modules

from langgraph_agentic_scientist import (
    LocalWorkspace,
    ModelRegistry,
    ScientistContext,
)
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import ScriptedModels


@pytest.fixture
def workspace(tmp_path: Path) -> LocalWorkspace:
    """A fresh host workspace in a temporary directory."""
    return LocalWorkspace(tmp_path / "ws")


@pytest.fixture
def make_context(workspace: LocalWorkspace) -> Callable[..., ScientistContext]:
    """Build a context with scripted models (one script per role) and a fast configuration."""

    def build(
        scripts: Mapping[Role | str, Sequence[Any]] | None = None,
        *,
        models: ModelRegistry | None = None,
        **kwargs: Any,
    ) -> ScientistContext:
        registry = models or ScriptedModels.from_scripts(scripts or {}, context_window=200_000)
        kwargs.setdefault("config", fast_config())
        kwargs.setdefault("workspace", workspace)
        return ScientistContext(models=registry, **kwargs)

    return build


@pytest.fixture(autouse=True)
def _quiet_known_warnings() -> None:
    warnings.filterwarnings("ignore", message=".*has no max_input_tokens profile.*")
