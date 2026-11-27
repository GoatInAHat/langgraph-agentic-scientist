"""The contract imported whole runs over a fixture that the importing suite defines."""

from __future__ import annotations

from pathlib import Path

import pytest

from langgraph_agentic_scientist.testing.workspace_contract import *
from langgraph_agentic_scientist.workspace import LocalWorkspace, Workspace


@pytest.fixture
def contract_workspace(tmp_path: Path) -> Workspace:
    return LocalWorkspace(tmp_path / "ws")
