"""The shipped workspace contract holds for LocalWorkspace and for the in-memory fake."""

from __future__ import annotations

from pathlib import Path

import pytest

from langgraph_agentic_scientist.testing import MemoryWorkspace
from langgraph_agentic_scientist.testing.workspace_contract import WorkspaceContract
from langgraph_agentic_scientist.workspace import LocalWorkspace, Workspace


class TestLocalWorkspaceContract(WorkspaceContract):
    @pytest.fixture
    def contract_workspace(self, tmp_path: Path) -> Workspace:
        return LocalWorkspace(tmp_path / "ws")


class TestMemoryWorkspaceContract(WorkspaceContract):
    runs_commands = False

    @pytest.fixture
    def contract_workspace(self) -> Workspace:
        return MemoryWorkspace()
