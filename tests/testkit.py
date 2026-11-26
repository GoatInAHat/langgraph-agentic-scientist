"""Helpers shared by the test packages (imported as ``testkit``; ``tests`` is on ``pythonpath``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from langgraph_agentic_scientist import ScientistConfig
from langgraph_agentic_scientist.config import AgentsConfig, ModelRetryConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fast_config(**updates: Any) -> ScientistConfig:
    """A configuration whose in-session model retries do not sleep."""
    agents = AgentsConfig(
        model_retry=ModelRetryConfig(max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0)
    )
    return ScientistConfig(agents=agents, **updates)


def thread(name: str) -> dict[str, Any]:
    """The run config of a thread."""
    return {"configurable": {"thread_id": name}}


@dataclass
class SessionRun:
    name: str
    run_id: UUID
    parent_run_id: UUID | None
    tags: list[str]
    metadata: dict[str, Any]


class SessionRoots(BaseCallbackHandler):
    """Records every run started under the package's session run name."""

    run_inline = True

    def __init__(self) -> None:
        self.runs: list[SessionRun] = []
        self.parents: dict[UUID, UUID | None] = {}

    def _record(self, kwargs: dict[str, Any]) -> None:
        run_id = kwargs["run_id"]
        self.parents[run_id] = kwargs.get("parent_run_id")
        name = str(kwargs.get("name") or "")
        if name.startswith("agsci:"):
            self.runs.append(
                SessionRun(
                    name=name,
                    run_id=run_id,
                    parent_run_id=kwargs.get("parent_run_id"),
                    tags=list(kwargs.get("tags") or []),
                    metadata=dict(kwargs.get("metadata") or {}),
                )
            )

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        self._record(kwargs)

    def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        self._record(kwargs)

    def is_root(self, run: SessionRun) -> bool:
        session_ids = {item.run_id for item in self.runs}
        parent = run.parent_run_id
        while parent is not None:
            if parent in session_ids:
                return False
            parent = self.parents.get(parent)
        return True
