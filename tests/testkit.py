"""Helpers shared by the test packages (imported as ``testkit``; ``tests`` is on ``pythonpath``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
