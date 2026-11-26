"""The effective configuration of the post-hoc phase."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..config import PosthocConfig
from ..core.runconfig import method_config


def posthoc_config(state: Mapping[str, Any]) -> PosthocConfig:
    """Return the recorded post-hoc configuration of a run, with the integrated pipeline's mode.

    Upstream's pipeline orchestrator runs the phase artifact-only
    (``execution.checkpoint_mode='optional'`` and ``execution.retrain_if_checkpoint_missing=false``,
    ``build_posthoc_cmd``); the input flag ``artifact_only`` selects that mode. Limits are read
    from the configuration recorded by the first node, never from the context.
    """
    config = method_config(state).posthoc
    if state.get("artifact_only"):
        return config.model_copy(
            update={"checkpoint_mode": "optional", "retrain_if_checkpoint_missing": False}
        )
    return config


__all__ = ["posthoc_config"]
