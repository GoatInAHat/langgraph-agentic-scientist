"""The run configuration: recorded by the first node, verified on every later entry.

The context (models, workspace, configuration, prompts) is passed again on every invocation and
is not checkpointed, so a resumed run could be given a different configuration. The first node of
every graph that can run alone, :func:`record_run_config`, therefore writes a
:class:`~langgraph_agentic_scientist.core.state.RunConfigRecord` into state: the method part of the
configuration, its SHA-256, the SHA-256 of every prompt and skill text the run can use (after
overrides), the package version and the state schema version. When a record is already present
(a resume, or a phase mounted under a parent that recorded it), the node recomputes both
fingerprints and raises :class:`~langgraph_agentic_scientist.core.errors.ConfigurationMismatch`,
naming what differs. Runtime-only settings may change. Nodes read limits through
:func:`method_config`, never from ``runtime.context.config``.

This generalises upstream's trusted resume policy (``pipeline/resume_config.py``
``write_resume_policy`` and ``read_idea_cycle_budget``), which records only the idea-cycle budget.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from importlib import metadata
from typing import Any

from langgraph.runtime import Runtime

from ..config import ScientistConfig
from ..context import ScientistContext
from .errors import ConfigurationMismatch, IncompatibleCheckpoint
from .state import STATE_SCHEMA_VERSION, RunConfigRecord

SUPPORTED_STATE_SCHEMA_VERSIONS = frozenset({STATE_SCHEMA_VERSION})
"""State schema versions this release reads (and migrates, when a migration is added)."""


def package_version() -> str:
    """Return the installed version of the package (``0.0.0`` when not installed)."""
    try:
        return metadata.version("langgraph-agentic-scientist")
    except metadata.PackageNotFoundError:
        return "0.0.0"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_config(context: ScientistContext) -> RunConfigRecord:
    """Return the record of a context's method configuration and prompt texts."""
    method = context.config.method_dump()
    return RunConfigRecord(
        method_config=method,
        config_sha256=_canonical_sha256(method),
        prompts_sha256=context.prompts.fingerprint(),
        prompt_digests=context.prompts.digests(),
        package_version=package_version(),
        state_schema_version=STATE_SCHEMA_VERSION,
    )


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


def differing_fields(recorded: RunConfigRecord, current: RunConfigRecord) -> list[str]:
    """Return the dotted configuration fields and prompt keys that differ between two records."""
    differences: list[str] = []
    if recorded.config_sha256 != current.config_sha256:
        before = _flatten(recorded.method_config)
        after = _flatten(current.method_config)
        differences += sorted(
            key for key in before.keys() | after.keys() if before.get(key) != after.get(key)
        )
        if not differences:
            differences.append("config")
    if recorded.prompts_sha256 != current.prompts_sha256:
        before_prompts = recorded.prompt_digests
        after_prompts = current.prompt_digests
        changed = sorted(
            f"prompt:{key}"
            for key in before_prompts.keys() | after_prompts.keys()
            if before_prompts.get(key) != after_prompts.get(key)
        )
        differences += changed or ["prompts"]
    return differences


def check_schema_version(recorded: RunConfigRecord) -> None:
    """Refuse a checkpoint whose state schema this release cannot read.

    Raises:
        IncompatibleCheckpoint: When no migration covers the recorded version.
    """
    if recorded.state_schema_version not in SUPPORTED_STATE_SCHEMA_VERSIONS:
        raise IncompatibleCheckpoint(recorded.state_schema_version, STATE_SCHEMA_VERSION)


def verify_run_config(recorded: RunConfigRecord, context: ScientistContext) -> None:
    """Check a context against a recorded configuration.

    Raises:
        IncompatibleCheckpoint: When the recorded state schema is not supported.
        ConfigurationMismatch: When a method field or a prompt text differs.
    """
    check_schema_version(recorded)
    current = build_run_config(context)
    differences = differing_fields(recorded, current)
    if differences:
        raise ConfigurationMismatch(differences)


async def record_run_config(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Graph node: record the run configuration, or verify the recorded one.

    Add it as the first node of every graph that can run alone. It returns ``{}`` when a record
    is present and matches, so a mounted phase leaves its parent's record unchanged.
    """
    recorded = state.get("run_config")
    if recorded is None:
        return {"run_config": build_run_config(runtime.context)}
    record = (
        recorded
        if isinstance(recorded, RunConfigRecord)
        else RunConfigRecord.model_validate(recorded)
    )
    verify_run_config(record, runtime.context)
    return {}


def method_config(state: Mapping[str, Any]) -> ScientistConfig:
    """Return the recorded method configuration of a run.

    Runtime-only settings are not recorded; read them from ``runtime.context.config``.

    Raises:
        KeyError: When the state has no recorded configuration (the graph lacks
            :func:`record_run_config`).
    """
    recorded = state.get("run_config")
    if recorded is None:
        raise KeyError("run_config is not recorded; add record_run_config as the first node")
    record = (
        recorded
        if isinstance(recorded, RunConfigRecord)
        else RunConfigRecord.model_validate(recorded)
    )
    return ScientistConfig.model_validate(record.method_config)


__all__ = [
    "SUPPORTED_STATE_SCHEMA_VERSIONS",
    "build_run_config",
    "check_schema_version",
    "differing_fields",
    "method_config",
    "package_version",
    "record_run_config",
    "verify_run_config",
]
