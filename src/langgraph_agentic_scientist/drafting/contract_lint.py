"""Lint and mechanical synchronisation of a drafted task contract.

Ports of upstream ``investigation/task_contract_lint.py`` (bounded multi-error findings for the
contract and ``requirements.txt``), ``task_drafting/contract_sync.py`` (refreshing the SHA-256
fields of paths the contract already declares, and the narrow rule that lets a repair remove an
unused dependency pin), ``task_drafting/rpc_transport_policy.py`` (the retired shared-filesystem
names ``EXCHANGE_DIR`` and ``AGENTIC_CANDIDATE_EXCHANGE``) and
``task_drafting/harness_capability_policy.py`` (evaluators must not read ``AGENTIC_HARNESS_*``
values or an invented proxy accessor). The rules and their messages are upstream's; file access
goes through a :class:`~langgraph_agentic_scientist.workspace.Workspace`.

The strict parser of ``task_contract.json`` itself is
:func:`~langgraph_agentic_scientist.core.contract.parse_task_contract`; the lint here reports every
independent finding at once, so that a repair agent can fix all of them in one pass.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from ..core.contract import (
    DEFAULT_MAX_CANDIDATE_RPC_CALLS,
    HARD_MAX_CANDIDATE_RPC_CALLS,
    MAX_TASK_CONTRACT_BYTES,
    TASK_CONTRACT_FILENAME,
    safe_contract_relative_path,
)
from ..core.errors import ProgramSpecError, TaskContractError
from ..evaluation.program_spec import resolve_entrypoint
from ..workspace.paths import join
from ..workspace.protocol import Workspace

_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PACKAGE_COMPONENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
_PINNED_VERSION = (
    r"(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*"
    r"(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?"
    r"(?:\+[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?"
)
_PINNED_REQUIREMENT_RE = re.compile(
    rf"^(?P<name>{_PACKAGE_COMPONENT}"
    rf"(?:\[{_PACKAGE_COMPONENT}(?:,{_PACKAGE_COMPONENT})*\])?)"
    rf"==(?P<version>{_PINNED_VERSION})$"
)
_PACKAGE_NAME_SEPARATOR_RE = re.compile(r"[-_.]+")
_MAX_FINDINGS = 64
_MAX_ITEMS = 64
_MAX_ARTIFACTS = 256
MAX_DEPENDENCY_ENTRIES = 256
_SYNC_MAX_DEPENDENCIES = 64
_TOP_FIELDS = {
    "schema_version",
    "canonical_task",
    "method",
    "candidate",
    "evaluators",
    "resources",
    "data",
    "metrics",
    "upstream",
    "provenance",
    "compute",
    "protocol",
    "feedback",
    "publication",
}
_UPSTREAM_FIELDS = {
    "repository",
    "revision",
    "package_version",
    "license",
    "entrypoints",
    "patches",
}
_COMPUTE_REQUIRED_FIELDS = {
    "accelerator",
    "accelerator_model",
    "gpu_count",
    "evaluator_gpu_count",
    "candidate_gpu_count",
    "cpu_cores",
    "memory_gb",
    "precision",
    "wall_clock_seconds",
}
_COMPUTE_OPTIONAL_FIELDS = {"max_candidate_rpc_calls", "execution_mode"}
_PROTOCOL_REQUIRED_FIELDS = {
    "search_seeds",
    "final_seeds",
    "final_reset_sequence",
    "baselines",
    "promotion_rule",
    "search_proxies",
}
_PROTOCOL_OPTIONAL_FIELDS = {
    "search_reset_sequence",
    "search_smoke_reset_sequence",
    "final_smoke_reset_sequence",
}
_METRIC_FIELDS = {"name", "data_role"}
_DEPENDENCY_FIELDS = {"name", "version"}
_FEEDBACK_FIELDS = {"search_visible", "final_sealed"}
_PUBLICATION_FIELDS = {"search_is_proxy", "requires_final_evaluation", "required_evidence"}


class ContractSyncError(ValueError):
    """Digest or dependency synchronisation cannot be done without semantic edits."""


# --- bounded findings ---------------------------------------------------------------------


class _Findings:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.omitted = 0

    def add(self, message: str) -> None:
        if message in self.items:
            return
        if len(self.items) < _MAX_FINDINGS:
            self.items.append(message)
        else:
            self.omitted += 1

    def result(self) -> tuple[str, ...]:
        if self.omitted:
            self.items.append(f"{self.omitted} additional bounded lint finding(s) omitted")
        return tuple(self.items)


def _mapping(value: object) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _safe_path(value: object) -> str | None:
    try:
        return safe_contract_relative_path(value, "path")
    except TaskContractError:
        return None


def _path_is_within(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts


def _object_fields(
    findings: _Findings, value: object, label: str, expected: set[str]
) -> dict[str, Any] | None:
    mapping = _mapping(value)
    if mapping is None:
        findings.add(f"{label} must be an object")
        return None
    missing = expected - mapping.keys()
    if missing:
        findings.add(f"{label} is missing required field(s): {', '.join(sorted(missing))}")
    unexpected_count = sum(key not in expected for key in mapping)
    if unexpected_count:
        findings.add(
            f"{label} has {unexpected_count} unexpected field(s); allowed fields: "
            + ", ".join(sorted(expected))
        )
    return mapping


def _object_fields_with_optional(
    findings: _Findings,
    value: object,
    label: str,
    required: set[str],
    optional: set[str],
) -> dict[str, Any] | None:
    mapping = _mapping(value)
    if mapping is None:
        findings.add(f"{label} must be an object")
        return None
    missing = required - mapping.keys()
    if missing:
        findings.add(f"{label} is missing required field(s): {', '.join(sorted(missing))}")
    allowed = required | optional
    unexpected_count = sum(key not in allowed for key in mapping)
    if unexpected_count:
        findings.add(
            f"{label} has {unexpected_count} unexpected field(s); allowed fields: "
            + ", ".join(sorted(allowed))
        )
    return mapping


def _items(
    findings: _Findings, value: object, label: str, *, limit: int = _MAX_ITEMS
) -> list[object]:
    if not isinstance(value, list):
        findings.add(f"{label} must be an array")
        return []
    if len(value) > limit:
        findings.add(f"{label} has {len(value) - limit} item(s) omitted from bounded lint")
    return value[:limit]


def normalized_dependency_pin(name: str, version: str) -> str:
    """Return ``name==version`` with upstream's normalisation (separators, case, extras)."""
    base_name, separator, extras_value = name.partition("[")
    normalized_name = _PACKAGE_NAME_SEPARATOR_RE.sub("-", base_name).lower()
    if separator:
        extras = extras_value.removesuffix("]").split(",")
        normalized_extras = sorted(
            {_PACKAGE_NAME_SEPARATOR_RE.sub("-", extra).lower() for extra in extras}
        )
        normalized_name += f"[{','.join(normalized_extras)}]"
    return f"{normalized_name}=={version}"


def _requirement_pins(requirements: bytes | None, findings: _Findings) -> set[str] | None:
    if requirements is None:
        findings.add("requirements.txt must be a safe regular file")
        return None
    if len(requirements) > MAX_TASK_CONTRACT_BYTES:
        findings.add(f"requirements.txt exceeds {MAX_TASK_CONTRACT_BYTES} byte harness limit")
        return None
    try:
        lines = requirements.decode("utf-8").splitlines()
    except UnicodeError:
        findings.add("requirements.txt must be readable UTF-8")
        return None
    pins: set[str] = set()
    requirement_rows = 0
    for line in lines:
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        requirement_rows += 1
        if requirement_rows > MAX_DEPENDENCY_ENTRIES:
            findings.add(f"requirements.txt exceeds {MAX_DEPENDENCY_ENTRIES} pinned entries")
            return None
        match = _PINNED_REQUIREMENT_RE.fullmatch(requirement)
        if match is None:
            findings.add("requirements.txt must contain only pinned package==version entries")
            return None
        pins.add(normalized_dependency_pin(match["name"], match["version"]))
    return pins


def contract_findings(
    contract_bytes: bytes,
    *,
    requirements: bytes | None = None,
    check_requirements: bool = True,
    require_explicit_rpc_budgets: bool = False,
) -> tuple[str, ...]:
    """Return independent actionable findings about a contract, without stopping at the first.

    Args:
        contract_bytes: The bytes of ``task_contract.json``.
        requirements: The bytes of ``requirements.txt``; ``None`` when it is not a safe regular
            file.
        check_requirements: Whether to compare ``requirements.txt`` with the declared
            dependencies.
        require_explicit_rpc_budgets: Require ``max_candidate_rpc_calls`` on both planes.

    Returns:
        At most 64 findings (plus a count of omitted ones); empty when the bytes are not JSON
        (the strict parser reports that).
    """
    findings = _Findings()
    if len(contract_bytes) > MAX_TASK_CONTRACT_BYTES:
        findings.add(f"task contract exceeds {MAX_TASK_CONTRACT_BYTES} byte harness limit")
        return findings.result()
    try:
        value = json.loads(contract_bytes)
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError):
        return ()
    contract = _object_fields(findings, value, "task contract", _TOP_FIELDS)
    if contract is None:
        return findings.result()

    canonical = _object_fields(
        findings, contract.get("canonical_task"), "canonical_task", {"path", "sha256"}
    )
    if canonical is not None and _text(canonical.get("path")) != "task.md":
        findings.add("canonical_task.path must be task.md")

    _object_fields(findings, contract.get("method"), "method", {"class", "description"})

    candidate = _object_fields(
        findings, contract.get("candidate"), "candidate", {"interface", "editable_paths"}
    )
    if candidate is not None:
        if _text(candidate.get("interface")) != "python_rpc_v1:handle":
            findings.add("candidate.interface must be python_rpc_v1:handle")
        for index, item in enumerate(
            _items(findings, candidate.get("editable_paths"), "candidate.editable_paths")
        ):
            path = _safe_path(item)
            if path is None:
                findings.add(
                    f"candidate.editable_paths[{index}] must be a safe POSIX relative path"
                )
            elif not _path_is_within(path, "program_0"):
                findings.add(f"candidate.editable_paths[{index}] must be inside program_0")

    roles = _object_fields(findings, contract.get("data"), "data", {"search_roles", "final_roles"})
    if roles is not None:
        for plane in ("search_roles", "final_roles"):
            for index, item in enumerate(_items(findings, roles.get(plane), f"data.{plane}")):
                role = _text(item)
                if role is None or not _ROLE_RE.fullmatch(role):
                    findings.add(f"data.{plane}[{index}] must be a snake_case role")

    metrics = _object_fields(findings, contract.get("metrics"), "metrics", {"search", "final"})
    if metrics is not None:
        for plane in ("search", "final"):
            label = f"metrics.{plane}"
            for index, metric in enumerate(_items(findings, metrics.get(plane), label)):
                _object_fields(findings, metric, f"{label}[{index}]", _METRIC_FIELDS)

    evaluators = _object_fields(
        findings, contract.get("evaluators"), "evaluators", {"search", "final"}
    )
    if evaluators is not None:
        if _text(evaluators.get("search")) != "evaluate.py":
            findings.add("evaluators.search must be evaluate.py")
        if _text(evaluators.get("final")) != "evaluate_final.py":
            findings.add("evaluators.final must be evaluate_final.py")
    resources = _object_fields(
        findings,
        contract.get("resources"),
        "resources",
        {"search_root", "final_root", "public_artifact_bundle"},
    )
    if resources is not None and (
        _text(resources.get("search_root")) != "resources/search"
        or _text(resources.get("final_root")) != "resources/final"
    ):
        findings.add("resource roots must be resources/search and resources/final")

    artifact_paths: set[str] = set()
    artifact_cross_reference_complete = True
    dependency_pins: set[str] = set()
    dependencies_complete = True
    provenance = _object_fields(
        findings, contract.get("provenance"), "provenance", {"dependencies", "artifacts"}
    )
    if provenance is not None:
        dependencies_value = provenance.get("dependencies")
        dependencies_complete = (
            isinstance(dependencies_value, list)
            and len(dependencies_value) <= MAX_DEPENDENCY_ENTRIES
        )
        for index, dependency in enumerate(
            _items(
                findings,
                dependencies_value,
                "provenance.dependencies",
                limit=MAX_DEPENDENCY_ENTRIES,
            )
        ):
            dependency_data = _object_fields(
                findings, dependency, f"provenance.dependencies[{index}]", _DEPENDENCY_FIELDS
            )
            if dependency_data is None:
                dependencies_complete = False
                continue
            name = _text(dependency_data.get("name"))
            version = _text(dependency_data.get("version"))
            if (
                name is None
                or version is None
                or _PINNED_REQUIREMENT_RE.fullmatch(f"{name}=={version}") is None
            ):
                findings.add(
                    f"provenance.dependencies[{index}] must name a pip-installable "
                    "package and version"
                )
                dependencies_complete = False
                continue
            dependency_pins.add(normalized_dependency_pin(name, version))
        artifact_values = provenance.get("artifacts")
        artifact_cross_reference_complete = (
            isinstance(artifact_values, list) and len(artifact_values) <= _MAX_ARTIFACTS
        )
        for index, artifact in enumerate(
            _items(findings, artifact_values, "provenance.artifacts", limit=_MAX_ARTIFACTS)
        ):
            artifact_data = _object_fields(
                findings,
                artifact,
                f"provenance.artifacts[{index}]",
                {"role", "path", "sha256", "source"},
            )
            if artifact_data is not None:
                path = _safe_path(artifact_data.get("path"))
                if path is not None:
                    artifact_paths.add(path)

    patch_paths: set[str] = set()
    upstream_value = contract.get("upstream")
    patch_cross_reference_complete = (
        isinstance(upstream_value, list) and len(upstream_value) <= _MAX_ITEMS
    )
    for index, item in enumerate(_items(findings, upstream_value, "upstream")):
        label = f"upstream[{index}]"
        item_data = _object_fields(findings, item, label, _UPSTREAM_FIELDS)
        if item_data is None:
            continue
        patch_values = item_data.get("patches")
        if not isinstance(patch_values, list) or len(patch_values) > _MAX_ITEMS:
            patch_cross_reference_complete = False
        for patch_index, value_item in enumerate(
            _items(findings, patch_values, f"{label}.patches")
        ):
            path = _safe_path(value_item)
            if path is None:
                findings.add(f"{label}.patches[{patch_index}] must be a safe POSIX relative path")
            else:
                patch_paths.add(path)
    missing_hash_count = len(patch_paths - artifact_paths)
    if artifact_cross_reference_complete and patch_cross_reference_complete and missing_hash_count:
        findings.add(
            f"{missing_hash_count} upstream patch path(s) lack matching provenance artifact "
            "checksums"
        )

    rpc_budgets: dict[str, int] = {}
    compute = _object_fields(findings, contract.get("compute"), "compute", {"search", "final"})
    if compute is not None:
        for plane in ("search", "final"):
            label = f"compute.{plane}"
            plane_data = _object_fields_with_optional(
                findings,
                compute.get(plane),
                label,
                _COMPUTE_REQUIRED_FIELDS,
                _COMPUTE_OPTIONAL_FIELDS,
            )
            if (
                require_explicit_rpc_budgets
                and plane_data is not None
                and "max_candidate_rpc_calls" not in plane_data
            ):
                findings.add(f"{label} is missing required field(s): max_candidate_rpc_calls")
            accelerator = _text(plane_data.get("accelerator")) if plane_data else None
            if accelerator is not None and accelerator.lower() not in {"cpu", "gpu"}:
                findings.add(f"{label}.accelerator must be 'cpu' or 'gpu'")
            if plane_data is not None:
                execution_mode = _text(plane_data.get("execution_mode", "isolated_v1"))
                if execution_mode != "isolated_v1":
                    findings.add(f"{label}.execution_mode must be 'isolated_v1'")
            rpc_budget = plane_data.get("max_candidate_rpc_calls") if plane_data else None
            if rpc_budget is None:
                rpc_budgets[plane] = DEFAULT_MAX_CANDIDATE_RPC_CALLS
            elif (
                isinstance(rpc_budget, bool)
                or not isinstance(rpc_budget, int)
                or not 1 <= rpc_budget <= HARD_MAX_CANDIDATE_RPC_CALLS
            ):
                findings.add(
                    f"{label}.max_candidate_rpc_calls must be an integer between 1 and "
                    f"{HARD_MAX_CANDIDATE_RPC_CALLS}"
                )
            else:
                rpc_budgets[plane] = rpc_budget

    protocol = _object_fields_with_optional(
        findings,
        contract.get("protocol"),
        "protocol",
        _PROTOCOL_REQUIRED_FIELDS,
        _PROTOCOL_OPTIONAL_FIELDS,
    )
    if protocol is not None:
        _protocol_findings(findings, protocol, rpc_budgets)

    _object_fields(findings, contract.get("feedback"), "feedback", _FEEDBACK_FIELDS)
    _object_fields(findings, contract.get("publication"), "publication", _PUBLICATION_FIELDS)
    if check_requirements and provenance is not None and dependencies_complete:
        requirement_pins = _requirement_pins(requirements, findings)
        if requirement_pins is not None and requirement_pins != dependency_pins:
            requirements_only = sorted(requirement_pins - dependency_pins)
            contract_only = sorted(dependency_pins - requirement_pins)
            findings.add(
                "requirements.txt and provenance.dependencies must contain the same pinned "
                f"packages; requirements-only={requirements_only} contract-only={contract_only}"
            )
    return findings.result()


def _protocol_findings(
    findings: _Findings, protocol: dict[str, Any], rpc_budgets: dict[str, int]
) -> None:
    seed_sets: dict[str, set[int]] = {}
    for field in ("search_seeds", "final_seeds"):
        label = f"protocol.{field}"
        raw_seeds = protocol.get(field)
        seeds = _items(findings, raw_seeds, label)
        if not seeds:
            findings.add(f"{label} must be a non-empty array")
        valid_seeds: list[int] = []
        for index, seed in enumerate(seeds):
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                findings.add(f"{label}[{index}] must be a non-negative integer")
            else:
                valid_seeds.append(seed)
        if (
            isinstance(raw_seeds, list)
            and len(raw_seeds) <= _MAX_ITEMS
            and len(valid_seeds) == len(raw_seeds)
        ):
            if len(valid_seeds) != len(set(valid_seeds)):
                findings.add(f"{label} must not contain duplicates")
            seed_sets[field] = set(valid_seeds)
    if seed_sets.get("search_seeds", set()).intersection(seed_sets.get("final_seeds", set())):
        findings.add("protocol search and final seeds must be disjoint")

    production_sequences: dict[str, list[int]] = {}
    for plane, field, seed_field in (
        ("search", "search_reset_sequence", "search_seeds"),
        ("final", "final_reset_sequence", "final_seeds"),
    ):
        label = f"protocol.{field}"
        raw_sequence = protocol.get(field)
        if field == "search_reset_sequence" and field not in protocol:
            raw_sequence = protocol.get("search_seeds")
        if not isinstance(raw_sequence, list):
            findings.add(f"{label} must be an array")
            continue
        if not raw_sequence:
            findings.add(f"{label} must be a non-empty array")
            continue
        rpc_budget = rpc_budgets.get(plane)
        if rpc_budget is not None and len(raw_sequence) > rpc_budget:
            findings.add(f"{label} exceeds compute.{plane}.max_candidate_rpc_calls")
        valid_sequence: list[int] = []
        for index, seed in enumerate(raw_sequence):
            if type(seed) is not int or seed < 0:
                findings.add(f"{label}[{index}] must be a non-negative integer")
            else:
                valid_sequence.append(seed)
        if len(valid_sequence) != len(raw_sequence):
            continue
        production_sequences[plane] = valid_sequence
        if seed_field in seed_sets and set(valid_sequence) != seed_sets[seed_field]:
            findings.add(f"{label} must cover exactly {seed_field}")

    for plane, field in (
        ("search", "search_smoke_reset_sequence"),
        ("final", "final_smoke_reset_sequence"),
    ):
        production_sequence = production_sequences.get(plane)
        raw_sequence = protocol.get(field)
        if field not in protocol:
            if not production_sequence:
                continue
            raw_sequence = [production_sequence[0]]
        label = f"protocol.{field}"
        if not isinstance(raw_sequence, list):
            findings.add(f"{label} must be an array")
            continue
        if not raw_sequence:
            findings.add(f"{label} must be a non-empty array")
            continue
        rpc_budget = rpc_budgets.get(plane)
        if rpc_budget is not None and len(raw_sequence) > rpc_budget:
            findings.add(f"{label} exceeds compute.{plane}.max_candidate_rpc_calls")
        valid_smoke_sequence: list[int] = []
        for index, seed in enumerate(raw_sequence):
            if type(seed) is not int or seed < 0:
                findings.add(f"{label}[{index}] must be a non-negative integer")
            else:
                valid_smoke_sequence.append(seed)
        if (
            production_sequence
            and len(valid_smoke_sequence) == len(raw_sequence)
            and any(seed != production_sequence[0] for seed in valid_smoke_sequence)
        ):
            findings.add(f"{label} must contain only the first production reset seed")

    raw_baselines = protocol.get("baselines")
    baselines = _items(findings, raw_baselines, "protocol.baselines")
    if not baselines:
        findings.add("protocol.baselines must be a non-empty array")
    valid_baselines: list[str] = []
    for index, baseline in enumerate(baselines):
        normalized_baseline = _text(baseline)
        if normalized_baseline is None:
            findings.add(f"protocol.baselines[{index}] must be a non-empty string")
        else:
            valid_baselines.append(normalized_baseline)
    if (
        isinstance(raw_baselines, list)
        and len(raw_baselines) <= _MAX_ITEMS
        and len(valid_baselines) == len(raw_baselines)
        and len(valid_baselines) != len(set(valid_baselines))
    ):
        findings.add("protocol.baselines must not contain duplicates")
    if _text(protocol.get("promotion_rule")) is None:
        findings.add("protocol.promotion_rule must be a non-empty string")
    raw_proxies = protocol.get("search_proxies")
    proxies = _items(findings, raw_proxies, "protocol.search_proxies")
    proxy_names: list[str] = []
    all_proxy_names_valid = True
    for index, proxy in enumerate(proxies):
        label = f"protocol.search_proxies[{index}]"
        proxy_data = _object_fields(findings, proxy, label, {"name", "preserves", "omits"})
        if proxy_data is None:
            all_proxy_names_valid = False
            continue
        for field in ("name", "preserves", "omits"):
            if _text(proxy_data.get(field)) is None:
                findings.add(f"{label}.{field} must be a non-empty string")
        name = _text(proxy_data.get("name"))
        if name is None:
            all_proxy_names_valid = False
        else:
            proxy_names.append(name)
    if (
        isinstance(raw_proxies, list)
        and len(raw_proxies) <= _MAX_ITEMS
        and all_proxy_names_valid
        and len(proxy_names) != len(set(proxy_names))
    ):
        findings.add("protocol.search_proxies names must be unique")


# --- workspace access ----------------------------------------------------------------------


async def read_requirements(workspace: Workspace, task_dir: str) -> bytes | None:
    """Return the bytes of ``requirements.txt`` when it is a regular file, else ``None``."""
    entry = await workspace.stat(join(task_dir, "requirements.txt"))
    if entry is None or entry.kind != "file":
        return None
    return await workspace.read_bytes(join(task_dir, "requirements.txt"))


async def preflight_findings(
    workspace: Workspace, task_dir: str, *, require_explicit_rpc_budgets: bool = True
) -> tuple[str, ...]:
    """Run :func:`contract_findings` on a staged task directory.

    An unreadable contract yields no findings (the strict parser reports it).
    """
    entry = await workspace.stat(join(task_dir, TASK_CONTRACT_FILENAME))
    if entry is None or entry.kind != "file":
        return ()
    contract = await workspace.read_bytes(join(task_dir, TASK_CONTRACT_FILENAME))
    return contract_findings(
        contract,
        requirements=await read_requirements(workspace, task_dir),
        check_requirements=True,
        require_explicit_rpc_budgets=require_explicit_rpc_budgets,
    )


# --- mechanical synchronisation (upstream contract_sync.py) ------------------------------------


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractSyncError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_contract_object(raw: bytes) -> dict[str, Any]:
    """Parse the bytes of a contract as a plain object, refusing duplicate keys and bad fields."""
    if len(raw) > MAX_TASK_CONTRACT_BYTES:
        raise ContractSyncError("task contract is too large")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_object_without_duplicates)
    except ContractSyncError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractSyncError("task contract is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ContractSyncError("task contract must be an object")
    missing = _TOP_FIELDS - payload.keys()
    unexpected = payload.keys() - _TOP_FIELDS
    if missing or unexpected:
        raise ContractSyncError(
            f"task contract fields mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    if payload.get("schema_version") != 1:
        raise ContractSyncError("task contract schema_version must be 1")
    return payload


async def _load_contract(workspace: Workspace, task_dir: str) -> dict[str, Any]:
    entry = await workspace.stat(join(task_dir, TASK_CONTRACT_FILENAME))
    if entry is None or entry.kind != "file":
        raise ContractSyncError("task contract must be a regular file")
    return load_contract_object(await workspace.read_bytes(join(task_dir, TASK_CONTRACT_FILENAME)))


def _object(value: object, label: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ContractSyncError(f"{label} must contain exactly {sorted(fields)}")
    return value


async def _declared_file(workspace: Workspace, task_dir: str, value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractSyncError(f"{label} must be a safe relative path")
    try:
        relative = PurePosixPath(safe_contract_relative_path(value, label))
    except TaskContractError as exc:
        raise ContractSyncError(str(exc)) from exc
    current = task_dir
    entry = None
    for part in relative.parts:
        current = join(current, part)
        entry = await workspace.stat(current)
        if entry is not None and entry.kind == "symlink":
            raise ContractSyncError(f"{label} must not contain symlinks")
    if entry is None or entry.kind != "file":
        raise ContractSyncError(f"{label} must identify an existing regular file")
    return current


def _normalized(name: str, version: str) -> tuple[str, str]:
    pin = normalized_dependency_pin(name, version)
    normalized_name, _, _ = pin.rpartition("==")
    return normalized_name, version


def _declared_dependencies(payload: Mapping[str, Any]) -> list[dict[str, str]]:
    provenance = _object(payload["provenance"], "provenance", {"dependencies", "artifacts"})
    dependencies = provenance["dependencies"]
    if not isinstance(dependencies, list) or len(dependencies) > _SYNC_MAX_DEPENDENCIES:
        raise ContractSyncError("provenance.dependencies must be a bounded array")
    parsed: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, value in enumerate(dependencies):
        dependency = _object(value, f"provenance.dependencies[{index}]", {"name", "version"})
        name = dependency["name"]
        version = dependency["version"]
        if (
            not isinstance(name, str)
            or not isinstance(version, str)
            or _PINNED_REQUIREMENT_RE.fullmatch(f"{name}=={version}") is None
        ):
            raise ContractSyncError("provenance.dependencies must contain pinned packages")
        key = _normalized(name, version)
        if key in seen:
            raise ContractSyncError("provenance.dependencies must not contain duplicates")
        seen.add(key)
        parsed.append({"name": name, "version": version})
    return parsed


def _requirement_keys(requirements: bytes | None) -> set[tuple[str, str]]:
    if requirements is None:
        raise ContractSyncError("requirements.txt must be a regular scaffold file")
    if len(requirements) > MAX_TASK_CONTRACT_BYTES:
        raise ContractSyncError("requirements.txt is too large")
    try:
        lines = requirements.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ContractSyncError("requirements.txt must be readable UTF-8") from exc
    keys: set[tuple[str, str]] = set()
    rows = 0
    for line in lines:
        requirement = line.strip()
        if not requirement or requirement.startswith("#"):
            continue
        rows += 1
        match = _PINNED_REQUIREMENT_RE.fullmatch(requirement)
        if rows > _SYNC_MAX_DEPENDENCIES or match is None:
            raise ContractSyncError("requirements.txt must contain bounded pinned packages")
        key = _normalized(match["name"], match["version"])
        if key in keys:
            raise ContractSyncError("requirements.txt must not contain duplicate packages")
        keys.add(key)
    return keys


async def _refresh_declared_digests(
    workspace: Workspace, payload: dict[str, Any], task_dir: str
) -> bool:
    canonical = _object(payload["canonical_task"], "canonical_task", {"path", "sha256"})
    if canonical["path"] != "task.md":
        raise ContractSyncError("canonical_task.path must remain task.md")
    provenance = _object(payload["provenance"], "provenance", {"dependencies", "artifacts"})
    artifacts = provenance["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ContractSyncError("provenance.artifacts must be a non-empty array")
    changed = False
    task_digest = hashlib.sha256(await workspace.read_bytes(join(task_dir, "task.md"))).hexdigest()
    if canonical["sha256"] != task_digest:
        canonical["sha256"] = task_digest
        changed = True
    seen_paths: set[str] = set()
    for index, value in enumerate(artifacts):
        artifact = _object(
            value, f"provenance.artifacts[{index}]", {"role", "path", "sha256", "source"}
        )
        path_value = artifact["path"]
        if not isinstance(path_value, str) or path_value in seen_paths:
            raise ContractSyncError("provenance artifact paths must be unique strings")
        seen_paths.add(path_value)
        declared = await _declared_file(
            workspace, task_dir, path_value, f"provenance.artifacts[{index}].path"
        )
        digest = hashlib.sha256(await workspace.read_bytes(declared)).hexdigest()
        if artifact["sha256"] != digest:
            artifact["sha256"] = digest
            changed = True
    return changed


def contract_text(payload: Mapping[str, Any]) -> str:
    """Serialise a contract the way upstream's synchronisation writes it."""
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


async def synchronize_declared_digests(workspace: Workspace, task_dir: str) -> bool:
    """Refresh only the canonical-task and artifact SHA-256 fields of already declared paths.

    Returns:
        Whether the contract file was rewritten.

    Raises:
        ContractSyncError: When a refresh would need a semantic edit.
    """
    payload = await _load_contract(workspace, task_dir)
    changed = await _refresh_declared_digests(workspace, payload, task_dir)
    if changed:
        await workspace.write_bytes(
            join(task_dir, TASK_CONTRACT_FILENAME), contract_text(payload).encode("utf-8")
        )
    return changed


async def declared_dependencies_match_requirements(workspace: Workspace, task_dir: str) -> bool:
    """Return whether ``requirements.txt`` pins exactly the declared dependencies."""
    payload = await _load_contract(workspace, task_dir)
    declared = {
        _normalized(dependency["name"], dependency["version"])
        for dependency in _declared_dependencies(payload)
    }
    requirements = await read_requirements(workspace, task_dir)
    required = _requirement_keys(requirements) if requirements is not None else set()
    return declared == required


async def retained_declared_dependencies(
    workspace: Workspace, task_dir: str, *, allowed_dependencies: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Return the allowed dependencies that the repaired ``requirements.txt`` still pins.

    Raises:
        ContractSyncError: When the repair pins a package the contract did not declare.
    """
    allowed = _declared_dependencies(
        {"provenance": {"dependencies": allowed_dependencies, "artifacts": []}}
    )
    allowed_keys = {_normalized(item["name"], item["version"]) for item in allowed}
    retained_keys = _requirement_keys(await read_requirements(workspace, task_dir))
    if not retained_keys.issubset(allowed_keys):
        raise ContractSyncError("repair may only remove pinned dependencies")
    return [item for item in allowed if _normalized(item["name"], item["version"]) in retained_keys]


async def synchronize_declared_dependency_removals(
    workspace: Workspace, task_dir: str, *, allowed_dependencies: list[dict[str, str]]
) -> bool:
    """Remove from the contract only the declared pins that a repair removed from the lock."""
    payload = await _load_contract(workspace, task_dir)
    current = _declared_dependencies(payload)
    retained = await retained_declared_dependencies(
        workspace, task_dir, allowed_dependencies=allowed_dependencies
    )
    if current not in (allowed_dependencies, retained):
        raise ContractSyncError("repair changed declared dependencies directly")
    changed = retained != current
    payload["provenance"]["dependencies"] = retained
    changed = await _refresh_declared_digests(workspace, payload, task_dir) or changed
    if changed:
        await workspace.write_bytes(
            join(task_dir, TASK_CONTRACT_FILENAME), contract_text(payload).encode("utf-8")
        )
    return changed


# --- isolated RPC transport and harness capability lints ---------------------------------------

_MAX_SOURCE_BYTES = 8 * 1024 * 1024
_MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_FILES = 1024
_MAX_TREE_ENTRIES = 4096
_MAX_TREE_DEPTH = 64
_RESERVED_NAMES = frozenset({"EXCHANGE_DIR", "AGENTIC_CANDIDATE_EXCHANGE"})
_RESERVED_HARNESS_PREFIX = "AGENTIC_HARNESS_"
_UNAVAILABLE_PROXY_ACCESSORS = frozenset({"candidate_peak_accelerator_memory_gib"})


def _display_path(relative: str) -> str:
    return json.dumps(relative, ensure_ascii=True)[1:-1]


def _docstring_nodes(tree: ast.AST) -> set[int]:
    nodes: set[int] = set()
    for owner in ast.walk(tree):
        body = getattr(owner, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            nodes.add(id(first.value))
    return nodes


def _reserved_occurrences(source: bytes, relative: str) -> tuple[tuple[int, str], ...] | None:
    try:
        tree = ast.parse(source, filename=relative)
    except (SyntaxError, UnicodeError, ValueError, RecursionError):
        return None
    docstrings = _docstring_nodes(tree)
    occurrences: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        name: str | None = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.arg):
            name = node.arg
        elif isinstance(node, ast.alias):
            name = node.name
        elif isinstance(node, ast.keyword):
            name = node.arg
        elif isinstance(node, ast.Constant) and id(node) not in docstrings:
            if isinstance(node.value, str):
                name = node.value
            elif isinstance(node.value, bytes):
                try:
                    name = node.value.decode("ascii")
                except UnicodeDecodeError:
                    name = None
        if name in _RESERVED_NAMES:
            occurrences.add((getattr(node, "lineno", 1), name or ""))
    return tuple(sorted(occurrences))


def _source_finding(relative: str, line: int, name: str) -> str:
    return (
        f"{_display_path(relative)}:{line}: isolated candidate RPC cannot use `{name}` because "
        "evaluator and candidate share no filesystem; send finite inline JSON, using bounded "
        "chunks below the 64 MiB message limit when needed"
    )


class _TraversalLimit(ValueError):
    pass


async def isolated_rpc_transport_findings(workspace: Workspace, task_dir: str) -> tuple[str, ...]:
    """Return bounded findings for the retired shared-filesystem RPC names.

    Scans the two evaluators and the Python files of ``program_0`` (and the declared
    entrypoint); the names ``EXCHANGE_DIR`` and ``AGENTIC_CANDIDATE_EXCHANGE`` must not occur
    outside docstrings, because the evaluator and the isolated candidate share no filesystem.
    """
    findings: list[str] = []
    state = {"files": 0, "bytes": 0}

    def add(message: str) -> None:
        if len(findings) < _MAX_FINDINGS:
            findings.append(message)

    try:
        entrypoint: str | None = (
            f"program_0/{await resolve_entrypoint(workspace, join(task_dir, 'program_0'))}"
        )
    except ProgramSpecError:
        entrypoint = None

    async def inspect(relative: str, size: int, kind: str) -> None:
        state["files"] += 1
        if state["files"] > _MAX_SOURCE_FILES:
            raise _TraversalLimit(
                f"isolated RPC source contains more than {_MAX_SOURCE_FILES} Python files"
            )
        if kind != "file":
            add(
                f"{_display_path(relative)}: isolated RPC source must be a regular non-symlink file"
            )
            return
        aggregate_remaining = _MAX_SOURCE_BYTES - state["bytes"]
        if size > min(_MAX_SOURCE_FILE_BYTES, aggregate_remaining):
            if aggregate_remaining >= _MAX_SOURCE_FILE_BYTES:
                raise _TraversalLimit(
                    f"{_display_path(relative)}: isolated RPC source exceeds the "
                    f"{_MAX_SOURCE_FILE_BYTES} byte per-file validation limit"
                )
            raise _TraversalLimit(
                f"isolated RPC source exceeds the {_MAX_SOURCE_BYTES} byte validation limit"
            )
        try:
            source = await workspace.read_bytes(join(task_dir, relative))
        except OSError:
            add(f"{_display_path(relative)}: isolated RPC source is unreadable")
            return
        state["bytes"] += len(source)
        occurrences = _reserved_occurrences(source, relative)
        if occurrences is None:
            add(f"{_display_path(relative)}: isolated RPC source could not be parsed safely")
            return
        for line, reserved_name in occurrences:
            add(_source_finding(relative, line, reserved_name))
            if len(findings) == _MAX_FINDINGS:
                return

    try:
        for evaluator in ("evaluate.py", "evaluate_final.py"):
            entry = await workspace.stat(join(task_dir, evaluator))
            if entry is not None:
                await inspect(evaluator, entry.size, entry.kind)
        program = await workspace.stat(join(task_dir, "program_0"))
        if program is None:
            return tuple(findings)
        if program.kind != "dir":
            add("program_0: isolated RPC source root must be a regular non-symlink directory")
            return tuple(findings)
        tree = await workspace.list_tree(join(task_dir, "program_0"))
        if len(tree) > _MAX_TREE_ENTRIES:
            raise _TraversalLimit(
                f"isolated RPC source tree contains more than {_MAX_TREE_ENTRIES} entries"
            )
        for entry in tree:
            if len(findings) == _MAX_FINDINGS:
                break
            relative = f"program_0/{entry.path}"
            if len(entry.path.split("/")) + 1 > _MAX_TREE_DEPTH:
                raise _TraversalLimit(
                    f"isolated RPC source tree exceeds the {_MAX_TREE_DEPTH} directory depth limit"
                )
            if entry.kind == "dir":
                continue
            if relative == entrypoint or entry.path.endswith(".py"):
                await inspect(relative, entry.size, entry.kind)
    except _TraversalLimit as exc:
        return (str(exc),)
    except OSError:
        return ("isolated RPC source tree is unreadable",)
    return tuple(findings)


class HarnessCapabilityFinding:
    """A reserved-environment or unsupported-accessor finding of one evaluator."""

    def __init__(self, component: str, diagnostic: str) -> None:
        self.component = component
        self.diagnostic = diagnostic


def _reserved_environment_occurrences(source: bytes, relative: str) -> tuple[tuple[int, str], ...]:
    try:
        tree = ast.parse(source, filename=relative)
    except (SyntaxError, UnicodeError, ValueError, RecursionError):
        return ()
    docstrings = _docstring_nodes(tree)
    occurrences = {
        (getattr(node, "lineno", 1), node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and id(node) not in docstrings
        and isinstance(node.value, str)
        and node.value.startswith(_RESERVED_HARNESS_PREFIX)
    }
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            name = node.args[1].value
        if name in _UNAVAILABLE_PROXY_ACCESSORS:
            occurrences.add((getattr(node, "lineno", 1), name or ""))
    return tuple(sorted(occurrences))


async def unavailable_harness_environment_findings(
    workspace: Workspace, task_dir: str, evaluators: Sequence[tuple[str, str]]
) -> tuple[HarnessCapabilityFinding, ...]:
    """Return findings for reserved harness environment inputs and unsupported proxy accessors.

    Args:
        workspace: The workspace holding the scaffold.
        task_dir: The staged task directory.
        evaluators: ``(component, relative path)`` pairs, for example
            ``("search_evaluator", "evaluate.py")``.
    """
    findings: list[HarnessCapabilityFinding] = []
    for component, relative in evaluators:
        entry = await workspace.stat(join(task_dir, relative))
        if entry is None or entry.kind != "file" or entry.size > 2 * 1024 * 1024:
            continue
        source = await workspace.read_bytes(join(task_dir, relative))
        for line, name in _reserved_environment_occurrences(source, relative):
            findings.append(
                HarnessCapabilityFinding(
                    component,
                    f"{relative}:{line}: `{name}` is reserved for harness-owned data but "
                    "is not supplied to task evaluators; correctness and metrics must use "
                    "evaluator-owned inputs",
                )
            )
    return tuple(findings)


__all__ = [
    "MAX_DEPENDENCY_ENTRIES",
    "ContractSyncError",
    "HarnessCapabilityFinding",
    "contract_findings",
    "contract_text",
    "declared_dependencies_match_requirements",
    "isolated_rpc_transport_findings",
    "load_contract_object",
    "normalized_dependency_pin",
    "preflight_findings",
    "read_requirements",
    "retained_declared_dependencies",
    "synchronize_declared_dependency_removals",
    "synchronize_declared_digests",
    "unavailable_harness_environment_findings",
]
