"""The task contract (``task_contract.json`` schema 1) and its candidate-visible projection.

This is a port of upstream ``src/investigation/task_contract.py`` (parsing, the canonical task
digest check, the scaffold fingerprint) and ``task_contract_types.py`` (the value types), with
the records as pydantic models so that they can be stored in graph state. The parsing rules and
their error messages are upstream's. File access goes through a
:class:`~langgraph_agentic_scientist.workspace.Workspace`, so a contract can be loaded from a
container or a remote sandbox as well as from the host. The lint rules of
``task_contract_lint.py`` belong to the drafting package.

The projection :func:`public_search_payload` is the data part of upstream
``public_task_contract.render_public_search_contract``: the candidate-visible subset of a
contract, without any final-evaluation field. The text around it is a prompt of the
investigation package.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field

from .errors import TaskContractError
from .state import StateModel

if TYPE_CHECKING:
    from ..workspace.protocol import Workspace

TASK_CONTRACT_FILENAME = "task_contract.json"
TASK_CONTRACT_SCHEMA_VERSION = 1
MAX_TASK_CONTRACT_BYTES = 2 * 1024 * 1024
MIN_COMPUTE_WALL_CLOCK_SECONDS = 120
DEFAULT_MAX_CANDIDATE_RPC_CALLS = 100_000
HARD_MAX_CANDIDATE_RPC_CALLS = 10_000_000
MAX_PUBLIC_SEARCH_CONTRACT_BYTES = 16 * 1024
MAX_PUBLIC_SEARCH_TEXT_BYTES = 2 * 1024
RESOURCE_MANIFEST_FILENAME = ".agentic_resource_manifest.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_FINAL_ROLE_NAMES = frozenset({"final", "test", "sealed_test", "private_test"})
_FLOATING_REVISIONS = frozenset({"head", "latest", "main", "master", "trunk"})
_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@-]*$")
_PUBLIC_BUNDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# --- value types (upstream task_contract_types.py) -----------------------------------------


class CanonicalTask(StateModel):
    """Path and digest of the canonical ``task.md``."""

    path: str
    sha256: str


class ContractMethod(StateModel):
    """The research method class and its description (``method.class`` upstream)."""

    class_name: str = Field(alias="class")
    description: str


class Candidate(StateModel):
    """The candidate interface and the paths a candidate may edit."""

    interface: str
    editable_paths: tuple[str, ...]


class Evaluators(StateModel):
    """Paths of the search and sealed final evaluators."""

    search: str
    final: str


class ResourceRoots(StateModel):
    """The search and final resource roots and the optional public artifact bundle."""

    search_root: str
    final_root: str
    public_artifact_bundle: str | None = None


class DataRoles(StateModel):
    """Data roles available to the search and to the final evaluation."""

    search_roles: tuple[str, ...]
    final_roles: tuple[str, ...]


class Metric(StateModel):
    """One declared metric with its data role."""

    name: str
    data_role: str


class Metrics(StateModel):
    """Declared search and final metrics."""

    search: tuple[Metric, ...]
    final: tuple[Metric, ...]


class Upstream(StateModel):
    """A pinned upstream dependency of the task."""

    repository: str
    revision: str
    package_version: str | None
    license: str
    entrypoints: tuple[str, ...]
    patches: tuple[str, ...]


class Dependency(StateModel):
    """A declared dependency with its version."""

    name: str
    version: str


class Artifact(StateModel):
    """A declared artifact with its digest and source."""

    role: str
    path: str
    sha256: str
    source: str


class Provenance(StateModel):
    """Declared dependencies and artifacts."""

    dependencies: tuple[Dependency, ...]
    artifacts: tuple[Artifact, ...]


class ComputePlane(StateModel):
    """The compute envelope of one evaluation plane."""

    accelerator: str
    accelerator_model: str
    gpu_count: int
    evaluator_gpu_count: int
    candidate_gpu_count: int
    cpu_cores: int
    memory_gb: int
    precision: str
    wall_clock_seconds: int
    max_candidate_rpc_calls: int
    execution_mode: str = "isolated_v1"


class Compute(StateModel):
    """Compute envelopes of the search and final planes."""

    search: ComputePlane
    final: ComputePlane


class SearchProxy(StateModel):
    """A declared search proxy: what it preserves and what it omits."""

    name: str
    preserves: str
    omits: str


class Protocol(StateModel):
    """Seeds, reset sequences, baselines and the promotion rule."""

    search_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    search_reset_sequence: tuple[int, ...]
    search_smoke_reset_sequence: tuple[int, ...]
    final_reset_sequence: tuple[int, ...]
    final_smoke_reset_sequence: tuple[int, ...]
    baselines: tuple[str, ...]
    promotion_rule: str
    search_proxies: tuple[SearchProxy, ...]


class Feedback(StateModel):
    """What the search may see and what stays sealed."""

    search_visible: tuple[str, ...]
    final_sealed: tuple[str, ...]


class Publication(StateModel):
    """Publication requirements of the task."""

    search_is_proxy: bool
    requires_final_evaluation: bool
    required_evidence: tuple[str, ...]


class TaskContract(StateModel):
    """A strict schema-version 1 task contract (upstream ``TaskContract``)."""

    contract_schema_version: int = Field(alias="schema_version")
    canonical_task: CanonicalTask
    method: ContractMethod
    candidate: Candidate
    evaluators: Evaluators
    resources: ResourceRoots
    data: DataRoles
    metrics: Metrics
    upstream: tuple[Upstream, ...]
    provenance: Provenance
    compute: Compute
    protocol: Protocol
    feedback: Feedback
    publication: Publication


# --- parsing (upstream task_contract.py) ----------------------------------------------------


def _mapping(value: object, label: str, fields: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TaskContractError(f"{label} must be an object")
    missing = fields - value.keys()
    unexpected = value.keys() - fields
    if missing:
        raise TaskContractError(f"{label} missing field(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise TaskContractError(f"{label} has unexpected field(s): {', '.join(sorted(unexpected))}")
    return value


def _mapping_with_optional(
    value: object, label: str, required_fields: set[str], optional_fields: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TaskContractError(f"{label} must be an object")
    missing = required_fields - value.keys()
    unexpected = value.keys() - required_fields - optional_fields
    if missing:
        raise TaskContractError(f"{label} missing field(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise TaskContractError(f"{label} has unexpected field(s): {', '.join(sorted(unexpected))}")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskContractError(f"{label} must be a non-empty string")
    return value.strip()


def safe_contract_relative_path(value: object, label: str) -> str:
    """Validate contract path syntax, including a leaf Python package initializer."""
    raw = _string(value, label)
    components = raw.split("/")
    ordinary_components = components[:-1] if components[-1] == "__init__.py" else components
    if "\\" in raw or any(
        not _SAFE_PATH_COMPONENT_RE.fullmatch(part) for part in ordinary_components
    ):
        raise TaskContractError(f"{label} must be a safe POSIX relative path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or raw in {"", "."}
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise TaskContractError(f"{label} must be a safe relative path")
    return path.as_posix()


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise TaskContractError(f"{label} must be a non-empty array")
    items = tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(items) != len(set(items)):
        raise TaskContractError(f"{label} must not contain duplicates")
    return items


def _optional_string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TaskContractError(f"{label} must be an array")
    items = tuple(_string(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(items) != len(set(items)):
        raise TaskContractError(f"{label} must not contain duplicates")
    return items


def _seed_tuple(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TaskContractError(f"{label} must be a non-empty array")
    seeds: list[int] = []
    for index, seed in enumerate(value):
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise TaskContractError(f"{label}[{index}] must be a non-negative integer")
        seeds.append(seed)
    if len(seeds) != len(set(seeds)):
        raise TaskContractError(f"{label} must not contain duplicates")
    return tuple(seeds)


def _reset_seed_tuple(
    value: object, label: str, max_candidate_rpc_calls: int, plane: str
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise TaskContractError(f"{label} must be a non-empty array")
    for index, seed in enumerate(value):
        if type(seed) is not int or seed < 0:
            raise TaskContractError(f"{label}[{index}] must be a non-negative integer")
    if len(value) > max_candidate_rpc_calls:
        raise TaskContractError(f"{label} exceeds compute.{plane}.max_candidate_rpc_calls")
    return tuple(value)


def _reset_sequence(
    value: object,
    *,
    label: str,
    statistical_seeds: tuple[int, ...],
    max_candidate_rpc_calls: int,
    plane: str,
) -> tuple[int, ...]:
    sequence = _reset_seed_tuple(value, label, max_candidate_rpc_calls, plane)
    if set(sequence) != set(statistical_seeds):
        seed_field = "search_seeds" if plane == "search" else "final_seeds"
        raise TaskContractError(f"{label} must cover exactly {seed_field}")
    return sequence


def _smoke_reset_sequence(
    value: object,
    *,
    label: str,
    production_sequence: tuple[int, ...],
    max_candidate_rpc_calls: int,
    plane: str,
) -> tuple[int, ...]:
    sequence = _reset_seed_tuple(value, label, max_candidate_rpc_calls, plane)
    if any(seed != production_sequence[0] for seed in sequence):
        raise TaskContractError(f"{label} must contain only the first production reset seed")
    return sequence


def _non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _parse_compute_plane(value: object, label: str) -> ComputePlane:
    data = _mapping_with_optional(
        value,
        label,
        {
            "accelerator",
            "accelerator_model",
            "gpu_count",
            "evaluator_gpu_count",
            "candidate_gpu_count",
            "cpu_cores",
            "memory_gb",
            "precision",
            "wall_clock_seconds",
        },
        {"max_candidate_rpc_calls", "execution_mode"},
    )
    execution_mode = _string(data.get("execution_mode", "isolated_v1"), f"{label}.execution_mode")
    if execution_mode != "isolated_v1":
        raise TaskContractError(f"{label}.execution_mode must be 'isolated_v1'")
    accelerator = _string(data["accelerator"], f"{label}.accelerator").lower()
    if accelerator not in {"cpu", "gpu"}:
        raise TaskContractError(f"{label}.accelerator must be 'cpu' or 'gpu'")
    gpu_count = data["gpu_count"]
    evaluator_gpu_count = data["evaluator_gpu_count"]
    candidate_gpu_count = data["candidate_gpu_count"]
    cpu_cores = data["cpu_cores"]
    memory_gb = data["memory_gb"]
    wall_clock_seconds = data["wall_clock_seconds"]
    max_candidate_rpc_calls = data.get("max_candidate_rpc_calls", DEFAULT_MAX_CANDIDATE_RPC_CALLS)
    if not _non_negative_int(gpu_count):
        raise TaskContractError(f"{label}.gpu_count must be a non-negative integer")
    for field_name, count in (
        ("evaluator_gpu_count", evaluator_gpu_count),
        ("candidate_gpu_count", candidate_gpu_count),
    ):
        if not _non_negative_int(count):
            raise TaskContractError(f"{label}.{field_name} must be a non-negative integer")
    if isinstance(cpu_cores, bool) or not isinstance(cpu_cores, int) or cpu_cores < 1:
        raise TaskContractError(f"{label}.cpu_cores must be a positive integer")
    if isinstance(memory_gb, bool) or not isinstance(memory_gb, int) or memory_gb < 1:
        raise TaskContractError(f"{label}.memory_gb must be a positive integer")
    if (
        isinstance(wall_clock_seconds, bool)
        or not isinstance(wall_clock_seconds, int)
        or wall_clock_seconds < MIN_COMPUTE_WALL_CLOCK_SECONDS
    ):
        raise TaskContractError(
            f"{label}.wall_clock_seconds must be an integer of at least "
            f"{MIN_COMPUTE_WALL_CLOCK_SECONDS} seconds"
        )
    if (
        isinstance(max_candidate_rpc_calls, bool)
        or not isinstance(max_candidate_rpc_calls, int)
        or not 1 <= max_candidate_rpc_calls <= HARD_MAX_CANDIDATE_RPC_CALLS
    ):
        raise TaskContractError(
            f"{label}.max_candidate_rpc_calls must be an integer between 1 and "
            f"{HARD_MAX_CANDIDATE_RPC_CALLS}"
        )
    if (accelerator == "cpu" and gpu_count != 0) or (accelerator == "gpu" and gpu_count < 1):
        raise TaskContractError(f"{label}.gpu_count is inconsistent with accelerator")
    if evaluator_gpu_count + candidate_gpu_count != gpu_count:
        raise TaskContractError(
            f"{label} evaluator_gpu_count + candidate_gpu_count must equal gpu_count"
        )
    accelerator_model = _string(data["accelerator_model"], f"{label}.accelerator_model")
    if accelerator == "cpu" and accelerator_model.lower() != "none":
        raise TaskContractError(f"{label}.accelerator_model must be 'none' for CPU")
    if accelerator == "gpu" and accelerator_model.lower() == "none":
        raise TaskContractError(f"{label}.accelerator_model must identify the GPU model")
    return ComputePlane(
        accelerator=accelerator,
        accelerator_model=accelerator_model,
        gpu_count=gpu_count,
        evaluator_gpu_count=evaluator_gpu_count,
        candidate_gpu_count=candidate_gpu_count,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        precision=_string(data["precision"], f"{label}.precision"),
        wall_clock_seconds=wall_clock_seconds,
        max_candidate_rpc_calls=max_candidate_rpc_calls,
        execution_mode=execution_mode,
    )


def _parse_search_proxies(value: object) -> tuple[SearchProxy, ...]:
    if not isinstance(value, list):
        raise TaskContractError("protocol.search_proxies must be an array")
    result: list[SearchProxy] = []
    for index, item in enumerate(value):
        label = f"protocol.search_proxies[{index}]"
        data = _mapping(item, label, {"name", "preserves", "omits"})
        result.append(
            SearchProxy(
                name=_string(data["name"], f"{label}.name"),
                preserves=_string(data["preserves"], f"{label}.preserves"),
                omits=_string(data["omits"], f"{label}.omits"),
            )
        )
    if len({proxy.name for proxy in result}) != len(result):
        raise TaskContractError("protocol.search_proxies names must be unique")
    return tuple(result)


def _parse_metrics(value: object, label: str) -> tuple[Metric, ...]:
    if not isinstance(value, list) or not value:
        raise TaskContractError(f"{label} must be a non-empty array")
    result: list[Metric] = []
    for index, item in enumerate(value):
        metric_label = f"{label}[{index}]"
        data = _mapping(item, metric_label, {"name", "data_role"})
        result.append(
            Metric(
                name=_string(data["name"], f"{metric_label}.name"),
                data_role=_string(data["data_role"], f"{metric_label}.data_role"),
            )
        )
    if len({metric.name for metric in result}) != len(result):
        raise TaskContractError(f"{label} metric names must be unique")
    return tuple(result)


def _paths_overlap(first: str, second: str) -> bool:
    first_parts = PurePosixPath(first).parts
    second_parts = PurePosixPath(second).parts
    shared = min(len(first_parts), len(second_parts))
    return first_parts[:shared] == second_parts[:shared]


def _path_is_within(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return len(path_parts) >= len(root_parts) and path_parts[: len(root_parts)] == root_parts


def parse_task_contract(value: object) -> TaskContract:
    """Parse a JSON-compatible value into a strict schema-version 1 contract.

    Raises:
        TaskContractError: With upstream's message, when any rule of the schema fails.
    """
    data = _mapping(
        value,
        "task contract",
        {
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
        },
    )
    version = data["schema_version"]
    if isinstance(version, bool) or version != TASK_CONTRACT_SCHEMA_VERSION:
        raise TaskContractError(
            f"unsupported schema_version={version!r}; expected {TASK_CONTRACT_SCHEMA_VERSION}"
        )

    canonical_data = _mapping(data["canonical_task"], "canonical_task", {"path", "sha256"})
    canonical = CanonicalTask(
        path=safe_contract_relative_path(canonical_data["path"], "canonical_task.path"),
        sha256=_string(canonical_data["sha256"], "canonical_task.sha256").lower(),
    )
    if not _SHA256_RE.fullmatch(canonical.sha256):
        raise TaskContractError("canonical_task.sha256 must be a lowercase SHA-256 digest")
    if canonical.path != "task.md":
        raise TaskContractError("canonical_task.path must be task.md")

    method_data = _mapping(data["method"], "method", {"class", "description"})
    method = ContractMethod(
        class_name=_string(method_data["class"], "method.class"),
        description=_string(method_data["description"], "method.description"),
    )
    candidate_data = _mapping(data["candidate"], "candidate", {"interface", "editable_paths"})
    editable_values = _string_tuple(candidate_data["editable_paths"], "candidate.editable_paths")
    candidate = Candidate(
        interface=_string(candidate_data["interface"], "candidate.interface"),
        editable_paths=tuple(
            safe_contract_relative_path(path, f"candidate.editable_paths[{index}]")
            for index, path in enumerate(editable_values)
        ),
    )
    if candidate.interface != "python_rpc_v1:handle":
        raise TaskContractError("candidate.interface must be python_rpc_v1:handle")
    if any(not _path_is_within(path, "program_0") for path in candidate.editable_paths):
        raise TaskContractError("candidate editable paths must be inside program_0")

    evaluator_data = _mapping(data["evaluators"], "evaluators", {"search", "final"})
    evaluators = Evaluators(
        search=safe_contract_relative_path(evaluator_data["search"], "evaluators.search"),
        final=safe_contract_relative_path(evaluator_data["final"], "evaluators.final"),
    )
    if evaluators.search == evaluators.final:
        raise TaskContractError("search and final evaluator paths must differ")
    if evaluators.search != "evaluate.py":
        raise TaskContractError("evaluators.search must be evaluate.py")
    if evaluators.final != "evaluate_final.py":
        raise TaskContractError("evaluators.final must be evaluate_final.py")

    resource_data = _mapping(
        data["resources"], "resources", {"search_root", "final_root", "public_artifact_bundle"}
    )
    raw_bundle = resource_data["public_artifact_bundle"]
    if raw_bundle is not None and (
        not isinstance(raw_bundle, str) or _PUBLIC_BUNDLE_RE.fullmatch(raw_bundle) is None
    ):
        raise TaskContractError("resources.public_artifact_bundle is invalid")
    resources = ResourceRoots(
        search_root=safe_contract_relative_path(
            resource_data["search_root"], "resources.search_root"
        ),
        final_root=safe_contract_relative_path(resource_data["final_root"], "resources.final_root"),
        public_artifact_bundle=raw_bundle,
    )
    if (resources.search_root, resources.final_root) != ("resources/search", "resources/final"):
        raise TaskContractError("resource roots must be resources/search and resources/final")
    if _paths_overlap(resources.search_root, resources.final_root):
        raise TaskContractError("search and final resource roots must not overlap")
    if _path_is_within(evaluators.search, resources.final_root):
        raise TaskContractError("search evaluator must not be inside the final resource root")
    if _path_is_within(evaluators.final, resources.search_root):
        raise TaskContractError("final evaluator must not be inside the search resource root")
    for editable_path in candidate.editable_paths:
        if _paths_overlap(editable_path, resources.final_root) or editable_path == evaluators.final:
            raise TaskContractError(
                f"candidate editable path must not overlap sealed final artifacts: {editable_path}"
            )

    role_data = _mapping(data["data"], "data", {"search_roles", "final_roles"})
    roles = DataRoles(
        search_roles=_string_tuple(role_data["search_roles"], "data.search_roles"),
        final_roles=_string_tuple(role_data["final_roles"], "data.final_roles"),
    )
    for role in (*roles.search_roles, *roles.final_roles):
        if not _ROLE_RE.fullmatch(role):
            raise TaskContractError(f"invalid data role: {role!r}")
    forbidden = _FINAL_ROLE_NAMES.intersection(roles.search_roles)
    if forbidden:
        raise TaskContractError(
            f"search data roles must not include final roles: {sorted(forbidden)}"
        )
    if set(roles.search_roles).intersection(roles.final_roles):
        raise TaskContractError("search and final data roles must be distinct")

    metric_data = _mapping(data["metrics"], "metrics", {"search", "final"})
    metrics = Metrics(
        search=_parse_metrics(metric_data["search"], "metrics.search"),
        final=_parse_metrics(metric_data["final"], "metrics.final"),
    )
    if "combined_score" not in {metric.name for metric in metrics.search}:
        raise TaskContractError("metrics.search must declare combined_score")
    for metric in metrics.search:
        if metric.data_role not in roles.search_roles or metric.data_role in _FINAL_ROLE_NAMES:
            raise TaskContractError(f"search metric {metric.name!r} must use a search data role")
    for metric in metrics.final:
        if metric.data_role not in roles.final_roles:
            raise TaskContractError(f"final metric {metric.name!r} must use a final data role")

    upstream_value = data["upstream"]
    if not isinstance(upstream_value, list):
        raise TaskContractError("upstream must be an array")
    upstream: list[Upstream] = []
    for index, item in enumerate(upstream_value):
        label = f"upstream[{index}]"
        item_data = _mapping(
            item,
            label,
            {"repository", "revision", "package_version", "license", "entrypoints", "patches"},
        )
        repository = _string(item_data["repository"], f"{label}.repository")
        revision = _string(item_data["revision"], f"{label}.revision")
        revision_lower = revision.lower()
        if (
            revision_lower in _FLOATING_REVISIONS
            or revision_lower.startswith("refs/heads/")
            or revision_lower.startswith("origin/")
        ):
            raise TaskContractError(f"{label}.revision must identify a pinned revision")
        package_version_value = item_data["package_version"]
        package_version = (
            None
            if package_version_value is None
            else _string(package_version_value, f"{label}.package_version")
        )
        upstream.append(
            Upstream(
                repository=repository,
                revision=revision,
                package_version=package_version,
                license=_string(item_data["license"], f"{label}.license"),
                entrypoints=_string_tuple(item_data["entrypoints"], f"{label}.entrypoints"),
                patches=tuple(
                    safe_contract_relative_path(path, f"{label}.patches[{patch_index}]")
                    for patch_index, path in enumerate(
                        _optional_string_list(item_data["patches"], f"{label}.patches")
                    )
                ),
            )
        )

    provenance_data = _mapping(data["provenance"], "provenance", {"dependencies", "artifacts"})
    dependency_values = provenance_data["dependencies"]
    if not isinstance(dependency_values, list):
        raise TaskContractError("provenance.dependencies must be an array")
    dependencies: list[Dependency] = []
    for index, item in enumerate(dependency_values):
        label = f"provenance.dependencies[{index}]"
        item_data = _mapping(item, label, {"name", "version"})
        dependencies.append(
            Dependency(
                name=_string(item_data["name"], f"{label}.name"),
                version=_string(item_data["version"], f"{label}.version"),
            )
        )
    artifact_values = provenance_data["artifacts"]
    if not isinstance(artifact_values, list):
        raise TaskContractError("provenance.artifacts must be an array")
    artifacts: list[Artifact] = []
    for index, item in enumerate(artifact_values):
        label = f"provenance.artifacts[{index}]"
        item_data = _mapping(item, label, {"role", "path", "sha256", "source"})
        digest = _string(item_data["sha256"], f"{label}.sha256").lower()
        if not _SHA256_RE.fullmatch(digest):
            raise TaskContractError(f"{label}.sha256 must be a lowercase SHA-256 digest")
        artifacts.append(
            Artifact(
                role=_string(item_data["role"], f"{label}.role"),
                path=safe_contract_relative_path(item_data["path"], f"{label}.path"),
                sha256=digest,
                source=_string(item_data["source"], f"{label}.source"),
            )
        )
    artifact_paths = [artifact.path for artifact in artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise TaskContractError("provenance.artifacts paths must be unique")
    declared_patch_paths = {patch for item in upstream for patch in item.patches}
    missing_patch_hashes = declared_patch_paths - set(artifact_paths)
    if missing_patch_hashes:
        raise TaskContractError(
            "upstream patches must have provenance artifact checksums: "
            + ", ".join(sorted(missing_patch_hashes))
        )
    provenance = Provenance(dependencies=tuple(dependencies), artifacts=tuple(artifacts))

    compute_data = _mapping(data["compute"], "compute", {"search", "final"})
    compute = Compute(
        search=_parse_compute_plane(compute_data["search"], "compute.search"),
        final=_parse_compute_plane(compute_data["final"], "compute.final"),
    )

    protocol_data = _mapping_with_optional(
        data["protocol"],
        "protocol",
        {
            "search_seeds",
            "final_seeds",
            "final_reset_sequence",
            "baselines",
            "promotion_rule",
            "search_proxies",
        },
        {"search_reset_sequence", "search_smoke_reset_sequence", "final_smoke_reset_sequence"},
    )
    search_seeds = _seed_tuple(protocol_data["search_seeds"], "protocol.search_seeds")
    final_seeds = _seed_tuple(protocol_data["final_seeds"], "protocol.final_seeds")
    search_reset_sequence = _reset_sequence(
        protocol_data.get("search_reset_sequence", list(search_seeds)),
        label="protocol.search_reset_sequence",
        statistical_seeds=search_seeds,
        max_candidate_rpc_calls=compute.search.max_candidate_rpc_calls,
        plane="search",
    )
    final_reset_sequence = _reset_sequence(
        protocol_data["final_reset_sequence"],
        label="protocol.final_reset_sequence",
        statistical_seeds=final_seeds,
        max_candidate_rpc_calls=compute.final.max_candidate_rpc_calls,
        plane="final",
    )
    protocol = Protocol(
        search_seeds=search_seeds,
        final_seeds=final_seeds,
        search_reset_sequence=search_reset_sequence,
        search_smoke_reset_sequence=_smoke_reset_sequence(
            protocol_data.get("search_smoke_reset_sequence", [search_reset_sequence[0]]),
            label="protocol.search_smoke_reset_sequence",
            production_sequence=search_reset_sequence,
            max_candidate_rpc_calls=compute.search.max_candidate_rpc_calls,
            plane="search",
        ),
        final_reset_sequence=final_reset_sequence,
        final_smoke_reset_sequence=_smoke_reset_sequence(
            protocol_data.get("final_smoke_reset_sequence", [final_reset_sequence[0]]),
            label="protocol.final_smoke_reset_sequence",
            production_sequence=final_reset_sequence,
            max_candidate_rpc_calls=compute.final.max_candidate_rpc_calls,
            plane="final",
        ),
        baselines=_string_tuple(protocol_data["baselines"], "protocol.baselines"),
        promotion_rule=_string(protocol_data["promotion_rule"], "protocol.promotion_rule"),
        search_proxies=_parse_search_proxies(protocol_data["search_proxies"]),
    )
    if set(search_seeds).intersection(final_seeds):
        raise TaskContractError("protocol search and final seeds must be disjoint")
    feedback_data = _mapping(data["feedback"], "feedback", {"search_visible", "final_sealed"})
    feedback = Feedback(
        search_visible=_string_tuple(feedback_data["search_visible"], "feedback.search_visible"),
        final_sealed=_string_tuple(feedback_data["final_sealed"], "feedback.final_sealed"),
    )
    leaked_feedback = set(feedback.search_visible).intersection(feedback.final_sealed)
    if leaked_feedback:
        raise TaskContractError(
            "feedback search-visible and final-sealed declarations overlap: "
            f"{sorted(leaked_feedback)}"
        )
    publication_data = _mapping(
        data["publication"],
        "publication",
        {"search_is_proxy", "requires_final_evaluation", "required_evidence"},
    )
    for name in ("search_is_proxy", "requires_final_evaluation"):
        if not isinstance(publication_data[name], bool):
            raise TaskContractError(f"publication.{name} must be a boolean")
    publication = Publication(
        search_is_proxy=publication_data["search_is_proxy"],
        requires_final_evaluation=publication_data["requires_final_evaluation"],
        required_evidence=_string_tuple(
            publication_data["required_evidence"], "publication.required_evidence"
        ),
    )
    return TaskContract(
        contract_schema_version=version,
        canonical_task=canonical,
        method=method,
        candidate=candidate,
        evaluators=evaluators,
        resources=resources,
        data=roles,
        metrics=metrics,
        upstream=tuple(upstream),
        provenance=provenance,
        compute=compute,
        protocol=protocol,
        feedback=feedback,
        publication=publication,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise TaskContractError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def parse_task_contract_bytes(raw: bytes) -> TaskContract:
    """Parse the bytes of a ``task_contract.json`` with upstream's size and duplicate-key rules."""
    if len(raw) > MAX_TASK_CONTRACT_BYTES:
        raise TaskContractError(
            f"task contract exceeds {MAX_TASK_CONTRACT_BYTES} byte harness limit"
        )
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except TaskContractError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise TaskContractError(f"failed to load {TASK_CONTRACT_FILENAME}: {exc}") from exc
    return parse_task_contract(value)


async def _is_symlink(workspace: Workspace, path: str) -> bool:
    entry = await workspace.stat(path)
    return entry is not None and entry.kind == "symlink"


async def _is_file(workspace: Workspace, path: str) -> bool:
    entry = await workspace.stat(path)
    return entry is not None and entry.kind == "file"


async def _is_dir(workspace: Workspace, path: str) -> bool:
    entry = await workspace.stat(path)
    return entry is not None and entry.kind == "dir"


def _join(root: str, relative: str) -> str:
    return f"{root.rstrip('/')}/{relative}" if root not in ("", ".") else relative


async def _reject_symlink_components(workspace: Workspace, task_dir: str, relative: str) -> None:
    current = task_dir
    for part in PurePosixPath(relative).parts:
        current = _join(current, part)
        if await _is_symlink(workspace, current):
            raise TaskContractError(f"declared artifact must not contain symlinks: {relative}")


async def _read_declared_file(workspace: Workspace, task_dir: str, relative: str) -> bytes:
    await _reject_symlink_components(workspace, task_dir, relative)
    path = _join(task_dir, relative)
    if not await _is_file(workspace, path):
        raise TaskContractError(f"declared artifact is not a file: {relative}")
    return await workspace.read_bytes(path)


async def load_task_contract(workspace: Workspace, task_dir: str) -> TaskContract:
    """Load ``<task_dir>/task_contract.json`` and verify its declared digests (upstream rules).

    Raises:
        TaskContractError: When the contract is invalid, a digest differs, or a frozen resource
            file has no provenance checksum.
    """
    contract_path = _join(task_dir, TASK_CONTRACT_FILENAME)
    if await _is_symlink(workspace, contract_path):
        raise TaskContractError(f"task contract must not be a symlink: {contract_path}")
    try:
        raw = await workspace.read_bytes(contract_path)
    except FileNotFoundError as exc:
        raise TaskContractError(f"failed to load {contract_path}: {exc}") from exc
    contract = parse_task_contract_bytes(raw)
    task_bytes = await _read_declared_file(workspace, task_dir, contract.canonical_task.path)
    if hashlib.sha256(task_bytes).hexdigest() != contract.canonical_task.sha256:
        raise TaskContractError("canonical task digest mismatch")
    for artifact in contract.provenance.artifacts:
        artifact_bytes = await _read_declared_file(workspace, task_dir, artifact.path)
        if hashlib.sha256(artifact_bytes).hexdigest() != artifact.sha256:
            raise TaskContractError(f"provenance artifact digest mismatch: {artifact.path}")
    declared_artifacts = {artifact.path for artifact in contract.provenance.artifacts}
    frozen_resource_files: set[str] = set()
    for root_relative in (contract.resources.search_root, contract.resources.final_root):
        await _reject_symlink_components(workspace, task_dir, root_relative)
        root = _join(task_dir, root_relative)
        if not await _is_dir(workspace, root):
            raise TaskContractError(f"declared resource root is not a directory: {root_relative}")
        for entry in await workspace.list_tree(root):
            relative = _join(root_relative, entry.path)
            if entry.kind == "symlink":
                raise TaskContractError(f"declared resource contains a symlink: {relative}")
            name = PurePosixPath(entry.path).name
            if entry.kind == "file" and name != RESOURCE_MANIFEST_FILENAME:
                frozen_resource_files.add(relative)
    missing_artifacts = frozen_resource_files - declared_artifacts
    if missing_artifacts:
        raise TaskContractError(
            "frozen resource files missing provenance checksums: "
            + ", ".join(sorted(missing_artifacts))
        )
    return contract


def _hash_file(hasher: Any, relative: str, content: bytes) -> None:
    hasher.update(
        b"file\0" + relative.encode() + b"\0" + str(len(content)).encode() + b"\0" + content
    )


async def task_scaffold_fingerprint(workspace: Workspace, task_dir: str) -> str:
    """Hash the contract, canonical task, evaluators, and both declared resource trees.

    The byte layout is upstream's ``task_scaffold_fingerprint``, so a fingerprint computed by the
    port equals the one upstream computes for the same scaffold.
    """
    contract = await load_task_contract(workspace, task_dir)
    hasher = hashlib.sha256()
    for relative in (
        TASK_CONTRACT_FILENAME,
        contract.canonical_task.path,
        contract.evaluators.search,
        contract.evaluators.final,
    ):
        _hash_file(hasher, relative, await _read_declared_file(workspace, task_dir, relative))
    for root_relative in (contract.resources.search_root, contract.resources.final_root):
        await _reject_symlink_components(workspace, task_dir, root_relative)
        root = _join(task_dir, root_relative)
        if not await _is_dir(workspace, root):
            raise TaskContractError(f"declared resource root is not a directory: {root_relative}")
        hasher.update(b"dir\0" + root_relative.encode() + b"\0")
        entries = sorted(await workspace.list_tree(root), key=lambda item: item.path)
        for entry in entries:
            relative = _join(root_relative, entry.path)
            if PurePosixPath(entry.path).name == RESOURCE_MANIFEST_FILENAME:
                continue
            if entry.kind == "symlink":
                raise TaskContractError(f"declared resource contains a symlink: {relative}")
            if entry.kind == "dir":
                hasher.update(b"dir\0" + relative.encode() + b"\0")
            elif entry.kind == "file":
                _hash_file(hasher, relative, await workspace.read_bytes(_join(task_dir, relative)))
            else:
                raise TaskContractError(f"unsupported declared resource: {relative}")
    return "sha256:" + hasher.hexdigest()


# --- candidate-visible projection (upstream public_task_contract.py) ------------------------


def _prompt_text(value: str, label: str) -> str:
    if len(value.encode("utf-8")) > MAX_PUBLIC_SEARCH_TEXT_BYTES or any(
        unicodedata.category(character).startswith("C") for character in value
    ):
        raise TaskContractError(f"unsafe prompt text in {label}")
    return value


def public_search_payload(contract: TaskContract) -> dict[str, Any]:
    """Return the candidate-visible subset of a contract, without final-evaluation fields.

    This is the ``payload`` of upstream ``render_public_search_contract``; the prompt text that
    frames it is rendered by the investigation package.
    """
    search_compute = contract.compute.search
    return {
        "candidate_interface": _prompt_text(contract.candidate.interface, "candidate.interface"),
        "search": {
            "metrics": [
                {
                    "name": _prompt_text(metric.name, "metrics.search.name"),
                    "data_role": _prompt_text(metric.data_role, "metrics.search.data_role"),
                }
                for metric in contract.metrics.search
            ],
            "precision": _prompt_text(search_compute.precision, "compute.search.precision"),
            "compute": {
                "execution_mode": _prompt_text(
                    search_compute.execution_mode, "compute.search.execution_mode"
                ),
                "accelerator": search_compute.accelerator,
                "accelerator_model": _prompt_text(
                    search_compute.accelerator_model, "compute.search.accelerator_model"
                ),
                "gpu_count": search_compute.gpu_count,
                "evaluator_gpu_count": search_compute.evaluator_gpu_count,
                "candidate_gpu_count": search_compute.candidate_gpu_count,
                "cpu_cores": search_compute.cpu_cores,
                "memory_gb": search_compute.memory_gb,
                "wall_clock_seconds": search_compute.wall_clock_seconds,
                "max_candidate_rpc_calls": search_compute.max_candidate_rpc_calls,
            },
            "seeds": list(contract.protocol.search_seeds),
        },
    }


def indented_public_search_payload(contract: TaskContract) -> str:
    """Return the payload serialized and indented exactly as upstream embeds it in the prompt."""
    serialized = json.dumps(public_search_payload(contract), indent=2, ensure_ascii=True)
    return "\n".join(f"    {line}" for line in serialized.split("\n"))


__all__ = [
    "DEFAULT_MAX_CANDIDATE_RPC_CALLS",
    "MAX_PUBLIC_SEARCH_CONTRACT_BYTES",
    "TASK_CONTRACT_FILENAME",
    "Artifact",
    "Candidate",
    "CanonicalTask",
    "Compute",
    "ComputePlane",
    "ContractMethod",
    "DataRoles",
    "Dependency",
    "Evaluators",
    "Feedback",
    "Metric",
    "Metrics",
    "Protocol",
    "Provenance",
    "Publication",
    "ResourceRoots",
    "SearchProxy",
    "TaskContract",
    "Upstream",
    "indented_public_search_payload",
    "load_task_contract",
    "parse_task_contract",
    "parse_task_contract_bytes",
    "public_search_payload",
    "safe_contract_relative_path",
    "task_scaffold_fingerprint",
]


def _complete_forward_references() -> None:
    from . import state

    state.TaskRef.model_rebuild(_types_namespace={"TaskContract": TaskContract})


_complete_forward_references()
