"""The contract of the task design loop: validation, patching and comparison of designs.

A port of upstream's ``src/task_drafting/design.py`` (strict, task-general design and review
contracts), of the data logic that its command-line adapter ``design_cli.py`` calls, of
``latent_review.py`` (rebinding unchanged full-review failures to a retained parent), and of the
Python snippets embedded in ``src/scripts/lib/task_design.sh`` (the repair feedback payload, the
repair memory, the focus rotation, the focused authorisation and the frontier tests). Messages of
:class:`TaskDesignError` are upstream's, because they travel back to the roles as the diagnostic
of a retry.

Everything here is a pure function. The roles answer through structured output, so the typed
shape of each record is guaranteed by pydantic; the rules that pydantic cannot state (hashes,
JSON pointers, scope containment, verdict arithmetic) are the ones upstream's loaders check.

Not ported: the three-review quorum (``aggregate_task_design_reviews``,
``task_design_adjudication_request``, ``load_task_design_adjudication``,
``task_design_scope_alternatives``) and the scope rotation that depends on it. Upstream's driver
runs one judge and never calls them (``run_design_consensus``), so there is no upstream behaviour
to match; ``docs/adaptations/drafting_design.md`` records this.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

from ..core.errors import ScientistError
from .design_models import (
    DESIGN_DIMENSIONS,
    IMMUTABLE_DESIGN_PATHS,
    MAX_DESIGN_ARTIFACT_BYTES,
    MAX_DESIGN_PATCH_OPERATIONS,
    DeltaAudit,
    DesignPatch,
    DesignProgress,
    DesignReview,
    RepairMemory,
    TaskDesign,
    canonical_json,
)

_SAFE_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+@-]*$")


class TaskDesignError(ScientistError, ValueError):
    """A design, review, patch or delta audit violates its contract (upstream's message)."""


# --- field validators ---------------------------------------------------------------------------


def _exact_fields(value: object, label: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TaskDesignError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise TaskDesignError(f"{label} fields mismatch: missing={missing}, extra={extra}")
    return value


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TaskDesignError(f"{label} must be a non-empty string")
    return " ".join(value.split())


def _text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise TaskDesignError(f"{label} must be a non-empty array")
    normalized = [_nonempty_text(item, f"{label}[]") for item in value]
    if len(set(normalized)) != len(normalized):
        raise TaskDesignError(f"{label} must contain unique values")
    return normalized


def _candidate_editable_paths(value: object) -> list[str]:
    paths = _text_list(value, "candidate.editable_paths")
    for raw in paths:
        path = PurePosixPath(raw)
        if (
            path.is_absolute()
            or "\\" in raw
            or any(not _SAFE_PATH_COMPONENT_RE.fullmatch(part) for part in raw.split("/"))
            or any(part in {"", ".", ".."} for part in path.parts)
            or not path.parts
            or path.parts[0] != "program_0"
            or len(path.parts) < 2
            or path.parts[1] == "program_spec.toml"
        ):
            raise TaskDesignError(
                "candidate.editable_paths must contain safe relative paths inside program_0"
            )
    return paths


def _possibly_empty_text_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TaskDesignError(f"{label} must be an array")
    normalized = [_nonempty_text(item, f"{label}[]") for item in value]
    if len(set(normalized)) != len(normalized):
        raise TaskDesignError(f"{label} must contain unique values")
    return normalized


def _seed_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise TaskDesignError(f"{label} must be a non-empty array")
    if any(type(seed) is not int or seed < 0 for seed in value):
        raise TaskDesignError(f"{label} must contain non-negative integers")
    if len(set(value)) != len(value):
        raise TaskDesignError(f"{label} must contain unique values")
    return value


def _validate_upstream(value: object) -> None:
    if not isinstance(value, list):
        raise TaskDesignError("upstream must be an array")
    repositories: set[str] = set()
    for index, item in enumerate(value):
        row = _exact_fields(
            item, f"upstream[{index}]", {"repository", "revision", "license", "entrypoints"}
        )
        repository = _nonempty_text(row["repository"], f"upstream[{index}].repository")
        if repository in repositories:
            raise TaskDesignError("upstream repositories must be unique")
        repositories.add(repository)
        _nonempty_text(row["revision"], f"upstream[{index}].revision")
        _nonempty_text(row["license"], f"upstream[{index}].license")
        _text_list(row["entrypoints"], f"upstream[{index}].entrypoints")


def _validate_plane(value: object, label: str) -> list[int]:
    plane = _exact_fields(
        value,
        label,
        {"resources", "seeds", "budget", "baselines", "metrics", "correctness", "smoke"},
    )
    _text_list(plane["resources"], f"{label}.resources")
    seeds = _seed_list(plane["seeds"], f"{label}.seeds")
    _text_list(plane["baselines"], f"{label}.baselines")
    _text_list(plane["metrics"], f"{label}.metrics")
    _text_list(plane["correctness"], f"{label}.correctness")
    _nonempty_text(plane["smoke"], f"{label}.smoke")
    budget = _exact_fields(
        plane["budget"],
        f"{label}.budget",
        {"unit", "training_steps", "wall_clock_seconds", "hardware", "precision"},
    )
    _nonempty_text(budget["unit"], f"{label}.budget.unit")
    _nonempty_text(budget["hardware"], f"{label}.budget.hardware")
    _nonempty_text(budget["precision"], f"{label}.budget.precision")
    for field in ("training_steps", "wall_clock_seconds"):
        if type(budget[field]) is not int or budget[field] < 1:
            raise TaskDesignError(f"{label}.budget.{field} must be a positive integer")
    return seeds


def validate_design_payload(payload: Mapping[str, Any], canonical_task: bytes) -> None:
    """Check a design object against upstream's TaskDesign v1 contract.

    Args:
        payload: The design as a JSON object.
        canonical_task: The bytes of ``task.md``; ``canonical_task.sha256`` must be their digest.

    Raises:
        TaskDesignError: With upstream's message when the design violates the contract.
    """
    data = _exact_fields(
        payload,
        "task design",
        {
            "schema_version",
            "canonical_task",
            "method",
            "upstream",
            "candidate",
            "planes",
            "feedback",
            "publication",
        },
    )
    if data["schema_version"] != 1 or type(data["schema_version"]) is not int:
        raise TaskDesignError("task design schema_version must be 1")
    canonical = _exact_fields(data["canonical_task"], "canonical_task", {"path", "sha256"})
    if canonical["path"] != "task.md":
        raise TaskDesignError("canonical_task.path must be task.md")
    if canonical["sha256"] != hashlib.sha256(canonical_task).hexdigest():
        raise TaskDesignError("canonical task hash does not match task.md")
    method = _exact_fields(data["method"], "method", {"class", "claim"})
    _nonempty_text(method["class"], "method.class")
    _nonempty_text(method["claim"], "method.claim")
    _validate_upstream(data["upstream"])
    candidate = _exact_fields(
        data["candidate"], "candidate", {"interface", "editable_paths", "state_semantics"}
    )
    interface = _nonempty_text(candidate["interface"], "candidate.interface")
    if interface != "python_rpc_v1:handle":
        raise TaskDesignError("candidate.interface must be python_rpc_v1:handle")
    _candidate_editable_paths(candidate["editable_paths"])
    _nonempty_text(candidate["state_semantics"], "candidate.state_semantics")
    planes = _exact_fields(data["planes"], "planes", {"search", "final"})
    search_seeds = _validate_plane(planes["search"], "planes.search")
    final_seeds = _validate_plane(planes["final"], "planes.final")
    if set(search_seeds).intersection(final_seeds):
        raise TaskDesignError("search and final seeds must be disjoint")
    feedback = _exact_fields(data["feedback"], "feedback", {"search_visible", "final_sealed"})
    visible = set(_text_list(feedback["search_visible"], "feedback.search_visible"))
    sealed = set(_text_list(feedback["final_sealed"], "feedback.final_sealed"))
    if visible.intersection(sealed):
        raise TaskDesignError("search-visible and final-sealed feedback must be disjoint")
    publication = _exact_fields(data["publication"], "publication", {"required_evidence"})
    _text_list(publication["required_evidence"], "publication.required_evidence")


def validate_design(design: TaskDesign, canonical_task: bytes) -> str:
    """Validate a design and return its fingerprint (upstream ``validate-design``).

    Raises:
        TaskDesignError: When the design violates the contract.
    """
    payload = design.payload()
    if len(canonical_json(payload).encode()) + 1 > MAX_DESIGN_ARTIFACT_BYTES:
        raise TaskDesignError("task design exceeds 1 MiB")
    validate_design_payload(payload, canonical_task)
    return design.fingerprint()


def design_from_payload(payload: Mapping[str, Any], canonical_task: bytes) -> TaskDesign:
    """Validate a design object and return it as a :class:`TaskDesign`.

    Raises:
        TaskDesignError: When the design violates the contract.
    """
    validate_design_payload(payload, canonical_task)
    return TaskDesign.model_validate(dict(payload))


# --- JSON pointers ------------------------------------------------------------------------------


def decode_json_pointer(path: object, *, allow_protected: bool = False) -> list[str]:
    """Decode a JSON pointer into tokens (upstream ``_decode_json_pointer``).

    The root, ``/schema_version`` and ``/canonical_task`` are protected unless
    ``allow_protected`` is true.

    Raises:
        TaskDesignError: For a malformed or protected pointer.
    """
    if not isinstance(path, str) or not path.startswith("/"):
        raise TaskDesignError("task design patch cannot replace the document root")
    decoded: list[str] = []
    for token in path[1:].split("/"):
        index = 0
        value = ""
        while index < len(token):
            if token[index] != "~":
                value += token[index]
                index += 1
                continue
            if index + 1 >= len(token) or token[index + 1] not in {"0", "1"}:
                raise TaskDesignError(f"invalid JSON pointer escape: {path}")
            value += "~" if token[index + 1] == "0" else "/"
            index += 2
        decoded.append(value)
    if not decoded or not decoded[0]:
        raise TaskDesignError("task design patch cannot replace the document root")
    if not allow_protected and decoded[0] in {"schema_version", "canonical_task"}:
        raise TaskDesignError(f"task design patch cannot modify {decoded[0]}")
    return decoded


def _array_index(token: str, length: int, path: str) -> int:
    if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
        raise TaskDesignError(f"invalid array index in JSON pointer: {path}")
    index = int(token)
    if index >= length:
        raise TaskDesignError(f"task design path does not exist: {path}")
    return index


def resolve_json_pointer(root: object, tokens: Sequence[str], path: str) -> object:
    """Return the value at the decoded pointer ``tokens`` of ``root``.

    Raises:
        TaskDesignError: When the path does not exist.
    """
    target = root
    for token in tokens:
        if isinstance(target, dict) and token in target:
            target = target[token]
        elif isinstance(target, list):
            target = target[_array_index(token, len(target), path)]
        else:
            raise TaskDesignError(f"task design path does not exist: {path}")
    return target


def _replace_pointer(root: object, tokens: Sequence[str], path: str, value: object) -> None:
    parent = resolve_json_pointer(root, tokens[:-1], path)
    leaf = tokens[-1]
    if isinstance(parent, dict) and leaf in parent:
        parent[leaf] = value
    elif isinstance(parent, list):
        parent[_array_index(leaf, len(parent), path)] = value
    else:
        raise TaskDesignError(f"task design patch path does not exist: {path}")


def paths_overlap(first: str, second: str) -> bool:
    """Return whether one JSON pointer equals or contains the other."""
    return first == second or first.startswith(f"{second}/") or second.startswith(f"{first}/")


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _changed_design_paths(parent: object, candidate: object, path: str = "") -> tuple[str, ...]:
    if isinstance(parent, dict) and isinstance(candidate, dict):
        changed: list[str] = []
        for key in sorted(parent.keys() | candidate.keys()):
            child_path = f"{path}/{_json_pointer_token(key)}"
            if key not in parent or key not in candidate:
                changed.append(child_path)
            else:
                changed.extend(_changed_design_paths(parent[key], candidate[key], child_path))
        return tuple(changed)
    if isinstance(parent, list) and isinstance(candidate, list):
        if len(parent) != len(candidate):
            return (path or "/",)
        return tuple(
            changed
            for index, (parent_item, candidate_item) in enumerate(
                zip(parent, candidate, strict=True)
            )
            for changed in _changed_design_paths(parent_item, candidate_item, f"{path}/{index}")
        )
    return () if type(parent) is type(candidate) and parent == candidate else (path or "/",)


def task_design_changed_paths(parent: TaskDesign, candidate: TaskDesign) -> tuple[str, ...]:
    """Return the leaf JSON pointers that differ between two designs (deterministic order)."""
    return _changed_design_paths(parent.payload(), candidate.payload())


# --- reviews ------------------------------------------------------------------------------------


def validate_review(review: DesignReview, design: TaskDesign) -> None:
    """Check a schema-3 review against the design it reviews (upstream ``validate-review``).

    The loop always requires schema 3 (``--require-schema-version 3``); the legacy schemas 1 and
    2 that upstream's loader also reads are not carried, because no wired path accepts them.

    Args:
        review: The judge's review.
        design: The reviewed design.

    Raises:
        TaskDesignError: With upstream's message when the review violates the contract.
    """
    payload = review.payload()
    if len(canonical_json(payload).encode()) + 1 > MAX_DESIGN_ARTIFACT_BYTES:
        raise TaskDesignError("task design review exceeds 1 MiB")
    schema_version = payload["schema_version"]
    if schema_version != 3:
        raise TaskDesignError("task design review schema_version must be 3")
    if payload["design_fingerprint"] != design.fingerprint():
        raise TaskDesignError("task design review fingerprint does not match design")
    _nonempty_text(payload["summary"], "task design review summary")
    design_payload = design.payload()
    rows: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(payload["dimensions"]):
        row = _exact_fields(
            value,
            f"dimensions[{index}]",
            {
                "id",
                "verdict",
                "evidence",
                "remediation",
                "design_paths",
                "task_evidence",
                "repair_paths",
            },
        )
        identifier = _nonempty_text(row["id"], f"dimensions[{index}].id")
        if identifier in rows:
            raise TaskDesignError(f"duplicate design dimension: {identifier}")
        rows[identifier] = row
    if set(rows) != set(DESIGN_DIMENSIONS):
        raise TaskDesignError("design review must contain exactly the required dimensions")
    failed: list[str] = []
    for identifier in DESIGN_DIMENSIONS:
        row = rows[identifier]
        verdict = row["verdict"]
        if verdict not in {"pass", "fail"}:
            raise TaskDesignError(f"dimension {identifier} verdict must be pass or fail")
        _text_list(row["evidence"], f"dimension {identifier} evidence")
        design_paths = _text_list(row["design_paths"], f"dimension {identifier} design_paths")
        _text_list(row["task_evidence"], f"dimension {identifier} task_evidence")
        for design_path in design_paths:
            tokens = decode_json_pointer(design_path, allow_protected=True)
            try:
                resolve_json_pointer(design_payload, tokens, design_path)
            except TaskDesignError as exc:
                raise TaskDesignError(
                    f"dimension {identifier} design path does not exist: {design_path}"
                ) from exc
        repair_paths = _possibly_empty_text_list(
            row["repair_paths"], f"dimension {identifier} repair_paths"
        )
        if verdict == "pass" and repair_paths:
            raise TaskDesignError(f"passed dimension {identifier} must not have repair_paths")
        for repair_path in repair_paths:
            try:
                tokens = decode_json_pointer(repair_path)
            except TaskDesignError as exc:
                raise TaskDesignError(
                    f"dimension {identifier} repair path is not writable: {repair_path}"
                ) from exc
            if any(paths_overlap(repair_path, immutable) for immutable in IMMUTABLE_DESIGN_PATHS):
                raise TaskDesignError(
                    f"dimension {identifier} repair path is immutable: {repair_path}"
                )
            try:
                resolve_json_pointer(design_payload, tokens, repair_path)
            except TaskDesignError as exc:
                raise TaskDesignError(
                    f"dimension {identifier} repair path does not exist: {repair_path}"
                ) from exc
            if not any(
                repair_path == design_path or repair_path.startswith(f"{design_path}/")
                for design_path in design_paths
            ):
                raise TaskDesignError(
                    f"dimension {identifier} repair path lacks matching design evidence: "
                    f"{repair_path}"
                )
        remediation = str(row["remediation"]).strip()
        if verdict == "fail":
            failed.append(identifier)
            if not remediation:
                raise TaskDesignError(f"failed dimension {identifier} requires remediation")
            if not repair_paths:
                raise TaskDesignError(f"failed dimension {identifier} requires repair_paths")
        elif remediation:
            raise TaskDesignError(f"passed dimension {identifier} must not have remediation")
    expected = "fail" if failed else "pass"
    if payload["verdict"] != expected:
        raise TaskDesignError("design review verdict does not match dimensions")


# --- patches ------------------------------------------------------------------------------------


def apply_design_patch(
    patch: DesignPatch,
    *,
    parent: TaskDesign,
    canonical_task: bytes,
    repair_review: DesignReview,
) -> TaskDesign:
    """Apply a bounded replace-only patch to a validated parent design.

    The failed rows of ``repair_review`` are the authority: every patched path must equal or
    descend from one of their ``repair_paths``. The result is validated like any design and must
    differ from the parent.

    Raises:
        TaskDesignError: With upstream's message when the patch or its scope is invalid.
    """
    payload = patch.payload()
    _exact_fields(
        payload, "task design patch", {"schema_version", "parent_fingerprint", "operations"}
    )
    if payload["schema_version"] != 1:
        raise TaskDesignError("task design patch schema_version must be 1")
    if payload["parent_fingerprint"] != parent.fingerprint():
        raise TaskDesignError("task design patch parent fingerprint does not match design")
    operations = payload["operations"]
    if not isinstance(operations, list) or not operations:
        raise TaskDesignError("task design patch operations must be a non-empty array")
    if len(operations) > MAX_DESIGN_PATCH_OPERATIONS:
        raise TaskDesignError(f"task design patch exceeds {MAX_DESIGN_PATCH_OPERATIONS} operations")
    candidate = deepcopy(parent.payload())
    if repair_review.verdict != "fail":
        raise TaskDesignError("task design repair review must fail")
    allowed_paths = tuple(
        decode_json_pointer(repair_path, allow_protected=True)
        for row in repair_review.dimensions
        if row.verdict == "fail"
        for repair_path in row.repair_paths
    )
    if not allowed_paths:
        raise TaskDesignError("task design repair review has no failed repair paths")
    seen_paths: set[str] = set()
    for index, operation in enumerate(operations):
        row = _exact_fields(operation, f"operations[{index}]", {"op", "path", "value"})
        if row["op"] != "replace":
            raise TaskDesignError("task design patch operations must use replace")
        pointer = row["path"]
        if not isinstance(pointer, str):
            raise TaskDesignError(f"operations[{index}].path must be a string")
        if pointer in seen_paths:
            raise TaskDesignError(f"duplicate task design patch path: {pointer}")
        seen_paths.add(pointer)
        tokens = decode_json_pointer(pointer)
        if any(paths_overlap(pointer, immutable) for immutable in IMMUTABLE_DESIGN_PATHS):
            raise TaskDesignError(f"task design patch path is immutable: {pointer}")
        if not any(tokens[: len(allowed)] == allowed for allowed in allowed_paths):
            raise TaskDesignError(
                f"task design patch path is outside failed repair paths: {pointer}"
            )
        _replace_pointer(candidate, tokens, pointer, deepcopy(row["value"]))
    repaired = design_from_payload(candidate, canonical_task)
    if repaired.fingerprint() == parent.fingerprint():
        raise TaskDesignError("task design patch must change the design")
    return repaired


# --- comparing reviews --------------------------------------------------------------------------


def compare_reviews(
    parent: DesignReview,
    candidate: DesignReview,
    *,
    parent_design: TaskDesign,
    candidate_design: TaskDesign,
) -> DesignProgress:
    """Describe whether a candidate strictly reduces the parent's failures.

    A newly failing dimension counts as regressed only when a path that changed between the two
    designs overlaps a path the candidate's row cites (upstream's causal test).
    """
    parent_failed = set(parent.failed_dimensions)
    candidate_failed = set(candidate.failed_dimensions)
    candidate_rows = {row.id: row for row in candidate.dimensions}
    changed_paths = _changed_design_paths(parent_design.payload(), candidate_design.payload())
    candidate_cited_paths = {
        identifier: tuple(dict.fromkeys([*row.design_paths, *row.repair_paths]))
        for identifier, row in candidate_rows.items()
    }
    causally_regressed = {
        identifier
        for identifier in candidate_failed - parent_failed
        if (
            not (cited_paths := candidate_cited_paths[identifier])
            or any(
                paths_overlap(changed_path, cited_path)
                for changed_path in changed_paths
                for cited_path in cited_paths
            )
        )
    }
    fixed = [i for i in DESIGN_DIMENSIONS if i in parent_failed - candidate_failed]
    regressed = [i for i in DESIGN_DIMENSIONS if i in causally_regressed]
    return DesignProgress(
        fixed=fixed,
        still_failed=[i for i in DESIGN_DIMENSIONS if i in parent_failed & candidate_failed],
        regressed=regressed,
        improved=bool(fixed) and not regressed,
    )


# --- delta audits -------------------------------------------------------------------------------


def _dimension_ids(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TaskDesignError(f"{label} must be an array")
    identifiers = tuple(_nonempty_text(item, f"{label}[]") for item in value)
    if len(set(identifiers)) != len(identifiers):
        raise TaskDesignError(f"{label} must contain unique values")
    unknown = set(identifiers) - set(DESIGN_DIMENSIONS)
    if unknown:
        raise TaskDesignError(f"{label} contains unknown dimensions: {sorted(unknown)}")
    return identifiers


def validate_delta(delta: DeltaAudit, *, parent: TaskDesign, candidate: TaskDesign) -> None:
    """Check a delta audit against the two designs it compares (upstream ``validate-delta``).

    Raises:
        TaskDesignError: With upstream's message when the audit violates the contract.
    """
    payload = delta.payload()
    schema_version = payload["schema_version"]
    if schema_version not in {1, 2}:
        raise TaskDesignError("task design delta review schema_version must be 1 or 2")
    if payload["parent_fingerprint"] != parent.fingerprint():
        raise TaskDesignError("task design delta parent fingerprint does not match")
    if payload["candidate_fingerprint"] != candidate.fingerprint():
        raise TaskDesignError("task design delta candidate fingerprint does not match")
    changed_paths = _changed_design_paths(parent.payload(), candidate.payload())
    candidate_payload = candidate.payload()
    _nonempty_text(payload["summary"], "task design delta review summary")
    _dimension_ids(payload["fixed_dimensions"], "fixed_dimensions")
    regressions = payload["regressions"]
    regressed: list[str] = []
    for index, value in enumerate(regressions):
        fields = {"dimension", "design_paths", "evidence", "remediation"}
        if schema_version == 2:
            fields.add("repair_paths")
        row = _exact_fields(value, f"regressions[{index}]", fields)
        identifier = _nonempty_text(row["dimension"], f"regressions[{index}].dimension")
        if identifier not in DESIGN_DIMENSIONS:
            raise TaskDesignError(f"unknown regressed dimension: {identifier}")
        if identifier in regressed:
            raise TaskDesignError(f"duplicate regressed dimension: {identifier}")
        regressed.append(identifier)
        for design_path in _text_list(row["design_paths"], f"regressions[{index}].design_paths"):
            try:
                resolve_json_pointer(
                    candidate_payload,
                    decode_json_pointer(design_path, allow_protected=True),
                    design_path,
                )
            except TaskDesignError as exc:
                raise TaskDesignError(
                    f"regression design path does not exist: {design_path}"
                ) from exc
            if not any(paths_overlap(design_path, changed) for changed in changed_paths):
                raise TaskDesignError(
                    f"regression design path was not changed by the candidate: {design_path}"
                )
        if schema_version == 2:
            for repair_path in _text_list(
                row["repair_paths"], f"regressions[{index}].repair_paths"
            ):
                try:
                    resolve_json_pointer(
                        candidate_payload,
                        decode_json_pointer(repair_path, allow_protected=True),
                        repair_path,
                    )
                except TaskDesignError as exc:
                    raise TaskDesignError(
                        f"regression repair path does not exist: {repair_path}"
                    ) from exc
        _text_list(row["evidence"], f"regressions[{index}].evidence")
        _nonempty_text(row["remediation"], f"regressions[{index}].remediation")
    expected = "fail" if regressed else "pass"
    if payload["verdict"] != expected:
        raise TaskDesignError("task design delta verdict does not match regressions")


def confirm_delta(
    delta: DeltaAudit, progress: DesignProgress, candidate_review: DesignReview
) -> None:
    """Require the delta auditor to account for every consensus-claimed fix.

    The full review names the candidate's failures; the delta audit independently decides
    whether the repair caused them. A disputed regression may therefore stay as a failed,
    quarantined frontier finding without blocking later repair.

    Raises:
        TaskDesignError: When the audit's fixes or repair paths contradict the consensus.
    """
    fixed = set(delta.fixed_dimensions)
    regressed = set(delta.regressed_dimensions)
    claimed = set(progress.fixed)
    claimed_regressed = set(progress.regressed)
    candidate_rows = {row.id: row for row in candidate_review.dimensions}
    if fixed - claimed or not claimed <= fixed | regressed:
        raise TaskDesignError("task design delta fixed dimensions do not match consensus progress")
    if claimed_regressed and delta.audit_schema_version < 2:
        raise TaskDesignError(
            "task design delta schema_version 1 cannot authorize a repair frontier"
        )
    delta_rows = {row.dimension: row for row in delta.regressions}
    for identifier in claimed_regressed & regressed:
        if delta_rows[identifier].repair_paths != candidate_rows[identifier].repair_paths:
            raise TaskDesignError("task design delta repair paths do not match candidate consensus")


def check_delta(
    delta: DeltaAudit,
    *,
    parent: TaskDesign,
    candidate: TaskDesign,
    parent_review: DesignReview,
    candidate_review: DesignReview,
) -> None:
    """Run upstream's ``validate-delta``: the audit's contract, then its consistency.

    Raises:
        TaskDesignError: When the audit is invalid.
    """
    validate_delta(delta, parent=parent, candidate=candidate)
    validate_review(candidate_review, candidate)
    validate_review(parent_review, parent)
    progress = compare_reviews(
        parent_review, candidate_review, parent_design=parent, candidate_design=candidate
    )
    confirm_delta(delta, progress, candidate_review)


# --- latent findings ----------------------------------------------------------------------------


def _unchanged_paths(paths: list[str], changed_paths: tuple[str, ...]) -> list[str]:
    return [
        path
        for path in paths
        if not any(paths_overlap(path, changed_path) for changed_path in changed_paths)
    ]


def rebind_latent_failures(
    *,
    parent: TaskDesign,
    candidate: TaskDesign,
    parent_review: DesignReview,
    candidate_review: DesignReview,
    delta: DeltaAudit,
    consensus_regressions: Sequence[str],
) -> DesignReview | None:
    """Rebind unchanged full-review failures to the retained parent (upstream ``latent_review``).

    Only candidate findings that concern paths the repair did not change, in dimensions that did
    not fail in the parent and did not regress, gain repair authority over the parent. Returns
    ``None`` when no finding qualifies or the synthesized review would exceed the size bound.
    """
    changed_paths = task_design_changed_paths(parent, candidate)
    direct_regressions = set(delta.regressed_dimensions) | set(consensus_regressions)
    parent_rows = {row.id: row.model_dump(mode="json") for row in parent_review.dimensions}
    latent_rows: dict[str, dict[str, Any]] = {}
    for source_row in candidate_review.dimensions:
        source = source_row.model_dump(mode="json")
        identifier = source["id"]
        if (
            source["verdict"] != "fail"
            or parent_rows[identifier]["verdict"] == "fail"
            or identifier in direct_regressions
        ):
            continue
        repair_paths = _unchanged_paths(source["repair_paths"], changed_paths)
        design_paths = _unchanged_paths(source["design_paths"], changed_paths)
        if (
            repair_paths != source["repair_paths"]
            or design_paths != source["design_paths"]
            or not repair_paths
            or not design_paths
        ):
            continue
        latent = deepcopy(source)
        latent["design_paths"] = design_paths
        latent_rows[identifier] = latent
    if not latent_rows:
        return None
    dimensions = [
        deepcopy(latent_rows[identifier])
        if identifier in latent_rows
        else deepcopy(parent_rows[identifier])
        for identifier in DESIGN_DIMENSIONS
    ]
    payload = {
        "schema_version": 3,
        "verdict": "fail",
        "summary": "Retained-parent repair authority with unchanged full-audit failures.",
        "design_fingerprint": parent.fingerprint(),
        "dimensions": dimensions,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    if len(encoded) + 1 > MAX_DESIGN_ARTIFACT_BYTES:
        return None
    return DesignReview.model_validate(payload)


# --- repair feedback ----------------------------------------------------------------------------


def failed_dimension_ids(review: DesignReview) -> list[str]:
    """Return the failed dimension ids in the order of the review's rows."""
    return [row.id for row in review.dimensions if row.verdict == "fail"]


def target_dimensions_text(review: DesignReview) -> str:
    """Return the target dimension ids of a focused review as upstream renders them (JSON)."""
    return json.dumps(failed_dimension_ids(review))


def validation_failure_payload(attempt: int, diagnostic: str) -> dict[str, Any]:
    """Return the feedback of a structural repair (``design_validation_failure_payload``).

    The diagnostic is cut at 8192 characters (``" [truncated]"`` is appended) and stripped.
    """
    text = diagnostic[:8193]
    if len(text) > 8192:
        text = text[:8192] + " [truncated]"
    return {"source": "schema_validation", "attempt": attempt, "diagnostic": text.strip()}


def _compact_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return value if len(value) <= limit else value[:limit] + " [truncated]"


def _compact_strings(value: object, *, count: int = 8, limit: int = 512) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_compact_text(item, limit) for item in value[:count] if isinstance(item, str)]


def _compact_finding(row: object) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    return {
        "id": _compact_text(row.get("id"), 128),
        "verdict": _compact_text(row.get("verdict"), 32),
        "design_paths": _compact_strings(row.get("design_paths"), limit=256),
        "evidence": _compact_strings(row.get("evidence"), count=2, limit=512),
        "remediation": _compact_text(row.get("remediation"), 1024),
    }


def _compact_regression(row: object) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    return {
        "dimension": _compact_text(row.get("dimension"), 128),
        "design_paths": _compact_strings(row.get("design_paths"), limit=256),
        "evidence": _compact_strings(row.get("evidence"), count=2, limit=512),
        "remediation": _compact_text(row.get("remediation"), 1024),
    }


def review_failure_payload(
    review: DesignReview,
    attempt: int,
    previous_progress: DesignProgress | None = None,
    repair_memory: RepairMemory | None = None,
) -> dict[str, Any]:
    """Return the repair feedback of a failed review (``design_review_failure_payload``).

    The payload carries the failed rows, the evidence of the passing rows as
    ``protected_dimensions``, a compact record of the previous rejected candidate and the repair
    memory. It is rendered into ``{{TASK_DESIGN_REVIEW}}`` as inert JSON data.
    """
    payload = review.payload()
    protected_dimensions = [
        {
            "id": str(row.get("id", ""))[:128],
            "design_paths": [
                value[:256] for value in row.get("design_paths", [])[:8] if isinstance(value, str)
            ],
            "evidence": [
                value[:512] for value in row.get("evidence", [])[:2] if isinstance(value, str)
            ],
        }
        for row in payload["dimensions"]
        if row.get("verdict") == "pass"
    ]
    payload["dimensions"] = [row for row in payload["dimensions"] if row.get("verdict") == "fail"]
    result: dict[str, Any] = {
        "source": "independent_review",
        "attempt": attempt,
        "review": payload,
        "protected_dimensions": protected_dimensions,
    }
    if previous_progress is not None:
        progress = previous_progress.payload()
        previous: dict[str, Any] = {
            "fixed": _compact_strings(progress.get("fixed"), count=16, limit=128),
            "still_failed": _compact_strings(progress.get("still_failed"), count=16, limit=128),
            "regressed": _compact_strings(progress.get("regressed"), count=16, limit=128),
            "improved": progress.get("improved") is True,
            "diagnostic": _compact_text(progress.get("diagnostic"), 4096),
        }
        findings = [
            compact
            for row in progress.get("candidate_findings", [])[:16]
            if (compact := _compact_finding(row)) is not None
        ]
        if findings:
            previous["candidate_findings"] = findings
        regressions = [
            compact
            for row in progress.get("delta_regressions", [])[:16]
            if (compact := _compact_regression(row)) is not None
        ]
        if regressions:
            previous["delta_regressions"] = regressions
        result["previous_candidate"] = previous
    if repair_memory is not None and isinstance(repair_memory.constraints, list):
        result["repair_memory"] = repair_memory.payload()
    return result


def focus_repair_payload(
    payload: Mapping[str, Any], focus_history: Sequence[str]
) -> tuple[dict[str, Any], str]:
    """Narrow a repair feedback to one failed dimension (``focus_design_repair_payload``).

    The dimension used least often in ``focus_history`` is chosen, in the canonical order of
    :data:`DESIGN_DIMENSIONS`, preferring one that regressed in the previous candidate. The
    previous candidate's fixed and still-failed lists and its findings are cut to that dimension;
    its regressions stay as negative constraints.

    Returns:
        The focused payload (``mode: "focused"``) and the focused dimension id.
    """
    focused_payload = deepcopy(dict(payload))
    focused_payload["mode"] = "focused"
    dimensions = focused_payload["review"]["dimensions"]
    dimensions_by_id = {row.get("id"): row for row in dimensions}
    ordered = [
        dimensions_by_id[identifier]
        for identifier in DESIGN_DIMENSIONS
        if identifier in dimensions_by_id
    ]
    previous = focused_payload.get("previous_candidate")
    regressed_ids: set[Any] = set()
    if isinstance(previous, dict):
        regressed_ids.update(previous.get("regressed", []))
        regressed_ids.update(
            row.get("dimension")
            for row in previous.get("delta_regressions", [])
            if isinstance(row, dict)
        )
    counts = Counter(item for item in focus_history if item in dimensions_by_id)
    least_count = min(counts.get(row["id"], 0) for row in ordered)
    least_used = [row for row in ordered if counts.get(row["id"], 0) == least_count]
    focused = next((row for row in least_used if row.get("id") in regressed_ids), least_used[0])
    focused_payload["review"]["dimensions"] = [focused]
    focused_ids = {focused["id"]}
    if isinstance(previous, dict):
        for key in ("fixed", "still_failed"):
            previous[key] = [item for item in previous.get(key, []) if item in focused_ids]
        previous["candidate_findings"] = [
            row for row in previous.get("candidate_findings", []) if row.get("id") in focused_ids
        ]
        if not previous["candidate_findings"]:
            previous.pop("candidate_findings")
    return focused_payload, str(focused["id"])


def focused_authorization(review: DesignReview, feedback: Mapping[str, Any]) -> DesignReview:
    """Return the review that authorises only the focused dimension.

    Upstream ``write_focused_design_review``.

    Every row other than the focused one passes with no remediation and no repair path, so that a
    patch is confined to the focused dimension's repair scope.

    Raises:
        TaskDesignError: When the feedback does not hold exactly one focused dimension.
    """
    rows = feedback.get("review", {}).get("dimensions", [])
    if feedback.get("mode") != "focused" or len(rows) != 1:
        raise TaskDesignError("focused repair feedback must contain exactly one dimension")
    focused_id = rows[0].get("id")
    payload = review.payload()
    for row in payload["dimensions"]:
        if row["id"] != focused_id:
            row["verdict"] = "pass"
            row["remediation"] = ""
            row["repair_paths"] = []
    payload["verdict"] = "fail"
    return DesignReview.model_validate(payload)


def enrich_progress(
    progress: DesignProgress, candidate_review: DesignReview, delta: DeltaAudit | None = None
) -> DesignProgress:
    """Add the candidate's failed rows and, after a delta audit, the auditor's view.

    With a delta audit the consensus fixed and regressed lists move aside and ``fixed``,
    ``regressed`` and ``improved`` follow the auditor (``enrich_design_progress``).
    """
    enriched = progress.model_copy(deep=True)
    enriched.candidate_findings = [
        row.model_dump(mode="json") for row in candidate_review.dimensions if row.verdict == "fail"
    ]
    if delta is not None:
        payload = delta.payload()
        enriched.consensus_fixed = list(enriched.fixed)
        enriched.consensus_regressed = list(enriched.regressed)
        enriched.delta_regressions = list(payload["regressions"])
        enriched.fixed = list(payload["fixed_dimensions"])
        enriched.regressed = list(dict.fromkeys(row["dimension"] for row in payload["regressions"]))
        enriched.improved = bool(enriched.fixed) and not enriched.regressed
    return enriched


def update_repair_memory(memory: RepairMemory, progress: DesignProgress) -> RepairMemory:
    """Add the regressions of a rejected candidate to the repair memory (at most 32 kept)."""
    regressed = {value for value in progress.regressed if isinstance(value, str)}
    rows: list[dict[str, Any]] = []
    for finding in progress.candidate_findings:
        if isinstance(finding, dict) and finding.get("id") in regressed:
            rows.append(
                {
                    "dimension": finding["id"],
                    "design_paths": finding.get("design_paths", []),
                    "evidence": finding.get("evidence", []),
                    "remediation": finding.get("remediation", ""),
                }
            )
    for finding in progress.delta_regressions:
        if isinstance(finding, dict) and finding.get("dimension") in regressed:
            rows.append(finding)

    def compact(row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "dimension": str(row.get("dimension", ""))[:128],
            "design_paths": [
                value[:256] for value in row.get("design_paths", [])[:8] if isinstance(value, str)
            ],
            "evidence": [
                value[:512] for value in row.get("evidence", [])[:2] if isinstance(value, str)
            ],
            "remediation": str(row.get("remediation", ""))[:1024],
        }

    constraints = [row for row in memory.constraints if isinstance(row, dict)]
    for row in rows:
        item = compact(row)
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        constraints = [
            existing
            for existing in constraints
            if json.dumps(existing, ensure_ascii=False, sort_keys=True) != key
        ]
        constraints.append(item)
    return RepairMemory(constraints=constraints[-32:])


# --- decisions on a candidate -------------------------------------------------------------------


def cleared_parent_failures(progress: DesignProgress) -> bool:
    """Whether a candidate fixed something and no parent failure remains."""
    return bool(progress.fixed) and not progress.still_failed


def regressions_were_already_failed(progress: DesignProgress) -> bool:
    """Whether every regression lies in a dimension that failed in the parent as well."""
    regressed = set(progress.regressed)
    return bool(regressed) and regressed <= set(progress.still_failed)


def has_only_indirect_regressions(
    progress: DesignProgress,
    *,
    parent: TaskDesign,
    candidate: TaskDesign,
    parent_review: DesignReview,
    candidate_review: DesignReview,
    delta: DeltaAudit,
) -> bool:
    """Whether the auditor confirms only regressions that the repair did not directly cause.

    True when some fix stands, every confirmed regression was a consensus regression, and every
    confirmed regression's repair paths lie outside the paths the candidate changed (upstream
    ``design_progress_has_only_indirect_regressions``): the regression is then a quarantined
    frontier finding rather than damage done by the patch.
    """
    consensus_regressed = set(progress.consensus_regressed or [])
    if not progress.fixed or not consensus_regressed:
        return False
    changed = task_design_changed_paths(parent, candidate)
    parent_failed = [row for row in parent_review.dimensions if row.verdict == "fail"]
    candidate_failed = [row for row in candidate_review.dimensions if row.verdict == "fail"]
    rows = {row.dimension: row for row in delta.regressions}
    confirmed = set(rows)
    if not confirmed or not confirmed <= consensus_regressed:
        return False
    if len(candidate_failed) > len(parent_failed) and confirmed == consensus_regressed:
        return False
    for identifier in confirmed:
        repair_paths = rows[identifier].repair_paths or []
        if not repair_paths or any(
            paths_overlap(repair_path, changed_path)
            for repair_path in repair_paths
            for changed_path in changed
        ):
            return False
    return True


__all__ = [
    "TaskDesignError",
    "apply_design_patch",
    "check_delta",
    "cleared_parent_failures",
    "compare_reviews",
    "confirm_delta",
    "decode_json_pointer",
    "design_from_payload",
    "enrich_progress",
    "failed_dimension_ids",
    "focus_repair_payload",
    "focused_authorization",
    "has_only_indirect_regressions",
    "paths_overlap",
    "rebind_latent_failures",
    "regressions_were_already_failed",
    "resolve_json_pointer",
    "review_failure_payload",
    "target_dimensions_text",
    "task_design_changed_paths",
    "update_repair_memory",
    "validate_delta",
    "validate_design",
    "validate_design_payload",
    "validate_review",
    "validation_failure_payload",
]
