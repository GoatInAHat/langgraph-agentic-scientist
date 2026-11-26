"""Shared state records, the base model of every stored record, and reducers.

Every pydantic model the package stores in graph state, passes through ``Send`` or returns from a
phase derives from :class:`StateModel`. That gives three guarantees:

* unknown fields are refused (``extra="forbid"``), as upstream's JSON validators refuse them;
* every record carries a class-level ``schema_version``, so a later release can refuse or migrate
  a checkpoint written by an older one
  (:func:`~langgraph_agentic_scientist.core.runconfig.record_run_config`);
* :func:`~langgraph_agentic_scientist.serde.serde_allowlist` can find every stored type by
  walking the subclasses of :class:`StateModel`, so a strict msgpack serializer accepts them.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from .contract import TaskContract

STATE_SCHEMA_VERSION = 1
"""Schema version of every state record of this release."""


class StateModel(BaseModel):
    """Base class of every record stored in graph state.

    Subclasses refuse unknown fields and carry the class-level ``schema_version`` of the release
    that defined them.
    """

    model_config = ConfigDict(extra="forbid", frozen=False, populate_by_name=True)

    schema_version: ClassVar[int] = STATE_SCHEMA_VERSION


def merge_dicts(left: Mapping[str, Any] | None, right: Mapping[str, Any] | None) -> dict[str, Any]:
    """Reducer for keyed maps: keys of ``right`` replace the same keys of ``left``.

    Used for ``phase_status``, ``hook_outputs`` and other keyed status maps, so that parallel
    branches can each write their own keys in one superstep.
    """
    merged: dict[str, Any] = dict(left or {})
    merged.update(right or {})
    return merged


append = operator.add
"""Reducer for append-only histories (``Annotated[list[X], append]``)."""


def utc_now() -> str:
    """Return the current UTC time as an ISO 8601 string, as upstream's ``utc_now``."""
    return datetime.now(UTC).isoformat()


# --- programs, tasks, rubrics -------------------------------------------------------------


class ProgramRef(StateModel):
    """A candidate program directory in the workspace.

    Attributes:
        path: Workspace-relative POSIX path of the ``program_<idx>`` directory.
        entrypoint: Entrypoint relative to ``path`` (default ``src/program.py``).
        digest: ``sha256:<hex>`` tree digest when known.
    """

    path: str
    entrypoint: str = "src/program.py"
    digest: str | None = None


class RubricSpec(StateModel):
    """One target rubric of an idea (upstream ``normalize_target_rubrics``).

    ``verification`` is upstream's optional static check for implementation rubrics:
    ``{"type": "source_contains", "patterns": [...]}``. ``metric``, ``operator`` and
    ``threshold`` are the structured comparison that upstream's harness incumbent gate carries
    (``combined_score > <incumbent>``); agent-authored rubrics have none, as upstream's
    normalisation keeps only ``id``, ``description``, ``type`` and ``verification``.
    """

    id: str
    description: str
    type: str = "implementation"
    verification: dict[str, Any] | None = None
    metric: str | None = None
    operator: str | None = None
    threshold: float | None = None


class RubricResult(StateModel):
    """The verdict on one rubric inside ``rubric_status.json``."""

    id: str
    passed: bool = Field(alias="pass")
    evidence: str = ""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class RubricStatus(StateModel):
    """The content of a ``rubric_status.json`` file (upstream field names)."""

    variant_id: str | None = None
    attempt_id: str
    idea_id: str
    overall_pass: bool
    rubrics: list[RubricResult]
    metrics: dict[str, Any] = Field(default_factory=dict)
    next_recommendation: str = ""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class TaskRef(StateModel):
    """The run-local task snapshot every phase reads.

    Attributes:
        root: Workspace-relative path of the task snapshot (upstream ``<run>/task``).
        task_md_sha256: Digest of ``task.md``.
        goal: Body of the ``Goal`` section of ``task.md``.
        rubrics: Body of the ``Rubrics`` section of ``task.md``.
        contract: The parsed ``task_contract.json`` when the task has one.
        seed_program: The baseline program (``program_0``).
        harness_evaluated: True when attempts are evaluated by the host (forced by a contract).
    """

    root: str
    task_md_sha256: str | None = None
    goal: str = ""
    rubrics: str = ""
    contract: TaskContract | None = None
    seed_program: ProgramRef | None = None
    harness_evaluated: bool = False


class IdeaBundle(StateModel):
    """A validated idea bundle (upstream ``normalize_idea_bundle`` output fields)."""

    idea_id: str
    title: str
    hypothesis: str
    implementation_plan: list[str]
    target_rubrics: list[RubricSpec]
    notes_for_agent: str = ""


class BestManifest(StateModel):
    """The content of ``best_program_manifest.json`` (upstream ``load_or_init_best_manifest``)."""

    program_dir: str
    entrypoint_path: str = "src/program.py"
    combined_score: float | None = None
    correct: bool | None = None
    source_attempt_id: str = "bootstrap"
    updated_at: str = Field(default_factory=utc_now)

    model_config = ConfigDict(extra="allow", populate_by_name=True)


EvaluationPlane = Literal[
    "search", "variant", "bootstrap", "smoke", "production_entry", "qualification"
]
"""The evaluation planes of :class:`~langgraph_agentic_scientist.evaluation.EvaluationRequest`."""


class EvaluationRecord(StateModel):
    """A host record of one candidate evaluation.

    Attributes:
        evaluation_id: The deterministic id the evaluation was requested under.
        program: The evaluated program.
        plane: The evaluation plane.
        status: The evaluator's status (``completed``, ``failed``, ``timeout``, ...).
        metrics: ``metrics.json`` of a completed evaluation.
        correct: ``correct.json`` of a completed evaluation.
        evidence_available: Whether a validated ``evaluation_evidence.json`` was produced.
        failure_category: Category of a failed evaluation, for example
            ``evaluator_output_contract_invalid``.
        summary: Bounded diagnostic text.
    """

    evaluation_id: str
    program: ProgramRef
    plane: EvaluationPlane = "search"
    status: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    correct: bool | None = None
    evidence_available: bool = False
    failure_category: str | None = None
    summary: str = ""
    attempt_id: str | None = None
    idea_id: str | None = None


PhaseState = Literal["pending", "completed", "degraded", "failed", "skipped"]


class PhaseStatus(StateModel):
    """Status of one pipeline phase with its reason."""

    status: PhaseState = "pending"
    reason: str | None = None


class Event(StateModel):
    """A structured event of a run (upstream event log, kept as state)."""

    kind: str
    phase: str | None = None
    step: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    at: str = Field(default_factory=utc_now)


FailureScope = Literal["task", "infrastructure", "method"]


class Failure(StateModel):
    """A classified failure.

    Attributes:
        category: Upstream's failure category, for example ``campaign_failed`` or
            ``evaluator_output_contract_invalid``.
        scope: Whether the task, the infrastructure or the method failed.
        detail: Bounded human-readable detail.
    """

    category: str
    scope: FailureScope = "method"
    detail: str = ""


class Question(StateModel):
    """A question the graph puts to a person through ``interrupt``.

    Attributes:
        kind: Stable identifier of the question, for example ``prune_confirmation``.
        text: The text shown to the person.
        options: The accepted answers.
        default: The answer a non-interactive policy gives (upstream's non-TTY behaviour).
        context: Extra JSON data for the person or the policy.
    """

    kind: str
    text: str
    options: list[str] = Field(default_factory=lambda: ["proceed", "abort"])
    default: str = "proceed"
    context: dict[str, Any] = Field(default_factory=dict)


class Answer(StateModel):
    """The answer to a :class:`Question`."""

    choice: str
    note: str = ""


class RunConfigRecord(StateModel):
    """The effective method configuration and prompt texts, recorded by the first node of a run.

    Attributes:
        method_config: The method part of :class:`~langgraph_agentic_scientist.ScientistConfig`
            as JSON (runtime-only fields excluded).
        config_sha256: SHA-256 of ``method_config`` in canonical JSON.
        prompts_sha256: SHA-256 of every prompt and skill text the run can use, after overrides.
        prompt_digests: Per-key SHA-256 of those texts, to name a differing prompt.
        package_version: Version of the package that recorded the configuration.
        state_schema_version: :data:`STATE_SCHEMA_VERSION` of that release.
    """

    method_config: dict[str, Any]
    config_sha256: str
    prompts_sha256: str
    prompt_digests: dict[str, str] = Field(default_factory=dict)
    package_version: str
    state_schema_version: int = STATE_SCHEMA_VERSION


HookOutputs = dict[str, Any]
"""Type of the ``hook_outputs`` state key: hook results keyed by step name."""


__all__ = [
    "STATE_SCHEMA_VERSION",
    "Answer",
    "BestManifest",
    "EvaluationPlane",
    "EvaluationRecord",
    "Event",
    "Failure",
    "FailureScope",
    "HookOutputs",
    "IdeaBundle",
    "PhaseState",
    "PhaseStatus",
    "ProgramRef",
    "Question",
    "RubricResult",
    "RubricSpec",
    "RubricStatus",
    "RunConfigRecord",
    "StateModel",
    "TaskRef",
    "append",
    "merge_dicts",
    "utc_now",
]
