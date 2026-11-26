"""The end-to-end pipeline: every phase of the method as one graph.

:func:`build_pipeline_graph` ports upstream's ``agsci run`` (``src/pipeline/cli.py``,
``phases.py``, ``config.py``, ``task_snapshot.py``): it takes a task directory, copies it once into
a run-local snapshot, and chains the phases in upstream's order::

    task_drafting -> investigation -> (downstream selection) -> ablations -> final_evaluation
    -> posthoc -> paper -> review -> revision

Each phase is a compiled graph, mounted through a wrapper node that maps the pipeline state to the
phase's input and returns the phase's result as a delta (never the child's whole state, so that
reducer keys are not duplicated). A router node precedes every phase. It skips an unselected phase
and applies upstream's degradation rules: optional phases (ablations, an optional final evaluation,
post-hoc analysis, review, revision) that fail are recorded as ``degraded`` and the run goes on; a
phase that cannot be skipped (drafting, investigation, paper, a final evaluation the task's
contract requires) fails the run when ``fail_fast`` is set, and its dependants are skipped
otherwise; the terminal host failures of upstream (``evaluator_output_contract_invalid`` and
``agent_log_persistence_failed``) and a drafted task that needs a final evaluation that is not
selected stop the run whatever ``fail_fast`` says.

The pipeline keeps no persistence of its own. The caller passes the checkpointer (and the store)
to :func:`build_pipeline_graph`; a run resumes from its checkpoint with the same thread id and a
context that carries the same method configuration and prompts (anything else raises
:class:`~langgraph_agentic_scientist.ConfigurationMismatch`).

Phase graphs come from ``GraphOverrides.phases`` (keyed by phase name, plus
``"downstream_selection"``) or from the phase package's factory. A phase graph is invoked with the
keys of the mapping below that its input schema declares (a phase graph that declares fewer keys
simply receives fewer), and returns its result under ``result`` or under the pipeline's key for
that phase (``drafting``, ``investigation``, ``downstream``, ``variants``, ``final``, ``posthoc``,
``paper``, ``review``, ``revision``):

========================  ================================================================
phase                     keys given
========================  ================================================================
``task_drafting``         ``task``, ``capacity``, ``run_id``, ``run_config``
``investigation``         ``task``, ``run_id``, ``run_config``
``downstream_selection``  the above and ``investigation``
``ablations``             the above and ``downstream``
``final_evaluation``      the above and ``downstream``
``posthoc``               the above and ``variants``
``paper``                 the above, ``mode``, ``final``, ``posthoc``, ``previous_paper_dir``,
                          ``edit_options``, ``paper_dir`` and ``manifest_dir`` (both ``paper``)
``review``                ``task``, ``run_id``, ``run_config``, ``paper``
``revision``              the above, ``review`` and the paper's grounding (``investigation``,
                          ``downstream``, ``variants``, ``final``, ``posthoc``, ``edit_options``)
========================  ================================================================
"""

from __future__ import annotations

import hashlib
import importlib
import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.types import Checkpointer
from pydantic import BaseModel

from .config import ALL_PHASES, Phase, ScientistConfig
from .context import GraphOverrides, ScientistContext
from .core.contract import TaskContract, load_task_contract
from .core.errors import ScientistError
from .core.render import task_sections
from .core.results import (
    DownstreamSelection,
    DraftingResult,
    EditOptions,
    InvestigationResult,
    PaperResult,
    PosthocResult,
    ReviewResult,
    RevisionResult,
    VariantsResult,
)
from .core.runconfig import method_config, record_run_config, verify_run_config
from .core.state import (
    Event,
    Failure,
    PhaseState,
    PhaseStatus,
    ProgramRef,
    RunConfigRecord,
    TaskRef,
    merge_dicts,
)
from .core.steps import StepGraph
from .evaluation.program_spec import resolve_entrypoint
from .evaluation.protocol import FinalEvaluationResult
from .workspace.digest import tree_digest
from .workspace.paths import (
    TASK_COPY_IGNORE_FILENAME,
    parse_ignore_patterns,
    safe_relpath,
    task_copy_path_is_ignored,
)
from .workspace.protocol import FileEntry, Workspace

TASK_SNAPSHOT_DIR = "task"
"""Workspace path of the run-local task snapshot (upstream ``<run>/task``)."""

PAPER_DIR = "paper"
"""Workspace path of the run's paper (upstream ``paths.paper_dir``)."""

TASK_SNAPSHOT_METADATA = ".agentic_task_snapshot.json"
"""Name of the snapshot's metadata file (upstream's name)."""

TASK_COPY_IGNORE_MAX_BYTES = 64 * 1024
TERMINAL_FAILURES = frozenset({"evaluator_output_contract_invalid", "agent_log_persistence_failed"})
"""Failure categories that stop a run whatever ``fail_fast`` says (upstream's exit codes 86, 87)."""

_DOWNSTREAM_PHASES: frozenset[str] = frozenset(
    {"ablations", "final_evaluation", "posthoc", "paper", "review", "revision"}
)
_DEPENDENCY: Mapping[str, str] = {
    "investigation": "task_drafting",
    "ablations": "investigation",
    "final_evaluation": "investigation",
    "posthoc": "investigation",
    "paper": "investigation",
    "review": "paper",
    "revision": "review",
}
_REQUIRED_FINAL_FAILURE = "required_final_evaluation_failed"

PipelineStatus = Literal["completed", "completed_degraded", "failed"]
Route = Literal["run", "skip", "stop"]


class PipelineConfigurationError(ScientistError, ValueError):
    """The phase selection, the task or the context cannot start a run.

    The counterpart of upstream's ``Config error`` messages (``validate_pipeline_config``): it is
    raised before any phase runs.
    """


class TaskSnapshotError(ScientistError):
    """The run-local task snapshot is unsafe, incomplete or does not match its recorded digest."""


# --- state ----------------------------------------------------------------------------------


class PipelineInput(TypedDict):
    """What :func:`build_pipeline_graph`'s graph is invoked with.

    ``task_dir`` is a path inside the context's workspace (the source of the run-local snapshot;
    it is not read again when a snapshot already exists). ``phases`` overrides the context's
    ``config.phases``. The optional results of earlier phases let a run start later, as upstream's
    ``execution.phases=paper paths.run_dir=...`` does: a selected phase whose dependency is not
    selected needs that dependency's result here.
    """

    task_dir: str
    run_id: NotRequired[str]
    phases: NotRequired[list[Phase]]
    previous_paper_dir: NotRequired[str]
    edit_options: NotRequired[EditOptions]
    drafting: NotRequired[DraftingResult]
    investigation: NotRequired[InvestigationResult]
    downstream: NotRequired[DownstreamSelection]
    variants: NotRequired[VariantsResult]
    final: NotRequired[FinalEvaluationResult]
    posthoc: NotRequired[PosthocResult]
    paper: NotRequired[PaperResult]
    review: NotRequired[ReviewResult]
    revision: NotRequired[RevisionResult]


class PipelineState(TypedDict, total=False):
    """The pipeline's state: the input, one result per phase and the bookkeeping of the run."""

    task_dir: str
    run_id: str
    run_config: RunConfigRecord | None
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    phases: list[Phase]
    previous_paper_dir: str
    edit_options: EditOptions
    task: TaskRef | None
    phase_status: Annotated[dict[str, PhaseStatus], merge_dicts]
    drafting: DraftingResult | None
    investigation: InvestigationResult | None
    downstream: DownstreamSelection | None
    variants: VariantsResult | None
    final: FinalEvaluationResult | None
    posthoc: PosthocResult | None
    paper: PaperResult | None
    review: ReviewResult | None
    revision: RevisionResult | None
    events: Annotated[list[Event], operator.add]
    failures: Annotated[list[Failure], operator.add]
    route: Route
    stop_reason: str | None
    status: PipelineStatus
    outcome: str | None


class PipelineOutput(TypedDict, total=False):
    """What a run returns.

    ``status`` is ``completed``, ``completed_degraded`` (an optional phase degraded) or
    ``failed``; ``outcome`` is ``completed_no_winner`` or ``not_qualified`` when the run ended
    that way, as upstream's ``pipeline_end`` event says.
    """

    run_id: str
    status: PipelineStatus
    outcome: str | None
    stop_reason: str | None
    phases: list[Phase]
    phase_status: dict[str, PhaseStatus]
    failures: list[Failure]
    task: TaskRef | None
    drafting: DraftingResult | None
    investigation: InvestigationResult | None
    downstream: DownstreamSelection | None
    variants: VariantsResult | None
    final: FinalEvaluationResult | None
    posthoc: PosthocResult | None
    paper: PaperResult | None
    review: ReviewResult | None
    revision: RevisionResult | None
    hook_outputs: dict[str, Any]
    events: list[Event]


# --- the table of phases --------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseSpec:
    """One mounted phase: its name, where its result lives and where its factory is."""

    name: str
    state_key: str
    result_type: type[BaseModel]
    module: str
    factory: str


PHASE_SPECS: tuple[PhaseSpec, ...] = (
    PhaseSpec("task_drafting", "drafting", DraftingResult, "drafting", "build_task_drafting_graph"),
    PhaseSpec(
        "investigation",
        "investigation",
        InvestigationResult,
        "investigation",
        "build_investigation_graph",
    ),
    PhaseSpec(
        "downstream_selection",
        "downstream",
        DownstreamSelection,
        "investigation",
        "build_downstream_selection_graph",
    ),
    PhaseSpec("ablations", "variants", VariantsResult, "variants", "build_variants_graph"),
    PhaseSpec(
        "final_evaluation",
        "final",
        FinalEvaluationResult,
        "final_evaluation",
        "build_final_evaluation_graph",
    ),
    PhaseSpec("posthoc", "posthoc", PosthocResult, "posthoc", "build_posthoc_graph"),
    PhaseSpec("paper", "paper", PaperResult, "paper", "build_paper_graph"),
    PhaseSpec("review", "review", ReviewResult, "review", "build_review_graph"),
    PhaseSpec("revision", "revision", RevisionResult, "revision", "build_revision_graph"),
)
"""The phases in the order the pipeline runs them (the downstream selection follows the search)."""


def _load_factory(spec: PhaseSpec) -> Any:
    module = importlib.import_module(f"{__package__}.{spec.module}")
    return getattr(module, spec.factory)


def _phase_graph(
    spec: PhaseSpec, overrides: GraphOverrides
) -> CompiledStateGraph[Any, Any, Any, Any]:
    replacement = overrides.phases.get(spec.name)
    if replacement is not None:
        return replacement
    graph: CompiledStateGraph[Any, Any, Any, Any] = _load_factory(spec)(
        overrides=overrides, name=spec.name
    )
    return graph


# --- small helpers --------------------------------------------------------------------------


def _coerce[M: BaseModel](model: type[M], value: Any) -> M | None:
    """Return ``value`` as a ``model`` (a state read back from JSON holds mappings)."""
    if value is None or isinstance(value, model):
        return value
    return model.model_validate(value)


def _task(state: Mapping[str, Any]) -> TaskRef | None:
    return _coerce(TaskRef, state.get("task"))


def _status_map(state: Mapping[str, Any]) -> dict[str, PhaseStatus]:
    return {
        key: PhaseStatus.model_validate(value) if isinstance(value, dict) else value
        for key, value in (state.get("phase_status") or {}).items()
    }


def _succeeded(state: Mapping[str, Any], phase: str) -> bool:
    status = _status_map(state).get(phase)
    return status is not None and status.status == "completed"


def _selected(state: Mapping[str, Any]) -> list[str]:
    return list(state.get("phases") or [])


def _verify(state: Mapping[str, Any], context: ScientistContext) -> None:
    """Refuse a context whose method configuration or prompts differ from the recorded ones."""
    recorded = _coerce(RunConfigRecord, state.get("run_config"))
    if recorded is not None:
        verify_run_config(recorded, context)


def _event(kind: str, phase: str | None = None, **data: Any) -> Event:
    return Event(kind=kind, phase=phase, data=data)


def normalize_phases(selected: Sequence[str]) -> list[Phase]:
    """Return the selected phases once each, in upstream's order.

    Raises:
        PipelineConfigurationError: When the selection is empty or names an unknown phase.
    """
    names = set(selected)
    if not names:
        raise PipelineConfigurationError("no phases selected")
    unknown = names - set(ALL_PHASES)
    if unknown:
        raise PipelineConfigurationError(
            f"invalid pipeline phases: {sorted(unknown)}; valid: {', '.join(ALL_PHASES)}"
        )
    return [phase for phase in ALL_PHASES if phase in names]


def requires_final_evaluation(task: TaskRef | None) -> bool:
    """Return whether the task's contract requires a final evaluation before a paper."""
    contract = task.contract if task is not None else None
    return contract is not None and contract.publication.requires_final_evaluation


def _final(state: Mapping[str, Any]) -> FinalEvaluationResult | None:
    return _coerce(FinalEvaluationResult, state.get("final"))


def _final_ok(state: Mapping[str, Any]) -> bool:
    if "final_evaluation" in _selected(state):
        return _succeeded(state, "final_evaluation")
    return _final(state) is not None


def _fallback_selected(state: Mapping[str, Any]) -> bool:
    selection = _coerce(DownstreamSelection, state.get("downstream"))
    return selection is not None and selection.status == "selected"


def _no_winner(state: Mapping[str, Any]) -> bool:
    investigation = _coerce(InvestigationResult, state.get("investigation"))
    return investigation is not None and investigation.status == "completed_no_winner"


def _posthoc_ready(result: PosthocResult | None) -> bool:
    return (
        result is not None
        and result.status == "completed"
        and any(analysis.status == "passed" for analysis in result.analyses)
    )


# --- the task snapshot ----------------------------------------------------------------------


def _unsafe_symlink(entry: FileEntry, targets: Mapping[str, str]) -> str | None:
    """Return why a symlink is unsafe (upstream ``_validate_symlink_targets``), or ``None``."""
    target = entry.target or ""
    if target.startswith("/"):
        return f"contains absolute symlink: {entry.path}"
    resolved = list(PurePosixPath(entry.path).parent.parts)
    pending = target.split("/")
    expansions = 0
    while pending:
        component = pending.pop(0)
        if component in {"", "."}:
            continue
        if component == "..":
            if not resolved:
                return f"contains escaping symlink: {entry.path}"
            resolved.pop()
            continue
        candidate = "/".join((*resolved, component))
        nested = targets.get(candidate)
        if nested is None:
            resolved.append(component)
            continue
        expansions += 1
        if expansions > 40:
            return f"contains invalid symlink: {entry.path}"
        if nested.startswith("/"):
            return f"contains absolute symlink: {candidate}"
        pending = [*nested.split("/"), *pending]
    return None


def _validate_source_tree(entries: Sequence[FileEntry], patterns: Sequence[str]) -> None:
    ignored_dirs = [
        entry.path
        for entry in entries
        if entry.kind == "dir"
        and task_copy_path_is_ignored(entry.path, is_dir=True, patterns=patterns)
    ]
    kept = [
        entry
        for entry in entries
        if not any(entry.path == base or entry.path.startswith(f"{base}/") for base in ignored_dirs)
        and not task_copy_path_is_ignored(entry.path, is_dir=entry.kind == "dir", patterns=patterns)
    ]
    targets = {entry.path: entry.target or "" for entry in kept if entry.kind == "symlink"}
    for entry in kept:
        if entry.kind == "other":
            raise TaskSnapshotError(f"task source contains unsupported entry: {entry.path}")
        if entry.kind == "symlink":
            problem = _unsafe_symlink(entry, targets)
            if problem is not None:
                raise TaskSnapshotError(f"task source {problem}")


async def _snapshot_digest(workspace: Workspace, root: str) -> str:
    """Return the tree digest of the snapshot without its metadata file, as bare hex."""
    entries = [
        entry for entry in await workspace.list_tree(root) if entry.path != TASK_SNAPSHOT_METADATA
    ]

    async def read(relative: str) -> bytes:
        return await workspace.read_bytes(f"{root}/{relative}")

    digest = await tree_digest(entries, read)
    return digest.removeprefix("sha256:")


async def _write_snapshot_metadata(workspace: Workspace, root: str, task_name: str) -> None:
    payload = json.dumps(
        {
            "source_digest": await _snapshot_digest(workspace, root),
            "task_name": task_name,
            "version": 2,
        },
        separators=(",", ":"),
    )
    await workspace.write_bytes(f"{root}/{TASK_SNAPSHOT_METADATA}", (payload + "\n").encode())


async def _read_snapshot_metadata(workspace: Workspace, root: str) -> dict[str, Any] | None:
    path = f"{root}/{TASK_SNAPSHOT_METADATA}"
    entry = await workspace.stat(path)
    if entry is None:
        return None
    if entry.kind != "file" or entry.size > 4096:
        raise TaskSnapshotError("task snapshot metadata is unsafe")
    try:
        metadata = json.loads((await workspace.read_bytes(path)).decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise TaskSnapshotError("task snapshot metadata is invalid") from exc
    if not isinstance(metadata, dict) or metadata.get("version") != 2:
        raise TaskSnapshotError("task snapshot metadata has an unsupported version")
    name = metadata.get("task_name")
    digest = metadata.get("source_digest")
    if not isinstance(name, str) or not name or "/" in name or "\\" in name:
        raise TaskSnapshotError("task snapshot metadata has an invalid task name")
    if not isinstance(digest, str) or len(digest) != 64:
        raise TaskSnapshotError("task snapshot metadata has an invalid source digest")
    return metadata


async def prepare_task_snapshot(workspace: Workspace, task_dir: str) -> None:
    """Copy the task into ``task/`` once; verify an existing snapshot and never refresh it.

    The copy honours ``.agentic_task_ignore``, refuses unsafe symlinks, and ends by writing the
    metadata file (upstream ``.agentic_task_snapshot.json``) that records the tree digest, so a
    snapshot without it is an interrupted copy and is made again. A snapshot with metadata must
    match its recorded digest.

    Raises:
        TaskSnapshotError: When the source is unsafe, overlaps the snapshot, or an existing
            snapshot does not match its digest.
    """
    root = TASK_SNAPSHOT_DIR
    existing = await workspace.stat(root)
    if existing is not None:
        if existing.kind != "dir":
            raise TaskSnapshotError("the run task snapshot must be a real directory")
        metadata = await _read_snapshot_metadata(workspace, root)
        if metadata is not None:
            if await _snapshot_digest(workspace, root) != metadata["source_digest"]:
                raise TaskSnapshotError("task snapshot content does not match its source digest")
            return
        await workspace.remove(root)
    source = safe_relpath(task_dir, allow_root=False)
    if source == root or source.startswith(f"{root}/") or root.startswith(f"{source}/"):
        raise TaskSnapshotError(
            f"task source and run task snapshot must not overlap: {task_dir!r} -> {root!r}"
        )
    source_entry = await workspace.stat(source)
    if source_entry is None or source_entry.kind != "dir":
        raise TaskSnapshotError(f"task source is not a real directory: {task_dir!r}")
    patterns: tuple[str, ...] = ()
    ignore_path = f"{source}/{TASK_COPY_IGNORE_FILENAME}"
    ignore_entry = await workspace.stat(ignore_path)
    if ignore_entry is not None:
        if ignore_entry.kind != "file" or ignore_entry.size > TASK_COPY_IGNORE_MAX_BYTES:
            raise TaskSnapshotError(f"{TASK_COPY_IGNORE_FILENAME} must be a small regular file")
        try:
            patterns = parse_ignore_patterns((await workspace.read_bytes(ignore_path)).decode())
        except UnicodeError as exc:
            raise TaskSnapshotError(f"{TASK_COPY_IGNORE_FILENAME} is invalid") from exc
    _validate_source_tree(await workspace.list_tree(source), patterns)
    await workspace.copy_tree(source, root, ignore=patterns)
    if await workspace.exists(f"{root}/{TASK_SNAPSHOT_METADATA}"):
        raise TaskSnapshotError(f"task source reserves {TASK_SNAPSHOT_METADATA}")
    await _write_snapshot_metadata(workspace, root, PurePosixPath(source).name)


async def reseal_task_snapshot(workspace: Workspace) -> None:
    """Record the digest of the snapshot as it is now (after drafting published a scaffold)."""
    metadata = await _read_snapshot_metadata(workspace, TASK_SNAPSHOT_DIR)
    if metadata is None:
        raise TaskSnapshotError("cannot reseal a task snapshot without metadata")
    await _write_snapshot_metadata(workspace, TASK_SNAPSHOT_DIR, str(metadata["task_name"]))


async def missing_scaffold_files(workspace: Workspace, task_dir: str) -> list[str]:
    """Return the scaffold files a task lacks (upstream ``missing_task_scaffold_files``).

    ``capture.py``, ``evaluate.py`` and ``program_0/program_spec.toml`` are required, and, when
    the task has a contract, ``task_contract.json`` and ``evaluate_final.py``. A declared
    entrypoint that is not a file counts as missing.
    """
    required = ["capture.py", "evaluate.py", "program_0/program_spec.toml"]
    if (await workspace.stat(f"{task_dir}/task_contract.json")) is not None:
        required += ["task_contract.json", "evaluate_final.py"]
    missing: list[str] = []
    for relative in required:
        entry = await workspace.stat(f"{task_dir}/{relative}")
        if entry is None or entry.kind != "file":
            missing.append(relative)
    if not missing:
        try:
            await resolve_entrypoint(workspace, f"{task_dir}/program_0")
        except ScientistError:
            missing.append("program_0/<entrypoint>")
    return missing


async def load_task_ref(workspace: Workspace, config: ScientistConfig) -> TaskRef:
    """Read the run-local snapshot into a :class:`~langgraph_agentic_scientist.core.state.TaskRef`.

    A task with a contract forces harness-evaluated attempts, as upstream's
    ``_enable_contracted_harness_mode`` does.
    """
    root = TASK_SNAPSHOT_DIR
    task_md: str | None = None
    task_md_sha: str | None = None
    entry = await workspace.stat(f"{root}/task.md")
    if entry is not None and entry.kind == "file":
        raw = await workspace.read_bytes(f"{root}/task.md")
        task_md = raw.decode("utf-8", errors="replace")
        task_md_sha = hashlib.sha256(raw).hexdigest()
    goal, rubrics = task_sections(task_md)
    contract: TaskContract | None = None
    if (await workspace.stat(f"{root}/task_contract.json")) is not None:
        contract = await load_task_contract(workspace, root)
    seed: ProgramRef | None = None
    if not await missing_scaffold_files(workspace, root):
        seed = ProgramRef(
            path=f"{root}/program_0",
            entrypoint=await resolve_entrypoint(workspace, f"{root}/program_0"),
            digest=await workspace.digest(f"{root}/program_0"),
        )
    return TaskRef(
        root=root,
        task_md_sha256=task_md_sha,
        goal=goal,
        rubrics=rubrics,
        contract=contract,
        seed_program=seed,
        harness_evaluated=contract is not None or config.investigation.harness_evaluated_attempts,
    )


# --- validating the selection ---------------------------------------------------------------


def validate_selection(
    state: Mapping[str, Any],
    config: ScientistConfig,
    *,
    drafting_pending: bool,
    context: ScientistContext,
    overrides: GraphOverrides,
) -> None:
    """Check a phase selection and the context against upstream's configuration rules.

    Covers upstream's ``validate_pipeline_config`` for what has a counterpart here: a downstream
    phase without its dependency selected needs that dependency's result in the input; ``review``
    and ``revision`` need a paper (or a completed review, for a revision alone); a ``revision``
    with a paper in the same selection needs ``review`` as well; a contracted paper phase needs
    the final evaluation selected (unless ``allow_not_qualified``); the phases that evaluate
    candidates need the context's evaluators; and a contracted task is refused an evaluator that
    does not isolate candidates (upstream ``task_contract_requires_container_isolation``) unless
    ``allow_unisolated_contracted`` is set.

    Raises:
        PipelineConfigurationError: Naming every problem found.
    """
    selected = set(_selected(state))
    task = _task(state)
    errors: list[str] = []

    investigation = _coerce(InvestigationResult, state.get("investigation"))
    if "investigation" not in selected and selected & {
        "ablations",
        "final_evaluation",
        "posthoc",
        "paper",
    }:
        if investigation is None:
            errors.append(
                "running ablations, final_evaluation, posthoc or paper without investigation "
                "needs the investigation result in the input"
            )
        elif investigation.status == "failed":
            errors.append("the investigation result given in the input did not complete")
    paper = _coerce(PaperResult, state.get("paper"))
    review = _coerce(ReviewResult, state.get("review"))
    if "review" in selected and "paper" not in selected and paper is None:
        errors.append("running review without paper needs the paper result in the input")
    if "revision" in selected:
        if "paper" not in selected and paper is None:
            errors.append("running revision without paper needs the paper result in the input")
        if "review" not in selected:
            if "paper" in selected:
                errors.append(
                    "running paper and revision requires review in the same phase selection"
                )
            elif review is None or review.status != "completed":
                errors.append(
                    "running revision without review needs a completed review in the input"
                )
    if (
        "paper" in selected
        and "final_evaluation" not in selected
        and config.final_outcome_policy != "allow_not_qualified"
        and requires_final_evaluation(task)
        and not drafting_pending
    ):
        errors.append("contracted paper phase requires final phase")
    if state.get("previous_paper_dir") and "paper" not in selected:
        errors.append("previous_paper_dir needs the paper phase")

    own = {phase for phase in selected if phase not in overrides.phases}
    if drafting_pending is False:
        own.discard("task_drafting")
    needs_evaluator = [
        phase
        for phase in ("task_drafting", "investigation", "ablations", "posthoc")
        if phase in own
    ]
    if needs_evaluator and context.evaluator is None:
        errors.append(f"the context has no evaluator, which {', '.join(needs_evaluator)} need")
    if "final_evaluation" in own and context.final_evaluator is None:
        errors.append("the context has no final_evaluator, which final_evaluation needs")
    if task is not None and task.contract is not None and not config.allow_unisolated_contracted:
        for phase, evaluator in (
            ("investigation", context.evaluator),
            ("ablations", context.evaluator),
            ("posthoc", context.evaluator),
            ("final_evaluation", context.final_evaluator),
        ):
            if phase in own and evaluator is not None and not evaluator.isolated:
                errors.append(
                    f"the task has a contract and the evaluator of {phase} does not isolate "
                    "candidates; use an isolating evaluator or set allow_unisolated_contracted"
                )
    if errors:
        raise PipelineConfigurationError("; ".join(errors))


# --- routers --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """What a router decides about a phase.

    Attributes:
        route: ``run`` the phase, ``skip`` it, or ``stop`` the run.
        status: The status recorded when the phase does not run.
        reason: The reason recorded with it.
        failure: A failure the decision records (a required final evaluation that failed).
    """

    route: Route
    status: PhaseState = "skipped"
    reason: str = ""
    failure: Failure | None = None


_RUN = Verdict("run")


def decide(state: Mapping[str, Any], config: ScientistConfig, phase: str) -> Verdict:
    """Return what the pipeline does with ``phase`` given the state so far.

    The rules are upstream's, in upstream's order (``cli.py`` ``_run_pipeline``): a stopped run
    stops; an unselected phase is skipped; downstream phases of an investigation without a winner
    are skipped unless a fallback program was frozen; post-hoc analysis is degraded when the
    selected ablations did not succeed; a paper (review, revision) is skipped when the task's
    contract requires a final evaluation that failed, or that did not qualify without a fallback
    program and without ``allow_not_qualified``; a phase whose selected dependency did not
    succeed is skipped.
    """
    if state.get("stop_reason"):
        return Verdict("stop")
    selected = set(_selected(state))
    if phase not in selected:
        return Verdict("skip", "skipped", "not selected")
    if (
        phase in _DOWNSTREAM_PHASES
        and _no_winner(state)
        and (config.no_winner_policy == "skip" or not _fallback_selected(state))
    ):
        return Verdict("skip", "skipped", "investigation completed without a qualifying winner")
    if phase == "posthoc" and "ablations" in selected and not _succeeded(state, "ablations"):
        return Verdict("skip", "degraded", "selected ablations did not succeed")
    task = _task(state)
    required = requires_final_evaluation(task)
    final = _final(state)
    not_qualified = required and final is not None and final.outcome == "not_qualified"
    allow = config.final_outcome_policy == "allow_not_qualified"
    final_failed = required and not _final_ok(state)
    unqualified_blocks = not_qualified and not _fallback_selected(state) and not allow
    failure: Failure | None = None
    if (
        phase in {"paper", "review", "revision"}
        and final_failed
        and not any(item.category == _REQUIRED_FINAL_FAILURE for item in state.get("failures", ()))
    ):
        failure = Failure(
            category=_REQUIRED_FINAL_FAILURE,
            scope="task",
            detail="the task's contract requires a successful final evaluation",
        )
    if phase == "paper" and required and (unqualified_blocks or final_failed):
        if unqualified_blocks:
            return Verdict("skip", "skipped", "required final evaluation did not qualify")
        stop = config.fail_fast
        return Verdict(
            "stop" if stop else "skip",
            "skipped",
            "required final evaluation did not succeed",
            failure or Failure(category=_REQUIRED_FINAL_FAILURE, scope="task"),
        )
    if phase in {"review", "revision"} and (final_failed or unqualified_blocks):
        reason = (
            "required final evaluation did not qualify"
            if unqualified_blocks
            else "required final evaluation failed"
        )
        return Verdict("skip", "skipped", reason, failure)
    dependency = _DEPENDENCY.get(phase)
    if dependency is not None and dependency in selected and not _succeeded(state, dependency):
        return Verdict("skip", "skipped", f"dependency {dependency} failed", failure)
    return Verdict("run", failure=failure)


def _gate_node(spec: PhaseSpec) -> Any:
    async def gate(state: PipelineState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        config = method_config(state)
        name = spec.name
        if name == "task_drafting":
            verdict = await _drafting_verdict(state, config, runtime)
        elif name == "downstream_selection":
            verdict = _downstream_verdict(state, config)
        else:
            verdict = decide(state, config, name)
        update: dict[str, Any] = {"route": verdict.route}
        if verdict.route == "stop" and not verdict.failure:
            return update
        failures = [verdict.failure] if verdict.failure is not None else []
        if verdict.route == "stop":
            update["stop_reason"] = verdict.failure.category if verdict.failure else "stopped"
        if verdict.route != "run":
            update["phase_status"] = {
                name: PhaseStatus(status=verdict.status, reason=verdict.reason or None)
            }
            update["events"] = [
                _event("phase_end", name, status=verdict.status, reason=verdict.reason)
            ]
            if name == "task_drafting" and verdict.status == "completed":
                update["drafting"] = DraftingResult(status="skipped", task=_task(state))
        if failures:
            update["failures"] = failures
        return update

    gate.__name__ = f"route_{spec.name}"
    return gate


async def _drafting_verdict(
    state: PipelineState, config: ScientistConfig, runtime: Runtime[ScientistContext]
) -> Verdict:
    if state.get("stop_reason"):
        return Verdict("stop")
    if "task_drafting" not in _selected(state):
        return Verdict("skip", "skipped", "not selected")
    missing = await missing_scaffold_files(runtime.context.workspace, TASK_SNAPSHOT_DIR)
    if not missing:
        return Verdict("skip", "completed", "task scaffold already present; nothing to draft")
    return _RUN


def _downstream_verdict(state: PipelineState, config: ScientistConfig) -> Verdict:
    if state.get("stop_reason"):
        return Verdict("stop")
    if not _no_winner(state) or not set(_selected(state)) & _DOWNSTREAM_PHASES:
        return Verdict("skip", "skipped", "no fallback program is needed")
    if config.no_winner_policy == "skip":
        return Verdict("skip", "skipped", "no_winner_policy is skip")
    if state.get("downstream") is not None:
        return Verdict("skip", "skipped", "a fallback program was given in the input")
    return _RUN


def _read_route(state: PipelineState) -> Route:
    return state.get("route", "run")


# --- mounting a phase -----------------------------------------------------------------------


def _accepted_keys(graph: Any) -> frozenset[str] | None:
    try:
        builder = graph.builder
        return frozenset(builder.schemas[builder.input_schema])
    except (AttributeError, KeyError, TypeError):
        return None


def _phase_input(spec: PhaseSpec, state: PipelineState, config: ScientistConfig) -> dict[str, Any]:
    name = spec.name
    values: dict[str, Any] = {
        "task": _task(state),
        "run_id": state.get("run_id"),
        "run_config": state.get("run_config"),
    }
    if name == "task_drafting":
        values["capacity"] = {
            "max_idea_cycles": config.investigation.max_idea_cycles,
            "max_program_evals_per_attempt": config.investigation.max_program_evals_per_attempt,
        }
    grounded = {"paper", "revision"}  # the revision runs the paper graph again
    if name in {"downstream_selection", "ablations", "final_evaluation", "posthoc", *grounded}:
        values["investigation"] = state.get("investigation")
    if name in {"ablations", "final_evaluation", "posthoc", *grounded}:
        values["downstream"] = state.get("downstream")
    if name in {"posthoc", *grounded}:
        values["variants"] = state.get("variants")
    if name in grounded:
        values["edit_options"] = state.get("edit_options")
        values["final"] = _final_for_paper(state, config)
        values["posthoc"] = _posthoc_for_paper(state)
    if name == "paper":
        values["mode"] = "edit" if state.get("previous_paper_dir") else "fresh"
        values["previous_paper_dir"] = state.get("previous_paper_dir")
        # upstream's pipeline writes the paper and its latest_manifest.json into one directory
        # (``--output-dir`` and ``--manifest-dir`` are both the run's paper directory), which is
        # where the revision looks for them
        values["paper_dir"] = PAPER_DIR
        values["manifest_dir"] = PAPER_DIR
    if name == "posthoc":
        values["artifact_only"] = True
    if name in {"review", "revision"}:
        paper = _coerce(PaperResult, state.get("paper"))
        values["paper"] = paper
        if paper is not None:
            values["pdf_path"] = paper.pdf_path
            values["paper_dir"] = paper.paper_dir
    if name == "review" and "revision" in _selected(state):
        values["artifact_stem"] = "ai_review.before_revision"
        values["manifest_key"] = "before_revision"
    if name == "revision":
        values["review"] = state.get("review")
    return {key: value for key, value in values.items() if value is not None}


def _final_for_paper(state: PipelineState, config: ScientistConfig) -> FinalEvaluationResult | None:
    """Return the final evaluation the paper may use (upstream ``include_final_evaluation``)."""
    final = _final(state)
    if final is None:
        return None
    if requires_final_evaluation(_task(state)) or final.outcome == "qualified":
        return final
    allowed = _fallback_selected(state) or config.final_outcome_policy == "allow_not_qualified"
    return final if allowed else None


def _posthoc_for_paper(state: PipelineState) -> PosthocResult | None:
    result = _coerce(PosthocResult, state.get("posthoc"))
    if "posthoc" in _selected(state):
        return result if _succeeded(state, "posthoc") else None
    return result if _posthoc_ready(result) else None


def _result_of(spec: PhaseSpec, output: Mapping[str, Any]) -> Any:
    for key in (spec.state_key, "result"):
        value = output.get(key)
        if value is not None:
            return _coerce(spec.result_type, value)
    return None


def _pick(status: Any, mapping: Mapping[str, PhaseState], default: PhaseState) -> PhaseState:
    return mapping.get(str(status), default)


def _assess(
    spec: PhaseSpec, result: Any, output: Mapping[str, Any], *, required_final: bool
) -> tuple[PhaseState, str | None, Failure | None]:
    """Map a phase's result to a status, a reason and, when it failed, a failure.

    Upstream's rules (``cli.py``): drafting, investigation, paper and a required final
    evaluation fail the run; ablations, an optional final evaluation, post-hoc analysis, review
    and revision only degrade it. Post-hoc analysis counts as completed only when a valid
    analysis exists (upstream ``_current_posthoc_artifacts_available``).
    """
    name = spec.name
    status = getattr(result, "status", None)
    reason: str | None = None
    failure: Failure | None = getattr(result, "failure", None)
    if failure is None and isinstance(output.get("failure"), Failure | dict):
        failure = _coerce(Failure, output["failure"])
    state: PhaseState
    if name == "task_drafting":
        state = _pick(status, {"published": "completed", "skipped": "completed"}, "failed")
    elif name == "investigation":
        state = "failed" if result is None or status == "failed" else "completed"
        reason = status if status == "completed_no_winner" else None
    elif name == "downstream_selection":
        state = _pick(status, {"selected": "completed"}, "skipped")
        reason = getattr(result, "reason", None) or None
    elif name == "ablations":
        state = _pick(status, {"completed": "completed", "skipped": "skipped"}, "degraded")
    elif name == "final_evaluation":
        if result is not None:
            state, reason = "completed", result.outcome
        else:
            state = "failed" if required_final else "degraded"
    elif name == "posthoc":
        if status == "skipped":
            state = "skipped"
        elif _posthoc_ready(result):
            state = "completed"
        else:
            state = "degraded"
            reason = "command_failed" if status in {None, "failed"} else "no_valid_analyses"
    elif name == "paper":
        state = _pick(status, {"complete": "completed", "skipped": "skipped"}, "failed")
        if status == "incomplete":
            reason = getattr(result, "incomplete_reason", None)
    elif name == "review":
        state = _pick(status, {"completed": "completed", "skipped": "skipped"}, "degraded")
    else:
        state = _pick(status, {"promoted": "completed", "skipped": "skipped"}, "degraded")
    if state in {"failed", "degraded"}:
        if failure is not None:
            reason = reason or failure.category
        else:
            reason = reason or f"{name} {status or 'returned no result'}"
            if state == "failed":
                failure = Failure(category=f"{name}_failed", detail=reason)
    return state, reason, failure


def _phase_node(spec: PhaseSpec, graph: CompiledStateGraph[Any, Any, Any, Any]) -> Any:
    accepted = _accepted_keys(graph)

    async def run_phase(
        state: PipelineState, config: RunnableConfig, runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        _verify(state, runtime.context)
        method = method_config(state)
        values = _phase_input(spec, state, method)
        if accepted is not None:
            values = {key: value for key, value in values.items() if key in accepted}
        started = _event("phase_start", spec.name)
        if spec.name == "review" and "pdf_path" not in values:
            skipped = "the paper result has no PDF to review"
            return {
                "phase_status": {spec.name: PhaseStatus(status="degraded", reason=skipped)},
                "events": [
                    started,
                    _event("phase_end", spec.name, status="degraded", reason=skipped),
                ],
            }
        output = await graph.ainvoke(values, config, context=runtime.context)
        result = _result_of(spec, output)
        required = requires_final_evaluation(_task(state))
        phase_state, reason, failure = _assess(spec, result, output, required_final=required)
        detail = {"detail": failure.detail} if failure is not None and failure.detail else {}
        update: dict[str, Any] = {
            "phase_status": {spec.name: PhaseStatus(status=phase_state, reason=reason)},
            "events": [
                started,
                _event("phase_end", spec.name, status=phase_state, reason=reason, **detail),
            ],
        }
        if result is not None:
            update[spec.state_key] = result
        hook_outputs = output.get("hook_outputs")
        if hook_outputs:
            update["hook_outputs"] = dict(hook_outputs)
        if failure is not None and phase_state == "failed":
            update["failures"] = [failure]
            terminal = spec.name in {"task_drafting", "investigation"} and (
                failure.category in TERMINAL_FAILURES
            )
            if method.fail_fast or terminal:
                update["stop_reason"] = failure.category
        return update

    run_phase.__name__ = spec.name
    return run_phase


# --- the steps around the phases ------------------------------------------------------------


def _snapshot_node(overrides: GraphOverrides) -> Any:
    async def snapshot_task(
        state: PipelineState, runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        return await _snapshot_task(state, runtime, overrides)

    return snapshot_task


async def _snapshot_task(
    state: PipelineState, runtime: Runtime[ScientistContext], overrides: GraphOverrides
) -> dict[str, Any]:
    """Copy the task, record the run id and phases, and validate the selection."""
    context = runtime.context
    _verify(state, context)
    config = method_config(state)
    run_id = state.get("run_id") or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if "/" in run_id or "\\" in run_id or run_id in {".", ".."} or "\0" in run_id:
        raise PipelineConfigurationError(
            f"run_id must be one non-empty path component, got {run_id!r}"
        )
    phases = normalize_phases(state.get("phases") or config.phases)
    await prepare_task_snapshot(context.workspace, state["task_dir"])
    task = await load_task_ref(context.workspace, config)
    working = {**state, "phases": phases, "task": task}
    drafting_pending = "task_drafting" in phases and bool(
        await missing_scaffold_files(context.workspace, TASK_SNAPSHOT_DIR)
    )
    validate_selection(
        working,
        config,
        drafting_pending=drafting_pending,
        context=context,
        overrides=overrides,
    )
    previous = state.get("previous_paper_dir")
    if previous and not await context.workspace.exists(previous):
        raise PipelineConfigurationError(f"previous_paper_dir does not exist: {previous!r}")
    return {
        "run_id": run_id,
        "phases": phases,
        "task": task,
        "stop_reason": None,
        "events": [
            _event(
                "pipeline_start",
                None,
                run_id=run_id,
                phases=list(phases),
                task_md_sha256=task.task_md_sha256,
                harness_evaluated=task.harness_evaluated,
            )
        ],
    }


def _check_node(overrides: GraphOverrides) -> Any:
    async def check_dependencies(
        state: PipelineState, runtime: Runtime[ScientistContext]
    ) -> dict[str, Any]:
        return await _check_dependencies(state, runtime, overrides)

    return check_dependencies


async def _check_dependencies(
    state: PipelineState, runtime: Runtime[ScientistContext], overrides: GraphOverrides
) -> dict[str, Any]:
    """Refresh the task after drafting and recheck what depends on its contract."""
    _verify(state, runtime.context)
    config = method_config(state)
    drafting = _coerce(DraftingResult, state.get("drafting"))
    update: dict[str, Any] = {}
    task = _task(state)
    if drafting is not None and drafting.status == "published":
        await reseal_task_snapshot(runtime.context.workspace)
        task = await load_task_ref(runtime.context.workspace, config)
        update["task"] = task
        selected = set(_selected(state))
        if (
            "paper" in selected
            and "final_evaluation" not in selected
            and requires_final_evaluation(task)
        ):
            failure = Failure(
                category="drafted_dependencies_invalid",
                scope="task",
                detail=(
                    "the drafted task requires final evaluation before paper; include "
                    "final_evaluation in the phases; the published task is preserved and no "
                    "downstream work was started"
                ),
            )
            update["failures"] = [failure]
            update["stop_reason"] = failure.category
            return update
        validate_selection(
            {**state, "task": task},
            config,
            drafting_pending=False,
            context=runtime.context,
            overrides=overrides,
        )
    return update


async def _finish(state: PipelineState) -> dict[str, Any]:
    """Record the phases a stopped run never reached, and compute the run's status and outcome."""
    statuses = _status_map(state)
    stop = state.get("stop_reason")
    unreached = (
        {
            name: PhaseStatus(status="skipped", reason=f"run stopped: {stop}")
            for name in _selected(state)
            if name not in statuses
        }
        if stop
        else {}
    )
    degraded = sorted(name for name, status in statuses.items() if status.status == "degraded")
    status: PipelineStatus = (
        "failed" if state.get("failures") else "completed_degraded" if degraded else "completed"
    )
    outcome: str | None = None
    if _no_winner(state):
        outcome = "completed_no_winner"
    else:
        final = _final(state)
        if final is not None and final.outcome == "not_qualified":
            outcome = "not_qualified"
    update: dict[str, Any] = {
        "status": status,
        "outcome": outcome,
        "events": [_event("pipeline_end", None, status=status, outcome=outcome, degraded=degraded)],
    }
    if unreached:
        update["phase_status"] = unreached
    return update


def build_pipeline_graph(
    *,
    checkpointer: Checkpointer = None,
    store: BaseStore | None = None,
    overrides: GraphOverrides | None = None,
    name: str = "pipeline",
) -> CompiledStateGraph[PipelineState, ScientistContext, PipelineInput, PipelineOutput]:
    """Build the end-to-end pipeline graph.

    The graph is invoked with a :class:`PipelineInput` and a
    :class:`~langgraph_agentic_scientist.ScientistContext`::

        graph = build_pipeline_graph(checkpointer=saver, store=store)
        out = await graph.ainvoke(
            {"task_dir": "inputs/circpack"},
            {"configurable": {"thread_id": "r1"}},
            context=ctx,
        )

    and resumes with the same thread id (``ainvoke(None, config, context=ctx)``, or
    ``Command(resume=...)`` after an interrupt).

    Args:
        checkpointer: The caller's checkpointer; ``None`` inherits the parent's when mounted.
        store: The caller's store (needed by parallel investigation workers that share memory).
        overrides: Replacement phase graphs (``overrides.phases``, keyed by phase name and
            ``"downstream_selection"``) and slots, which are handed on to every phase factory.
        name: The graph's name.

    Returns:
        The compiled graph.
    """
    overrides = overrides or GraphOverrides()
    graph = StepGraph(
        "pipeline",
        PipelineState,
        input_schema=PipelineInput,
        output_schema=PipelineOutput,
        overrides=overrides,
    )
    graph.add_node("record_run_config", record_run_config)
    graph.step("snapshot_task", _snapshot_node(overrides))
    graph.step("check_dependencies", _check_node(overrides))
    graph.step("finish", _finish)
    graph.add_edge(START, "record_run_config")
    graph.add_edge("record_run_config", "snapshot_task")

    chain: list[PhaseSpec] = list(PHASE_SPECS)
    gates = [f"route_{spec.name}" for spec in chain]
    first_after = {"task_drafting": "check_dependencies"}
    for index, spec in enumerate(chain):
        gate = gates[index]
        following = first_after.get(spec.name) or (
            gates[index + 1] if index + 1 < len(chain) else "finish"
        )
        graph.add_node(gate, _gate_node(spec))
        graph.step(spec.name, _phase_node(spec, _phase_graph(spec, overrides)))
        graph.add_conditional_edges(
            gate, _read_route, {"run": spec.name, "skip": following, "stop": "finish"}
        )
        graph.add_edge(spec.name, following)
    graph.add_edge("snapshot_task", gates[0])
    graph.add_edge("check_dependencies", gates[1])
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "PAPER_DIR",
    "PHASE_SPECS",
    "TASK_SNAPSHOT_DIR",
    "TASK_SNAPSHOT_METADATA",
    "TERMINAL_FAILURES",
    "PhaseSpec",
    "PipelineConfigurationError",
    "PipelineInput",
    "PipelineOutput",
    "PipelineState",
    "TaskSnapshotError",
    "Verdict",
    "build_pipeline_graph",
    "decide",
    "load_task_ref",
    "missing_scaffold_files",
    "normalize_phases",
    "prepare_task_snapshot",
    "requires_final_evaluation",
    "reseal_task_snapshot",
    "validate_selection",
]
