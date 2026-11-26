"""Tools the package gives agents in place of upstream helpers that cross the trust boundary.

Upstream agents run three helper programs of upstream's own package: the trusted clone
(``python -I -m src.investigation.program_clone``), the evaluation wrapper
(``python -I -m src.investigation.evaluate_program``) and the task-gate client
(``python "$TASK_GATE_CLIENT" acceptance-smoke``). The evaluator and the gate must stay out of the
candidate's reach, so the port gives them as tools (adaptation A1): :func:`make_clone_program_tool`,
:func:`make_evaluate_program_tool`, :func:`make_task_gate_tool`. :func:`make_report_blocker_tool`
replaces "exit non-zero and report the blocker" (A2); :func:`make_nl_tool` and
:func:`make_file_sha256_tool` give read-only roles the ``nl -ba`` and SHA-256 reads their prompts
name (A7). Memory and rubric placeholders have no tools: the staged upstream scripts serve.

Each tool is built per session and closed over the workspace, the evaluator or the gate, and the
role's root. Paths given by the agent are relative to its root (its working directory), start
with ``/`` naming that root as Deep Agents' file tools do, or start with the path at which its
shell sees the root (:func:`~langgraph_agentic_scientist.workspace.protocol.agent_path`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool

from ..core.errors import ProgramSpecError, WorkspacePathError
from ..core.interfaces import GateCheck, TaskGate
from ..core.state import EvaluationPlane, ProgramRef, TaskRef
from ..evaluation.ids import id_to_path
from ..evaluation.program_clone import clone_program, require_candidate_source_ready
from ..evaluation.program_spec import resolve_entrypoint
from ..evaluation.protocol import EvaluationRequest, EvaluationResult, Evaluator
from ..prompts import PromptRegistry
from ..workspace.paths import join, safe_relpath
from ..workspace.protocol import Workspace, agent_path

CLONE_PROGRAM = "clone_program"
EVALUATE_PROGRAM = "evaluate_program"
TASK_GATE = "task_gate"
REPORT_BLOCKER = "report_blocker"
NL = "nl"
FILE_SHA256 = "file_sha256"
MAX_NL_BYTES = 4 * 1024 * 1024
_RESULT_FILES = ("metrics.json", "correct.json", "evaluation_manifest.json")


def _description(prompts: PromptRegistry, name: str) -> str:
    return prompts.text(f"harness/{name}").strip()


def _in_root(workspace: Workspace, root: str, path: str, *, read_only: bool = False) -> str:
    """Map an agent path (relative, or absolute as its file tools see it) to a workspace path."""
    directory = agent_path(workspace, root, read_only=read_only).rstrip("/")
    if directory and (path == directory or path.startswith(f"{directory}/")):
        path = path[len(directory) :]
    return join(root, safe_relpath(path))


def _display(root: str, path: str) -> str:
    relative = safe_relpath(path)
    return f"./{relative}" if relative else "."


# --- clone_program -------------------------------------------------------------------------


def make_clone_program_tool(
    workspace: Workspace, root: str, *, prompts: PromptRegistry
) -> BaseTool:
    """Return the ``clone_program`` tool for an agent rooted at ``root``."""

    @tool(CLONE_PROGRAM, description=_description(prompts, CLONE_PROGRAM))
    async def clone_program_tool(from_dir: str, to_dir: str) -> str:
        try:
            source = _in_root(workspace, root, from_dir)
            destination = _in_root(workspace, root, to_dir)
            await require_candidate_source_ready(workspace, src_dir=source, dst_dir=destination)
            report = await clone_program(workspace, src_dir=source, dst_dir=destination)
        except (ValueError, FileNotFoundError, FileExistsError, ProgramSpecError) as exc:
            return f"clone refused: {exc}"
        lines = [f"cloned {_display(root, from_dir)} -> {_display(root, to_dir)}"]
        lines.append(f"files copied: {len(report.copied)}")
        if report.skipped_symlinks:
            lines.append(
                "symlinks not carried (copy their targets by hand if needed): "
                + ", ".join(report.skipped_symlinks)
            )
        return "\n".join(lines)

    return clone_program_tool


# --- evaluate_program ----------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluateProgramSettings:
    """What an ``evaluate_program`` tool needs to request evaluations.

    Attributes:
        task: The task evaluated on.
        id_prefix: Prefix of the evaluation ids of this session (for example
            ``"<run>/w0/c003"``); the n-th call of the session gets ``<prefix>/eval_<n>``.
        budget: Calls allowed in the session (upstream ``max_program_evals_per_attempt``); 0
            refuses every call, as upstream's post-hoc executor has no evaluation budget.
        plane: Evaluation plane of the requests.
        timeout_seconds: Evaluation timeout passed to the evaluator.
        environment: Extra environment variables for the evaluator.
    """

    task: TaskRef
    id_prefix: str
    budget: int
    plane: EvaluationPlane = "search"
    timeout_seconds: float | None = None
    environment: dict[str, str] = field(default_factory=dict)


def _call_index(runtime: ToolRuntime[Any, Any], name: str) -> int:
    """Return the position of this call among the session's calls of tool ``name``.

    Counting from the checkpointed message history makes the count survive a crash: a call
    re-executed on resume keeps its index, hence its evaluation id.
    """
    messages = (runtime.state or {}).get("messages", []) if isinstance(runtime.state, dict) else []
    index = 0
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call.get("id") == runtime.tool_call_id:
                return index
            if call.get("name") == name:
                index += 1
    return index


async def _publish_results(
    workspace: Workspace, program_dir: str, result: EvaluationResult
) -> None:
    """Publish the validated summaries at the program root (upstream wrapper semantics)."""
    payloads = {
        "metrics.json": json.dumps(result.metrics, indent=2, sort_keys=True) + "\n",
        "correct.json": json.dumps({"correct": result.correct}, indent=2, sort_keys=True) + "\n",
    }
    for name, text in payloads.items():
        await workspace.write_bytes(f"{program_dir}/{name}", text.encode("utf-8"))
    manifest = {
        "producer": "isolated_evaluator",
        "schema_version": 1,
        "program_id": program_dir.rsplit("/", 1)[-1],
        "output_dir": result.output_dir or "",
        "returncode": 0,
        "published_summaries": ["metrics.json", "correct.json"],
        "evaluation_id": result.evaluation_id,
    }
    await workspace.write_bytes(
        f"{program_dir}/evaluation_manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


async def _published_for(workspace: Workspace, program_dir: str, evaluation_id: str) -> bool:
    try:
        manifest = json.loads(await workspace.read_bytes(f"{program_dir}/evaluation_manifest.json"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return False
    return isinstance(manifest, dict) and manifest.get("evaluation_id") == evaluation_id


def make_evaluate_program_tool(
    workspace: Workspace,
    evaluator: Evaluator | None,
    root: str,
    settings: EvaluateProgramSettings,
    *,
    prompts: PromptRegistry,
) -> BaseTool:
    """Return the ``evaluate_program`` tool for an agent rooted at ``root``.

    Every call consumes one evaluation of ``settings.budget``, including failed and interrupted
    calls; a call beyond the budget is refused before any evaluation. The evaluation id of the
    n-th call is ``<id_prefix>/eval_<n>``, so a call re-executed after a crash asks the evaluator
    for the same id. The tool message carries the :class:`EvaluationResult` as its artifact.
    """

    @tool(
        EVALUATE_PROGRAM,
        description=_description(prompts, EVALUATE_PROGRAM),
        response_format="content_and_artifact",
    )
    async def evaluate_program_tool(
        program_dir: str,
        runtime: ToolRuntime,
        num_replications: int = 1,
    ) -> tuple[str, dict[str, Any] | None]:
        index = _call_index(runtime, EVALUATE_PROGRAM)
        if index >= settings.budget:
            return (
                f"Evaluation refused: the evaluation budget of this attempt is consumed "
                f"({index} of {settings.budget} evaluate_program calls already made). Do not "
                "evaluate again; verify rubrics from the existing evidence, write memory and stop "
                "with cap_reached.",
                None,
            )
        eval_id = f"{id_to_path(settings.id_prefix)}/eval_{index + 1:02d}"
        try:
            program = _in_root(workspace, root, program_dir)
        except WorkspacePathError as exc:
            return f"evaluation wrapper error: {exc}", None
        name = program.rsplit("/", 1)[-1]
        if not name.startswith("program_") or not name[len("program_") :].isdigit():
            return (
                f"evaluation wrapper error: program_dir must be named program_<index>: {name}",
                None,
            )
        if num_replications < 1:
            return "evaluation wrapper error: --num-replications must be positive", None
        if evaluator is None:
            return "evaluation wrapper error: no evaluator is configured for this run", None
        if not await _published_for(workspace, program, eval_id):
            for filename in _RESULT_FILES:
                if await workspace.exists(f"{program}/{filename}"):
                    return (
                        f"evaluation wrapper error: Evaluation result already exists: {filename}",
                        None,
                    )
        try:
            entrypoint = await resolve_entrypoint(workspace, program)
        except ProgramSpecError as exc:
            return f"evaluation wrapper error: {exc}", None
        request = EvaluationRequest(
            evaluation_id=eval_id,
            plane=settings.plane,
            program=ProgramRef(path=program, entrypoint=entrypoint),
            task=settings.task,
            num_replications=num_replications,
            timeout_seconds=settings.timeout_seconds,
            environment=dict(settings.environment),
        )
        result = await evaluator.evaluate(request)
        if result.status == "completed" and not await _published_for(workspace, program, eval_id):
            await _publish_results(workspace, program, result)
        lines = [
            f"evaluation outputs: {result.output_dir or '(evaluator-owned)'}",
            f"evaluation id: {eval_id} (call {index + 1} of {settings.budget})",
            f"status: {result.status}",
        ]
        if result.status == "completed":
            lines.append(
                f"combined_score: {result.metrics.get('combined_score')}; "
                f"correct: {str(result.correct).lower()}"
            )
            lines.append(f"published: {_display(root, program_dir)}/metrics.json, correct.json")
        if result.failure_category:
            lines.append(f"failure category: {result.failure_category}")
        if result.summary:
            lines.append(result.summary)
        return "\n".join(lines), result.model_dump(mode="json", by_alias=True)

    return evaluate_program_tool


def evaluations_from_messages(messages: list[Any]) -> list[EvaluationResult]:
    """Collect the evaluation results that ``evaluate_program`` calls attached to tool messages."""
    results: list[EvaluationResult] = []
    for message in messages:
        if (
            getattr(message, "type", None) != "tool"
            or getattr(message, "name", None) != EVALUATE_PROGRAM
        ):
            continue
        artifact = getattr(message, "artifact", None)
        if isinstance(artifact, dict):
            results.append(EvaluationResult.model_validate(artifact))
    return results


# --- task_gate ------------------------------------------------------------------------------


_GATE_CHECKS: tuple[str, ...] = (
    "acceptance-smoke",
    "contract",
    "runtime",
    "search-smoke",
    "final-smoke",
    "production-entry",
)


def make_task_gate_tool(
    workspace: Workspace,
    gate: TaskGate,
    task: TaskRef,
    *,
    id_prefix: str,
    max_checks: int,
    prompts: PromptRegistry,
) -> BaseTool:
    """Return the in-session ``task_gate`` tool (upstream's gate client), bounded per repair."""

    @tool(TASK_GATE, description=_description(prompts, TASK_GATE))
    async def task_gate_tool(check: str, runtime: ToolRuntime) -> str:
        if check not in _GATE_CHECKS:
            return f"unknown check {check!r}; use one of: {', '.join(_GATE_CHECKS)}"
        index = _call_index(runtime, TASK_GATE)
        if index >= max_checks:
            return (
                f"request refused: the gate service is closed after {max_checks} checks in this "
                "repair; this is operational, not a new scaffold failure"
            )
        report = await gate.run_check(
            _gate_check(check),
            task=task,
            workspace=workspace,
            evaluation_id=f"{id_to_path(id_prefix)}/gate_{index + 1:02d}",
        )
        status = "passed" if report.passed else "failed"
        lines = [f"{check}: {status}"]
        if report.category:
            lines.append(f"category: {report.category}")
        if report.stage:
            lines.append(f"stage: {report.stage}")
        if report.diagnostic:
            lines.append(report.diagnostic)
        return "\n".join(lines)

    return task_gate_tool


def _gate_check(check: str) -> GateCheck:
    return check  # type: ignore[return-value]


# --- report_blocker -------------------------------------------------------------------------


def make_report_blocker_tool(*, prompts: PromptRegistry) -> BaseTool:
    """Return ``report_blocker``, which records the blocker and ends the session."""

    @tool(
        REPORT_BLOCKER,
        description=_description(prompts, REPORT_BLOCKER),
        return_direct=True,
        response_format="content_and_artifact",
    )
    def report_blocker_tool(text: str) -> tuple[str, dict[str, str]]:
        return "Blocker recorded. The session ends now.", {"blocker": text}

    return report_blocker_tool


def blocker_from_messages(messages: list[Any]) -> str | None:
    """Return the last blocker reported through ``report_blocker`` in a session, if any."""
    blocker: str | None = None
    for message in messages:
        if (
            getattr(message, "type", None) == "tool"
            and getattr(message, "name", None) == REPORT_BLOCKER
        ):
            artifact = getattr(message, "artifact", None)
            if isinstance(artifact, dict) and isinstance(artifact.get("blocker"), str):
                blocker = artifact["blocker"]
    return blocker


# --- nl and file_sha256 ---------------------------------------------------------------------


def make_nl_tool(workspace: Workspace, root: str, *, prompts: PromptRegistry) -> BaseTool:
    """Return the read-only ``nl`` tool (``nl -ba`` line numbering)."""

    @tool(NL, description=_description(prompts, NL))
    async def nl_tool(path: str) -> str:
        try:
            target = _in_root(workspace, root, path, read_only=True)
            data = await workspace.read_bytes(target)
        except (WorkspacePathError, FileNotFoundError, IsADirectoryError, PermissionError) as exc:
            return f"nl: {path}: {exc}"
        truncated = len(data) > MAX_NL_BYTES
        text = data[:MAX_NL_BYTES].decode("utf-8", errors="replace")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        numbered = "\n".join(f"{index:>6}\t{line}" for index, line in enumerate(lines, start=1))
        if truncated:
            numbered += f"\n[output truncated after {MAX_NL_BYTES} bytes]"
        return numbered

    return nl_tool


def make_file_sha256_tool(workspace: Workspace, root: str, *, prompts: PromptRegistry) -> BaseTool:
    """Return the read-only ``file_sha256`` tool (``sha256sum`` of one file)."""

    @tool(FILE_SHA256, description=_description(prompts, FILE_SHA256))
    async def file_sha256_tool(path: str) -> str:
        try:
            data = await workspace.read_bytes(_in_root(workspace, root, path, read_only=True))
        except (WorkspacePathError, FileNotFoundError, IsADirectoryError, PermissionError) as exc:
            return f"sha256sum: {path}: {exc}"
        return f"{hashlib.sha256(data).hexdigest()}  {path}"

    return file_sha256_tool


__all__ = [
    "CLONE_PROGRAM",
    "EVALUATE_PROGRAM",
    "FILE_SHA256",
    "NL",
    "REPORT_BLOCKER",
    "TASK_GATE",
    "EvaluateProgramSettings",
    "blocker_from_messages",
    "evaluations_from_messages",
    "make_clone_program_tool",
    "make_evaluate_program_tool",
    "make_file_sha256_tool",
    "make_nl_tool",
    "make_report_blocker_tool",
    "make_task_gate_tool",
]
