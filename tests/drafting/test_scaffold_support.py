"""Builders shared by the scaffold tests: a drafted scaffold, scripted answers, a context.

Not a test module. It writes the contracted toy task of ``tests/fixtures/core`` (a complete scaffold
that passes the contract lint) into a workspace, builds the structured answers of the alignment
judge from the prompt it is given, and assembles a context with scripted models and fakes. The tests
import it as ``test_scaffold_support``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from langgraph_agentic_scientist import LocalWorkspace, ScientistContext
from langgraph_agentic_scientist.config import AgentsConfig, ModelRetryConfig, ScientistConfig
from langgraph_agentic_scientist.core.render import task_sections
from langgraph_agentic_scientist.core.state import ProgramRef, TaskRef
from langgraph_agentic_scientist.drafting.models import (
    REQUIRED_ALIGNMENT_DIMENSIONS,
    AlignmentDimension,
    AlignmentEvidence,
    AlignmentReport,
)
from langgraph_agentic_scientist.evaluation.protocol import EvaluationRequest, EvaluationResult
from langgraph_agentic_scientist.testing import (
    FakeEvaluator,
    ScriptedModels,
    structured_call,
)
from langgraph_agentic_scientist.workspace.protocol import Workspace

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "core" / "contract_task"
CAPTURE = '"""Compatibility artifact."""\n\n\nclass OutputCapture:\n    """Capture nothing."""\n'
REQUIREMENTS = "python==3.12\n"
RECURSION = {"recursion_limit": 3000}


def fast(**drafting: Any) -> ScientistConfig:
    """A configuration without in-session retry sleeps, with drafting limits overridden."""
    from langgraph_agentic_scientist.config import DraftingConfig

    return ScientistConfig(
        agents=AgentsConfig(
            model_retry=ModelRetryConfig(
                max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0
            )
        ),
        drafting=DraftingConfig(**drafting),
    )


async def write_scaffold(
    workspace: Workspace,
    root: str = "task",
    *,
    capture: bool = True,
    requirements: bool = True,
    only_task_md: bool = False,
) -> TaskRef:
    """Write the toy task (and, unless ``only_task_md``, its complete scaffold) under ``root``."""
    for path in sorted(FIXTURE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(FIXTURE).as_posix()
        if only_task_md and relative != "task.md":
            continue
        await workspace.write_bytes(f"{root}/{relative}", path.read_bytes())
    if capture and not only_task_md:
        await workspace.write_bytes(f"{root}/capture.py", CAPTURE.encode())
    if requirements and not only_task_md:
        await workspace.write_bytes(f"{root}/requirements.txt", REQUIREMENTS.encode())
    task_md = (FIXTURE / "task.md").read_text(encoding="utf-8")
    goal, rubrics = task_sections(task_md)
    return TaskRef(
        root=root,
        task_md_sha256=hashlib.sha256(task_md.encode()).hexdigest(),
        goal=goal,
        rubrics=rubrics,
        seed_program=ProgramRef(path=f"{root}/program_0", entrypoint="src/program.py"),
    )


def fixture_result(request: EvaluationRequest) -> EvaluationResult:
    """The answer of a healthy evaluator for every plane of the toy task."""
    if request.evaluator == "final":
        metrics: dict[str, Any] = {"final_score": 0.5}
    else:
        metrics = {"combined_score": -1.0, "accuracy": 0.0}
    return EvaluationResult(
        evaluation_id=request.evaluation_id,
        status="completed",
        metrics=metrics,
        correct=True,
        feasible=True if request.plane == "qualification" else None,
        summary="ok",
        duration_seconds=1.0,
    )


def make_evaluator(
    handler: Callable[[EvaluationRequest], EvaluationResult] | None = None, **kwargs: Any
) -> FakeEvaluator:
    """A fake evaluator answering every request with ``handler`` (healthy by default)."""
    return FakeEvaluator(default=handler or fixture_result, **kwargs)


# --- the judge -----------------------------------------------------------------------------


def prompt_text(messages: Sequence[BaseMessage]) -> str:
    """The first human message of a session: the rendered prompt."""
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    raise AssertionError("no human message")


def expected_fingerprints(messages: Sequence[BaseMessage]) -> dict[str, str]:
    """Read the expected fingerprint object out of the judge's prompt."""
    match = re.search(r"Expected fingerprints: `(\{.*?\})`", prompt_text(messages))
    assert match, "the judge prompt names no fingerprints"
    return dict(json.loads(match.group(1)))


def alignment_report(
    fingerprints: Mapping[str, str],
    *,
    failing: Sequence[str] = (),
    path: str = "task.md",
    line: int = 1,
) -> AlignmentReport:
    """A report that passes every dimension except ``failing``, with valid evidence."""
    dimensions = [
        AlignmentDimension(
            id=identifier,
            verdict="fail" if identifier in failing else "pass",
            evidence=[AlignmentEvidence(path=path, line=line, detail=f"checked {identifier}")],
            remediation=f"fix {identifier}" if identifier in failing else "",
        )
        for identifier in REQUIRED_ALIGNMENT_DIMENSIONS
    ]
    return AlignmentReport(
        report_schema_version=1,
        verdict="fail" if failing else "pass",
        summary="failing" if failing else "all dimensions pass",
        fingerprints=dict(fingerprints),
        dimensions=dimensions,
    )


def judge_step(
    *,
    failing: Sequence[str] = (),
    mutate: Callable[[AlignmentReport], AlignmentReport] | None = None,
) -> Callable[[list[BaseMessage]], AIMessage]:
    """A script step: the judge answers for the fingerprints in the prompt it is given."""

    def step(messages: list[BaseMessage]) -> AIMessage:
        report = alignment_report(expected_fingerprints(messages), failing=failing)
        if mutate is not None:
            report = mutate(report)
        return structured_call(AlignmentReport, report)

    return step


# --- a context -----------------------------------------------------------------------------


def make_context(
    workspace: LocalWorkspace,
    scripts: Mapping[str, Sequence[Any]],
    *,
    evaluator: FakeEvaluator | None = None,
    config: ScientistConfig | None = None,
    **kwargs: Any,
) -> tuple[ScientistContext, ScriptedModels]:
    """A context with scripted models (one script per role) and a fake evaluator."""
    models = ScriptedModels.from_scripts(dict(scripts), context_window=200_000)
    context = ScientistContext(
        models=models,
        workspace=workspace,
        evaluator=evaluator if evaluator is not None else make_evaluator(),
        config=config or fast(),
        **kwargs,
    )
    return context, models


def scaffold_files() -> dict[str, str]:
    """The files a drafter creates for the toy task (the scaffold of ``tests/fixtures/core``)."""
    files: dict[str, str] = {}
    for path in sorted(FIXTURE.rglob("*")):
        relative = path.relative_to(FIXTURE).as_posix()
        if path.is_file() and relative not in ("task.md", "program_0/program_spec.toml"):
            files[relative] = path.read_text(encoding="utf-8")
    files["capture.py"] = CAPTURE
    files["requirements.txt"] = REQUIREMENTS
    return files


def drafter_script(
    *, skip: Sequence[str] = (), transform: Mapping[str, Callable[[str], str]] | None = None
) -> list[Any]:
    """A drafter that creates the toy task's scaffold in one turn, then stops.

    Args:
        skip: Files it does not create.
        transform: Per file, a function applied to the content it writes (to draft a defect).
    """
    from langgraph_agentic_scientist.testing import final, tool_calls

    changes = dict(transform or {})
    writes = [
        (
            "write_file",
            {"file_path": f"/task/{path}", "content": changes.get(path, lambda x: x)(text)},
        )
        for path, text in scaffold_files().items()
        if path not in skip
    ]
    return [tool_calls(*writes), final("scaffold created")]


def write_file_call(path: str, content: str) -> AIMessage:
    """An agent turn that writes a file with Deep Agents' ``write_file`` tool."""
    from langgraph_agentic_scientist.testing import tool_call

    return tool_call("write_file", {"file_path": path, "content": content})


__all__ = [
    "CAPTURE",
    "FIXTURE",
    "RECURSION",
    "REQUIREMENTS",
    "alignment_report",
    "drafter_script",
    "expected_fingerprints",
    "fast",
    "fixture_result",
    "judge_step",
    "make_context",
    "make_evaluator",
    "prompt_text",
    "scaffold_files",
    "write_file_call",
    "write_scaffold",
]
