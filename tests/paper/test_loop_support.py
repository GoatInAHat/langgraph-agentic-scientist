"""Shared fixtures of the paper-loop tests: a task, paper data, scripted roles, stub slots.

Every test runs offline: scripted chat models, :class:`FakePaperToolchain` instead of LaTeX, a
stub ``paper.references`` slot (the host side's references graph is tested on its own), and a
:class:`LocalWorkspace` in a temporary directory.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from langgraph_agentic_scientist import GraphOverrides, ScientistContext
from langgraph_agentic_scientist.config import (
    AgentsConfig,
    ModelRetryConfig,
    PaperConfig,
    ScientistConfig,
)
from langgraph_agentic_scientist.core.results import GroundingSource, PaperData, ValidationReport
from langgraph_agentic_scientist.core.state import TaskRef
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper import build_paper_graph
from langgraph_agentic_scientist.paper.models import (
    AuditPayload,
    DraftGapsAnswer,
    ReviewPayload,
    StrategyAnswer,
)
from langgraph_agentic_scientist.testing import (
    FakePaperToolchain,
    ScriptedModels,
    final,
    structured_call,
    tool_call,
)

TASK_MD = "# Goal\nPack circles densely.\n\n# Rubrics\n- The packing is valid.\n"
TEX = (
    "\\documentclass{article}\n\\begin{document}\nA grounded result of 2.6.\n"
    "\\bibliography{references}\n\\end{document}\n"
)


SCOPE_AUDIT = {
    "manuscript_focus": {"shared_method_summary": "Dense packing.", "scope_strategy": "Local."},
    "shared_method_elements": ["greedy placement"],
    "scoped_method_elements": [],
    "claims_requiring_scoping": [],
}
"""A ``method_scope_audit.json`` that passes the scope-audit check."""


def paper_data(**extra: Any) -> PaperData:
    """A prebuilt paper data object with one grounding source."""
    return PaperData(
        generated_at="2026-01-01T00:00:00+00:00",
        totals={"evaluated_programs": 3},
        grounding_sources={
            "result.best.combined_score": GroundingSource(type="metric", value=2.6),
        },
        **extra,
    )


async def setup_task(workspace: Any, root: str = "run/task") -> TaskRef:
    """Write the task snapshot and return its reference."""
    await workspace.write_bytes(f"{root}/task.md", TASK_MD.encode())
    return TaskRef(root=root, goal="Pack circles densely.", rubrics="- The packing is valid.")


async def references_stub(
    state: Mapping[str, Any], runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """A ``paper.references`` slot that writes a one-entry bibliography."""
    ws = runtime.context.workspace
    paper_dir = state["paper_dir"]
    if state.get("reference_mode") != "preserve":
        await ws.write_bytes(
            f"{paper_dir}/references.bib",
            b"@article{a2020,\n  title={A},\n  author={B},\n  year={2020}\n}\n",
        )
        await ws.write_bytes(
            f"{paper_dir}/references_provenance.json", json.dumps({"references": []}).encode()
        )
        return {"references": {"status": "completed"}}
    return {"references": {"status": "preserved"}}


def write_paper(content: str = TEX, *, extra: Mapping[str, str] | None = None) -> list[AIMessage]:
    """Writer script steps: write ``paper.tex`` (and ``extra`` files), then stop."""
    steps = [tool_call("write_file", {"file_path": "/paper.tex", "content": content})]
    for path, text in (extra or {}).items():
        steps.append(tool_call("write_file", {"file_path": f"/{path}", "content": text}))
    steps.append(final("draft written"))
    return steps


def edit_outputs(content: str = TEX) -> list[AIMessage]:
    """Writer steps of edit mode: the manuscript, the claims and the scope-audit pair."""
    return write_paper(
        content,
        extra={
            "claims_grounding.json": json.dumps({"claims": []}),
            "method_scope_audit.json": json.dumps(SCOPE_AUDIT),
            "method_scope_audit.md": "# Method scope audit\n",
        },
    )


def no_gaps() -> AIMessage:
    """The draft-gap analyst's answer when no citation gaps exist."""
    return structured_call(
        DraftGapsAnswer,
        {"memo": "# Gaps\nNone.", "gaps": {"status": "no_gaps", "summary": "coverage is adequate"}},
    )


def audit_pass() -> AIMessage:
    """The evidence auditor's pass."""
    return structured_call(
        AuditPayload, {"verdict": "pass", "summary": "Claims match the evidence.", "findings": []}
    )


def audit_fail(finding_id: str = "results-overclaim-1") -> AIMessage:
    """The evidence auditor's fail with one finding."""
    return structured_call(
        AuditPayload,
        {
            "verdict": "fail",
            "summary": "One claim overstates the evidence.",
            "findings": [
                {
                    "id": finding_id,
                    "section": "Results",
                    "claim": "consistently better",
                    "issue": "only one benchmark",
                    "evidence": "claims_grounding.json claim[0]",
                    "fix": "narrow the sentence",
                }
            ],
        },
    )


def review(status: str = "revise") -> AIMessage:
    """The in-loop reviewer's answer."""
    major = (
        [
            {
                "id": "major-1",
                "section": "Results",
                "issue": "overclaim",
                "evidence": ["paper.tex line 3"],
                "requested_change": "narrow it",
            }
        ]
        if status == "revise"
        else []
    )
    return structured_call(
        ReviewPayload,
        {
            "status": status,
            "summary": "A short review.",
            "major_comments": major,
            "score": {"value": 5, "label": "borderline"},
        },
    )


def strategy(
    family: str = "paper_standalone", *, status: str = "ready", **fields: Any
) -> AIMessage:
    """The strategist's answer (a valid strategy by default)."""
    payload: dict[str, Any] = {
        "status": status,
        "selected_prompt_family": family,
        "recommended_title": "Dense packing",
        "central_question": "How dense?",
        "central_thesis": "Denser.",
        "narrative_rationale": "Grounded.",
        "method_groups": [{"name": "proposed", "role": "proposed", "include_reason": "main"}],
        "results_focus": ["score"],
        "figure_plan": ["bar"],
        "figure_specs": [],
        "supplementary_plan": {
            "purpose": "depth",
            "planned_subsections": [
                {
                    "heading": "Details",
                    "grounded_content": "settings",
                    "source_ids": ["result.best.combined_score"],
                }
            ],
        },
        "forbidden_headings": [],
        "forbidden_phrases": [],
        "notes_for_writer": [],
        "failure_reason": "",
    }
    payload.update(fields)
    return structured_call(StrategyAnswer, {"memo": "# Strategy\nDense.", "strategy": payload})


def fast_config(**updates: Any) -> ScientistConfig:
    """A configuration whose in-session model retries do not sleep."""
    agents = AgentsConfig(
        model_retry=ModelRetryConfig(max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0)
    )
    return ScientistConfig(agents=agents, **updates)


def paper_config(**updates: Any) -> PaperConfig:
    """A paper configuration with small budgets."""
    values: dict[str, Any] = {"max_repair_rounds": 3, "timeout_seconds": None}
    values.update(updates)
    return PaperConfig(**values)


def build(
    *,
    slots: Mapping[str, Any] | None = None,
) -> Any:
    """The paper graph with the stub references slot and an in-memory checkpointer."""
    merged: dict[str, Any] = {"paper.references": references_stub}
    merged.update(slots or {})
    return build_paper_graph(checkpointer=InMemorySaver(), overrides=GraphOverrides(slots=merged))


def models(scripts: Mapping[Role, Sequence[Any]]) -> ScriptedModels:
    """Scripted models, one script per role."""
    return ScriptedModels.from_scripts(scripts, context_window=200_000)


def reports(*passed: bool | ValidationReport) -> FakePaperToolchain:
    """A toolchain that returns validation reports in order (``True`` passes)."""
    items = [
        item
        if isinstance(item, ValidationReport)
        else ValidationReport(passed=bool(item), errors=[] if item else ["hard error A"])
        for item in passed
    ]
    return FakePaperToolchain(items)


async def run(
    graph: Any,
    context: ScientistContext,
    task: TaskRef,
    *,
    thread: str = "t",
    **inputs: Any,
) -> dict[str, Any]:
    """Run the paper graph to its end."""
    payload: dict[str, Any] = {"task": task, "run_id": "r1", "paper_data": paper_data()}
    payload.update(inputs)
    return dict(
        await graph.ainvoke(payload, {"configurable": {"thread_id": thread}}, context=context)
    )


def prompts_of(models_: ScriptedModels, role: Role) -> list[str]:
    """The first user message of every session of ``role``."""
    out: list[str] = []
    for call in models_.scripted(role).calls:
        human = [m for m in call if isinstance(m, BaseMessage) and m.type == "human"]
        if human and isinstance(human[0].content, str) and (not out or out[-1] != human[0].content):
            out.append(human[0].content)
    return out


ContextFactory = Callable[..., ScientistContext]


def test_support_builds_the_graph() -> None:
    graph = build()
    assert "paper.write.before" in graph.get_graph().nodes
