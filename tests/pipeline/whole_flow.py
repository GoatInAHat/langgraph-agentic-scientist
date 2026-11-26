"""A scripted run of the whole method on the package's own phase graphs (no phase stubs).

Every role answers through a policy that reads its own session's messages, so one policy serves
every session of the role and the run does not depend on how many sessions a phase opens. The
evaluators, the reference retriever and the LaTeX toolchain are the package's fakes; everything
else (the graphs, the prompts, the host checks, the workspace) is the package's own code.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langgraph_agentic_scientist import (
    Hooks,
    LocalWorkspace,
    ScientistConfig,
    ScientistContext,
)
from langgraph_agentic_scientist.config import (
    AgentsConfig,
    InvestigationConfig,
    ModelRetryConfig,
    PaperConfig,
    PosthocConfig,
    ReviewConfig,
    VariantsConfig,
)
from langgraph_agentic_scientist.core.interfaces import ReferenceRecord
from langgraph_agentic_scientist.investigation.ideas import IdeaDraft
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.paper.models import (
    AuditPayload,
    DraftGapsAnswer,
    PosthocRanking,
)
from langgraph_agentic_scientist.paper.references import ReferencePlanAnswer
from langgraph_agentic_scientist.posthoc.models import PosthocBundle, SelectionAnswer
from langgraph_agentic_scientist.review.models import ReviewerAnswer
from langgraph_agentic_scientist.testing import (
    FakeEvaluator,
    FakeFinalEvaluator,
    FakePaperToolchain,
    FakeRetriever,
    ScriptedModels,
    completed_result,
    final,
    structured_call,
    tool_call,
    tool_calls,
    write_minimal_task,
)
from langgraph_agentic_scientist.testing.fixtures import MINIMAL_EVALUATOR
from langgraph_agentic_scientist.variants.models import (
    VariantBundleAnswer,
    VariantDecisionsAnswer,
    VariantSummariesAnswer,
)

ALL_PHASES = (
    "task_drafting",
    "investigation",
    "ablations",
    "final_evaluation",
    "posthoc",
    "paper",
    "review",
    "revision",
)
TASK_DIR = "inputs/minimal"
PAPER_TEXT = (
    "We study a program that sets a score. The candidate raises the score from one to six and "
    "the sealed final evaluation confirms it. An ablation removes the change. "
) * 3
TEX = (
    "\\documentclass{article}\n\\begin{document}\n"
    "The candidate reaches a combined score of 6.0.\n"
    "\\bibliography{references}\n\\end{document}\n"
)
SCOPE_AUDIT = {
    "manuscript_focus": {"shared_method_summary": "Raise SCORE.", "scope_strategy": "Local."},
    "shared_method_elements": ["edit SCORE"],
    "scoped_method_elements": [],
    "claims_requiring_scoping": [],
}
UNFILLED = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


# --- reading a session ----------------------------------------------------------------------


def prompt_of(messages: Sequence[BaseMessage]) -> str:
    """The session's prompt: its first human message."""
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else json.dumps(content)
    return ""


def done(messages: Sequence[BaseMessage]) -> int:
    """How many tool results the session has seen so far."""
    return sum(isinstance(message, ToolMessage) for message in messages)


Policy = Callable[[list[BaseMessage]], AIMessage]


def steps(*turns: Callable[[list[BaseMessage]], AIMessage] | AIMessage) -> Policy:
    """A policy that plays ``turns`` in order, one per tool result seen, then stops."""

    def policy(messages: list[BaseMessage]) -> AIMessage:
        index = done(messages)
        if index >= len(turns):
            return final("done")
        turn = turns[index]
        return turn(messages) if callable(turn) else turn

    return policy


def answer(schema: type[Any], payload: Any) -> Policy:
    """A structured role that answers at once."""
    return lambda messages: structured_call(schema, payload)


# --- the roles --------------------------------------------------------------------------------


def memory_line(messages: list[BaseMessage]) -> AIMessage:
    match = re.search(r"Attempt ID: `([^`]+)`", prompt_of(messages))
    record = {
        "timestamp": "2026-01-01T00:00:00Z",
        "step": "edit",
        "scope": "attempt",
        "summary": "raised SCORE to 6",
        "evidence": "program_1/src/program.py",
        "next_action": "evaluate",
        "attempt_id": match.group(1) if match else "unknown",
        "program_id": "program_1",
    }
    return tool_call(
        "write_file",
        {"file_path": "/_shared_memory/memory.jsonl", "content": json.dumps(record) + "\n"},
    )


IDEA = {
    "idea_id": "raise_score",
    "title": "Raise the score",
    "hypothesis": "A larger SCORE is better while it stays at most 10.",
    "implementation_plan": ["Set SCORE to 6.0"],
    "target_rubrics": [
        {"id": "agent_metric", "description": "combined_score > 5", "type": "metric"},
        {
            "id": "sets_score",
            "description": "The program sets SCORE to 6.",
            "type": "implementation",
            "verification": {"type": "source_contains", "patterns": ["SCORE = 6.0"]},
        },
    ],
    "notes_for_agent": "Edit only SCORE.",
}

CANDIDATE_AUTHOR = steps(
    tool_call("execute", {"command": "cp -r program_0 program_1"}),
    tool_call(
        "edit_file",
        {
            "file_path": "/program_1/src/program.py",
            "old_string": "SCORE = 1.0",
            "new_string": "SCORE = 6.0",
        },
    ),
    memory_line,
)


def _variant_editor(messages: list[BaseMessage]) -> AIMessage:
    match = re.search(
        r"Create exactly \./program_(\d+)/ by cloning \./program_(\d+)/", prompt_of(messages)
    )
    assert match, "the variant editor's prompt names the program to create"
    if done(messages):
        return final("candidate created")
    index, previous = match.groups()
    return tool_call(
        "clone_program", {"from_dir": f"program_{previous}", "to_dir": f"program_{index}"}
    )


def _variant_verifier(messages: list[BaseMessage]) -> AIMessage:
    match = re.search(r"evaluated \./program_(\d+)/", prompt_of(messages))
    assert match, "the variant verifier's prompt names the evaluated program"
    path = f"/program_{match.group(1)}/rubric_status.json"
    edits = [
        ('"evidence": ""', '"evidence": "checked against the results"'),
        ('"pass": false', '"pass": true'),
    ]
    index = done(messages)
    if index >= len(edits):
        return final("verified")
    old, new = edits[index]
    return tool_call(
        "edit_file", {"file_path": path, "old_string": old, "new_string": new, "replace_all": True}
    )


VARIANT_BUNDLE = {
    "decision": "propose_variants",
    "summary": "One ablation of the change.",
    "candidates": [
        {
            "variant_id": "no_raise",
            "title": "Without the raised score",
            "kind": "ablation",
            "hypothesis": "The raise is what improves the score.",
            "relevance_reason": "isolates the innovation",
            "feasibility_reason": "one line",
            "target_components": ["SCORE"],
            "implementation_plan": ["set SCORE back to 1.0"],
            "expected_effect": "lower score",
        }
    ],
}

ANALYSIS_ID = "score_trace"


def _analysis_outputs(messages: list[BaseMessage]) -> AIMessage:
    rubric_ids = [
        "analysis_summary_present",
        "analysis_manifest_present",
        "interestingness_rubric_present",
        "required_outputs_present",
        "claim_grounded",
        "checkpoint_guardrail_respected",
    ]
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "attempt_id": "attempt_1",
        "summary": "the score is six",
        "claim": "the candidate sets SCORE to 6.0",
        "grounded": True,
        "reran_evaluator": False,
        "primary_metric_tolerance": 0.01,
        "outputs": {"tables": ["score.csv"]},
        "evidence_paths": ["task_bundle/program_0/src/program.py"],
        "checkpoint_paths": [],
    }
    files = {
        "analysis_summary.md": "# Summary\nThe candidate sets SCORE to 6.0.\n",
        "interestingness_rubric.json": json.dumps({"novelty": "low"}),
        "analysis_manifest.json": json.dumps(manifest),
        "rubric_status.json": json.dumps(
            {
                "analysis_id": ANALYSIS_ID,
                "overall_pass": True,
                "rubrics": [{"id": rubric, "pass": True} for rubric in rubric_ids],
            }
        ),
        "score.csv": "score\n6.0\n",
    }
    return tool_calls(
        *(
            ("write_file", {"file_path": f"/{path}", "content": text})
            for path, text in files.items()
        )
    )


def _write(path: str, content: str) -> AIMessage:
    return tool_call("write_file", {"file_path": f"/{path}", "content": content})


PAPER_WRITER = steps(_write("paper.tex", TEX))
REVISION_WRITER = steps(
    _write("paper.tex", TEX.replace("reaches", "reaches, on one task,")),
    _write("claims_grounding.json", json.dumps({"claims": []})),
    _write("method_scope_audit.json", json.dumps(SCOPE_AUDIT)),
    _write("method_scope_audit.md", "# Method scope audit\n"),
)

REVIEW = {
    "Summary": "The paper raises a score.",
    "Strengths": ["Clear."],
    "Weaknesses": ["One task.", "Broad claims."],
    "Originality": 2,
    "Quality": 3,
    "Clarity": 3,
    "Significance": 2,
    "Questions": ["Does it generalise?"],
    "Limitations": ["Single task."],
    "Ethical Concerns": False,
    "Soundness": 3,
    "Presentation": 3,
    "Contribution": 2,
    "Overall": 4,
    "Confidence": 4,
    "Decision": "Reject",
}

REFERENCE_PLAN = {
    "memo": "References for a study of a score.",
    "plan": {
        "status": "ready",
        "central_topic": "Program search",
        "queries": [
            {"query": "program search", "category": "core_method", "rationale": "method"},
            {"query": "random search", "category": "baseline_comparator", "rationale": "base"},
            {"query": "ablation", "category": "result_driven_theme", "rationale": "ablation"},
        ],
    },
}


def policies() -> dict[Role, Policy]:
    """One policy per role of the method that the run uses."""
    return {
        Role.IDEATOR: answer(IdeaDraft, IDEA),
        Role.CANDIDATE_AUTHOR: CANDIDATE_AUTHOR,
        Role.RETROFIT_AGENT: steps(),
        Role.VARIANT_GENERATOR: answer(VariantBundleAnswer, VARIANT_BUNDLE),
        Role.VARIANT_REVIEWER: answer(
            VariantDecisionsAnswer,
            {"decisions": [{"variant_id": "no_raise", "accepted": True, "reason": "isolates"}]},
        ),
        Role.VARIANT_EDITOR: _variant_editor,
        Role.VARIANT_VERIFIER: _variant_verifier,
        Role.README_SUMMARISER: answer(
            VariantSummariesAnswer,
            {"task_summary": "Raise a score.", "variant_summaries": {"no_raise": "Undoes it."}},
        ),
        Role.ANALYSIS_GENERATOR: answer(
            PosthocBundle,
            {
                "decision": "propose_analyses",
                "summary": "One trace of the score.",
                "analyses": [
                    {
                        "id": ANALYSIS_ID,
                        "title": "Trace of the score",
                        "family": "score_trace",
                        "hypothesis": "the score is the raised constant",
                        "relevance_reason": "supports the claim",
                        "feasibility_reason": "reads the program",
                        "required_artifacts": [],
                        "requires_retrain": False,
                        "implementation_plan": ["read the program", "write the table"],
                        "expected_claim": "the score is 6.0",
                        "expected_outputs": ["score.csv"],
                    }
                ],
            },
        ),
        Role.ANALYSIS_SELECTOR: answer(
            SelectionAnswer,
            {"decisions": [{"id": ANALYSIS_ID, "accepted": True, "reason": "grounded"}]},
        ),
        Role.ANALYSIS_EXECUTOR: steps(_analysis_outputs),
        Role.REFERENCE_PLANNER: answer(ReferencePlanAnswer, REFERENCE_PLAN),
        Role.POSTHOC_RANKER: answer(
            PosthocRanking,
            {
                "ranked_analyses": [
                    {
                        "analysis_id": ANALYSIS_ID,
                        "rank": 1,
                        "relevance_to_thesis": "high",
                        "relevance_justification": "it is the result",
                        "recommended_usage": "supporting",
                        "key_finding_for_paper": "SCORE is 6.0",
                    }
                ],
                "ranking_rationale": "only one analysis",
            },
        ),
        Role.PAPER_WRITER: PAPER_WRITER,
        Role.DRAFT_GAP_ANALYST: answer(
            DraftGapsAnswer,
            {"memo": "# Gaps\nNone.", "gaps": {"status": "no_gaps", "summary": "adequate"}},
        ),
        Role.EVIDENCE_AUDITOR: answer(
            AuditPayload, {"verdict": "pass", "summary": "Claims match.", "findings": []}
        ),
        Role.AI_REVIEWER: answer(ReviewerAnswer, {"thought": "notes", "review": REVIEW}),
        Role.AREA_CHAIR: answer(ReviewerAnswer, {"thought": "meta", "review": REVIEW}),
        Role.REVISION_WRITER: REVISION_WRITER,
    }


def scripted_models(copies: int = 200) -> ScriptedModels:
    """Scripted models: each role replays its policy for up to ``copies`` model calls."""
    return ScriptedModels.from_scripts(
        {role: [policy] * copies for role, policy in policies().items()},
        context_window=200_000,
    )


# --- the fakes and the context ----------------------------------------------------------------


def evaluator() -> FakeEvaluator:
    """Baseline 1.0; every other program 6.0."""
    return FakeEvaluator(
        default=lambda request: completed_result(
            request, combined_score=1.0 if request.evaluation_id.endswith("baseline") else 6.0
        )
    )


def final_evaluator() -> FakeFinalEvaluator:
    """A sealed final evaluation that qualifies the frozen candidate."""
    return FakeFinalEvaluator()


def references() -> FakeRetriever:
    """Ten distinct records for every query (the paper needs at least eight)."""
    records = [
        ReferenceRecord(
            key=f"ref{number}",
            title=f"Reference {number}",
            bibtex=f"@article{{ref{number},\n  title={{Reference {number}}},\n  year={{2020}}\n}}",
            source="semantic_scholar",
            provenance={"id": f"ref.s2.ref{number}", "paper_id": f"p{number}"},
        )
        for number in range(10)
    ]
    return FakeRetriever(default=records)


def config(**updates: Any) -> ScientistConfig:
    """The method's configuration for the scripted run: one idea cycle, small budgets."""
    values: dict[str, Any] = {
        "phases": ALL_PHASES,
        "agents": AgentsConfig(
            model_retry=ModelRetryConfig(
                max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0
            )
        ),
        "investigation": InvestigationConfig(max_idea_cycles=1, harness_evaluated_attempts=True),
        "variants": VariantsConfig(max_evals_per_variant=1),
        "posthoc": PosthocConfig(checkpoint_mode="optional", max_selected_analyses=1),
        "paper": PaperConfig(max_repair_rounds=2),
        "review": ReviewConfig(num_reviewers=2, fewshot=False),
    }
    values.update(updates)
    return ScientistConfig(**values)


def context(
    workspace: LocalWorkspace,
    *,
    models: ScriptedModels | None = None,
    evaluator_: FakeEvaluator | None = None,
    final_: FakeFinalEvaluator | None = None,
    hooks: Hooks | None = None,
    **updates: Any,
) -> ScientistContext:
    """A context over ``workspace`` with the scripted roles and the fakes; ``updates`` change
    the configuration."""
    return ScientistContext(
        hooks=hooks or Hooks(),
        models=models or scripted_models(),
        workspace=workspace,
        evaluator=evaluator_ or evaluator(),
        final_evaluator=final_ or final_evaluator(),
        retriever=references(),
        paper_toolchain=FakePaperToolchain(pdf_text=PAPER_TEXT),
        config=config(**updates),
    )


async def write_task(workspace: LocalWorkspace) -> str:
    """Write the minimal task, with a scaffold, where the pipeline reads it."""
    await write_minimal_task(workspace, TASK_DIR)
    await workspace.write_bytes(f"{TASK_DIR}/capture.py", b"# capture\n")
    # the sealed final evaluator (the fake final evaluator stands in for running it)
    await workspace.write_bytes(f"{TASK_DIR}/evaluate_final.py", MINIMAL_EVALUATOR.encode("utf-8"))
    return TASK_DIR


def unfilled_placeholders(models: ScriptedModels) -> dict[str, set[str]]:
    """The ``{{KEY}}`` placeholders left in any role's prompt, by role."""
    found: dict[str, set[str]] = {}
    for role in policies():
        for call in models.scripted(role).calls:
            leftovers = set(UNFILLED.findall(prompt_of(call)))
            if leftovers:
                found.setdefault(str(role), set()).update(leftovers)
    return found
