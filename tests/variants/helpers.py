"""Scripts, fixtures and loaders shared by the variants tests (offline, scripted models)."""

from __future__ import annotations

import ast
import logging
import re
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langgraph_agentic_scientist import ScientistConfig, VariantsConfig
from langgraph_agentic_scientist.config import AgentsConfig, ModelRetryConfig
from langgraph_agentic_scientist.core.state import ProgramRef, TaskRef
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import (
    final,
    structured_call,
    tool_call,
    write_minimal_task,
    write_program,
)
from langgraph_agentic_scientist.variants.models import (
    VariantBundleAnswer,
    VariantDecisionsAnswer,
    VariantSummariesAnswer,
)
from langgraph_agentic_scientist.workspace import Workspace

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = REPO_ROOT / "third_party" / "ai-scientist-v3" / "src" / "variants_pipeline"

BEST_SOURCE = "SCORE = 3.0\n# ablate_me: reheating\n"
SEED_SOURCE = "SCORE = 1.0\n"


def variants_config(**updates: Any) -> ScientistConfig:
    """A configuration with no model-retry sleeps and the given variants settings."""
    agents = AgentsConfig(
        model_retry=ModelRetryConfig(max_retries=0, initial_delay=0, backoff_factor=0, max_delay=0)
    )
    return ScientistConfig(agents=agents, variants=VariantsConfig(**updates))


async def make_study(
    workspace: Workspace, *, best_source: str = BEST_SOURCE
) -> tuple[TaskRef, ProgramRef]:
    """Write a task (seed score 1.0) and a better program; return the task and the best program."""
    task = await write_minimal_task(workspace, "task", program=SEED_SOURCE)
    best = await write_program(workspace, "runs/best/program_1", best_source)
    return task, best


def _first_human(messages: Sequence[BaseMessage]) -> str:
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content
            return content if isinstance(content, str) else str(content)
    return ""


def _tool_messages(messages: Sequence[BaseMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, ToolMessage))


def cloning_editor(copies: int = 80) -> list[Callable[[list[BaseMessage]], AIMessage]]:
    """A script for the variant editor that clones the previous program and stops.

    The step reads the program indices from the prompt, so one script serves every variant the
    graph runs in parallel.
    """

    def step(messages: list[BaseMessage]) -> AIMessage:
        prompt = _first_human(messages)
        match = re.search(r"Create exactly \./program_(\d+)/ by cloning \./program_(\d+)/", prompt)
        assert match, prompt
        index, previous = match.groups()
        if _tool_messages(messages):
            return final("candidate created")
        return tool_call(
            "clone_program", {"from_dir": f"program_{previous}", "to_dir": f"program_{index}"}
        )

    return [step] * copies


def idle_verifier(copies: int = 80) -> list[Callable[[list[BaseMessage]], AIMessage]]:
    """A script for a verifier that never fills the scaffold (it stays invalid)."""
    return [lambda messages: final("I verified nothing")] * copies


def idle_editor(copies: int = 80) -> list[Callable[[list[BaseMessage]], AIMessage]]:
    """A script for an editor that never creates the program."""
    return [lambda messages: final("I did nothing")] * copies


def judging_verifier(
    *,
    passing: bool | set[int] = True,
    evidence: str = "checked",
    copies: int = 80,
) -> list[Callable[[list[BaseMessage]], AIMessage]]:
    """A script for the variant verifier that fills the host's scaffold.

    It edits ``pass`` and ``evidence`` of every rubric, and leaves ``overall_pass`` as the host
    wrote it (false), so a passing result proves the host recomputed it. ``passing`` is true for
    every program, false for none, or the set of program indices whose rubrics all pass.
    """

    def step(messages: list[BaseMessage]) -> AIMessage:
        prompt = _first_human(messages)
        match = re.search(r"evaluated \./program_(\d+)/", prompt)
        assert match, prompt
        path = f"/program_{match.group(1)}/rubric_status.json"
        done = _tool_messages(messages)
        index = int(match.group(1))
        passes = passing if isinstance(passing, bool) else index in passing
        if done == 0:
            return tool_call(
                "edit_file",
                {
                    "file_path": path,
                    "old_string": '"evidence": ""',
                    "new_string": f'"evidence": "{evidence}"',
                    "replace_all": True,
                },
            )
        if done == 1 and passes:
            return tool_call(
                "edit_file",
                {
                    "file_path": path,
                    "old_string": '"pass": false',
                    "new_string": '"pass": true',
                    "replace_all": True,
                },
            )
        return final("verified")

    return [step] * copies


def proposal(variant_id: str, kind: str = "ablation", **fields: Any) -> dict[str, Any]:
    """A candidate entry of the generator's bundle."""
    return {
        "variant_id": variant_id,
        "title": f"Title of {variant_id}",
        "kind": kind,
        "hypothesis": f"Hypothesis of {variant_id}",
        "relevance_reason": "relevant",
        "feasibility_reason": "feasible",
        "target_components": ["reheating"],
        "implementation_plan": ["remove the schedule"],
        "expected_effect": "lower score",
        **fields,
    }


def bundle_answer(*entries: dict[str, Any], decision: str = "propose_variants") -> AIMessage:
    """The generator's structured answer."""
    return structured_call(
        VariantBundleAnswer, {"decision": decision, "summary": "s", "candidates": list(entries)}
    )


def decisions_answer(*decisions: tuple[str, bool, str]) -> AIMessage:
    """The reviewer's structured answer."""
    return structured_call(
        VariantDecisionsAnswer,
        {
            "decisions": [
                {"variant_id": vid, "accepted": accepted, "reason": reason}
                for vid, accepted, reason in decisions
            ]
        },
    )


def summaries_answer(task_summary: str, **variants: str) -> AIMessage:
    """The README summariser's structured answer."""
    return structured_call(
        VariantSummariesAnswer, {"task_summary": task_summary, "variant_summaries": variants}
    )


def study_scripts(
    *,
    generator: Sequence[Any] = (),
    reviewer: Sequence[Any] = (),
    editor: Sequence[Any] | None = None,
    verifier: Sequence[Any] | None = None,
    summariser: Sequence[Any] = (),
    retrofit: Sequence[Any] | None = None,
) -> dict[Role, Sequence[Any]]:
    """Scripts for every role (defaults: no-op retrofit, cloning editor, passing verifier)."""
    return {
        Role.RETROFIT_AGENT: retrofit if retrofit is not None else [final("retrofitted")],
        Role.VARIANT_GENERATOR: generator,
        Role.VARIANT_REVIEWER: reviewer,
        Role.VARIANT_EDITOR: editor if editor is not None else cloning_editor(),
        Role.VARIANT_VERIFIER: verifier if verifier is not None else judging_verifier(),
        Role.README_SUMMARISER: summariser,
    }


# --- loading code of the upstream variants pipeline, to compare with it -----------------------


def load_upstream(
    filename: str, names: Sequence[str], namespace: Mapping[str, Any] | None = None
) -> types.SimpleNamespace:
    """Execute selected top-level definitions of a vendored upstream module.

    The upstream modules import the rest of upstream's package, which the repository does not
    carry. This loader parses the file, keeps the top-level functions and assignments called
    ``names`` and runs them in a namespace that holds ``namespace`` (stubs for what they use).
    """
    tree = ast.parse((UPSTREAM / filename).read_text(encoding="utf-8"))
    chosen: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            chosen.append(node)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in names for t in targets):
                chosen.append(node)
    missing = set(names) - {
        n.name if isinstance(n, ast.FunctionDef) else _target_name(n) for n in chosen
    }
    assert not missing, f"not found in {filename}: {missing}"
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.Module(body=[future, *chosen], type_ignores=[])
    ast.fix_missing_locations(module)
    scope: dict[str, Any] = {
        "__name__": "upstream_" + filename,
        "logger": logging.getLogger("upstream"),
        **(namespace or {}),
    }
    exec(compile(module, filename, "exec"), scope)
    return types.SimpleNamespace(**scope)


def _target_name(node: ast.stmt) -> str:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]  # type: ignore[attr-defined]
    for target in targets:
        if isinstance(target, ast.Name):
            return target.id
    return ""


async def make_variant_workspaces(
    ctx: Any,
    workspace: Workspace,
    task: TaskRef,
    best: ProgramRef,
    *,
    candidates: Sequence[Any] = (),
    max_evals: int = 2,
) -> tuple[list[Any], Any]:
    """Extract, retrofit and materialise variants without running a graph.

    Returns the specs and the :class:`RunVariantInput` builder's ingredients (run config, digest).
    """
    from langgraph_agentic_scientist.core.runconfig import build_run_config
    from langgraph_agentic_scientist.variants.digests import managed_evaluator_tree_digest
    from langgraph_agentic_scientist.variants.extract import extract_bundle
    from langgraph_agentic_scientist.variants.materialize import materialize_variant_workspaces
    from langgraph_agentic_scientist.variants.models import VariantSelection
    from langgraph_agentic_scientist.variants.retrofit import finish_retrofit, seal_and_copy

    bundle = await extract_bundle(
        workspace, task=task, output_dir="out", best=best, meta={"entrypoint_path": best.entrypoint}
    )
    report = await seal_and_copy(
        workspace,
        bundle_dir=bundle.extracted,
        output_bundle_dir="out/_retrofitted_bundle",
        task_source_dir=task.root,
    )
    await finish_retrofit(workspace, report, session_ok=True)
    specs = await materialize_variant_workspaces(
        workspace,
        ctx.prompts,
        bundle_dir=bundle.extracted,
        retrofitted_bundle_dir="out/_retrofitted_bundle",
        variants_run_dir="out",
        selection=VariantSelection(accepted=list(candidates)),
        agent="test",
        agent_model=None,
        max_evals_per_variant=max_evals,
    )
    extra = {
        "run_config": build_run_config(ctx),
        "digest": await managed_evaluator_tree_digest(workspace, task.root),
    }
    return specs, extra
