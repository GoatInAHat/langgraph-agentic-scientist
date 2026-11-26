"""Stage 2: propose analysis candidates (upstream ``generate.py``).

Role R24 (the analysis generator) is a structured role: it answers with a
:class:`~langgraph_agentic_scientist.posthoc.models.PosthocBundle` and the host writes it as
``generation/posthoc_bundle.json`` (and ``change_summary.json``), the files upstream's agent and
host produced. The agent reads the staged evidence under ``generation/evidence/`` through a
read-only backend. Upstream's tolerance is kept: a session that produces no answer yields no
candidates, an entry with an unsafe id is skipped, and list-like fields are normalised.
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.runtime import Runtime

from ..agents.runner import run_structured_role
from ..config import PosthocConfig
from ..context import ScientistContext
from ..core.state import Event, Failure
from ..models import Role
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .common import read_text_or_empty
from .extract import BundleValidationError, verify_staged_evidence
from .layout import Layout
from .models import AnalysisCandidate, PosthocBundle, PosthocState
from .settings import posthoc_config

PHASE = "posthoc"


async def stage_generation_evidence(
    workspace: Workspace, ready_dir: str, generation_dir: str, config: PosthocConfig
) -> None:
    """Copy the verified evidence of the analysis-ready bundle into the generation directory.

    Raises:
        RuntimeError: When the staged evidence is unsafe.
    """
    source = join(ready_dir, "evidence")
    entry = await workspace.stat(source)
    if entry is None:
        return
    if entry.kind != "dir":
        raise RuntimeError(f"Unsafe staged evidence directory: {source}")
    await verify_staged_evidence(workspace, ready_dir, config)
    destination = join(generation_dir, "evidence")
    await workspace.remove(destination)
    await workspace.copy_tree(source, destination)


async def build_generation_prompt(
    workspace: Workspace, ctx: ScientistContext, layout: Layout, evaluator: str
) -> str:
    """Render ``posthoc_generate.md`` from the files of the analysis-ready bundle."""
    ready = layout.ready
    values = {
        "TASK_MD": await read_text_or_empty(workspace, join(ready, "task.md")),
        "EXTRACTION_MANIFEST": await read_text_or_empty(
            workspace, join(ready, "extraction_manifest.json")
        ),
        "EVIDENCE_MANIFEST": await read_text_or_empty(
            workspace, join(ready, "evidence_manifest.json")
        ),
        "BOOTSTRAP_REPORT": await read_text_or_empty(workspace, layout.bootstrap_report),
        "EVALUATOR_SOURCE": await read_text_or_empty(workspace, join(ready, evaluator)),
        "RESULTS_JSON": await read_text_or_empty(
            workspace, join(ready, "program_0", "_results.json")
        ),
    }
    return ctx.prompts.render("posthoc/posthoc_generate", values, fragments=False)


def parse_posthoc_bundle(bundle: PosthocBundle) -> tuple[str, list[AnalysisCandidate]]:
    """Return the decision and the candidates of a bundle (upstream ``_parse_posthoc_bundle``).

    ``no_relevant_analyses`` yields none; otherwise every proposal with a safe id becomes a
    candidate.
    """
    if bundle.decision == "no_relevant_analyses":
        return bundle.decision, []
    candidates: list[AnalysisCandidate] = []
    for proposal in bundle.analyses:
        candidate = AnalysisCandidate.from_proposal(proposal)
        if candidate is not None:
            candidates.append(candidate)
    return bundle.decision, candidates


async def generate_candidates(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the analysis generator (role R24) and record its candidates."""
    ctx = runtime.context
    workspace = ctx.workspace
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    task = state["task"]
    evaluator = task.contract.evaluators.search if task.contract is not None else "evaluate.py"
    try:
        await stage_generation_evidence(workspace, layout.ready, layout.generation, config)
    except (BundleValidationError, RuntimeError) as exc:
        return {
            "phase_error": Failure(category="evidence_invalid", scope="task", detail=str(exc)),
            "events": [
                Event(kind="posthoc_evidence_invalid", phase=PHASE, data={"detail": str(exc)})
            ],
        }
    prompt = await build_generation_prompt(workspace, ctx, layout, evaluator)
    outcome = await run_structured_role(
        role=Role.ANALYSIS_GENERATOR,
        schema=PosthocBundle,
        runtime=runtime,
        root=layout.generation,
        read_only=True,
        prompt=prompt,
        timeout=config.generation_timeout_seconds,
    )
    bundle = outcome.structured
    if not isinstance(bundle, PosthocBundle):
        return {
            "candidates": [],
            "generation": {
                "decision": None,
                "num_candidates": 0,
                "exit_code": 124 if outcome.timed_out else 1,
            },
            "events": [
                Event(
                    kind="posthoc_generation_empty",
                    phase=PHASE,
                    data={"failure": outcome.failure.category if outcome.failure else None},
                )
            ],
        }
    await workspace.write_bytes(
        join(layout.generation, "posthoc_bundle.json"),
        json.dumps(bundle.model_dump(mode="json"), indent=2).encode("utf-8"),
    )
    decision, candidates = parse_posthoc_bundle(bundle)
    summary = {
        "decision": decision,
        "num_candidates": len(candidates),
        "exit_code": 0,
        "summary": bundle.summary,
        "agent_output": outcome.final_text,
        "prompt_path": None,
    }
    await workspace.write_bytes(
        join(layout.generation, "change_summary.json"),
        json.dumps(summary, indent=2).encode("utf-8"),
    )
    return {
        "candidates": candidates,
        "generation": summary,
        "events": [
            Event(
                kind="posthoc_candidates",
                phase=PHASE,
                data={"decision": decision, "count": len(candidates)},
            )
        ],
    }


__all__ = [
    "build_generation_prompt",
    "generate_candidates",
    "parse_posthoc_bundle",
    "stage_generation_evidence",
]
