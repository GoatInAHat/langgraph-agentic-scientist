"""Stage 3: select analysis candidates (upstream ``select.py``).

Two steps. ``select`` runs role R25 (the analysis selector, structured, read-only) and fails open:
a selector that gives no answer accepts every candidate, as does a candidate the selector did not
review. ``apply_constraints`` then applies upstream's hard constraints in code, whatever the
selector said: ablation families, ``requires_retrain`` and plans that imply retraining or an
evaluator rerun are rejected, required artifacts must exist in the analysis-ready bundle, ids and
(family, title) pairs are deduplicated, one analysis per family, at most
``max_selected_analyses``. The result is written as ``generation/posthoc_selection.json``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langgraph.runtime import Runtime

from ..agents.runner import run_structured_role
from ..context import ScientistContext
from ..core.errors import WorkspacePathError
from ..core.state import Event
from ..models import Role
from ..workspace.paths import join, safe_relpath
from ..workspace.protocol import Workspace
from .common import read_text_or_empty
from .layout import Layout
from .models import (
    AnalysisCandidate,
    AnalysisSelection,
    PosthocState,
    RejectedCandidate,
    SelectionAnswer,
)
from .settings import posthoc_config

PHASE = "posthoc"
BLOCKED_FAMILY_TOKENS = ("ablation",)
RETRAIN_PLAN_TOKENS = (
    "retrain",
    "train from scratch",
    "fine-tun",
    "finetun",
    "invoke evaluate.py",
    "rerun evaluation",
    "run evaluation",
)


# --- hard constraints ---------------------------------------------------------------------------


async def artifact_exists(workspace: Workspace, bundle_dir: str, artifact: str) -> bool:
    """Return whether a required artifact exists in the bundle.

    The name is tried as given, then below ``program_0/`` and ``reference_program/``; a path
    through a symlink or outside the bundle never exists.
    """
    normalized = artifact.strip()
    if normalized.startswith("task_bundle/"):
        normalized = normalized[len("task_bundle/") :]
    if not normalized or normalized.startswith("/") or ".." in normalized.split("/"):
        return False
    names = [normalized]
    if not normalized.startswith(("program_0/", "reference_program/")):
        names.extend([f"program_0/{normalized}", f"reference_program/{normalized}"])
    for name in names:
        try:
            relative = safe_relpath(name, allow_root=False)
        except WorkspacePathError:
            continue
        current = ""
        safe = True
        for part in relative.split("/"):
            current = f"{current}/{part}" if current else part
            entry = await workspace.stat(join(bundle_dir, current))
            if entry is not None and entry.kind == "symlink":
                safe = False
                break
        if safe and await workspace.stat(join(bundle_dir, relative)) is not None:
            return True
    return False


async def feasibility_error(
    workspace: Workspace,
    *,
    candidate: AnalysisCandidate,
    bundle_dir: str,
    bootstrap_ready: bool,
) -> str | None:
    """Return why a candidate is not feasible for the analysis-only phase, or ``None``."""
    family_key = candidate.family.strip().lower()
    if any(token in family_key for token in BLOCKED_FAMILY_TOKENS):
        return "post-hoc analysis pipeline excludes ablations; use the dedicated ablation pipeline"

    if candidate.requires_retrain:
        if not bootstrap_ready:
            return (
                "post-hoc analysis pipeline is analysis-only and bootstrap is not relevant for "
                "retraining candidates"
            )
        return (
            "post-hoc analysis pipeline is analysis-only; candidates must not require retraining "
            "or evaluator reruns"
        )

    plan_text = " ".join(candidate.implementation_plan).lower()
    if any(token in plan_text for token in RETRAIN_PLAN_TOKENS):
        return (
            "post-hoc analysis pipeline is analysis-only; use existing artifacts rather than "
            "retraining or evaluator reruns"
        )

    missing_artifacts = [
        artifact
        for artifact in candidate.required_artifacts
        if not await artifact_exists(workspace, bundle_dir, artifact)
    ]
    if missing_artifacts:
        return f"missing required artifacts: {missing_artifacts}"
    return None


async def apply_selection_constraints(
    workspace: Workspace,
    *,
    candidates: list[AnalysisCandidate],
    llm_decisions: Mapping[str, tuple[bool, str]],
    max_selected_analyses: int,
    bundle_dir: str,
    bootstrap_ready: bool,
) -> AnalysisSelection:
    """Apply the hard constraints to the selector's decisions (upstream rules)."""
    accepted: list[AnalysisCandidate] = []
    rejected: list[RejectedCandidate] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str]] = set()
    seen_families: set[str] = set()

    for candidate in candidates:
        accepted_by_llm, reason = llm_decisions.get(candidate.analysis_id, (True, "accepted"))
        if not accepted_by_llm:
            rejected.append(RejectedCandidate(candidate=candidate, reason=reason))
            continue

        error = await feasibility_error(
            workspace, candidate=candidate, bundle_dir=bundle_dir, bootstrap_ready=bootstrap_ready
        )
        if error is not None:
            rejected.append(RejectedCandidate(candidate=candidate, reason=error))
            continue

        if candidate.analysis_id in seen_ids:
            rejected.append(RejectedCandidate(candidate=candidate, reason="duplicate analysis_id"))
            continue
        seen_ids.add(candidate.analysis_id)

        dedup_key = (candidate.family.strip().lower(), candidate.title.strip().lower())
        if dedup_key in seen_keys:
            rejected.append(
                RejectedCandidate(candidate=candidate, reason="duplicate family/title proposal")
            )
            continue
        seen_keys.add(dedup_key)

        if candidate.family.strip().lower() in seen_families:
            rejected.append(
                RejectedCandidate(candidate=candidate, reason="family already represented")
            )
            continue

        if len(accepted) >= max_selected_analyses:
            rejected.append(
                RejectedCandidate(candidate=candidate, reason="max_selected_analyses cap reached")
            )
            continue

        seen_families.add(candidate.family.strip().lower())
        accepted.append(candidate)

    return AnalysisSelection(accepted=accepted, rejected=rejected)


def selection_payload(selection: AnalysisSelection, max_selected_analyses: int) -> dict[str, Any]:
    """Return upstream's ``posthoc_selection.json`` content."""
    return {
        "schema_version": 1,
        "max_selected_analyses": max_selected_analyses,
        "accepted": [candidate.to_dict() for candidate in selection.accepted],
        "rejected": [
            {**item.candidate.to_dict(), "reason": item.reason} for item in selection.rejected
        ],
    }


# --- the selector -------------------------------------------------------------------------------


def decisions_from_answer(
    answer: SelectionAnswer, candidates: list[AnalysisCandidate]
) -> dict[str, tuple[bool, str]]:
    """Map the selector's decisions by id; a candidate it did not review is accepted."""
    decisions = {item.id: (item.accepted, item.reason) for item in answer.decisions if item.id}
    for candidate in candidates:
        decisions.setdefault(candidate.analysis_id, (True, "accepted (not reviewed by selector)"))
    return decisions


async def select_analyses(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Run the analysis selector (role R25); fail open when it gives no answer."""
    ctx = runtime.context
    workspace = ctx.workspace
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    candidates = list(state.get("candidates") or [])
    if not candidates:
        return {"llm_decisions": {}}
    values = {
        "TASK_MD": await read_text_or_empty(workspace, join(layout.ready, "task.md")),
        "BOOTSTRAP_REPORT": await read_text_or_empty(workspace, layout.bootstrap_report),
        "CANDIDATES_JSON": json.dumps([candidate.to_dict() for candidate in candidates], indent=2),
    }
    prompt = ctx.prompts.render("posthoc/posthoc_select", values, fragments=False)
    outcome = await run_structured_role(
        role=Role.ANALYSIS_SELECTOR,
        schema=SelectionAnswer,
        runtime=runtime,
        root=layout.generation,
        read_only=True,
        prompt=prompt,
        timeout=config.selection_timeout_seconds,
    )
    answer = outcome.structured
    if not isinstance(answer, SelectionAnswer):
        decisions = {
            candidate.analysis_id: (True, "accepted (selector unavailable)")
            for candidate in candidates
        }
    else:
        decisions = decisions_from_answer(answer, candidates)
    return {
        "llm_decisions": {
            key: {"accepted": accepted, "reason": reason}
            for key, (accepted, reason) in decisions.items()
        }
    }


async def apply_constraints(
    state: PosthocState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Apply the hard constraints and write ``generation/posthoc_selection.json``."""
    workspace = runtime.context.workspace
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    candidates = list(state.get("candidates") or [])
    bootstrap = state.get("bootstrap")
    decisions = {
        key: (bool(value["accepted"]), str(value["reason"]))
        for key, value in (state.get("llm_decisions") or {}).items()
    }
    selection = await apply_selection_constraints(
        workspace,
        candidates=candidates,
        llm_decisions=decisions,
        max_selected_analyses=config.max_selected_analyses,
        bundle_dir=layout.ready,
        bootstrap_ready=bool(bootstrap.bootstrap_ready) if bootstrap is not None else False,
    )
    await workspace.write_bytes(
        join(layout.generation, "posthoc_selection.json"),
        json.dumps(selection_payload(selection, config.max_selected_analyses), indent=2).encode(
            "utf-8"
        ),
    )
    return {
        "selection": selection,
        "events": [
            Event(
                kind="posthoc_selection",
                phase=PHASE,
                data={"accepted": len(selection.accepted), "rejected": len(selection.rejected)},
            )
        ],
    }


__all__ = [
    "BLOCKED_FAMILY_TOKENS",
    "RETRAIN_PLAN_TOKENS",
    "apply_constraints",
    "apply_selection_constraints",
    "artifact_exists",
    "decisions_from_answer",
    "feasibility_error",
    "select_analyses",
    "selection_payload",
]
