"""Stage 3: variant selection (upstream ``select.py``).

A separate model call (the variant reviewer, a structured role, no skills) reviews the proposals
and filters them by relevance, feasibility, single-change scope and evaluability. The selection
is **fail-open**: when the reviewer fails, times out or answers nothing, every candidate is
accepted; a candidate the reviewer did not mention is accepted too. Hard caps are enforced by the
host afterwards, in candidate order: at most ``max_ablations`` ablations and
``max_extra_baselines`` baselines.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence

from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .generate import compute_program_diff
from .models import (
    RejectedCandidate,
    VariantCandidate,
    VariantDecisionsAnswer,
    VariantSelection,
)
from .workspace_ops import read_text_or_none

logger = logging.getLogger(__name__)

SELECT_PROMPT_KEY = "variants/variant_select"


def candidates_json(candidates: Sequence[VariantCandidate]) -> str:
    """Return the candidates as the prompt shows them (upstream's field subset, indent 2)."""
    return json.dumps(
        [
            {
                "variant_id": c.variant_id,
                "title": c.title,
                "kind": c.kind,
                "hypothesis": c.hypothesis,
                "relevance_reason": c.relevance_reason,
                "feasibility_reason": c.feasibility_reason,
                "target_components": c.target_components,
                "implementation_plan": c.implementation_plan,
                "expected_effect": c.expected_effect,
            }
            for c in candidates
        ],
        indent=2,
    )


async def build_selection_prompt(
    workspace: Workspace,
    prompts: PromptRegistry,
    candidates: Sequence[VariantCandidate],
    bundle_dir: str,
    *,
    max_chars: int,
) -> str:
    """Build the reviewer's prompt (task, program diff and the proposed candidates)."""
    values = {
        "TASK_MD": await read_text_or_none(workspace, join(bundle_dir, "task.md")) or "",
        "PROGRAM_DIFF": await compute_program_diff(workspace, bundle_dir, max_chars=max_chars),
        "CANDIDATES_JSON": candidates_json(candidates),
    }
    return prompts.render(SELECT_PROMPT_KEY, values, fragments=False)


def fail_open_decisions(
    candidates: Sequence[VariantCandidate], reason: str
) -> dict[str, tuple[bool, str]]:
    """Accept every candidate with ``reason`` (the reviewer failed)."""
    return {candidate.variant_id: (True, reason) for candidate in candidates}


def decisions_from_answer(
    candidates: Sequence[VariantCandidate], answer: VariantDecisionsAnswer
) -> dict[str, tuple[bool, str]]:
    """Return the reviewer's decisions by variant id; unmentioned candidates are accepted."""
    decisions: dict[str, tuple[bool, str]] = {}
    for entry in answer.decisions:
        decisions[entry.variant_id] = (entry.accepted, entry.reason)
    for candidate in candidates:
        if candidate.variant_id not in decisions:
            decisions[candidate.variant_id] = (True, "accepted (not evaluated by agent)")
    return decisions


def apply_hard_caps(
    candidates: Sequence[VariantCandidate],
    llm_decisions: Mapping[str, tuple[bool, str]],
    *,
    max_ablations: int,
    max_extra_baselines: int,
) -> VariantSelection:
    """Apply the reviewer's decisions and enforce the hard caps (upstream ``_apply_hard_caps``)."""
    accepted: list[VariantCandidate] = []
    rejected: list[RejectedCandidate] = []
    ablation_count = 0
    baseline_count = 0
    for candidate in candidates:
        is_accepted, reason = llm_decisions.get(candidate.variant_id, (True, "no decision"))
        if not is_accepted:
            rejected.append(RejectedCandidate(candidate=candidate, reason=reason))
            continue
        if candidate.kind == "ablation":
            if ablation_count >= max_ablations:
                rejected.append(
                    RejectedCandidate(
                        candidate=candidate, reason=f"ablation cap reached ({max_ablations})"
                    )
                )
                continue
            ablation_count += 1
        elif candidate.kind == "baseline":
            if baseline_count >= max_extra_baselines:
                rejected.append(
                    RejectedCandidate(
                        candidate=candidate, reason=f"baseline cap reached ({max_extra_baselines})"
                    )
                )
                continue
            baseline_count += 1
        accepted.append(candidate)
    return VariantSelection(accepted=accepted, rejected=rejected, controls=["control_best"])


def selection_json(selection: VariantSelection) -> dict[str, object]:
    """Return the final ``variant_selection.json`` content (upstream ``_write_selection_json``)."""
    return {
        "controls": selection.controls,
        "accepted": [
            {
                "variant_id": c.variant_id,
                "title": c.title,
                "kind": c.kind,
                "hypothesis": c.hypothesis,
                "source": "custom" if c.description_body else "generated",
            }
            for c in selection.accepted
        ],
        "rejected": [
            {
                "variant_id": item.candidate.variant_id,
                "title": item.candidate.title,
                "kind": item.candidate.kind,
                "reason": item.reason,
            }
            for item in selection.rejected
        ],
        "num_accepted": len(selection.accepted),
        "num_rejected": len(selection.rejected),
    }


__all__ = [
    "SELECT_PROMPT_KEY",
    "apply_hard_caps",
    "build_selection_prompt",
    "candidates_json",
    "decisions_from_answer",
    "fail_open_decisions",
    "selection_json",
]
