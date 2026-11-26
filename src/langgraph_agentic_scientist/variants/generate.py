"""Stage 2: variant candidate generation (upstream ``generate.py``).

The variant generator (a structured role, no skills) reads the task description, the extraction
manifest, the diff between the best program and the reference (cut at 50,000 characters) and the
evaluator source, and proposes zero or more baselines and ablations. Zero is valid. Ablations must
target innovations of the best program, which the prompt says and the reviewer checks.

The agent answers through structured output. The host screens the proposals the way upstream's
parser skipped malformed entries: an id that is not snake case, a reserved id or a duplicate
drops that candidate (with its reason recorded).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..prompts import PromptRegistry
from ..workspace.paths import join
from ..workspace.protocol import Workspace, is_dir
from .constants import (
    EXTRACTION_MANIFEST_FILENAME,
    MAX_VARIANT_ID_LENGTH,
    RESERVED_VARIANT_IDS,
    VARIANT_ID_RE,
)
from .models import (
    VariantBundleAnswer,
    VariantCandidate,
    candidate_from_proposal,
)
from .workspace_ops import read_text_or_none

logger = logging.getLogger(__name__)

GENERATE_PROMPT_KEY = "variants/variant_generate"
DIFF_TIMEOUT_SECONDS = 30
DIFF_OUTPUT_BYTES = 8 * 1024 * 1024


async def compute_program_diff(workspace: Workspace, bundle_dir: str, *, max_chars: int) -> str:
    """Return a unified diff between ``reference_program`` and ``program_0`` of a bundle.

    Port of upstream ``_compute_program_diff``: ``diff -ruN`` run in the bundle (through
    :meth:`Workspace.run`, with directory names as the headers instead of absolute paths), the
    text cut at ``max_chars`` with upstream's marker. A bundle without both programs gives
    ``(no reference program available for diff)``; no output gives ``(programs are identical)``;
    a failing or missing ``diff`` command gives ``(diff unavailable)``.
    """
    ref_dir = join(bundle_dir, "reference_program")
    best_dir = join(bundle_dir, "program_0")
    if not await is_dir(workspace, ref_dir) or not await is_dir(workspace, best_dir):
        return "(no reference program available for diff)"
    try:
        result = await workspace.run(
            ["diff", "-ruN", "reference_program", "program_0"],
            cwd=bundle_dir,
            timeout=DIFF_TIMEOUT_SECONDS,
            max_output_bytes=DIFF_OUTPUT_BYTES,
        )
    except Exception:
        return "(diff unavailable)"
    if result.timed_out:
        return "(diff unavailable)"
    diff_text = result.stdout
    if not diff_text:
        return "(programs are identical)"
    if len(diff_text) > max_chars:
        diff_text = diff_text[:max_chars] + "\n... (diff truncated)"
    return diff_text


async def build_generation_prompt(
    workspace: Workspace, prompts: PromptRegistry, bundle_dir: str, *, max_chars: int
) -> str:
    """Build the generator's prompt from the extracted bundle."""
    values = {
        "TASK_MD": await read_text_or_none(workspace, join(bundle_dir, "task.md")) or "",
        "EXTRACTION_MANIFEST": await read_text_or_none(
            workspace, join(bundle_dir, EXTRACTION_MANIFEST_FILENAME)
        )
        or "",
        "PROGRAM_DIFF": await compute_program_diff(workspace, bundle_dir, max_chars=max_chars),
        "EVALUATOR_SUMMARY": await read_text_or_none(workspace, join(bundle_dir, "evaluate.py"))
        or "",
    }
    return prompts.render(GENERATE_PROMPT_KEY, values, fragments=False)


@dataclass
class ScreenedCandidates:
    """The proposals the host kept and the ones it dropped, with the reason."""

    candidates: list[VariantCandidate]
    dropped: list[tuple[str, str]]


def screen_candidates(answer: VariantBundleAnswer) -> ScreenedCandidates:
    """Parse a generator's answer into candidates, skipping malformed entries.

    ``no_relevant_variants`` gives no candidates, whatever the list holds (upstream's rule). An
    id that does not match ``^[a-z][a-z0-9_]*$`` (at most 64 characters), a reserved id
    (``control_best``, ``baseline_reference``) or an id that already occurred is dropped.
    """
    if answer.decision == "no_relevant_variants":
        return ScreenedCandidates([], [])
    kept: list[VariantCandidate] = []
    dropped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for proposal in answer.candidates:
        variant_id = proposal.variant_id
        if not VARIANT_ID_RE.match(variant_id) or len(variant_id) > MAX_VARIANT_ID_LENGTH:
            dropped.append((variant_id, "malformed variant id"))
        elif variant_id in RESERVED_VARIANT_IDS:
            dropped.append((variant_id, "reserved variant id"))
        elif variant_id in seen:
            dropped.append((variant_id, "duplicate variant id"))
        else:
            seen.add(variant_id)
            kept.append(candidate_from_proposal(proposal))
    for variant_id, reason in dropped:
        logger.warning("Skipping malformed candidate entry %r: %s", variant_id, reason)
    return ScreenedCandidates(kept, dropped)


__all__ = [
    "GENERATE_PROMPT_KEY",
    "ScreenedCandidates",
    "build_generation_prompt",
    "compute_program_diff",
    "screen_candidates",
]
