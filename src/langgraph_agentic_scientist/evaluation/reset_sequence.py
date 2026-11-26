"""Candidate reset-transcript validation shared across evaluation phases.

Port of upstream ``src/investigation/reset_sequence.py``. The diagnostic text is package data
(``prompts/workflow/evaluation/reset_sequence_diagnostic.md``), extracted from upstream's
f-string and rendered by the prompt registry.
"""

from __future__ import annotations

from ..prompts import PromptRegistry

RESET_SEQUENCE_DIAGNOSTIC_KEY = "workflow/evaluation/reset_sequence_diagnostic"


def declared_reset_sequence_matches(
    observed_reset_seeds: tuple[object, ...], declared_reset_seeds: tuple[int, ...]
) -> bool:
    """Accept the exact primary transcript, including repeats, and one bounded replay."""
    if not declared_reset_seeds:
        return False
    replay_sequence = (*declared_reset_seeds, declared_reset_seeds[0])
    return observed_reset_seeds in (declared_reset_seeds, replay_sequence)


def reduced_draft_reset_sequence_matches(
    observed_reset_seeds: tuple[object, ...], declared_reset_seeds: tuple[int, ...]
) -> bool:
    """Validate a reduced draft smoke transcript against its first seed."""
    return declared_reset_sequence_matches(observed_reset_seeds, declared_reset_seeds[:1])


def _safe_seed(value: object) -> int | str:
    if isinstance(value, int) and not isinstance(value, bool):
        if value.bit_length() <= 63:
            return value
        return f"<int bits={value.bit_length()}>"
    return f"<{type(value).__name__}>"


def reset_sequence_diagnostic(
    observed_reset_seeds: tuple[object, ...],
    expected_primary_reset_seeds: tuple[int, ...],
    *,
    context: str,
    prompts: PromptRegistry | None = None,
) -> str:
    """Describe a reset mismatch without rendering arbitrary candidate values."""
    expected: list[int | str] = [_safe_seed(value) for value in expected_primary_reset_seeds[:8]]
    if len(expected_primary_reset_seeds) > len(expected):
        expected.append("<truncated>")
    observed: list[int | str] = [_safe_seed(value) for value in observed_reset_seeds[:8]]
    if len(observed_reset_seeds) > len(observed):
        observed.append("<truncated>")
    replay_seed = (
        _safe_seed(expected_primary_reset_seeds[0]) if expected_primary_reset_seeds else "<none>"
    )
    registry = prompts or PromptRegistry.default()
    return registry.render(
        RESET_SEQUENCE_DIAGNOSTIC_KEY,
        {
            "CONTEXT": context,
            "EXPECTED": str(expected),
            "OBSERVED": str(observed),
            "COUNT": str(len(observed_reset_seeds)),
            "REPLAY_SEED": str(replay_seed),
        },
        fragments=False,
    )


__all__ = [
    "RESET_SEQUENCE_DIAGNOSTIC_KEY",
    "declared_reset_sequence_matches",
    "reduced_draft_reset_sequence_matches",
    "reset_sequence_diagnostic",
]
