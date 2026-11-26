"""Checks of comparison inputs and candidates (upstream ``reference_preflight.py`` and the
candidate validation of ``host_eval.py``)."""

from __future__ import annotations

from ..core.errors import ProgramSpecError
from ..evaluation.program_spec import resolve_entrypoint
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .constants import BASELINE_REFERENCE_ID, EVALUATION_PENDING_FILENAME, EVALUATOR_ARTIFACTS
from .workspace_ops import read_text_or_none

REFERENCE_DOC_NAMES = (
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "reference_performance.md",
)


async def validate_reference_workspace(
    workspace: Workspace, task_bundle: str, variant_id: str, base_program: str
) -> None:
    """Reject external references whose required comparison inputs are absent.

    A baseline whose base program is an external reference (other than the reserved
    ``baseline_reference``) needs ``control_program/program_spec.toml`` to compare against and
    documentation in ``program_0/`` (a README or ``reference_performance.md``) for the
    performance check.

    Raises:
        ValueError: With upstream's message.
    """
    if base_program != "reference" or variant_id == BASELINE_REFERENCE_ID:
        return
    control_spec = await read_text_or_none(
        workspace, join(task_bundle, "control_program/program_spec.toml")
    )
    if control_spec is None or not control_spec.strip():
        raise ValueError(f"Reference {variant_id!r} requires control_program")
    for filename in REFERENCE_DOC_NAMES:
        text = await read_text_or_none(workspace, join(task_bundle, "program_0", filename))
        if text is not None and text.strip():
            return
    raise ValueError(
        f"Reference {variant_id!r} requires documentation in program_0/ "
        "(README or reference_performance.md) for the performance comparison"
    )


async def validate_candidate(
    workspace: Workspace, task_bundle: str, program_index: int
) -> tuple[str, str]:
    """Check a candidate before its evaluation; return ``(program_dir, entrypoint)``.

    The candidate directory must exist, hold no evaluator artifact and have a valid
    ``program_spec.toml`` with an entrypoint that is a regular file (upstream
    ``_validate_candidate``).

    Raises:
        ValueError: With upstream's message.
    """
    if program_index < 0:
        raise ValueError("program index must be non-negative")
    program_dir = join(task_bundle, f"program_{program_index}")
    entry = await workspace.stat(program_dir)
    if entry is None or entry.kind != "dir":
        raise ValueError(f"missing candidate directory: {program_dir}")
    for artifact in EVALUATOR_ARTIFACTS:
        if await workspace.stat(join(program_dir, artifact)) is not None:
            raise ValueError(
                f"candidate contains evaluator artifact: {join(program_dir, artifact)}"
            )
    if await workspace.stat(join(program_dir, EVALUATION_PENDING_FILENAME)) is not None:
        raise ValueError(
            "candidate contains evaluator artifact: "
            f"{join(program_dir, EVALUATION_PENDING_FILENAME)}"
        )
    try:
        entrypoint = await resolve_entrypoint(workspace, program_dir)
    except ProgramSpecError as exc:
        raise ValueError(f"candidate is not a valid program: {exc}") from exc
    return program_dir, entrypoint


__all__ = ["REFERENCE_DOC_NAMES", "validate_candidate", "validate_reference_workspace"]
