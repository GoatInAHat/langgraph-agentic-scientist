"""Evidence of repair rounds and the repair context of the reference planner.

Upstream ``paper_diagnostics.py``. Every round of a planning or validation stage is archived under
``paper_diagnostics/<stage>/round_NNNN/attempt_NNN/`` without replacing the latest aliases or
earlier snapshots. For the reference plan, the context of a repair round (the allowed categories
and the previous artifacts, as untrusted data) is appended to the planner prompt, and two
consecutive rounds that produce the same invalid plan with the same errors end the stage
(:func:`record_reference_round`).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from typing import Any

from ...prompts import PromptRegistry
from ...workspace.paths import join
from ...workspace.protocol import Workspace
from .plan import VALID_QUERY_CATEGORIES

REPAIR_CONTEXT_KEY = "workflow/paper/reference_repair_context"
"""The registry key of the fixed text of the repair context."""

_STAGE_RE = re.compile(r"[a-z_]+")


async def _read_artifact(workspace: Workspace, root: str, name: str) -> bytes | None:
    entry = await workspace.stat(join(root, name))
    if name != name.rsplit("/", 1)[-1] or (entry is not None and entry.kind == "symlink"):
        raise ValueError(f"Unsafe diagnostic artifact: {name}")
    if entry is None or entry.kind != "file":
        return None
    return await workspace.read_bytes(join(root, name))


async def _stage_dir(workspace: Workspace, root: str, stage: str) -> str:
    if not _STAGE_RE.fullmatch(stage):
        raise ValueError("Invalid diagnostic stage")
    directory = root
    for part in ("paper_diagnostics", stage):
        directory = join(directory, part)
        entry = await workspace.stat(directory)
        if entry is not None and entry.kind == "symlink":
            raise ValueError(f"Diagnostic directory is a symlink: {directory}")
    return directory


async def archive_round(
    workspace: Workspace, root: str, stage: str, round_number: int, names: Sequence[str]
) -> str:
    """Copy current outputs without replacing latest aliases or earlier snapshots.

    Returns the path of the new ``attempt_NNN`` directory.
    """
    if round_number < 0:
        raise ValueError("Invalid diagnostic round")
    stage_dir = await _stage_dir(workspace, root, stage)
    round_dir = join(stage_dir, f"round_{round_number:04d}")
    entry = await workspace.stat(round_dir)
    if entry is not None and entry.kind == "symlink":
        raise ValueError("Diagnostic round directory is a symlink")
    attempt = 1
    while await workspace.exists(join(round_dir, f"attempt_{attempt:03d}")):
        attempt += 1
    destination = join(round_dir, f"attempt_{attempt:03d}")
    for name in names:
        data = await _read_artifact(workspace, root, name)
        if data is not None:
            await workspace.write_bytes(join(destination, name), data)
    return destination


async def reference_repair_context(
    workspace: Workspace, root: str, stage: str, *, prompts: PromptRegistry | None = None
) -> str:
    """Return the context appended to the planner prompt of a repair round.

    It names the allowed query categories and quotes the previous plan, memo and validation
    feedback of ``stage`` (``reference_query_plan`` or ``postdraft_reference_query_plan``) as data.
    """
    registry = prompts or PromptRegistry.default()
    fixed = registry.render(
        REPAIR_CONTEXT_KEY,
        {"ALLOWED_QUERY_CATEGORIES": ", ".join(sorted(VALID_QUERY_CATEGORIES))},
        fragments=False,
    )
    sections: list[str] = []
    for name in (f"{stage}.json", f"{stage}.md", f"{stage}_validation_feedback.json"):
        data = await _read_artifact(workspace, root, name)
        text = data.decode("utf-8", errors="replace") if data is not None else "<missing>"
        fence = "`" * max(3, max((len(run) + 1 for run in re.findall(r"`+", text)), default=3))
        sections.extend((f"\nPrevious artifact: {name}", fence + "text", text, fence))
    return fixed + "\n" + "\n".join(sections)


async def record_reference_round(
    workspace: Workspace, root: str, stage: str, round_number: int
) -> bool:
    """Archive a plan and detect consecutive identical invalid plan and error pairs.

    Returns True when this round repeats the previous one (the same artifacts and the same hard
    errors, both invalid): upstream ends the stage in that case.
    """
    feedback_name = f"{stage}_validation_feedback.json"
    names = [f"{stage}.json", f"{stage}.md", feedback_name, f"{stage}_validation_feedback.md"]
    snapshot = await archive_round(workspace, root, stage, round_number, names)
    feedback: dict[str, Any] = json.loads(
        (await _read_artifact(workspace, root, feedback_name) or b"{}").decode()
    )
    errors = feedback.get("hard_errors", [])
    content = [await _read_artifact(workspace, root, name) for name in names[:2]]
    signature = hashlib.sha256(
        json.dumps(
            {
                "artifacts": [data.hex() if data is not None else None for data in content],
                "errors": errors,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    stage_dir = await _stage_dir(workspace, root, stage)
    previous_path = join(stage_dir, "last_round.json")
    entry = await workspace.stat(previous_path)
    if entry is not None and entry.kind == "symlink":
        raise ValueError("Diagnostic state is a symlink")
    previous: dict[str, Any] = (
        json.loads((await workspace.read_bytes(previous_path)).decode())
        if entry is not None and entry.kind == "file"
        else {}
    )
    repeated = bool(
        round_number > 0
        and errors
        and previous.get("invalid")
        and previous.get("round") == round_number - 1
        and previous.get("signature") == signature
    )
    state = {
        "signature": signature,
        "invalid": bool(errors),
        "repeated": repeated,
        "round": round_number,
    }
    await workspace.write_bytes(
        join(snapshot, "diagnostic_state.json"), json.dumps(state, indent=2).encode()
    )
    await workspace.write_bytes(previous_path, json.dumps(state).encode())
    return repeated


__all__ = [
    "REPAIR_CONTEXT_KEY",
    "archive_round",
    "record_reference_round",
    "reference_repair_context",
]
