"""Task references of extra tasks (cross-task generalisation)."""

from __future__ import annotations

import hashlib
import json

from ..core.contract import load_task_contract
from ..core.errors import ProgramSpecError, TaskContractError
from ..core.render import task_sections
from ..core.state import ProgramRef, TaskRef
from ..evaluation.program_spec import resolve_entrypoint
from ..workspace.paths import join, safe_relpath
from ..workspace.protocol import Workspace, is_dir
from .constants import TASK_SNAPSHOT_METADATA_FILENAME
from .workspace_ops import read_text_or_none


async def task_ref_for(workspace: Workspace, root: str) -> TaskRef:
    """Build the :class:`~langgraph_agentic_scientist.TaskRef` of a task directory of the workspace.

    Reads ``task.md`` (its ``Goal`` and ``Rubrics`` sections and digest), the task contract when
    ``task_contract.json`` exists (a contracted task is harness-evaluated) and the seed program
    ``program_0``.

    Raises:
        ValueError: When the directory does not exist or holds no ``evaluate.py``.
        TaskContractError: When a contract exists and is invalid.
    """
    root = safe_relpath(root, allow_root=False)
    if not await is_dir(workspace, root):
        raise ValueError(f"task directory does not exist: {root}")
    task_md = await read_text_or_none(workspace, join(root, "task.md"))
    goal, rubrics = task_sections(task_md)
    contract = None
    if await workspace.stat(join(root, "task_contract.json")) is not None:
        contract = await load_task_contract(workspace, root)
    evaluator = contract.evaluators.search if contract is not None else "evaluate.py"
    if await workspace.stat(join(root, evaluator)) is None:
        raise ValueError(f"task directory holds no evaluator: {join(root, evaluator)}")
    seed: ProgramRef | None = None
    if await is_dir(workspace, join(root, "program_0")):
        try:
            seed = ProgramRef(
                path=join(root, "program_0"),
                entrypoint=await resolve_entrypoint(workspace, join(root, "program_0")),
            )
        except ProgramSpecError:
            seed = None
    return TaskRef(
        root=root,
        task_md_sha256=hashlib.sha256(task_md.encode("utf-8")).hexdigest() if task_md else None,
        goal=goal,
        rubrics=rubrics,
        contract=contract,
        seed_program=seed,
        harness_evaluated=contract is not None,
    )


async def primary_task_name(workspace: Workspace, root: str) -> str:
    """Return the name of the primary task: the source's name a run-local snapshot records.

    Upstream's pipeline passes the source directory's name as ``paths.task_name``; a snapshot
    directory (``<run>/task``) records it in its metadata. Without one the name is the last
    component of ``root``, then ``original``.
    """
    text = await read_text_or_none(workspace, join(root, TASK_SNAPSHOT_METADATA_FILENAME))
    if text is not None:
        try:
            recorded = json.loads(text)
        except json.JSONDecodeError:
            recorded = None
        name = recorded.get("task_name") if isinstance(recorded, dict) else None
        if isinstance(name, str) and name and "/" not in name and "\\" not in name:
            return name
    return root.rstrip("/").rsplit("/", 1)[-1] or "original"


__all__ = ["TaskContractError", "primary_task_name", "task_ref_for"]
