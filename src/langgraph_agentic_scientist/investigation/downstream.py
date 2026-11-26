"""No-winner downstream selection (upstream ``pipeline/downstream_selection.py``).

When the investigation ends without a winner and ``no_winner_policy`` is ``best_available``, the
downstream phases need a program to study. The rules: among the completed search evaluations with
``correct=true`` and a finite score, choose by score (descending), then attempt order, then program
index, excluding the seeded ``program_0`` copies; a failed rubric does not exclude a program.
Without such a candidate use the retained baseline if it is valid and scored; without either, skip
the downstream phases explicitly. The chosen program is frozen under ``<run>/downstream/program``
and recorded in ``<run>/.downstream_selection.json``, which binds the digests of the source and the
frozen copy. A resume verifies the record and never selects again.
"""

from __future__ import annotations

import json
import math
import posixpath
import re
from typing import Any

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..context import GraphOverrides, ScientistContext
from ..core.errors import ScientistError
from ..core.results import DownstreamSelection, InvestigationResult
from ..core.runconfig import record_run_config
from ..core.state import BestManifest, EvaluationRecord, ProgramRef, TaskRef, utc_now
from ..core.steps import StepGraph
from ..evaluation.program_clone import clone_program
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .models import DownstreamInput, DownstreamOutput, DownstreamState

SELECTION_FILE = ".downstream_selection.json"
PROGRAM_RE = re.compile(r"program_([1-9][0-9]*)")


class StaleDownstreamSelection(ScientistError):
    """A recorded downstream selection no longer matches the files it binds."""


def _score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def _candidate_key(record: EvaluationRecord, order: int) -> tuple[float, int, int] | None:
    if record.plane != "search" or record.status != "completed" or record.correct is not True:
        return None
    score = _score(record.metrics.get("combined_score"))
    match = PROGRAM_RE.fullmatch(record.program.path.rsplit("/", 1)[-1])
    if score is None or match is None:
        return None
    return (-score, order, int(match.group(1)))


def select_fallback(investigation: InvestigationResult) -> DownstreamSelection:
    """Apply upstream's no-winner rules to an investigation's records.

    Candidates are the completed, correct, finitely scored search evaluations of programs
    ``program_1`` and above, ranked by score (descending), then by the order of their attempts,
    then by program index. Otherwise the retained baseline is used when the best manifest is
    correct and finitely scored. Otherwise the selection is ``skipped``.
    """
    if investigation.status != "completed_no_winner":
        return DownstreamSelection(
            status="skipped",
            reason=f"the investigation ended {investigation.status}, not completed_no_winner",
        )
    attempt_order = {
        attempt.attempt_id: index for index, attempt in enumerate(investigation.attempts)
    }
    ranked: list[tuple[tuple[float, int, int], EvaluationRecord]] = []
    for position, record in enumerate(investigation.evaluations):
        order = attempt_order.get(record.attempt_id or "", len(attempt_order) + position)
        key = _candidate_key(record, order)
        if key is not None:
            ranked.append((key, record))
    if ranked:
        key, chosen = min(ranked, key=lambda item: item[0])
        return DownstreamSelection(
            status="selected",
            program=chosen.program,
            source="candidate",
            evaluation_id=chosen.evaluation_id,
            combined_score=-key[0],
            reason="the best valid scored candidate of the search",
        )
    best: BestManifest | None = investigation.best
    if best is not None and best.correct is True and _score(best.combined_score) is not None:
        return DownstreamSelection(
            status="selected",
            program=ProgramRef(path=best.program_dir, entrypoint=best.entrypoint_path),
            source="baseline",
            evaluation_id=(
                investigation.baseline.evaluation_id if investigation.baseline else None
            ),
            combined_score=_score(best.combined_score),
            reason="no valid scored candidate; the retained baseline is valid and scored",
        )
    return DownstreamSelection(
        status="skipped", reason="no valid scored candidate and no valid scored baseline"
    )


def _run_dir(task: TaskRef) -> str:
    return posixpath.dirname(task.root.rstrip("/"))


async def _read_selection(ws: Workspace, path: str) -> dict[str, Any] | None:
    try:
        data = json.loads((await ws.read_bytes(path)).decode("utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise StaleDownstreamSelection(f"invalid downstream selection manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise StaleDownstreamSelection("invalid downstream selection manifest")
    return data


async def select_fallback_node(
    state: DownstreamState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Choose the fallback program (a pure function of the investigation's records)."""
    return {"proposal": select_fallback(state["investigation"])}


async def freeze_program(
    state: DownstreamState, runtime: Runtime[ScientistContext]
) -> dict[str, Any]:
    """Freeze the chosen program, or verify the frozen one on resume; never select again.

    The manifest is written last: it is the commit point, and a frozen tree without it is
    redone from the same source. A recorded selection whose frozen tree changed raises
    :class:`StaleDownstreamSelection`.
    """
    ws = runtime.context.workspace
    task = state["task"]
    run_dir = _run_dir(task)
    manifest_path = join(run_dir, SELECTION_FILE)
    frozen_path = join(run_dir, "downstream", "program")
    recorded = await _read_selection(ws, manifest_path)
    if recorded is not None:
        digest = await ws.digest(frozen_path) if await ws.exists(frozen_path) else None
        if digest != recorded.get("program_tree_sha256"):
            raise StaleDownstreamSelection(
                "invalid or stale downstream selection: the frozen snapshot changed"
            )
        return {"downstream": DownstreamSelection.model_validate(recorded["selection"])}
    proposal = state.get("proposal")
    assert proposal is not None
    if proposal.status != "selected" or proposal.program is None:
        return {"downstream": proposal}
    source = proposal.program
    await ws.remove(frozen_path)
    await clone_program(ws, src_dir=source.path, dst_dir=frozen_path)
    frozen = ProgramRef(
        path=frozen_path, entrypoint=source.entrypoint, digest=await ws.digest(frozen_path)
    )
    selection = proposal.model_copy(update={"program": frozen, "selection_path": manifest_path})
    manifest = {
        "schema_version": 1,
        "investigation_outcome": "completed_no_winner",
        "selection_kind": "non_winner_candidate" if proposal.source == "candidate" else "baseline",
        "source_program_dir": source.path,
        "source_tree_sha256": await ws.digest(source.path),
        "program_dir": frozen_path,
        "program_tree_sha256": frozen.digest,
        "selection": selection.model_dump(mode="json"),
        "frozen_at": utc_now(),
    }
    await ws.write_bytes(
        manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return {"downstream": selection}


def build_downstream_selection_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "downstream_selection",
) -> CompiledStateGraph[Any, ScientistContext, DownstreamInput, DownstreamOutput]:
    """Build the graph that selects and freezes the no-winner fallback program.

    Steps (hook keys ``downstream.<step>``): ``select_fallback`` and ``freeze_program``. The
    result is a :class:`~langgraph_agentic_scientist.core.results.DownstreamSelection`:
    ``selected`` with the frozen program, or ``skipped`` with the reason.

    Args:
        checkpointer: The checkpointer when run at the top level; ``None`` inherits the parent's.
        store: The store when run at the top level.
        overrides: Replacement slots and hooks' graph overrides.
        name: The compiled graph's name.
    """
    sg = StepGraph(
        "downstream",
        DownstreamState,
        input_schema=DownstreamInput,
        output_schema=DownstreamOutput,
        overrides=overrides,
    )
    sg.add_node("record_run_config", record_run_config)
    sg.step("select_fallback", select_fallback_node)
    sg.step("freeze_program", freeze_program)
    sg.add_edge(START, "record_run_config")
    sg.add_edge("record_run_config", "select_fallback")
    sg.add_edge("select_fallback", "freeze_program")
    sg.add_edge("freeze_program", END)
    return sg.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "SELECTION_FILE",
    "StaleDownstreamSelection",
    "build_downstream_selection_graph",
    "select_fallback",
]
