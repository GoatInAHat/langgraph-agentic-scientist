"""Deterministic evaluation ids.

Every evaluation is requested under an id that the graph derives from the run and the position of
the evaluation in the method (worker, cycle, program, plane). A resumed node therefore asks again
under the same id, and the evaluator's idempotence (see
:mod:`~langgraph_agentic_scientist.evaluation.protocol`) turns the second request into a lookup.
"""

from __future__ import annotations

import re

_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]*")


def _part(value: object) -> str:
    text = str(value)
    if not _PART.fullmatch(text):
        raise ValueError(f"unsafe evaluation id component: {text!r}")
    return text


def evaluation_id(run_id: str, *parts: object) -> str:
    """Join a run id and components into an evaluation id, refusing unsafe components."""
    return "/".join([_part(run_id), *(_part(part) for part in parts)])


def campaign_baseline_id(run_id: str) -> str:
    """Id of the campaign's one baseline evaluation (the harness incumbent)."""
    return evaluation_id(run_id, "baseline")


def worker_baseline_id(run_id: str, worker: int) -> str:
    """Id of the harness incumbent's baseline evaluation of worker ``worker``."""
    return evaluation_id(run_id, f"w{worker}", "baseline")


def candidate_id(run_id: str, worker: int, cycle: int, program: str) -> str:
    """Id of the evaluation of ``program`` in cycle ``cycle`` of worker ``worker``."""
    return evaluation_id(run_id, f"w{worker}", f"c{cycle:03d}", program)


def final_id(run_id: str) -> str:
    """Id of the sealed final evaluation of a run."""
    return evaluation_id(run_id, "final")


def id_to_path(evaluation_id_value: str) -> str:
    """Return a workspace-safe directory name for an evaluation id."""
    parts = evaluation_id_value.split("/")
    return "/".join(_part(part) for part in parts)


__all__ = [
    "campaign_baseline_id",
    "candidate_id",
    "evaluation_id",
    "final_id",
    "id_to_path",
    "worker_baseline_id",
]
