"""Small task scaffolds and records for tests.

:func:`write_minimal_task` writes a task that any workspace can hold and
:class:`~langgraph_agentic_scientist.evaluation.LocalCommandEvaluator` can run: a ``task.md``
with ``Goal`` and ``Rubrics`` sections, an ``evaluate.py`` that runs the program and writes
``metrics.json`` and ``correct.json``, and a ``program_0`` with upstream's default
``program_spec.toml``.
"""

from __future__ import annotations

import hashlib

from ..core.state import ProgramRef, TaskRef
from ..evaluation.program_spec import render_default_program_spec_toml
from ..workspace.protocol import Workspace

MINIMAL_TASK_MD = """# Minimal task

## Goal
Make `SCORE` in the program as large as possible while keeping it at most 10.

## Rubrics
- The program sets `SCORE` to a number.
- `combined_score` exceeds 5.
"""

MINIMAL_EVALUATOR = '''"""Evaluate a program that sets SCORE (correct when SCORE <= 10)."""

import argparse
import json
import os
import runpy

parser = argparse.ArgumentParser()
parser.add_argument("--program_path", required=True)
parser.add_argument("--results_dir", required=True)
parser.add_argument("--num_replications", type=int, default=1)
args = parser.parse_args()
namespace = runpy.run_path(args.program_path)
score = float(namespace.get("SCORE", 0.0))
os.makedirs(args.results_dir, exist_ok=True)
with open(os.path.join(args.results_dir, "metrics.json"), "w") as handle:
    json.dump({"combined_score": score}, handle)
with open(os.path.join(args.results_dir, "correct.json"), "w") as handle:
    json.dump({"correct": score <= 10}, handle)
print(f"combined_score={score}")
'''

MINIMAL_PROGRAM = "SCORE = 1.0\n"


async def write_minimal_task(
    workspace: Workspace, root: str = "task", *, program: str = MINIMAL_PROGRAM
) -> TaskRef:
    """Write the minimal task under ``root`` and return its :class:`TaskRef`."""
    await workspace.write_bytes(f"{root}/task.md", MINIMAL_TASK_MD.encode("utf-8"))
    await workspace.write_bytes(f"{root}/evaluate.py", MINIMAL_EVALUATOR.encode("utf-8"))
    await workspace.write_bytes(
        f"{root}/program_0/program_spec.toml", render_default_program_spec_toml().encode("utf-8")
    )
    await workspace.write_bytes(f"{root}/program_0/src/program.py", program.encode("utf-8"))
    return minimal_task_ref(root)


def minimal_task_ref(root: str = "task") -> TaskRef:
    """Return the :class:`TaskRef` of the minimal task at ``root``."""
    return TaskRef(
        root=root,
        task_md_sha256=hashlib.sha256(MINIMAL_TASK_MD.encode("utf-8")).hexdigest(),
        goal="Make `SCORE` in the program as large as possible while keeping it at most 10.",
        rubrics="- The program sets `SCORE` to a number.\n- `combined_score` exceeds 5.",
        seed_program=ProgramRef(path=f"{root}/program_0", entrypoint="src/program.py"),
    )


async def write_program(workspace: Workspace, program_dir: str, source: str) -> ProgramRef:
    """Write a program directory with the default spec and ``src/program.py``."""
    await workspace.write_bytes(
        f"{program_dir}/program_spec.toml", render_default_program_spec_toml().encode("utf-8")
    )
    await workspace.write_bytes(f"{program_dir}/src/program.py", source.encode("utf-8"))
    return ProgramRef(path=program_dir, entrypoint="src/program.py")


__all__ = [
    "MINIMAL_EVALUATOR",
    "MINIMAL_PROGRAM",
    "MINIMAL_TASK_MD",
    "minimal_task_ref",
    "write_minimal_task",
    "write_program",
]
