"""A checkpoint of the pipeline's state survives strict msgpack with ``serde_allowlist()``."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]


def test_an_interrupted_run_resumes_through_a_strict_serializer(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    shutil.copytree(
        REPO_ROOT / "examples" / "circpack",
        workspace / "inputs" / "circpack",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    env = {
        **os.environ,
        "LANGGRAPH_STRICT_MSGPACK": "true",
        "PYTHONPATH": os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]),
    }
    done = subprocess.run(
        [sys.executable, str(HERE / "strict_msgpack_run.py"), str(workspace), "inputs/circpack"],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    summary = json.loads(done.stdout.strip().splitlines()[-1])
    assert summary["status"] == "completed"
    phases = ("investigation", "ablations", "final_evaluation", "posthoc", "paper", "review")
    assert summary["calls"] == {name: 1 for name in (*phases, "revision")}
