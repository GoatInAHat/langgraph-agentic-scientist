"""The pipeline on a Postgres checkpointer under strict msgpack, killed and resumed.

Runs when ``LAS_TEST_POSTGRES_URL`` names a database the test may create its tables in (the
checkpointer's own tables, with a thread id made for the test); skipped otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

URL = os.environ.get("LAS_TEST_POSTGRES_URL")
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

pytestmark = pytest.mark.skipif(not URL, reason="LAS_TEST_POSTGRES_URL is not set")


def _run(mode: str, thread_id: str, workspace: Path) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "LANGGRAPH_STRICT_MSGPACK": "true",
        "PYTHONPATH": os.pathsep.join([str(HERE), os.environ.get("PYTHONPATH", "")]),
    }
    assert URL is not None
    return subprocess.run(
        [
            sys.executable,
            str(HERE / "postgres_run.py"),
            mode,
            URL,
            thread_id,
            str(workspace),
            "inputs/circpack",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_a_killed_run_resumes_from_postgres_with_strict_msgpack(tmp_path: Path) -> None:
    pytest.importorskip("langgraph.checkpoint.postgres")
    workspace = tmp_path / "ws"
    shutil.copytree(
        REPO_ROOT / "examples" / "circpack",
        workspace / "inputs" / "circpack",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    thread_id = f"las-pipeline-{uuid.uuid4().hex}"
    first = _run("first", thread_id, workspace)
    assert first.returncode == 9, first.stderr[-2000:]  # killed inside the post-hoc phase
    log = tmp_path / "calls.log"
    assert log.read_text().split() == ["investigation", "ablations", "final_evaluation", "posthoc"]
    second = _run("resume", thread_id, workspace)
    assert second.returncode == 0, second.stderr[-3000:]
    summary = json.loads(second.stdout.strip().splitlines()[-1])
    assert summary["status"] == "completed"
    assert set(summary["phases"].values()) <= {"completed", "skipped"}
    # nothing that had completed ran again: only the killed phase restarted
    assert log.read_text().split() == [
        "investigation",
        "ablations",
        "final_evaluation",
        "posthoc",
        "posthoc",
        "paper",
        "review",
        "revision",
    ]
