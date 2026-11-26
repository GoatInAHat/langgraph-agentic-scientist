"""The review and revision graphs checkpoint and resume under strict msgpack with the allowlist."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("pymupdf")

TESTS = Path(__file__).resolve().parents[1]

SCRIPT = textwrap.dedent(
    """
    import asyncio, sys, tempfile
    from pathlib import Path

    sys.path[:0] = [sys.argv[1] + "/revision", sys.argv[1]]
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde import _msgpack
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from revision_helpers import StubPaper, answer, review_dict, write_original

    from langgraph_agentic_scientist import (
        GraphOverrides, Hooks, LocalWorkspace, ReviewConfig, ScientistContext, serde_allowlist,
    )
    from langgraph_agentic_scientist.models import Role
    from langgraph_agentic_scientist.review import build_review_graph
    from langgraph_agentic_scientist.revision import build_revision_graph
    from langgraph_agentic_scientist.testing import ScriptedModels
    from testkit import fast_config

    assert _msgpack.STRICT_MSGPACK_ENABLED, "run with LANGGRAPH_STRICT_MSGPACK=true"


    def saver():
        return InMemorySaver(
            serde=JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist())
        )


    async def main():
        root = Path(tempfile.mkdtemp())
        workspace = LocalWorkspace(root / "ws")
        write_original(workspace.root)
        crashes = {"left": 1}

        async def crash_once(state, runtime):
            if crashes["left"]:
                crashes["left"] -= 1
                raise RuntimeError("crash")

        models = ScriptedModels.from_scripts(
            {Role.AI_REVIEWER: [answer()] * 2, Role.AREA_CHAIR: [answer()] * 2},
            context_window=200_000,
        )
        ctx = ScientistContext(
            models=models,
            workspace=workspace,
            config=fast_config(review=ReviewConfig(num_reviewers=1)),
            hooks=Hooks(before={"revision.second_review": [crash_once]}),
        )
        stub = StubPaper(workspace.root)

        async def hashes(directory, runtime):
            workspace = runtime.context.workspace
            data = await workspace.read_bytes(directory + "/paper_evidence_audit.json")
            return __import__("json").loads(data)["input_sha256"]

        async def scope(directory, runtime):
            return True

        graph = build_revision_graph(
            checkpointer=saver(),
            overrides=GraphOverrides(
                slots={
                    "revision.paper": stub,
                    "revision.audit_input_hashes": hashes,
                    "revision.scope_audit": scope,
                }
            ),
        )
        config = {"configurable": {"thread_id": "strict"}}
        try:
            await graph.ainvoke({"paper_dir": "paper"}, config, context=ctx)
        except RuntimeError:
            pass
        out = await graph.ainvoke(None, config, context=ctx)
        assert out["result"].status == "promoted", out["result"]
        assert len(stub.calls) == 1

        review = build_review_graph(checkpointer=saver())
        out = await review.ainvoke(
            {"pdf_path": "p/p.pdf", "paper_text": "x" * 200},
            {"configurable": {"thread_id": "review"}},
            context=ctx,
        )
        assert out["result"].status == "completed", out["result"]
        print("ok")


    asyncio.run(main())
    """
)


def test_the_graphs_run_and_resume_under_strict_msgpack() -> None:
    env = {**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true"}
    completed = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(TESTS)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
        cwd=TESTS.parent,
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "ok" in completed.stdout
