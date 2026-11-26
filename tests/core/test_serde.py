"""serde_allowlist(): every stored type is allowed, and state survives strict msgpack checkpoints."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

from langgraph_agentic_scientist import serde_allowlist
from langgraph_agentic_scientist.core.state import StateModel

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "core"


def _all_subclasses(base: type) -> set[type]:
    found: set[type] = set()
    pending = list(base.__subclasses__())
    while pending:
        cls = pending.pop()
        if cls not in found:
            found.add(cls)
            pending.extend(cls.__subclasses__())
    return found


def test_every_state_model_is_allowlisted() -> None:
    allowed = set(serde_allowlist())
    for cls in _all_subclasses(StateModel):
        if "<locals>" in cls.__qualname__:
            continue
        assert (cls.__module__, cls.__qualname__) in allowed, cls
    assert ("langgraph_agentic_scientist.models", "Role") in allowed


SCRIPT = textwrap.dedent(
    """
    import asyncio, json, sys
    from pathlib import Path
    from typing import Any, TypedDict
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.serde import _msgpack
    from langgraph.graph import START, StateGraph
    from langgraph.runtime import Runtime
    from langgraph.types import Command, interrupt
    from pydantic import BaseModel

    from langgraph_agentic_scientist import (
        AIReview, LocalWorkspace, PaperData, ScientistContext, serde_allowlist,
    )
    from langgraph_agentic_scientist.agents.runner import run_structured_role
    from langgraph_agentic_scientist.config import AgentsConfig, ModelRetryConfig, ScientistConfig
    from langgraph_agentic_scientist.core.contract import parse_task_contract
    from langgraph_agentic_scientist.core.results import GroundingSource, IdeaRecord
    from langgraph_agentic_scientist.core.state import (
        IdeaBundle, ProgramRef, RubricResult, RubricSpec, RubricStatus, RunConfigRecord, TaskRef,
    )
    from langgraph_agentic_scientist.evaluation import (
        EvaluationEvidence, EvaluationResult, FreezeRecord, SealedInputs,
    )
    from langgraph_agentic_scientist.models import Role
    from langgraph_agentic_scientist.testing import ScriptedModels, structured_call
    from langgraph_agentic_scientist.workspace import SnapshotRef

    assert _msgpack.STRICT_MSGPACK_ENABLED, "run with LANGGRAPH_STRICT_MSGPACK=true"

    class Verdict(BaseModel):
        verdict: str

    contract = parse_task_contract(json.loads(Path(sys.argv[1]).read_text()))
    review = {k: 3 for k in ("Originality", "Quality", "Clarity", "Significance", "Soundness",
              "Presentation", "Contribution", "Confidence")}
    review.update({"Summary": "s", "Strengths": ["a"], "Weaknesses": ["b"], "Questions": [],
                   "Limitations": [], "Ethical Concerns": False, "Overall": 6, "Decision": "Accept"})
    records = {
        "task": TaskRef(root="task", contract=contract, seed_program=ProgramRef(path="task/program_0")),
        "evaluation": EvaluationResult(
            evaluation_id="r/1", status="completed", metrics={"combined_score": 1.5, "detail": {"a": [1]}},
            correct=True,
            evidence=EvaluationEvidence.model_validate(
                {"schema_version": 1, "episodes": [{"seed": 1, "score": 1.0, "correct": True, "duration_seconds": 0.1}]}
            ),
        ),
        "rubric": RubricStatus(attempt_id="a", idea_id="i", overall_pass=False,
                               rubrics=[RubricResult.model_validate({"id": "r", "pass": True, "evidence": "e"})]),
        "paper": PaperData(generated_at="now", grounding_sources={
            "result.best": GroundingSource.model_validate({"type": "metric", "value": 2.0, "program_dir": "p"})},
            evidence_sources={"investigation": {"status": "included"}}),
        "review": AIReview.model_validate(review),
        "snapshot": SnapshotRef(snapshot_id="s", path="paper"),
        "freeze": FreezeRecord(program=ProgramRef(path="p"), program_tree_sha256="sha256:" + "0" * 64,
                               task_tree_sha256="t", evaluator_sha256="e", frozen_at="now",
                               sealed_inputs=SealedInputs(ref="draw-1", drawn_at="now")),
        "idea": IdeaRecord(idea_id="i", idea_dir="ideas/01_i", cycle=1, bundle=IdeaBundle(
            idea_id="i", title="t", hypothesis="h", implementation_plan=["p"],
            target_rubrics=[RubricSpec(id="r", description="d")])),
        "role": Role.DESIGN_JUDGE,
    }

    class State(TypedDict, total=False):
        records: dict[str, Any]
        verdict: Verdict | None
        answer: str

    async def hold(state: State, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        outcome = await run_structured_role(role=Role.DESIGN_JUDGE, schema=Verdict, runtime=runtime,
                                            root="judge", prompt="judge")
        answer = interrupt("resume me")
        return {"verdict": outcome.structured, "answer": answer}

    async def main() -> None:
        serde = JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist(Verdict))
        saver = InMemorySaver(serde=serde)
        graph = StateGraph(State, context_schema=ScientistContext)
        graph.add_node("hold", hold)
        graph.add_edge(START, "hold")
        app = graph.compile(checkpointer=saver)
        models = ScriptedModels.from_scripts(
            {Role.DESIGN_JUDGE: [structured_call(Verdict, {"verdict": "fine"})]}, context_window=100000)
        config = ScientistConfig(agents=AgentsConfig(model_retry=ModelRetryConfig(max_retries=0)))
        ctx = ScientistContext(models=models, workspace=LocalWorkspace(sys.argv[2]), config=config)
        cfg = {"configurable": {"thread_id": "strict"}}
        await app.ainvoke({"records": records}, cfg, context=ctx)
        snapshot = await app.aget_state(cfg)
        restored = snapshot.values["records"]
        for key, value in records.items():
            assert restored[key] == value, (key, restored[key])
            assert type(restored[key]) is type(value), (key, type(restored[key]))
        out = await app.ainvoke(Command(resume="ok"), cfg, context=ctx)
        assert out["verdict"] == Verdict(verdict="fine"), out["verdict"]
        assert out["answer"] == "ok"
        print("STRICT-OK")

    asyncio.run(main())
    """
)


def test_records_round_trip_under_strict_msgpack(tmp_path: Path) -> None:
    script = tmp_path / "strict_roundtrip.py"
    script.write_text(SCRIPT, encoding="utf-8")
    env = {**os.environ, "LANGGRAPH_STRICT_MSGPACK": "true"}
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(FIXTURES / "contract_task" / "task_contract.json"),
            str(tmp_path / "ws"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "STRICT-OK" in completed.stdout
