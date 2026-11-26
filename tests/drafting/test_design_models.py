"""The records of the design loop: JSON shape, aliases, checkpoint serialisation."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import ValidationError
from test_design_support import (
    BUDGET,
    TASK_MD,
    delta_payload,
    design_payload,
    judge,
    make_design,
    patch_payload,
    patcher,
    regression,
    review_payload,
)

from langgraph_agentic_scientist import serde_allowlist
from langgraph_agentic_scientist.drafting.design_models import (
    DESIGN_DIMENSIONS,
    Capacity,
    DeltaAudit,
    DesignLedgerEntry,
    DesignPatch,
    DesignProgress,
    DesignReview,
    RepairMemory,
    TaskDesign,
    canonical_json,
)
from langgraph_agentic_scientist.drafting.graph_design import build_task_design_graph
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.testing import ScriptedModels, minimal_task_ref, structured_call


def test_the_dimensions_are_upstreams_eleven_in_order() -> None:
    assert len(DESIGN_DIMENSIONS) == 11
    assert DESIGN_DIMENSIONS[0] == "goal_and_method_fidelity"
    assert DESIGN_DIMENSIONS[-1] == "publication_scope"


def test_a_design_dumps_to_upstreams_json_keys() -> None:
    design = make_design()
    assert design.payload() == design_payload()
    schema = TaskDesign.model_json_schema(by_alias=True)
    assert "schema_version" in schema["properties"] and "class_name" not in json.dumps(schema)
    assert set(schema["properties"]) == {
        "schema_version",
        "canonical_task",
        "method",
        "upstream",
        "candidate",
        "planes",
        "feedback",
        "publication",
    }
    # the field that clashes with a Python keyword is "class" in the schema the designer sees
    assert "class" in schema["$defs"]["DesignMethod"]["properties"]


def test_a_design_by_python_names_and_by_aliases_is_the_same_design() -> None:
    by_alias = TaskDesign.model_validate(design_payload())
    method = by_alias.method
    assert method.class_name == "learned optimization"
    renamed = design_payload()
    renamed["method"] = {"class_name": "learned optimization", "claim": "Improve held-out loss."}
    assert TaskDesign.model_validate(renamed).fingerprint() == by_alias.fingerprint()


def test_unknown_fields_and_wrong_types_are_refused_by_the_schema() -> None:
    payload = design_payload()
    payload["unexpected"] = 1
    with pytest.raises(ValidationError):
        TaskDesign.model_validate(payload)
    payload = design_payload()
    payload["planes"]["search"]["budget"]["training_steps"] = True
    with pytest.raises(ValidationError):
        TaskDesign.model_validate(payload)
    payload = design_payload()
    payload["candidate"]["state_semantics"] = 7
    with pytest.raises(ValidationError):
        TaskDesign.model_validate(payload)


def test_the_canonical_json_is_sorted_and_compact() -> None:
    assert canonical_json({"b": 1, "a": [1, "é"]}) == '{"a":[1,"é"],"b":1}'


def test_a_review_names_its_failed_dimensions_in_canonical_order() -> None:
    review = DesignReview.model_validate(
        review_payload("f" * 64, {"reproducibility": BUDGET, "compute_realism": BUDGET})
    )
    assert review.failed_dimensions == ("compute_realism", "reproducibility")
    assert review.payload()["schema_version"] == 3


def test_a_review_verdict_outside_pass_and_fail_is_refused() -> None:
    payload = review_payload("f" * 64)
    payload["verdict"] = "maybe"
    with pytest.raises(ValidationError):
        DesignReview.model_validate(payload)


def test_a_legacy_delta_regression_has_no_repair_paths_in_its_json() -> None:
    row = regression("reproducibility", "/method/claim")
    del row["repair_paths"]
    audit = DeltaAudit.model_validate(
        {
            **delta_payload("a" * 64, "b" * 64, fixed=[], regressions=[]),
            "schema_version": 1,
            "regressions": [row],
            "verdict": "fail",
        }
    )
    assert "repair_paths" not in audit.payload()["regressions"][0]
    assert audit.regressed_dimensions == ("reproducibility",)


def test_a_patch_keeps_any_json_value() -> None:
    patch = DesignPatch.model_validate(patch_payload("a" * 64, "/x", {"nested": [1, None, "s"]}))
    assert patch.payload()["operations"][0]["value"] == {"nested": [1, None, "s"]}
    assert patch.operations[0].op == "replace"


def test_progress_serialises_the_keys_upstream_writes() -> None:
    progress = DesignProgress(fixed=["a"], improved=True)
    assert set(progress.payload()) == {
        "fixed",
        "still_failed",
        "regressed",
        "improved",
        "candidate_findings",
    }
    after_delta = DesignProgress(consensus_fixed=[], consensus_regressed=[], audit_failure="x")
    assert {"consensus_fixed", "consensus_regressed", "delta_regressions", "audit_failure"} <= set(
        after_delta.payload()
    )


def test_the_repair_memory_is_versioned_json_and_the_ledger_line_is_sorted() -> None:
    assert RepairMemory().payload() == {"schema_version": 1, "constraints": []}
    entry = DesignLedgerEntry(event="rejected", payload={"fixed": ["a"], "improved": False})
    assert json.loads(entry.line()) == {"event": "rejected", "fixed": ["a"], "improved": False}
    assert entry.line().index('"event"') < entry.line().index('"fixed"')


def test_capacity_has_upstreams_defaults() -> None:
    assert Capacity().max_idea_cycles == 20 and Capacity().max_program_evals_per_attempt == 4
    with pytest.raises(ValidationError):
        Capacity(max_idea_cycles=0)


def test_every_record_is_allowlisted_and_survives_a_checkpoint_round_trip() -> None:
    serde = JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist())
    design = make_design()
    review = DesignReview.model_validate(
        review_payload(design.fingerprint(), {"compute_realism": BUDGET})
    )
    audit = DeltaAudit.model_validate(
        delta_payload(
            design.fingerprint(), design.fingerprint(), fixed=["compute_realism"], regressions=[]
        )
    )
    values: list[Any] = [
        design,
        review,
        DesignPatch.model_validate(patch_payload(design.fingerprint(), "/x", [1])),
        audit,
        DesignProgress(fixed=["a"], candidate_findings=[{"id": "a"}]),
        RepairMemory(constraints=[{"dimension": "a"}]),
        DesignLedgerEntry(event="reviewed", payload={"a": 1}),
        Capacity(max_idea_cycles=3),
    ]
    for value in values:
        loaded = serde.loads_typed(serde.dumps_typed(value))
        assert type(loaded) is type(value)
        assert loaded.model_dump(by_alias=True) == value.model_dump(by_alias=True)
    assert loaded == values[-1]


async def test_the_graph_checkpoints_with_the_allowlisted_serializer(
    make_context: Any, workspace: Any
) -> None:
    await workspace.write_bytes("run/task/task.md", TASK_MD.encode("utf-8"))
    design = make_design()
    models = ScriptedModels.from_scripts(
        {
            Role.TASK_DESIGNER: [structured_call(TaskDesign, design)],
            Role.DESIGN_JUDGE: [
                judge({"compute_realism": "/planes/search/budget/wall_clock_seconds"}),
                judge(),
                judge(),
            ],
            Role.DESIGN_PATCH_REPAIRER: [patcher("/planes/search/budget/wall_clock_seconds", 7200)],
        },
        context_window=200_000,
    )
    saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=serde_allowlist()))
    app = build_task_design_graph(checkpointer=saver)
    out = await app.ainvoke(
        {"task": minimal_task_ref("run/task")},
        {"configurable": {"thread_id": "strict"}},
        context=make_context(models=models),
    )
    assert out["status"] == "accepted"
    state = await app.aget_state({"configurable": {"thread_id": "strict"}})
    assert state.values["repair_base"].fingerprint() == out["design"].fingerprint()
