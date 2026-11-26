"""The contract functions of the design loop, ported from upstream's design tests.

Messages are upstream's: they travel back to the roles as the diagnostic of a retry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from test_design_support import (
    BUDGET,
    CLAIM,
    TASK_MD,
    delta_payload,
    design_payload,
    make_design,
    patch_payload,
    regression,
    review_payload,
)

from langgraph_agentic_scientist.drafting.design import (
    TaskDesignError,
    apply_design_patch,
    check_delta,
    compare_reviews,
    confirm_delta,
    decode_json_pointer,
    design_from_payload,
    enrich_progress,
    focus_repair_payload,
    focused_authorization,
    has_only_indirect_regressions,
    rebind_latent_failures,
    review_failure_payload,
    task_design_changed_paths,
    update_repair_memory,
    validate_delta,
    validate_design,
    validate_review,
    validation_failure_payload,
)
from langgraph_agentic_scientist.drafting.design_models import (
    DESIGN_DIMENSIONS,
    DeltaAudit,
    DesignPatch,
    DesignProgress,
    DesignReview,
    RepairMemory,
    TaskDesign,
)

TASK = TASK_MD.encode("utf-8")
WALL = "/planes/search/budget/wall_clock_seconds"


def _review(
    design: TaskDesign, failures: dict[str, str] | None = None, **kwargs: Any
) -> DesignReview:
    return DesignReview.model_validate(review_payload(design.fingerprint(), failures, **kwargs))


def _row(payload: dict[str, Any], dimension: str) -> dict[str, Any]:
    return next(row for row in payload["dimensions"] if row["id"] == dimension)


def _patch(parent: TaskDesign, path: str, value: Any, **extra: Any) -> DesignPatch:
    payload = patch_payload(parent.fingerprint(), path, value)
    payload.update(extra)
    return DesignPatch.model_validate(payload)


# --- designs ------------------------------------------------------------------------------------


def test_a_complete_design_validates_and_has_a_stable_fingerprint() -> None:
    design = make_design()
    assert validate_design(design, TASK) == design.fingerprint()
    assert len(design.fingerprint()) == 64
    assert design.payload() == design_payload()
    # the design with no upstream is allowed when the task names none
    assert validate_design(make_design(lambda payload: payload.update(upstream=[])), TASK)


def _overlap_seeds(payload: dict[str, Any]) -> None:
    payload["planes"]["final"]["seeds"] = [1]


def _custom_interface(payload: dict[str, Any]) -> None:
    payload["candidate"]["interface"] = "custom:run"


def _editable(path: str) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        payload["candidate"]["editable_paths"] = [path]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda p: p["canonical_task"].update(sha256="0" * 64),
            "canonical task hash",
            id="task-hash-mismatch",
        ),
        pytest.param(_overlap_seeds, "seeds must be disjoint", id="overlapping-seeds"),
        pytest.param(_custom_interface, "python_rpc_v1:handle", id="noncanonical-interface"),
        pytest.param(
            lambda p: p["feedback"].update(final_sealed=["search metrics"]),
            "feedback must be disjoint",
            id="overlapping-feedback",
        ),
        pytest.param(
            lambda p: p["upstream"].append(dict(p["upstream"][0])),
            "upstream repositories must be unique",
            id="duplicate-upstream",
        ),
        pytest.param(
            lambda p: p["planes"]["search"]["budget"].update(wall_clock_seconds=0),
            "wall_clock_seconds must be a positive integer",
            id="zero-budget",
        ),
        pytest.param(
            lambda p: p["planes"]["search"].update(seeds=[1, 1]),
            "must contain unique values",
            id="duplicate-seeds",
        ),
        pytest.param(
            lambda p: p["planes"]["search"].update(seeds=[-1]),
            "non-negative integers",
            id="negative-seed",
        ),
        pytest.param(
            lambda p: p["method"].update(claim="   "),
            "method.claim must be a non-empty string",
            id="blank-claim",
        ),
        *[
            pytest.param(_editable(path), "inside program_0", id=f"editable:{path}")
            for path in (
                "candidate/solver.py",
                "program_0/../candidate/solver.py",
                "/program_0/src",
                "program_0//src",
                "program_0/src/file:name.py",
                "program_0/src/*.py",
                "program_0/src/",
                "program_0",
                "program_0/program_spec.toml",
            )
        ],
    ],
)
def test_invalid_designs_are_rejected_with_upstreams_message(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    design = make_design(mutate)
    with pytest.raises(TaskDesignError, match=message):
        validate_design(design, TASK)


def test_a_design_with_extra_or_missing_fields_is_refused_when_built_from_a_payload() -> None:
    payload = design_payload()
    payload["extra"] = 1
    with pytest.raises(TaskDesignError, match=r"fields mismatch: missing=\[\], extra=\['extra'\]"):
        design_from_payload(payload, TASK)
    payload = design_payload()
    del payload["planes"]["final"]
    with pytest.raises(TaskDesignError, match="planes fields mismatch"):
        design_from_payload(payload, TASK)


def test_a_rich_value_where_a_string_belongs_is_refused() -> None:
    payload = design_payload()
    payload["candidate"]["state_semantics"] = {"reset": True}
    with pytest.raises(
        TaskDesignError, match=r"candidate\.state_semantics must be a non-empty string"
    ):
        design_from_payload(payload, TASK)


def test_the_fingerprint_is_upstreams_canonical_hash() -> None:
    import hashlib
    import json

    design = make_design()
    canonical = json.dumps(
        design_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert design.fingerprint() == hashlib.sha256(canonical.encode()).hexdigest()


# --- reviews ------------------------------------------------------------------------------------


def test_a_review_binds_its_verdict_to_the_dimensions() -> None:
    design = make_design()
    review = _review(design, {"compute_realism": BUDGET})
    validate_review(review, design)
    assert review.verdict == "fail"
    assert review.failed_dimensions == ("compute_realism",)
    payload = review_payload(design.fingerprint(), {"compute_realism": BUDGET})
    payload["verdict"] = "pass"
    with pytest.raises(TaskDesignError, match="verdict does not match dimensions"):
        validate_review(DesignReview.model_validate(payload), design)


def test_a_review_of_another_design_or_another_schema_is_refused() -> None:
    design = make_design()
    stale = DesignReview.model_validate(review_payload("0" * 64))
    with pytest.raises(TaskDesignError, match="fingerprint does not match design"):
        validate_review(stale, design)
    payload = review_payload(design.fingerprint())
    payload["schema_version"] = 2
    with pytest.raises(TaskDesignError, match="schema_version must be 3"):
        validate_review(DesignReview.model_validate(payload), design)


def test_a_review_must_hold_each_dimension_exactly_once() -> None:
    design = make_design()
    payload = review_payload(design.fingerprint())
    payload["dimensions"].pop()
    with pytest.raises(TaskDesignError, match="exactly the required dimensions"):
        validate_review(DesignReview.model_validate(payload), design)
    payload = review_payload(design.fingerprint())
    payload["dimensions"][1]["id"] = payload["dimensions"][0]["id"]
    with pytest.raises(TaskDesignError, match="duplicate design dimension"):
        validate_review(DesignReview.model_validate(payload), design)


@pytest.mark.parametrize(
    ("failed", "overrides", "message"),
    [
        (
            True,
            {"repair_paths": []},
            "failed dimension compute_realism requires repair_paths",
        ),
        (
            False,
            {"repair_paths": ["/method/claim"]},
            "passed dimension compute_realism must not have repair_paths",
        ),
        (
            True,
            {"repair_paths": ["/canonical_task/sha256"]},
            "repair path is not writable",
        ),
        (True, {"repair_paths": ["/candidate/interface"]}, "repair path is immutable"),
        (True, {"repair_paths": ["/candidate"]}, "repair path is immutable"),
        (True, {"repair_paths": ["/missing"]}, "repair path does not exist"),
        (
            True,
            {"design_paths": ["/method/claim"], "repair_paths": [BUDGET]},
            "repair path lacks matching design evidence",
        ),
        (True, {"remediation": ""}, "failed dimension compute_realism requires remediation"),
        (
            False,
            {"remediation": "Fix it."},
            "passed dimension compute_realism must not have remediation",
        ),
        (True, {"design_paths": ["/missing"]}, "design path does not exist"),
        (True, {"evidence": []}, "evidence must be a non-empty array"),
        (True, {"task_evidence": ["x", "x"]}, "task_evidence must contain unique values"),
    ],
)
def test_invalid_repair_scope_and_row_rules(
    failed: bool, overrides: dict[str, Any], message: str
) -> None:
    design = make_design()
    payload = review_payload(design.fingerprint(), {"compute_realism": BUDGET} if failed else None)
    _row(payload, "compute_realism").update(overrides)
    payload["verdict"] = "fail" if failed else "pass"
    with pytest.raises(TaskDesignError, match=message):
        validate_review(DesignReview.model_validate(payload), design)


# --- patches ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pointer", "value"),
    [
        (WALL, 7200),
        pytest.param("/upstream/0/revision", "fedcba9876543210", id="bounded-array-index"),
    ],
)
def test_a_patch_changes_only_its_declared_paths(pointer: str, value: Any) -> None:
    design = make_design()
    review = _review(design, {"compute_realism": pointer})
    candidate = apply_design_patch(
        _patch(design, pointer, value), parent=design, canonical_task=TASK, repair_review=review
    )
    payload = candidate.payload()
    for token in pointer.strip("/").split("/"):
        payload = payload[int(token)] if isinstance(payload, list) else payload[token]
    assert payload == value
    assert candidate.payload()["planes"]["final"] == design.payload()["planes"]["final"]
    assert candidate.payload()["method"] == design.payload()["method"]


def test_a_patch_cannot_replace_an_immutable_ancestor() -> None:
    design = make_design()
    review = _review(design, {"candidate_editability": "/candidate"})
    replacement = {**design.payload()["candidate"], "state_semantics": "Changed."}
    with pytest.raises(TaskDesignError, match="patch path is immutable"):
        apply_design_patch(
            _patch(design, "/candidate", replacement),
            parent=design,
            canonical_task=TASK,
            repair_review=review,
        )


def test_a_patch_needs_failed_repair_authority_covering_its_path() -> None:
    design = make_design()
    outside = _review(design, {"compute_realism": BUDGET}, design_paths={"compute_realism": CLAIM})
    with pytest.raises(TaskDesignError, match="outside failed repair paths"):
        apply_design_patch(
            _patch(design, CLAIM, "Unrelated."),
            parent=design,
            canonical_task=TASK,
            repair_review=outside,
        )
    passing = _review(design)
    with pytest.raises(TaskDesignError, match="repair review must fail"):
        apply_design_patch(
            _patch(design, CLAIM, "Unrelated."),
            parent=design,
            canonical_task=TASK,
            repair_review=passing,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"parent_fingerprint": "0" * 64}, "parent fingerprint"),
        ({"schema_version": 2}, "schema_version must be 1"),
        ({"operations": []}, "non-empty array"),
        (
            {"operations": [{"op": "replace", "path": "", "value": {}}]},
            "root",
        ),
        (
            {
                "operations": [
                    {"op": "replace", "path": "/canonical_task/sha256", "value": "0" * 64}
                ]
            },
            "canonical_task",
        ),
        ({"operations": [{"op": "remove", "path": "/method/claim", "value": None}]}, "replace"),
        (
            {
                "operations": [
                    {"op": "replace", "path": CLAIM, "value": "a"},
                    {"op": "replace", "path": CLAIM, "value": "b"},
                ]
            },
            "duplicate task design patch path",
        ),
        (
            {"operations": [{"op": "replace", "path": "/method/missing", "value": "a"}]},
            "patch path does not exist",
        ),
        (
            {"operations": [{"op": "replace", "path": "/upstream/5/revision", "value": "a"}]},
            "path does not exist",
        ),
        (
            {"operations": [{"op": "replace", "path": "/upstream/01/revision", "value": "a"}]},
            "invalid array index",
        ),
        (
            {"operations": [{"op": "replace", "path": "/method/~2", "value": "a"}]},
            "invalid JSON pointer escape",
        ),
        (
            {"operations": [{"op": "replace", "path": CLAIM, "value": "Improve held-out loss."}]},
            "must change the design",
        ),
    ],
)
def test_unsafe_patches_are_refused(mutation: dict[str, Any], message: str) -> None:
    design = make_design()
    review = _review(
        design, {"goal_and_method_fidelity": "/method", "upstream_grounding": "/upstream"}
    )
    payload = patch_payload(design.fingerprint(), CLAIM, "Better.")
    payload.update(mutation)
    with pytest.raises(TaskDesignError, match=message):
        apply_design_patch(
            DesignPatch.model_validate(payload),
            parent=design,
            canonical_task=TASK,
            repair_review=review,
        )


def test_a_patch_that_makes_the_design_invalid_is_refused() -> None:
    design = make_design()
    review = _review(design, {"goal_and_method_fidelity": "/method"})
    with pytest.raises(TaskDesignError, match=r"method\.claim must be a non-empty string"):
        apply_design_patch(
            _patch(design, CLAIM, ""), parent=design, canonical_task=TASK, repair_review=review
        )
    with pytest.raises(TaskDesignError, match=r"method\.claim must be a non-empty string"):
        apply_design_patch(
            _patch(design, CLAIM, {"nested": 1}),
            parent=design,
            canonical_task=TASK,
            repair_review=review,
        )


def test_a_patch_may_not_exceed_sixty_four_operations() -> None:
    design = make_design()
    review = _review(design, {"goal_and_method_fidelity": "/method"})
    payload = patch_payload(design.fingerprint(), CLAIM, "x")
    payload["operations"] = [
        {"op": "replace", "path": f"/method/claim{index}", "value": "x"} for index in range(65)
    ]
    with pytest.raises(TaskDesignError, match="exceeds 64 operations"):
        apply_design_patch(
            DesignPatch.model_validate(payload),
            parent=design,
            canonical_task=TASK,
            repair_review=review,
        )


def test_json_pointer_tokens_decode_escapes() -> None:
    assert decode_json_pointer("/a~1b/c~0d") == ["a/b", "c~d"]
    with pytest.raises(TaskDesignError):
        decode_json_pointer("a/b")


# --- comparing reviews --------------------------------------------------------------------------


def _bump_wall_clock(payload: dict[str, Any]) -> None:
    payload["planes"]["search"]["budget"]["wall_clock_seconds"] = 7200


def _change_claim(payload: dict[str, Any]) -> None:
    payload["method"]["claim"] = "Changed claim."


@pytest.mark.parametrize(
    ("mutate", "failures", "design_paths", "improved", "regressed"),
    [
        pytest.param(None, None, None, True, (), id="strict-monotonic-progress"),
        pytest.param(
            _bump_wall_clock,
            {"reproducibility": "/candidate/state_semantics"},
            None,
            True,
            (),
            id="ignores-new-finding-on-unchanged-paths",
        ),
        pytest.param(
            _change_claim,
            {"reproducibility": BUDGET},
            {"reproducibility": "/method"},
            False,
            ("reproducibility",),
            id="rejects-new-failure-on-changed-ancestor",
        ),
    ],
)
def test_comparing_reviews(
    mutate: Callable[[dict[str, Any]], None] | None,
    failures: dict[str, str] | None,
    design_paths: dict[str, str] | None,
    improved: bool,
    regressed: tuple[str, ...],
) -> None:
    parent_design = make_design()
    candidate_design = make_design(mutate) if mutate else parent_design
    parent = _review(parent_design, {"compute_realism": BUDGET})
    candidate = _review(candidate_design, failures, design_paths=design_paths)
    progress = compare_reviews(
        parent, candidate, parent_design=parent_design, candidate_design=candidate_design
    )
    assert progress.improved is improved
    assert progress.fixed == ["compute_realism"]
    assert tuple(progress.regressed) == regressed


def test_still_failed_dimensions_are_reported_in_canonical_order() -> None:
    design = make_design()
    parent = _review(design, {"reproducibility": BUDGET, "compute_realism": BUDGET})
    candidate = _review(design, {"reproducibility": BUDGET})
    progress = compare_reviews(parent, candidate, parent_design=design, candidate_design=design)
    assert progress.fixed == ["compute_realism"]
    assert progress.still_failed == ["reproducibility"]
    assert progress.improved is True


# --- delta audits -------------------------------------------------------------------------------


def _audit(parent: TaskDesign, candidate: TaskDesign, **kwargs: Any) -> DeltaAudit:
    return DeltaAudit.model_validate(
        delta_payload(parent.fingerprint(), candidate.fingerprint(), **kwargs)
    )


def test_a_delta_regression_must_cite_changed_paths() -> None:
    parent = make_design()
    candidate = make_design(_change_claim)
    delta = _audit(
        parent,
        candidate,
        fixed=["compute_realism"],
        regressions=[regression("reproducibility", CLAIM)],
    )
    validate_delta(delta, parent=parent, candidate=candidate)
    assert delta.verdict == "fail" and delta.regressed_dimensions == ("reproducibility",)
    unchanged = _audit(
        parent, candidate, fixed=[], regressions=[regression("reproducibility", BUDGET)]
    )
    with pytest.raises(TaskDesignError, match="was not changed by the candidate"):
        validate_delta(unchanged, parent=parent, candidate=candidate)


@pytest.mark.parametrize(
    ("payload_update", "message"),
    [
        ({"verdict": "pass"}, "verdict does not match regressions"),
        ({"parent_fingerprint": "0" * 64}, "parent fingerprint does not match"),
        ({"candidate_fingerprint": "0" * 64}, "candidate fingerprint does not match"),
        ({"fixed_dimensions": ["nonsense"]}, "unknown dimensions"),
        ({"fixed_dimensions": ["compute_realism", "compute_realism"]}, "unique values"),
        ({"summary": " "}, "summary must be a non-empty string"),
    ],
)
def test_invalid_delta_audits_are_refused(payload_update: dict[str, Any], message: str) -> None:
    parent = make_design()
    candidate = make_design(_change_claim)
    payload = delta_payload(
        parent.fingerprint(),
        candidate.fingerprint(),
        fixed=["compute_realism"],
        regressions=[regression("reproducibility", CLAIM)],
    )
    payload.update(payload_update)
    with pytest.raises(TaskDesignError, match=message):
        validate_delta(DeltaAudit.model_validate(payload), parent=parent, candidate=candidate)


def test_duplicate_and_unknown_regressions_are_refused() -> None:
    parent = make_design()
    candidate = make_design(_change_claim)
    duplicate = _audit(
        parent,
        candidate,
        fixed=[],
        regressions=[regression("reproducibility", CLAIM), regression("reproducibility", CLAIM)],
    )
    with pytest.raises(TaskDesignError, match="duplicate regressed dimension"):
        validate_delta(duplicate, parent=parent, candidate=candidate)
    unknown = _audit(parent, candidate, fixed=[], regressions=[regression("nonsense", CLAIM)])
    with pytest.raises(TaskDesignError, match="unknown regressed dimension"):
        validate_delta(unknown, parent=parent, candidate=candidate)


def _add_entrypoint(payload: dict[str, Any]) -> None:
    payload["upstream"][0]["entrypoints"] = ["train.py", "invalid smoke claim"]


@pytest.mark.parametrize(
    ("mutate", "delta", "candidate_failures", "claimed", "message"),
    [
        pytest.param(
            _change_claim,
            {"fixed": [], "regressions": []},
            None,
            DesignProgress(fixed=["compute_realism"]),
            "fixed dimensions",
            id="must-confirm-every-consensus-fix",
        ),
        pytest.param(
            _change_claim,
            {"fixed": ["compute_realism"], "regressions": []},
            {"reproducibility": CLAIM},
            DesignProgress(fixed=["compute_realism"], regressed=["reproducibility"]),
            None,
            id="can-dispute-consensus-regression",
        ),
        pytest.param(
            _change_claim,
            {
                "fixed": ["compute_realism"],
                "regressions": [regression("reproducibility", CLAIM, ["/upstream"])],
            },
            {"reproducibility": CLAIM},
            DesignProgress(fixed=["compute_realism"], regressed=["reproducibility"]),
            "repair paths do not match",
            id="repair-paths-must-match-candidate-consensus",
        ),
        pytest.param(
            _add_entrypoint,
            {
                "fixed": ["compute_realism", "reproducibility"],
                "regressions": [regression("upstream_grounding", "/upstream/0/entrypoints/1")],
            },
            None,
            DesignProgress(fixed=["upstream_grounding", "compute_realism", "reproducibility"]),
            None,
            id="can-dispute-consensus-fix-as-regression",
        ),
    ],
)
def test_the_delta_must_account_for_the_consensus_progress(
    mutate: Callable[[dict[str, Any]], None],
    delta: dict[str, Any],
    candidate_failures: dict[str, str] | None,
    claimed: DesignProgress,
    message: str | None,
) -> None:
    parent = make_design()
    candidate = make_design(mutate)
    audit = _audit(parent, candidate, **delta)
    review = _review(candidate, candidate_failures)
    if message is None:
        confirm_delta(audit, claimed, review)
    else:
        with pytest.raises(TaskDesignError, match=message):
            confirm_delta(audit, claimed, review)


def test_a_legacy_delta_cannot_authorize_a_repair_frontier() -> None:
    parent = make_design()
    candidate = make_design(_change_claim)
    payload = delta_payload(
        parent.fingerprint(),
        candidate.fingerprint(),
        fixed=["compute_realism"],
        regressions=[regression("reproducibility", CLAIM)],
    )
    payload["schema_version"] = 1
    for row in payload["regressions"]:
        del row["repair_paths"]
    audit = DeltaAudit.model_validate(payload)
    validate_delta(audit, parent=parent, candidate=candidate)
    review = _review(candidate, {"reproducibility": CLAIM})
    claimed = DesignProgress(fixed=["compute_realism"], regressed=["reproducibility"])
    with pytest.raises(TaskDesignError, match="cannot authorize a repair frontier"):
        confirm_delta(audit, claimed, review)


def test_check_delta_runs_the_whole_contract() -> None:
    parent = make_design()
    candidate = make_design(_change_claim)
    parent_review = _review(parent, {"compute_realism": BUDGET})
    candidate_review = _review(candidate)
    audit = _audit(parent, candidate, fixed=["compute_realism"])
    check_delta(
        audit,
        parent=parent,
        candidate=candidate,
        parent_review=parent_review,
        candidate_review=candidate_review,
    )
    claims_nothing = _audit(parent, candidate, fixed=[])
    with pytest.raises(TaskDesignError, match="do not match consensus progress"):
        check_delta(
            claims_nothing,
            parent=parent,
            candidate=candidate,
            parent_review=parent_review,
            candidate_review=candidate_review,
        )


def test_indirect_regressions_are_those_outside_the_changed_paths() -> None:
    parent = make_design()
    candidate = make_design(_bump_wall_clock)
    parent_review = _review(parent, {"compute_realism": WALL})
    candidate_review = _review(
        candidate,
        {"reproducibility": "/upstream/0/revision"},
        design_paths={"reproducibility": WALL},
    )
    consensus = compare_reviews(
        parent_review, candidate_review, parent_design=parent, candidate_design=candidate
    )
    assert consensus.regressed == ["reproducibility"]
    indirect = _audit(
        parent,
        candidate,
        fixed=["compute_realism"],
        regressions=[regression("reproducibility", WALL, ["/upstream/0/revision"])],
    )
    progress = enrich_progress(consensus, candidate_review, indirect)
    assert has_only_indirect_regressions(
        progress,
        parent=parent,
        candidate=candidate,
        parent_review=parent_review,
        candidate_review=candidate_review,
        delta=indirect,
    )
    direct = _audit(
        parent,
        candidate,
        fixed=["compute_realism"],
        regressions=[regression("reproducibility", WALL, [WALL])],
    )
    progress = enrich_progress(consensus, candidate_review, direct)
    assert not has_only_indirect_regressions(
        progress,
        parent=parent,
        candidate=candidate,
        parent_review=parent_review,
        candidate_review=candidate_review,
        delta=direct,
    )


# --- latent findings ----------------------------------------------------------------------------


def test_latent_failures_are_rebound_only_when_unchanged_and_not_regressions() -> None:
    parent = make_design()
    candidate = make_design(_bump_wall_clock)
    parent_review = _review(parent, {"compute_realism": WALL})
    candidate_review = _review(
        candidate,
        {
            "metrics_and_rubrics": "/planes/search/metrics",
            "reproducibility": WALL,
            "publication_scope": WALL,
        },
    )
    delta = _audit(parent, candidate, fixed=["compute_realism"])
    delta = delta.model_copy(
        update={
            "regressions": [
                regression("reproducibility", WALL).__class__
                and _regression_model("reproducibility", WALL)
            ],
            "verdict": "fail",
        }
    )
    rebound = rebind_latent_failures(
        parent=parent,
        candidate=candidate,
        parent_review=parent_review,
        candidate_review=candidate_review,
        delta=delta,
        consensus_regressions=(),
    )
    assert rebound is not None
    assert rebound.design_fingerprint == parent.fingerprint()
    assert rebound.failed_dimensions == ("metrics_and_rubrics", "compute_realism")
    rows = {row.id: row for row in rebound.dimensions}
    assert rows["compute_realism"].model_dump() == _find(parent_review, "compute_realism")
    assert rows["metrics_and_rubrics"].repair_paths == ["/planes/search/metrics"]
    assert rows["reproducibility"].verdict == "pass"
    assert rows["publication_scope"].verdict == "pass"
    validate_review(rebound, parent)


def _regression_model(dimension: str, path: str) -> Any:
    from langgraph_agentic_scientist.drafting.design_models import DeltaRegression

    return DeltaRegression.model_validate(regression(dimension, path))


def _find(review: DesignReview, dimension: str) -> dict[str, Any]:
    return next(row.model_dump() for row in review.dimensions if row.id == dimension)


def test_latent_rebinding_returns_none_without_safe_findings() -> None:
    parent = make_design()
    candidate = make_design(_bump_wall_clock)
    parent_review = _review(parent, {"compute_realism": WALL})
    delta = _audit(parent, candidate, fixed=["compute_realism"])
    # a failure on a changed path is the repair's own
    changed = _review(candidate, {"reproducibility": WALL})
    assert (
        rebind_latent_failures(
            parent=parent,
            candidate=candidate,
            parent_review=parent_review,
            candidate_review=changed,
            delta=delta,
            consensus_regressions=(),
        )
        is None
    )
    # a consensus regression overrides a disputing delta
    unchanged = _review(candidate, {"metrics_and_rubrics": "/planes/search/metrics"})
    assert (
        rebind_latent_failures(
            parent=parent,
            candidate=candidate,
            parent_review=parent_review,
            candidate_review=unchanged,
            delta=delta,
            consensus_regressions=("metrics_and_rubrics",),
        )
        is None
    )
    # a finding mixing changed and unchanged evidence is not safe
    mixed = review_payload(
        candidate.fingerprint(), {"metrics_and_rubrics": "/planes/search/metrics"}
    )
    _row(mixed, "metrics_and_rubrics")["design_paths"] = ["/planes/search/metrics", WALL]
    assert (
        rebind_latent_failures(
            parent=parent,
            candidate=candidate,
            parent_review=parent_review,
            candidate_review=DesignReview.model_validate(mixed),
            delta=delta,
            consensus_regressions=(),
        )
        is None
    )


def test_a_rebound_review_above_the_size_limit_is_dropped() -> None:
    parent = make_design()
    candidate = make_design(_bump_wall_clock)
    parent_payload = review_payload(parent.fingerprint(), {"compute_realism": WALL})
    _row(parent_payload, "compute_realism")["evidence"] = ["p" * 600_000]
    candidate_payload = review_payload(
        candidate.fingerprint(), {"metrics_and_rubrics": "/planes/search/metrics"}
    )
    _row(candidate_payload, "metrics_and_rubrics")["evidence"] = ["c" * 600_000]
    assert (
        rebind_latent_failures(
            parent=parent,
            candidate=candidate,
            parent_review=DesignReview.model_validate(parent_payload),
            candidate_review=DesignReview.model_validate(candidate_payload),
            delta=_audit(parent, candidate, fixed=["compute_realism"]),
            consensus_regressions=(),
        )
        is None
    )


def test_changed_paths_are_leaf_pointers_and_whole_arrays_when_lengths_differ() -> None:
    parent = make_design()
    candidate = make_design(
        lambda payload: (
            payload["planes"]["search"]["budget"].update(wall_clock_seconds=1),
            payload["upstream"][0]["entrypoints"].append("other.py"),
        )
    )
    assert task_design_changed_paths(parent, candidate) == (
        "/planes/search/budget/wall_clock_seconds",
        "/upstream/0/entrypoints",
    )


# --- repair feedback ----------------------------------------------------------------------------


def test_the_validation_failure_payload_is_bounded() -> None:
    payload = validation_failure_payload(2, "x" * 9000 + "\n")
    assert payload["source"] == "schema_validation" and payload["attempt"] == 2
    assert payload["diagnostic"].endswith(" [truncated]")
    assert len(payload["diagnostic"]) == 8192 + len(" [truncated]")
    assert validation_failure_payload(0, " short \n")["diagnostic"] == "short"


def test_the_review_failure_payload_keeps_failures_and_protects_passing_dimensions() -> None:
    design = make_design()
    review = _review(design, {"compute_realism": WALL})
    payload = review_failure_payload(review, 3)
    assert payload["source"] == "independent_review" and payload["attempt"] == 3
    assert [row["id"] for row in payload["review"]["dimensions"]] == ["compute_realism"]
    protected = {row["id"] for row in payload["protected_dimensions"]}
    assert protected == set(DESIGN_DIMENSIONS) - {"compute_realism"}
    assert all(
        set(row) == {"id", "design_paths", "evidence"} for row in payload["protected_dimensions"]
    )
    assert "previous_candidate" not in payload and "repair_memory" not in payload
    progress = DesignProgress(
        fixed=["upstream_grounding"],
        regressed=["reproducibility"],
        candidate_findings=[{"id": "reproducibility", "remediation": "r" * 2000}],
        delta_regressions=[{"dimension": "reproducibility", "remediation": "m"}],
        diagnostic="d" * 5000,
    )
    memory = RepairMemory(constraints=[{"dimension": "reproducibility"}])
    payload = review_failure_payload(review, 4, progress, memory)
    previous = payload["previous_candidate"]
    assert previous["fixed"] == ["upstream_grounding"]
    assert previous["diagnostic"].endswith(" [truncated]")
    assert previous["candidate_findings"][0]["remediation"].endswith(" [truncated]")
    assert previous["delta_regressions"][0]["dimension"] == "reproducibility"
    assert payload["repair_memory"] == {
        "schema_version": 1,
        "constraints": [{"dimension": "reproducibility"}],
    }


def _failing(*identifiers: str) -> dict[str, Any]:
    return {"review": {"dimensions": [{"id": i, "verdict": "fail"} for i in identifiers]}}


def _previous(**fields: Any) -> dict[str, Any]:
    return {"fixed": [], "still_failed": [], "regressed": [], "delta_regressions": [], **fields}


THREE = ("metrics_and_rubrics", "execution_feasibility", "reproducibility")


def _focused_ids(payloads: list[dict[str, Any]]) -> list[str]:
    history: list[str] = []
    for payload in payloads:
        history.append(focus_repair_payload(payload, history)[1])
    return history


@pytest.mark.parametrize(
    ("payload", "attempts", "expected"),
    [
        pytest.param({"review": _failing(*THREE)["review"]}, 3, list(THREE), id="rotates"),
        pytest.param(
            {
                "review": _failing(
                    "reproducibility", "metrics_and_rubrics", "execution_feasibility"
                )["review"],
                "previous_candidate": _previous(
                    still_failed=list(THREE), regressed=["execution_feasibility"]
                ),
            },
            3,
            ["execution_feasibility", "metrics_and_rubrics", "reproducibility"],
            id="prioritizes-regression-then-rotates",
        ),
        pytest.param(
            _failing("reproducibility", "compute_realism", "metrics_and_rubrics"),
            3,
            ["metrics_and_rubrics", "compute_realism", "reproducibility"],
            id="stable-canonical-dimension-order",
        ),
        pytest.param(
            {
                **_failing("reproducibility", "metrics_and_rubrics"),
                "previous_candidate": _previous(
                    fixed=["compute_realism"],
                    still_failed=["metrics_and_rubrics", "reproducibility"],
                ),
            },
            4,
            ["metrics_and_rubrics", "reproducibility", "metrics_and_rubrics", "reproducibility"],
            id="excludes-fixed-or-removed-dimensions",
        ),
    ],
)
def test_focus_rotation(payload: dict[str, Any], attempts: int, expected: list[str]) -> None:
    assert _focused_ids([payload for _ in range(attempts)]) == expected


def test_changing_regressions_cannot_starve_persistent_failures() -> None:
    payloads = [
        {
            **_failing(*THREE),
            "previous_candidate": _previous(still_failed=list(THREE), regressed=[regressed]),
        }
        for regressed in THREE
    ]
    assert _focused_ids(payloads) == list(THREE)


def test_focus_keeps_regressions_as_constraints_and_cuts_the_rest() -> None:
    payload = {
        **_failing("search_proxy_fidelity", "metrics_and_rubrics"),
        "previous_candidate": _previous(
            fixed=["search_proxy_fidelity"],
            still_failed=["metrics_and_rubrics"],
            regressed=["metrics_and_rubrics"],
            candidate_findings=[
                {"id": "search_proxy_fidelity", "remediation": "stale"},
                {"id": "metrics_and_rubrics", "remediation": "align metrics"},
            ],
            delta_regressions=[
                {"dimension": "metrics_and_rubrics", "remediation": "align metrics"}
            ],
        ),
    }
    focused, dimension = focus_repair_payload(payload, [])
    assert dimension == "metrics_and_rubrics" and focused["mode"] == "focused"
    assert focused["review"]["dimensions"] == [{"id": "metrics_and_rubrics", "verdict": "fail"}]
    assert focused["previous_candidate"]["fixed"] == []
    assert focused["previous_candidate"]["candidate_findings"] == [
        {"id": "metrics_and_rubrics", "remediation": "align metrics"}
    ]
    payload = {
        **_failing("search_proxy_fidelity"),
        "previous_candidate": _previous(
            still_failed=["search_proxy_fidelity"],
            regressed=["reproducibility"],
            candidate_findings=[{"id": "reproducibility", "remediation": "drift"}],
            delta_regressions=[{"dimension": "reproducibility", "remediation": "do not weaken"}],
        ),
    }
    focused, dimension = focus_repair_payload(payload, [])
    assert dimension == "search_proxy_fidelity"
    assert "candidate_findings" not in focused["previous_candidate"]
    assert focused["previous_candidate"]["regressed"] == ["reproducibility"]
    assert focused["previous_candidate"]["delta_regressions"][0]["remediation"] == "do not weaken"
    # the input is not mutated
    assert "mode" not in payload


def test_a_focused_authorization_confines_a_patch_to_the_focused_dimension() -> None:
    design = make_design()
    review = _review(design, {"compute_realism": BUDGET, "reproducibility": CLAIM})
    feedback = {"mode": "focused", "review": _failing("compute_realism")["review"]}
    authorization = focused_authorization(review, feedback)
    validate_review(authorization, design)
    assert authorization.failed_dimensions == ("compute_realism",)
    with pytest.raises(TaskDesignError, match="outside failed repair paths"):
        apply_design_patch(
            _patch(design, CLAIM, "Changed."),
            parent=design,
            canonical_task=TASK,
            repair_review=authorization,
        )
    with pytest.raises(TaskDesignError, match="exactly one dimension"):
        focused_authorization(review, {"mode": "focused", "review": _failing("a", "b")["review"]})


def test_the_repair_memory_keeps_regressions_and_at_most_thirty_two() -> None:
    progress = DesignProgress(
        regressed=["reproducibility"],
        candidate_findings=[
            {
                "id": "reproducibility",
                "design_paths": ["/method"],
                "evidence": ["e"],
                "remediation": "r",
            },
            {"id": "other", "remediation": "ignored"},
        ],
        delta_regressions=[
            {"dimension": "reproducibility", "design_paths": ["/method/claim"], "evidence": ["e"]}
        ],
    )
    memory = update_repair_memory(RepairMemory(), progress)
    assert [row["dimension"] for row in memory.constraints] == ["reproducibility"] * 2
    again = update_repair_memory(memory, progress)
    assert again.constraints == memory.constraints, (
        "an identical constraint moves, it is not repeated"
    )
    many = RepairMemory(constraints=[{"dimension": f"d{index}"} for index in range(40)])
    assert len(update_repair_memory(many, progress).constraints) == 32


def test_enriching_progress_with_a_delta_moves_the_consensus_aside() -> None:
    design = make_design()
    parent = make_design()
    candidate_review = _review(design, {"reproducibility": CLAIM})
    consensus = DesignProgress(
        fixed=["compute_realism"], regressed=["reproducibility"], improved=False
    )
    enriched = enrich_progress(consensus, candidate_review)
    assert enriched.candidate_findings[0]["id"] == "reproducibility"
    assert enriched.consensus_fixed is None
    delta = _audit(
        parent, make_design(_change_claim), fixed=["compute_realism", "upstream_grounding"]
    )
    enriched = enrich_progress(consensus, candidate_review, delta)
    assert enriched.consensus_regressed == ["reproducibility"]
    assert enriched.fixed == ["compute_realism", "upstream_grounding"]
    assert enriched.regressed == [] and enriched.improved is True
