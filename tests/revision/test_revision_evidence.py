"""Evidence retention: containment of the original's data and the passing-audit check."""

from __future__ import annotations

from typing import Any

import pytest
from revision_helpers import audit_files, put, sha

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.revision.evidence import (
    EvidenceError,
    candidate_audit_hashes,
    contains,
    contains_normalized,
    immutable_paper_evidence,
    normalize,
    published_audit_hashes,
    validate_passing_audits,
    validate_semantic_evidence,
)


def test_a_mapping_contains_what_it_extends() -> None:
    assert contains({"a": 1, "b": {"c": 2, "d": 3}}, {"a": 1, "b": {"c": 2}})
    assert not contains({"a": 1}, {"a": 1, "b": 2})
    assert not contains({"a": 1}, {"a": 2})
    assert not contains([1], {"a": 1})


def test_a_list_must_hold_a_distinct_matching_item_for_every_required_one() -> None:
    assert contains([{"id": 1, "x": 1}, {"id": 2}], [{"id": 2}, {"id": 1}])
    assert contains([1, 2, 3], [3, 1])
    assert not contains([1, 2], [1, 1])  # distinct items
    assert not contains([1], [1, 2])
    # the most specific items are matched first, so a general one cannot take a specific one's place
    assert contains([{"a": 1, "b": 2}, {"a": 1}], [{"a": 1}, {"a": 1, "b": 2}])
    assert not contains([{"a": 1, "b": 2}], [{"a": 1}, {"a": 1, "b": 2}])


def test_volatile_keys_and_absolute_locations_do_not_count() -> None:
    original = {
        "generated_at": "then",
        "results_dir": "/a/b",
        "artifacts": {"x": "/a/x", "y": "relative/y"},
        "rows": [{"path": "/a/p", "n": 1}],
        "note": "/not/a/location",
    }
    revised = {
        "generated_at": "now",
        "results_dir": "/c/d",
        "artifacts": {"x": "/c/x", "y": "relative/y"},
        "rows": [{"path": "/c/q", "n": 1}],
        "note": "/not/a/location",
    }
    assert contains_normalized(revised, original)
    assert not contains_normalized({**revised, "artifacts": {"y": "elsewhere/y"}}, original)
    assert normalize({"updated_at": 1, "keep": 2}) == {"keep": 2}


def test_editorial_annotations_of_analyses_are_not_evidence() -> None:
    data = {"posthoc_analyses": [{"id": 1, "relevance_to_thesis": "x", "n": 1}, "plain"]}
    assert immutable_paper_evidence(data) == {"posthoc_analyses": [{"id": 1, "n": 1}, "plain"]}
    assert immutable_paper_evidence({"other": 1}) == {"other": 1}
    revised = {"posthoc_analyses": [{"id": 1, "relevance_to_thesis": "y", "n": 1}, "plain"]}
    validate_semantic_evidence(data, revised)


def test_the_lost_inventory_is_named_and_otherwise_the_data_in_general() -> None:
    source = {"all_programs": [1], "other": {"k": 1}}
    with pytest.raises(EvidenceError, match="program inventory"):
        validate_semantic_evidence(source, {"all_programs": [], "other": {"k": 1}})
    with pytest.raises(EvidenceError, match="paper data evidence"):
        validate_semantic_evidence(source, {"all_programs": [1], "other": {}})


def test_the_hash_filters_match_what_each_stage_changes() -> None:
    hashes = {
        "evidence-source/latest_manifest.json": "a",
        "evidence-source/ai_review.after_revision.json": "b",
        "evidence-source/revisions/revision_01/x": "c",
        "tex-source/paper.tex": "d",
    }
    assert candidate_audit_hashes(hashes) == {
        "evidence-source/ai_review.after_revision.json": "b",
        "evidence-source/revisions/revision_01/x": "c",
        "tex-source/paper.tex": "d",
    }
    assert published_audit_hashes(hashes) == {"tex-source/paper.tex": "d"}


def _write(workspace: LocalWorkspace, inputs: dict[str, str], **overrides: Any) -> None:
    for name in ("method_scope_audit.json", "method_scope_audit.md"):
        put(workspace.root, f"d/{name}", "x")
    for name, payload in audit_files(inputs).items():
        put(workspace.root, f"d/{name}", overrides.get(name, payload))


async def test_the_checks_that_were_not_made_are_named(workspace: LocalWorkspace) -> None:
    _write(workspace, {"tex-source/paper.tex": sha("t")})
    skipped = await validate_passing_audits(workspace, "d", published=False)
    assert skipped == ["method_scope_audit validity", "audit input hashes"]


async def test_a_recomputed_hash_set_replaces_the_comparison_of_the_records(
    workspace: LocalWorkspace,
) -> None:
    inputs = {"tex-source/paper.tex": sha("t"), "evidence-source/latest_manifest.json": sha("m")}
    _write(workspace, inputs)

    async def current(directory: str, runtime: Any) -> dict[str, str]:
        return {
            "tex-source/paper.tex": sha("t"),
            "evidence-source/latest_manifest.json": sha("changed after the audit"),
        }

    async def valid(directory: str, runtime: Any) -> bool:
        return True

    # the completion manifest is excluded: the host writes it after the final audit
    skipped = await validate_passing_audits(
        workspace, "d", published=False, audit_input_hashes=current, scope_audit=valid
    )
    assert skipped == []

    async def stale(directory: str, runtime: Any) -> dict[str, str]:
        return {"tex-source/paper.tex": sha("edited since")}

    with pytest.raises(EvidenceError, match="current passing"):
        await validate_passing_audits(workspace, "d", published=False, audit_input_hashes=stale)

    async def empty(directory: str, runtime: Any) -> dict[str, str]:
        return {}

    with pytest.raises(EvidenceError, match="current passing"):
        await validate_passing_audits(workspace, "d", published=False, audit_input_hashes=empty)


async def test_a_function_that_fails_is_a_failed_check(workspace: LocalWorkspace) -> None:
    _write(workspace, {"tex-source/paper.tex": sha("t")})

    async def broken(directory: str, runtime: Any) -> dict[str, str]:
        raise OSError("cannot read")

    with pytest.raises(EvidenceError, match="could not verify current agentic evidence audit"):
        await validate_passing_audits(workspace, "d", published=False, audit_input_hashes=broken)

    async def scope_broken(directory: str, runtime: Any) -> bool:
        raise ValueError("bad audit")

    with pytest.raises(EvidenceError, match="could not verify revision method-scope audit"):
        await validate_passing_audits(workspace, "d", published=False, scope_audit=scope_broken)


async def test_a_pass_needs_zero_findings_in_the_audit_and_in_its_record(
    workspace: LocalWorkspace,
) -> None:
    inputs = {"tex-source/paper.tex": sha("t")}
    audit, report = audit_files(inputs).values()
    _write(workspace, inputs, **{"paper_evidence_audit.json": {**audit, "finding_count": 1}})
    with pytest.raises(EvidenceError, match="current passing"):
        await validate_passing_audits(workspace, "d", published=False)
    _write(
        workspace,
        inputs,
        **{
            "validation_report.json": {
                **report,
                "agentic_evidence_audit": {**report["agentic_evidence_audit"], "findings": [1]},
            }
        },
    )
    with pytest.raises(EvidenceError, match="current passing"):
        await validate_passing_audits(workspace, "d", published=False)


async def test_an_audit_file_that_is_not_json_is_named(workspace: LocalWorkspace) -> None:
    _write(workspace, {"tex-source/paper.tex": sha("t")})
    put(workspace.root, "d/paper_evidence_audit.json", "{not json")
    with pytest.raises(EvidenceError, match="invalid agentic paper-evidence audit"):
        await validate_passing_audits(workspace, "d", published=False)
    put(workspace.root, "d/paper_evidence_audit.json", "[]")
    with pytest.raises(EvidenceError, match="must contain a JSON object"):
        await validate_passing_audits(workspace, "d", published=False)
