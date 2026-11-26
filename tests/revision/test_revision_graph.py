"""The revision graph alone: preconditions, edit, retention, promotion and the second review."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from revision_helpers import (
    StubPaper,
    answer,
    audit_files,
    paper_data,
    put,
    read_json,
    review_dict,
    sha,
    tree,
    write_original,
    write_revision,
)
from testkit import fast_config

from langgraph_agentic_scientist import (
    GraphOverrides,
    Hooks,
    LocalWorkspace,
    ReviewConfig,
    ScientistContext,
)
from langgraph_agentic_scientist.core.results import (
    EditOptions,
    PaperResult,
    ReviewResult,
    RevisionResult,
)
from langgraph_agentic_scientist.models import Role
from langgraph_agentic_scientist.revision import build_revision_graph
from langgraph_agentic_scientist.revision import graph as revision_graph
from langgraph_agentic_scientist.testing import ScriptedChatModel, ScriptedModels

pytest.importorskip("pymupdf")  # the second review reads the published PDF

CONFIG = fast_config(review=ReviewConfig(num_reviewers=1))


def _context(
    make_context: Callable[..., ScientistContext],
    *,
    reviewer: list[Any] | None = None,
    chair: list[Any] | None = None,
    **kwargs: Any,
) -> ScientistContext:
    scripts = {
        Role.AI_REVIEWER: reviewer if reviewer is not None else [answer(review_dict(Overall=5))],
        Role.AREA_CHAIR: chair if chair is not None else [answer(review_dict(Overall=5))],
    }
    return make_context(scripts, config=CONFIG, **kwargs)


def _model(ctx: ScientistContext, role: Role) -> ScriptedChatModel:
    assert isinstance(ctx.models, ScriptedModels)
    return ctx.models.scripted(role)


async def _recorded_hashes(directory: str, runtime: Any) -> Mapping[str, str]:
    """Stand-in for the paper package's recomputation: the hashes the directory's audit recorded."""
    data = await runtime.context.workspace.read_bytes(f"{directory}/paper_evidence_audit.json")
    return json.loads(data)["input_sha256"]  # type: ignore[no-any-return]


async def _scope_ok(directory: str, runtime: Any) -> bool:
    return True


def _graph(stub: Any, **slots: Any) -> Any:
    overrides = GraphOverrides(
        slots={
            "revision.paper": stub,
            "revision.audit_input_hashes": _recorded_hashes,
            "revision.scope_audit": _scope_ok,
            **slots,
        }
    )
    return build_revision_graph(checkpointer=InMemorySaver(), overrides=overrides)


async def _run(
    graph: Any, ctx: ScientistContext, *, thread: str = "t", **inputs: Any
) -> RevisionResult:
    out = await graph.ainvoke(
        {"paper_dir": "paper", **inputs}, {"configurable": {"thread_id": thread}}, context=ctx
    )
    result = out["result"]
    assert isinstance(result, RevisionResult)
    return result


def _published_digest(root: Path) -> dict[str, bytes]:
    return tree(root, "paper")


# --- the promoted revision ---------------------------------------------------------------------


async def test_a_revision_is_published_with_the_original_kept_and_reviewed_again(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    original = write_original(root)
    stub = StubPaper(root)
    ctx = _context(make_context)

    result = await _run(_graph(stub), ctx)

    assert result.status == "promoted"
    assert result.failure is None
    assert result.before_dir == "paper/revisions/revision_01/before"
    assert result.second_review is not None
    assert result.second_review.status == "completed"
    assert result.second_review.review_path == "paper/ai_review.after_revision.json"

    # the published paper is the revision
    assert (root / "paper" / "paper.tex").read_text() == "\\documentclass{article} revised"
    assert (root / "paper" / "figures" / "f2.png").read_bytes() == b"png-new"
    assert (root / "paper" / "figures" / "f1.png").read_bytes() == b"png-revised"
    assert (root / "paper" / "new_note.txt").read_text() == "new note"
    assert result.paper is not None
    assert result.paper.paper_dir == "paper"
    assert result.paper.pdf_path == "paper/paper.pdf"

    # the original is kept, byte for byte
    before = root / "paper" / "revisions" / "revision_01" / "before"
    assert (before / "paper.tex").read_bytes() == original["paper.tex"]
    assert (before / "paper_data.json").read_bytes() == original["paper_data.json"]
    assert (before / "original_manifest.json").read_bytes() == original["latest_manifest.json"]
    assert (before / "ai_review.before_revision.json").read_bytes() == original[
        "ai_review.before_revision.json"
    ]
    assert (before / "figures" / "f1.png").read_bytes() == b"png-original"
    complete = read_json(root, "paper/revisions/revision_01/before/snapshot_complete.json")
    assert complete["status"] == "complete"
    assert "paper.tex" in complete["files"]
    assert complete["figure_files"] == ["f1.png"]

    # original-only audited inputs left the paper for the archive, with their relative paths
    assert not (root / "paper" / "old_note.txt").exists()
    assert not (root / "paper" / "sources" / "a.txt").exists()
    assert not (root / "paper" / "sources" / "deep" / "b.txt").exists()
    assert (before / "ancillary" / "old_note.txt").read_text() == "old note"
    assert (before / "ancillary" / "sources" / "a.txt").read_text() == "tree file a"
    assert (before / "ancillary" / "sources" / "deep" / "b.txt").read_text() == "tree file b"

    # the manifest says what was done, with the revision's paths moved into the paper
    manifest = read_json(root, "paper/latest_manifest.json")
    assert manifest["paper_pdf"] == "paper/paper.pdf"
    assert manifest["figures"] == ["paper/figures/f1.png", "paper/figures/f2.png"]
    assert manifest["revision"] == {
        "status": "promoted",
        "source_dir": "revision/attempt_01",
        "original_snapshot": "paper/revisions/revision_01/before",
        "before_review": "paper/ai_review.before_revision.json",
        "rereview_status": "completed",
        "after_review": "paper/ai_review.after_revision.json",
    }
    assert manifest["before_revision"]["review_json"] == "paper/ai_review.before_revision.json"
    assert manifest["after_revision"]["status"] == "completed"
    state = read_json(root, "paper/revisions/revision_01/promotion_state.json")
    assert state == {"status": "completed", "revision_dir": "revision/attempt_01"}

    # the second review is the published paper's, and the first review is untouched
    assert (root / "paper" / "ai_review.after_revision.json").is_file()
    assert (root / "paper" / "ai_review.before_revision.json").read_bytes() == original[
        "ai_review.before_revision.json"
    ]
    assert _model(ctx, Role.AI_REVIEWER).call_count == 1


async def test_the_paper_graph_is_given_the_revision_inputs_and_the_fixed_options(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    write_original(workspace.root)
    stub = StubPaper(workspace.root)
    ctx = _context(make_context)
    options = EditOptions(
        edit_programs=["program_3"], reference_mode="refresh", figure_refresh_mode="preserve"
    )

    await _run(_graph(stub), ctx, edit_options=options, run_id="run-7")

    (payload,) = stub.calls
    assert payload["mode"] == "revision"
    assert payload["previous_paper_dir"] == "paper"
    assert payload["output_dir"] == "revision/attempt_01"
    assert payload["run_id"] == "run-7"
    chosen = payload["edit_options"]
    assert chosen.reference_mode == "preserve"
    assert chosen.figure_refresh_mode == "auto"
    assert chosen.skip_postdraft_enrichment is True
    assert chosen.edit_programs == ["program_3"]
    review = payload["external_review"]
    assert review.status == "completed"
    assert review.review_path == "paper/ai_review.before_revision.json"
    assert review.aggregate is not None
    assert review.aggregate.overall == 4
    assert payload["external_review_path"] == "paper/ai_review.before_revision.json"
    feedback = payload["revision_feedback"]
    assert feedback.untrusted_reviewer_data is True
    assert feedback.values["Weaknesses"] == ["Small evaluation.", "Claims are broad."]
    assert payload["run_config"] is not None


@pytest.mark.parametrize("has_them", [True, False])
async def test_post_hoc_and_final_results_reach_the_edit_only_when_the_paper_has_them(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace, has_them: bool
) -> None:
    # as upstream's revision command: the paper edit is given the post-hoc directory and the final
    # evaluation only when the paper being revised carries them (otherwise --no-final-evaluation)
    root = workspace.root
    write_original(root)
    data = paper_data()
    if not has_them:
        data["posthoc_analyses"] = []
        data.pop("final_evaluation")
        put(root, "paper/paper_data.json", data)
    stub = StubPaper(root, data=paper_data(revised=True) if has_them else data)
    marker = {"given": True}
    await _run(_graph(stub), _context(make_context), posthoc=marker, final=marker)
    payload = stub.calls[0]
    assert (payload.get("posthoc") == marker) is has_them
    assert (payload.get("final") == marker) is has_them


async def test_the_events_carry_the_before_and_after_assessments(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    write_original(workspace.root)
    ctx = _context(make_context, reviewer=[answer(review_dict(Overall=6, Decision="Accept"))])
    graph = _graph(StubPaper(workspace.root))
    await _run(graph, ctx)
    state = await graph.aget_state({"configurable": {"thread_id": "t"}})
    event = next(e for e in state.values["events"] if e.kind == "revision_result")
    assert event.data["before_overall_score"] == 4
    assert event.data["after_overall_score"] == 6
    assert event.data["overall_score_delta"] == 2
    assert event.data["before_decision"] == "Reject"
    assert event.data["after_decision"] == "Reject"  # the area chair's decision, scores averaged


async def test_a_saved_attempt_that_still_passes_is_published_without_a_new_edit(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    write_revision(root, "revision/attempt_01")
    stub = StubPaper(root)
    result = await _run(_graph(stub), _context(make_context))
    assert result.status == "promoted"
    assert stub.calls == []
    assert (root / "paper" / "paper.tex").read_text().endswith("revised")


async def test_an_invalid_saved_attempt_is_skipped_for_the_next_number(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    write_revision(root, "revision/attempt_01", skip=("validation_report.json",))
    stub = StubPaper(root)
    result = await _run(_graph(stub), _context(make_context))
    assert result.status == "promoted"
    assert [call["output_dir"] for call in stub.calls] == ["revision/attempt_02"]


async def test_the_revision_root_is_a_setting(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    stub = StubPaper(root)
    result = await _run(_graph(stub), _context(make_context), revision_root="work/rev")
    assert result.status == "promoted"
    assert stub.calls[0]["output_dir"] == "work/rev/attempt_01"


# --- the original is retained when the revision is not published --------------------------------


async def test_a_revision_that_loses_evidence_is_not_published(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    before = _published_digest(root)
    lost = paper_data(revised=True)
    lost["final_evaluation"] = {"outcome": "qualified", "metrics": {"score": 0.99}}
    stub = StubPaper(root, data=lost)
    ctx = _context(make_context)

    result = await _run(_graph(stub), ctx)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "revision_evidence_invalid"
    assert "revision loses or changes final evaluation inventory" in result.failure.detail
    assert _published_digest(root) == before
    assert not (root / "paper" / "revisions").exists()
    assert _model(ctx, Role.AI_REVIEWER).call_count == 0  # no second review of an unpublished paper


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.pop("all_programs"), "program inventory"),
        (lambda data: data["all_programs"].pop(0), "program inventory"),
        (lambda data: data.pop("grounding_sources"), "grounding inventory"),
        (lambda data: data.pop("posthoc_analyses"), "posthoc inventory"),
        (lambda data: data.pop("downstream_study"), "downstream study context"),
        (lambda data: data.pop("final_evaluation"), "final evaluation inventory"),
    ],
    ids=["programs", "one-program", "grounding", "posthoc", "downstream", "final"],
)
async def test_each_lost_inventory_is_named(
    make_context: Callable[..., ScientistContext],
    workspace: LocalWorkspace,
    change: Callable[[dict[str, Any]], Any],
    message: str,
) -> None:
    root = workspace.root
    write_original(root)
    data = paper_data(revised=True)
    change(data)
    result = await _run(_graph(StubPaper(root, data=data)), _context(make_context))
    assert result.status == "failed"
    assert result.failure is not None
    assert message in result.failure.detail


async def test_volatile_keys_locations_and_editorial_annotations_may_change(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    data = paper_data(revised=True)
    data["posthoc_analyses"][0]["key_finding_for_paper"] = "rewritten for the reviewer"
    result = await _run(_graph(StubPaper(root, data=data)), _context(make_context))
    assert result.status == "promoted"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"audit_override": {"verdict": "fail"}},
            "revision lacks a current passing agentic evidence audit",
        ),
        (
            {
                "audit_override": {
                    "finding_count": 1,
                    "findings": [{"id": "f1"}],
                }
            },
            "revision lacks a current passing agentic evidence audit",
        ),
        ({"report_override": {"ok": False}}, "revision validation did not pass"),
        (
            {"report_override": {"agentic_evidence_audit": {"checked": False}}},
            "revision lacks a current passing agentic evidence audit",
        ),
        ({"skip": ("method_scope_audit.md",)}, "method-scope audit: method_scope_audit.md"),
        ({"skip": ("method_scope_audit.json",)}, "method-scope audit: method_scope_audit.json"),
        ({"skip": ("paper_evidence_audit.json",)}, "agentic paper-evidence audit"),
        ({"skip": ("validation_report.json",)}, "validation_report.json"),
        ({"skip": ("paper.pdf",)}, "paper.pdf"),
        ({"skip": ("latest_manifest.json",)}, "latest_manifest.json"),
        ({"manifest": {"paper_pdf": "elsewhere/paper.pdf"}}, "does not identify the revised PDF"),
        ({"manifest": {}}, "does not identify the revised PDF"),
    ],
    ids=[
        "audit-fail",
        "audit-findings",
        "report-not-ok",
        "report-audit-unchecked",
        "scope-md-missing",
        "scope-json-missing",
        "audit-missing",
        "report-missing",
        "pdf-missing",
        "manifest-missing",
        "manifest-other-pdf",
        "manifest-without-pdf",
    ],
)
async def test_a_revision_without_a_passing_current_audit_is_not_published(
    make_context: Callable[..., ScientistContext],
    workspace: LocalWorkspace,
    kwargs: dict[str, Any],
    message: str,
) -> None:
    root = workspace.root
    write_original(root)
    before = _published_digest(root)
    result = await _run(_graph(StubPaper(root, **kwargs)), _context(make_context))
    assert result.status == "failed"
    assert result.failure is not None
    assert message in result.failure.detail
    assert _published_digest(root) == before


async def test_an_audit_whose_hashes_differ_from_the_reports_is_stale(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    stub = StubPaper(
        root,
        audit_override={"input_sha256": {"tex-source/paper.tex": sha("changed after the audit")}},
    )
    result = await _run(_graph(stub), _context(make_context))
    assert result.status == "failed"
    assert "current passing agentic evidence audit" in (
        result.failure.detail if result.failure else ""
    )


async def test_the_paper_packages_own_checks_can_be_supplied(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    seen: list[str] = []

    async def hashes(directory: str, runtime: Any) -> Mapping[str, str]:
        seen.append(directory)
        return {
            "evidence-source/paper_data.json": sha("data"),
            "tex-source/paper.tex": sha("tex2"),
            "evidence-source/new_note.txt": sha("new note"),
            "evidence-source/latest_manifest.json": sha("completion2"),
        }

    async def scope(directory: str, runtime: Any) -> bool:
        return True

    graph = _graph(
        StubPaper(root),
        **{"revision.audit_input_hashes": hashes, "revision.scope_audit": scope},
    )
    result = await _run(graph, _context(make_context))
    assert result.status == "promoted"
    assert seen == ["revision/attempt_01", "paper"]  # the revision, then the published paper


async def test_hashes_that_do_not_match_the_recomputed_ones_refuse_the_revision(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)

    async def hashes(directory: str, runtime: Any) -> Mapping[str, str]:
        return {"tex-source/paper.tex": sha("something else entirely")}

    async def scope(directory: str, runtime: Any) -> bool:
        return False

    before = _published_digest(root)
    graph = _graph(StubPaper(root), **{"revision.audit_input_hashes": hashes})
    result = await _run(graph, _context(make_context))
    assert result.status == "failed"
    assert _published_digest(root) == before

    graph = _graph(StubPaper(root), **{"revision.scope_audit": scope})
    result = await _run(graph, _context(make_context), thread="scope")
    assert result.status == "failed"
    assert "method-scope audit" in (result.failure.detail if result.failure else "")


async def test_a_paper_revision_that_does_not_complete_keeps_the_original(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    before = _published_digest(root)
    stub = StubPaper(
        root,
        result=lambda directory: PaperResult(
            status="incomplete", paper_dir=directory, incomplete_reason="paper validation failed"
        ),
    )
    ctx = _context(make_context)
    result = await _run(_graph(stub), ctx)
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "paper_revision_failed"
    assert "paper validation failed" in result.failure.detail
    assert result.paper is not None
    assert result.paper.status == "incomplete"
    assert _published_digest(root) == before
    assert _model(ctx, Role.AI_REVIEWER).call_count == 0


async def test_a_paper_graph_that_raises_keeps_the_original(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    before = _published_digest(root)
    result = await _run(
        _graph(StubPaper(root, error=RuntimeError("the writer died"))), _context(make_context)
    )
    assert result.status == "failed"
    assert result.failure is not None
    assert "the writer died" in result.failure.detail
    assert _published_digest(root) == before


async def test_a_publication_that_fails_midway_puts_the_original_back(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    before = _published_digest(root)
    # the audit binds an input that the revision does not have: copying it fails after the
    # snapshot, the archive and the overlay have been made
    bound = {
        "evidence-source/paper_data.json": sha("data"),
        "evidence-source/not_in_the_revision.txt": sha("x"),
    }
    result = await _run(_graph(StubPaper(root, bound=bound)), _context(make_context))
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "revision_publication_failed"
    assert "not a regular file" in result.failure.detail
    assert _published_digest(root) == before
    assert not (root / "paper" / "revisions").exists()


async def test_a_second_review_that_fails_keeps_the_revision_and_degrades_the_phase(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    bad = answer(review_dict(Overall=0))
    ctx = _context(make_context, reviewer=[bad, bad])

    result = await _run(_graph(StubPaper(root)), ctx)

    assert result.status == "degraded"
    assert result.second_review is not None
    assert result.second_review.status == "failed"
    assert result.failure is not None
    assert result.paper is not None
    assert (root / "paper" / "paper.tex").read_text().endswith("revised")
    assert (root / "paper" / "revisions" / "revision_01" / "before" / "paper.tex").is_file()
    manifest = read_json(root, "paper/latest_manifest.json")
    assert manifest["revision"]["status"] == "promoted"
    assert manifest["revision"]["rereview_status"] == "failed"
    assert "rereview_error" in manifest["revision"]
    assert manifest["revision"]["after_review"] is None
    assert not (root / "paper" / "ai_review.after_revision.json").exists()


async def test_a_second_review_slot_that_raises_degrades_the_phase_too(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)

    async def broken(payload: Any, runtime: Any) -> Any:
        raise RuntimeError("the reviewer service is down")

    result = await _run(_graph(StubPaper(root), review=broken), _context(make_context))
    assert result.status == "degraded"
    assert result.failure is not None
    assert "reviewer service is down" in result.failure.detail
    assert (root / "paper" / "paper.tex").read_text().endswith("revised")


async def test_the_second_review_can_be_replaced(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    seen: list[dict[str, Any]] = []

    async def replacement(payload: dict[str, Any], runtime: Any) -> ReviewResult:
        seen.append(payload)
        return ReviewResult(status="completed", review_path="somewhere.json")

    result = await _run(_graph(StubPaper(root), review=replacement), _context(make_context))
    assert result.status == "promoted"
    assert seen[0]["pdf_path"] == "paper/paper.pdf"
    assert seen[0]["artifact_stem"] == "ai_review.after_revision"
    assert seen[0]["manifest_key"] == "after_revision"
    assert read_json(root, "paper/latest_manifest.json")["revision"]["after_review"] == (
        "somewhere.json"
    )


# --- preconditions ------------------------------------------------------------------------------


async def test_without_a_completed_first_review_nothing_is_edited(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    (root / "paper" / "ai_review.before_revision.json").unlink()
    (root / "paper" / "ai_review.json").unlink()
    manifest = read_json(root, "paper/latest_manifest.json")
    manifest.pop("before_revision")
    put(root, "paper/latest_manifest.json", manifest)
    stub = StubPaper(root)
    result = await _run(_graph(stub), _context(make_context))
    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.category == "revision_precondition_failed"
    assert "completed initial AI review is unavailable" in result.failure.detail
    assert stub.calls == []


async def test_a_failed_review_result_is_not_a_first_review(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    write_original(workspace.root)
    stub = StubPaper(workspace.root)
    result = await _run(_graph(stub), _context(make_context), review=ReviewResult(status="failed"))
    assert result.status == "failed"
    assert stub.calls == []


async def test_the_latest_review_becomes_the_named_first_review(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    named = root / "paper" / "ai_review.before_revision.json"
    named.unlink()
    (root / "paper" / "ai_review.before_revision.log").unlink()
    put(root, "paper/ai_review.log", "legacy log\n")
    manifest = read_json(root, "paper/latest_manifest.json")
    manifest.pop("before_revision")
    put(root, "paper/latest_manifest.json", manifest)
    original_review = (root / "paper" / "ai_review.json").read_bytes()

    result = await _run(_graph(StubPaper(root)), _context(make_context))

    assert result.status == "promoted"
    assert named.read_bytes() == original_review
    assert (root / "paper" / "ai_review.before_revision.log").read_text() == "legacy log\n"
    manifest = read_json(root, "paper/latest_manifest.json")
    assert manifest["before_revision"]["review_json"] == "paper/ai_review.before_revision.json"
    assert (
        root / "paper" / "revisions" / "revision_01" / "before" / "ai_review.before_revision.json"
    ).read_bytes() == original_review


async def test_a_recorded_first_review_that_is_gone_is_not_replaced_by_the_latest(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    (root / "paper" / "ai_review.before_revision.json").unlink()
    result = await _run(_graph(StubPaper(root)), _context(make_context))
    assert result.status == "failed"
    assert result.failure is not None
    assert "immutable initial AI review is missing or invalid" in result.failure.detail


@pytest.mark.parametrize(
    ("review", "message"),
    [
        (b"not json", "valid UTF-8 JSON"),
        (b"\xff\xfe", "valid UTF-8 JSON"),
        (b"[1, 2]", "must be a JSON object"),
        (json.dumps(review_dict(Summary=["x"])).encode(), "Summary must be a string"),
        (json.dumps(review_dict(Strengths="x")).encode(), "Strengths must be a list of strings"),
        (json.dumps(review_dict(Weaknesses=["a", 1])).encode(), "list of strings"),
        (json.dumps(review_dict(Questions=["q"] * 101)).encode(), "at most 100 items"),
        (b" " * (1024 * 1024 + 1), "too large"),
    ],
)
async def test_a_review_that_cannot_serve_as_feedback_stops_the_revision(
    make_context: Callable[..., ScientistContext],
    workspace: LocalWorkspace,
    review: bytes,
    message: str,
) -> None:
    root = workspace.root
    write_original(root)
    put(root, "paper/ai_review.before_revision.json", review)
    stub = StubPaper(root)
    result = await _run(_graph(stub), _context(make_context))
    assert result.status == "failed"
    assert result.failure is not None
    assert message in result.failure.detail
    assert stub.calls == []


@pytest.mark.parametrize("manifest", [None, "not json", "[]"])
async def test_a_paper_without_a_usable_manifest_cannot_be_revised(
    make_context: Callable[..., ScientistContext],
    workspace: LocalWorkspace,
    manifest: str | None,
) -> None:
    root = workspace.root
    write_original(root)
    path = root / "paper" / "latest_manifest.json"
    if manifest is None:
        path.unlink()
    else:
        path.write_text(manifest)
    result = await _run(_graph(StubPaper(root)), _context(make_context))
    assert result.status == "failed"
    assert result.failure is not None
    assert "published paper manifest" in result.failure.detail


# --- replay of a promoted revision --------------------------------------------------------------


async def test_a_promoted_revision_is_not_edited_again(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    stub = StubPaper(root)
    ctx = _context(make_context)
    first = await _run(_graph(stub), ctx)
    assert first.status == "promoted"
    published = _published_digest(root)

    again = await _run(_graph(stub), ctx, thread="again")

    assert again.status == "promoted"
    assert len(stub.calls) == 1  # no second edit
    assert _model(ctx, Role.AI_REVIEWER).call_count == 1  # no third review
    assert again.second_review is not None
    assert again.second_review.review_path == "paper/ai_review.after_revision.json"
    assert again.before_dir == "paper/revisions/revision_01/before"
    assert _published_digest(root) == published


async def test_a_promoted_revision_without_its_second_review_is_only_reviewed(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    stub = StubPaper(root)
    bad = answer(review_dict(Overall=0))
    ctx = _context(make_context, reviewer=[bad, bad])
    degraded = await _run(_graph(stub), ctx)
    assert degraded.status == "degraded"

    ctx = _context(make_context, reviewer=[answer(review_dict(Overall=7))])
    again = await _run(_graph(stub), ctx, thread="retry")

    assert again.status == "promoted"
    assert len(stub.calls) == 1
    assert _model(ctx, Role.AI_REVIEWER).call_count == 1
    manifest = read_json(root, "paper/latest_manifest.json")
    assert manifest["revision"]["rereview_status"] == "completed"
    assert "rereview_error" not in manifest["revision"]
    assert read_json(root, "paper/ai_review.after_revision.json")["Overall"] == 7


async def test_a_promoted_paper_that_lost_its_snapshot_is_refused(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    stub = StubPaper(root)
    await _run(_graph(stub), _context(make_context))
    import shutil

    shutil.rmtree(root / "paper" / "revisions")
    result = await _run(_graph(stub), _context(make_context), thread="gone")
    assert result.status == "failed"
    assert result.failure is not None
    assert "original_snapshot is missing" in result.failure.detail


# --- hooks and resume ---------------------------------------------------------------------------


async def test_hooks_fire_at_every_step_in_order(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    write_original(workspace.root)
    seen: list[str] = []

    def hook(name: str) -> Callable[..., Any]:
        async def run(state: Mapping[str, Any], runtime: Any) -> None:
            seen.append(name)

        return run

    steps = (
        "check_preconditions",
        "stage_revision",
        "paper",
        "retention_check",
        "snapshot_original",
        "promote",
        "second_review",
        "finalize",
    )
    ctx = _context(
        make_context,
        hooks=Hooks(
            before={f"revision.{step}": [hook(f"before {step}")] for step in steps},
            after={f"revision.{step}": [hook(f"after {step}")] for step in steps},
        ),
    )
    await _run(_graph(StubPaper(workspace.root)), ctx)
    assert seen == [f"{when} {step}" for step in steps for when in ("before", "after")]


async def test_a_hook_may_change_the_edit_options_of_the_staging_step(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    write_original(workspace.root)
    stub = StubPaper(workspace.root)

    async def choose(state: Mapping[str, Any], runtime: Any) -> dict[str, Any]:
        return {"edit_options": EditOptions(old_programs=["program_0"])}

    ctx = _context(make_context, hooks=Hooks(after={"revision.stage_revision": [choose]}))
    await _run(_graph(stub), ctx)
    chosen = stub.calls[0]["edit_options"]
    assert chosen.old_programs == ["program_0"]
    assert chosen.reference_mode == "preserve"


async def test_a_crash_before_the_promotion_does_not_repeat_the_edit_or_the_review(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    crashes = {"left": 1}

    async def crash_once(state: Mapping[str, Any], runtime: Any) -> None:
        if crashes["left"]:
            crashes["left"] -= 1
            raise RuntimeError("the host died")

    stub = StubPaper(root)
    ctx = _context(make_context, hooks=Hooks(before={"revision.promote": [crash_once]}))
    graph = _graph(stub)
    config = {"configurable": {"thread_id": "crash"}}
    with pytest.raises(RuntimeError, match="host died"):
        await graph.ainvoke({"paper_dir": "paper"}, config, context=ctx)
    assert len(stub.calls) == 1
    assert not (root / "paper" / "revisions").exists()

    out = await graph.ainvoke(None, config, context=ctx)

    assert out["result"].status == "promoted"
    assert len(stub.calls) == 1  # the edit is not done twice
    assert _model(ctx, Role.AI_REVIEWER).call_count == 1
    assert (root / "paper" / "revisions" / "revision_01" / "before" / "paper.tex").is_file()


async def test_the_second_review_is_the_review_graph_by_default(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    root = workspace.root
    write_original(root)
    ctx = _context(make_context, reviewer=[answer(review_dict(Overall=8))])
    result = await _run(_graph(StubPaper(root)), ctx)
    assert result.second_review is not None
    assert result.second_review.aggregate is not None
    assert result.second_review.aggregate.overall == 8  # the chair's answer, scores replaced
    assert [r.overall for r in result.second_review.reviews] == [8]


async def test_the_revision_needs_a_paper_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(revision_graph, "importlib", SimpleNamespace(import_module=missing))
    with pytest.raises(ImportError, match=r"revision\.paper"):
        build_revision_graph(overrides=GraphOverrides())


async def test_the_package_paper_graph_is_the_default_slot(
    make_context: Callable[..., ScientistContext],
    workspace: LocalWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_original(workspace.root)
    stub = StubPaper(workspace.root)
    built: list[dict[str, Any]] = []

    def build_paper_graph(**kwargs: Any) -> StubPaper:
        built.append(kwargs)
        return stub

    fake = SimpleNamespace(
        import_module=lambda name: SimpleNamespace(build_paper_graph=build_paper_graph)
    )
    monkeypatch.setattr(revision_graph, "importlib", fake)
    overrides = GraphOverrides()
    graph = build_revision_graph(checkpointer=InMemorySaver(), overrides=overrides)
    assert built == [{"overrides": overrides, "name": "revision_paper"}]
    result = await _run(graph, _context(make_context))
    assert result.status == "promoted"
    assert len(stub.calls) == 1


async def test_the_paper_packages_checks_are_the_default(
    make_context: Callable[..., ScientistContext],
    workspace: LocalWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_original(workspace.root)
    seen: list[tuple[str, str]] = []

    async def audit_input_sha256(ws: Any, directory: str) -> dict[str, str]:
        seen.append(("hashes", directory))
        data = await ws.read_bytes(f"{directory}/paper_evidence_audit.json")
        return json.loads(data)["input_sha256"]  # type: ignore[no-any-return]

    async def validate_method_scope_audit_in(ws: Any, directory: str) -> dict[str, Any]:
        seen.append(("scope", directory))
        return {"ok": True}

    modules = {
        "langgraph_agentic_scientist.paper.audit": SimpleNamespace(
            audit_input_sha256=audit_input_sha256
        ),
        "langgraph_agentic_scientist.paper.validation.scope_audit": SimpleNamespace(
            validate_method_scope_audit_in=validate_method_scope_audit_in
        ),
    }
    fake = SimpleNamespace(import_module=lambda name: modules[name])
    monkeypatch.setattr(revision_graph, "importlib", fake)
    overrides = GraphOverrides(slots={"revision.paper": StubPaper(workspace.root)})
    graph = build_revision_graph(checkpointer=InMemorySaver(), overrides=overrides)
    result = await _run(graph, _context(make_context))
    assert result.status == "promoted"
    assert ("hashes", "revision/attempt_01") in seen
    assert ("scope", "revision/attempt_01") in seen
    assert ("hashes", "paper") in seen  # the published paper is checked again
    assert result.paper is not None


async def test_without_the_paper_packages_checks_the_recorded_files_are_compared(
    make_context: Callable[..., ScientistContext],
    workspace: LocalWorkspace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_original(workspace.root)

    def missing(name: str) -> Any:
        raise ImportError(name)

    monkeypatch.setattr(revision_graph, "importlib", SimpleNamespace(import_module=missing))
    overrides = GraphOverrides(slots={"revision.paper": StubPaper(workspace.root)})
    graph = build_revision_graph(checkpointer=InMemorySaver(), overrides=overrides)
    ctx = _context(make_context)
    result = await _run(graph, ctx)
    assert result.status == "promoted"
    state = await graph.aget_state({"configurable": {"thread_id": "t"}})
    assert state.values["retention"].skipped_checks == [
        "method_scope_audit validity",
        "audit input hashes",
    ]
    assert state.values["promotion"].skipped_checks == [
        "method_scope_audit validity",
        "audit input hashes",
    ]


async def test_the_paper_packages_real_audit_hashes_agree_with_a_published_revision(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    audit = pytest.importorskip("langgraph_agentic_scientist.paper.audit")
    if not hasattr(audit, "audit_input_sha256"):
        pytest.skip("the paper package has no audit_input_sha256 yet")
    root = workspace.root
    write_original(root)

    class Consistent(StubPaper):
        """A paper graph whose revision records the hashes the paper package computes for it."""

        async def __call__(self, payload: dict[str, Any], runtime: Any) -> PaperResult:
            result = await super().__call__(payload, runtime)
            directory = payload["output_dir"]
            hashes = await audit.audit_input_sha256(runtime.context.workspace, directory)
            files = audit_files(hashes)
            for name, content in files.items():
                put(self.root, f"{directory}/{name}", content)
            return result

    overrides = GraphOverrides(
        slots={"revision.paper": Consistent(root), "revision.scope_audit": _scope_ok}
    )
    graph = build_revision_graph(checkpointer=InMemorySaver(), overrides=overrides)
    result = await _run(graph, _context(make_context))
    assert result.failure is None, result.failure
    assert result.status == "promoted"


class _PaperIn(TypedDict, total=False):
    mode: str
    previous_paper_dir: str
    output_dir: str
    external_review: Any
    edit_options: Any
    revision_feedback: Any
    run_config: Any


class _PaperOut(TypedDict):
    result: PaperResult


def _compiled_paper(root: Path, seen: list[str], *, ask: bool = False) -> Any:
    """A compiled paper graph of its own state, as the real one is: used as the slot."""

    async def revise(state: _PaperIn, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
        seen.append(f"revise {state['mode']} with {type(runtime.context).__name__}")
        if ask:
            assert interrupt("may the writer start?") == "yes"
        directory = state["output_dir"]
        write_revision(root, directory)
        return {"result": PaperResult(status="complete", paper_dir=directory)}

    builder: StateGraph[_PaperIn, ScientistContext, _PaperIn, _PaperOut] = StateGraph(
        _PaperIn, context_schema=ScientistContext, input_schema=_PaperIn, output_schema=_PaperOut
    )
    builder.add_node("revise", revise)
    builder.add_edge(START, "revise")
    return builder.compile(name="paper")


async def test_a_compiled_paper_graph_runs_as_a_subgraph_of_the_revision(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    write_original(workspace.root)
    seen: list[str] = []
    graph = _graph(_compiled_paper(workspace.root, seen))
    result = await _run(graph, _context(make_context))
    assert result.status == "promoted"
    assert seen == ["revise revision with ScientistContext"]  # the run context and the input
    assert result.paper is not None
    assert result.paper.paper_dir == "paper"  # as published


async def test_a_question_asked_inside_the_paper_graph_reaches_the_caller_and_resumes(
    make_context: Callable[..., ScientistContext], workspace: LocalWorkspace
) -> None:
    write_original(workspace.root)
    seen: list[str] = []
    ctx = _context(make_context)
    graph = _graph(_compiled_paper(workspace.root, seen, ask=True))
    config = {"configurable": {"thread_id": "ask"}}

    out = await graph.ainvoke({"paper_dir": "paper"}, config, context=ctx)
    assert out["__interrupt__"][0].value == "may the writer start?"
    assert not (workspace.root / "paper" / "revisions").exists()

    out = await graph.ainvoke(Command(resume="yes"), config, context=ctx)
    assert out["result"].status == "promoted"
    assert (workspace.root / "paper" / "revisions" / "revision_01" / "before").is_dir()
