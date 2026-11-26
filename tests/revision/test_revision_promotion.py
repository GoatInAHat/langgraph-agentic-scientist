"""Publication of a revision: bound inputs, manifests, the workspace checks and the rollback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from revision_helpers import PAPER, put, read_json, sha, tree, write_original, write_revision

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.revision.evidence import EvidenceError
from langgraph_agentic_scientist.revision.promotion import (
    BEFORE_DIR_NAME,
    CopyBudget,
    PublicationError,
    audit_bound_paths,
    copy_audit_bound_inputs,
    copy_directory_contents,
    initial_review_must_be_preserved,
    names_path,
    promote_revision,
    promoted_original_snapshot,
    rebase_manifest,
    validate_revision_workspace,
    workspace_path,
)

H = sha("x")


# --- audit-bound paths --------------------------------------------------------------------------


def test_bound_files_lose_their_prefixes_and_trees_stand_for_what_is_in_them() -> None:
    files, trees = audit_bound_paths(
        {
            "evidence-source/paper_data.json": H,
            "tex-source/paper.tex": H,
            "compile-input/main.tex": H,
            "validator-input/v.json": H,
            "figure-asset/f.png": H,
            "plain/name.txt": H,
            "evidence-tree/sources": H,
            "evidence-tree/sources/deep": H,
            "evidence-source/sources/inside.txt": H,
        }
    )
    assert trees == ["sources"]  # the nested tree and the file inside it are covered by it
    assert files == [
        "f.png",
        "main.tex",
        "paper.tex",
        "paper_data.json",
        "plain/name.txt",
        "v.json",
    ]


@pytest.mark.parametrize(
    "inputs",
    [
        {"evidence-source/../etc": H},
        {"evidence-source//x": H},
        {"/abs": H},
        {"evidence-source/a\\b": H},
        {"evidence-source/": H},
        {"evidence-source/x": "short"},
        {"evidence-source/x": "A" * 64},
        {"evidence-source/x": 5},
        {"evidence-tree/t": H, "evidence-source/t": H},
        [("evidence-source/x", H)],
        None,
    ],
)
def test_unsafe_or_malformed_bound_paths_are_refused(inputs: Any) -> None:
    with pytest.raises(PublicationError):
        audit_bound_paths(inputs)


def test_too_many_bound_paths_are_refused() -> None:
    with pytest.raises(PublicationError, match="invalid input_sha256 paths"):
        audit_bound_paths({f"evidence-source/f{index}": H for index in range(4097)})


async def test_a_bound_tree_replaces_what_is_at_its_target(workspace: LocalWorkspace) -> None:
    put(workspace.root, "rev/sources/a.txt", "new a")
    put(workspace.root, "rev/sources/sub/b.txt", "new b")
    put(workspace.root, "paper/sources/old.txt", "stale")
    await copy_audit_bound_inputs(
        workspace, "rev", "paper", {"evidence-tree/sources": H}, CopyBudget()
    )
    assert tree(workspace.root, "paper/sources") == {"a.txt": b"new a", "sub/b.txt": b"new b"}


async def test_a_bound_symlink_is_refused(workspace: LocalWorkspace, tmp_path: Path) -> None:
    put(workspace.root, "rev/real.txt", "r")
    os.symlink(workspace.root / "rev" / "real.txt", workspace.root / "rev" / "link.txt")
    with pytest.raises(PublicationError, match="not a regular file"):
        await copy_audit_bound_inputs(
            workspace, "rev", "paper", {"evidence-source/link.txt": H}, CopyBudget()
        )


# --- copying ------------------------------------------------------------------------------------


async def test_a_directory_is_merged_into_its_target(workspace: LocalWorkspace) -> None:
    put(workspace.root, "src/a.png", b"a")
    put(workspace.root, "src/sub/b.png", b"b")
    put(workspace.root, "dst/keep.png", b"keep")
    copied = await copy_directory_contents(workspace, "src", "dst", CopyBudget())
    assert copied == ["a.png", "sub/b.png"]
    assert tree(workspace.root, "dst") == {"a.png": b"a", "sub/b.png": b"b", "keep.png": b"keep"}


async def test_a_symlink_in_a_copied_tree_is_refused(workspace: LocalWorkspace) -> None:
    put(workspace.root, "src/a.png", b"a")
    os.symlink("a.png", workspace.root / "src" / "link.png")
    with pytest.raises(PublicationError, match="unexpected symlink"):
        await copy_directory_contents(workspace, "src", "dst", CopyBudget())


def test_the_copy_bounds_are_upstreams() -> None:
    budget = CopyBudget()
    with pytest.raises(PublicationError, match="too large"):
        budget.add(size=1024 * 1024 * 1024 + 1)
    budget = CopyBudget(entries=100_000)
    with pytest.raises(PublicationError, match="too many entries"):
        budget.add()


# --- manifests ----------------------------------------------------------------------------------


def test_paths_in_either_form_name_workspace_paths(workspace: LocalWorkspace) -> None:
    host = workspace.host_path("paper/paper.pdf")
    assert workspace_path(workspace, "paper/paper.pdf") == "paper/paper.pdf"
    assert workspace_path(workspace, "./paper/paper.pdf") == "paper/paper.pdf"
    assert workspace_path(workspace, host) == "paper/paper.pdf"
    assert workspace_path(workspace, "/somewhere/else/paper.pdf") is None
    assert workspace_path(workspace, "../outside") is None
    assert names_path(workspace, host, "paper/paper.pdf")
    assert names_path(workspace, "paper/paper.pdf", "paper/paper.pdf")
    assert not names_path(workspace, 5, "paper/paper.pdf")
    assert not names_path(workspace, "other/paper.pdf", "paper/paper.pdf")


def test_a_manifest_is_rebased_in_the_form_its_paths_have(workspace: LocalWorkspace) -> None:
    old_host = workspace.host_path("rev/attempt_01/paper.pdf")
    manifest: dict[str, Any] = {
        "paper_pdf": "rev/attempt_01/paper.pdf",
        "absolute": old_host,
        "figures": ["rev/attempt_01/figures/f.png", "elsewhere/x.png"],
        "nested": {"dir": "rev/attempt_01"},
        "text": "rev/attempt_011/not-inside",
        "number": 3,
        "empty": "",
    }
    moved = rebase_manifest(workspace, manifest, old_root="rev/attempt_01", new_root="paper")
    assert moved == {
        "paper_pdf": "paper/paper.pdf",
        "absolute": workspace.host_path("paper/paper.pdf"),
        "figures": ["paper/figures/f.png", "elsewhere/x.png"],
        "nested": {"dir": "paper"},
        "text": "rev/attempt_011/not-inside",
        "number": 3,
        "empty": "",
    }


def test_a_manifest_path_that_climbs_out_of_the_revision_is_refused(
    workspace: LocalWorkspace,
) -> None:
    with pytest.raises(PublicationError, match="escapes revision root"):
        rebase_manifest(
            workspace,
            {"x": "rev/attempt_01/../attempt_01/x"},
            old_root="rev/attempt_01",
            new_root="paper",
        )
    # a path with dots that leaves the root is not ours to move
    assert rebase_manifest(
        workspace, {"x": "rev/attempt_01/../other"}, old_root="rev/attempt_01", new_root="paper"
    ) == {"x": "rev/attempt_01/../other"}


# --- the workspace check ------------------------------------------------------------------------


async def test_a_complete_validated_revision_passes(workspace: LocalWorkspace) -> None:
    write_revision(workspace.root, "rev")
    manifest = await validate_revision_workspace(workspace, "rev")
    assert manifest["paper_pdf"] == "rev/paper.pdf"


async def test_a_manifest_may_name_the_pdf_by_its_host_path(workspace: LocalWorkspace) -> None:
    write_revision(
        workspace.root, "rev", manifest={"paper_pdf": workspace.host_path("rev/paper.pdf")}
    )
    await validate_revision_workspace(workspace, "rev")


async def test_a_revision_directory_must_exist_and_be_a_directory(
    workspace: LocalWorkspace,
) -> None:
    with pytest.raises(PublicationError, match="must be a regular directory"):
        await validate_revision_workspace(workspace, "missing")
    put(workspace.root, "file", "x")
    with pytest.raises(PublicationError, match="must be a regular directory"):
        await validate_revision_workspace(workspace, "file")


async def test_a_symlinked_file_or_figure_is_refused(workspace: LocalWorkspace) -> None:
    write_revision(workspace.root, "rev")
    os.symlink("f1.png", workspace.root / "rev" / "figures" / "link.png")
    with pytest.raises(PublicationError, match="irregular entry"):
        await validate_revision_workspace(workspace, "rev")
    os.unlink(workspace.root / "rev" / "figures" / "link.png")
    (workspace.root / "rev" / "paper.pdf").unlink()
    os.symlink("paper.tex", workspace.root / "rev" / "paper.pdf")
    with pytest.raises(PublicationError, match="expected regular file"):
        await validate_revision_workspace(workspace, "rev")


async def test_a_revision_that_did_not_validate_is_refused(workspace: LocalWorkspace) -> None:
    write_revision(workspace.root, "rev", report_override={"ok": False})
    with pytest.raises(PublicationError, match="validation did not pass"):
        await validate_revision_workspace(workspace, "rev")


async def test_the_validation_report_may_say_passed_instead_of_ok(
    workspace: LocalWorkspace,
) -> None:
    write_revision(workspace.root, "rev")
    report = read_json(workspace.root, "rev/validation_report.json")
    report.pop("ok")
    report["passed"] = True
    put(workspace.root, "rev/validation_report.json", report)
    await validate_revision_workspace(workspace, "rev")


# --- the first review ---------------------------------------------------------------------------


async def test_the_first_review_must_be_preserved_in_the_cases_upstream_names(
    workspace: LocalWorkspace,
) -> None:
    root = workspace.root
    put(root, "paper/other.txt", "x")
    assert not await initial_review_must_be_preserved(workspace, "paper")
    put(root, "paper/latest_manifest.json", {"revision": {"status": "promoted"}})
    assert await initial_review_must_be_preserved(workspace, "paper")
    put(
        root,
        "paper/latest_manifest.json",
        {
            "before_revision": {
                "status": "completed",
                "review_json": "paper/ai_review.before_revision.json",
            }
        },
    )
    assert await initial_review_must_be_preserved(workspace, "paper")
    put(
        root,
        "paper/latest_manifest.json",
        {
            "before_revision": {
                "status": "failed",
                "review_json": "paper/ai_review.before_revision.json",
            }
        },
    )
    assert not await initial_review_must_be_preserved(workspace, "paper")
    put(root, "paper/revisions/revision_01/promotion_state.json", {"status": "promoting"})
    assert await initial_review_must_be_preserved(workspace, "paper")
    put(root, "paper/revisions/revision_01/promotion_state.json", {"status": "completed"})
    assert not await initial_review_must_be_preserved(workspace, "paper")


async def test_a_recorded_original_snapshot_must_be_inside_the_revisions_and_a_directory(
    workspace: LocalWorkspace,
) -> None:
    root = workspace.root
    put(root, "paper/revisions/revision_01/before/x", "x")
    ok = await promoted_original_snapshot(
        workspace, "paper", {"original_snapshot": "paper/revisions/revision_01/before"}
    )
    assert ok == "paper/revisions/revision_01/before"
    for value, message in (
        (None, "missing original_snapshot"),
        ("", "missing original_snapshot"),
        ("elsewhere/before", "outside the paper revision root"),
        ("paper/revisions/revision_02/before", "is missing"),
        ("paper/revisions/revision_01/before/x", "not a directory"),
    ):
        with pytest.raises(PublicationError, match=message):
            await promoted_original_snapshot(workspace, "paper", {"original_snapshot": value})
    os.symlink("revision_01", root / "paper" / "revisions" / "linked")
    with pytest.raises(PublicationError, match="symlink"):
        await promoted_original_snapshot(
            workspace, "paper", {"original_snapshot": "paper/revisions/linked/before"}
        )


# --- promotion and rollback ---------------------------------------------------------------------


async def test_a_promotion_that_died_midway_is_restarted_from_the_original(
    workspace: LocalWorkspace,
) -> None:
    root = workspace.root
    original = write_original(root)
    write_revision(root, "revision/attempt_01")
    snapshot = await workspace.snapshot(PAPER)
    # an earlier attempt overlaid some files and left a half-written snapshot behind
    put(root, "paper/paper.tex", "half overlaid")
    put(root, f"paper/{BEFORE_DIR_NAME}/junk.txt", "partial")
    (root / "paper" / "old_note.txt").unlink()

    outcome = await promote_revision(
        workspace,
        paper_dir=PAPER,
        revision_dir="revision/attempt_01",
        before_review_path="paper/ai_review.before_revision.json",
        original=snapshot,
    )

    before = root / "paper" / "revisions" / "revision_01" / "before"
    assert outcome.before_dir == "paper/revisions/revision_01/before"
    assert not (before / "junk.txt").exists()
    assert (before / "paper.tex").read_bytes() == original["paper.tex"]
    assert (before / "ancillary" / "old_note.txt").read_text() == "old note"
    assert outcome.archived == ["old_note.txt", "sources/a.txt", "sources/deep/b.txt"]
    assert (root / "paper" / "paper.tex").read_text().endswith("revised")


async def test_a_failed_promotion_restores_the_original_exactly(workspace: LocalWorkspace) -> None:
    root = workspace.root
    write_original(root)
    write_revision(root, "revision/attempt_01", data={"final_evaluation": {"outcome": "changed"}})
    original_tree = tree(root, PAPER)
    snapshot = await workspace.snapshot(PAPER)
    with pytest.raises(EvidenceError, match="loses or changes"):
        await promote_revision(
            workspace,
            paper_dir=PAPER,
            revision_dir="revision/attempt_01",
            before_review_path="paper/ai_review.before_revision.json",
            original=snapshot,
        )
    assert tree(root, PAPER) == original_tree
    assert not (root / "paper" / "revisions").exists()


async def test_the_snapshot_is_created_once(workspace: LocalWorkspace) -> None:
    root = workspace.root
    write_original(root)
    write_revision(root, "revision/attempt_01")
    put(root, f"paper/{BEFORE_DIR_NAME}/already", "an earlier promotion")
    snapshot = await workspace.snapshot(PAPER)  # the original already has a snapshot directory
    with pytest.raises(PublicationError, match="revision snapshot already exists"):
        await promote_revision(
            workspace,
            paper_dir=PAPER,
            revision_dir="revision/attempt_01",
            before_review_path="paper/ai_review.before_revision.json",
            original=snapshot,
        )
    assert (root / "paper" / BEFORE_DIR_NAME / "already").read_text() == "an earlier promotion"


async def test_the_first_review_must_be_a_file_of_the_paper_directory(
    workspace: LocalWorkspace,
) -> None:
    root = workspace.root
    write_original(root)
    write_revision(root, "revision/attempt_01")
    put(root, "elsewhere/review.json", "{}")
    snapshot = await workspace.snapshot(PAPER)
    with pytest.raises(PublicationError, match="file of the paper directory"):
        await promote_revision(
            workspace,
            paper_dir=PAPER,
            revision_dir="revision/attempt_01",
            before_review_path="elsewhere/review.json",
            original=snapshot,
        )


async def test_without_an_original_audit_nothing_is_archived(workspace: LocalWorkspace) -> None:
    root = workspace.root
    write_original(root)
    (root / "paper" / "paper_evidence_audit.json").unlink()
    write_revision(root, "revision/attempt_01")
    snapshot = await workspace.snapshot(PAPER)
    outcome = await promote_revision(
        workspace,
        paper_dir=PAPER,
        revision_dir="revision/attempt_01",
        before_review_path="paper/ai_review.before_revision.json",
        original=snapshot,
    )
    assert outcome.archived == []
    assert (root / "paper" / "old_note.txt").is_file()
