"""``seed_edit_workspace`` of ``paper_edit.sh`` and ``paper_revision_workspace.py``."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from langgraph_agentic_scientist.core.results import EditOptions
from langgraph_agentic_scientist.paper import workspace as paper_workspace
from langgraph_agentic_scientist.paper.workspace import (
    PaperPreparationError,
    seed_edit_workspace,
)

OLD = "old/paper"
NEW = "new/paper"


async def _seed(ctx: Any, paper_data: dict[str, Any] | None = None) -> dict[str, Any]:
    return await seed_edit_workspace(
        ctx,
        paper_dir=NEW,
        previous_paper_dir=OLD,
        options=EditOptions(),
        reference_mode="refresh",
        paper_data=paper_data,
    )


async def _context(workspace: Any) -> dict[str, Any]:
    return json.loads((await workspace.read_bytes(f"{NEW}/paper_edit_context.json")).decode())


async def test_posthoc_analyses_of_the_new_data_alone_are_reported_as_seeded(
    make_context: Any, workspace: Any
) -> None:
    await workspace.write_bytes(f"{OLD}/paper.tex", b"old")
    seeded = await _seed(make_context(), {"posthoc_analyses": [{"analysis_id": "a1", "rank": 1}]})
    assert "posthoc_analyses" in seeded["seeded"]
    assert "posthoc_analyses" in (await _context(workspace))["seeded_workspace_artifacts"]


@pytest.mark.parametrize(
    "previous",
    [
        b"{not json",
        b"[1, 2]",
        json.dumps({"posthoc_analyses": [{"analysis_id": "a1", "rank": 0}]}).encode(),
        json.dumps({"posthoc_analyses": [{"analysis_id": "a1", "rank": True}]}).encode(),
    ],
)
async def test_previous_paper_data_that_upstream_refuses_fails_the_preparation(
    make_context: Any, workspace: Any, previous: bytes
) -> None:
    await workspace.write_bytes(f"{OLD}/paper_data.json", previous)
    with pytest.raises(PaperPreparationError, match="Could not preserve revision posthoc"):
        await _seed(make_context(), {})


async def test_a_seeded_artifact_that_is_a_link_is_recorded_as_skipped(
    make_context: Any, workspace: Any
) -> None:
    await workspace.write_bytes(f"{OLD}/real.tex", b"old")
    os.symlink("real.tex", workspace.root / OLD / "paper.tex")
    seeded = await _seed(make_context())
    assert "paper.tex" not in seeded["seeded"]
    skipped = (await _context(workspace))["skipped_snapshot_entries"]
    assert {"path": f"{OLD}/paper.tex", "reason": "symlink"} in skipped
    assert {"path": f"{OLD}/paper.tex", "reason": "non_regular"} in skipped


async def test_the_snapshot_is_taken_again_from_the_edited_paper(
    make_context: Any, workspace: Any
) -> None:
    await workspace.write_bytes(f"{OLD}/paper.tex", b"old")
    await workspace.write_bytes(f"{NEW}/previous_paper/stale.tex", b"partial copy")
    await _seed(make_context())
    assert not await workspace.exists(f"{NEW}/previous_paper/stale.tex")
    assert await workspace.read_bytes(f"{NEW}/previous_paper/paper.tex") == b"old"


async def test_a_snapshot_of_more_files_than_upstream_allows_is_refused(
    make_context: Any, workspace: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper_workspace, "MAX_SNAPSHOT_FILES", 2)
    for name in ("a.tex", "b.tex", "c.tex"):
        await workspace.write_bytes(f"{OLD}/sections/{name}", b"x")
    await workspace.write_bytes(f"{OLD}/notes/d.tex", b"not snapshotted")
    with pytest.raises(PaperPreparationError, match="too many files"):
        await _seed(make_context())


async def test_a_snapshot_larger_than_upstream_allows_is_refused(
    make_context: Any, workspace: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(paper_workspace, "MAX_SNAPSHOT_BYTES", 4)
    await workspace.write_bytes(f"{OLD}/paper.tex", b"12345")
    with pytest.raises(PaperPreparationError, match="too large"):
        await _seed(make_context())
