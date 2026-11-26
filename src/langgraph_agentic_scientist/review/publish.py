"""Publishing a review next to the paper: artifacts and the manifest entry.

Upstream's final review phase (``final_ai_review.py``) leaves, in the paper directory,
``<stem>.json`` and ``<stem>.log`` and the compatibility copies ``ai_review.json`` and
``ai_review.log`` of the latest review, and records the outcome in ``latest_manifest.json``
under ``manifest_key`` and under ``ai_review``. A failed review clears the JSON copy instead of
leaving a stale assessment, and writes the failure to the log. The review is never gating: a
failure is reported, and the paper stays.

The port writes the same files through the workspace and updates ``latest_manifest.json`` when it
exists in the paper directory (a manuscript reviewed outside the pipeline has none). Paths in the
manifest are workspace paths, which the revision reads back.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from ..workspace import Workspace, is_file, join, write_text
from .models import DEFAULT_ARTIFACT_STEM

LATEST_REVIEW = "ai_review.json"
LATEST_LOG = "ai_review.log"
MANIFEST_NAME = "latest_manifest.json"
MAX_MANIFEST_BYTES = 1024 * 1024
"""Upstream reads the paper manifest with a 1 MiB bound (``_PAPER_MANIFEST_MAX_BYTES``)."""

_FAILURE_HEADER = "[ai review failure]"


class ManifestError(ValueError):
    """The paper manifest exists but is not a JSON object."""


async def read_manifest(workspace: Workspace, paper_dir: str) -> dict[str, Any] | None:
    """Read ``latest_manifest.json`` of ``paper_dir``; ``None`` when there is none.

    Raises:
        ManifestError: When the file is not valid JSON, not an object, or larger than 1 MiB.
    """
    path = join(paper_dir, MANIFEST_NAME)
    if not await is_file(workspace, path):
        return None
    data = await workspace.read_bytes(path)
    if len(data) > MAX_MANIFEST_BYTES:
        raise ManifestError("paper manifest is too large")
    try:
        manifest = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ManifestError("invalid paper manifest JSON") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("paper manifest must contain a JSON object")
    return manifest


async def write_manifest(workspace: Workspace, paper_dir: str, manifest: dict[str, Any]) -> None:
    """Write ``latest_manifest.json`` (sorted keys, two-space indent, as upstream)."""
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    await write_text(workspace, join(paper_dir, MANIFEST_NAME), text)


def failure_log(log_text: str, error: str) -> str:
    """Return the log of a failed review: the notes, then the failure record (upstream's form)."""
    record = f"{_FAILURE_HEADER}\n{error}"
    current = log_text.rstrip()
    return f"{current}\n\n{record}\n" if current else f"{record}\n"


async def publish_review(
    workspace: Workspace,
    *,
    paper_dir: str,
    artifact_stem: str,
    manifest_key: str,
    status: Literal["completed", "failed"],
    payload: dict[str, Any] | None,
    log_text: str,
    error: str | None,
    update_manifest: bool = True,
) -> str | None:
    """Write the review artifacts and the manifest entry; return the review path (or ``None``).

    Args:
        workspace: The workspace.
        paper_dir: The paper directory the artifacts are written to.
        artifact_stem: ``<stem>.json`` and ``<stem>.log`` are written, and the latest copies
            ``ai_review.json`` and ``ai_review.log``.
        manifest_key: The key of the status in ``latest_manifest.json``.
        status: ``completed`` writes the review; ``failed`` clears the latest JSON copy and
            writes the failure to the log.
        payload: The review as upstream's JSON object (for ``completed``).
        log_text: The notes of the run.
        error: The failure (for ``failed``).
        update_manifest: Whether to read and update ``latest_manifest.json``.

    Raises:
        ManifestError: When the manifest is present but unusable; nothing is then written.
    """
    manifest = await read_manifest(workspace, paper_dir) if update_manifest else None
    review_path = join(paper_dir, f"{artifact_stem}.json")
    log_path = join(paper_dir, f"{artifact_stem}.log")
    latest_review = join(paper_dir, LATEST_REVIEW)
    latest_log = join(paper_dir, LATEST_LOG)
    completed = status == "completed" and payload is not None
    if completed:
        assert payload is not None
        text = json.dumps(payload, indent=2) + "\n"
        await write_text(workspace, review_path, text)
        if review_path != latest_review:
            await write_text(workspace, latest_review, text)
        log = log_text
    else:
        await workspace.remove(latest_review)
        log = failure_log(log_text, error or "review failed")
    await write_text(workspace, log_path, log)
    if log_path != latest_log:
        await write_text(workspace, latest_log, log)
    if manifest is not None:
        state = "completed" if completed else "failed"
        manifest[manifest_key] = {
            "status": state,
            "review_json": review_path if completed else None,
            "log": log_path,
        }
        manifest["ai_review"] = {
            "status": state,
            "review_json": latest_review if completed else None,
            "log": latest_log,
        }
        await write_manifest(workspace, paper_dir, manifest)
    return review_path if completed else None


async def read_review_json(workspace: Workspace, path: str) -> dict[str, Any]:
    """Read a published review as a JSON object (at most 4 MiB, as upstream).

    Raises:
        ValueError: When the file is not a JSON object or is too large.
    """
    data = await workspace.read_bytes(path)
    if len(data) > 4 * 1024 * 1024:
        raise ValueError(f"{path} is too large")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} does not contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


__all__ = [
    "DEFAULT_ARTIFACT_STEM",
    "LATEST_LOG",
    "LATEST_REVIEW",
    "MANIFEST_NAME",
    "ManifestError",
    "failure_log",
    "publish_review",
    "read_manifest",
    "read_review_json",
    "write_manifest",
]
