"""Stage 6: aggregate and publish the valid analyses (upstream ``publish.py``).

The stage chooses no winner: ranking is the paper's job. It writes ``ranking.json`` (the
compatibility file listing ``valid`` and ``rejected`` analyses, ``delegated_to_paper_agent``
true), ``latest_manifest.json`` (the hand-off to the paper, only when an analysis is valid) and
``pipeline_summary.json``, all below the output directory; the paper's data builder reads
``ranking.json`` and the attempt directories it names.

Upstream also copies ``latest_manifest.json`` to the parent of the output directory, the stable
pointer that finds the newest run among several below ``posthoc/``. The port does the same when
the parent is a directory of the workspace and leaves the workspace root alone (a root-level
``latest_manifest.json`` would collide with the paper's).
"""

from __future__ import annotations

from typing import Any

from langgraph.runtime import Runtime

from ..context import ScientistContext
from ..core.results import AnalysisResult, PosthocResult
from ..core.state import Event, Failure, utc_now
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .common import read_json, write_json
from .layout import Layout
from .models import AnalysisRunResult, AnalysisSpec, PosthocState
from .settings import posthoc_config

PHASE = "posthoc"


def rejection_reason(result: AnalysisRunResult, checkpoint_mode: str) -> str:
    """Return why an analysis is not published (upstream ``_rejection_reason``)."""
    if result.status == "timeout":
        return "timeout"
    if result.selected_attempt is None:
        return "no_successful_attempt"
    if not result.rubric_passed:
        return "rubric_failed"
    if not result.required_outputs_present:
        return "missing_required_outputs"
    if result.guardrail_passed is False:
        return "guardrail_failed"
    if checkpoint_mode == "required" and not selected_attempt_has_checkpoint(result):
        return "checkpoint_required"
    return "invalid"


def selected_attempt_has_checkpoint(result: AnalysisRunResult) -> bool:
    """Whether the selected attempt validated a checkpoint."""
    return any(
        attempt.attempt_id == result.selected_attempt and attempt.checkpoint_validated
        for attempt in result.attempts
    )


def is_valid(result: AnalysisRunResult, checkpoint_mode: str) -> bool:
    """Whether an analysis is published as valid (upstream's conditions)."""
    return not (
        result.status != "completed"
        or not result.selected_attempt
        or not result.rubric_passed
        or not result.required_outputs_present
        or result.guardrail_passed is False
        or (checkpoint_mode == "required" and not selected_attempt_has_checkpoint(result))
    )


async def _remove_stale_manifests(workspace: Workspace, output: str) -> None:
    await workspace.remove(join(output, "latest_manifest.json"))
    parent = output.rpartition("/")[0]
    if not parent:
        return
    stable = join(parent, "latest_manifest.json")
    payload = await read_json(workspace, stable)
    if payload and payload.get("output_dir") == output:
        await workspace.remove(stable)


async def aggregate_and_publish_valid_analyses(
    workspace: Workspace,
    *,
    layout: Layout,
    specs: list[AnalysisSpec],
    results: list[AnalysisRunResult],
    checkpoint_mode: str = "required",
) -> dict[str, Any]:
    """Write ``ranking.json``, ``latest_manifest.json`` and ``pipeline_summary.json``."""
    output = layout.output
    by_id = {result.analysis_id: result for result in results}
    valid_entries: list[dict[str, Any]] = []
    rejected_entries: list[dict[str, Any]] = []

    for spec in specs:
        result = by_id.get(spec.analysis_id)
        identity = {"analysis_id": spec.analysis_id, "title": spec.title, "family": spec.family}
        if result is None:
            rejected_entries.append({**identity, "reason": "not_executed"})
            continue
        if not is_valid(result, checkpoint_mode):
            rejected_entries.append(
                {**identity, "reason": rejection_reason(result, checkpoint_mode)}
            )
            continue
        assert result.selected_attempt is not None
        attempt_dir = join(spec.analysis_dir, result.selected_attempt)
        valid_entries.append(
            {
                **identity,
                "analysis_dir": spec.analysis_dir,
                "selected_attempt": result.selected_attempt,
                "selected_attempt_dir": attempt_dir,
                "status": result.status,
                "attempt_count": result.attempt_count,
                "analysis_manifest_path": join(attempt_dir, "analysis_manifest.json"),
                "analysis_summary_path": join(attempt_dir, "analysis_summary.md"),
                "interestingness_rubric_path": join(attempt_dir, "interestingness_rubric.json"),
            }
        )

    publication = {
        "published_at": utc_now(),
        "delegated_to_paper_agent": True,
        "valid": valid_entries,
        "rejected": rejected_entries,
        "selected_analysis_id": None,
    }
    await write_json(workspace, join(output, "ranking.json"), publication)

    latest_manifest_path: str | None = None
    if valid_entries:
        latest = {
            "manifest_version": 1,
            "created_at": utc_now(),
            "output_dir": output,
            "delegated_to_paper_agent": True,
            "selected_analysis_id": None,
            "valid_analyses": valid_entries,
            "rejected_analyses": rejected_entries,
        }
        manifest_path = join(output, "latest_manifest.json")
        await write_json(workspace, manifest_path, latest)
        parent = output.rpartition("/")[0]
        if parent:
            await workspace.write_bytes(
                join(parent, "latest_manifest.json"), await workspace.read_bytes(manifest_path)
            )
        latest_manifest_path = manifest_path
    else:
        await _remove_stale_manifests(workspace, output)

    summary = {
        "status": "completed" if valid_entries else "no_valid_analyses",
        "selected_analysis_id": None,
        "delegated_to_paper_agent": True,
        "total_analyses": len(specs),
        "total_valid": len(valid_entries),
        "total_rejected": len(rejected_entries),
        "ranking_path": join(output, "ranking.json"),
        "latest_manifest_path": latest_manifest_path,
    }
    await write_json(workspace, join(output, "pipeline_summary.json"), summary)
    return publication


def analysis_results(
    publication: dict[str, Any],
    specs: list[AnalysisSpec],
    results: list[AnalysisRunResult],
    manifests: dict[str, dict[str, Any]],
) -> list[AnalysisResult]:
    """Return the core :class:`AnalysisResult` of every materialised analysis.

    A published analysis is ``passed`` and carries its selected attempt's directory and manifest;
    the others are ``timeout``, ``skipped`` (not executed) or ``failed``.
    """
    by_id = {result.analysis_id: result for result in results}
    valid = {item["analysis_id"]: item for item in publication["valid"]}
    rejected = {item["analysis_id"]: item["reason"] for item in publication["rejected"]}
    records: list[AnalysisResult] = []
    for spec in specs:
        run = by_id.get(spec.analysis_id)
        if spec.analysis_id in valid:
            entry = valid[spec.analysis_id]
            manifest = manifests.get(spec.analysis_id, {})
            outputs = manifest.get("outputs", {})
            paths: list[str] = []
            if isinstance(outputs, dict):
                for value in outputs.values():
                    if isinstance(value, list):
                        paths.extend(str(item) for item in value)
            elif isinstance(outputs, list):
                paths.extend(str(item) for item in outputs)
            records.append(
                AnalysisResult(
                    analysis_id=spec.analysis_id,
                    family=spec.family,
                    title=spec.title,
                    status="passed",
                    attempts=run.attempt_count if run else 0,
                    analysis_dir=entry["selected_attempt_dir"],
                    outputs=paths,
                    manifest=manifest,
                )
            )
            continue
        reason = rejected.get(spec.analysis_id, "not_executed")
        status = (
            "timeout"
            if reason == "timeout"
            else "skipped"
            if reason == "not_executed"
            else "failed"
        )
        records.append(
            AnalysisResult(
                analysis_id=spec.analysis_id,
                family=spec.family,
                title=spec.title,
                status=status,
                attempts=run.attempt_count if run else 0,
                analysis_dir=spec.analysis_dir,
                failure=Failure(category=f"analysis_{reason}", scope="task", detail=reason),
            )
        )
    return records


async def publish(state: PosthocState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    """Graph step: publish the valid analyses and build the phase result."""
    workspace = runtime.context.workspace
    config = posthoc_config(state)
    layout = Layout.of(state.get("output_dir"))
    bootstrap = state.get("bootstrap")
    bootstrap_dump = bootstrap.model_dump(mode="json") if bootstrap is not None else {}
    error = state.get("phase_error")
    if error is not None:
        result = PosthocResult(
            status="failed", output_dir=layout.output, bootstrap=bootstrap_dump, failure=error
        )
        return {
            "posthoc": result,
            "events": [
                Event(
                    kind="phase_end", phase=PHASE, data={"status": "failed", "detail": error.detail}
                )
            ],
        }
    specs = list(state.get("specs") or [])
    order = {spec.analysis_id: index for index, spec in enumerate(specs)}
    runs = sorted(state.get("analysis_runs") or [], key=lambda item: order.get(item.analysis_id, 0))
    publication = await aggregate_and_publish_valid_analyses(
        workspace,
        layout=layout,
        specs=specs,
        results=runs,
        checkpoint_mode=config.checkpoint_mode,
    )
    manifests: dict[str, dict[str, Any]] = {}
    for item in publication["valid"]:
        loaded = await read_json(workspace, item["analysis_manifest_path"])
        manifests[item["analysis_id"]] = loaded or {}
    analyses = analysis_results(publication, specs, runs, manifests)
    if publication["valid"]:
        result = PosthocResult(
            status="completed",
            output_dir=layout.output,
            bootstrap=bootstrap_dump,
            analyses=analyses,
        )
    else:
        ready = bootstrap is not None and bootstrap.bootstrap_ready
        category = "no_valid_analyses" if ready else "bootstrap_not_ready"
        detail = (
            "the phase published no valid analysis"
            if ready
            else f"no valid analysis and bootstrap was not ready: "
            f"{bootstrap.failure_reason if bootstrap is not None else None}"
        )
        result = PosthocResult(
            status="degraded",
            output_dir=layout.output,
            bootstrap=bootstrap_dump,
            analyses=analyses,
            failure=Failure(category=category, scope="task", detail=detail),
        )
    return {
        "publication": publication,
        "posthoc": result,
        "events": [
            Event(
                kind="phase_end",
                phase=PHASE,
                data={"status": result.status, "valid": len(publication["valid"])},
            )
        ],
    }


__all__ = [
    "aggregate_and_publish_valid_analyses",
    "analysis_results",
    "is_valid",
    "publish",
    "rejection_reason",
    "selected_attempt_has_checkpoint",
]
