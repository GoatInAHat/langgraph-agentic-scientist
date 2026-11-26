"""Stage 0: extract the best-program bundle (upstream ``extract.py`` and
``best_program_extraction.py``).

The bundle, ``_extracted_bundle/``, is the phase's working directory: the task-level files, the
best program as ``program_0``, the reference program (the seed, or the program a caller names) as
``reference_program``, and ``extraction_manifest.json`` with the provenance and fingerprints that
bind it to the task and the program it was made from.

Upstream resolves the best program from a results directory (a frozen downstream selection, the
best-program manifest, or a scan). The port is given it typed: an explicit program, the
downstream selection, or the investigation's best manifest, in that order. The bundle is built in
place and the manifest is written last, so a bundle without a manifest is a crashed extraction and
is rebuilt, while a bundle whose manifest disagrees with the inputs is refused.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ..core.errors import ProgramSpecError
from ..core.results import DownstreamSelection, InvestigationResult
from ..core.state import ProgramRef, TaskRef
from ..evaluation.program_clone import clone_program
from ..evaluation.program_spec import resolve_entrypoint
from ..workspace.paths import join
from ..workspace.protocol import Workspace, is_dir
from .constants import (
    EXTRACTED_BUNDLE_DIR,
    EXTRACTION_MANIFEST_FILENAME,
    EXTRACTION_VERSION,
)
from .digests import managed_evaluator_tree_digest, tree_fingerprint
from .materialize import task_ignore_patterns
from .models import BundleRef
from .workspace_ops import (
    copy_tree_replacing,
    extract_copy_patterns,
    make_dir,
    read_json_object,
    write_json,
)

logger = logging.getLogger(__name__)

REQUIRED_BUNDLE_ENTRIES = ("program_0", "evaluate.py", EXTRACTION_MANIFEST_FILENAME)


class BundleValidationError(RuntimeError):
    """The extracted bundle is incomplete, or does not belong to this run's inputs."""


class ExtractionError(RuntimeError):
    """Extraction validation or materialisation failed (upstream ``ExtractionError``)."""


def resolve_best_program(
    *,
    best: ProgramRef | None,
    downstream: DownstreamSelection | None,
    investigation: InvestigationResult | None,
) -> tuple[ProgramRef, dict[str, Any]]:
    """Resolve the program the variants are built around and its selection metadata.

    Order: an explicit ``best`` program; the frozen downstream selection (the no-winner
    fallback); the investigation's best-program manifest.

    Raises:
        ExtractionError: When none of them names a program.
    """
    if best is not None:
        return best, {
            "entrypoint_path": best.entrypoint,
            "combined_score": None,
            "correct": None,
            "source_attempt_id": None,
            "parallel_worker_index": None,
        }
    if downstream is not None and downstream.status == "selected" and downstream.program:
        program = downstream.program
        return program, {
            "entrypoint_path": program.entrypoint,
            "combined_score": downstream.combined_score,
            "correct": True,
            "source_attempt_id": downstream.evaluation_id,
            "parallel_worker_index": None,
        }
    if investigation is not None and investigation.best is not None:
        manifest = investigation.best
        extra = manifest.model_extra or {}
        return ProgramRef(path=manifest.program_dir, entrypoint=manifest.entrypoint_path), {
            "entrypoint_path": manifest.entrypoint_path,
            "combined_score": manifest.combined_score,
            "correct": manifest.correct,
            "source_attempt_id": manifest.source_attempt_id,
            "parallel_worker_index": extra.get("parallel_worker_index"),
        }
    raise ExtractionError(
        "No best program: give an explicit program, a downstream selection or an investigation "
        "result with a best-program manifest."
    )


def _json_fingerprint(payload: Mapping[str, Any], *, excluded: set[str] | None = None) -> str:
    skip = excluded or set()
    normalized = {key: value for key, value in payload.items() if key not in skip}
    encoded = json.dumps(
        normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def capture_source_provenance(
    workspace: Workspace,
    *,
    task: TaskRef,
    best: ProgramRef,
    meta: Mapping[str, Any],
    downstream: DownstreamSelection | None,
) -> dict[str, str | None]:
    """Fingerprint the exact task and winning-program inputs of an extraction."""
    selection_payload = {
        "program_dir": best.path,
        "entrypoint_path": meta.get("entrypoint_path"),
        "combined_score": meta.get("combined_score"),
        "correct": meta.get("correct"),
        "source_attempt_id": meta.get("source_attempt_id"),
        "parallel_worker_index": meta.get("parallel_worker_index"),
    }
    return {
        "source_downstream_selection_fingerprint": (
            _json_fingerprint(downstream.model_dump(mode="json"))
            if downstream is not None and downstream.status == "selected"
            else None
        ),
        "source_best_program_dir": best.path,
        "source_best_program_fingerprint": await tree_fingerprint(workspace, best.path),
        "source_selection_fingerprint": _json_fingerprint(selection_payload),
        "source_task_fingerprint": await managed_evaluator_tree_digest(workspace, task.root),
    }


async def validate_bundle(workspace: Workspace, bundle: str) -> None:
    """Raise :class:`BundleValidationError` if the bundle lacks a required entry."""
    missing = [
        name for name in REQUIRED_BUNDLE_ENTRIES if await workspace.stat(join(bundle, name)) is None
    ]
    if missing:
        raise BundleValidationError(f"Extracted bundle at {bundle} is missing: {missing}")
    if not await is_dir(workspace, join(bundle, "reference_program")):
        logger.warning(
            "Extracted bundle has no reference_program/ "
            "— baseline_reference variant will be skipped."
        )


async def _bundle_fingerprint(workspace: Workspace, bundle: str) -> str:
    return await tree_fingerprint(workspace, bundle, excluded={EXTRACTION_MANIFEST_FILENAME})


async def validate_reusable_bundle(
    workspace: Workspace,
    bundle: str,
    *,
    expected: Mapping[str, Any],
) -> None:
    """Validate that an existing bundle belongs to this run's inputs and is intact.

    ``expected`` holds the provenance (and baseline override) the manifest must carry.

    Raises:
        BundleValidationError: With upstream's wording where it has one.
    """
    await validate_bundle(workspace, bundle)
    manifest = await read_json_object(workspace, join(bundle, EXTRACTION_MANIFEST_FILENAME))
    if manifest is None:
        raise BundleValidationError(f"Existing extraction manifest is invalid: {bundle}")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise BundleValidationError(
                f"Existing extraction manifest {key} does not match: expected {value!r}, "
                f"got {manifest.get(key)!r}"
            )
    bundle_fingerprint = manifest.get("bundle_fingerprint")
    if not isinstance(bundle_fingerprint, str):
        raise BundleValidationError("Existing extraction manifest has no bundle fingerprint")
    if await _bundle_fingerprint(workspace, bundle) != bundle_fingerprint:
        raise BundleValidationError(
            "Existing extracted bundle fingerprint does not match bundled contents"
        )
    payload_fingerprint = manifest.get("manifest_payload_fingerprint")
    if not isinstance(payload_fingerprint, str):
        raise BundleValidationError("Existing extraction manifest has no payload fingerprint")
    if (
        _json_fingerprint(manifest, excluded={"manifest_payload_fingerprint"})
        != payload_fingerprint
    ):
        raise BundleValidationError(
            "Existing extraction manifest payload fingerprint does not match"
        )


async def extract_bundle(
    workspace: Workspace,
    *,
    task: TaskRef,
    output_dir: str,
    best: ProgramRef,
    meta: Mapping[str, Any],
    downstream: DownstreamSelection | None = None,
    baseline_program_override: str | None = None,
) -> BundleRef:
    """Extract the best-program bundle and validate it for the variants phase.

    Args:
        workspace: The run's workspace.
        task: The run-local task snapshot (``task.md``, evaluators, ``program_0``).
        output_dir: The phase's output directory; the bundle is ``<output_dir>/_extracted_bundle``.
        best: The best program.
        meta: Its selection metadata (see :func:`resolve_best_program`).
        downstream: The frozen downstream selection, when the best program is its fallback.
        baseline_program_override: A program directory that replaces ``reference_program/``; only
            program files are taken from it, task-level files still come from the task.

    Returns:
        The reference to the bundle.

    Raises:
        ExtractionError: When the task or the best program is unusable.
        BundleValidationError: When an existing bundle does not match the inputs.
    """
    bundle = join(output_dir, EXTRACTED_BUNDLE_DIR)
    task_root = task.root
    if not await is_dir(workspace, task_root):
        raise ExtractionError(f"Task directory does not exist: {task_root}")
    seed = join(task_root, "program_0")
    if not await is_dir(workspace, seed):
        raise ExtractionError(f"Task directory missing program_0: {task_root}")
    if not await is_dir(workspace, best.path):
        raise ExtractionError(f"Best program directory does not exist: {best.path}")
    try:
        entrypoint = await resolve_entrypoint(workspace, best.path)
    except ProgramSpecError as exc:
        raise ExtractionError(
            f"Cannot resolve entrypoint in best program {best.path}: {exc}"
        ) from exc
    if baseline_program_override is not None and not await is_dir(
        workspace, baseline_program_override
    ):
        raise BundleValidationError(
            f"Baseline program override is not a directory: {baseline_program_override}"
        )

    provenance = await capture_source_provenance(
        workspace, task=task, best=best, meta=meta, downstream=downstream
    )
    override_fingerprint = (
        await tree_fingerprint(workspace, baseline_program_override)
        if baseline_program_override is not None
        else None
    )
    expected = {
        **provenance,
        "source_task_dir": task_root,
        "baseline_program_override": baseline_program_override,
        "baseline_program_override_fingerprint": override_fingerprint,
    }

    manifest_path = join(bundle, EXTRACTION_MANIFEST_FILENAME)
    if await workspace.stat(bundle) is not None:
        if await workspace.stat(manifest_path) is not None:
            await validate_reusable_bundle(workspace, bundle, expected=expected)
            manifest = await read_json_object(workspace, manifest_path) or {}
            logger.info("Reusing existing extracted bundle: %s", bundle)
            return BundleRef(
                output_dir=output_dir,
                extracted=bundle,
                has_reference=await is_dir(workspace, join(bundle, "reference_program")),
                entrypoint=entrypoint,
                manifest=manifest,
            )
        logger.info(
            "Discarding a bundle without a manifest (an interrupted extraction): %s", bundle
        )
        await workspace.remove(bundle)

    logger.info(
        "Extracting best-program bundle: task=%s best=%s -> %s", task_root, best.path, bundle
    )
    await make_dir(workspace, output_dir)
    # Task-level files, the reference program and the best program.
    patterns = extract_copy_patterns(await task_ignore_patterns(workspace, task_root))
    await copy_tree_replacing(workspace, task_root, bundle, ignore=patterns)
    reference_source = baseline_program_override if baseline_program_override is not None else seed
    await clone_program(
        workspace, src_dir=reference_source, dst_dir=join(bundle, "reference_program")
    )
    await clone_program(workspace, src_dir=best.path, dst_dir=join(bundle, "program_0"))

    current = await capture_source_provenance(
        workspace, task=task, best=best, meta=meta, downstream=downstream
    )
    if current != provenance:
        raise BundleValidationError("Best-program or task inputs changed during bundle extraction")

    built: dict[str, Any] = {
        "extraction_version": EXTRACTION_VERSION,
        "extracted_at": datetime.now(UTC).isoformat(),
        "source_task_dir": task_root,
        "source_best_program_dir": best.path,
        "reference_program_dir": join(bundle, "reference_program"),
        "extracted_program_dir": join(bundle, "program_0"),
        "entrypoint_path": meta.get("entrypoint_path") or entrypoint,
        "combined_score": meta.get("combined_score"),
        "correct": meta.get("correct"),
        "source_attempt_id": meta.get("source_attempt_id"),
        "parallel_worker_index": meta.get("parallel_worker_index"),
        "downstream_selection": (
            downstream.model_dump(mode="json")
            if downstream is not None and downstream.status == "selected"
            else None
        ),
        **provenance,
        "baseline_program_override": baseline_program_override,
        "baseline_program_override_fingerprint": override_fingerprint,
    }
    built["bundle_fingerprint"] = await _bundle_fingerprint(workspace, bundle)
    built["manifest_payload_fingerprint"] = _json_fingerprint(
        built, excluded={"manifest_payload_fingerprint"}
    )
    await write_json(workspace, manifest_path, built, sort_keys=True, trailing_newline=True)
    await validate_bundle(workspace, bundle)
    logger.info("Bundle extraction complete: %s", bundle)
    return BundleRef(
        output_dir=output_dir,
        extracted=bundle,
        has_reference=await is_dir(workspace, join(bundle, "reference_program")),
        entrypoint=entrypoint,
        manifest=built,
    )


__all__ = [
    "BundleValidationError",
    "ExtractionError",
    "capture_source_provenance",
    "extract_bundle",
    "resolve_best_program",
    "validate_bundle",
    "validate_reusable_bundle",
]
