"""Stage 1: prepare analysis bundles, preserving brokered candidates without agent edits
(upstream ``retrofit.py`` and ``retrofit_preservation.py``).

A *legacy executable* program is retrofitted by the retrofit agent: it adds a ``--results_dir``
argument and writes three analysis CSVs, without touching the program's computation. A
*contracted* ``python_rpc_v1:handle`` candidate is copied unchanged by the host; its evaluator
owns analysis output.

The evaluator is sealed around the agent: the host digests the source bundle's evaluator tree and
keeps the bytes of ``evaluate.py``, lets the agent work on a copy, restores ``evaluate.py`` and
removes the completion marker afterwards, digest-checks the source tree, and writes
``.retrofit_complete.json`` only on success. :func:`retrofit_bundle_complete` verifies the marker
(schema 1 for an instrumented program, schema 2 with the preserved-tree digests for a brokered
copy) against the bundle.

The phase runs this as three steps (``retrofit_prepare``, the agent session, ``retrofit_finish``)
so that the session is the only thing its node does; the functions here are those steps.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from langgraph.runtime import Runtime

from ..agents.runner import run_workspace_role
from ..context import ScientistContext
from ..core.contract import load_task_contract, task_scaffold_fingerprint
from ..core.errors import ProgramSpecError, TaskContractError
from ..evaluation.program_spec import load_program_spec, resolve_entrypoint
from ..models import Role
from ..prompts import PromptRegistry, render_text
from ..workspace.paths import join
from ..workspace.protocol import Workspace, is_file
from .constants import (
    ANALYSIS_CSV_SCHEMAS,
    EXTRACTION_MANIFEST_FILENAME,
    MAX_EVALUATOR_BYTES,
    RETROFIT_COMPLETE_FILENAME,
)
from .digests import evaluator_tree_digest, strict_tree_digest
from .models import RetrofitReport
from .workspace_ops import read_text_or_none, stage_active_files

logger = logging.getLogger(__name__)

TASK_RUNTIME_ROOT_ENTRIES = frozenset({".venv"})
BUNDLE_DIGEST_OMISSIONS = TASK_RUNTIME_ROOT_ENTRIES | {RETROFIT_COMPLETE_FILENAME}
CONTRACT_FILENAME = "task_contract.json"
BROKERED_INTERFACE = "python_rpc_v1:handle"
PROMPT_KEY = "variants/variant_retrofit"


async def _read_bounded_file(workspace: Workspace, path: str, maximum: int) -> bytes:
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        raise ValueError(f"{path} must be a regular file")
    if entry.size > maximum:
        raise ValueError(f"{path} exceeds {maximum} bytes")
    return await workspace.read_bytes(path)


async def brokered_preservation_metadata(
    workspace: Workspace, bundle_dir: str, task_source_dir: str | None
) -> dict[str, str] | None:
    """Require a valid contract before treating a program as host-brokered.

    Looks for ``task_contract.json`` in the task source directory, then in the bundle. Returns
    ``None`` when there is none (a legacy executable).

    Raises:
        ValueError: When the contract is invalid or names an unsupported candidate interface.
    """
    sources = [bundle_dir]
    if task_source_dir is not None:
        sources.insert(0, task_source_dir)
    for source in sources:
        if await workspace.stat(join(source, CONTRACT_FILENAME)) is None:
            continue
        try:
            contract = await load_task_contract(workspace, source)
        except TaskContractError as exc:
            raise ValueError(str(exc)) from exc
        if contract.candidate.interface != BROKERED_INTERFACE:
            raise ValueError("unsupported candidate interface for brokered retrofit")
        return {
            "mode": "brokered_preserve",
            "source_program_tree_sha256": await strict_tree_digest(
                workspace, join(bundle_dir, "program_0")
            ),
            "source_bundle_tree_sha256": await strict_tree_digest(
                workspace, bundle_dir, omitted_root_entries=BUNDLE_DIGEST_OMISSIONS
            ),
            "task_scaffold_sha256": await task_scaffold_fingerprint(workspace, source),
        }
    return None


async def preserved_program_digest(
    workspace: Workspace, bundle_dir: str, metadata: Mapping[str, str]
) -> str:
    """Check a brokered copy against its source and return the preserved program's digest.

    Raises:
        ValueError: When the copy holds an unstaged task runtime or differs from the source.
    """
    for name in TASK_RUNTIME_ROOT_ENTRIES:
        if await workspace.stat(join(bundle_dir, name)) is not None:
            raise ValueError("brokered copy contains an unstaged task runtime")
    digest = await strict_tree_digest(workspace, join(bundle_dir, "program_0"))
    if digest != metadata["source_program_tree_sha256"]:
        raise ValueError("brokered retrofit changed the selected program")
    bundle_digest = await strict_tree_digest(
        workspace, bundle_dir, omitted_root_entries=BUNDLE_DIGEST_OMISSIONS
    )
    if bundle_digest != metadata["source_bundle_tree_sha256"]:
        raise ValueError("brokered retrofit changed the sealed bundle")
    return digest


async def retrofit_bundle_complete(
    workspace: Workspace,
    bundle_dir: str,
    source_dir: str,
    *,
    task_source_dir: str | None = None,
) -> bool:
    """Verify the evaluator seal and any required brokered-program preservation.

    ``bundle_dir`` is the retrofitted bundle, ``source_dir`` the bundle it was made from.
    """
    try:
        marker = json.loads(
            await _read_bounded_file(workspace, join(bundle_dir, RETROFIT_COMPLETE_FILENAME), 4096)
        )
        evaluator = await _read_bounded_file(
            workspace, join(bundle_dir, "evaluate.py"), MAX_EVALUATOR_BYTES
        )
        source_digest = await evaluator_tree_digest(workspace, source_dir)
        preservation = await brokered_preservation_metadata(workspace, source_dir, task_source_dir)
        if preservation is not None:
            copied_digest = await preserved_program_digest(workspace, bundle_dir, preservation)
            if not isinstance(marker, dict) or not (
                marker.get("schema_version") == 2
                and all(marker.get(key) == value for key, value in preservation.items())
                and marker.get("copied_program_tree_sha256") == copied_digest
            ):
                return False
        elif not isinstance(marker, dict) or marker.get("schema_version") != 1:
            return False
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError, FileNotFoundError):
        return False
    return bool(
        isinstance(marker, dict)
        and marker.get("evaluator_sha256") == hashlib.sha256(evaluator).hexdigest()
        and marker.get("source_tree_sha256") == source_digest
    )


async def seal_and_copy(
    workspace: Workspace,
    *,
    bundle_dir: str,
    output_bundle_dir: str,
    task_source_dir: str | None,
    agent_model: str | None = None,
) -> RetrofitReport:
    """Seal the evaluator, copy the bundle and, for a brokered candidate, finish the retrofit.

    Returns a report. For a legacy executable the report has ``exit_code`` 1 and no ``error``:
    the retrofit agent still has to run and :func:`finish_retrofit` completes the report. For a
    brokered candidate the copy is verified, the marker (schema 2) is written and the report is
    complete (``exit_code`` 0, ``mode="brokered_preserve"``). A failure has ``error`` set.
    """
    report = RetrofitReport(
        bundle_dir=bundle_dir, output_bundle_dir=output_bundle_dir, agent_model=agent_model
    )
    try:
        preservation = await brokered_preservation_metadata(workspace, bundle_dir, task_source_dir)
        sealed_source_digest = await evaluator_tree_digest(workspace, bundle_dir)
        sealed_evaluator = await _read_bounded_file(
            workspace, join(bundle_dir, "evaluate.py"), MAX_EVALUATOR_BYTES
        )
    except (OSError, ValueError, UnicodeError) as error:
        return report.model_copy(update={"error": f"Cannot seal evaluator: {error}"})
    report = report.model_copy(
        update={
            "preservation": preservation,
            "sealed_source_tree_sha256": sealed_source_digest,
            "sealed_evaluator_sha256": hashlib.sha256(sealed_evaluator).hexdigest(),
        }
    )

    await workspace.remove(output_bundle_dir)
    # The task runtime (.venv) is the caller's: it is never copied, brokered or not.
    await workspace.copy_tree(bundle_dir, output_bundle_dir, ignore=["/.venv/"])
    await workspace.remove(join(output_bundle_dir, RETROFIT_COMPLETE_FILENAME))
    logger.info("Copied bundle to %s", output_bundle_dir)

    if preservation is None:
        return report
    try:
        copied_digest = await preserved_program_digest(workspace, output_bundle_dir, preservation)
        if await evaluator_tree_digest(workspace, bundle_dir) != sealed_source_digest:
            raise ValueError("Sealed evaluator source changed during retrofit")
        if (
            await brokered_preservation_metadata(workspace, bundle_dir, task_source_dir)
            != preservation
        ):
            raise ValueError("Brokered task or selected program changed during retrofit")
        marker = {
            "schema_version": 2,
            "evaluator_sha256": report.sealed_evaluator_sha256,
            "source_tree_sha256": sealed_source_digest,
            "copied_program_tree_sha256": copied_digest,
            **preservation,
        }
        await workspace.write_bytes(
            join(output_bundle_dir, RETROFIT_COMPLETE_FILENAME),
            (json.dumps(marker, sort_keys=True) + "\n").encode(),
        )
    except (OSError, ValueError, UnicodeError) as error:
        return report.model_copy(update={"error": f"Cannot preserve brokered program: {error}"})
    logger.info("Preserved brokered candidate; host evaluator owns analysis outputs")
    return report.model_copy(
        update={"exit_code": 0, "mode": "brokered_preserve", "program_tree_sha256": copied_digest}
    )


def format_csv_schemas() -> str:
    """Format the CSV schema definitions for the prompt (upstream ``_format_csv_schemas``)."""
    lines: list[str] = []
    for filename, columns in ANALYSIS_CSV_SCHEMAS.items():
        lines.append(f"### {filename}")
        lines.append(f"Required columns: {', '.join(columns)}")
        lines.append("")
    return "\n".join(lines)


async def build_retrofit_prompt(
    workspace: Workspace,
    prompts: PromptRegistry,
    bundle_dir: str,
    *,
    reference_evaluator_source: str = "",
) -> str:
    """Build the retrofit agent's prompt from the copied bundle (``_build_retrofit_prompt``)."""
    program_dir = join(bundle_dir, "program_0")
    program_source = ""
    if await is_file(workspace, join(program_dir, "program_spec.toml")):
        try:
            await load_program_spec(workspace, program_dir)
            entrypoint = await resolve_entrypoint(workspace, program_dir)
            program_source = (await workspace.read_bytes(join(program_dir, entrypoint))).decode(
                "utf-8"
            )
        except (ProgramSpecError, OSError, UnicodeError):
            logger.warning("Could not load program spec, using fallback")
    values = {
        "PROGRAM_SOURCE": program_source,
        "EVALUATOR_SOURCE": await read_text_or_none(workspace, join(bundle_dir, "evaluate.py"))
        or "",
        "EXTRACTION_MANIFEST": await read_text_or_none(
            workspace, join(bundle_dir, EXTRACTION_MANIFEST_FILENAME)
        )
        or "",
        "CSV_SCHEMAS": format_csv_schemas(),
        "TASK_DESCRIPTION": await read_text_or_none(workspace, join(bundle_dir, "task.md")) or "",
        "REFERENCE_EVALUATOR_SOURCE": reference_evaluator_source,
    }
    return render_text(prompts.text(PROMPT_KEY), values)


async def finish_retrofit(
    workspace: Workspace,
    report: RetrofitReport,
    *,
    session_ok: bool,
    session_error: str | None = None,
) -> RetrofitReport:
    """Restore the sealed evaluator, check the source and write the completion marker.

    Runs after the retrofit agent's session whatever its outcome (upstream does this in a
    ``finally``): ``evaluate.py`` of the retrofitted copy is rewritten from the sealed source
    bundle and the completion marker removed. When the session failed or timed out, the report
    is a failure; otherwise the marker (schema 1) is written.
    """
    if report.mode == "brokered_preserve" or report.error is not None:
        return report
    source_evaluator = join(report.bundle_dir, "evaluate.py")
    target_evaluator = join(report.output_bundle_dir, "evaluate.py")
    try:
        sealed = await _read_bounded_file(workspace, source_evaluator, MAX_EVALUATOR_BYTES)
        if hashlib.sha256(sealed).hexdigest() != report.sealed_evaluator_sha256:
            raise ValueError("Sealed evaluator changed during retrofit")
        await workspace.remove(join(report.output_bundle_dir, RETROFIT_COMPLETE_FILENAME))
        await workspace.write_bytes(target_evaluator, sealed)
    except (OSError, ValueError) as error:
        return report.model_copy(update={"error": f"Failed to restore sealed evaluator: {error}"})
    try:
        unchanged = (
            await evaluator_tree_digest(workspace, report.bundle_dir)
            == report.sealed_source_tree_sha256
        )
    except (OSError, ValueError) as error:
        return report.model_copy(
            update={"error": f"Cannot verify sealed evaluator source: {error}"}
        )
    if not unchanged:
        return report.model_copy(
            update={"error": "Sealed evaluator source changed during retrofit"}
        )
    if not session_ok:
        logger.warning("Retrofit agent did not finish: %s", session_error)
        return report.model_copy(
            update={"exit_code": 1, "error": session_error or "Agent exited with code 1"}
        )
    marker = {
        "schema_version": 1,
        "evaluator_sha256": report.sealed_evaluator_sha256,
        "source_tree_sha256": report.sealed_source_tree_sha256,
    }
    try:
        await workspace.write_bytes(
            join(report.output_bundle_dir, RETROFIT_COMPLETE_FILENAME),
            (json.dumps(marker, sort_keys=True) + "\n").encode(),
        )
    except (OSError, ValueError) as error:
        return report.model_copy(
            update={"exit_code": 1, "error": f"Failed to publish retrofit completion: {error}"}
        )
    logger.info("Retrofit agent completed successfully")
    return report.model_copy(update={"exit_code": 0})


async def run_retrofit_session(
    runtime: Runtime[ScientistContext],
    report: RetrofitReport,
    *,
    reference_evaluator_source: str = "",
    timeout: float | None,
) -> dict[str, Any]:
    """Run one session of the retrofit agent (R16) on the copied bundle.

    No skills are staged: upstream's retrofit passes no ``--stage-skills``. Returns the node
    update ``{"retrofit_session_ok": bool, "retrofit_session_error": str | None}``.
    """
    ctx = runtime.context
    ws = ctx.workspace
    root = report.output_bundle_dir
    await stage_active_files(ws, root)
    prompt = await build_retrofit_prompt(
        ws, ctx.prompts, root, reference_evaluator_source=reference_evaluator_source
    )
    outcome = await run_workspace_role(
        role=Role.RETROFIT_AGENT,
        runtime=runtime,
        root=root,
        prompt=prompt,
        timeout=timeout,
    )
    error: str | None = None
    if outcome.timed_out:
        error = "Agent exited with code 124"
    elif outcome.failure is not None:
        error = f"Agent exited with code 1 ({outcome.failure.category}): {outcome.failure.detail}"
    return {"retrofit_session_ok": outcome.ok, "retrofit_session_error": error}


__all__ = [
    "brokered_preservation_metadata",
    "build_retrofit_prompt",
    "finish_retrofit",
    "format_csv_schemas",
    "preserved_program_digest",
    "retrofit_bundle_complete",
    "run_retrofit_session",
    "seal_and_copy",
]
