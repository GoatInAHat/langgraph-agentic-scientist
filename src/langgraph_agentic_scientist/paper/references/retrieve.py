"""Retrieval fallbacks and the reference files (upstream ``paper.sh`` retrieval step).

Upstream retrieves with Semantic Scholar first; when that fails or returns fewer than the minimum,
it runs the vendored ``openalex_fetch.py`` script, then the vendored ``scholarly_fetch.py``
script, and writes ``references.bib`` and ``references_provenance.json`` from the first source
that reaches the minimum. The scripts belong to the ``retrieve-references`` skill (vendored
unchanged); here they run through the workspace, in the paper directory, from the copies the step
writes there.

The functions of this module are the retrieval step's pure parts and its fallback runner; the
references graph orders them.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ...core.interfaces import ReferenceRecord
from ...prompts import PromptRegistry
from ...workspace.paths import join
from ...workspace.protocol import Workspace
from .bibtex import bibtex_file_text, load_bibtex_rows, merge_reference_rows

OPENALEX_SEARCH_LIMIT = 10
OPENALEX_TIMEOUT_SECONDS = 240
SCHOLARLY_SEARCH_LIMIT = 10
SCHOLARLY_TOTAL_TIMEOUT_SECONDS = 300
SCHOLARLY_PROCESS_TIMEOUT_SECONDS = SCHOLARLY_TOTAL_TIMEOUT_SECONDS + 60
SCHOLARLY_PROGRESS_INTERVAL_SECONDS = 10
SCRIPTS_DIR = ".agents/skills/retrieve-references/scripts"
"""Where the fallback scripts are written in the paper directory."""


@dataclass
class FallbackOutcome:
    """The result of the script fallbacks.

    Attributes:
        records: The references upstream would write (the OpenAlex set when it reaches the
            minimum, else the merged OpenAlex and scholarly set when that does), or empty.
        openalex_error: Why the OpenAlex script failed, if it did.
        scholarly_error: Why the scholarly script failed, if it did.
        sources: The fallbacks that contributed references, in order.
    """

    records: list[ReferenceRecord] = field(default_factory=list)
    openalex_error: str | None = None
    scholarly_error: str | None = None
    sources: list[str] = field(default_factory=list)


def row_to_record(row: Mapping[str, Any], *, source: str) -> ReferenceRecord:
    """Return a record for a merged reference row (a provenance row with ``bibtex``)."""
    year = row.get("year")
    title = row.get("title")
    venue = row.get("venue")
    provenance = {key: value for key, value in row.items() if key != "bibtex"}
    return ReferenceRecord(
        key=str(row["citation_key"]),
        title=title if isinstance(title, str) else "",
        year=year if isinstance(year, int) and not isinstance(year, bool) else None,
        venue=venue if isinstance(venue, str) else None,
        bibtex=str(row["bibtex"]),
        source=source,
        provenance=provenance,
    )


def record_provenance_row(record: ReferenceRecord) -> dict[str, Any]:
    """Return the provenance row of a record (its provenance, with the citation key)."""
    row = dict(record.provenance)
    row["citation_key"] = record.key
    return row


def reference_provenance_payload(
    records: Sequence[ReferenceRecord],
    *,
    generated_at: object,
    results_dir: str,
    campaign_results_dir: str,
    queries: Sequence[str],
    plan_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the content of ``references_provenance.json`` (upstream payload)."""
    refs = [record_provenance_row(record) for record in records]
    return {
        "generated_at": generated_at,
        "results_dir": results_dir,
        "campaign_results_dir": campaign_results_dir,
        "queries": list(queries),
        "reference_query_plan": dict(plan_metadata),
        "count": len(refs),
        "references": refs,
    }


async def write_reference_outputs(
    workspace: Workspace,
    paper_dir: str,
    records: Sequence[ReferenceRecord],
    payload: Mapping[str, Any],
    *,
    bib_name: str = "references.bib",
    provenance_name: str = "references_provenance.json",
) -> None:
    """Write the ``.bib`` file and the provenance file of a reference set.

    Raises:
        ValueError: When no record carries BibTeX.
    """
    await workspace.write_bytes(
        join(paper_dir, bib_name),
        bibtex_file_text([record.bibtex or "" for record in records]).encode("utf-8"),
    )
    await workspace.write_bytes(
        join(paper_dir, provenance_name),
        json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8"),
    )


def retrieval_failure_message(
    s2_error: str | None, openalex_error: str | None, scholarly_error: str | None
) -> str:
    """Return upstream's failure message for a retrieval that produced no references."""
    if scholarly_error and openalex_error and s2_error:
        return (
            f"Reference retrieval failed. Semantic Scholar: {s2_error}; "
            f"OpenAlex: {openalex_error}; scholarly: {scholarly_error}"
        )
    if scholarly_error and s2_error:
        return (
            f"Reference retrieval failed. Semantic Scholar: {s2_error}; "
            f"scholarly: {scholarly_error}"
        )
    if scholarly_error and openalex_error:
        return (
            f"Reference retrieval failed. OpenAlex: {openalex_error}; scholarly: {scholarly_error}"
        )
    if scholarly_error:
        return f"Reference retrieval failed via scholarly fallback: {scholarly_error}"
    if openalex_error:
        return f"Reference retrieval failed via OpenAlex fallback: {openalex_error}"
    if s2_error:
        return s2_error
    return "Reference retrieval failed without producing references.bib"


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join((text or "").splitlines()[-lines:]).strip()


async def _read_rows(
    workspace: Workspace, paper_dir: str, bib_name: str, json_name: str
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    bib_path = join(paper_dir, bib_name)
    json_path = join(paper_dir, json_name)
    bib_rows: list[dict[str, str]] = []
    prov_rows: list[dict[str, Any]] = []
    bib_entry = await workspace.stat(bib_path)
    if bib_entry is not None and bib_entry.kind == "file":
        bib_rows = load_bibtex_rows((await workspace.read_bytes(bib_path)).decode("utf-8"))
    json_entry = await workspace.stat(json_path)
    if json_entry is not None and json_entry.kind == "file":
        payload = json.loads((await workspace.read_bytes(json_path)).decode("utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("references"), list):
            prov_rows = [row for row in payload["references"] if isinstance(row, dict)]
    return bib_rows, prov_rows


async def _run_script(
    workspace: Workspace,
    paper_dir: str,
    script_name: str,
    arguments: Sequence[str],
    *,
    python: str,
    timeout: float,
    label: str,
    timeout_seconds: int,
) -> str | None:
    """Run one fallback script; return its error text, or ``None`` when it succeeded."""
    script_path = join(paper_dir, SCRIPTS_DIR, script_name)
    try:
        run = await workspace.run(
            [python, workspace.host_path(script_path), *arguments],
            cwd=paper_dir,
            timeout=timeout,
        )
    except OSError as exc:
        return f"{label} fallback could not run: {exc}"
    if run.timed_out:
        return f"{label} fallback timed out after {timeout_seconds}s"
    if run.returncode != 0:
        return (
            _tail(run.stderr)
            or _tail(run.stdout)
            or f"{label} fallback failed with code {run.returncode}"
        )
    return None


async def run_script_fallbacks(
    workspace: Workspace,
    paper_dir: str,
    *,
    queries: Sequence[str],
    min_refs: int,
    max_refs: int,
    s2_error: str | None,
    s2_count: int,
    python: str | None = None,
    prompts: PromptRegistry | None = None,
) -> FallbackOutcome:
    """Run the OpenAlex and scholarly fallbacks as upstream ``paper.sh`` does.

    OpenAlex runs when Semantic Scholar failed or returned fewer than ``min_refs`` references; its
    references are used when they reach ``min_refs``. The scholarly script runs when that left no
    references, asks for the references still missing, and its references merged after OpenAlex's
    are used when they reach ``min_refs``. Both scripts run in ``paper_dir`` and write
    ``references.openalex.*`` and ``references.scholarly.*`` files there.
    """
    interpreter = python or sys.executable
    registry = prompts or PromptRegistry.default()
    scripts = registry.skill_files("retrieve-references")
    for name in ("openalex_fetch.py", "scholarly_fetch.py"):
        await workspace.write_bytes(join(paper_dir, SCRIPTS_DIR, name), scripts[f"scripts/{name}"])
    outcome = FallbackOutcome()

    openalex_bib, openalex_json = "references.openalex.bib", "references.openalex.json"
    scholarly_bib, scholarly_json = "references.scholarly.bib", "references.scholarly.json"

    if s2_error is not None or s2_count < min_refs:
        arguments = [
            "--min-refs",
            str(min_refs),
            "--max-refs",
            str(max_refs),
            "--search-limit",
            str(OPENALEX_SEARCH_LIMIT),
            "--allow-partial-output",
            "--out-bib",
            openalex_bib,
            "--out-json",
            openalex_json,
        ]
        for query in queries:
            arguments.extend(["--query", query])
        outcome.openalex_error = await _run_script(
            workspace,
            paper_dir,
            "openalex_fetch.py",
            arguments,
            python=interpreter,
            timeout=OPENALEX_TIMEOUT_SECONDS,
            label="OpenAlex",
            timeout_seconds=OPENALEX_TIMEOUT_SECONDS,
        )
        bib_rows, prov_rows = await _read_rows(workspace, paper_dir, openalex_bib, openalex_json)
        if await workspace.exists(join(paper_dir, openalex_bib)) and await workspace.exists(
            join(paper_dir, openalex_json)
        ):
            refs, _ = merge_reference_rows((bib_rows, prov_rows))
            if refs:
                outcome.sources.append("openalex")
                if len(refs) >= min_refs:
                    outcome.records = [row_to_record(row, source="openalex") for row in refs]
                    return outcome

    openalex_existing_bib, openalex_existing_prov = await _read_rows(
        workspace, paper_dir, openalex_bib, openalex_json
    )
    scholarly_min = max(1, min_refs - len(openalex_existing_bib))
    scholarly_max = max(1, max_refs - len(openalex_existing_bib))
    arguments = [
        "--min-refs",
        str(scholarly_min),
        "--max-refs",
        str(scholarly_max),
        "--search-limit",
        str(SCHOLARLY_SEARCH_LIMIT),
        "--total-timeout-seconds",
        str(SCHOLARLY_TOTAL_TIMEOUT_SECONDS),
        "--progress-interval-seconds",
        str(SCHOLARLY_PROGRESS_INTERVAL_SECONDS),
        "--allow-partial-output",
        "--out-bib",
        scholarly_bib,
        "--out-json",
        scholarly_json,
    ]
    for query in queries:
        arguments.extend(["--query", query])
    outcome.scholarly_error = await _run_script(
        workspace,
        paper_dir,
        "scholarly_fetch.py",
        arguments,
        python=interpreter,
        timeout=SCHOLARLY_PROCESS_TIMEOUT_SECONDS,
        label="scholarly",
        timeout_seconds=SCHOLARLY_PROCESS_TIMEOUT_SECONDS,
    )
    scholarly_rows = await _read_rows(workspace, paper_dir, scholarly_bib, scholarly_json)
    merged, _ = merge_reference_rows(
        (openalex_existing_bib, openalex_existing_prov), scholarly_rows
    )
    if merged:
        outcome.sources.append("scholarly")
        if len(merged) >= min_refs:
            openalex_keys = {row["citation_key"] for row in openalex_existing_bib}
            outcome.records = [
                row_to_record(
                    row, source="openalex" if row["citation_key"] in openalex_keys else "scholarly"
                )
                for row in merged
            ]
    return outcome


__all__ = [
    "SCRIPTS_DIR",
    "FallbackOutcome",
    "record_provenance_row",
    "reference_provenance_payload",
    "retrieval_failure_message",
    "row_to_record",
    "run_script_fallbacks",
    "write_reference_outputs",
]
