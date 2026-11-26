"""Merging reference artifacts (upstream ``merge_reference_artifacts``).

The post-draft enrichment retrieves additional references for the gaps of a draft and merges
them into ``references.bib`` and ``references_provenance.json``: the existing entries come first,
an incremental entry with a citation key already present is dropped, the queries are merged
without duplicates, and every query plan that contributed is kept.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ...core.state import utc_now
from ...workspace.paths import join
from ...workspace.protocol import Workspace
from .bibtex import load_bibtex_rows, merge_reference_rows


def _merge_query_lists(*query_lists: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for queries in query_lists:
        for query in queries:
            clean = " ".join(str(query).split()).strip()
            if not clean:
                continue
            key = clean.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(clean)
    return merged


def _merge_reference_query_plans(*payloads: Mapping[str, Any]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for payload in payloads:
        metadata = payload.get("reference_query_plan")
        if isinstance(metadata, dict) and metadata:
            plans.append(metadata)
        legacy_list = payload.get("reference_query_plans")
        if isinstance(legacy_list, list):
            plans.extend(row for row in legacy_list if isinstance(row, dict) and row)
    return plans


def _provenance_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("references")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def merge_reference_payloads(
    *,
    existing_bib: str,
    existing_provenance: Mapping[str, Any],
    incremental_bib: str,
    incremental_provenance: Mapping[str, Any],
    results_dir: str,
    campaign_results_dir: str,
) -> tuple[str, dict[str, Any]]:
    """Merge existing and incremental references; return the ``.bib`` text and the provenance.

    Raises:
        ValueError: When no BibTeX entry is available.
    """
    merged_refs, merged_prov = merge_reference_rows(
        (load_bibtex_rows(existing_bib), _provenance_rows(existing_provenance)),
        (load_bibtex_rows(incremental_bib), _provenance_rows(incremental_provenance)),
    )
    if not merged_refs:
        raise ValueError("No BibTeX entries available to merge.")
    bib_text = "\n\n".join(str(row["bibtex"]).strip() for row in merged_refs) + "\n"
    payload = {
        "generated_at": incremental_provenance.get("generated_at")
        or existing_provenance.get("generated_at")
        or utc_now(),
        "results_dir": results_dir,
        "campaign_results_dir": campaign_results_dir,
        "queries": _merge_query_lists(
            list(existing_provenance.get("queries", [])),
            list(incremental_provenance.get("queries", [])),
        ),
        "reference_query_plan": incremental_provenance.get("reference_query_plan")
        or existing_provenance.get("reference_query_plan", {}),
        "reference_query_plans": _merge_reference_query_plans(
            existing_provenance, incremental_provenance
        ),
        "count": len(merged_prov),
        "references": merged_prov,
    }
    return bib_text, payload


async def _read_json_object(workspace: Workspace, path: str) -> dict[str, Any]:
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        return {}
    payload = json.loads((await workspace.read_bytes(path)).decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


async def _read_text_or_empty(workspace: Workspace, path: str) -> str:
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        return ""
    return (await workspace.read_bytes(path)).decode("utf-8")


async def merge_reference_artifacts(
    workspace: Workspace,
    paper_dir: str,
    *,
    existing_bib: str = "references.pre_postdraft.bib",
    existing_provenance: str = "references.pre_postdraft.json",
    incremental_bib: str = "postdraft_references.bib",
    incremental_provenance: str = "postdraft_references_provenance.json",
    output_bib: str = "references.bib",
    output_provenance: str = "references_provenance.json",
    results_dir: str = "",
    campaign_results_dir: str = "",
) -> dict[str, Any]:
    """Merge existing and incremental reference files of a paper directory (upstream function).

    The file names default to those of upstream's post-draft step. Writes ``output_bib`` and
    ``output_provenance`` and returns the provenance payload.

    Raises:
        ValueError: When no BibTeX entry is available.
    """
    bib_text, payload = merge_reference_payloads(
        existing_bib=await _read_text_or_empty(workspace, join(paper_dir, existing_bib)),
        existing_provenance=await _read_json_object(
            workspace, join(paper_dir, existing_provenance)
        ),
        incremental_bib=await _read_text_or_empty(workspace, join(paper_dir, incremental_bib)),
        incremental_provenance=await _read_json_object(
            workspace, join(paper_dir, incremental_provenance)
        ),
        results_dir=results_dir,
        campaign_results_dir=campaign_results_dir,
    )
    await workspace.write_bytes(join(paper_dir, output_bib), bib_text.encode("utf-8"))
    await workspace.write_bytes(
        join(paper_dir, output_provenance),
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"),
    )
    return payload


__all__ = ["merge_reference_artifacts", "merge_reference_payloads"]
