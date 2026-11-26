"""Finalisation of a paper (``paper_finalize.py`` ``finalize_paper_artifacts``).

After the loop ends with a validated paper, upstream appends a fallback ``paper`` entry to the
global memory when the writers recorded none, and writes ``latest_manifest.json`` (in the paper
root, or the manifest directory a caller names) with the paths of every published artefact and
the state of the embedded Supplementary Information. An incomplete paper is not finalised
(upstream exits with status 2 before this step); the port returns its result with the reasons.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from ..core.results import PaperResult, ValidationReport
from ..core.state import Failure, utc_now
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .contexts import read_json_or_none, read_text_or_none, write_json

if TYPE_CHECKING:
    from ..context import ScientistContext

SUPPLEMENT_BOOLEAN_CHECKS = (
    "checked",
    "appendix_present",
    "heading_present",
    "placed_after_references",
    "pre_bibliography_clearpage_present",
    "clearpage_present",
    "rendered_checked",
    "rendered_heading_present",
)
SUPPLEMENT_COUNTS = (
    ("visible_character_count", 100),
    ("substantive_element_count", 1),
    ("rendered_visible_character_count", 100),
)


def supplement_manifest_state(validation: Mapping[str, Any] | None) -> tuple[bool, str]:
    """``_supplement_manifest_state``."""
    if validation is None:
        return False, "missing_validation"
    checks = all(validation.get(key) is True for key in SUPPLEMENT_BOOLEAN_CHECKS)
    counts = True
    for key, minimum in SUPPLEMENT_COUNTS:
        value = validation.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            counts = False
            break
    if validation.get("hard_errors") == [] and checks and counts:
        return True, "validated"
    return False, "validation_failed"


def format_memory_markdown_entry(entry: Mapping[str, Any]) -> str:
    """Upstream ``format_memory_markdown_entry`` (``orchestrator/state.py``)."""

    def field(key: str, missing: str) -> str:
        return str(entry.get(key, "")).strip() or missing

    lines = [
        f"### {field('timestamp', 'unknown_ts')} | step={field('step', 'unknown_step')} | "
        f"scope={field('scope', 'unknown_scope')}",
        f"- summary: {field('summary', '(missing summary)')}",
        f"- evidence: {field('evidence', '(missing evidence)')}",
        f"- next_action: {field('next_action', '(missing next_action)')}",
    ]
    for key in ("idea_id", "attempt_id", "program_id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"- {key}: {value.strip()}")
    metrics = entry.get("metrics")
    if isinstance(metrics, dict) and metrics:
        lines.append(f"- metrics: {json.dumps(metrics, sort_keys=True)}")
    rubrics = entry.get("rubrics")
    if isinstance(rubrics, list) and rubrics:
        lines.append(f"- rubrics: {json.dumps(rubrics, sort_keys=True)}")
    tags = entry.get("tags")
    if isinstance(tags, list) and tags:
        lines.append(f"- tags: {', '.join(str(item) for item in tags)}")
    return "\n".join(lines) + "\n\n"


async def _append(workspace: Workspace, path: str, text: str) -> None:
    current = await read_text_or_none(workspace, path) or ""
    await workspace.write_bytes(path, (current + text).encode("utf-8"))


async def append_fallback_memory(
    workspace: Workspace, memory_dirs: Sequence[str], supplement_embedded: bool
) -> bool:
    """Append upstream's fallback ``paper`` memory entry when no writer recorded one.

    ``memory_dirs`` are the paper workspace's ``global_memory`` copy and, when known, the global
    memory directory it was copied from (upstream appends through a symlink to the latter).
    Returns whether an entry was appended.
    """
    primary = memory_dirs[0]
    text = await read_text_or_none(workspace, join(primary, "memory.jsonl")) or ""
    for line in text.splitlines():
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed.get("step") == "paper":
            return False
    evidence = (
        "paper.tex (validated embedded Supplementary Information)"
        if supplement_embedded
        else "paper.tex (Supplementary Information validation incomplete)"
    )
    entry = {
        "timestamp": utc_now(),
        "step": "paper",
        "scope": "paper",
        "summary": "paper artifacts generated (auto-healed by paper.sh)",
        "evidence": f"{evidence} + claims_grounding.json + validation_report.json",
        "next_action": "publish latest_manifest.json",
    }
    for directory in dict.fromkeys(memory_dirs):
        if not await workspace.exists(directory):
            continue
        await _append(
            workspace, join(directory, "memory.jsonl"), json.dumps(entry, sort_keys=True) + "\n"
        )
        await _append(workspace, join(directory, "memory.md"), format_memory_markdown_entry(entry))
    return True


async def finalize_paper_artifacts(
    ctx: ScientistContext,
    *,
    paper_dir: str,
    manifest_root: str,
    memory_dirs: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """Write ``latest_manifest.json`` and the fallback memory entry; return the manifest path
    and the manifest."""
    ws = ctx.workspace
    report_value = await read_json_or_none(ws, join(paper_dir, "validation_report.json"))
    report: dict[str, Any] = report_value if isinstance(report_value, dict) else {}
    supplement_value = report.get("supplementary_information_validation")
    supplement = supplement_value if isinstance(supplement_value, dict) else None
    embedded, status = supplement_manifest_state(supplement)
    await append_fallback_memory(ws, memory_dirs, embedded)
    paper_data = await read_json_or_none(ws, join(paper_dir, "paper_data.json"))

    async def present(name: str) -> str | None:
        path = join(paper_dir, name)
        return path if await ws.exists(path) else None

    manifest = {
        "generated_at": utc_now(),
        "paper_profile": paper_data.get("paper_profile", "default")
        if isinstance(paper_data, Mapping)
        else "default",
        "paper_dir": paper_dir,
        "paper_tex": join(paper_dir, "paper.tex"),
        "paper_pdf": join(paper_dir, "paper.pdf"),
        "supplementary_information": {
            "embedded": embedded,
            "status": status,
            "paper_tex": join(paper_dir, "paper.tex"),
            "paper_pdf": join(paper_dir, "paper.pdf"),
            "validation": supplement,
        },
        "paper_prompt": join(paper_dir, "paper_prompt.md"),
        "paper_data": join(paper_dir, "paper_data.json"),
        "references_bib": join(paper_dir, "references.bib"),
        "references_provenance": join(paper_dir, "references_provenance.json"),
        "claims_grounding": join(paper_dir, "claims_grounding.json"),
        "figure_manifest": await present("figure_manifest.json"),
        "paper_evidence_audit_json": await present("paper_evidence_audit.json"),
        "paper_evidence_audit_md": await present("paper_evidence_audit.md"),
        "paper_evidence_audit_feedback_json": await present("paper_evidence_audit_feedback.json"),
        "paper_evidence_audit_feedback_md": await present("paper_evidence_audit_feedback.md"),
        "validation_feedback_json": await present("validation_feedback.json"),
        "validation_feedback_md": await present("validation_feedback.md"),
        "validation_report": await present("validation_report.json"),
    }
    path = join(manifest_root, "latest_manifest.json")
    await write_json(ws, path, manifest)
    return path, manifest


def paper_result(
    *,
    status: Literal["complete", "incomplete", "failed", "skipped"],
    paper_dir: str | None,
    validation: ValidationReport | None,
    manifest_path: str | None = None,
    reasons: Sequence[str] = (),
    failure: Failure | None = None,
) -> PaperResult:
    """The :class:`PaperResult` of a finished loop."""
    return PaperResult(
        status=status,
        paper_dir=paper_dir,
        final_dir=paper_dir,
        pdf_path=join(paper_dir, "paper.pdf") if paper_dir else None,
        tex_path=join(paper_dir, "paper.tex") if paper_dir else None,
        manifest_path=manifest_path,
        incomplete_reason="; ".join(reasons) or None,
        validation=validation,
        failure=failure,
    )


__all__ = [
    "append_fallback_memory",
    "finalize_paper_artifacts",
    "format_memory_markdown_entry",
    "paper_result",
    "supplement_manifest_state",
]
