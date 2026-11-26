"""The evidence-audit gate (R33): read-only, on a disposable copy, with an input-hash check.

Ports ``run_agentic_evidence_audit_gate`` (``paper.sh``), ``run_required_evidence_audit``
(``paper_edit.sh``) and the helpers of ``paper_evidence_audit.py``. Each contract attempt:

1. ``prepare``: render the audit prompt and keep it as ``paper_evidence_audit_prompt*.md``;
   check the final-evaluation grounding of ``paper_data.json`` against the verified paper data;
   copy the paper directory to a disposable workspace beside it (the agents' skill directory
   excluded), refresh the scientific-evidence verification report there when the verifier is
   staged, build ``paper_evidence_audit_packet.md`` and record the SHA-256 of the audit inputs;
   install the packet and the hash manifest in the paper directory.
2. ``session``: the evidence auditor reads the copy through a read-only backend and answers with
   :class:`~langgraph_agentic_scientist.paper.models.AuditPayload` (adaptation A5).
3. ``finalize``: remove the copy; recompute the input hashes of the paper directory and refuse an
   audit whose inputs changed; write ``paper_evidence_audit.json``/``.md``, the audit feedback and
   the merged ``validation_report.json``; classify as upstream's return codes do: 0 pass,
   2 findings, 1 operational failure, 3 contract exhausted after two attempts.

The gate is the replaceable slot ``paper.audit``: :func:`build_audit_graph` returns its default.
"""

from __future__ import annotations

import hashlib
import json
import operator
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph import END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Checkpointer

from ..agents.runner import agent_node_retry_policy, run_structured_role
from ..context import GraphOverrides, ScientistContext
from ..core.results import PaperData
from ..core.runconfig import method_config
from ..core.state import Event, RunConfigRecord, TaskRef, merge_dicts
from ..core.steps import StepGraph
from ..models import Role
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .contexts import (
    read_json_or_none,
    read_text_or_none,
    render_stage_prompt,
    workflow_text,
    write_json,
    write_stage_contract_feedback,
    write_stage_execution_feedback,
    write_text,
    write_validation_feedback_from_current_report,
)
from .models import AuditPayload, AuditResult

MAX_TEX_CHARS = 30000
MAX_JSON_CHARS = 14000
MAX_CONTRACT_ATTEMPTS = 2
"""Upstream ``max_contract_attempts`` of the audit gate."""

AUDIT_INPUT_PATHS = (
    "paper.tex",
    "claims_grounding.json",
    "paper_data.json",
    "references.bib",
    "references_provenance.json",
    "figure_manifest.json",
    "scientific-evidence/verification-report.json",
)
VALIDATION_INPUT_PATHS = (
    *AUDIT_INPUT_PATHS,
    "VENUE.md",
    "paper_authors.tex",
    "paper_strategy.md",
    "paper_strategy.json",
    "method_scope_audit.md",
    "method_scope_audit.json",
)
AUDIT_INPUT_HASH_MANIFEST = ".paper_evidence_audit_input_sha256.json"
VOLATILE_VERIFICATION_FIELDS = frozenset({"root_dir", "verified_at"})
EVIDENCE_SCAN_EXTENSIONS = frozenset(
    {
        ".bash",
        ".cfg",
        ".conf",
        ".csv",
        ".ini",
        ".js",
        ".json",
        ".jsonl",
        ".jsx",
        ".md",
        ".py",
        ".r",
        ".R",
        ".rst",
        ".sh",
        ".sql",
        ".tex",
        ".toml",
        ".ts",
        ".tsv",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
        ".zsh",
    }
)
EVIDENCE_SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "scientific-evidence",
        ".agents",
    }
)
EVIDENCE_GENERATED_NAMES = frozenset(
    {
        "validation_feedback.json",
        "validation_feedback.md",
        "validation_report.json",
        "verification-report.json",
        ".latest_manifest.lock",
    }
)
GRAPHICS_SUFFIXES = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".eps"})
PROMPT_ARTIFACT_RE = re.compile(r".+_prompt(?:_round_\d+(?:_contract_retry_\d+)?)?\.md$")
SCIENTIFIC_EVIDENCE_SCRIPT = ".agents/skills/scientific-evidence/scripts/scientific_evidence.py"


# --- hashing -------------------------------------------------------------------------------------


def _strip_watermark(text: str) -> str:
    try:
        from .watermark import strip_managed_block
    except ImportError:  # pragma: no cover - the host side ships it
        return text
    return strip_managed_block(text)


def _stable_input_bytes(relative: str, data: bytes) -> bytes:
    if relative == "paper.tex":
        return _strip_watermark(data.decode("utf-8", errors="replace")).encode("utf-8")
    if relative == "scientific-evidence/verification-report.json":
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return data
        if isinstance(payload, dict):
            stable = {k: v for k, v in payload.items() if k not in VOLATILE_VERIFICATION_FIELDS}
            return json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return data


def is_generated_audit_artifact(relative: str) -> bool:
    """``_is_generated_audit_artifact``: files the host writes while an audit runs."""
    parts = PurePosixPath(relative).parts
    if parts and parts[0] in ("paper_diagnostics", "revisions"):
        return True
    if relative == "scientific-evidence/verification-report.json":
        return True
    if relative in ("metrics/cost_events.jsonl", "metrics/cost_summary.json"):
        return True
    if len(parts) != 1:
        return False
    name = parts[0]
    return bool(
        name in EVIDENCE_GENERATED_NAMES
        or name == AUDIT_INPUT_HASH_MANIFEST
        or name.startswith("paper_evidence_audit")
        or PROMPT_ARTIFACT_RE.fullmatch(name)
    )


async def paper_input_sha256(
    workspace: Workspace, paper_dir: str, paths: tuple[str, ...] = AUDIT_INPUT_PATHS
) -> dict[str, str]:
    """``_paper_input_sha256`` over ``paths`` plus the referenced figures."""
    hashes: dict[str, str] = {}
    for relative in paths:
        path = join(paper_dir, relative)
        entry = await workspace.stat(path)
        if entry is None or entry.kind != "file":
            continue
        data = _stable_input_bytes(relative, await workspace.read_bytes(path))
        hashes[relative] = hashlib.sha256(data).hexdigest()
    figures = join(paper_dir, "figures")
    if await workspace.exists(figures):
        for entry in await workspace.list_tree(figures, with_digest=True):
            if (
                entry.kind == "file"
                and PurePosixPath(entry.path).suffix.lower() in GRAPHICS_SUFFIXES
            ):
                digest = (entry.sha256 or "").removeprefix("sha256:")
                if not digest:
                    digest = hashlib.sha256(
                        await workspace.read_bytes(join(figures, entry.path))
                    ).hexdigest()
                hashes[f"figure-asset/figures/{entry.path}"] = digest
    return hashes


async def audit_input_sha256(workspace: Workspace, paper_dir: str) -> dict[str, str]:
    """``audit_input_sha256``: the audit inputs plus every evidence source of the workspace."""
    hashes = await paper_input_sha256(workspace, paper_dir)
    for entry in await workspace.list_tree(paper_dir, with_digest=True):
        if entry.kind != "file":
            continue
        parts = PurePosixPath(entry.path).parts
        if any(part in EVIDENCE_SKIP_DIR_NAMES for part in parts[:-1]):
            continue
        if is_generated_audit_artifact(entry.path):
            continue
        name = parts[-1]
        suffix = PurePosixPath(name).suffix
        if not (
            name.endswith(".prov.json")
            or suffix in EVIDENCE_SCAN_EXTENSIONS
            or name in ("Dockerfile", "Makefile")
        ):
            continue
        digest = (entry.sha256 or "").removeprefix("sha256:")
        if not digest:
            digest = hashlib.sha256(
                await workspace.read_bytes(join(paper_dir, entry.path))
            ).hexdigest()
        hashes[f"evidence-source/{entry.path}"] = digest
    ledger = join(paper_dir, "scientific-evidence/ledger.jsonl")
    if await workspace.exists(ledger):
        hashes["evidence-source/scientific-evidence/ledger.jsonl"] = hashlib.sha256(
            await workspace.read_bytes(ledger)
        ).hexdigest()
    return hashes


# --- packet ------------------------------------------------------------------------------------


def truncate(text: str, *, max_chars: int) -> str:
    """Upstream ``_truncate``."""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return f"{stripped[:max_chars].rstrip()}\n\n[truncated after {max_chars} chars]"


def json_excerpt(payload: Any, *, max_chars: int = MAX_JSON_CHARS) -> str:
    """Upstream ``_json_excerpt``."""
    return truncate(json.dumps(payload, indent=2, sort_keys=True), max_chars=max_chars)


def _error_summary(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, Mapping) and isinstance(payload.get("_json_error"), str):
        return {"available": False, "error": payload["_json_error"]}
    return None


def final_outcome_summary(paper_data: Mapping[str, Any]) -> dict[str, Any] | None:
    """Upstream ``final_outcome_summary``."""
    final = paper_data.get("final_evaluation")
    if not isinstance(final, Mapping):
        return None
    return {k: final[k] for k in ("status", "correct", "outcome", "limitations") if k in final}


def split_manuscript(tex: str) -> tuple[str, str | None]:
    """Split the manuscript from its Supplementary Information (the host side's splitter)."""
    try:
        from .validation.supplement import split_manuscript_and_supplement
    except ImportError:  # pragma: no cover - the host side ships it
        return tex, None
    return split_manuscript_and_supplement(tex)


def _paper_data_summary(payload: Any) -> dict[str, Any]:
    error = _error_summary(payload)
    if error is not None:
        return error
    if not isinstance(payload, Mapping):
        return {"available": False}
    grounding = payload.get("grounding_sources")
    claims = payload.get("claims")
    totals = payload.get("totals")
    return {
        "available": True,
        "paper_profile": payload.get("paper_profile", "default"),
        "downstream_study": payload.get("downstream_study"),
        "final_evaluation": final_outcome_summary(payload),
        "grounding_source_count": len(grounding) if isinstance(grounding, Mapping) else 0,
        "claim_count": len(claims) if isinstance(claims, list) else 0,
        "totals": totals if isinstance(totals, Mapping) else {},
    }


def _evidence_summary(payload: Any) -> dict[str, Any]:
    error = _error_summary(payload)
    if error is not None:
        return error
    if not isinstance(payload, Mapping):
        return {"available": False}
    raw_summary = payload.get("summary")
    summary: Mapping[str, Any] = raw_summary if isinstance(raw_summary, Mapping) else {}
    raw_results = payload.get("results")
    results: list[Any] = raw_results if isinstance(raw_results, list) else []
    preview: list[dict[str, Any]] = []
    for row in results:
        if not isinstance(row, Mapping) or row.get("status") != "dangling":
            continue
        preview.append(
            {
                "subject_path": row.get("subject_path"),
                "raw_reference": row.get("raw_reference"),
                "target": row.get("target"),
                "messages": row.get("messages") if isinstance(row.get("messages"), list) else [],
            }
        )
        if len(preview) >= 10:
            break
    keys = (
        "total_references",
        "resolved_references",
        "anchored_references",
        "dangling_references",
        "manifests_checked",
    )
    return {
        "available": True,
        "summary": {key: summary.get(key, 0) for key in keys},
        "dangling_preview": preview,
    }


def _validation_summary(payload: Any) -> dict[str, Any]:
    error = _error_summary(payload)
    if error is not None:
        return error
    if not isinstance(payload, Mapping):
        return {"available": False}
    raw_evidence = payload.get("evidence_validation")
    evidence: Mapping[str, Any] = raw_evidence if isinstance(raw_evidence, Mapping) else {}
    raw_summary = evidence.get("summary")
    summary: Mapping[str, Any] = raw_summary if isinstance(raw_summary, Mapping) else {}
    return {
        "available": True,
        "ok": payload.get("ok"),
        "pdf_path": payload.get("pdf_path"),
        "evidence_validation": {
            "checked": bool(evidence.get("checked", False)),
            "bundle_present": bool(evidence.get("bundle_present", False)),
            "paper_token_count": evidence.get("paper_token_count", 0),
            **{
                key: summary.get(key, 0)
                for key in (
                    "total_references",
                    "resolved_references",
                    "anchored_references",
                    "dangling_references",
                )
            },
        },
    }


async def build_audit_packet(ctx: ScientistContext, paper_dir: str, output_path: str) -> None:
    """``build_audit_packet``: the compact entry point of the evidence auditor.

    Raises:
        FileNotFoundError: When ``paper.tex`` is missing.
    """
    ws = ctx.workspace
    tex = await read_text_or_none(ws, join(paper_dir, "paper.tex"))
    if tex is None:
        raise FileNotFoundError(
            f"Missing required manuscript artifact: {join(paper_dir, 'paper.tex')}"
        )
    lines = [
        "# Agentic evidence audit packet",
        "",
        f"- generated_at: `{datetime.now(UTC).isoformat()}`",
        f"- paper_dir: `{paper_dir}`",
        "",
        *workflow_text(ctx, "audit_packet_intro.md").split("\n"),
        "",
        "## Artifact checklist",
        "",
    ]
    for name in (
        "paper.tex",
        "claims_grounding.json",
        "paper_data.json",
        "validation_report.json",
        "scientific-evidence/verification-report.json",
        "references_provenance.json",
    ):
        state = "present" if await ws.exists(join(paper_dir, name)) else "missing"
        lines.append(f"- `{name}` — `{state}`")
    manuscript, supplement = split_manuscript(tex)
    lines.extend(
        [
            "",
            "## validation report summary",
            "",
            "```json",
            json_excerpt(
                _validation_summary(
                    await read_json_or_none(ws, join(paper_dir, "validation_report.json"))
                )
            ),
            "```",
            "",
            "## scientific-evidence verification summary",
            "",
            "```json",
            json_excerpt(
                _evidence_summary(
                    await read_json_or_none(
                        ws, join(paper_dir, "scientific-evidence/verification-report.json")
                    )
                )
            ),
            "```",
            "",
            "## claims_grounding excerpt",
            "",
            "```json",
            json_excerpt(await read_json_or_none(ws, join(paper_dir, "claims_grounding.json"))),
            "```",
            "",
            "## paper_data summary",
            "",
            "```json",
            json_excerpt(
                _paper_data_summary(await read_json_or_none(ws, join(paper_dir, "paper_data.json")))
            ),
            "```",
            "",
            "## references provenance summary",
            "",
            "```json",
            json_excerpt(
                await read_json_or_none(ws, join(paper_dir, "references_provenance.json"))
            ),
            "```",
            "",
            "## manuscript source excerpt",
            "",
            "```tex",
            truncate(manuscript, max_chars=MAX_TEX_CHARS),
            "```",
            "",
        ]
    )
    if supplement is not None:
        lines.extend(
            [
                "## supplementary information excerpt",
                "",
                "```tex",
                truncate(supplement, max_chars=MAX_TEX_CHARS),
                "```",
                "",
            ]
        )
    await write_text(ws, output_path, "\n".join(lines))


async def refresh_evidence_report(ctx: ScientistContext, root: str, script: str) -> str | None:
    """``_refresh_evidence_verification_report``: rerun the vendored verifier when it is staged.

    Returns an error message when the verifier fails operationally, ``None`` otherwise (also when
    there is no evidence trail or no staged verifier).
    """
    ws = ctx.workspace
    if not await ws.exists(join(root, "scientific-evidence")) or not await ws.exists(script):
        return None
    output = join(root, "scientific-evidence", ".verification-report.tmp.json")
    result = await ws.run(
        [
            "python3",
            ws.host_path(script),
            "verify",
            "--root-dir",
            ws.host_path(root),
            "--output",
            ws.host_path(output),
        ],
        cwd=root,
        timeout=300,
    )
    if result.returncode not in (0, 1) or not await ws.exists(output):
        detail = (result.stdout + "\n" + result.stderr).strip()
        await ws.remove(output)
        return (
            "Failed to refresh scientific-evidence verification report in readonly workspace: "
            f"verifier exited {result.returncode}. {detail}"
        )
    await ws.write_bytes(
        join(root, "scientific-evidence", "verification-report.json"), await ws.read_bytes(output)
    )
    await ws.remove(output)
    return None


def verify_final_outcome(actual: Any, expected: PaperData | None) -> str | None:
    """The final-evaluation check of ``prepare_readonly_paper_audit_workspace``.

    ``paper_data.json`` must keep the verified ``final_evaluation`` and every ``final.*`` and
    ``study.final.*`` grounding source of the paper data the phase built. Returns the error.
    """
    if expected is None:
        return None
    reference = expected.model_dump(mode="json", by_alias=True)
    if "final_evaluation" not in reference:
        return None
    if not isinstance(actual, Mapping):
        return "Paper final evidence must contain an object"
    if actual.get("final_evaluation") != reference.get("final_evaluation"):
        return "Paper final evidence differs from verified final evaluation"

    def final_sources(data: Mapping[str, Any]) -> dict[str, Any]:
        sources = data.get("grounding_sources")
        if not isinstance(sources, Mapping):
            return {}
        return {k: v for k, v in sources.items() if k.startswith(("final.", "study.final."))}

    if not isinstance(actual.get("grounding_sources"), Mapping):
        return "Paper final evidence grounding is missing"
    if final_sources(actual) != final_sources(reference):
        return "Paper final evidence grounding differs from verified final evaluation"
    return None


def render_audit_markdown(audit: AuditPayload) -> str:
    """Upstream ``render_audit_markdown``."""
    lines = [
        "# Agentic evidence audit",
        "",
        f"- verdict: `{audit.verdict}`",
        f"- finding_count: `{len(audit.findings)}`",
        "",
        "## Summary",
        audit.summary,
        "",
        "## Findings",
    ]
    if not audit.findings:
        lines.extend(["- None.", ""])
        return "\n".join(lines)
    for index, finding in enumerate(audit.findings, start=1):
        lines.extend(
            [
                f"### {index}. {finding.id} — {finding.section}",
                f"- claim: {finding.claim}",
                f"- issue: {finding.issue}",
                f"- evidence: {finding.evidence}",
                f"- requested_fix: {finding.fix}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


async def write_audit_artifacts(
    workspace: Workspace, paper_dir: str, audit: AuditPayload, input_hashes: Mapping[str, str]
) -> None:
    """``finalize_audit`` after parsing: the audit JSON and markdown, the feedback, and the
    ``agentic_evidence_audit`` block of ``validation_report.json``."""
    findings = [finding.model_dump(mode="json") for finding in audit.findings]
    raw_path = join(paper_dir, "paper_evidence_audit_raw.txt")
    json_path = join(paper_dir, "paper_evidence_audit.json")
    md_path = join(paper_dir, "paper_evidence_audit.md")
    await write_text(
        workspace, raw_path, json.dumps(audit.model_dump(mode="json"), indent=2) + "\n"
    )
    await write_text(workspace, md_path, render_audit_markdown(audit))
    feedback = {
        "generated_at": datetime.now(UTC).isoformat(),
        "ok": audit.verdict == "pass",
        "verdict": audit.verdict,
        "summary": audit.summary,
        "finding_count": len(findings),
        "audit_json_path": json_path,
        "audit_markdown_path": md_path,
        "raw_output_path": raw_path,
        "findings": findings,
    }
    await write_json(workspace, join(paper_dir, "paper_evidence_audit_feedback.json"), feedback)
    lines = [
        "# Agentic evidence audit feedback",
        "",
        f"- generated_at: `{feedback['generated_at']}`",
        f"- ok: `{feedback['ok']}`",
        f"- verdict: `{audit.verdict}`",
        f"- finding_count: `{len(findings)}`",
        f"- audit_json_path: `{json_path}`",
        f"- audit_markdown_path: `{md_path}`",
        f"- raw_output_path: `{raw_path}`",
        "",
        "## Summary",
        audit.summary,
        "",
        "## Findings",
    ]
    if findings:
        for item in findings:
            lines.append(f"- `{item['id']}` ({item['section']}): {item['issue']}")
            lines.append(f"  - claim: {item['claim']}")
            lines.append(f"  - evidence: {item['evidence']}")
            lines.append(f"  - requested_fix: {item['fix']}")
    else:
        lines.append("- None.")
    lines.append("")
    await write_text(
        workspace, join(paper_dir, "paper_evidence_audit_feedback.md"), "\n".join(lines)
    )
    report_path = join(paper_dir, "validation_report.json")
    report = await read_json_or_none(workspace, report_path)
    if not isinstance(report, dict) or "_json_error" in report:
        report = {}
    report["agentic_evidence_audit"] = {
        "checked": True,
        "verdict": audit.verdict,
        "summary": audit.summary,
        "finding_count": len(findings),
        "findings": findings,
        "input_sha256": dict(input_hashes),
    }
    await write_json(workspace, report_path, report)
    payload = audit.model_dump(mode="json")
    payload["finding_count"] = len(findings)
    payload["input_sha256"] = dict(input_hashes)
    await write_json(workspace, json_path, payload)


STALE_AUDIT_ARTIFACTS = (
    "paper_evidence_audit_stage_feedback.json",
    "paper_evidence_audit_stage_feedback.md",
    "paper_evidence_audit_packet.md",
    "paper_evidence_audit_input_sha256.json",
    "paper_evidence_audit_raw.txt",
    "paper_evidence_audit.json",
    "paper_evidence_audit.md",
    "paper_evidence_audit_feedback.json",
    "paper_evidence_audit_feedback.md",
)


# --- the gate as a graph ---------------------------------------------------------------------


class AuditInput(TypedDict):
    """What the audit gate reads from the paper loop."""

    run_config: RunConfigRecord | None
    task: TaskRef
    paper_dir: str
    mode: NotRequired[str]
    paper_data: NotRequired[PaperData | None]
    writer_prompt_key: NotRequired[str]
    writer_skill: NotRequired[str]
    paper_format: NotRequired[str]
    two_stage: NotRequired[bool]
    repair_round: NotRequired[int]


class AuditOutput(TypedDict):
    """What the audit gate returns: the outcome and its events."""

    audit: AuditResult
    events: list[Event]


class AuditState(AuditInput, total=False):
    """The gate's own state across its contract attempts."""

    audit: AuditResult
    events: Annotated[list[Event], operator.add]
    hook_outputs: Annotated[dict[str, Any], merge_dicts]
    contract_attempt: int
    copy_dir: str | None
    prompt: str
    prompt_path: str
    input_hashes: dict[str, str]
    payload: AuditPayload | None
    session_failure: str | None
    contract_failure: str | None
    operational: str | None


def _copy_dir(paper_dir: str, tag: str) -> str:
    """The disposable copy of one audit attempt, beside the paper directory."""
    parent = paper_dir.rsplit("/", 1)[0] if "/" in paper_dir else ""
    name = paper_dir.rsplit("/", 1)[-1]
    return join(parent, ".paper_evidence_audit_workspaces", f"{name}-{tag}")


async def _prepare(state: AuditState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ctx = runtime.context
    ws = ctx.workspace
    paper_dir = state["paper_dir"]
    attempt = int(state.get("contract_attempt") or 0) + 1
    audit_round = int(state.get("repair_round") or 0) + 1
    if attempt == 1:
        for name in STALE_AUDIT_ARTIFACTS:
            await ws.remove(join(paper_dir, name))
    if audit_round == 1 and attempt == 1 and state.get("mode", "fresh") == "fresh":
        prompt_name = "paper_evidence_audit_prompt.md"
    elif attempt == 1:
        prompt_name = f"paper_evidence_audit_prompt_round_{audit_round}.md"
    else:
        prompt_name = f"paper_evidence_audit_prompt_round_{audit_round}_contract_retry_{attempt}.md"
    prompt = await render_stage_prompt(state, runtime, "paper/evidence_audit")
    prompt_path = join(paper_dir, prompt_name)
    await write_text(ws, prompt_path, prompt)
    update: dict[str, Any] = {
        "contract_attempt": attempt,
        "prompt": prompt,
        "prompt_path": prompt_path,
        "payload": None,
        "session_failure": None,
        "contract_failure": None,
        "operational": None,
        "copy_dir": None,
    }
    error = verify_final_outcome(
        await read_json_or_none(ws, join(paper_dir, "paper_data.json")), state.get("paper_data")
    )
    if error is not None:
        update["operational"] = error
        return update
    copy_dir = _copy_dir(paper_dir, f"round{audit_round}-attempt{attempt}")
    await ws.remove(copy_dir)
    await ws.copy_tree(paper_dir, copy_dir, ignore=("/.agents/",))
    refresh_error = await refresh_evidence_report(
        ctx, copy_dir, join(paper_dir, SCIENTIFIC_EVIDENCE_SCRIPT)
    )
    if refresh_error is not None:
        await ws.remove(copy_dir)
        update["operational"] = refresh_error
        return update
    try:
        await build_audit_packet(ctx, copy_dir, join(copy_dir, "paper_evidence_audit_packet.md"))
    except FileNotFoundError as exc:
        await ws.remove(copy_dir)
        update["operational"] = str(exc)
        return update
    hashes = await audit_input_sha256(ws, copy_dir)
    await write_json(ws, join(copy_dir, AUDIT_INPUT_HASH_MANIFEST), hashes)
    await ws.write_bytes(
        join(paper_dir, "paper_evidence_audit_packet.md"),
        await ws.read_bytes(join(copy_dir, "paper_evidence_audit_packet.md")),
    )
    await write_json(ws, join(paper_dir, "paper_evidence_audit_input_sha256.json"), hashes)
    update["copy_dir"] = copy_dir
    update["input_hashes"] = hashes
    return update


async def _session(state: AuditState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    if state.get("operational") or not state.get("copy_dir"):
        return {}
    cfg = method_config(state).paper
    outcome = await run_structured_role(
        role=Role.EVIDENCE_AUDITOR,
        schema=AuditPayload,
        runtime=runtime,
        root=state["copy_dir"],
        read_only=True,
        prompt=state["prompt"],
        timeout=cfg.timeout_seconds,
    )
    if isinstance(outcome.structured, AuditPayload):
        return {"payload": outcome.structured}
    assert outcome.failure is not None
    if outcome.failure.category == "structured_response_missing":
        return {"contract_failure": outcome.failure.detail or "no structured audit answer"}
    return {"session_failure": f"{outcome.failure.category}: {outcome.failure.detail}"}


async def _finalize(state: AuditState, runtime: Runtime[ScientistContext]) -> dict[str, Any]:
    ws = runtime.context.workspace
    paper_dir = state["paper_dir"]
    attempt = int(state.get("contract_attempt") or 1)
    copy_dir = state.get("copy_dir")
    if copy_dir:
        await ws.remove(copy_dir)
    prompt_path = state.get("prompt_path") or join(paper_dir, "paper_evidence_audit_prompt.md")
    stage_name = PurePosixPath(prompt_path).name.removesuffix(".md").replace("_prompt", "")
    if state.get("operational"):
        return {
            "audit": AuditResult(
                outcome="operational_failure", attempts=attempt, detail=str(state["operational"])
            ),
            "events": [
                Event(
                    kind="audit_operational_failure",
                    phase="paper",
                    step="audit",
                    data={"detail": str(state["operational"])[:2000]},
                )
            ],
        }
    payload = state.get("payload")
    if payload is not None:
        refresh_error = await refresh_evidence_report(
            runtime.context, paper_dir, join(paper_dir, SCIENTIFIC_EVIDENCE_SCRIPT)
        )
        current = await audit_input_sha256(ws, paper_dir)
        if refresh_error is not None or current != state.get("input_hashes"):
            detail = refresh_error or (
                "Paper inputs changed after the read-only audit snapshot was prepared"
            )
            return {
                "audit": AuditResult(
                    outcome="operational_failure", payload=payload, attempts=attempt, detail=detail
                ),
                "events": [
                    Event(
                        kind="audit_input_changed",
                        phase="paper",
                        step="audit",
                        data={"detail": detail},
                    )
                ],
            }
        await write_audit_artifacts(ws, paper_dir, payload, current)
        await ws.remove(join(paper_dir, "paper_evidence_audit_stage_feedback.json"))
        await ws.remove(join(paper_dir, "paper_evidence_audit_stage_feedback.md"))
        if payload.verdict == "pass":
            await write_validation_feedback_from_current_report(ws, paper_dir)
            return {"audit": AuditResult(outcome="pass", payload=payload, attempts=attempt)}
        await write_validation_feedback_from_current_report(
            ws,
            paper_dir,
            "agentic evidence audit failed: see "
            f"{join(paper_dir, 'paper_evidence_audit_feedback.json')} and "
            f"{join(paper_dir, 'paper_evidence_audit.md')}",
        )
        return {"audit": AuditResult(outcome="findings", payload=payload, attempts=attempt)}
    if state.get("session_failure"):
        await write_stage_execution_feedback(
            ws,
            paper_dir,
            feedback_json="paper_evidence_audit_stage_feedback.json",
            feedback_md="paper_evidence_audit_stage_feedback.md",
            stage_name=stage_name,
            failure_detail=str(state["session_failure"]),
            prompt_path=prompt_path,
            expected_outputs=(),
        )
        return {
            "audit": AuditResult(
                outcome="operational_failure",
                attempts=attempt,
                detail=str(state["session_failure"]),
            )
        }
    await write_stage_contract_feedback(
        ws,
        paper_dir,
        feedback_json="paper_evidence_audit_stage_feedback.json",
        feedback_md="paper_evidence_audit_stage_feedback.md",
        stage_name=stage_name,
        prompt_path=prompt_path,
        expected_outputs=(),
        hard_errors=[
            f"evidence audit stage '{stage_name}' wrote an invalid audit artifact: "
            f"{state.get('contract_failure') or 'no structured audit answer'}"
        ],
    )
    if attempt >= MAX_CONTRACT_ATTEMPTS:
        return {
            "audit": AuditResult(
                outcome="contract_exhausted",
                attempts=attempt,
                detail=str(state.get("contract_failure") or ""),
            ),
        }
    return {}


def _after_finalize(state: AuditState) -> str:
    return "done" if state.get("audit") is not None else "retry"


def build_audit_graph(
    *,
    checkpointer: Checkpointer = None,
    store: Any = None,
    overrides: GraphOverrides | None = None,
    name: str = "paper.audit",
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Return the default ``paper.audit`` slot: prepare, session and finalize per contract
    attempt (at most two), mounted as a subgraph of the paper loop.

    It runs mounted in the paper graph or alone (its input schema is the slot's input).

    Args:
        checkpointer: The caller's checkpointer when run alone; ``None`` inherits the parent's.
        store: The caller's store when run alone.
        overrides: Replacement slots (the steps carry hooks under the graph's phase name).
        name: The compiled graph's name.
    """
    graph = StepGraph(
        "paper.audit",
        AuditState,
        input_schema=AuditInput,
        output_schema=AuditOutput,
        overrides=overrides,
    )
    graph.step("prepare", _prepare)
    graph.step("session", _session, retry_policy=agent_node_retry_policy())
    graph.step("finalize", _finalize)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "session")
    graph.add_edge("session", "finalize")
    graph.add_conditional_edges("finalize", _after_finalize, {"done": END, "retry": "prepare"})
    return graph.compile(checkpointer=checkpointer, store=store, name=name)


__all__ = [
    "AUDIT_INPUT_PATHS",
    "MAX_CONTRACT_ATTEMPTS",
    "VALIDATION_INPUT_PATHS",
    "AuditInput",
    "AuditOutput",
    "AuditState",
    "audit_input_sha256",
    "build_audit_graph",
    "build_audit_packet",
    "final_outcome_summary",
    "is_generated_audit_artifact",
    "paper_input_sha256",
    "refresh_evidence_report",
    "render_audit_markdown",
    "verify_final_outcome",
    "write_audit_artifacts",
]
