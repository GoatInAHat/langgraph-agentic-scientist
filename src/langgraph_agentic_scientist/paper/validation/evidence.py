"""Scientific-evidence validation: every ``@evidence[[...]]`` link must resolve.

Upstream ``validate_evidence_bundle`` and its helpers (``paper_step.py``). The bundle is opt-in:
when ``scientific-evidence/`` is missing and no ``@evidence[[...]]`` token exists anywhere under
the paper directory, the check stays silent. When it is missing but tokens exist, it fails hard.
Otherwise the skill's own verifier script (vendored unchanged under
``skills/scientific-evidence/scripts``) classifies every link as ``resolved``, ``anchored`` or
``dangling``; the validator runs it through the workspace and validates its report.

The scan and the TeX dependency walk read a
:class:`~langgraph_agentic_scientist.paper.validation.files.PaperTree`; running the verifier is
asynchronous.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import os
import posixpath
import secrets
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Any

from ...prompts import PromptRegistry
from ...workspace.paths import join
from ...workspace.protocol import Workspace
from .files import PaperTree, resolve_inside
from .supplement import mask_tex_verbatim, strip_false_conditionals
from .tex import (
    AGENT_SKILL_DIR_PREFIXES,
    DYNAMIC_TEX_IMPORT_RE,
    EVIDENCE_REF_RE,
    PROMPT_ARTIFACT_RE,
    TEX_IMPORT_COMMAND_RE,
    TEX_INPUT_RE,
    TEX_LOCAL_SUPPORT_COMMAND_RE,
    TEX_LOCAL_SUPPORT_RE,
    _strip_tex_comments,
)

GENERATED_EVIDENCE_SCAN_ARTIFACT_NAMES = {
    "validation_feedback.json",
    "validation_feedback.md",
    "verification-report.json",
}
ROOT_GENERATED_EVIDENCE_SCAN_ARTIFACT_NAMES = {"validation_report.json"}
ROOT_EVIDENCE_ARCHIVE_DIR_PREFIXES = (("previous_paper",),)
GENERATED_EVIDENCE_SCAN_ARTIFACT_PREFIXES = ("paper_evidence_audit",)
EVIDENCE_BUNDLE_DIRNAME = "scientific-evidence"
EVIDENCE_SCAN_SUFFIXES = (
    ".tex",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".py",
    ".sh",
    ".r",
    ".csv",
    ".tsv",
)
EVIDENCE_SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    EVIDENCE_BUNDLE_DIRNAME,
}

VERIFICATION_SUMMARY_KEYS = (
    "total_references",
    "resolved_references",
    "anchored_references",
    "dangling_references",
    "manifests_checked",
)

WORKSPACE_ALIAS_DIRS = frozenset({"task", "results", "global_memory", "aggregated", "extra_tasks"})
"""Directories of the paper workspace that alias the run's task, results and memory.

Upstream makes them symbolic links, which its scan does not enter; a workspace without symbolic
links may hold copies, which the scan must not enter either.
"""

EVIDENCE_VERIFIER_PATH = ".agents/skills/scientific-evidence/scripts/scientific_evidence.py"
"""Where the validator places its own copy of the verifier in the paper directory."""


def _is_excluded_evidence_scan_path(rel: str) -> bool:
    """Return True for operational artifacts that should not enter evidence scans."""
    path = PurePosixPath(rel)
    rel_parts = path.parts
    if any(rel_parts[: len(prefix)] == prefix for prefix in AGENT_SKILL_DIR_PREFIXES):
        return True
    if any(rel_parts[: len(prefix)] == prefix for prefix in ROOT_EVIDENCE_ARCHIVE_DIR_PREFIXES):
        return True
    if rel_parts and rel_parts[0] in WORKSPACE_ALIAS_DIRS and len(rel_parts) > 1:
        return True
    if any(part in EVIDENCE_SKIP_DIR_NAMES for part in rel_parts):
        return True
    if len(rel_parts) == 1 and rel_parts[0] in ROOT_GENERATED_EVIDENCE_SCAN_ARTIFACT_NAMES:
        return True
    if path.name in GENERATED_EVIDENCE_SCAN_ARTIFACT_NAMES:
        return True
    if any(path.name.startswith(prefix) for prefix in GENERATED_EVIDENCE_SCAN_ARTIFACT_PREFIXES):
        return True
    return bool(PROMPT_ARTIFACT_RE.fullmatch(path.name))


def _tex_candidate(
    tree: PaperTree, source: str, base: str, target: str, *, suffixed: bool
) -> list[str]:
    """Return the workspace-relative candidates for an import ``target`` seen from ``base``."""
    joined = posixpath.normpath(posixpath.join(base, target))
    if joined.startswith("/") or joined == ".." or joined.startswith("../"):
        raise RuntimeError(
            f"paper.tex dependency escapes the paper workspace: {joined.lstrip('/')}"
        )
    relative = PurePosixPath(joined)
    if any(relative.parts[: len(prefix)] == prefix for prefix in AGENT_SKILL_DIR_PREFIXES):
        raise RuntimeError(f"paper.tex must not import agent skill files: {source}")
    candidates = [joined]
    if suffixed and not relative.suffix:
        candidates.append(relative.with_suffix(".tex").as_posix())
    return candidates


def _existing_file_dependency(tree: PaperTree, candidate: str) -> str | None:
    """Return ``candidate`` when it names a file reachable without symbolic links."""
    kind = tree.kind(candidate)
    if kind is None:
        return None
    if kind in {"dir", "other"}:
        return None
    resolved = resolve_inside(tree, candidate)
    if resolved is None:
        if kind == "symlink" and tree.readlink(candidate) is not None:
            target = tree.readlink(candidate) or ""
            if target.startswith("/") or ".." in PurePosixPath(target).parts:
                raise RuntimeError(f"paper.tex dependency escapes the paper workspace: {candidate}")
        return None
    if resolved != candidate:
        raise RuntimeError(f"paper.tex dependency must not use symlinks: {candidate}")
    if tree.kind(resolved) != "file":
        return None
    return resolved


def paper_tex_dependency_paths(tree: PaperTree) -> set[str]:
    """Return existing local files recursively imported by ``paper.tex``."""
    pending = ["paper.tex"]
    visited: set[str] = set()
    dependencies: set[str] = set()
    while pending:
        source = pending.pop()
        if source in visited or tree.kind(source) != "file":
            continue
        visited.add(source)
        raw = tree.read(source)
        if raw is None:
            continue
        try:
            text = strip_false_conditionals(
                mask_tex_verbatim(_strip_tex_comments(raw.decode("utf-8")))
            )
        except UnicodeDecodeError:
            continue
        if DYNAMIC_TEX_IMPORT_RE.search(text):
            raise RuntimeError(f"paper.tex dependency uses a dynamic import command: {source}")
        input_matches = list(TEX_INPUT_RE.finditer(text))
        parsed_import_starts = {match.start() for match in input_matches}
        if any(
            match.start() not in parsed_import_starts
            for match in TEX_IMPORT_COMMAND_RE.finditer(text)
        ):
            raise RuntimeError(f"paper.tex dependency uses a dynamic import command: {source}")
        source_parent = posixpath.dirname(source)
        for match in input_matches:
            target_text = (match.group("braced") or match.group("unbraced")).strip()
            if "\\" in target_text or "#" in target_text:
                raise RuntimeError(f"paper.tex dependency uses a dynamic import target: {source}")
            candidates: list[str] = []
            for base in ("", source_parent):
                candidates.extend(_tex_candidate(tree, source, base, target_text, suffixed=True))
            for candidate in candidates:
                found = _existing_file_dependency(tree, candidate)
                if found is None:
                    continue
                dependencies.add(found)
                pending.append(found)
                break
        support_matches = list(TEX_LOCAL_SUPPORT_RE.finditer(text))
        parsed_support_starts = {match.start() for match in support_matches}
        if any(
            match.start() not in parsed_support_starts
            for match in TEX_LOCAL_SUPPORT_COMMAND_RE.finditer(text)
        ):
            raise RuntimeError(f"paper.tex dependency uses a dynamic support import: {source}")
        for match in support_matches:
            suffix = ".cls" if "class" in match.group("command").lower() else ".sty"
            for raw_target in match.group("targets").split(","):
                target_text = raw_target.strip()
                if "\\" in target_text or "#" in target_text:
                    raise RuntimeError(
                        f"paper.tex dependency uses a dynamic support import: {source}"
                    )
                target = PurePosixPath(target_text)
                if not target.name:
                    continue
                if not target.suffix:
                    target = target.with_suffix(suffix)
                for base in ("", source_parent):
                    candidate = _tex_candidate(
                        tree, source, base, target.as_posix(), suffixed=False
                    )[0]
                    found = _existing_file_dependency(tree, candidate)
                    if found is None:
                        continue
                    dependencies.add(found)
                    pending.append(found)
                    break
    return dependencies


def scan_paper_for_evidence_tokens(tree: PaperTree) -> list[dict[str, Any]]:
    """Return every `@evidence[[...]]` occurrence found in paper text-like files."""
    occurrences: list[dict[str, Any]] = []
    imported_paths = paper_tex_dependency_paths(tree)
    for rel in tree.files():
        if rel not in imported_paths and _is_excluded_evidence_scan_path(rel):
            continue
        if (
            rel not in imported_paths
            and PurePosixPath(rel).suffix.lower() not in EVIDENCE_SCAN_SUFFIXES
        ):
            continue
        raw = tree.read(rel)
        if raw is None:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in EVIDENCE_REF_RE.finditer(line):
                occurrences.append({"file": rel, "line": line_no, "token": match.group(0)})
    return occurrences


def _is_usable_verification_payload(payload: object, *, expected_root: str) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_name") != "scientific-evidence.verification-report":
        return False
    if not all(
        isinstance(payload.get(key), str) and payload[key] for key in ("verified_at", "root_dir")
    ):
        return False
    if os.path.realpath(os.path.expanduser(payload["root_dir"])) != os.path.realpath(expected_root):
        return False
    summary = payload.get("summary")
    results = payload.get("results")
    if not isinstance(summary, dict) or not isinstance(results, list):
        return False
    if any(
        not isinstance(summary.get(key), int)
        or isinstance(summary.get(key), bool)
        or summary[key] < 0
        for key in VERIFICATION_SUMMARY_KEYS
    ):
        return False
    statuses: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            return False
        required_strings = ("subject_path", "subject_kind", "raw_reference", "target", "status")
        if any(not isinstance(result.get(key), str) or not result[key] for key in required_strings):
            return False
        status_value = result["status"]
        if status_value not in {"resolved", "anchored", "dangling"}:
            return False
        if result["subject_kind"] not in {"text-artifact", "manifest", "ledger-entry", "other"}:
            return False
        if any(
            key in result and (not isinstance(result[key], str) or not result[key])
            for key in ("locator", "resolved_path")
        ):
            return False
        if "hash_checked" in result and not isinstance(result["hash_checked"], bool):
            return False
        messages = result.get("messages")
        if not isinstance(messages, list) or any(
            not isinstance(message, str) for message in messages
        ):
            return False
        statuses.append(status_value)
    return bool(
        summary["total_references"] == len(results)
        and summary["resolved_references"] == statuses.count("resolved")
        and summary["anchored_references"] == statuses.count("anchored")
        and summary["dangling_references"] == statuses.count("dangling")
    )


async def run_evidence_verifier(
    workspace: Workspace,
    paper_dir: str,
    *,
    python: str,
    prompts: PromptRegistry | None = None,
    timeout: float | None = 600,
) -> tuple[dict[str, Any] | None, str | None]:
    """Run the vendored verifier on the paper directory and return its validated report.

    The verifier writes into a temporary file of the bundle directory; the report replaces
    ``scientific-evidence/verification-report.json`` only when it is schema-valid and its exit
    status agrees with its dangling count (upstream ``_run_evidence_verifier_atomically``).
    The validator writes its own copy of the verifier script before running it, so that a
    modified staged copy cannot change the verdict.
    """
    registry = prompts or PromptRegistry.default()
    script = registry.skill_files("scientific-evidence")["scripts/scientific_evidence.py"]
    script_path = join(paper_dir, EVIDENCE_VERIFIER_PATH)
    await workspace.write_bytes(script_path, script)
    bundle_dir = join(paper_dir, EVIDENCE_BUNDLE_DIRNAME)
    verification_report_path = join(bundle_dir, "verification-report.json")
    temporary = join(bundle_dir, f".verification-report.{secrets.token_hex(8)}.json")
    await workspace.write_bytes(temporary, b"")
    try:
        try:
            proc = await workspace.run(
                [
                    python,
                    workspace.host_path(script_path),
                    "verify",
                    "--root-dir",
                    workspace.host_path(paper_dir),
                    "--output",
                    workspace.host_path(temporary),
                ],
                cwd=paper_dir,
                timeout=timeout,
            )
        except OSError as exc:
            return None, f"scientific-evidence verifier could not run: {exc}"
        stdout_tail = (proc.stdout or "")[-1000:]
        stderr_tail = (proc.stderr or "")[-1000:]
        if proc.returncode not in {0, 1}:
            return None, (
                "scientific-evidence verifier failed "
                f"(rc={proc.returncode}). stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}"
            )
        try:
            payload = json.loads((await workspace.read_bytes(temporary)).decode("utf-8"))
        except (FileNotFoundError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return None, f"scientific-evidence verifier produced an invalid report: {exc}"
        if not _is_usable_verification_payload(
            payload, expected_root=workspace.host_path(paper_dir)
        ):
            return None, "scientific-evidence verifier produced a schema-invalid report"
        summary = payload["summary"]
        has_dangling = summary["dangling_references"] > 0
        if (proc.returncode == 1) != has_dangling:
            return None, (
                "scientific-evidence verifier exit status disagrees with dangling reference count"
            )
        await workspace.write_bytes(
            verification_report_path,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        return payload, None
    finally:
        await workspace.remove(temporary)


def evidence_report_skeleton(paper_dir_label: str, bundle_present: bool) -> dict[str, Any]:
    """Return the empty ``evidence_validation`` report of upstream."""
    return {
        "checked": False,
        "bundle_path": f"{paper_dir_label.rstrip('/')}/{EVIDENCE_BUNDLE_DIRNAME}",
        "bundle_present": bundle_present,
        "verification_report_path": None,
        "summary": {
            "total_references": 0,
            "resolved_references": 0,
            "anchored_references": 0,
            "dangling_references": 0,
            "manifests_checked": 0,
        },
        "dangling": [],
        "paper_token_count": 0,
        "hard_errors": [],
    }


async def validate_evidence_bundle(
    workspace: Workspace,
    paper_dir: str,
    tree: PaperTree,
    *,
    python: str,
    prompts: PromptRegistry | None = None,
    timeout: float | None = 600,
) -> dict[str, Any]:
    """Trace every `@evidence[[...]]` link in the scientific-evidence bundle.

    Delegates to the canonical scientific-evidence skill verifier so the skill stays the single
    source of truth for classification semantics (`resolved`, `anchored`, `dangling`). Returns a
    structured report for ``report['evidence_validation']``.
    """
    bundle_kind = tree.kind(EVIDENCE_BUNDLE_DIRNAME)
    paper_label = workspace.host_path(paper_dir)
    bundle_dir_label = f"{paper_label.rstrip('/')}/{EVIDENCE_BUNDLE_DIRNAME}"
    tokens_in_paper = scan_paper_for_evidence_tokens(tree)
    report = evidence_report_skeleton(paper_label, bundle_kind == "dir")
    report["paper_token_count"] = len(tokens_in_paper)

    if bundle_kind == "symlink":
        report["hard_errors"].append(
            f"scientific-evidence bundle must be a real directory: {bundle_dir_label}"
        )
        return report
    if bundle_kind != "dir":
        if tokens_in_paper:
            first = tokens_in_paper[0]
            report["hard_errors"].append(
                f"scientific-evidence/ bundle missing under {paper_label} but paper text carries "
                f"{len(tokens_in_paper)} `@evidence[[...]]` token(s); first occurrence: "
                f"{first['file']}:{first['line']} {first['token']}. "
                "Generate the bundle with `scientific_evidence.sh emit-bundle --root-dir <paper_dir>` "
                "or remove the evidence tokens."
            )
        return report

    verification_report_path = f"{bundle_dir_label}/verification-report.json"
    report["checked"] = True
    report["verification_report_path"] = verification_report_path
    payload, verification_error = await run_evidence_verifier(
        workspace, paper_dir, python=python, prompts=prompts, timeout=timeout
    )
    if verification_error is not None or payload is None:
        report["hard_errors"].append(verification_error or "evidence verifier failed")
        return report

    summary = payload.get("summary")
    if isinstance(summary, dict):
        for key in report["summary"]:
            value = summary.get(key)
            if isinstance(value, int):
                report["summary"][key] = value

    raw_results = payload.get("results")
    results: Sequence[Any] = raw_results if isinstance(raw_results, list) else []
    dangling: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("status") != "dangling":
            continue
        messages = item.get("messages")
        if not isinstance(messages, list):
            messages = []
        dangling.append(
            {
                "subject_path": item.get("subject_path"),
                "subject_kind": item.get("subject_kind"),
                "raw_reference": item.get("raw_reference"),
                "target": item.get("target"),
                "locator": item.get("locator"),
                "messages": [str(message) for message in messages],
            }
        )
    report["dangling"] = dangling

    if dangling:
        preview_limit = 5
        parts = [
            f"scientific-evidence verifier found {len(dangling)} dangling `@evidence[[...]]` link(s).",
        ]
        for item in dangling[:preview_limit]:
            subject = item.get("subject_path") or "<unknown-subject>"
            token = item.get("raw_reference") or item.get("target") or "<unknown-token>"
            detail = "; ".join(item.get("messages") or []) or "dangling link"
            parts.append(f"{subject} {token} -> {detail}")
        if len(dangling) > preview_limit:
            parts.append(f"(+{len(dangling) - preview_limit} more; see {verification_report_path})")
        parts.append(
            "Fix or remove every dangling link, rerun "
            "`scientific_evidence.sh verify --root-dir <paper_dir>`, and repeat "
            "until zero dangling references remain."
        )
        report["hard_errors"].append(" ".join(parts))

    return report
