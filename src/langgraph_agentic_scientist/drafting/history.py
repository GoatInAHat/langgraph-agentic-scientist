"""Retained history for repair prompts, and the sanitisation that every diagnostic passes through.

Two histories are kept as records in graph state and rendered into the prompts of later repairs,
with upstream's bounds (``task_drafting/alignment_history.py`` and ``repair_context.py``):

* the **alignment history**: the failed dimensions of every validated alignment report (at most the
  last 16 reports; three evidence items per finding, 512 bytes per detail, 1024 per remediation),
  rendered after prompt expansion into the judge prompt and the repair prompts. It is evidence for
  re-checking the current tree, never acceptance authority;
* the **rejected-repair history**: a bounded source-only diff of every rejected qualification
  repair (64 KiB), with the reports that rejected it (4 KiB each in the prompt), rendered into the
  repair prompts only, never into the judge's.

The module also holds the sanitisation of diagnostics (:func:`redact_sensitive_text` and
:func:`contains_sensitive_output`, upstream's ``checkpoint_security`` and ``runtime_output``
helpers): a diagnostic that contains a recognisable credential is replaced by a digest, and every
non-printable character, a newline included, is shown as an escape, exactly as upstream shows it.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

from ..core.render import json_text
from ..workspace.paths import join
from ..workspace.protocol import Workspace
from .models import (
    REQUIRED_ALIGNMENT_DIMENSIONS,
    AlignmentFinding,
    AlignmentHistoryEntry,
    AlignmentReport,
    RejectedRepair,
)

MAX_PROMPT_BYTES = 64 * 1024
MAX_ALIGNMENT_REPORTS = 16
MAX_REPAIR_RECORDS = 8
MAX_DIFF_BYTES = 64 * 1024
MAX_RECORD_BYTES = 192 * 1024
MAX_ALIGNMENT_RECORD_BYTES = 64 * 1024
_MAX_FILE_BYTES = 512 * 1024
_MAX_SOURCE_BYTES = 4 * 1024 * 1024
_EXCLUDED = {"resources", "data", "datasets", "artifacts", "results", "cache", "__pycache__"}
_FINGERPRINT_RE = re.compile(r"sha256:[a-f0-9]{64}")

# --- credential recognition (upstream checkpoint_security.py and runtime_output.py) ------------

_UNIVERSAL_SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(
        rb"\b(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{16,}|sk_live_[A-Za-z0-9]{16,}|"
        rb"gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        rb"glpat-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|"
        rb"xox[baprs]-[A-Za-z0-9-]{20,}|AIza[0-9A-Za-z_-]{30,})\b"
    ),
    re.compile(rb"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
)
_ASSIGNMENT_RE = re.compile(
    rb"(?i)(?:api[_-]?key|api[_-]?token|auth[_-]?token|access[_-]?token|"
    rb"refresh[_-]?token|client[_-]?secret|secret[_-]?key|aws[_-]?secret[_-]?access[_-]?key|"
    rb"password)"
    rb"[\"']?\s*[:=]\s*(?:[\"'](?P<quoted>[^\"'\r\n]{12,})[\"']|"
    rb"(?P<bare>[^\s,#;]{12,}))"
)
_PLACEHOLDER_VALUES = frozenset(
    {
        b"a_secure_password",
        b"changeme",
        b"dummy",
        b"example",
        b"placeholder_value",
        b"redacted",
        b"your-api-key-here",
    }
)
_SOURCE_REFERENCE_RE = re.compile(
    rb"(?i)^(?:args|cfg|config|response|self|settings)\.[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$"
)
_SOURCE_CALL_RE = re.compile(rb"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\(.*\)$")
_TYPE_EXPRESSION_RE = re.compile(rb"^[A-Za-z_]\w*\[[^\]]+\]$")
_TEMPLATE_EXPRESSION_RE = re.compile(
    rb"^(?:\$[A-Z_][A-Z0-9_]*|\$\{[A-Z_][A-Z0-9_]*\}|\{\{[^{}]+\}\})$"
)
_TRANSPORT_CREDENTIAL = re.compile(
    r"(?i)(?:\bauthorization\s*:\s*(?:bearer|basic)\s+\S+|"
    r"\b[a-z][a-z0-9+.-]*://[^\s/@]+@)"
)
_SENSITIVE_ENV_NAME = re.compile(r"(?i)AUTH|CREDENTIAL|KEY|PASSWORD|PROXY|SECRET|TOKEN")
_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|api[_-]?token|auth(?:orization)?|credentials?|"
    r"password|secret|token|access[_-]?token|client[_-]?secret)"
    r"[\"']?\s*[:=]\s*\S+"
)
_MIN_ENV_SECRET_CHARS = 8
"""Shortest environment value treated as a secret to look for in a text. Upstream looks for any
non-empty value; a value such as ``1`` or ``true`` (flags often live in variables whose names
contain AUTH or KEY) occurs in every text and would hide every diagnostic, so shorter values are
not searched for."""
_SENSITIVE_DIRECTORY_NAMES = frozenset(
    {".agent_home", ".aws", ".azure", ".docker", ".gnupg", ".kaggle", ".kube", ".ssh"}
)
_SENSITIVE_PATH_SUFFIXES = (
    (".cache", "huggingface"),
    (".config", "gcloud"),
    (".config", "gh"),
    (".config", "huggingface"),
)
_SENSITIVE_FILE_NAMES = frozenset(
    {
        ".git-credentials",
        ".netrc",
        ".npmrc",
        ".pypirc",
        "application_default_credentials.json",
        "auth.json",
        "credentials",
        "credentials.json",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
        "kaggle.json",
        "service-account.json",
        "token",
        "token.json",
    }
)
_SENSITIVE_FILE_SUFFIXES = frozenset({".key", ".p12", ".pfx"})
_SAFE_ENV_TEMPLATE_SUFFIXES = (".example", ".sample", ".template")


def _contains_literal_assignment(content: bytes) -> bool:
    for match in _ASSIGNMENT_RE.finditer(content):
        value = match.group("quoted") or match.group("bare")
        if value.lower() in _PLACEHOLDER_VALUES:
            continue
        if _TEMPLATE_EXPRESSION_RE.fullmatch(value):
            continue
        if any(
            pattern.fullmatch(value)
            for pattern in (_SOURCE_REFERENCE_RE, _SOURCE_CALL_RE, _TYPE_EXPRESSION_RE)
        ):
            continue
        return True
    return False


def contains_secret(content: bytes) -> bool:
    """Return whether ``content`` holds a recognisable credential (upstream ``contains_secret``)."""
    return any(
        pattern.search(content) for pattern in _UNIVERSAL_SECRET_PATTERNS
    ) or _contains_literal_assignment(content)


def is_sensitive_path(path: str) -> bool:
    """Return whether a relative path names a conventional credential location."""
    relative = PurePosixPath(path)
    lowered = tuple(part.lower() for part in relative.parts)
    if any(part in _SENSITIVE_DIRECTORY_NAMES for part in lowered):
        return True
    if any(
        lowered[index : index + len(suffix)] == suffix
        for suffix in _SENSITIVE_PATH_SUFFIXES
        for index in range(len(lowered) - len(suffix) + 1)
    ):
        return True
    name = relative.name.lower()
    if name in _SENSITIVE_FILE_NAMES or relative.suffix.lower() in _SENSITIVE_FILE_SUFFIXES:
        return True
    return name == ".env" or (
        name.startswith(".env.") and not name.endswith(_SAFE_ENV_TEMPLATE_SUFFIXES)
    )


def _display_character(character: str) -> str:
    codepoint = ord(character)
    if character.isprintable():
        return character
    if codepoint <= 0xFF:
        return f"\\x{codepoint:02x}"
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    return f"\\U{codepoint:08x}"


def redact_sensitive_text(value: str) -> str:
    """Replace a diagnostic that contains a credential by a digest; escape non-printables.

    Every character that is not printable, a newline included, is shown as an escape sequence
    (``\\x0a``): upstream's redaction does so, and the diagnostics that reach a repair prompt
    are therefore single lines.
    """
    encoded = value.encode("utf-8", errors="surrogateescape")
    if not contains_secret(encoded):
        return "".join(_display_character(character) for character in value)
    digest = hashlib.sha256(encoded).hexdigest()[:12]
    return f"<redacted-sensitive-text:{digest}>"


def contains_sensitive_output(text: str, environment: Mapping[str, str] | None = None) -> bool:
    """Detect diagnostic credentials, including short literals and credential URLs."""
    env = os.environ if environment is None else environment
    return bool(
        _TRANSPORT_CREDENTIAL.search(text)
        or _CREDENTIAL_ASSIGNMENT.search(text)
        or any(
            len(secret) >= _MIN_ENV_SECRET_CHARS
            and _SENSITIVE_ENV_NAME.search(name)
            and secret in text
            for name, secret in env.items()
        )
        or contains_secret(text.encode("utf-8", errors="replace"))
    )


def clip_utf8(text: str, size: int) -> str:
    """Cut ``text`` to ``size`` UTF-8 bytes on a character boundary."""
    return text.encode("utf-8")[:size].decode("utf-8", errors="ignore")


def safe_text(text: str) -> str:
    """Replace ``text`` by a marker when it contains sensitive output."""
    if contains_sensitive_output(text):
        return "<redacted-sensitive-text>"
    return text


def bounded_text(value: object, limit: int) -> str:
    """Return ``value`` as text, redacted and cut to ``limit`` bytes (upstream ``_text``)."""
    return clip_utf8(redact_sensitive_text(safe_text(str(value))), limit)


def has_transport_credential(text: str) -> bool:
    """Return whether ``text`` carries an authorization header or a credential URL."""
    return bool(_TRANSPORT_CREDENTIAL.search(text))


def fingerprint_or_none(value: object) -> str | None:
    """Return ``value`` when it is a ``sha256:<hex>`` fingerprint, else ``None``."""
    return value if isinstance(value, str) and _FINGERPRINT_RE.fullmatch(value) else None


# --- alignment history ---------------------------------------------------------------------


def _finding(dimension_id: str, evidence: Sequence[object], remediation: str) -> AlignmentFinding:
    items = list(evidence)
    truncated = len(items) > 3 or len(remediation.encode()) > 1024
    entries: list[dict[str, object]] = []
    for item in items[:3]:
        path = str(getattr(item, "path", ""))
        detail = str(getattr(item, "detail", ""))
        line = getattr(item, "line", None)
        truncated |= len(path.encode()) > 160 or len(detail.encode()) > 512
        entries.append(
            {
                "path": bounded_text(path, 160),
                "detail": bounded_text(detail, 512),
                "line": line if type(line) is int else None,
            }
        )
    return AlignmentFinding(
        id=dimension_id,
        remediation=bounded_text(remediation, 1024),
        evidence=entries,
        truncated=truncated,
    )


def capture_alignment_report(
    report: AlignmentReport, *, task_fingerprint: str, name: str
) -> AlignmentHistoryEntry:
    """Capture a validated alignment report for the history (upstream ``capture_alignment_report``).

    Args:
        report: The validated report.
        task_fingerprint: The tree fingerprint of the scaffold the judge reviewed.
        name: The report's name in the history (``alignment_<n>.json``).

    Raises:
        ValueError: When the report or the fingerprint is not valid history material.
    """
    if fingerprint_or_none(task_fingerprint) is None:
        raise ValueError("invalid reviewed task fingerprint")
    verdict = report.verdict.strip()
    if verdict not in ("pass", "fail"):
        raise ValueError("invalid historical verdict")
    declared = fingerprint_or_none(report.fingerprints.get("declared_scaffold"))
    if declared is None:
        raise ValueError("invalid historical declared fingerprint")
    if len(report.dimensions) > len(REQUIRED_ALIGNMENT_DIMENSIONS):
        raise ValueError("invalid alignment dimensions")
    findings: list[AlignmentFinding] = []
    seen: set[str] = set()
    for dimension in report.dimensions:
        identifier = dimension.id.strip()
        if identifier not in REQUIRED_ALIGNMENT_DIMENSIONS or identifier in seen:
            raise ValueError("invalid alignment dimension identifier")
        seen.add(identifier)
        dimension_verdict = dimension.verdict.strip()
        if dimension_verdict not in ("pass", "fail"):
            raise ValueError("invalid historical verdict")
        if dimension_verdict == "pass":
            continue
        findings.append(_finding(identifier, dimension.evidence, dimension.remediation))
    if (verdict == "fail") != bool(findings):
        raise ValueError("historical verdict disagrees with dimensions")
    report_bytes = json_text(report.model_dump(mode="json", by_alias=True)).encode("utf-8")
    entry = AlignmentHistoryEntry(
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        reviewed_task_fingerprint=task_fingerprint,
        declared_scaffold_fingerprint=declared,
        historical_verdict="pass" if verdict == "pass" else "fail",
        findings=findings,
        report_name=name,
    )
    if len(entry.model_dump_json().encode()) > MAX_ALIGNMENT_RECORD_BYTES:
        raise ValueError("alignment history record exceeds limit")
    return entry


def render_alignment_history(
    entries: Sequence[AlignmentHistoryEntry], *, current_fingerprint: str, heading: str
) -> str:
    """Render only prior independent findings; no source diffs or runtime failure excerpts.

    Args:
        entries: The retained entries, oldest first.
        current_fingerprint: The tree fingerprint of the scaffold being prompted about.
        heading: The text of ``workflow/drafting/alignment_history_heading``.

    Returns:
        The text to append after the prompt (empty without history), at most 64 KiB.
    """
    if not entries:
        return ""
    selected = list(entries)[-MAX_ALIGNMENT_REPORTS:]
    records: list[dict[str, Any]] = [
        {
            "report": entry.report_name,
            "report_sha256": entry.report_sha256,
            "reviewed_task_fingerprint": entry.reviewed_task_fingerprint,
            "declared_scaffold_fingerprint": entry.declared_scaffold_fingerprint,
            "historical_verdict": entry.historical_verdict,
            "findings": [
                {
                    "id": finding.id,
                    "verdict": "fail",
                    "remediation": finding.remediation,
                    "evidence": [dict(item) for item in finding.evidence],
                    "truncated": finding.truncated,
                }
                for finding in entry.findings
            ],
        }
        for entry in selected
    ]
    payload = {
        "current_task_fingerprint": fingerprint_or_none(current_fingerprint),
        "omitted_older_reports": max(0, len(entries) - MAX_ALIGNMENT_REPORTS),
        "alignment_history": records,
    }
    limit = 1024
    while True:
        text = heading + json.dumps(payload, ensure_ascii=False)
        if len(text.encode()) <= MAX_PROMPT_BYTES:
            return text
        limit //= 2
        for record in records:
            record["prompt_truncated"] = True
            for finding in record["findings"]:
                finding["remediation"] = clip_utf8(finding["remediation"], limit)
                if limit < 128:
                    finding["evidence"] = []
                for evidence in finding["evidence"]:
                    evidence["detail"] = clip_utf8(evidence["detail"], limit)


# --- rejected-repair history ---------------------------------------------------------------


async def collect_sources(workspace: Workspace, root: str) -> tuple[dict[str, str], int]:
    """Read the source files of a scaffold tree that a rejected-repair diff may cover.

    Python, shell and TOML files, ``task_contract.json`` and ``requirements.txt`` outside hidden
    and data directories, each up to 512 KiB and 4 MiB in all; unreadable, oversized or
    credential-like files are counted as omitted (upstream ``_source_files``).

    Returns:
        The files by relative path, and the number of omitted files.
    """
    files: dict[str, str] = {}
    omitted = 0
    remaining = _MAX_SOURCE_BYTES
    for entry in await workspace.list_tree(root):
        parts = entry.path.split("/")
        if entry.kind != "file":
            continue
        if any(part.startswith(".") or part in _EXCLUDED for part in parts[:-1]):
            continue
        name = parts[-1]
        if name.startswith(".") or is_sensitive_path(entry.path):
            continue
        if contains_sensitive_output(entry.path):
            omitted += 1
            continue
        if PurePosixPath(name).suffix not in {".py", ".sh", ".toml"} and name not in {
            "task_contract.json",
            "requirements.txt",
        }:
            continue
        limit = min(_MAX_FILE_BYTES, remaining)
        if entry.size > limit:
            omitted += 1
            continue
        try:
            content = await workspace.read_bytes(join(root, entry.path))
            remaining -= len(content)
            text = content.decode("utf-8")
            if b"\0" in content or contains_sensitive_output(text):
                raise ValueError("binary or sensitive source")
            files[entry.path] = text
        except (OSError, ValueError, UnicodeError):
            omitted += 1
    return files, omitted


def capture_rejected_repair(
    before: Mapping[str, str],
    after: Mapping[str, str],
    *,
    omitted_files: int,
    base_fingerprint: str,
    reports: Iterable[tuple[str, bytes | None]],
    rejected_fingerprint: str | None = None,
) -> RejectedRepair:
    """Capture provisional edits before they are restored (upstream ``capture_rejected_repair``).

    Args:
        before: The source files at the base snapshot (:func:`collect_sources`).
        after: The source files of the rejected tree.
        omitted_files: Files omitted from both collections.
        base_fingerprint: The tree fingerprint of the base snapshot.
        reports: Up to three ``(name, content)`` pairs of the reports that rejected the repair;
            ``None`` content marks an unavailable report.
        rejected_fingerprint: The tree fingerprint of the rejected scaffold.

    Raises:
        ValueError: When the base fingerprint is invalid or the record exceeds 192 KiB.
    """
    if fingerprint_or_none(base_fingerprint) is None:
        raise ValueError("invalid base fingerprint")
    fragments: list[str] = []
    remaining = MAX_DIFF_BYTES
    truncated = False
    for name in sorted(before.keys() | after.keys()):
        if before.get(name) == after.get(name):
            continue
        for line in difflib.unified_diff(
            before.get(name, "").splitlines(keepends=True),
            after.get(name, "").splitlines(keepends=True),
            fromfile="base/" + name,
            tofile="rejected/" + name,
        ):
            if len(line.encode()) > remaining:
                fragments.append(clip_utf8(line, remaining))
                truncated = True
                break
            fragments.append(line)
            remaining -= len(line.encode())
        if truncated:
            break
    evidence: list[dict[str, object]] = []
    for name, content in list(reports)[:3]:
        if content is None:
            evidence.append({"unavailable": True})
            continue
        text = redact_sensitive_text(safe_text(content.decode("utf-8", errors="replace")))
        evidence.append(
            {
                "name": redact_sensitive_text(safe_text(name)),
                "sha256": hashlib.sha256(content).hexdigest(),
                "text": clip_utf8(text, 16 * 1024),
                "truncated": len(text.encode()) > 16 * 1024,
            }
        )
    record = RejectedRepair(
        base_fingerprint=base_fingerprint,
        rejected_fingerprint=fingerprint_or_none(rejected_fingerprint),
        reports=evidence,
        diff=safe_text("".join(fragments)),
        truncated=truncated,
        omitted_files=omitted_files,
    )
    if len(record.model_dump_json().encode()) > MAX_RECORD_BYTES:
        raise ValueError("repair context record exceeds limit")
    return record


def render_repair_history(
    records: Sequence[RejectedRepair], *, current_fingerprint: str, heading: str
) -> str:
    """Render bounded inert JSON of the rejected repairs, for repair agents only.

    Args:
        records: The retained records, oldest first (the last eight are rendered).
        current_fingerprint: The tree fingerprint of the scaffold being prompted about.
        heading: The text of ``workflow/drafting/repair_history_heading``.

    Returns:
        The text to append after the prompt (empty without history), at most 64 KiB.
    """
    rendered: list[dict[str, Any]] = []
    for item in list(records)[-MAX_REPAIR_RECORDS:]:
        rendered.append(
            {
                "base_fingerprint": fingerprint_or_none(item.base_fingerprint),
                "rejected_fingerprint": fingerprint_or_none(item.rejected_fingerprint),
                "diff": clip_utf8(safe_text(item.diff), MAX_DIFF_BYTES),
                "truncated": item.truncated is True,
                "reports": [
                    {
                        "text": clip_utf8(
                            redact_sensitive_text(safe_text(str(report.get("text", "")))), 4096
                        ),
                        "sha256": str(report.get("sha256", ""))
                        if re.fullmatch(r"[a-f0-9]{64}", str(report.get("sha256", "")))
                        else None,
                    }
                    for report in item.reports[:3]
                ],
            }
        )
    if not rendered:
        return ""

    def payload_text() -> str:
        return json.dumps(
            {
                "current_task_fingerprint": fingerprint_or_none(current_fingerprint),
                "rejected_repairs": rendered,
            },
            ensure_ascii=True,
        )

    text = payload_text()
    while len((heading + text).encode()) > MAX_PROMPT_BYTES:
        has_diffs = any(record["diff"] for record in rendered)
        for record in rendered:
            record["diff"] = clip_utf8(
                str(record.get("diff", "")), len(str(record.get("diff", "")).encode()) // 2
            )
            if has_diffs:
                record["prompt_truncated"] = True
                continue
            record["reports"] = [
                {
                    "text": clip_utf8(
                        str(report.get("text", "")), len(str(report.get("text", "")).encode()) // 2
                    ),
                    "sha256": report.get("sha256"),
                }
                for report in record.get("reports", [])
            ]
            record["prompt_truncated"] = True
        text = payload_text()
    return heading + text


__all__ = [
    "MAX_ALIGNMENT_REPORTS",
    "MAX_DIFF_BYTES",
    "MAX_PROMPT_BYTES",
    "MAX_REPAIR_RECORDS",
    "bounded_text",
    "capture_alignment_report",
    "capture_rejected_repair",
    "clip_utf8",
    "collect_sources",
    "contains_secret",
    "contains_sensitive_output",
    "fingerprint_or_none",
    "has_transport_credential",
    "is_sensitive_path",
    "redact_sensitive_text",
    "render_alignment_history",
    "render_repair_history",
    "safe_text",
]
