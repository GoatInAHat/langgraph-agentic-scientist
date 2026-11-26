"""Fingerprints of the validation inputs and of a validation failure.

Upstream ``validation_input_sha256`` (``paper_evidence_audit.py``) and
``validation_failure_fingerprint`` (``paper_step.py``). A repeated failure with unchanged inputs
ends the repair loop: the fingerprint binds the normalised hard errors to the hashes of the
files the validators read (the manuscript without its managed watermark block, the claims, data
and reference files, the TeX sources it imports, the bibliography styles it names, and the
figure files and prompt logs of the manifest).
"""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Sequence
from itertools import chain
from pathlib import PurePosixPath
from typing import Any

from ..watermark import strip_managed_block
from .files import PaperTree, is_regular_nofollow
from .graphics import (
    GRAPHICS_SUFFIXES,
    candidate_graphics_paths,
    iter_graphics_paths,
    iter_graphics_search_path_declarations,
    iter_tex_import_paths,
    iter_tex_support_paths,
)
from .supplement import mask_false_conditionals, mask_tex_verbatim, strip_tex_comments
from .tex import AGENT_SKILL_DIR_PREFIXES

AUDIT_INPUT_PATHS = (
    "paper.tex",
    "claims_grounding.json",
    "paper_data.json",
    "references.bib",
    "references_provenance.json",
    "figure_manifest.json",
    "scientific-evidence/verification-report.json",
)
"""The inputs of the evidence audit (upstream ``AUDIT_INPUT_PATHS``)."""

VALIDATION_INPUT_PATHS = (
    *AUDIT_INPUT_PATHS,
    "VENUE.md",
    "paper_authors.tex",
    "paper_strategy.md",
    "paper_strategy.json",
    "method_scope_audit.md",
    "method_scope_audit.json",
)
"""The canonical inputs of the validation gate (upstream ``VALIDATION_INPUT_PATHS``)."""

_MAX_AUDIT_TEX_BYTES = 16 * 1024 * 1024
_MAX_VALIDATION_METADATA_BYTES = 16 * 1024 * 1024
_MAX_VALIDATION_METADATA_TOTAL_BYTES = 64 * 1024 * 1024
_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_COUNT = 16
_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_BYTES = 16 * 1024 * 1024
_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_TARGET_CHARS = 1024
_MAX_AUDIT_TEX_SOURCE_COUNT = 256
_MAX_AUDIT_FIGURE_COUNT = 256
_MAX_AUDIT_GRAPHICS_PATH_COUNT = 32
_MAX_AUDIT_GRAPHICS_PATH_CHARS = 1024
_MAX_AUDIT_FIGURE_REFERENCE_CHARS = 1024
_MAX_AUDIT_FIGURE_CANDIDATE_COUNT = 16 * 1024
_MAX_AUDIT_FIGURE_CANDIDATE_CHARS = 16 * 1024 * 1024
_MAX_AUDIT_FIGURE_BYTES = 512 * 1024 * 1024
_MAX_VALIDATION_TEX_CONDITIONAL_TOKENS = 1024
_TEX_CONDITIONAL_TOKEN_RE = re.compile(r"\\(?:if[A-Za-z@]*|else|fi)\b")
_BIBLIOGRAPHY_STYLE_RE = re.compile(r"\\bibliographystyle(?![A-Za-z@])\s*\{(?P<target>[^{}]+)\}")
_VOLATILE_VERIFICATION_FIELDS = {"root_dir", "verified_at"}
_AGENT_NAMESPACE_NAMES = {prefix[0] for prefix in AGENT_SKILL_DIR_PREFIXES}
_AUDIT_FIGURE_SUFFIXES = set(GRAPHICS_SUFFIXES)


def _read_bounded(tree: PaperTree, rel: str, *, byte_limit: int, description: str) -> bytes | None:
    if not is_regular_nofollow(tree, rel):
        return None
    data = tree.read(rel)
    if data is None:
        return None
    if len(data) > byte_limit:
        raise ValueError(f"{description} exceeds {byte_limit} bytes")
    return data


def _validated_graphics_search_path(raw_path: str) -> PurePosixPath:
    if len(raw_path) > _MAX_AUDIT_GRAPHICS_PATH_CHARS:
        raise ValueError("Paper graphics search path exceeds the audit path limit")
    path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or not path.parts
        or path.parts[0] != "figures"
        or not raw_path.endswith("/")
        or "#" in raw_path
        or any(part in {"", ".", ".."} or "\\" in part for part in path.parts)
    ):
        raise ValueError(f"Unsafe graphicspath in paper.tex: {raw_path!r}")
    return path


def _validated_figure_candidate(candidate: str) -> PurePosixPath:
    path = PurePosixPath(candidate)
    if (
        path.is_absolute()
        or not path.parts
        or "#" in candidate
        or any(part in {"", ".", ".."} or "\\" in part for part in path.parts)
        or path.suffix.casefold() not in _AUDIT_FIGURE_SUFFIXES
    ):
        raise ValueError(f"Unsafe figure reference in paper.tex: {candidate!r}")
    return path


def _paper_tex_sources(tree: PaperTree) -> list[tuple[PurePosixPath, bytes]]:
    pending = [PurePosixPath("paper.tex")]
    queued = set(pending)
    visited: set[PurePosixPath] = set()
    sources: list[tuple[PurePosixPath, bytes]] = []
    total_bytes = 0
    conditional_tokens = 0
    while pending:
        relative_path = pending.pop()
        if relative_path in visited:
            continue
        visited.add(relative_path)
        data = _read_bounded(
            tree,
            relative_path.as_posix(),
            byte_limit=_MAX_AUDIT_TEX_BYTES - total_bytes,
            description=f"TeX source {relative_path}",
        )
        if data is None:
            continue
        total_bytes += len(data)
        text = data.decode("utf-8")
        visible_conditionals = mask_tex_verbatim(strip_tex_comments(text))
        conditional_tokens += len(_TEX_CONDITIONAL_TOKEN_RE.findall(visible_conditionals))
        if conditional_tokens > _MAX_VALIDATION_TEX_CONDITIONAL_TOKENS:
            raise ValueError("Paper TeX exceeds the conditional parsing limit")
        sources.append((relative_path, data))
        dependency_targets = chain(iter_tex_import_paths(text), iter_tex_support_paths(text))
        for raw_target in dependency_targets:
            if "\\" in raw_target or "#" in raw_target:
                raise ValueError(f"Dynamic TeX import target: {raw_target!r}")
            target = PurePosixPath(raw_target)
            if target.is_absolute():
                raise ValueError(f"TeX import escapes the paper workspace: {raw_target!r}")
            for base in (PurePosixPath(), relative_path.parent):
                candidate = posixpath.normpath((base / target).as_posix())
                if candidate.startswith("/") or candidate == ".." or candidate.startswith("../"):
                    raise ValueError(f"TeX import escapes the paper workspace: {raw_target!r}")
                candidate_relative = PurePosixPath(candidate)
                if (
                    candidate_relative.parts
                    and candidate_relative.parts[0] in _AGENT_NAMESPACE_NAMES
                ):
                    raise ValueError(f"Paper must not import agent files: {raw_target!r}")
                variants = [candidate_relative]
                if not candidate_relative.suffix:
                    variants.append(candidate_relative.with_suffix(".tex"))
                for variant in variants:
                    if variant in queued:
                        continue
                    queued.add(variant)
                    if len(queued) > _MAX_AUDIT_TEX_SOURCE_COUNT:
                        raise ValueError(
                            f"Paper imports more than {_MAX_AUDIT_TEX_SOURCE_COUNT} TeX sources"
                        )
                    pending.append(variant)
    return sources


def _referenced_bibliography_style_paths(
    tree: PaperTree, tex_sources: Sequence[tuple[PurePosixPath, bytes]]
) -> list[PurePosixPath]:
    styles: set[PurePosixPath] = set()
    candidates: set[PurePosixPath] = set()
    for source_path, source_bytes in tex_sources:
        if source_path.suffix.casefold() != ".tex":
            continue
        text = mask_false_conditionals(
            mask_tex_verbatim(strip_tex_comments(source_bytes.decode("utf-8")))
        )
        for match in _BIBLIOGRAPHY_STYLE_RE.finditer(text):
            raw_target = match.group("target").strip()
            if len(raw_target) > _MAX_VALIDATION_BIBLIOGRAPHY_STYLE_TARGET_CHARS:
                raise ValueError("Bibliography style target is too long")
            if "\\" in raw_target or "#" in raw_target:
                raise ValueError(f"Dynamic bibliography style target: {raw_target!r}")
            target = PurePosixPath(raw_target)
            if target.is_absolute() or any(part in {"", ".", ".."} for part in target.parts):
                raise ValueError(f"Unsafe bibliography style target: {raw_target!r}")
            if not target.suffix:
                target = target.with_suffix(".bst")
            candidates.add(target)
            if len(candidates) > _MAX_VALIDATION_BIBLIOGRAPHY_STYLE_COUNT:
                raise ValueError(
                    "Paper references more than "
                    f"{_MAX_VALIDATION_BIBLIOGRAPHY_STYLE_COUNT} bibliography styles"
                )
            if is_regular_nofollow(tree, target.as_posix()):
                styles.add(target)
    return sorted(styles, key=lambda path: path.as_posix())


def _referenced_figure_paths(
    tree: PaperTree, tex_sources: Sequence[tuple[PurePosixPath, bytes]]
) -> list[PurePosixPath]:
    search_paths: tuple[PurePosixPath, ...] | None = None
    for _, source_bytes in tex_sources:
        tex_text = source_bytes.decode("utf-8")
        for raw_declaration in iter_graphics_search_path_declarations(tex_text):
            if len(raw_declaration) > _MAX_AUDIT_GRAPHICS_PATH_COUNT:
                raise ValueError(
                    f"Paper declares more than {_MAX_AUDIT_GRAPHICS_PATH_COUNT} graphics search paths"
                )
            declaration = tuple(
                _validated_graphics_search_path(raw_path) for raw_path in raw_declaration
            )
            if search_paths is not None and declaration != search_paths:
                raise ValueError("Paper declares conflicting graphics search paths")
            search_paths = declaration

    candidates: dict[str, bool] = {}
    reference_count = 0
    candidate_count = 0
    candidate_chars = 0

    def consume_candidate_budget(candidate: str) -> None:
        nonlocal candidate_count, candidate_chars
        candidate_count += 1
        candidate_chars += len(candidate)
        if (
            candidate_count > _MAX_AUDIT_FIGURE_CANDIDATE_COUNT
            or candidate_chars > _MAX_AUDIT_FIGURE_CANDIDATE_CHARS
        ):
            raise ValueError("Paper figure references exceed the audit expansion limit")

    def add_candidate(candidate: str, *, support_file: bool) -> None:
        candidates[candidate] = candidates.get(candidate, True) and support_file

    for source_path, source_bytes in tex_sources:
        tex_text = source_bytes.decode("utf-8")
        support_file = source_path.suffix.casefold() in {".cls", ".sty"}
        for raw_path in iter_graphics_paths(tex_text):
            reference_count += 1
            if reference_count > _MAX_AUDIT_FIGURE_COUNT:
                raise ValueError(
                    f"Paper references more than {_MAX_AUDIT_FIGURE_COUNT} figure assets"
                )
            if len(raw_path) > _MAX_AUDIT_FIGURE_REFERENCE_CHARS:
                raise ValueError("Paper figure reference exceeds the audit path limit")
            if ("#" in raw_path or "\\" in raw_path) and support_file:
                continue
            for candidate in sorted(candidate_graphics_paths(raw_path)):
                candidate_path = _validated_figure_candidate(candidate)
                consume_candidate_budget(candidate)
                if candidate_path.parts and candidate_path.parts[0] == "figures":
                    add_candidate(candidate, support_file=support_file)
                    continue
                if search_paths:
                    if is_regular_nofollow(tree, candidate_path.as_posix()):
                        raise ValueError(
                            "Unsafe figure reference shadows graphicspath asset in "
                            f"paper root: {candidate!r}"
                        )
                    for search_path in search_paths:
                        resolved_candidate = (search_path / candidate_path).as_posix()
                        consume_candidate_budget(resolved_candidate)
                        resolved_path = _validated_figure_candidate(resolved_candidate)
                        if is_regular_nofollow(tree, resolved_path.as_posix()):
                            add_candidate(resolved_candidate, support_file=support_file)
                            break
                    continue
                add_candidate(candidate, support_file=support_file)
    paths: list[PurePosixPath] = []
    for candidate, support_file in sorted(candidates.items()):
        path = _validated_figure_candidate(candidate)
        if (len(path.parts) < 2 or path.parts[0] != "figures") and not support_file:
            raise ValueError(f"Unsafe figure reference in paper.tex: {candidate!r}")
        if support_file and not is_regular_nofollow(tree, path.as_posix()):
            continue
        paths.append(path)
    return paths


def _json_object_from_bytes(data: bytes, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def _manifest_validator_input_paths(
    tree: PaperTree,
) -> tuple[set[PurePosixPath], list[PurePosixPath]]:
    manifest_data = _read_bounded(
        tree,
        "figure_manifest.json",
        byte_limit=_MAX_VALIDATION_METADATA_BYTES,
        description="validation input figure_manifest.json",
    )
    if manifest_data is None:
        return set(), []
    payload = _json_object_from_bytes(manifest_data, "figure_manifest.json")
    rows = payload.get("figures")
    if not isinstance(rows, list):
        return set(), []
    if len(rows) > _MAX_AUDIT_FIGURE_COUNT:
        raise ValueError(f"Figure manifest contains more than {_MAX_AUDIT_FIGURE_COUNT} figures")

    figure_paths: set[PurePosixPath] = set()
    prompt_logs: set[PurePosixPath] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        files = row.get("files")
        nested_png = files.get("png") if isinstance(files, dict) else None
        nested_pdf = files.get("pdf") if isinstance(files, dict) else None
        raw_paths = (row.get("path_png", nested_png), row.get("path_pdf", nested_pdf))
        for raw_path_value in raw_paths:
            raw_path = raw_path_value.strip() if isinstance(raw_path_value, str) else ""
            if not raw_path:
                continue
            path = _validated_figure_candidate(raw_path)
            if len(path.parts) < 2 or path.parts[0] != "figures":
                continue
            figure_paths.add(path)
        prompt_log_value = row.get("prompt_log_path")
        figure_id_value = row.get("figure_id")
        if isinstance(prompt_log_value, str) and isinstance(figure_id_value, str):
            raw_prompt_path = prompt_log_value.strip()
            expected = f"figures/{figure_id_value}.prompts.json"
            if raw_prompt_path == expected:
                prompt_logs.add(PurePosixPath(raw_prompt_path))
    return figure_paths, sorted(prompt_logs, key=lambda path: path.as_posix())


def _file_hex(tree: PaperTree, rel: str) -> str | None:
    digest = tree.sha256(rel)
    if digest is None:
        data = tree.read(rel)
        return None if data is None else hashlib.sha256(data).hexdigest()
    return digest.removeprefix("sha256:")


def validation_input_sha256(tree: PaperTree) -> dict[str, str]:
    """Hash the canonical validation inputs without unrelated workspace churn."""
    return paper_input_sha256(tree, VALIDATION_INPUT_PATHS)


def paper_input_sha256(tree: PaperTree, input_paths: Sequence[str]) -> dict[str, str]:
    """Hash ``input_paths`` with the TeX sources, styles, figures and prompt logs they reference.

    Upstream ``_paper_input_sha256``.
    """
    hashes: dict[str, str] = {}
    remaining_metadata_bytes = _MAX_VALIDATION_METADATA_TOTAL_BYTES
    for relative_path in input_paths:
        per_file_limit = (
            _MAX_AUDIT_TEX_BYTES if relative_path == "paper.tex" else _MAX_VALIDATION_METADATA_BYTES
        )
        data = _read_bounded(
            tree,
            relative_path,
            byte_limit=min(per_file_limit, remaining_metadata_bytes),
            description=f"validation input {relative_path}",
        )
        if data is None:
            continue
        remaining_metadata_bytes -= len(data)
        if relative_path == "paper.tex":
            # The managed watermark block is regenerated before every compile and is not
            # manuscript content; keep hashes independent of it.
            data = strip_managed_block(data.decode("utf-8")).encode("utf-8")
        if relative_path == "scientific-evidence/verification-report.json":
            payload = json.loads(data.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object in {relative_path}")
            stable_payload = {
                key: value
                for key, value in payload.items()
                if key not in _VOLATILE_VERIFICATION_FIELDS
            }
            data = json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode()
        hashes[relative_path] = hashlib.sha256(data).hexdigest()

    tex_sources = _paper_tex_sources(tree)
    for tex_relative_path, source_bytes in tex_sources:
        if tex_relative_path != PurePosixPath("paper.tex"):
            hashes[f"tex-source/{tex_relative_path.as_posix()}"] = hashlib.sha256(
                source_bytes
            ).hexdigest()
    remaining_style_bytes = _MAX_VALIDATION_BIBLIOGRAPHY_STYLE_BYTES
    for style_path in _referenced_bibliography_style_paths(tree, tex_sources):
        style_data = _read_bounded(
            tree,
            style_path.as_posix(),
            byte_limit=min(_MAX_VALIDATION_METADATA_BYTES, remaining_style_bytes),
            description=f"bibliography style {style_path}",
        )
        if style_data is not None:
            remaining_style_bytes -= len(style_data)
            hashes[f"compile-input/{style_path.as_posix()}"] = hashlib.sha256(
                style_data
            ).hexdigest()

    figure_bytes = 0
    manifest_figures, prompt_logs = _manifest_validator_input_paths(tree)
    figure_paths = set(_referenced_figure_paths(tree, tex_sources))
    figure_paths.update(manifest_figures)
    for figure_path in sorted(figure_paths, key=lambda path: path.as_posix()):
        if not is_regular_nofollow(tree, figure_path.as_posix()):
            continue
        entry_size = getattr(tree, "size", lambda _rel: None)(figure_path.as_posix())
        if entry_size is not None:
            figure_bytes += int(entry_size)
            if figure_bytes > _MAX_AUDIT_FIGURE_BYTES:
                raise ValueError(f"Referenced figure assets exceed {_MAX_AUDIT_FIGURE_BYTES} bytes")
        digest = _file_hex(tree, figure_path.as_posix())
        if digest is not None:
            hashes[f"figure-asset/{figure_path.as_posix()}"] = digest
    for prompt_log_path in prompt_logs:
        prompt_data = _read_bounded(
            tree,
            prompt_log_path.as_posix(),
            byte_limit=min(_MAX_VALIDATION_METADATA_BYTES, remaining_metadata_bytes),
            description=f"concept figure prompt log {prompt_log_path}",
        )
        if prompt_data is not None:
            remaining_metadata_bytes -= len(prompt_data)
            hashes[f"validator-input/{prompt_log_path.as_posix()}"] = hashlib.sha256(
                prompt_data
            ).hexdigest()
    return hashes


def validation_hard_error_fingerprint(hard_errors: Sequence[object]) -> str | None:
    """Return a stable fingerprint for a validation error set (``None`` when empty)."""
    if not hard_errors or any(not isinstance(error, str) for error in hard_errors):
        return None
    normalized_errors = sorted(
        {
            normalized
            for error in hard_errors
            if (normalized := re.sub(r"\s+", " ", str(error)).strip())
        }
    )
    if not normalized_errors:
        return None
    serialized = json.dumps(normalized_errors, ensure_ascii=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(serialized).hexdigest()


def validation_failure_fingerprint(hard_errors: Sequence[object], tree: PaperTree) -> str | None:
    """Fingerprint validation errors together with their current paper inputs."""
    error_fingerprint = validation_hard_error_fingerprint(hard_errors)
    if error_fingerprint is None:
        return None
    try:
        input_hashes = validation_input_sha256(tree)
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise RuntimeError("Could not fingerprint paper validation inputs") from exc
    if not input_hashes:
        return None
    serialized = json.dumps(
        {"hard_errors": error_fingerprint, "paper_inputs": input_hashes},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


__all__ = [
    "AUDIT_INPUT_PATHS",
    "VALIDATION_INPUT_PATHS",
    "paper_input_sha256",
    "validation_failure_fingerprint",
    "validation_hard_error_fingerprint",
    "validation_input_sha256",
]
