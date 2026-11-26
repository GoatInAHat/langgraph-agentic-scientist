"""Figure-manifest validation: schema, readability budgets and the figure files.

Upstream ``validate_figure_manifest`` and its helpers (``paper_step.py``): the manifest is
normalised from either supported shape, every row is checked against the strategy's figure specs,
the readability budgets, the PNG and PDF files and the grounding ids, and ``paper.tex``'s graphics
references are matched to the manifest. Issues are split into hard errors and soft readability
warnings. The file system is read through a
:class:`~langgraph_agentic_scientist.paper.validation.files.PaperTree`, so the same function
serves the host-side gate and the ``validate-paper`` skill's script (which passes a directory).
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .figure_manifest import (
    SUPPORTED_FIGURE_GENERATORS,
    SUPPORTED_FIGURE_TYPES,
    NormalizedFigureManifestFigure,
    format_figure_manifest_validation_error,
    normalize_figure_manifest,
)
from .files import PaperTree, as_tree, lexists, read_regular_nofollow
from .graphics import read_tex_file_bounded, resolve_graphics_references
from .tex import read_json

DEFAULT_CATEGORY_BUDGET = {
    "grouped_bar": 6,
    "bar": 8,
    "line": 8,
    "box": 6,
    "violin": 6,
    "scatter": 8,
    "heatmap": 8,
    "table_matrix": 8,
    "table": 8,
    "composite": 6,
    "diagram": 1,
}
DEFAULT_SERIES_BUDGET = {
    "grouped_bar": 4,
    "bar": 4,
    "line": 4,
    "box": 4,
    "violin": 4,
    "scatter": 4,
    "heatmap": 8,
    "table_matrix": 8,
    "table": 8,
    "composite": 4,
    "diagram": 1,
}
STRATEGY_FIGURE_TYPE_ALIASES = {
    "flow_diagram": "diagram",
    "horizontal_bar": "bar",
}


def _normalized_strategy_figure_type(value: object) -> str:
    figure_type = str(value).strip()
    return STRATEGY_FIGURE_TYPE_ALIASES.get(figure_type, figure_type)


def read_png_size(source: PaperTree | Path | str, rel: str | None = None) -> tuple[int, int]:
    """Read PNG width and height from the IHDR chunk (a tree file, or a host path)."""
    if rel is None:
        with Path(source).open("rb") as handle:  # type: ignore[arg-type]
            header = handle.read(24)
        return _read_png_size_header(header, str(source))
    data = read_regular_nofollow(as_tree(source), rel, 24)
    return _read_png_size_header(data or b"", rel)


def _read_png_size_header(header: bytes, path_label: str) -> tuple[int, int]:
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Invalid PNG file: {path_label}")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


CONCEPT_FIGURE_GENERATOR = "generate-concept-figure"
CONCEPT_FIGURE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def is_valid_concept_figure_id(figure_id: str) -> bool:
    return CONCEPT_FIGURE_ID_PATTERN.fullmatch(figure_id) is not None


def _read_json_object_nofollow(tree: PaperTree, rel: str) -> dict[str, Any]:
    data = read_regular_nofollow(tree, rel)
    if data is None:
        raise RuntimeError(f"Could not safely open JSON artifact: {rel}")
    try:
        payload = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON in {rel}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {rel}")
    return payload


def _reject_unfinished_concept_prompt_log(payload: Any, path_label: str) -> None:
    if isinstance(payload, dict) and (
        payload.get("placeholder") is True
        or str(payload.get("status", "")).lower() in {"pending", "placeholder"}
    ):
        raise RuntimeError(f"Concept prompt log is unfinished: {path_label}")


def _concept_prompt_log_rel(figure_id: str) -> str:
    if not is_valid_concept_figure_id(figure_id):
        raise RuntimeError(f"Invalid concept figure_id: {figure_id!r}")
    return f"figures/{figure_id}.prompts.json"


def read_concept_figure_prompt_log(
    paper_dir: PaperTree | Path | str,
    figure_id: str,
    prompt_log_path: str,
) -> dict[str, Any]:
    """Read a non-placeholder prompt log from its canonical safe path."""
    tree = as_tree(paper_dir)
    expected_path = _concept_prompt_log_rel(figure_id)
    if prompt_log_path != expected_path:
        raise RuntimeError(
            f"Concept figure {figure_id!r} must use prompt_log_path {expected_path!r}"
        )
    payload = _read_json_object_nofollow(tree, prompt_log_path)
    if not payload:
        raise RuntimeError(f"Concept prompt log is an empty placeholder: {prompt_log_path}")
    _reject_unfinished_concept_prompt_log(payload, prompt_log_path)
    return payload


def _read_regular_figure_artifact_nofollow(
    paper_dir: PaperTree,
    artifact_rel: str,
    expected_suffix: str,
    read_limit: int | None,
) -> bytes | None:
    components = artifact_rel.split("/")
    if (
        len(components) < 2
        or components[0] != "figures"
        or any(component in {"", ".", ".."} or "\\" in component for component in components)
        or Path(components[-1]).suffix.lower() != expected_suffix
    ):
        return None
    return read_regular_nofollow(paper_dir, artifact_rel, read_limit)


def classify_figure_validation_errors(errors: list[str]) -> tuple[list[str], list[str]]:
    """Split figure validation issues into hard blockers and soft readability warnings."""
    soft_patterns = (
        re.compile(r"^figures\[\d+\] exceeds max total panels:"),
        re.compile(r"^figures\[\d+\] exceeds max panels per row:"),
        re.compile(r"^figures\[\d+\] panels_per_row exceeds n_panels"),
        re.compile(r"^figures\[\d+\] exceeds category budget in panel"),
        re.compile(r"^figures\[\d+\] exceeds series budget in panel"),
        re.compile(r"^figures\[\d+\] labels too long without abbreviation"),
        re.compile(r"^figures\[\d+\]\.font_sizes\."),
        re.compile(r"^figures\[\d+\]\.min_font_size_pt below minimum:"),
        re.compile(r"^figures\[\d+\] missing font size metadata"),
        re.compile(r"^figures\[\d+\] PNG width too small:"),
        re.compile(r"^figures\[\d+\] PNG height too small:"),
        re.compile(
            r"^figures\[\d+\] paper\.tex references raster PNG "
            r"despite available vector PDF$"
        ),
    )
    hard_errors: list[str] = []
    soft_warnings: list[str] = []
    for error in errors:
        if any(pattern.match(error) for pattern in soft_patterns):
            soft_warnings.append(error)
        else:
            hard_errors.append(error)
    return hard_errors, soft_warnings


FIGURE_QUALITY_WARNING_MARKERS = (
    "PNG width too small",
    "PNG height too small",
    "references raster PNG despite available vector PDF",
)


def has_actionable_figure_quality_warnings(warnings: list[str]) -> bool:
    """Return whether validation found a figure issue suitable for targeted repair."""
    return any(
        marker in warning for warning in warnings for marker in FIGURE_QUALITY_WARNING_MARKERS
    )


def validate_figure_manifest(
    paper_dir: PaperTree | Path | str,
    *,
    tex_text: str,
    allowed_source_ids: set[str],
    manifest_path_label: str | None = None,
) -> dict[str, Any]:
    """Validate figure manifest artifacts and readability budgets.

    ``paper_dir`` is a tree of the paper directory, or a host directory path. The report's
    ``manifest_path`` is ``manifest_path_label`` (default: the resolved path of a directory, or
    the relative path ``figure_manifest.json`` for another tree).
    """
    tree = as_tree(paper_dir)
    strategy_path = "paper_strategy.json"
    manifest_path = "figure_manifest.json"
    if manifest_path_label is None:
        manifest_path_label = (
            str((Path(paper_dir_root) / manifest_path).resolve())
            if (paper_dir_root := _directory_root(tree)) is not None
            else manifest_path
        )
    errors: list[str] = []
    graphics_resolution_failed = False
    try:
        graphics_resolutions = resolve_graphics_references(tex_text, tree)
    except ValueError as exc:
        errors.append(f"paper.tex graphics validation failed: {exc}")
        graphics_resolutions = []
        graphics_resolution_failed = True
    tex_graphics = [resolution.raw_path for resolution in graphics_resolutions]
    graphics_candidates: dict[str, set[str]] = {raw_path: set() for raw_path in tex_graphics}
    for resolution in graphics_resolutions:
        if resolution.resolved_path is not None:
            graphics_candidates[resolution.raw_path].add(resolution.resolved_path)
    explicitly_referenced_pngs = {
        resolution.resolved_path
        for resolution in graphics_resolutions
        if resolution.explicitly_png and resolution.resolved_path is not None
    }

    try:
        strategy = read_json(tree, strategy_path, required=False)
    except RuntimeError as exc:
        strategy = {}
        errors.append(str(exc))

    figure_specs = strategy.get("figure_specs", [])
    if figure_specs and not isinstance(figure_specs, list):
        errors.append("paper_strategy.json field must be a list: figure_specs")
        figure_specs = []

    concept_spec_count = sum(
        1
        for spec in figure_specs
        if isinstance(spec, dict)
        and str(spec.get("preferred_generator", "")).strip() == CONCEPT_FIGURE_GENERATOR
        and spec.get("required") is not False
    )
    if concept_spec_count > 1:
        errors.append(
            f"paper_strategy.json may contain at most one {CONCEPT_FIGURE_GENERATOR} spec"
        )

    manifest_required = (
        bool(tex_graphics) or bool(figure_specs) or lexists(tree, manifest_path) or bool(errors)
    )
    if not manifest_required:
        return {
            "checked": False,
            "required": False,
            "figure_count": 0,
            "referenced_paths": [],
            "manifest_path": manifest_path_label,
            "errors": errors,
            "hard_errors": [],
            "soft_warnings": [],
        }

    manifest: dict[str, Any] = {}
    figure_rows: list[NormalizedFigureManifestFigure] = []
    if not lexists(tree, manifest_path):
        errors.append(f"Missing required artifact: {manifest_path_label}")
    else:
        try:
            manifest = read_json(tree, manifest_path)
        except RuntimeError as exc:
            errors.append(str(exc))
        else:
            try:
                figure_rows = normalize_figure_manifest(manifest)
            except ValidationError as exc:
                errors.append(
                    "figure_manifest.json failed schema validation: "
                    f"{format_figure_manifest_validation_error(exc)}"
                )

    spec_map: dict[str, dict[str, Any]] = {}
    for idx, spec in enumerate(figure_specs):
        if not isinstance(spec, dict):
            errors.append(f"figure_specs[{idx}] must be an object")
            continue
        figure_id = str(spec.get("figure_id", "")).strip()
        if not figure_id:
            errors.append(f"figure_specs[{idx}] missing non-empty figure_id")
            continue
        preferred_generator = str(spec.get("preferred_generator", "")).strip()
        if preferred_generator and preferred_generator not in SUPPORTED_FIGURE_GENERATORS:
            allowed = ", ".join(sorted(SUPPORTED_FIGURE_GENERATORS))
            errors.append(f"figure_specs[{idx}].preferred_generator must be one of: {allowed}")
        if preferred_generator == CONCEPT_FIGURE_GENERATOR and not is_valid_concept_figure_id(
            figure_id
        ):
            errors.append(f"figure_specs[{idx}] invalid figure_id for {CONCEPT_FIGURE_GENERATOR}")
            continue
        figure_type = _normalized_strategy_figure_type(spec.get("figure_type", ""))
        if (
            bool(spec.get("required", False))
            and figure_type == "diagram"
            and not preferred_generator
        ):
            errors.append(f"figure_specs[{idx}] missing preferred_generator for required diagram")
        spec_map[figure_id] = spec

    seen_ids: set[str] = set()
    manifest_paths: set[str] = set()

    for idx, row in enumerate(figure_rows):
        figure_id = row.figure_id
        if figure_id in seen_ids:
            errors.append(f"duplicate figure_id: {figure_id}")
        seen_ids.add(figure_id)

        path_pdf_raw = row.path_pdf.strip()
        path_png_raw = row.path_png.strip()
        figure_type = row.figure_type.strip()
        if figure_type not in SUPPORTED_FIGURE_TYPES:
            errors.append(f"figures[{idx}] unsupported figure_type: {figure_type!r}")

        png_header = _read_regular_figure_artifact_nofollow(
            tree,
            path_png_raw,
            ".png",
            24,
        )
        manifest_paths.add(Path(path_png_raw).as_posix())
        # A PDF is optional for concept output and documented failed renders.
        # Validate it only when the row declares one.
        pdf_header = None
        if path_pdf_raw:
            pdf_header = _read_regular_figure_artifact_nofollow(
                tree,
                path_pdf_raw,
                ".pdf",
                5,
            )
            manifest_paths.add(Path(path_pdf_raw).as_posix())
            if pdf_header is None:
                errors.append(f"figures[{idx}] missing PDF artifact: {path_pdf_raw}")
        if png_header is None:
            errors.append(f"figures[{idx}] missing PNG artifact: {path_png_raw}")

        n_panels = row.n_panels
        panels_per_row = row.panels_per_row
        categories_per_panel = row.categories_per_panel
        series_per_panel = row.series_per_panel
        max_label_chars = row.max_label_chars
        uses_abbrev_labels = row.uses_abbrev_labels
        uses_progression = row.uses_progression
        source_ids = row.source_ids
        generator = (
            row.generator.strip()
            if isinstance(row.generator, str) and row.generator.strip()
            else None
        )
        attempted_generator = (
            row.attempted_generator.strip()
            if isinstance(row.attempted_generator, str) and row.attempted_generator.strip()
            else None
        )
        prompt_log_path_raw = (
            row.prompt_log_path.strip()
            if isinstance(row.prompt_log_path, str) and row.prompt_log_path.strip()
            else None
        )
        fallback_reason = (
            row.fallback_reason.strip()
            if isinstance(row.fallback_reason, str) and row.fallback_reason.strip()
            else None
        )

        if n_panels > 4:
            errors.append(f"figures[{idx}] exceeds max total panels: {n_panels} > 4")
        if panels_per_row > 3:
            errors.append(f"figures[{idx}] exceeds max panels per row: {panels_per_row} > 3")
        elif panels_per_row > n_panels:
            errors.append(f"figures[{idx}] panels_per_row exceeds n_panels")

        category_budget = DEFAULT_CATEGORY_BUDGET.get(figure_type, 6)
        series_budget = DEFAULT_SERIES_BUDGET.get(figure_type, 4)
        spec = spec_map.get(figure_id)
        preferred_generator = ""
        if spec is not None:
            category_budget = int(spec.get("max_categories_per_panel", category_budget))
            series_budget = int(spec.get("max_series_per_panel", series_budget))
            strategy_figure_type = _normalized_strategy_figure_type(
                spec.get("figure_type", figure_type)
            )
            if figure_type != strategy_figure_type:
                errors.append(f"figures[{idx}] figure_type does not match strategy for {figure_id}")
            if n_panels > int(spec.get("max_panels", 4)):
                errors.append(f"figures[{idx}] exceeds strategy max_panels for {figure_id}")
            if uses_progression and not bool(spec.get("allow_progression", False)):
                errors.append(f"figures[{idx}] progression forbidden by strategy for {figure_id}")
            preferred_generator = str(spec.get("preferred_generator", "")).strip()
            if preferred_generator == "generate-concept-figure":
                if generator == "generate-concept-figure":
                    if prompt_log_path_raw is None:
                        errors.append(
                            f"figures[{idx}] missing prompt_log_path for generate-concept-figure output"
                        )
                    else:
                        try:
                            prompt_log = read_concept_figure_prompt_log(
                                tree,
                                figure_id,
                                prompt_log_path_raw,
                            )
                        except RuntimeError as exc:
                            errors.append(f"figures[{idx}] invalid prompt log: {exc}")
                        else:
                            if prompt_log.get("status") == "failed":
                                errors.append(
                                    f"figures[{idx}] prompt log records a failed render "
                                    "but generator claims generate-concept-figure"
                                )
                elif fallback_reason is None:
                    errors.append(
                        f"figures[{idx}] bypassed preferred_generator generate-concept-figure without fallback_reason"
                    )
                elif generator is None:
                    if attempted_generator != CONCEPT_FIGURE_GENERATOR:
                        errors.append(
                            f"figures[{idx}] failed concept render missing attempted_generator"
                        )
                    if prompt_log_path_raw is None:
                        errors.append(
                            f"figures[{idx}] missing prompt_log_path for failed concept render"
                        )
                    else:
                        try:
                            prompt_log = read_concept_figure_prompt_log(
                                tree,
                                figure_id,
                                prompt_log_path_raw,
                            )
                        except RuntimeError as exc:
                            errors.append(f"figures[{idx}] invalid failure prompt log: {exc}")
                        else:
                            if prompt_log.get("status") != "failed":
                                errors.append(
                                    f"figures[{idx}] fallback prompt log must record status failed"
                                )
                            if prompt_log.get("attempted_generator") != CONCEPT_FIGURE_GENERATOR:
                                errors.append(
                                    f"figures[{idx}] failure prompt log must record "
                                    f"attempted_generator {CONCEPT_FIGURE_GENERATOR}"
                                )
                            if prompt_log.get("reason") != fallback_reason:
                                errors.append(
                                    f"figures[{idx}] fallback_reason does not match prompt log"
                                )

        if attempted_generator == CONCEPT_FIGURE_GENERATOR and (
            spec is None or preferred_generator != CONCEPT_FIGURE_GENERATOR
        ):
            errors.append(
                f"figures[{idx}] records a concept render attempt without a matching strategy spec"
            )
        if generator == CONCEPT_FIGURE_GENERATOR and (
            spec is None or preferred_generator != CONCEPT_FIGURE_GENERATOR
        ):
            errors.append(
                f"figures[{idx}] claims {CONCEPT_FIGURE_GENERATOR} output without "
                "a matching strategy spec"
            )

        for panel_idx, value in enumerate(categories_per_panel):
            if value > category_budget:
                errors.append(
                    f"figures[{idx}] exceeds category budget in panel {panel_idx}: {value} > {category_budget}"
                )
        for panel_idx, value in enumerate(series_per_panel):
            if value > series_budget:
                errors.append(
                    f"figures[{idx}] exceeds series budget in panel {panel_idx}: {value} > {series_budget}"
                )

        if max_label_chars is not None and max_label_chars > 18 and uses_abbrev_labels is False:
            errors.append(f"figures[{idx}] labels too long without abbreviation")

        minimums = {"title": 10.0, "axis": 9.0, "ticks": 8.0, "legend": 8.0}
        if row.font_sizes is not None:
            for key, minimum in minimums.items():
                font_value = row.font_sizes.get(key)
                if font_value is None:
                    errors.append(f"figures[{idx}].font_sizes.{key} missing")
                elif float(font_value) < minimum:
                    errors.append(
                        f"figures[{idx}].font_sizes.{key} below minimum: {font_value} < {minimum}"
                    )
        elif row.min_font_size_pt is not None:
            minimum = 7.0 if figure_type == "diagram" else 8.0
            if row.min_font_size_pt < minimum:
                errors.append(
                    f"figures[{idx}].min_font_size_pt below minimum: {row.min_font_size_pt} < {minimum}"
                )
        else:
            errors.append(f"figures[{idx}] missing font size metadata")

        if png_header is not None:
            width, height = _read_png_size_header(png_header, path_png_raw)
            if row.placement == "full_width":
                min_width = 2400
            elif row.placement == "column":
                min_width = 1200
            else:
                min_width = 1600 if n_panels > 2 else 1200
            min_height = 900 if n_panels > 2 else 700
            if width < min_width:
                errors.append(f"figures[{idx}] PNG width too small: {width} < {min_width}")
            if height < min_height:
                errors.append(f"figures[{idx}] PNG height too small: {height} < {min_height}")

        normalized_png_path = Path(path_png_raw).as_posix()
        pdf_is_vector_candidate = pdf_header is not None and pdf_header.startswith(b"%PDF-")
        if (
            path_pdf_raw
            and pdf_is_vector_candidate
            and normalized_png_path in explicitly_referenced_pngs
        ):
            errors.append(
                f"figures[{idx}] paper.tex references raster PNG despite available vector PDF"
            )

        if not source_ids:
            if figure_type != "diagram":
                errors.append(f"figures[{idx}].source_ids must be non-empty list")
        else:
            for source_id in source_ids:
                if source_id not in allowed_source_ids:
                    errors.append(f"figures[{idx}] unknown source id: {source_id!r}")

    for figure_id, spec in spec_map.items():
        if bool(spec.get("required", False)) and figure_id not in seen_ids:
            errors.append(f"Missing required figure from strategy: {figure_id}")

    if not graphics_resolution_failed:
        referenced_manifest_paths: set[str] = set()
        for raw_path in tex_graphics:
            candidates = graphics_candidates[raw_path]
            matched = {candidate for candidate in candidates if candidate in manifest_paths}
            if not matched:
                errors.append(
                    f"paper.tex references figure missing from figure_manifest.json: {raw_path}"
                )
                continue
            referenced_manifest_paths.update(matched)

        for manifest_rel_path in sorted(manifest_paths):
            if manifest_rel_path.endswith(".png"):
                counterpart = Path(manifest_rel_path).with_suffix(".pdf").as_posix()
                if counterpart in referenced_manifest_paths:
                    continue
            if manifest_rel_path.endswith(".pdf"):
                counterpart = Path(manifest_rel_path).with_suffix(".png").as_posix()
                if counterpart in referenced_manifest_paths:
                    continue
            if manifest_rel_path not in referenced_manifest_paths and tex_graphics:
                errors.append(
                    f"figure_manifest entry not referenced by paper.tex: {manifest_rel_path}"
                )

    hard_errors, soft_warnings = classify_figure_validation_errors(errors)
    return {
        "checked": True,
        "required": manifest_required,
        "figure_count": len(figure_rows),
        "referenced_paths": sorted(manifest_paths),
        "manifest_path": manifest_path_label,
        "errors": errors,
        "hard_errors": hard_errors,
        "soft_warnings": soft_warnings,
    }


def _directory_root(tree: PaperTree) -> Path | None:
    root = getattr(tree, "root", None)
    return Path(root) if isinstance(root, Path) else None


__all__ = [
    "CONCEPT_FIGURE_GENERATOR",
    "DEFAULT_CATEGORY_BUDGET",
    "DEFAULT_SERIES_BUDGET",
    "FIGURE_QUALITY_WARNING_MARKERS",
    "classify_figure_validation_errors",
    "has_actionable_figure_quality_warnings",
    "is_valid_concept_figure_id",
    "read_concept_figure_prompt_log",
    "read_json",
    "read_png_size",
    "read_tex_file_bounded",
    "validate_figure_manifest",
]
