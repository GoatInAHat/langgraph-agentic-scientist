"""Page-budget validation from a compiled PDF.

Upstream ``paper_step.py`` (``read_pdf_page_texts``, ``read_pdf_page_layouts``,
``compute_last_body_page_fill_ratio``, ``validate_page_budget`` and their helpers). The geometry
and the rules are carried over unchanged; the PDF arrives as bytes (the workspace may be remote),
and the enforced page budgets come from the venue catalog instead of a module constant.
"""

# ruff: noqa: E501

from __future__ import annotations

import io
import math
import re
from typing import Any

from ...venues import VenueCatalog
from ..watermark import WATERMARK_TEXT
from .supplement import RENDERED_REFERENCE_HEADING_RE as REFERENCE_HEADING_RE
from .supplement import find_supplement_start_page
from .venues import detect_paper_format


def _pdf_reader(pdf: bytes) -> Any:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise RuntimeError(
            "pypdf is required to inspect the compiled paper.pdf; install the `review` extra"
        ) from exc
    return PdfReader(io.BytesIO(pdf))


def read_pdf_page_texts(pdf: bytes) -> list[str]:
    """Extract page text from a compiled PDF."""
    reader = _pdf_reader(pdf)
    return [page.extract_text() or "" for page in reader.pages]


def _fragment_y_position(*, user_matrix: Any, text_matrix: Any) -> float | None:
    """Prefer text-space y coordinates when user-space coordinates collapse to zero."""
    candidates: list[float] = []
    for matrix in (text_matrix, user_matrix):
        if not isinstance(matrix, (list, tuple)) or len(matrix) <= 5:
            continue
        try:
            candidate = float(matrix[5])
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            candidates.append(candidate)
    if not candidates:
        return None

    for candidate in candidates:
        if abs(candidate) > 1e-6:
            return candidate
    return candidates[0]


def _fragment_x_position(*, user_matrix: Any, text_matrix: Any) -> float | None:
    """Prefer text-space x coordinates when user-space coordinates collapse to zero."""
    candidates: list[float] = []
    for matrix in (text_matrix, user_matrix):
        if not isinstance(matrix, (list, tuple)) or len(matrix) <= 4:
            continue
        try:
            candidate = float(matrix[4])
        except (TypeError, ValueError):
            continue
        if math.isfinite(candidate):
            candidates.append(candidate)
    if not candidates:
        return None

    for candidate in candidates:
        if abs(candidate) > 1e-6:
            return candidate
    return candidates[0]


def _graphic_bounds_from_matrix(matrix: Any) -> dict[str, float] | None:
    """Return the page-space bounds of a unit-square PDF graphic."""
    if not isinstance(matrix, (list, tuple)) or len(matrix) < 6:
        return None
    try:
        a, b, c, d, e, f = (float(matrix[index]) for index in range(6))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (a, b, c, d, e, f)):
        return None
    linear_scale = max(abs(a), abs(b), abs(c), abs(d), 1.0)
    normalized_determinant = (a / linear_scale) * (d / linear_scale) - (b / linear_scale) * (
        c / linear_scale
    )
    if not math.isfinite(normalized_determinant) or abs(normalized_determinant) <= 1e-12:
        return None

    corners = (
        (e, f),
        (a + e, b + f),
        (c + e, d + f),
        (a + c + e, b + d + f),
    )
    if not all(math.isfinite(value) for point in corners for value in point):
        return None
    x_values = [point[0] for point in corners]
    y_values = [point[1] for point in corners]
    bounds = {
        "x_min": min(x_values),
        "x_max": max(x_values),
        "y_min": min(y_values),
        "y_max": max(y_values),
    }
    if bounds["x_max"] <= bounds["x_min"] or bounds["y_max"] <= bounds["y_min"]:
        return None
    return bounds


def _page_image_names(page: Any) -> set[str]:
    """Return image XObject names declared by a PDF page."""
    try:
        resource_owner = page
        visited: set[int] = set()
        resources = None
        while resource_owner is not None and id(resource_owner) not in visited:
            visited.add(id(resource_owner))
            resources = resource_owner.get("/Resources")
            if resources is not None:
                break
            parent = resource_owner.get("/Parent")
            resource_owner = parent.get_object() if hasattr(parent, "get_object") else parent
        resources = resources or {}
        if hasattr(resources, "get_object"):
            resources = resources.get_object()
        xobjects = resources.get("/XObject") or {}
        if hasattr(xobjects, "get_object"):
            xobjects = xobjects.get_object()
        return {
            str(name)
            for name, reference in xobjects.items()
            if str(reference.get_object().get("/Subtype")) == "/Image"
        }
    except (AttributeError, KeyError, TypeError, ValueError):
        return set()


def _multiply_pdf_matrices(left: Any, right: Any) -> list[float] | None:
    """Multiply two six-value PDF transformation matrices."""
    if not all(isinstance(matrix, (list, tuple)) and len(matrix) >= 6 for matrix in (left, right)):
        return None
    try:
        m = [float(value) for value in left[:6]]
        n = [float(value) for value in right[:6]]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (*m, *n)):
        return None
    return [
        m[0] * n[0] + m[1] * n[2],
        m[0] * n[1] + m[1] * n[3],
        m[2] * n[0] + m[3] * n[2],
        m[2] * n[1] + m[3] * n[3],
        m[4] * n[0] + m[5] * n[2] + n[4],
        m[4] * n[1] + m[5] * n[3] + n[5],
    ]


def _read_page_graphics(
    page: Any,
    *,
    max_operations: int = 100_000,
    max_state_depth: int = 1024,
    max_unique_graphics: int = 1024,
) -> list[dict[str, float]]:
    """Read top-level image placements without recursing into Form XObjects."""
    image_names = _page_image_names(page)
    if not image_names or max_operations <= 0 or max_state_depth <= 0 or max_unique_graphics <= 0:
        return []
    try:
        operations = page.get_contents().operations
    except (AttributeError, KeyError, TypeError, ValueError):
        return []
    if len(operations) > max_operations:
        return []

    current_matrix = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    matrix_stack: list[list[float]] = []
    unique_bounds: set[tuple[float, float, float, float]] = set()
    for operands, operator in operations:
        if operator == b"q":
            if len(matrix_stack) >= max_state_depth:
                return []
            matrix_stack.append(current_matrix.copy())
            continue
        if operator == b"Q":
            current_matrix = matrix_stack.pop() if matrix_stack else [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            continue
        if operator == b"cm":
            combined = _multiply_pdf_matrices(operands, current_matrix)
            current_matrix = combined or [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            continue
        if operator != b"Do" or not isinstance(operands, (list, tuple)) or not operands:
            continue
        if str(operands[0]) not in image_names:
            continue
        bounds = _graphic_bounds_from_matrix(current_matrix)
        if bounds is None:
            continue
        bounds_key = (
            bounds["x_min"],
            bounds["x_max"],
            bounds["y_min"],
            bounds["y_max"],
        )
        if bounds_key not in unique_bounds and len(unique_bounds) >= max_unique_graphics:
            return []
        unique_bounds.add(bounds_key)

    return [
        {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
        for x_min, x_max, y_min, y_max in sorted(unique_bounds)
    ]


def read_pdf_page_layouts(pdf: bytes) -> list[dict[str, Any]]:
    """Extract per-page text and image bounds for layout-aware validation."""
    reader = _pdf_reader(pdf)
    page_layouts: list[dict[str, Any]] = []

    for page in reader.pages:
        fragments: list[dict[str, float | str]] = []
        graphics = _read_page_graphics(page)

        def _visitor_text(
            text: Any,
            user_matrix: Any,
            text_matrix: Any,
            font_dict: Any,
            font_size: Any,
        ) -> None:
            del font_dict
            if not isinstance(text, str) or not text.strip():
                return

            y_value = _fragment_y_position(
                user_matrix=user_matrix,
                text_matrix=text_matrix,
            )
            if y_value is None:
                return
            x_value = _fragment_x_position(
                user_matrix=user_matrix,
                text_matrix=text_matrix,
            )
            if x_value is None:
                return

            try:
                font_size_value = float(font_size)
            except (TypeError, ValueError):
                font_size_value = 0.0
            if not math.isfinite(font_size_value):
                font_size_value = 0.0

            fragments.append(  # noqa: B023 - the visitor runs within this page's iteration
                {
                    "text": text,
                    "x": x_value,
                    "y": y_value,
                    "font_size": max(font_size_value, 0.0),
                }
            )

        page_text = page.extract_text(visitor_text=_visitor_text) or ""
        page_x_min = float(page.mediabox.left)
        page_x_max = float(page.mediabox.right)
        page_y_min = float(page.mediabox.bottom)
        page_y_max = float(page.mediabox.top)
        page_layouts.append(
            {
                "text": page_text,
                "page_width": page_x_max - page_x_min,
                "page_height": page_y_max - page_y_min,
                "page_x_min": page_x_min,
                "page_x_max": page_x_max,
                "page_y_min": page_y_min,
                "page_y_max": page_y_max,
                "fragments": fragments,
                "graphics": graphics,
            }
        )

    return page_layouts


def _is_meaningful_text_fragment(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and bool(re.search(r"[A-Za-z]", stripped))


def _reference_heading_y(page_layout: dict[str, Any]) -> float | None:
    fragments = page_layout.get("fragments", [])
    if not isinstance(fragments, list):
        return None
    heading_positions: list[float] = []
    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        text = fragment.get("text")
        y_value = fragment.get("y")
        if not isinstance(text, str):
            continue
        if y_value is None:
            continue
        try:
            y_float = float(y_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y_float):
            continue
        if REFERENCE_HEADING_RE.fullmatch(text.strip()):
            heading_positions.append(y_float)
    if not heading_positions:
        return None
    return max(heading_positions)


def _body_text_bounds(
    page_layout: dict[str, Any],
    *,
    exclude_reference_region: bool,
    column_index: int | None = None,
    body_columns: int = 1,
) -> tuple[float, float] | None:
    fragments = page_layout.get("fragments", [])
    if not isinstance(fragments, list):
        return None

    reference_heading_y = _reference_heading_y(page_layout) if exclude_reference_region else None
    page_width_raw = page_layout.get("page_width")
    if page_width_raw is None:
        page_width_raw = 0.0
    try:
        page_width = float(page_width_raw)
    except (TypeError, ValueError):
        page_width = 0.0
    try:
        page_x_min = float(page_layout.get("page_x_min", 0.0))
        page_x_max = float(page_layout.get("page_x_max", page_x_min + page_width))
        page_y_min = float(page_layout.get("page_y_min", 0.0))
        page_y_max = float(
            page_layout.get("page_y_max", page_y_min + float(page_layout.get("page_height", 0.0)))
        )
    except (TypeError, ValueError):
        page_x_min = 0.0
        page_x_max = page_width
        page_y_min = 0.0
        page_y_max = float(page_layout.get("page_height", 0.0) or 0.0)
    column_boundary = page_width / body_columns if body_columns > 0 else 0.0
    y_tops: list[float] = []
    y_bottoms: list[float] = []

    for fragment in fragments:
        if not isinstance(fragment, dict):
            continue
        text = fragment.get("text")
        if not isinstance(text, str) or not _is_meaningful_text_fragment(text):
            continue
        x_raw = fragment.get("x")
        y_raw = fragment.get("y")
        if x_raw is None or y_raw is None:
            continue
        try:
            x_value = float(x_raw)
            y_value = float(y_raw)
            font_size = float(fragment.get("font_size", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(x_value):
            continue
        if not math.isfinite(y_value):
            continue
        if column_index is not None and body_columns > 1 and page_width > 0:
            column_start = column_boundary * column_index
            column_end = column_boundary * (column_index + 1)
            if not (column_start <= x_value < column_end):
                continue
        if (
            exclude_reference_region
            and reference_heading_y is not None
            and y_value <= reference_heading_y
        ):
            continue
        if not math.isfinite(font_size) or font_size < 0:
            font_size = 0.0
        y_bottoms.append(y_value)
        y_tops.append(y_value + font_size)

    graphics = page_layout.get("graphics", [])
    if isinstance(graphics, list):
        for graphic in graphics:
            if not isinstance(graphic, dict):
                continue
            try:
                x_min = float(graphic["x_min"])
                x_max = float(graphic["x_max"])
                y_min = float(graphic["y_min"])
                y_max = float(graphic["y_max"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (x_min, x_max, y_min, y_max)):
                continue
            if x_max <= x_min or y_max <= y_min:
                continue
            x_min = max(x_min, page_x_min)
            x_max = min(x_max, page_x_max)
            y_min = max(y_min, page_y_min)
            y_max = min(y_max, page_y_max)
            if x_max <= x_min or y_max <= y_min:
                continue
            if column_index is not None and body_columns > 1 and page_width > 0:
                column_start = page_x_min + column_boundary * column_index
                column_end = page_x_min + column_boundary * (column_index + 1)
                if x_max <= column_start or x_min >= column_end:
                    continue
                x_min = max(x_min, column_start)
                x_max = min(x_max, column_end)
            if exclude_reference_region and reference_heading_y is not None:
                if y_max <= reference_heading_y:
                    continue
                y_min = max(y_min, reference_heading_y)
            y_bottoms.append(y_min)
            y_tops.append(y_max)

    if not y_tops or not y_bottoms:
        return None
    return max(y_tops), min(y_bottoms)


def compute_last_body_page_fill_ratio(
    page_layouts: list[dict[str, Any]],
    *,
    last_body_page_number: int,
    reference_start_page: int | None,
    body_columns: int = 1,
) -> float | None:
    """Estimate how much of the usable text span the last counted body page occupies."""
    if last_body_page_number < 1 or last_body_page_number > len(page_layouts):
        return None

    usable_bounds: list[tuple[float, float]] = []
    for page_number in range(1, last_body_page_number + 1):
        bounds = _body_text_bounds(
            page_layouts[page_number - 1],
            exclude_reference_region=page_number == reference_start_page,
            body_columns=body_columns,
        )
        if bounds is not None:
            usable_bounds.append(bounds)

    if not usable_bounds:
        return None

    usable_top = max(top for top, _ in usable_bounds)
    usable_bottom = min(bottom for _, bottom in usable_bounds)
    usable_height = usable_top - usable_bottom
    if usable_height <= 0:
        return None

    last_page = page_layouts[last_body_page_number - 1]
    if body_columns <= 1:
        last_bounds = _body_text_bounds(
            last_page,
            exclude_reference_region=last_body_page_number == reference_start_page,
            body_columns=body_columns,
        )
        if last_bounds is None:
            return None
        last_top, last_bottom = last_bounds
        fill_ratio = (last_top - last_bottom) / usable_height
        return max(0.0, min(fill_ratio, 1.0))

    column_fill_ratios: list[float] = []
    for column_index in range(body_columns):
        last_bounds = _body_text_bounds(
            last_page,
            exclude_reference_region=last_body_page_number == reference_start_page,
            column_index=column_index,
            body_columns=body_columns,
        )
        if last_bounds is None:
            column_fill_ratios.append(0.0)
            continue
        last_top, last_bottom = last_bounds
        fill_ratio = (last_top - last_bottom) / usable_height
        column_fill_ratios.append(max(0.0, min(fill_ratio, 1.0)))
    if not column_fill_ratios:
        return None
    return sum(column_fill_ratios) / len(column_fill_ratios)


def measure_last_body_page_fill_ratio(
    pdf: bytes,
    *,
    last_body_page_number: int,
    reference_start_page: int | None,
    body_columns: int = 1,
) -> float | None:
    """Read the compiled PDF and estimate last-page body-text occupancy."""
    page_layouts = read_pdf_page_layouts(pdf)
    return compute_last_body_page_fill_ratio(
        page_layouts,
        last_body_page_number=last_body_page_number,
        reference_start_page=reference_start_page,
        body_columns=body_columns,
    )


def find_reference_start_page(page_texts: list[str]) -> int | None:
    """Return the first 1-indexed page that visibly starts the references section."""
    for page_index, page_text in enumerate(page_texts, start=1):
        if REFERENCE_HEADING_RE.search(page_text):
            return page_index
    return None


def _is_watermark_text_line(line: str) -> bool:
    """True for extracted-text lines contributed by the mandatory page watermark.

    Extraction may keep the overlay as one line or split it into fragments, so any
    line whose letters form a contiguous chunk of the watermark text is treated as
    watermark output rather than manuscript body text.
    """
    normalized = re.sub(r"[^A-Za-z]", "", line).casefold()
    if not normalized:
        return False
    return normalized in re.sub(r"[^A-Za-z]", "", WATERMARK_TEXT).casefold()


def reference_page_has_body_prefix(page_text: str) -> bool:
    """Heuristically detect substantive body text before a references heading on the same page."""
    match = REFERENCE_HEADING_RE.search(page_text)
    if match is None:
        return False

    prefix = page_text[: match.start()]
    meaningful_lines: list[str] = []
    for raw_line in prefix.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not re.search(r"[A-Za-z]", line):
            continue
        if _is_watermark_text_line(line):
            continue
        meaningful_lines.append(line)

    if not meaningful_lines:
        return False
    if len(meaningful_lines) >= 2:
        return True

    only_line = meaningful_lines[0]
    if re.search(r"[.!?;]", only_line):
        return True
    if len(re.findall(r"[A-Za-z]+", only_line)) >= 4:
        return True
    return len(only_line) >= 20


def validate_page_budget(
    *,
    tex_text: str,
    pdf: bytes,
    page_texts: list[str] | None = None,
    catalog: VenueCatalog | None = None,
) -> dict[str, Any]:
    """Validate template-coupled page budgets from the compiled PDF."""
    catalog = catalog or VenueCatalog.default()
    paper_format = detect_paper_format(tex_text, catalog)
    budget = catalog.page_limits().get(paper_format) if paper_format is not None else None
    report: dict[str, Any] = {
        "checked": budget is not None,
        "paper_format": paper_format,
        "min_body_pages": None,
        "min_last_body_page_fill_ratio": None,
        "max_body_pages": None,
        "exclude_references": False,
        "total_pages": None,
        "reference_start_page": None,
        "supplement_start_page": None,
        "supplement_pages": 0,
        "body_pages": None,
        "last_body_page_number": None,
        "last_body_page_fill_ratio": None,
        "last_body_page_counted": None,
        "errors": [],
    }
    if page_texts is None and budget is None:
        return report

    if page_texts is None:
        page_texts = read_pdf_page_texts(pdf)
    total_pages = len(page_texts)
    reference_start_page = find_reference_start_page(page_texts)
    supplement_start_page = find_supplement_start_page(page_texts)
    report["total_pages"] = total_pages
    report["reference_start_page"] = reference_start_page
    report["supplement_start_page"] = supplement_start_page
    report["supplement_pages"] = (
        total_pages - supplement_start_page + 1 if supplement_start_page is not None else 0
    )

    if budget is None:
        return report

    min_body_pages = int(budget["min_body_pages"]) if "min_body_pages" in budget else None
    body_columns = int(budget.get("body_columns", 1))
    min_last_body_page_fill_ratio = (
        float(budget["min_last_body_page_fill_ratio"])
        if "min_last_body_page_fill_ratio" in budget
        else None
    )
    max_body_pages = int(budget["max_body_pages"])
    exclude_references = bool(budget.get("exclude_references", False))

    report["min_body_pages"] = min_body_pages
    report["min_last_body_page_fill_ratio"] = min_last_body_page_fill_ratio
    report["max_body_pages"] = max_body_pages
    report["exclude_references"] = exclude_references

    if exclude_references:
        if reference_start_page is None:
            if total_pages <= max_body_pages:
                raw_body_pages = total_pages
            else:
                report["errors"].append(
                    f"{paper_format} page-limit validation could not locate the references section in paper.pdf; total pages = {total_pages}, limit = {max_body_pages} excluding references"
                )
                return report
        else:
            reference_page_text = page_texts[reference_start_page - 1]
            raw_body_pages = max(reference_start_page - 1, 0)
            if reference_page_has_body_prefix(reference_page_text):
                raw_body_pages += 1
    else:
        raw_body_pages = total_pages

    body_pages = raw_body_pages
    last_body_page_number = raw_body_pages if raw_body_pages > 0 else None
    report["last_body_page_number"] = last_body_page_number

    if (
        min_body_pages is not None
        and min_last_body_page_fill_ratio is not None
        and raw_body_pages == min_body_pages
        and last_body_page_number is not None
    ):
        fill_ratio = measure_last_body_page_fill_ratio(
            pdf,
            last_body_page_number=last_body_page_number,
            reference_start_page=reference_start_page,
            body_columns=body_columns,
        )
        report["last_body_page_fill_ratio"] = fill_ratio
        if fill_ratio is None:
            report["last_body_page_counted"] = False
            report["errors"].append(
                f"{paper_format} page minimum validation could not determine the last counted body page fill ratio for page {last_body_page_number}"
            )
            report["body_pages"] = body_pages
            return report
        if fill_ratio < min_last_body_page_fill_ratio:
            body_pages = max(raw_body_pages - 1, 0)
            report["last_body_page_counted"] = False
            report["errors"].append(
                f"{paper_format} last counted body page fill ratio {fill_ratio:.2f} is below required {min_last_body_page_fill_ratio:.2f}; page {last_body_page_number} does not count toward minimum page requirement"
            )
        else:
            report["last_body_page_counted"] = True

    report["body_pages"] = body_pages

    if min_body_pages is not None and body_pages < min_body_pages:
        qualifier = " excluding references" if exclude_references else ""
        if min_last_body_page_fill_ratio is not None:
            qualifier += (
                f"; the final body page counts only when at least "
                f"{min_last_body_page_fill_ratio:.0%} full"
            )
        report["errors"].append(
            f"{paper_format} page minimum not met: body uses {body_pages} pages (min {min_body_pages}{qualifier})"
        )

    if raw_body_pages > max_body_pages:
        qualifier = " excluding references" if exclude_references else ""
        report["errors"].append(
            f"{paper_format} page limit exceeded: body uses {raw_body_pages} pages (max {max_body_pages}{qualifier})"
        )
    return report
