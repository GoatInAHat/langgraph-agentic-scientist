"""Context strings and placeholder values used by several phases.

Ports of upstream helpers that turn files into prompt text: the ``task.md`` sections
(``files.read_task_sections``, ``extract_markdown_section``), ``relative_display_path``, bounded
UTF-8 cuts, and the base64 blocks upstream's shell drivers embed in repair prompts.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
"""A Markdown ATX heading (upstream ``constants.MARKDOWN_HEADING_RE``)."""


def extract_markdown_section(markdown: str, heading: str) -> str:
    """Extract a Markdown section body by heading title (case-insensitive), as upstream does."""
    matches = list(MARKDOWN_HEADING_RE.finditer(markdown))
    target = heading.strip().lower()
    for index, match in enumerate(matches):
        if match.group(1).strip().lower() != target:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        return markdown[start:end].strip()
    return ""


def task_sections(task_md: str | None) -> tuple[str, str]:
    """Return the ``Goal`` and ``Rubrics`` bodies of a ``task.md`` with upstream's fallbacks."""
    if task_md is None:
        return "(missing `task.md`)", "(missing `task.md`)"
    goal = extract_markdown_section(task_md, "Goal") or "(missing Goal section)"
    rubrics = extract_markdown_section(task_md, "Rubrics") or "(missing Rubrics section)"
    return goal, rubrics


def standard_values(
    *, task_dir: str, results_dir: str, task_md: str | None, task_md_path: str | None = None
) -> dict[str, str]:
    """Return the placeholders every upstream prompt receives (``files.load_instructions``).

    ``TASK_DIR``, ``RESULTS_DIR``, ``TASK_MD_PATH``, ``TASK_GOAL`` and ``TASK_RUBRICS``; the paths
    are rendered as the role sees them (``.`` in a search workspace, ``./task`` in a paper
    workspace).
    """
    goal, rubrics = task_sections(task_md)
    md_path = task_md_path or (f"{task_dir.rstrip('/')}/task.md" if task_dir else "task.md")
    return {
        "TASK_DIR": task_dir,
        "RESULTS_DIR": results_dir,
        "TASK_MD_PATH": md_path,
        "TASK_GOAL": goal,
        "TASK_RUBRICS": rubrics,
    }


def relative_display_path(path: str, *, base_dir: str) -> str:
    """Render workspace path ``path`` relative to ``base_dir`` as upstream renders prompt paths.

    The result is ``.``, ``./<path>`` or a ``../`` path.
    """
    target = [part for part in PurePosixPath(path).parts if part not in ("", ".")]
    base = [part for part in PurePosixPath(base_dir).parts if part not in ("", ".")]
    common = 0
    while common < min(len(target), len(base)) and target[common] == base[common]:
        common += 1
    parts = [".."] * (len(base) - common) + target[common:]
    if not parts:
        return "."
    relative = "/".join(parts)
    return relative if relative.startswith("../") else f"./{relative}"


def truncate_utf8(text: str, max_bytes: int, *, marker: str = "") -> str:
    """Cut ``text`` to at most ``max_bytes`` UTF-8 bytes on a character boundary."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    room = max(max_bytes - len(marker.encode("utf-8")), 0)
    return data[:room].decode("utf-8", errors="ignore") + marker


def tail_utf8(text: str, max_bytes: int) -> str:
    """Keep the last ``max_bytes`` UTF-8 bytes of ``text`` on a character boundary."""
    data = text.encode("utf-8")
    if len(data) <= max_bytes:
        return text
    return data[-max_bytes:].decode("utf-8", errors="ignore")


def base64_tail(payload: bytes | str, max_bytes: int) -> str:
    """Return base64 of the last ``max_bytes`` bytes of a payload.

    Upstream's drafting driver bounds each diagnostic it embeds in a repair prompt this way
    (``encoded_file`` in ``task.sh``: the tail is kept, then encoded).
    """
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return base64.b64encode(data).decode("ascii")


def split_repair_policy(template: str) -> tuple[str, str]:
    """Split ``repair_task.md`` into its boundary policy and its repair policy (upstream rule).

    The boundary policy is the text before ``# Role``; the repair policy is the text after
    ``# Repair policy``. The qualification repair prompt embeds both (``task.sh``).

    Raises:
        ValueError: When the template has no ``# Repair policy`` heading.
    """
    boundary_policy = template.partition("# Role\n")[0]
    _, heading, repair_policy = template.partition("# Repair policy\n")
    if not heading:
        raise ValueError("Shared scaffold repair policy is missing")
    return boundary_policy, repair_policy


def json_text(value: Any, *, indent: int | None = 2) -> str:
    """Serialize a JSON value the way upstream writes JSON files (sorted keys, newline)."""
    return json.dumps(value, indent=indent, sort_keys=True, ensure_ascii=False) + "\n"


def bullet_list(items: Mapping[str, str] | list[str], *, empty: str = "- none") -> str:
    """Render items as ``- `` bullets, or ``empty`` when there are none."""
    if isinstance(items, Mapping):
        lines = [f"- {key}: {value}" for key, value in items.items()]
    else:
        lines = [f"- {item}" for item in items]
    return "\n".join(lines) if lines else empty


__all__ = [
    "MARKDOWN_HEADING_RE",
    "base64_tail",
    "bullet_list",
    "extract_markdown_section",
    "json_text",
    "relative_display_path",
    "split_repair_policy",
    "standard_values",
    "tail_utf8",
    "task_sections",
    "truncate_utf8",
]
