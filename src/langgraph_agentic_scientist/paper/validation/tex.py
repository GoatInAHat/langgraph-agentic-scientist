"""LaTeX text helpers of the paper validation gates.

Upstream ``paper_step.py`` (the regular expressions, comment stripping, visible-text reduction,
citation and BibTeX key extraction, required sections and balanced-brace parsing). The functions
are pure functions on manuscript text, carried over unchanged; only ``read_json`` reads a file
and takes a :class:`~langgraph_agentic_scientist.paper.validation.files.PaperTree`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .files import PaperTree, as_tree, read_regular_nofollow
from .graphics import FIGURE_INCLUDE_RE

SECTION_PATTERNS = {
    "abstract": re.compile(
        r"\\begin\{abstract\}|\\section\*?\{[^}]*abstract[^}]*\}",
        re.IGNORECASE,
    ),
    "introduction": re.compile(r"\\section\*?\{[^}]*introduction[^}]*\}", re.IGNORECASE),
    "related work": re.compile(r"\\section\*?\{[^}]*related work[^}]*\}", re.IGNORECASE),
    "method": re.compile(r"\\section\*?\{[^}]*method[^}]*\}", re.IGNORECASE),
    "results": re.compile(r"\\section\*?\{[^}]*results[^}]*\}", re.IGNORECASE),
    "conclusion": re.compile(r"\\section\*?\{[^}]*conclusion[^}]*\}", re.IGNORECASE),
    "references": re.compile(
        r"\\bibliography\{|\\printbibliography|\\begin\{thebibliography\}|\\section\*?\{[^}]*references[^}]*\}",
        re.IGNORECASE,
    ),
}
CITATION_KEY_RE = re.compile(r"@\w+\{([^,]+),")
TEX_CITATION_RE = re.compile(r"\\cite[a-zA-Z*]*\{([^}]+)\}")
TEX_INPUT_RE = re.compile(
    r"\\(?:input|include)(?![A-Za-z@])\s*"
    r"(?:\{(?P<braced>[^{}]+)\}|(?P<unbraced>[^\s%{}]+))"
)
TEX_IMPORT_COMMAND_RE = re.compile(r"\\(?:input|include)(?![A-Za-z@])")
TEX_LOCAL_SUPPORT_RE = re.compile(
    r"\\(?P<command>usepackage|RequirePackage|documentclass|LoadClass)"
    r"(?![A-Za-z@])(?:\s*\[[^\]]*\])?\s*\{(?P<targets>[^{}]+)\}"
)
TEX_LOCAL_SUPPORT_COMMAND_RE = re.compile(
    r"\\(?:usepackage|RequirePackage|documentclass|LoadClass)(?![A-Za-z@])"
)
DYNAMIC_TEX_IMPORT_RE = re.compile(r"\\csname\s*(?:input|include)\s*\\endcsname")
TEX_BIBLIOGRAPHY_RE = re.compile(r"\\bibliography(?![A-Za-z@])\s*\{(?P<targets>[^{}]+)\}")
TEX_BIBLIOGRAPHY_COMMAND_RE = re.compile(r"\\bibliography(?![A-Za-z@])")
TEX_BIB_RESOURCE_RE = re.compile(
    r"\\addbibresource(?![A-Za-z@])(?:\s*\[[^\]]*\])?\s*\{(?P<target>[^{}]+)\}"
)
TEX_BIB_RESOURCE_COMMAND_RE = re.compile(r"\\addbibresource(?![A-Za-z@])")
TEX_BIBLIOGRAPHY_STYLE_RE = re.compile(r"\\bibliographystyle(?![A-Za-z@])\s*\{(?P<target>[^{}]+)\}")
TEX_BIBLIOGRAPHY_STYLE_COMMAND_RE = re.compile(r"\\bibliographystyle(?![A-Za-z@])")
BIBTEX_AUX_DATA_RE = re.compile(r"\\bibdata\s*\{(?P<targets>[^{}]+)\}")
BIBTEX_AUX_DATA_COMMAND_RE = re.compile(r"\\bibdata(?![A-Za-z@])")
BIBTEX_AUX_STYLE_RE = re.compile(r"\\bibstyle\s*\{(?P<target>[^{}]+)\}")
BIBTEX_AUX_STYLE_COMMAND_RE = re.compile(r"\\bibstyle(?![A-Za-z@])")
BIBTEX_AUX_INPUT_RE = re.compile(r"\\@input\s*\{(?P<target>[^{}]+)\}")
BIBTEX_AUX_INPUT_COMMAND_RE = re.compile(r"\\@input(?![A-Za-z@])")
EVIDENCE_REF_RE = re.compile(r"@evidence\[\[[^\]]+\]\]")
PROMPT_ARTIFACT_RE = re.compile(r".+_prompt(?:_round_\d+(?:_contract_retry_\d+)?)?\.md$")
AGENT_SKILL_DIR_PREFIXES = (
    (".claude", "skills"),
    (".codex", "skills"),
    (".cursor", "skills"),
    (".opencode", "skills"),
    (".gemini", "skills"),
    (".pi", "agent", "skills"),
    (".agents", "skills"),
)


def read_json(
    source: PaperTree | Path | str, rel: str | None = None, *, required: bool = True
) -> dict[str, Any]:
    """Read a JSON object from a paper tree (or, with a single path, from the host).

    ``read_json(tree, "paper_data.json")`` reads a file of a tree; ``read_json(path)`` reads a
    host file (the signature of upstream's ``read_json``, which the ``validate-paper`` skill's
    script calls).
    """
    if rel is None:
        path = Path(source)  # type: ignore[arg-type]
        if not path.exists():
            if required:
                raise RuntimeError(f"Missing required artifact: {path}")
            return {}
        raw = path.read_bytes()
        label = str(path)
    else:
        tree = as_tree(source)
        data = read_regular_nofollow(tree, rel)
        if data is None:
            if required:
                raise RuntimeError(f"Missing required artifact: {rel}")
            return {}
        raw = data
        label = rel
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"Invalid JSON in {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in {label}")
    return payload


def extract_bibtex_keys(bibtex_text: str) -> set[str]:
    """Extract bibtex keys."""
    return {
        match.group(1).strip()
        for match in CITATION_KEY_RE.finditer(bibtex_text)
        if match.group(1).strip()
    }


def extract_tex_citations(tex_text: str) -> set[str]:
    """Extract tex citations."""
    keys: set[str] = set()
    for match in TEX_CITATION_RE.finditer(tex_text):
        raw = match.group(1).strip()
        if not raw:
            continue
        for part in raw.split(","):
            key = part.strip()
            if key:
                keys.add(key)
    return keys


def missing_required_sections(tex_text: str) -> list[str]:
    """Handle missing required sections logic for this module."""
    missing: list[str] = []
    for section, pattern in SECTION_PATTERNS.items():
        if not pattern.search(tex_text):
            missing.append(section)
    return missing


def _strip_tex_comments(tex_text: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""
    cleaned_lines: list[str] = []
    for line in tex_text.splitlines():
        kept: list[str] = []
        i = 0
        while i < len(line):
            if line.startswith(r"\verb", i):
                delimiter_index = i + len(r"\verb")
                if delimiter_index < len(line) and line[delimiter_index] == "*":
                    delimiter_index += 1
                if (
                    delimiter_index < len(line)
                    and not line[delimiter_index].isspace()
                    and not line[delimiter_index].isalpha()
                ):
                    closing_index = line.find(line[delimiter_index], delimiter_index + 1)
                    if closing_index >= 0:
                        kept.append(line[i : closing_index + 1])
                        i = closing_index + 1
                        continue
            char = line[i]
            if char == "%" and not _is_escaped_tex_character(line, i):
                break
            kept.append(char)
            i += 1
        cleaned_lines.append("".join(kept))
    return "\n".join(cleaned_lines)


def _is_escaped_tex_character(text: str, index: int) -> bool:
    backslash_count = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslash_count += 1
        cursor -= 1
    return backslash_count % 2 == 1


def visible_tex_text(tex_text: str) -> str:
    """Reduce LaTeX source to a closer approximation of visible manuscript text."""
    visible = _strip_tex_comments(tex_text)
    visible = FIGURE_INCLUDE_RE.sub(" ", visible)
    visible = re.sub(
        r"\\(?:bibliography|bibliographystyle|label|ref|eqref|pageref|cite[a-zA-Z*]*)\*?(?:\[[^\]]*\])?\{[^}]*\}",
        " ",
        visible,
    )
    visible = re.sub(r"\\(?:input|include)\{[^}]*\}", " ", visible)
    visible = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", visible)
    visible = re.sub(r"[{}]", " ", visible)
    visible = re.sub(r"\s+", " ", visible)
    return visible.strip()


def _visible_tex_text(tex_text: str) -> str:
    """Backward-compatible private alias for visible manuscript text extraction."""
    return visible_tex_text(tex_text)


def _extract_tex_command_body(tex_text: str, command: str) -> str:
    """Extract a LaTeX command body with balanced-brace parsing."""
    match = re.search(rf"\\{re.escape(command)}\s*\{{", tex_text)
    if match is None:
        return ""

    result = _extract_balanced_tex_body(tex_text, match.end())
    return result[0].strip() if result is not None else ""


def _extract_balanced_tex_body(tex_text: str, start: int) -> tuple[str, int] | None:
    """Extract content after an opening brace and return its closing position."""
    depth = 1
    idx = start
    while idx < len(tex_text):
        char = tex_text[idx]
        if char == "{" and not _is_escaped_tex_character(tex_text, idx):
            depth += 1
        elif char == "}" and not _is_escaped_tex_character(tex_text, idx):
            depth -= 1
            if depth == 0:
                return tex_text[start:idx], idx + 1
        idx += 1
    return None


def _extract_tex_title(tex_text: str) -> str:
    return _extract_tex_command_body(tex_text, "title")


def _extract_tex_section_body(tex_text: str, heading: str) -> str:
    pattern = re.compile(r"\\section\*?\{(?P<title>[^}]*)\}", re.IGNORECASE)
    matches = list(pattern.finditer(tex_text))
    aliases = {
        "introduction": ("introduction", "introduction and motivation"),
        "discussion": ("discussion", "general discussion"),
        "conclusion": ("conclusion", "conclusions"),
    }
    allowed_titles = aliases.get(heading.strip().lower(), (heading.strip().lower(),))
    for idx, match in enumerate(matches):
        normalized_title = match.group("title").strip().lower()
        if not any(alias in normalized_title for alias in allowed_titles):
            continue
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(tex_text)
        return tex_text[start:end].strip()
    return ""


def _contains_any_term(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _normalize_tex_whitespace(text: str) -> str:
    return " ".join(text.split())
