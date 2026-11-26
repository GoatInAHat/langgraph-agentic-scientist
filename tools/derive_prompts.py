#!/usr/bin/env python3
"""Build the package's prompt and skill files from the vendored upstream files.

Every prompt, skill and workflow text of the package is derived from an upstream file under
``third_party/ai-scientist-v3/`` by the changes declared in
``src/langgraph_agentic_scientist/prompts/changes/*.toml``. Nothing else may differ from upstream,
and this tool is the only way package prompt files are made. The files under ``prompts/harness/``
are the package's own texts and are exempt.

A changes file holds two kinds of entries::

    [[extract]]                     # a text that lives in upstream code
    upstream = "src/investigation/reset_sequence.py"
    lines = "47-51"                 # the string expression lies within these lines
    output = "extracted/core/reset_sequence_diagnostic.md"
    names = { "len(observed_reset_seeds)" = "COUNT" }   # f-string fields -> {{PLACEHOLDERS}}
    reason = "..."

    [[file]]                        # a package file derived from an upstream (or extracted) file
    upstream = "src/agents/skills/memory-log/SKILL.md"
    target = "skills/memory-log/SKILL.md"
    reason = "..."

    [[file.change]]                 # one exact replacement; ``old`` must occur ``count`` times
    kind = "A4"                     # A1-A8 (DESIGN section 4.0) or "other"
    old = "..."
    new = "..."
    count = 1
    reason = "..."

An extraction writes the value of the string (quoting removed; f-string fields become
``{{NAME}}`` placeholders, named after the field's variable or by ``names``) after a one-line
header naming its source; deriving a package file from an extracted file drops that header.

Usage::

    python tools/derive_prompts.py           # write the package files
    python tools/derive_prompts.py --check   # verify them (CI and the tests)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party" / "ai-scientist-v3"
PACKAGE = REPO_ROOT / "src" / "langgraph_agentic_scientist"
CHANGES_DIR = PACKAGE / "prompts" / "changes"
KINDS = {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "other"}
EXTRACT_HEADER_PREFIX = "<!-- Extracted from "
# Package directories whose every file must be derived, and the exemptions inside them.
DERIVED_ROOTS = ("prompts", "skills")
EXEMPT_PREFIXES = ("prompts/changes/", "prompts/harness/")
EXEMPT_FILES = ("prompts/__init__.py",)


@dataclass
class Change:
    """One declared replacement."""

    kind: str
    old: str
    new: str
    count: int
    reason: str


@dataclass
class FileEntry:
    """One derived package file."""

    upstream: str
    target: str
    reason: str
    source: str
    changes: list[Change] = field(default_factory=list)


@dataclass
class Extraction:
    """One text extracted from upstream code."""

    upstream: str
    lines: tuple[int, int]
    output: str
    names: dict[str, str]
    reason: str
    source: str


class DerivationError(Exception):
    """A changes file is inconsistent with the upstream files."""


def _safe(base: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DerivationError(f"unsafe path in changes file: {relative!r}")
    return base.joinpath(*path.parts)


def load_changes(changes_dir: Path = CHANGES_DIR) -> tuple[list[Extraction], list[FileEntry]]:
    """Read every changes file."""
    extractions: list[Extraction] = []
    files: list[FileEntry] = []
    for toml_path in sorted(changes_dir.glob("*.toml")):
        source = toml_path.name
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        for raw in data.get("extract", []):
            start, _, end = str(raw["lines"]).partition("-")
            extractions.append(
                Extraction(
                    upstream=str(raw["upstream"]),
                    lines=(int(start), int(end or start)),
                    output=str(raw["output"]),
                    names={str(k): str(v) for k, v in dict(raw.get("names", {})).items()},
                    reason=str(raw.get("reason", "")),
                    source=source,
                )
            )
        for raw in data.get("file", []):
            changes = []
            for change in raw.get("change", []):
                kind = str(change.get("kind", ""))
                if kind not in KINDS:
                    raise DerivationError(f"{source}: unknown change kind {kind!r}")
                if not str(change.get("reason", "")).strip():
                    raise DerivationError(f"{source}: a change to {raw['target']} has no reason")
                changes.append(
                    Change(
                        kind=kind,
                        old=str(change["old"]),
                        new=str(change["new"]),
                        count=int(change.get("count", 1)),
                        reason=str(change["reason"]),
                    )
                )
            if not str(raw.get("reason", "")).strip():
                raise DerivationError(f"{source}: file {raw['target']} has no reason")
            files.append(
                FileEntry(
                    upstream=str(raw["upstream"]),
                    target=str(raw["target"]),
                    reason=str(raw["reason"]),
                    source=source,
                    changes=changes,
                )
            )
    return extractions, files


def _commit() -> str:
    return (THIRD_PARTY / "UPSTREAM_COMMIT").read_text(encoding="utf-8").strip()


def _string_nodes(tree: ast.AST, start: int, end: int) -> list[ast.expr]:
    candidates: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant | ast.JoinedStr | ast.BinOp):
            continue
        if isinstance(node, ast.Constant) and not isinstance(node.value, str):
            continue
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.Add):
            continue
        node_end = node.end_lineno or node.lineno
        if node.lineno >= start and node_end <= end:
            candidates.append(node)
    outermost = [
        node
        for node in candidates
        if not any(
            other is not node
            and other.lineno <= node.lineno
            and (other.end_lineno or 0) >= (node.end_lineno or 0)
            and node in list(ast.walk(other))
            for other in candidates
        )
    ]
    return outermost


def _render_string(node: ast.expr, names: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _render_string(node.left, names) + _render_string(node.right, names)
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                if value.conversion != -1 or value.format_spec is not None:
                    raise DerivationError(
                        f"f-string field with conversion or format: {ast.unparse(value)}"
                    )
                expression = ast.unparse(value.value)
                if expression in names:
                    parts.append("{{" + names[expression] + "}}")
                elif isinstance(value.value, ast.Name):
                    parts.append("{{" + value.value.id.upper() + "}}")
                else:
                    raise DerivationError(f"name the f-string field {expression!r} in `names`")
            else:
                raise DerivationError(f"unsupported f-string part: {ast.dump(value)}")
        return "".join(parts)
    raise DerivationError(f"not a string expression: {ast.dump(node)}")


def extract(extraction: Extraction) -> str:
    """Return the content of an extracted file (header line and string value)."""
    path = _safe(THIRD_PARTY, extraction.upstream)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    start, end = extraction.lines
    nodes = _string_nodes(tree, start, end)
    if len(nodes) != 1:
        raise DerivationError(
            f"{extraction.source}: expected one string expression in "
            f"{extraction.upstream}:{start}-{end}, found {len(nodes)}"
        )
    value = _render_string(nodes[0], extraction.names)
    header = (
        f"{EXTRACT_HEADER_PREFIX}{extraction.upstream} lines {start}-{end} of AI-Scientist-v3 "
        f"{_commit()[:12]} by tools/derive_prompts.py; deriving a package file drops this line. -->"
    )
    return f"{header}\n{value}"


def derive(entry: FileEntry) -> bytes:
    """Return the bytes of a package file derived from its upstream file."""
    source_path = _safe(THIRD_PARTY, entry.upstream)
    data = source_path.read_bytes()
    if entry.upstream.startswith("extracted/"):
        text = data.decode("utf-8")
        first, _, rest = text.partition("\n")
        if not first.startswith(EXTRACT_HEADER_PREFIX):
            raise DerivationError(f"{entry.upstream} lacks the extraction header")
        data = rest.encode("utf-8")
    if not entry.changes:
        return data
    text = data.decode("utf-8")
    for change in entry.changes:
        found = text.count(change.old)
        if found != change.count:
            raise DerivationError(
                f"{entry.source}: in {entry.upstream} the old string of a {change.kind} change "
                f"occurs {found} time(s), declared {change.count}: {change.old[:80]!r}"
            )
        text = text.replace(change.old, change.new)
    return text.encode("utf-8")


def _manifest_digests() -> dict[str, str]:
    data = tomllib.loads((THIRD_PARTY / "MANIFEST.toml").read_text(encoding="utf-8"))
    return {str(item["path"]): str(item["sha256"]) for item in data.get("file", [])}


def _package_files() -> list[str]:
    found: list[str] = []
    for root in DERIVED_ROOTS:
        base = PACKAGE / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(PACKAGE).as_posix()
            if relative in EXEMPT_FILES or relative.startswith(EXEMPT_PREFIXES):
                continue
            found.append(relative)
    return found


def run(*, check: bool) -> list[str]:
    """Derive (or check) every declared file; return the list of problems."""
    problems: list[str] = []
    try:
        extractions, files = load_changes()
    except (DerivationError, KeyError, ValueError) as exc:
        return [str(exc)]
    manifest = _manifest_digests()
    for extraction in extractions:
        if extraction.upstream not in manifest:
            problems.append(f"{extraction.source}: {extraction.upstream} is not vendored")
            continue
        try:
            content = extract(extraction)
        except (DerivationError, OSError, SyntaxError) as exc:
            problems.append(str(exc))
            continue
        output = _safe(THIRD_PARTY, extraction.output)
        if not extraction.output.startswith("extracted/"):
            problems.append(f"{extraction.source}: extraction output must be under extracted/")
            continue
        if check:
            if not output.is_file() or output.read_text(encoding="utf-8") != content:
                problems.append(f"extracted file differs from its extraction: {extraction.output}")
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(content, encoding="utf-8")
    targets: dict[str, str] = {}
    for entry in files:
        if entry.target in targets:
            problems.append(
                f"{entry.target} is declared twice ({targets[entry.target]}, {entry.source})"
            )
            continue
        targets[entry.target] = entry.source
        if not entry.upstream.startswith("extracted/"):
            if entry.upstream not in manifest:
                problems.append(f"{entry.source}: {entry.upstream} is not vendored")
                continue
            digest = hashlib.sha256(_safe(THIRD_PARTY, entry.upstream).read_bytes()).hexdigest()
            if digest != manifest[entry.upstream]:
                problems.append(f"vendored file differs from upstream: {entry.upstream}")
                continue
        try:
            content = derive(entry)
        except (DerivationError, OSError, UnicodeError) as exc:
            problems.append(str(exc))
            continue
        target = _safe(PACKAGE, entry.target)
        if check:
            if not target.is_file() or target.read_bytes() != content:
                problems.append(f"package file differs from its derivation: {entry.target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    for relative in _package_files():
        if relative not in targets:
            problems.append(f"package file is not declared in any changes file: {relative}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args(argv)
    problems = run(check=args.check)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print("prompt derivation " + ("verified" if args.check else "written"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
