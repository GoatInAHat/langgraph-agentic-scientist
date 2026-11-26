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

``kind`` selects the shape of the text (``expression`` when omitted):

* ``expression``: the one string expression within ``lines`` (concatenations and f-strings
  folded); in a file that is not Python the lines are dedented Python code of an embedded heredoc,
  with ``strip_prefix`` and ``strip_suffix`` removed before parsing.
* ``elements``: every string expression within ``lines`` (the elements of a list), joined by
  newlines.
* ``fstring``: the one f-string within ``lines``.
* ``print``: the argument of the one ``print(...)`` statement of embedded Python.
* ``assignment``: the module-level assignment ``name``; ``decode = "base85-bz2"`` decodes the
  value and ``json_key`` then takes one key of the decoded JSON.
* ``dict_value``: the value of ``key`` in the module-level dictionary ``name``.
* ``shell_string``: the quoted value of a shell assignment or string; ``quote`` is ``double``
  (default), ``single`` or ``ansi``; ``strip_prefix`` and ``strip_suffix`` remove the assignment
  and the closing quote; expansions (``$x``, ``${x#y}``) need an entry in ``names``.
* ``shell_concat``: the lines of ``lines`` are ``NAME+=$'...'``, ``NAME+="..."`` or
  ``NAME+="$var"`` appends; the expansions need an entry in ``names``.
* ``heredoc``: the body of a quoted heredoc; the line before ``lines`` opens it and the line
  after closes it.

A shape the extractor does not recognise is an error. A ``[[pin]]`` entry holds an ``extracted``
file and a ``package`` source with the same locator fields; the tool extracts the text from the
package's code and requires it to equal the extracted text, so an in-code constant cannot drift
from upstream.

Usage::

    python tools/derive_prompts.py           # write the package files
    python tools/derive_prompts.py --check   # verify them (CI and the tests)
"""

from __future__ import annotations

import argparse
import ast
import base64
import bz2
import hashlib
import json
import re
import sys
import textwrap
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
    """One text extracted from code (upstream's, or the package's for a pin)."""

    upstream: str
    lines: tuple[int, int]
    output: str
    names: dict[str, str]
    reason: str
    source: str
    kind: str = "expression"
    name: str = ""
    key: str = ""
    decode: str = ""
    json_key: str = ""
    quote: str = "double"
    strip_prefix: str = ""
    strip_suffix: str = ""


EXTRACT_KINDS = {
    "expression",
    "elements",
    "fstring",
    "print",
    "assignment",
    "dict_value",
    "shell_string",
    "shell_concat",
    "heredoc",
}
NAMED_KINDS = {"assignment", "dict_value"}


class DerivationError(Exception):
    """A changes file is inconsistent with the upstream files."""


def _safe(base: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise DerivationError(f"unsafe path in changes file: {relative!r}")
    return base.joinpath(*path.parts)


def _locator(raw: dict[str, object], source: str, path_key: str, output: str) -> Extraction:
    kind = str(raw.get("kind", "expression"))
    if kind not in EXTRACT_KINDS:
        raise DerivationError(f"{source}: unknown extraction kind {kind!r}")
    start, end = 0, 0
    if kind in NAMED_KINDS:
        if not str(raw.get("name", "")):
            raise DerivationError(f"{source}: a {kind} extraction needs `name`")
    else:
        first, _, last = str(raw["lines"]).partition("-")
        start, end = int(first), int(last or first)
    names_raw = raw.get("names", {})
    return Extraction(
        upstream=str(raw[path_key]),
        lines=(start, end),
        output=output,
        names={str(k): str(v) for k, v in dict(names_raw).items()},  # type: ignore[call-overload]
        reason=str(raw.get("reason", "")),
        source=source,
        kind=kind,
        name=str(raw.get("name", "")),
        key=str(raw.get("key", "")),
        decode=str(raw.get("decode", "")),
        json_key=str(raw.get("json_key", "")),
        quote=str(raw.get("quote", "double")),
        strip_prefix=str(raw.get("strip_prefix", "")),
        strip_suffix=str(raw.get("strip_suffix", "")),
    )


def load_pins(changes_dir: Path = CHANGES_DIR) -> list[Extraction]:
    """Read the pins of every changes file; ``output`` of a pin is its extracted file."""
    pins: list[Extraction] = []
    for toml_path in sorted(changes_dir.glob("*.toml")):
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        for raw in data.get("pin", []):
            pins.append(_locator(raw, toml_path.name, "package", str(raw["extracted"])))
    return pins


def load_changes(changes_dir: Path = CHANGES_DIR) -> tuple[list[Extraction], list[FileEntry]]:
    """Read every changes file."""
    extractions: list[Extraction] = []
    files: list[FileEntry] = []
    for toml_path in sorted(changes_dir.glob("*.toml")):
        source = toml_path.name
        data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        for raw in data.get("extract", []):
            extractions.append(_locator(raw, source, "upstream", str(raw["output"])))
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
    expression = ast.unparse(node)
    if expression in names:
        return "{{" + names[expression] + "}}"
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
                field_expression = ast.unparse(value.value)
                if field_expression in names:
                    parts.append("{{" + names[field_expression] + "}}")
                elif isinstance(value.value, ast.Name):
                    parts.append("{{" + value.value.id.upper() + "}}")
                else:
                    raise DerivationError(
                        f"name the f-string field {field_expression!r} in `names`"
                    )
            else:
                raise DerivationError(f"unsupported f-string part: {ast.dump(value)}")
        return "".join(parts)
    raise DerivationError(f"not a string expression: {expression!r}; name it in `names`")


def _selected(source: str, extraction: Extraction) -> list[str]:
    start, end = extraction.lines
    lines = source.split("\n")
    if start < 1 or end > len(lines) or end < start:
        raise DerivationError(f"{extraction.source}: lines {start}-{end} are outside the file")
    return lines[start - 1 : end]


def _strip(text: str, extraction: Extraction) -> str:
    if extraction.strip_prefix:
        if not text.startswith(extraction.strip_prefix):
            raise DerivationError(
                f"{extraction.source}: {extraction.upstream}:{extraction.lines[0]} does not "
                f"start with {extraction.strip_prefix!r}"
            )
        text = text[len(extraction.strip_prefix) :]
    if extraction.strip_suffix:
        if not text.endswith(extraction.strip_suffix):
            raise DerivationError(
                f"{extraction.source}: {extraction.upstream}:{extraction.lines[1]} does not "
                f"end with {extraction.strip_suffix!r}"
            )
        text = text[: -len(extraction.strip_suffix)]
    return text


def _top_level_value(tree: ast.Module, name: str) -> ast.expr:
    found = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign | ast.AnnAssign)
        and node.value is not None
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    if len(found) != 1:
        raise DerivationError(f"expected one module-level assignment of {name}, found {len(found)}")
    return found[0]


def _decode_value(value: str, extraction: Extraction) -> str:
    if not extraction.decode:
        return value
    if extraction.decode != "base85-bz2":
        raise DerivationError(f"{extraction.source}: unknown decoding {extraction.decode!r}")
    decoded = bz2.decompress(base64.b85decode(value)).decode("utf-8")
    if extraction.json_key:
        return str(json.loads(decoded)[extraction.json_key])
    return decoded


def _python_text(
    source: str, extraction: Extraction, *, embedded: bool
) -> tuple[str, tuple[int, int]]:
    start, end = extraction.lines
    kind = extraction.kind
    if kind in NAMED_KINDS:
        value = _top_level_value(ast.parse(source), extraction.name)
        if kind == "dict_value":
            if not isinstance(value, ast.Dict):
                raise DerivationError(f"{extraction.name} is not a dictionary literal")
            matches = [
                item
                for dict_key, item in zip(value.keys, value.values, strict=True)
                if isinstance(dict_key, ast.Constant) and dict_key.value == extraction.key
            ]
            if len(matches) != 1:
                raise DerivationError(f"{extraction.name} has no single key {extraction.key!r}")
            value = matches[0]
        text = _decode_value(_render_string(value, extraction.names), extraction)
        return text, (value.lineno, value.end_lineno or value.lineno)
    if embedded:
        snippet = _strip(
            textwrap.dedent("\n".join(_selected(source, extraction))).strip(), extraction
        )
        if kind == "print":
            (statement,) = ast.parse(snippet).body
            if not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)):
                raise DerivationError(f"{extraction.source}: lines {start}-{end} are not a call")
            call = statement.value
            if not (isinstance(call.func, ast.Name) and call.func.id == "print"):
                raise DerivationError(f"{extraction.source}: lines {start}-{end} are not print")
            if len(call.args) != 1 or call.keywords:
                raise DerivationError(f"{extraction.source}: print takes one argument here")
            return _render_string(call.args[0], extraction.names), (start, end)
        if kind == "elements":
            listing = ast.parse(f"[\n{snippet}\n]", mode="eval").body
            assert isinstance(listing, ast.List)
            return "\n".join(_render_string(e, extraction.names) for e in listing.elts), (
                start,
                end,
            )
        expression = ast.parse(f"(\n{snippet}\n)", mode="eval").body
        return _render_string(expression, extraction.names), (start, end)
    nodes = _string_nodes(ast.parse(source), start, end)
    if kind == "fstring":
        nodes = [node for node in nodes if isinstance(node, ast.JoinedStr)]
    if kind == "elements":
        if not nodes:
            raise DerivationError(
                f"{extraction.source}: no string in {extraction.upstream}:{start}-{end}"
            )
        nodes.sort(key=lambda node: (node.lineno, node.col_offset))
        return "\n".join(_render_string(node, extraction.names) for node in nodes), (start, end)
    if len(nodes) != 1:
        raise DerivationError(
            f"{extraction.source}: expected one string expression in "
            f"{extraction.upstream}:{start}-{end}, found {len(nodes)}"
        )
    return _render_string(nodes[0], extraction.names), (start, end)


_ANSI_ESCAPES = {"n": "\n", "t": "\t", "\\": "\\", "'": "'", '"': '"'}
_EXPANSION = re.compile(r"\$(?:\{[^}]*\}|[A-Za-z_][A-Za-z0-9_]*)")


def _shell_value(body: str, quote: str, names: dict[str, str], where: str) -> str:
    if quote == "single":
        if "'" in body:
            raise DerivationError(f"{where}: a single quote inside a single-quoted string")
        return body
    out: list[str] = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == "\\":
            following = body[index + 1 : index + 2]
            if quote == "ansi":
                if following not in _ANSI_ESCAPES:
                    raise DerivationError(f"{where}: unsupported ANSI-C escape \\{following}")
                out.append(_ANSI_ESCAPES[following])
            elif following in ('"', "\\", "$", "`"):
                out.append(following)
            else:
                raise DerivationError(f"{where}: unsupported escape \\{following}")
            index += 2
            continue
        if quote == "ansi":
            out.append(char)
            index += 1
            continue
        if char == "`":
            raise DerivationError(f"{where}: an unescaped backtick (command substitution)")
        if char == "$":
            match = _EXPANSION.match(body, index)
            if match is None:
                raise DerivationError(
                    f"{where}: unsupported expansion at {body[index : index + 20]!r}"
                )
            token = match.group(0)
            if token not in names:
                raise DerivationError(f"{where}: name the expansion {token!r} in `names`")
            out.append("{{" + names[token] + "}}")
            index = match.end()
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _shell_text(source: str, extraction: Extraction) -> str:
    start, end = extraction.lines
    where = f"{extraction.source}: {extraction.upstream}:{start}-{end}"
    selected = _selected(source, extraction)
    if extraction.kind == "heredoc":
        if start < 2 or end >= len(source.split("\n")):
            raise DerivationError(f"{where}: the heredoc has no opening or closing line")
        lines = source.split("\n")
        opener = re.search(r"<<-?\s*'([A-Za-z_]+)'\s*$", lines[start - 2])
        if opener is None:
            raise DerivationError(f"{where}: the line before is not a quoted heredoc opener")
        if lines[end] != opener.group(1):
            raise DerivationError(f"{where}: the line after is not the heredoc delimiter")
        return "".join(line + "\n" for line in selected)
    if extraction.kind == "shell_concat":
        out: list[str] = []
        for line in selected:
            text = line.strip()
            match = re.fullmatch(r"\w+\+=(?P<value>\$'.*'|\".*\")", text)
            if match is None:
                raise DerivationError(f"{where}: not an append of a quoted string: {text[:60]!r}")
            value = match.group("value")
            if value.startswith("$'"):
                out.append(_shell_value(value[2:-1], "ansi", extraction.names, where))
            else:
                out.append(_shell_value(value[1:-1], "double", extraction.names, where))
        return "".join(out)
    text = "\n".join(selected)
    if extraction.strip_prefix:
        text = text.lstrip()
    text = _strip(text, extraction)
    return _shell_value(text, extraction.quote, extraction.names, where)


def text_of(extraction: Extraction, root: Path) -> tuple[str, tuple[int, int]]:
    """Return the value that ``extraction`` names in the file under ``root``, and its lines."""
    source = _safe(root, extraction.upstream).read_text(encoding="utf-8")
    if extraction.kind in {"shell_string", "shell_concat", "heredoc"}:
        return _shell_text(source, extraction), extraction.lines
    return _python_text(source, extraction, embedded=not extraction.upstream.endswith(".py"))


def extract(extraction: Extraction) -> str:
    """Return the content of an extracted file (header line and string value)."""
    value, (start, end) = text_of(extraction, THIRD_PARTY)
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
    try:
        pins = load_pins()
    except (DerivationError, KeyError, ValueError) as exc:
        return [*problems, str(exc)]
    for pin in pins:
        if not pin.output.startswith("extracted/"):
            problems.append(f"{pin.source}: a pin names an extracted file, not {pin.output}")
            continue
        try:
            actual, _ = text_of(pin, PACKAGE)
            expected = _safe(THIRD_PARTY, pin.output).read_text(encoding="utf-8")
        except (DerivationError, OSError, SyntaxError) as exc:
            problems.append(f"{pin.source}: {pin.upstream}: {exc}")
            continue
        if actual != expected.partition("\n")[2]:
            problems.append(
                f"package code differs from the extracted text: {pin.upstream} ({pin.output})"
            )
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
