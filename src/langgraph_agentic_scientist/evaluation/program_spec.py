"""``program_spec.toml``: the specification of a candidate program (upstream schema 1).

Port of upstream ``src/investigation/program_spec.py``. Parsing and entrypoint resolution work on
text and POSIX paths, so that a spec can be read through any
:class:`~langgraph_agentic_scientist.workspace.Workspace`.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..core.errors import ProgramSpecError
from ..core.state import StateModel

if TYPE_CHECKING:
    from ..workspace.protocol import Workspace

PROGRAM_SPEC_FILENAME = "program_spec.toml"
PROGRAM_SPEC_SCHEMA_VERSION = 1
DEFAULT_ENTRYPOINT = "src/program.py"


class ProgramEntrypoint(StateModel):
    """The program entrypoint: path below ``code_root``, language and type."""

    path: str
    language: str
    type: str


class ProgramCopySpec(StateModel):
    """Copy rules applied by the trusted clone."""

    exclude: tuple[str, ...] = ()
    include: tuple[str, ...] = ()


class ProgramIntegrationSpec(StateModel):
    """Integration metadata of a program."""

    kind: str
    notes: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class ProgramSpec(StateModel):
    """A validated ``program_spec.toml``."""

    spec_schema_version: int = Field(alias="schema_version")
    code_root: str
    entrypoint: ProgramEntrypoint
    copy_rules: ProgramCopySpec = Field(alias="copy")
    integration: ProgramIntegrationSpec


def default_program_spec() -> ProgramSpec:
    """Return the canonical v1 default spec."""
    return ProgramSpec(
        spec_schema_version=PROGRAM_SPEC_SCHEMA_VERSION,
        code_root="src",
        entrypoint=ProgramEntrypoint(path="program.py", language="python", type="script"),
        copy_rules=ProgramCopySpec(
            exclude=(
                "**/__pycache__/**",
                "**/*.pyc",
                ".venv/**",
                "node_modules/**",
                ".git/**",
                ".mypy_cache/**",
                ".pytest_cache/**",
                ".ruff_cache/**",
                ".tox/**",
                ".nox/**",
                "dist/**",
                "build/**",
            ),
            include=(),
        ),
        integration=ProgramIntegrationSpec(kind="generic", notes="", extra={}),
    )


def render_default_program_spec_toml() -> str:
    """Render the canonical v1 default specification as TOML (upstream byte layout)."""
    spec = default_program_spec()

    def quote(value: str) -> str:
        return json.dumps(value, ensure_ascii=False)

    lines = [
        f"schema_version = {spec.spec_schema_version}",
        f"code_root = {quote(spec.code_root)}",
        "",
        "[entrypoint]",
        f"path = {quote(spec.entrypoint.path)}",
        f"language = {quote(spec.entrypoint.language)}",
        f"type = {quote(spec.entrypoint.type)}",
        "",
        "[copy]",
        "exclude = [",
        *(f"  {quote(pattern)}," for pattern in spec.copy_rules.exclude),
        "]",
        "include = [",
        *(f"  {quote(pattern)}," for pattern in spec.copy_rules.include),
        "]",
        "",
        "[integration]",
        f"kind = {quote(spec.integration.kind)}",
        f"notes = {quote(spec.integration.notes)}",
        "",
    ]
    return "\n".join(lines)


def _expect_str(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ProgramSpecError(f"{label} must be a string")
    stripped = value.strip()
    if not stripped:
        raise ProgramSpecError(f"{label} must be non-empty")
    return stripped


def _expect_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProgramSpecError(f"{label} must be an integer")
    return value


def _expect_str_list(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProgramSpecError(f"{label} must be an array of strings")
    out: list[str] = []
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            raise ProgramSpecError(f"{label}[{idx}] must be a string")
        stripped = item.strip()
        if stripped:
            out.append(stripped)
    return tuple(out)


def _ensure_safe_relpath(raw: str, *, label: str) -> str:
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise ProgramSpecError(f"{label} must be a relative path (got absolute path)")
    if any(part == ".." for part in path.parts):
        raise ProgramSpecError(f"{label} must not contain '..'")
    if raw.strip() in (".", ""):
        raise ProgramSpecError(f"{label} must not be '.' or empty")
    return path.as_posix()


def parse_program_spec(text: str, *, source: str = PROGRAM_SPEC_FILENAME) -> ProgramSpec:
    """Parse and validate the text of a ``program_spec.toml`` (upstream rules and messages)."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ProgramSpecError(f"Failed to parse {source}: {exc}") from exc
    schema_version = _expect_int(data.get("schema_version"), label="schema_version")
    if schema_version != PROGRAM_SPEC_SCHEMA_VERSION:
        raise ProgramSpecError(
            f"Unsupported schema_version={schema_version} (expected {PROGRAM_SPEC_SCHEMA_VERSION})"
        )
    code_root = _ensure_safe_relpath(
        _expect_str(data.get("code_root"), label="code_root"), label="code_root"
    )
    if code_root != "src":
        raise ProgramSpecError("code_root must be 'src' for schema_version=1")
    entrypoint = data.get("entrypoint")
    if not isinstance(entrypoint, dict):
        raise ProgramSpecError("entrypoint must be a TOML table")
    entry_path = _ensure_safe_relpath(
        _expect_str(entrypoint.get("path"), label="entrypoint.path"), label="entrypoint.path"
    )
    entry_language = _expect_str(entrypoint.get("language"), label="entrypoint.language")
    entry_type = _expect_str(entrypoint.get("type"), label="entrypoint.type").lower()
    if entry_type not in ("script", "module"):
        raise ProgramSpecError("entrypoint.type must be 'script' or 'module'")
    copy_spec = data.get("copy")
    if not isinstance(copy_spec, dict):
        raise ProgramSpecError("copy must be a TOML table")
    exclude = _expect_str_list(copy_spec.get("exclude"), label="copy.exclude")
    include = _expect_str_list(copy_spec.get("include"), label="copy.include")
    integration = data.get("integration")
    if not isinstance(integration, dict):
        raise ProgramSpecError("integration must be a TOML table")
    kind = _expect_str(integration.get("kind"), label="integration.kind")
    notes = str(integration.get("notes") or "")
    extra = {k: v for k, v in integration.items() if k not in ("kind", "notes")}
    return ProgramSpec(
        spec_schema_version=schema_version,
        code_root=code_root,
        entrypoint=ProgramEntrypoint(path=entry_path, language=entry_language, type=entry_type),
        copy_rules=ProgramCopySpec(exclude=exclude, include=include),
        integration=ProgramIntegrationSpec(kind=kind, notes=notes, extra=extra),
    )


def entrypoint_relpath(spec: ProgramSpec) -> str:
    """Return the entrypoint relative to the program directory, refusing escapes.

    Mirrors upstream ``resolve_entrypoint``: ``code_root`` must stay under the program directory
    and the entrypoint under ``code_root``.
    """
    root_parts = [part for part in PurePosixPath(spec.code_root).parts if part != "."]
    if any(part == ".." for part in root_parts):
        raise ProgramSpecError("code_root escapes program_dir")
    entry_parts = [part for part in PurePosixPath(spec.entrypoint.path).parts if part != "."]
    if not entry_parts or any(part == ".." for part in entry_parts):
        raise ProgramSpecError("Entrypoint escapes code_root")
    return "/".join([*root_parts, *entry_parts])


async def load_program_spec(workspace: Workspace, program_dir: str) -> ProgramSpec:
    """Read and validate ``<program_dir>/program_spec.toml`` through a workspace."""
    path = (
        f"{program_dir.rstrip('/')}/{PROGRAM_SPEC_FILENAME}"
        if program_dir
        else PROGRAM_SPEC_FILENAME
    )
    entry = await workspace.stat(path)
    if entry is None:
        raise ProgramSpecError(f"Missing {PROGRAM_SPEC_FILENAME} at {path}")
    if entry.kind != "file":
        raise ProgramSpecError(f"{path} must be a file")
    try:
        text = (await workspace.read_bytes(path)).decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ProgramSpecError(f"Failed to parse {path}: {exc}") from exc
    return parse_program_spec(text, source=path)


async def resolve_entrypoint(workspace: Workspace, program_dir: str) -> str:
    """Return the entrypoint of a program relative to its directory, checking it is a file."""
    spec = await load_program_spec(workspace, program_dir)
    relative = entrypoint_relpath(spec)
    path = f"{program_dir.rstrip('/')}/{relative}" if program_dir else relative
    entry = await workspace.stat(path)
    if entry is None or entry.kind != "file":
        raise ProgramSpecError(f"Program entrypoint must be a real file: {path}")
    return relative


__all__ = [
    "DEFAULT_ENTRYPOINT",
    "PROGRAM_SPEC_FILENAME",
    "PROGRAM_SPEC_SCHEMA_VERSION",
    "ProgramCopySpec",
    "ProgramEntrypoint",
    "ProgramIntegrationSpec",
    "ProgramSpec",
    "default_program_spec",
    "entrypoint_relpath",
    "load_program_spec",
    "parse_program_spec",
    "render_default_program_spec_toml",
    "resolve_entrypoint",
]
