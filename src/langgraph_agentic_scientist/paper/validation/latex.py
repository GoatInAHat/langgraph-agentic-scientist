"""LaTeX compilation and the validation of its inputs.

Upstream ``compile_latex`` and the TeX and BibTeX dependency checks of ``paper_step.py``: the
bibliography must be the canonical ``references.bib``, local styles and packages must not come
from the staged agent skills, every input the compiler records (``-recorder``) must be inside the
paper workspace or an approved TeX root, and the compile runs four commands (``pdflatex``,
``bibtex``, ``pdflatex``, ``pdflatex``) without shell escape.

The commands run through :meth:`~langgraph_agentic_scientist.Workspace.run`; the checks read a
:class:`~langgraph_agentic_scientist.paper.validation.files.PaperTree`. Upstream's containerised
compile (``container_image``) is the caller's workspace: a container workspace runs the same
commands inside its container.
"""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from ...workspace.paths import join
from ...workspace.protocol import Workspace
from .files import PaperTree, is_regular_nofollow, lexists, read_text, resolve_inside
from .supplement import mask_tex_verbatim, strip_false_conditionals
from .tex import (
    AGENT_SKILL_DIR_PREFIXES,
    BIBTEX_AUX_DATA_COMMAND_RE,
    BIBTEX_AUX_DATA_RE,
    BIBTEX_AUX_INPUT_COMMAND_RE,
    BIBTEX_AUX_INPUT_RE,
    BIBTEX_AUX_STYLE_COMMAND_RE,
    BIBTEX_AUX_STYLE_RE,
    TEX_BIB_RESOURCE_COMMAND_RE,
    TEX_BIB_RESOURCE_RE,
    TEX_BIBLIOGRAPHY_COMMAND_RE,
    TEX_BIBLIOGRAPHY_RE,
    TEX_BIBLIOGRAPHY_STYLE_COMMAND_RE,
    TEX_BIBLIOGRAPHY_STYLE_RE,
    _strip_tex_comments,
)

LATEX_SYSTEM_INPUT_ROOTS = tuple(
    Path(path)
    for path in (
        "/etc/texmf",
        "/opt/texlive",
        "/usr/local/share/miktex-texmf",
        "/usr/local/share/texmf",
        "/usr/local/texlive",
        "/usr/share/texlive",
        "/usr/share/texmf",
        "/usr/share/texmf-dist",
        "/var/lib/miktex-texmf",
        "/var/lib/texmf",
        "/nix/store",
    )
)

LATEX_ENV_ALLOWLIST = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "SOURCE_DATE_EPOCH",
    "FORCE_SOURCE_DATE",
)


class PaperValidationInfrastructureError(RuntimeError):
    """Raised when paper validation cannot run because its tooling is unavailable."""


def default_latex_environment() -> dict[str, str]:
    """Return upstream's clean compile environment from the host's environment.

    The allowlisted variables of the current process, with ``HOME`` and ``TEXMFHOME`` pointed at
    ``/nonexistent`` so that inherited Kpathsea overrides and user TeX trees cannot change the
    compile. A caller whose workspace runs elsewhere (a container, a remote sandbox) passes its
    own environment to the toolchain.
    """
    env = {name: os.environ[name] for name in LATEX_ENV_ALLOWLIST if name in os.environ}
    env["HOME"] = "/nonexistent"
    env["TEXMFHOME"] = "/nonexistent"
    return env


def validate_latex_recorder_inputs(tree: PaperTree, *, root: str, recorder_text: str) -> None:
    """Reject compiled TeX dependencies that enter staged agent skill trees.

    ``root`` is the paper directory as the compiler saw it (the workspace's ``host_path``);
    ``recorder_text`` is the content of the ``.fls`` file.
    """
    recorder_lines = recorder_text.splitlines()
    recorded_pwd = next(
        (
            PurePosixPath(line.removeprefix("PWD ").strip())
            for line in recorder_lines
            if line.startswith("PWD ") and line.removeprefix("PWD ").strip()
        ),
        PurePosixPath(root),
    )
    root_path = PurePosixPath(root)
    for line in recorder_lines:
        if not line.startswith("INPUT "):
            continue
        input_text = line.removeprefix("INPUT ").strip()
        if not input_text:
            continue
        input_path = PurePosixPath(input_text)
        relative_candidate: PurePosixPath | None = None
        if input_path.is_absolute():
            input_path = PurePosixPath(posixpath.normpath(input_path.as_posix()))
            for workspace_root in (root_path, recorded_pwd):
                try:
                    relative_candidate = input_path.relative_to(workspace_root)
                    break
                except ValueError:
                    continue
            if relative_candidate is None and _is_approved_latex_system_input(input_path):
                continue
        else:
            recorded_candidate = PurePosixPath(
                posixpath.normpath((recorded_pwd / input_path).as_posix())
            )
            try:
                relative_candidate = recorded_candidate.relative_to(recorded_pwd)
            except ValueError:
                relative_candidate = None
        if relative_candidate is None:
            raise RuntimeError(
                f"compiled paper dependency is outside approved TeX roots: {input_path.as_posix()}"
            )
        workspace_candidates = [relative_candidate]
        if lexists(tree, relative_candidate.as_posix()):
            resolved_relative = resolve_inside(tree, relative_candidate.as_posix())
            if resolved_relative is None:
                raise RuntimeError(
                    "compiled paper dependency escapes the paper workspace: "
                    f"{relative_candidate.as_posix()}"
                )
            workspace_candidates.append(PurePosixPath(resolved_relative))
        blocked_candidate = next(
            (
                candidate
                for candidate in workspace_candidates
                if any(
                    candidate.parts[: len(prefix)] == prefix for prefix in AGENT_SKILL_DIR_PREFIXES
                )
            ),
            None,
        )
        if blocked_candidate is not None:
            raise RuntimeError(
                "compiled paper must not read staged agent skill files: "
                f"{blocked_candidate.as_posix()}"
            )


def _is_approved_latex_system_input(input_path: PurePosixPath) -> bool:
    host_path = Path(input_path.as_posix())
    for system_root in LATEX_SYSTEM_INPUT_ROOTS:
        if not host_path.is_relative_to(system_root):
            continue
        if not host_path.exists():
            return True
        try:
            resolved_input = host_path.resolve(strict=True)
        except OSError:
            return False
        return any(
            resolved_input.is_relative_to(root.resolve())
            for root in LATEX_SYSTEM_INPUT_ROOTS
            if root.exists()
        )
    return False


def validate_bibtex_dependencies(tree: PaperTree, tex_text: str) -> None:
    """Restrict BibTeX data and local styles to trusted manuscript files."""
    structural_tex = strip_false_conditionals(mask_tex_verbatim(_strip_tex_comments(tex_text)))
    bibliography_matches = list(TEX_BIBLIOGRAPHY_RE.finditer(structural_tex))
    resource_matches = list(TEX_BIB_RESOURCE_RE.finditer(structural_tex))
    style_matches = list(TEX_BIBLIOGRAPHY_STYLE_RE.finditer(structural_tex))
    _reject_unparsed_tex_commands(
        structural_tex, TEX_BIBLIOGRAPHY_COMMAND_RE, bibliography_matches, "bibliography"
    )
    _reject_unparsed_tex_commands(
        structural_tex, TEX_BIB_RESOURCE_COMMAND_RE, resource_matches, "bibliography resource"
    )
    _reject_unparsed_tex_commands(
        structural_tex,
        TEX_BIBLIOGRAPHY_STYLE_COMMAND_RE,
        style_matches,
        "bibliography style",
    )
    bibliography_targets = [
        target.strip()
        for match in bibliography_matches
        for target in match.group("targets").split(",")
    ]
    bibliography_targets.extend(match.group("target").strip() for match in resource_matches)
    for target in bibliography_targets:
        _validate_canonical_bibliography_target(tree, target)
    for match in style_matches:
        _validate_bibliography_style_target(tree, match.group("target").strip())


def validate_bibtex_aux_dependencies(tree: PaperTree, *, aux_name: str = "paper.aux") -> None:
    """Validate the exact bibliography inputs emitted for BibTeX."""
    aux_texts = _read_bibtex_aux_graph(tree, aux_name)
    data_matches: list[re.Match[str]] = []
    style_matches: list[re.Match[str]] = []
    for aux_text in aux_texts:
        current_data_matches = list(BIBTEX_AUX_DATA_RE.finditer(aux_text))
        current_style_matches = list(BIBTEX_AUX_STYLE_RE.finditer(aux_text))
        _reject_unparsed_tex_commands(
            aux_text,
            BIBTEX_AUX_DATA_COMMAND_RE,
            current_data_matches,
            "BibTeX data",
        )
        _reject_unparsed_tex_commands(
            aux_text,
            BIBTEX_AUX_STYLE_COMMAND_RE,
            current_style_matches,
            "BibTeX style",
        )
        data_matches.extend(current_data_matches)
        style_matches.extend(current_style_matches)
    if len(data_matches) != 1 or len(style_matches) != 1:
        raise RuntimeError("paper.aux must declare exactly one bibliography database and style")
    for target in data_matches[0].group("targets").split(","):
        _validate_canonical_bibliography_target(tree, target.strip())
    _validate_bibliography_style_target(tree, style_matches[0].group("target").strip())


def _relative_inside(path: str) -> str:
    candidate = posixpath.normpath(path)
    if candidate.startswith("/") or candidate == ".." or candidate.startswith("../"):
        raise RuntimeError("BibTeX aux input escapes the paper workspace")
    return candidate


def _read_bibtex_aux_graph(tree: PaperTree, initial_path: str) -> list[str]:
    pending = [initial_path]
    visited: set[str] = set()
    aux_texts: list[str] = []
    while pending:
        candidate = _relative_inside(pending.pop())
        if any(
            PurePosixPath(candidate).parts[: len(prefix)] == prefix
            for prefix in AGENT_SKILL_DIR_PREFIXES
        ):
            raise RuntimeError("BibTeX must not read agent skill aux files")
        if tree.kind(candidate) is None or resolve_inside(tree, candidate) is None:
            raise RuntimeError(f"BibTeX aux input is missing or unsafe: {candidate}")
        if not is_regular_nofollow(tree, candidate):
            if resolve_inside(tree, candidate) != candidate:
                raise RuntimeError(f"BibTeX aux input must not use symlinks: {candidate}")
            raise RuntimeError(f"BibTeX aux input is missing or unsafe: {candidate}")
        if candidate in visited:
            continue
        visited.add(candidate)
        aux_text = _strip_tex_comments(read_text(tree, candidate) or "")
        input_matches = list(BIBTEX_AUX_INPUT_RE.finditer(aux_text))
        _reject_unparsed_tex_commands(
            aux_text, BIBTEX_AUX_INPUT_COMMAND_RE, input_matches, "BibTeX aux input"
        )
        aux_texts.append(aux_text)
        pending.extend(match.group("target").strip() for match in input_matches)
    return aux_texts


def _reject_unparsed_tex_commands(
    tex_text: str,
    command_pattern: re.Pattern[str],
    parsed_matches: list[re.Match[str]],
    label: str,
) -> None:
    parsed_starts = {match.start() for match in parsed_matches}
    if any(match.start() not in parsed_starts for match in command_pattern.finditer(tex_text)):
        raise RuntimeError(f"paper.tex uses a dynamic {label} command")


def _validate_canonical_bibliography_target(tree: PaperTree, target_text: str) -> None:
    if "\\" in target_text or "#" in target_text:
        raise RuntimeError("paper.tex uses a dynamic bibliography target")
    target = PurePosixPath(target_text)
    if not target.suffix:
        target = target.with_suffix(".bib")
    candidate = posixpath.normpath(target.as_posix())
    if candidate != "references.bib":
        raise RuntimeError("paper.tex must use the canonical references.bib bibliography")
    if lexists(tree, candidate) and resolve_inside(tree, candidate) != candidate:
        raise RuntimeError("paper.tex bibliography must not use symlinks")


def _validate_bibliography_style_target(tree: PaperTree, target_text: str) -> None:
    if "\\" in target_text or "#" in target_text:
        raise RuntimeError("paper.tex uses a dynamic bibliography style target")
    target = PurePosixPath(target_text)
    if not target.suffix:
        target = target.with_suffix(".bst")
    candidate = posixpath.normpath(target.as_posix())
    if target.parent == PurePosixPath(".") and not lexists(tree, candidate):
        return
    if candidate.startswith("/") or candidate == ".." or candidate.startswith("../"):
        raise RuntimeError("paper.tex bibliography style escapes the paper workspace")
    if any(
        PurePosixPath(candidate).parts[: len(prefix)] == prefix
        for prefix in AGENT_SKILL_DIR_PREFIXES
    ):
        raise RuntimeError("paper.tex must not use agent skill bibliography styles")
    if tree.kind(candidate) is None:
        raise RuntimeError(f"paper.tex local bibliography style is missing: {candidate}")
    if resolve_inside(tree, candidate) != candidate:
        raise RuntimeError("paper.tex bibliography style must not use symlinks")
    if tree.kind(candidate) != "file":
        raise RuntimeError(f"paper.tex local bibliography style is missing: {candidate}")


LATEX_COMMAND_FLAGS = (
    "-recorder",
    "-no-shell-escape",
    "-interaction=nonstopmode",
    "-halt-on-error",
)
"""The flags of every ``pdflatex`` run (upstream ``compile_latex``)."""

TreeLoader = Callable[[], "Awaitable[PaperTree]"]


def latex_commands(
    tex_name: str, *, pdflatex: str = "pdflatex", bibtex: str = "bibtex"
) -> list[list[str]]:
    """Return the four compile commands: ``pdflatex``, ``bibtex``, ``pdflatex``, ``pdflatex``."""
    tex_command = [pdflatex, *LATEX_COMMAND_FLAGS, tex_name]
    return [tex_command, [bibtex, Path(tex_name).stem], tex_command, tex_command]


async def compile_latex(
    workspace: Workspace,
    paper_dir: str,
    *,
    load_tree: TreeLoader,
    tex_name: str = "paper.tex",
    env: Mapping[str, str] | None = None,
    pdflatex: str = "pdflatex",
    bibtex: str = "bibtex",
    timeout: float | None = 1800,
) -> list[dict[str, Any]]:
    """Compile ``paper_dir/tex_name`` and return the log of each command.

    Before ``bibtex`` the emitted ``.aux`` graph is validated, and after each ``pdflatex`` the
    recorded inputs are validated (``load_tree`` returns a fresh tree of the paper directory).

    Raises:
        PaperValidationInfrastructureError: When a command cannot be started.
        RuntimeError: When a command fails or an input is refused.
    """
    commands = latex_commands(tex_name, pdflatex=pdflatex, bibtex=bibtex)
    stem = Path(tex_name).stem
    recorder_rel = f"{stem}.fls"
    recorder_path = join(paper_dir, recorder_rel)
    environment = dict(default_latex_environment() if env is None else env)
    logs: list[dict[str, Any]] = []
    root = workspace.host_path(paper_dir)
    for cmd in commands:
        uses_recorder = "-recorder" in cmd
        if cmd[0] == bibtex:
            validate_bibtex_aux_dependencies(await load_tree(), aux_name=f"{stem}.aux")
        if uses_recorder:
            await workspace.remove(recorder_path)
        try:
            proc = await workspace.run(cmd, cwd=paper_dir, env=environment, timeout=timeout)
        except FileNotFoundError as exc:
            raise PaperValidationInfrastructureError(
                f"Missing required command(s): {cmd[0]}. Install TeX Live (pdflatex + bibtex) "
                "or give the workspace an environment that has them."
            ) from exc
        logs.append(
            {
                "command": cmd,
                "returncode": proc.returncode,
                "stdout_tail": (proc.stdout or "")[-2000:],
                "stderr_tail": (proc.stderr or "")[-2000:],
            }
        )
        if uses_recorder and (proc.returncode == 0 or await workspace.exists(recorder_path)):
            if not await workspace.exists(recorder_path):
                raise RuntimeError(f"LaTeX recorder output missing: {recorder_rel}")
            recorder_text = (await workspace.read_bytes(recorder_path)).decode(
                "utf-8", errors="replace"
            )
            validate_latex_recorder_inputs(
                await load_tree(), root=root, recorder_text=recorder_text
            )
        if proc.returncode != 0:
            raise RuntimeError(f"LaTeX command failed: {' '.join(cmd)}")
    return logs


__all__ = [
    "LATEX_COMMAND_FLAGS",
    "LATEX_ENV_ALLOWLIST",
    "LATEX_SYSTEM_INPUT_ROOTS",
    "PaperValidationInfrastructureError",
    "compile_latex",
    "default_latex_environment",
    "latex_commands",
    "validate_bibtex_aux_dependencies",
    "validate_bibtex_dependencies",
    "validate_latex_recorder_inputs",
]
