"""System TeX inputs linked from one approved tree into another are approved, as on Debian."""

from __future__ import annotations

import importlib.util
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest

import langgraph_agentic_scientist
from langgraph_agentic_scientist.paper.validation import latex

SCRIPT = (
    Path(langgraph_agentic_scientist.__file__).parent
    / "skills/validate-paper/scripts/validate_compile_inputs.py"
)


def _trees(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    texmf = tmp_path / "usr/share/texmf"
    texlive = tmp_path / "usr/share/texlive"
    elsewhere = tmp_path / "home/user"
    (texlive / "texmf-dist/web2c").mkdir(parents=True)
    (texmf / "web2c").mkdir(parents=True)
    elsewhere.mkdir(parents=True)
    (texlive / "texmf-dist/web2c/texmf.cnf").write_text("% cnf\n")
    (elsewhere / "evil.sty").write_text("% sty\n")
    linked = texmf / "web2c/texmf.cnf"
    linked.symlink_to("../../texlive/texmf-dist/web2c/texmf.cnf")
    escaping = texmf / "web2c/evil.sty"
    escaping.symlink_to(elsewhere / "evil.sty")
    return texmf, texlive, linked, escaping


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_compile_inputs", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_validator_approves_a_link_between_system_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texmf, texlive, linked, escaping = _trees(tmp_path)
    monkeypatch.setattr(latex, "LATEX_SYSTEM_INPUT_ROOTS", (texlive, texmf))
    assert latex._is_approved_latex_system_input(PurePosixPath(linked.as_posix()))
    assert not latex._is_approved_latex_system_input(PurePosixPath(escaping.as_posix()))


def test_the_staged_script_approves_a_link_between_system_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texmf, texlive, linked, escaping = _trees(tmp_path)
    script = _script()
    monkeypatch.setattr(script, "LATEX_SYSTEM_INPUT_ROOTS", (texlive, texmf))
    assert script._is_approved_latex_system_input(linked)
    assert not script._is_approved_latex_system_input(escaping)
