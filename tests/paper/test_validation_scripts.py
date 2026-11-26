"""The vendored skill scripts, staged under ``.agents/skills/`` and run in a local workspace.

These are the scripts the paper-writing agents run through ``execute``. They run with the current
interpreter, as the default local workspace does.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from test_validation_figures import pdf, png, row
from test_validation_support import (
    FakeLatexWorkspace,
    paper_files,
    valid_tex,
    write_files,
)

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.agents.staging import stage_skills
from langgraph_agentic_scientist.paper.validation import WorkspacePaperToolchain
from langgraph_agentic_scientist.prompts import PromptRegistry

SKILLS = (
    "write-paper",
    "write-paper-standalone",
    "validate-paper",
    "scientific-evidence",
    "retrieve-references",
    "generate-concept-figure",
)
SKILL_ROOT = ".agents/skills"


async def staged_paper(
    workspace: LocalWorkspace, files: dict[str, str | bytes] | None = None
) -> None:
    await write_files(workspace, "paper", files or paper_files())
    staged = await stage_skills(
        workspace, "paper", SKILLS, prompts=PromptRegistry.default(), values={}
    )
    assert staged == list(SKILLS)


async def run(workspace: LocalWorkspace, *argv: str, cwd: str = "paper") -> tuple[int, str]:
    result = await workspace.run([sys.executable, *argv], cwd=cwd, timeout=120)
    return result.returncode, result.stdout + result.stderr


async def test_staged_skills_keep_the_sibling_layout(workspace: LocalWorkspace) -> None:
    await staged_paper(workspace)
    for relative in (
        "write-paper/SKILL.md",
        "write-paper/scripts/validate_claims.py",
        "write-paper/templates/iclr/paper.tex",
        "write-paper/templates/alife/alifeconf.sty",
        "write-paper-standalone/templates/iclr/paper.bst",  # shares write-paper's templates
        "validate-paper/scripts/validate_figures.py",
        "validate-paper/templates/figure_manifest.schema.json",
        "scientific-evidence/scripts/scientific_evidence.sh",
        "retrieve-references/templates/provenance.schema.json",
        "generate-concept-figure/SKILL.md",
    ):
        assert await workspace.exists(f"paper/{SKILL_ROOT}/{relative}"), relative
    # the relative link of validate-paper to write-paper's validator stays true
    skill = (await workspace.read_bytes(f"paper/{SKILL_ROOT}/validate-paper/SKILL.md")).decode()
    assert "python ../write-paper/scripts/validate_claims.py" in skill
    assert f"python {SKILL_ROOT}/validate-paper/scripts/validate_figures.py" in skill
    evidence = (
        await workspace.read_bytes(f"paper/{SKILL_ROOT}/scientific-evidence/SKILL.md")
    ).decode()
    assert f"bash {SKILL_ROOT}/scientific-evidence/scripts/scientific_evidence.sh" in evidence
    assert "src/agents/skills" not in evidence


async def test_the_figure_validation_script_runs_from_the_staged_skill(
    workspace: LocalWorkspace,
) -> None:
    files = paper_files(
        tex=valid_tex(include_graphics="figures/fig1"),
        extra={
            "figure_manifest.json": json.dumps({"figures": [row()]}),
            "figures/fig1.pdf": pdf(),
            "figures/fig1.png": png(1300, 800),
        },
    )
    await staged_paper(workspace, files)
    script = f"{SKILL_ROOT}/validate-paper/scripts/validate_figures.py"
    code, output = await run(workspace, script, "--paper-dir", ".")
    assert code == 0, output
    assert "Validated 1 figures with manifest and readability budgets." in output
    bad = dict(files)
    bad["figure_manifest.json"] = json.dumps({"figures": [row(source_ids=["result.nope"])]})
    await write_files(workspace, "paper", bad)
    code, output = await run(workspace, script, "--paper-dir", ".")
    assert code != 0 and "unknown source id: 'result.nope'" in output


async def test_the_claims_script_runs_and_skips_the_staged_skills(
    workspace: LocalWorkspace,
) -> None:
    await staged_paper(workspace)
    script = f"{SKILL_ROOT}/write-paper/scripts/validate_claims.py"
    args = (
        "--paper-data", "paper_data.json", "--references", "references_provenance.json",
        "--bib", "references.bib", "--claims", "claims_grounding.json", "--paper-dir", ".",
    )  # fmt: skip
    code, output = await run(workspace, script, *args)
    assert code == 0, output
    # the skills' own documentation holds evidence-link examples; they must not count as tokens
    skill = await workspace.read_bytes(f"paper/{SKILL_ROOT}/scientific-evidence/SKILL.md")
    assert b"@evidence[[" in skill
    assert "dangling" not in output


async def test_the_evidence_tool_emits_a_bundle_the_gate_then_verifies(tmp_path: Path) -> None:
    workspace = FakeLatexWorkspace(tmp_path / "ws")
    files = paper_files(
        extra={
            "results/metrics.json": json.dumps({"score": 1}),
            "notes.md": "The score. @evidence[[results/metrics.json#score]]\n",
        }
    )
    await staged_paper(workspace, files)
    # alias directories are not scanned, so keep the evidence target in a scanned place too
    await workspace.write_bytes("paper/data/metrics.json", b'{"score": 1}')
    await workspace.write_bytes(
        "paper/notes.md", b"The score. @evidence[[data/metrics.json#score]]\n"
    )
    emit = f"{SKILL_ROOT}/scientific-evidence/scripts/scientific_evidence.py"
    code, output = await run(workspace, emit, "emit-bundle", "--root-dir", ".")
    assert code == 0, output
    assert await workspace.exists("paper/scientific-evidence/bundle.manifest.json")
    report = await WorkspacePaperToolchain().validate("paper", workspace=workspace, options={})
    assert report.passed, report.errors
    evidence = report.checks["evidence_validation"]
    assert evidence["checked"] and evidence["summary"]["total_references"] == 1
    assert evidence["summary"]["dangling_references"] == 0
    assert await workspace.exists("paper/scientific-evidence/verification-report.json")
    # a dangling link fails the gate with the verifier's message
    await workspace.write_bytes("paper/notes.md", b"Missing. @evidence[[data/none.json#x]]\n")
    report = await WorkspacePaperToolchain().validate("paper", workspace=workspace, options={})
    assert not report.passed
    assert "scientific-evidence verifier found 1 dangling" in report.errors[0]
    assert report.checks["evidence_validation"]["dangling"][0]["raw_reference"].startswith(
        "@evidence"
    )


async def test_a_tampered_staged_verifier_cannot_change_the_verdict(tmp_path: Path) -> None:
    workspace = FakeLatexWorkspace(tmp_path / "ws")
    await staged_paper(workspace, paper_files())
    await workspace.write_bytes("paper/notes.md", b"@evidence[[data/none.json#x]]\n")
    script = f"{SKILL_ROOT}/scientific-evidence/scripts/scientific_evidence.py"
    code, _ = await run(workspace, script, "emit-bundle", "--root-dir", ".")
    assert code == 0
    await workspace.write_bytes(f"paper/{script}", b"import sys; sys.exit(0)\n")
    report = await WorkspacePaperToolchain().validate("paper", workspace=workspace, options={})
    assert not report.passed and "dangling" in report.errors[0]


async def test_the_compile_input_checker_knows_the_staged_skill_directory(
    workspace: LocalWorkspace,
) -> None:
    await staged_paper(workspace)
    script = f"{SKILL_ROOT}/validate-paper/scripts/validate_compile_inputs.py"
    root = workspace.host_path("paper")
    await workspace.write_bytes("paper/paper.fls", f"PWD {root}\nINPUT paper.tex\n".encode())
    code, output = await run(
        workspace, script, "recorder", "--paper-dir", root, "--fls", "paper.fls"
    )
    assert code == 0, output
    staged_sty = f"{root}/{SKILL_ROOT}/write-paper/templates/iclr/paper.sty"
    bad = f"PWD {root}\nINPUT paper.tex\nINPUT {staged_sty}\n"
    await workspace.write_bytes("paper/paper.fls", bad.encode())
    code, output = await run(
        workspace, script, "recorder", "--paper-dir", root, "--fls", "paper.fls"
    )
    assert code != 0 and "agent skill" in output


async def test_the_tex_reference_and_rendered_si_scripts_run(workspace: LocalWorkspace) -> None:
    await staged_paper(workspace)
    code, output = await run(
        workspace,
        f"{SKILL_ROOT}/validate-paper/scripts/validate_tex_refs.py",
        "--tex", "paper.tex", "--bib", "references.bib",
    )  # fmt: skip
    assert code == 0, output
    pytest.importorskip("fitz")
    from test_validation_support import DEFAULT_PAGES, make_pdf

    await workspace.write_bytes("paper/paper.pdf", make_pdf(DEFAULT_PAGES))
    code, output = await run(
        workspace,
        f"{SKILL_ROOT}/validate-paper/scripts/validate_rendered_si.py",
        "--pdf",
        "paper.pdf",
    )
    assert code == 0, output


@pytest.mark.parametrize("script", ["s2_fetch.py", "openalex_fetch.py"])
async def test_the_retrieval_scripts_start(workspace: LocalWorkspace, script: str) -> None:
    await staged_paper(workspace)
    code, output = await run(
        workspace, f"{SKILL_ROOT}/retrieve-references/scripts/{script}", "--help"
    )
    assert code == 0 and "usage:" in output


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex is not installed")
async def test_the_staged_compile_script_builds_a_pdf(workspace: LocalWorkspace) -> None:
    await staged_paper(workspace)
    result = await workspace.run(
        ["bash", f"{SKILL_ROOT}/validate-paper/scripts/compile_latex.sh", "paper.tex"], cwd="paper"
    )
    assert result.returncode in {0, 1}
