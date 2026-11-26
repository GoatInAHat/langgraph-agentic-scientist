"""Shared factories of the paper validation tests (derived from upstream's test support).

The valid manuscript below is the one upstream's tests use (Apache License 2.0, Sakana AI). A
fake workspace stands in for the LaTeX tools: ``pdflatex`` writes a PDF built from scripted page
texts, the recorder file and the auxiliary file; everything else runs for real.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from langgraph_agentic_scientist import LocalWorkspace
from langgraph_agentic_scientist.workspace import CommandResult

SI_BODY = (
    "We report additional grounded evaluation details, implementation settings, and "
    "failure-analysis context that support reproducibility without carrying the central claim.\n"
)
ALIFE_AUTHORS = (
    "Ada Testwright$^{1}$ \\and Ben Mockley$^{1}$ \\\\\n\\mbox{}\\\\\n$^{1}$Fictional Institute"
)
BODY_PAGE = "Introduction\nThe body of the paper is here."
REFS_PAGE = "References\n[1] Example"
SUPPLEMENT_PAGE = (
    "A Supplementary Information\nExtended grounded evaluation and implementation details "
    "support reproducibility, including additional controls, robustness checks, limitations, "
    "and failure analysis."
)
DEFAULT_PAGES = [BODY_PAGE, REFS_PAGE, SUPPLEMENT_PAGE]


def valid_tex(*, include_graphics: str | None = None, preamble: str = "") -> str:
    """A manuscript that passes every source gate."""
    graphics_package = "\\usepackage{graphicx}\n" if include_graphics else ""
    graphics_block = (
        f"\\begin{{figure}}\n\\includegraphics{{{include_graphics}}}\n"
        "\\caption{Example figure.}\n\\end{figure}\n"
        if include_graphics
        else ""
    )
    return (
        "\\documentclass{article}\n"
        f"{graphics_package}{preamble}"
        "\\begin{document}\n"
        "\\begin{abstract}\nSummary.\n\\end{abstract}\n"
        "\\section{Introduction}\nIntro \\cite{RefKey2024}.\n"
        "\\section{Related Work}\nPrior art.\n"
        "\\section{Method}\nMethod.\n"
        "\\section{Results}\n"
        f"{graphics_block}"
        "Results.\n"
        "\\section{Conclusion}\nDone.\n"
        "\\clearpage\n"
        "\\bibliographystyle{plain}\n"
        "\\bibliography{references}\n"
        "\\clearpage\n\\appendix\n"
        "\\section{Supplementary Information}\n"
        "\\subsection{Extended evaluation details}\n"
        f"{SI_BODY}"
        "\\end{document}\n"
    )


def valid_alife_tex(author: str = ALIFE_AUTHORS) -> str:
    """A manuscript with ALife markers and framing language."""
    return (
        "\\documentclass{article}\n"
        "\\usepackage{natbib,alifeconf}\n"
        "\\title{Artificial Life Example of Adaptive Computation}\n"
        f"\\author{{{author}}}\n"
        "\\begin{document}\n\\maketitle\n"
        "\\begin{abstract}\n"
        "Artificial life summary about adaptive computation and biologically plausible learning.\n"
        "\\end{abstract}\n"
        "\\section{Introduction}\n"
        "Artificial life intro on life-like adaptive systems and local learning "
        "\\cite{RefKey2024}.\n"
        "\\section{Related Work}\nPrior art.\n"
        "\\section{Method}\nMethod.\n"
        "\\section{Results}\nResults.\n"
        "\\section{Conclusion}\n"
        "Artificial life conclusion on adaptive computation under biologically plausible "
        "constraints.\n"
        "\\clearpage\n"
        "\\bibliographystyle{apalike}\n"
        "\\bibliography{references}\n"
        "\\clearpage\n\\appendix\n"
        "\\section{Supplementary Information}\n"
        "\\subsection{Extended evaluation details}\n"
        f"{SI_BODY}"
        "\\end{document}\n"
    )


BIB = "@article{RefKey2024,\n  title={An Example},\n  author={Someone, A.},\n  year={2024}\n}\n"


def make_pdf(pages: Sequence[str]) -> bytes:
    """Build a PDF with one text page per entry (needs PyMuPDF, an extra of the package)."""
    fitz = pytest.importorskip("fitz")
    document = fitz.open()
    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text, fontsize=11)
    data: bytes = document.tobytes()
    document.close()
    return data


def paper_files(
    *,
    tex: str | None = None,
    profile: str = "default",
    contract: Mapping[str, Any] | None = None,
    sources: Mapping[str, Any] | None = None,
    claims: list[dict[str, Any]] | None = None,
    extra: Mapping[str, str | bytes] | None = None,
) -> dict[str, str | bytes]:
    """The files of a paper directory that passes the source gates."""
    grounding = dict(
        sources or {"result.best_correct_score": {"id": "x", "type": "metric", "value": 1}}
    )
    files: dict[str, str | bytes] = {
        "paper.tex": tex if tex is not None else valid_tex(),
        "references.bib": BIB,
        "references_provenance.json": json.dumps(
            {"references": [{"id": "ref.s2.abc", "citation_key": "RefKey2024"}]}
        ),
        "claims_grounding.json": json.dumps(
            {
                "claims": claims
                or [
                    {
                        "claim": "The best score is one.",
                        "source_ids": ["result.best_correct_score"],
                        "citation_keys": ["RefKey2024"],
                    }
                ]
            }
        ),
        "paper_data.json": json.dumps(
            {
                "paper_profile": profile,
                "grounding_sources": grounding,
                "presentation_contract": dict(contract or {}),
            }
        ),
    }
    files.update(extra or {})
    return files


class FakeLatexWorkspace(LocalWorkspace):
    """A local workspace whose ``pdflatex`` and ``bibtex`` are scripted.

    ``pdflatex`` writes ``paper.pdf`` (from ``pages``), the recorder file ``paper.fls`` (with the
    inputs in ``recorded_inputs``) and ``paper.aux``; ``bibtex`` succeeds. Commands are recorded.
    """

    def __init__(
        self,
        root: Path,
        *,
        pages: Sequence[str] = DEFAULT_PAGES,
        recorded_inputs: Sequence[str] = ("paper.tex",),
        aux: str = "\\bibstyle{plain}\n\\bibdata{references}\n",
        pdflatex_returncode: int = 0,
    ) -> None:
        super().__init__(root)
        self.pages = list(pages)
        self.recorded_inputs = list(recorded_inputs)
        self.aux = aux
        self.pdflatex_returncode = pdflatex_returncode
        self.commands: list[list[str]] = []

    async def run(self, argv: Sequence[str], **kwargs: Any) -> CommandResult:  # type: ignore[override]
        """Script the LaTeX tools; run anything else."""
        command = list(argv)
        if command[0] not in {"pdflatex", "bibtex"}:
            return await super().run(argv, **kwargs)
        self.commands.append(command)
        cwd = kwargs["cwd"]
        prefix = f"{cwd}/" if cwd else ""
        if command[0] == "pdflatex":
            await self.write_bytes(f"{prefix}paper.pdf", make_pdf(self.pages))
            lines = [f"PWD {self.host_path(cwd)}"]
            lines += [f"INPUT {name}" for name in self.recorded_inputs]
            await self.write_bytes(f"{prefix}paper.fls", ("\n".join(lines) + "\n").encode())
            await self.write_bytes(f"{prefix}paper.aux", self.aux.encode())
            return CommandResult(
                argv=command, returncode=self.pdflatex_returncode, stdout="This is pdfTeX"
            )
        return CommandResult(argv=command, returncode=0, stdout="This is BibTeX")


async def write_files(
    workspace: LocalWorkspace, directory: str, files: Mapping[str, str | bytes]
) -> None:
    """Write files into a directory of a workspace."""
    for name, content in files.items():
        data = content.encode("utf-8") if isinstance(content, str) else content
        await workspace.write_bytes(f"{directory}/{name}", data)


def test_support_manuscript_is_well_formed() -> None:
    """The support manuscript names the sections the gates require."""
    assert "\\section{Supplementary Information}" in valid_tex()
