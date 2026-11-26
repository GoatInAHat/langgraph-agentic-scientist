"""The paper validation gate and LaTeX compile (upstream ``validate_and_compile_paper``).

:class:`WorkspacePaperToolchain` is the default
:class:`~langgraph_agentic_scientist.core.interfaces.PaperToolchain`. Its gates run in upstream's
order and the first failing gate ends the run with its message, as upstream's
``validate_and_compile_paper`` raises at the first failure:

1. required artifacts exist (``paper.tex``, ``references.bib``);
2. the managed watermark is (re)inserted into ``paper.tex`` before the manuscript is read, or,
   when it is switched off, a managed block already there is removed;
3. required sections; Supplementary Information (source); BibTeX dependencies; BibTeX keys and
   citations;
4. ``references_provenance.json`` against ``references.bib``; ``claims_grounding.json`` against
   the grounding sources of ``paper_data.json`` and the references;
5. the scientific-evidence bundle (every ``@evidence[[...]]`` link must resolve);
6. visible language (standalone papers); venue alignment (author block and framing);
7. the figure manifest and the readability budgets;
8. the LaTeX compile (``pdflatex``, ``bibtex``, ``pdflatex``, ``pdflatex``) with its inputs
   validated; the rendered Supplementary Information; the page budget.

``validation_report.json`` is written to the paper directory, with upstream's keys, whether the
gate passes or fails. The returned
:class:`~langgraph_agentic_scientist.core.results.ValidationReport` carries the same report in
``checks`` and, for a failure, a fingerprint of the errors and the paper inputs, so that a repair
that changes nothing can be recognised.

Files are read through the workspace (a snapshot of the paper directory is taken once, after the
watermark is applied); commands run through :meth:`~langgraph_agentic_scientist.Workspace.run`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from ...core.interfaces import CompileResult
from ...core.results import ValidationReport
from ...prompts import PromptRegistry
from ...venues import VenueCatalog
from ...workspace.paths import join
from ...workspace.protocol import FileEntry, Workspace
from ..watermark import (
    WATERMARK_TEXT,
    PaperWatermarkError,
    apply_paper_watermark,
    has_managed_watermark,
    strip_managed_block,
)
from .evidence import (
    EVIDENCE_BUNDLE_DIRNAME,
    EVIDENCE_SCAN_SUFFIXES,
    WORKSPACE_ALIAS_DIRS,
    _is_excluded_evidence_scan_path,
    validate_evidence_bundle,
)
from .figures import validate_figure_manifest
from .files import PaperTree, SnapshotTree, read_text
from .fingerprint import VALIDATION_INPUT_PATHS, validation_failure_fingerprint
from .graphics import MAX_GRAPHICS_TEX_CHARS
from .grounding import validate_claims_grounding, validate_reference_provenance
from .language import validate_visible_language
from .latex import (
    PaperValidationInfrastructureError,
    compile_latex,
    validate_bibtex_dependencies,
)
from .page_budget import read_pdf_page_texts, validate_page_budget
from .supplement import (
    validate_rendered_supplementary_information,
    validate_supplementary_information,
)
from .tex import (
    extract_bibtex_keys,
    extract_tex_citations,
    missing_required_sections,
    read_json,
)
from .venues import detect_paper_format, validate_venue_alignment

TEX_FILE_SUFFIXES = (".tex", ".sty", ".cls", ".bst", ".bib", ".aux", ".fls")
"""Files read wherever they are in the paper directory (TeX sources, styles, bibliography)."""

_MAX_FILE_BYTES = 64 * 1024 * 1024
_HEADER_BYTES = 64


def _include_for_validation(rel: str, entry: FileEntry) -> int | None:
    """Return how many bytes of a paper-directory file the validation tree keeps."""
    parts = rel.split("/")
    if parts[0] == ".agents":
        return None
    if len(parts) > 1 and parts[0] in WORKSPACE_ALIAS_DIRS:
        return None
    suffix = PurePosixPath(rel).suffix.lower()
    if rel in VALIDATION_INPUT_PATHS:
        return _MAX_FILE_BYTES
    if suffix in TEX_FILE_SUFFIXES and not _is_excluded_evidence_scan_path(rel):
        return _MAX_FILE_BYTES
    if parts[0] == "figures" and len(parts) > 1:
        if suffix in {".png", ".pdf"}:
            return _HEADER_BYTES
        if suffix == ".json":
            return 16 * 1024 * 1024
        return None
    if suffix in EVIDENCE_SCAN_SUFFIXES and not _is_excluded_evidence_scan_path(rel):
        return 16 * 1024 * 1024
    return None


def _include_for_compile(rel: str, entry: FileEntry) -> int | None:
    parts = rel.split("/")
    if parts[0] == ".agents" or (len(parts) > 1 and parts[0] in WORKSPACE_ALIAS_DIRS):
        return None
    if PurePosixPath(rel).suffix.lower() in TEX_FILE_SUFFIXES:
        return _MAX_FILE_BYTES
    return None


async def load_validation_tree(workspace: Workspace, paper_dir: str) -> SnapshotTree:
    """Load the files the validation gates read from a paper directory of a workspace."""
    return await SnapshotTree.load(workspace, paper_dir, include=_include_for_validation)


async def _load_compile_tree(workspace: Workspace, paper_dir: str) -> SnapshotTree:
    return await SnapshotTree.load(workspace, paper_dir, include=_include_for_compile)


def new_validation_report(workspace: Workspace, paper_dir: str) -> dict[str, Any]:
    """Return upstream's empty ``validation_report.json`` for a paper directory."""

    def host(rel: str) -> str:
        return workspace.host_path(join(paper_dir, rel))

    return {
        "ok": False,
        "paper_dir": workspace.host_path(paper_dir),
        "paper_profile": "default",
        "missing_sections": [],
        "citation_count": 0,
        "bibtex_key_count": 0,
        "claim_count": 0,
        "feedback": {
            "figure_validation_errors": [],
            "figure_validation_warnings": [],
            "language_validation_warnings": [],
        },
        "figure_validation": {
            "checked": False,
            "required": False,
            "figure_count": 0,
            "referenced_paths": [],
            "manifest_path": host("figure_manifest.json"),
            "errors": [],
            "hard_errors": [],
            "soft_warnings": [],
        },
        "reference_validation": {
            "checked": False,
            "provenance_path": host("references_provenance.json"),
            "provenance_reference_count": 0,
            "errors": [],
        },
        "evidence_validation": {
            "checked": False,
            "bundle_path": host(EVIDENCE_BUNDLE_DIRNAME),
            "bundle_present": False,
            "verification_report_path": None,
            "summary": {
                "total_references": 0,
                "resolved_references": 0,
                "anchored_references": 0,
                "dangling_references": 0,
                "manifests_checked": 0,
            },
            "dangling": [],
            "paper_token_count": 0,
            "hard_errors": [],
        },
        "standalone_language_validation": {
            "checked": False,
            "paper_profile": "default",
            "hard_errors": [],
            "soft_warnings": [],
            "forbidden_terms": [],
        },
        "venue_validation": {
            "checked": False,
            "paper_format": None,
            "hard_errors": [],
            "soft_warnings": [],
        },
        "supplementary_information_validation": {
            "checked": False,
            "appendix_present": False,
            "heading_present": False,
            "placed_after_references": False,
            "pre_bibliography_clearpage_present": False,
            "clearpage_present": False,
            "visible_character_count": 0,
            "substantive_element_count": 0,
            "rendered_checked": False,
            "rendered_heading_present": False,
            "rendered_visible_character_count": 0,
            "hard_errors": [],
        },
        "page_validation": {
            "checked": False,
            "paper_format": None,
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
        },
        "watermark_validation": {
            "checked": False,
            "text": WATERMARK_TEXT,
            "applied": False,
            "changed": False,
            "removed": False,
            "hard_errors": [],
        },
        "latex_logs": [],
        "pdf_path": None,
        "error": None,
    }


def _tex_text(tree: PaperTree, rel: str) -> str:
    data = tree.read(rel)
    if data is None:
        raise ValueError(f"Unable to open regular TeX file: {rel}")
    if len(data) > MAX_GRAPHICS_TEX_CHARS:
        raise ValueError("Paper TeX exceeds the graphics resolution limit")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"TeX input is not valid UTF-8: {rel}") from exc


def _list_payload(report: Mapping[str, Any], key: str, what: str, kind: str) -> list[Any]:
    value = report.get(key, [])
    if not isinstance(value, list):
        raise RuntimeError(f"{what} returned invalid {kind} payload")
    return value


class WorkspacePaperToolchain:
    """Validates and compiles a paper directory of a workspace.

    Args:
        venues: The venue catalog (default: the package's ``iclr`` and ``alife``).
        prompts: Where the vendored evidence verifier comes from (default: the package's).
        python: The interpreter that runs the evidence verifier through the workspace (default:
            the current interpreter, which suits :class:`LocalWorkspace`; pass ``"python3"`` for
            a workspace that runs elsewhere).
        latex_env: The whole environment of the compile commands (default: upstream's clean
            allowlisted environment built from this process, see
            :func:`~langgraph_agentic_scientist.paper.validation.latex.default_latex_environment`;
            pass your own for a workspace that runs elsewhere).
        pdflatex: The ``pdflatex`` command.
        bibtex: The ``bibtex`` command.
        watermark: Whether the managed watermark block is inserted before every compile
            (default: yes, as upstream). A caller that publishes into a paper of its own, with
            its own disclosure, switches it off; ``options["watermark"]`` of a call overrides.
        compile_timeout: Wall clock of each compile command, in seconds.
        evidence_timeout: Wall clock of the evidence verifier, in seconds.
    """

    def __init__(
        self,
        *,
        venues: VenueCatalog | None = None,
        prompts: PromptRegistry | None = None,
        python: str | None = None,
        latex_env: Mapping[str, str] | None = None,
        pdflatex: str = "pdflatex",
        bibtex: str = "bibtex",
        watermark: bool = True,
        compile_timeout: float | None = 1800,
        evidence_timeout: float | None = 600,
    ) -> None:
        self.venues = venues or VenueCatalog.default()
        self.prompts = prompts
        self.python = python or sys.executable
        self.latex_env = dict(latex_env) if latex_env is not None else None
        self.pdflatex = pdflatex
        self.bibtex = bibtex
        self.watermark = watermark
        self.compile_timeout = compile_timeout
        self.evidence_timeout = evidence_timeout

    # --- PaperToolchain ---------------------------------------------------------------

    async def validate(
        self, paper_dir: str, *, workspace: Workspace, options: dict[str, Any]
    ) -> ValidationReport:
        """Run the validation gates of ``validate_and_compile_paper`` on a paper directory.

        Writes ``validation_report.json`` into ``paper_dir`` and returns the report. A failing
        gate is a report with ``passed=False`` and the gate's message in ``errors``; a missing
        compiler is also a failure (``checks["infrastructure_error"]`` is then true).

        Args:
            paper_dir: Workspace-relative path of the paper directory.
            workspace: The workspace holding it.
            options: ``{"watermark": bool}`` overrides the toolchain's watermark switch; other
                keys are ignored.
        """
        watermark = bool(options.get("watermark", self.watermark)) if options else self.watermark
        report = new_validation_report(workspace, paper_dir)
        infrastructure = False
        failure: Exception | None = None
        try:
            await self._run_gates(workspace, paper_dir, report, watermark=watermark)
        except PaperValidationInfrastructureError as exc:
            report["error"] = str(exc)
            infrastructure = True
            failure = exc
        except Exception as exc:
            report["error"] = str(exc)
            failure = exc
        await workspace.write_bytes(
            join(paper_dir, "validation_report.json"),
            json.dumps(report, indent=2, sort_keys=True).encode("utf-8"),
        )
        return await self._model(
            workspace, paper_dir, report, failed=failure is not None, infrastructure=infrastructure
        )

    async def compile(self, paper_dir: str, *, workspace: Workspace) -> CompileResult:
        """Insert the watermark (when enabled) and compile ``paper.tex`` into ``paper.pdf``."""
        pdf_rel = join(paper_dir, "paper.pdf")
        try:
            if self.watermark:
                await apply_paper_watermark(workspace, paper_dir)
            logs = await compile_latex(
                workspace,
                paper_dir,
                load_tree=lambda: _load_compile_tree(workspace, paper_dir),
                env=self.latex_env,
                pdflatex=self.pdflatex,
                bibtex=self.bibtex,
                timeout=self.compile_timeout,
            )
        except (PaperWatermarkError, PaperValidationInfrastructureError, RuntimeError) as exc:
            return CompileResult(success=False, log=str(exc))
        if not await workspace.exists(pdf_rel):
            return CompileResult(success=False, log=f"Expected compiled PDF missing: {pdf_rel}")
        page_count: int | None = None
        try:
            page_count = len(read_pdf_page_texts(await workspace.read_bytes(pdf_rel)))
        except Exception:
            page_count = None
        log = "\n".join(
            f"$ {' '.join(entry['command'])}\n{entry['stdout_tail']}{entry['stderr_tail']}"
            for entry in logs
        )
        return CompileResult(success=True, pdf_path=pdf_rel, log=log, page_count=page_count)

    async def pdf_text(self, pdf_path: str, *, workspace: Workspace) -> str:
        """Extract the text of a PDF (pages joined by blank lines)."""
        return "\n\n".join(read_pdf_page_texts(await workspace.read_bytes(pdf_path)))

    # --- the gates ---------------------------------------------------------------------

    async def _run_gates(
        self, workspace: Workspace, paper_dir: str, report: dict[str, Any], *, watermark: bool
    ) -> None:
        tex_rel = "paper.tex"
        bib_rel = "references.bib"
        if await workspace.stat(join(paper_dir, tex_rel)) is None:
            raise RuntimeError(f"Missing required artifact: {tex_rel}")
        if await workspace.stat(join(paper_dir, bib_rel)) is None:
            raise RuntimeError(f"Missing required artifact: {bib_rel}")

        # Host-owned watermark: (re)inserted before the manuscript is read, hashed or compiled so
        # that every published PDF carries it.
        watermark_report = report["watermark_validation"]
        if watermark:
            watermark_report["checked"] = True
            try:
                watermark_report["changed"] = await apply_paper_watermark(workspace, paper_dir)
            except PaperWatermarkError as exc:
                watermark_report["hard_errors"].append(str(exc))
                raise RuntimeError(f"paper watermark failed: {exc}") from exc

        tree = await load_validation_tree(workspace, paper_dir)
        tex_text = _tex_text(tree, tex_rel)
        if watermark:
            if not has_managed_watermark(tex_text):
                watermark_report["hard_errors"].append(
                    "managed watermark block missing from paper.tex after insertion"
                )
                raise RuntimeError("paper watermark failed: managed block missing after insertion")
            watermark_report["applied"] = True
        elif has_managed_watermark(tex_text):
            tex_text = strip_managed_block(tex_text)
            await workspace.write_bytes(join(paper_dir, tex_rel), tex_text.encode("utf-8"))
            tree = await load_validation_tree(workspace, paper_dir)
            watermark_report["removed"] = True
        bib_text = read_text(tree, bib_rel) or ""
        venue_brief_text = read_text(tree, "VENUE.md") or ""
        missing_sections = missing_required_sections(tex_text)
        report["missing_sections"] = missing_sections
        if missing_sections:
            raise RuntimeError(f"paper.tex missing required sections: {missing_sections}")

        supplementary_report = validate_supplementary_information(tex_text)
        report["supplementary_information_validation"] = supplementary_report
        supplementary_errors = supplementary_report.get("hard_errors", [])
        if not isinstance(supplementary_errors, list):
            raise RuntimeError(
                "Supplementary Information validation returned invalid hard_errors payload"
            )
        if supplementary_errors:
            raise RuntimeError(
                "Supplementary Information validation failed: "
                + "; ".join(str(error) for error in supplementary_errors)
            )

        validate_bibtex_dependencies(tree, tex_text)

        bib_keys = extract_bibtex_keys(bib_text)
        report["bibtex_key_count"] = len(bib_keys)
        if not bib_keys:
            raise RuntimeError("references.bib did not contain any BibTeX keys")
        cited_keys = extract_tex_citations(tex_text)
        report["citation_count"] = len(cited_keys)
        unknown_citations = sorted(key for key in cited_keys if key not in bib_keys)
        if unknown_citations:
            raise RuntimeError(f"paper.tex cites unknown BibTeX keys: {unknown_citations}")

        claims_payload = read_json(tree, "claims_grounding.json")
        report["claim_count"] = len(claims_payload.get("claims", []))
        paper_data = read_json(tree, "paper_data.json")
        references_provenance = read_json(tree, "references_provenance.json")
        paper_profile = str(paper_data.get("paper_profile", "default"))
        report["paper_profile"] = paper_profile
        grounding_sources = paper_data.get("grounding_sources")
        if not isinstance(grounding_sources, dict):
            raise RuntimeError("paper_data.json missing object field: grounding_sources")
        allowed_source_ids = {str(key) for key in grounding_sources}

        provenance_ids, reference_errors = validate_reference_provenance(
            references_payload=references_provenance,
            bib_keys=bib_keys,
        )
        report["reference_validation"] = {
            "checked": True,
            "provenance_path": workspace.host_path(join(paper_dir, "references_provenance.json")),
            "provenance_reference_count": len(provenance_ids),
            "errors": [str(error) for error in reference_errors],
        }
        if reference_errors:
            raise RuntimeError(
                "references provenance validation failed: " + "; ".join(reference_errors)
            )
        allowed_source_ids.update(provenance_ids)

        claim_errors = validate_claims_grounding(
            claims_payload=claims_payload,
            allowed_source_ids=allowed_source_ids,
            valid_citation_keys=bib_keys,
        )
        if claim_errors:
            raise RuntimeError(
                "claims_grounding.json validation failed: " + "; ".join(claim_errors)
            )

        evidence_report = await validate_evidence_bundle(
            workspace,
            paper_dir,
            tree,
            python=self.python,
            prompts=self.prompts,
            timeout=self.evidence_timeout,
        )
        report["evidence_validation"] = evidence_report
        evidence_errors = _list_payload(
            evidence_report, "hard_errors", "evidence validation", "hard_errors"
        )
        if evidence_errors:
            raise RuntimeError(
                "scientific-evidence validation failed: "
                + "; ".join(str(error) for error in evidence_errors)
            )

        language_report = validate_visible_language(
            tex_text,
            paper_profile=paper_profile,
            presentation_contract=paper_data.get("presentation_contract", {}),
        )
        report["standalone_language_validation"] = language_report
        report["feedback"]["language_validation_warnings"] = [
            str(message) for message in language_report.get("soft_warnings", [])
        ]
        language_errors = _list_payload(
            language_report, "hard_errors", "standalone language validation", "hard_errors"
        )
        if language_errors:
            raise RuntimeError(
                "standalone paper language validation failed: "
                + "; ".join(str(error) for error in language_errors)
            )

        paper_format = detect_paper_format(tex_text, self.venues)
        venue_report = validate_venue_alignment(
            tex_text,
            venue_brief_text=venue_brief_text,
            paper_format=paper_format,
            paper_dir=tree,
            catalog=self.venues,
        )
        report["venue_validation"] = venue_report
        venue_warnings = _list_payload(
            venue_report, "soft_warnings", "venue validation", "soft_warnings"
        )
        report["feedback"]["language_validation_warnings"].extend(
            str(message) for message in venue_warnings
        )
        venue_errors = _list_payload(venue_report, "hard_errors", "venue validation", "hard_errors")
        if venue_errors:
            raise RuntimeError(
                "venue validation failed: " + "; ".join(str(error) for error in venue_errors)
            )

        figure_report = validate_figure_manifest(
            tree,
            tex_text=tex_text,
            allowed_source_ids=allowed_source_ids,
            manifest_path_label=workspace.host_path(join(paper_dir, "figure_manifest.json")),
        )
        report["figure_validation"] = figure_report
        figure_errors = _list_payload(
            figure_report, "errors", "figure_manifest validation", "errors"
        )
        report["feedback"]["figure_validation_errors"] = [str(error) for error in figure_errors]
        figure_hard_errors = _list_payload(
            figure_report, "hard_errors", "figure_manifest validation", "hard_errors"
        )
        figure_soft_warnings = _list_payload(
            figure_report, "soft_warnings", "figure_manifest validation", "soft_warnings"
        )
        report["feedback"]["figure_validation_warnings"] = [
            str(message) for message in figure_soft_warnings
        ]
        if figure_hard_errors:
            raise RuntimeError(
                "figure validation failed: " + "; ".join(str(error) for error in figure_hard_errors)
            )

        latex_logs = await compile_latex(
            workspace,
            paper_dir,
            load_tree=lambda: _load_compile_tree(workspace, paper_dir),
            env=self.latex_env,
            pdflatex=self.pdflatex,
            bibtex=self.bibtex,
            timeout=self.compile_timeout,
        )
        report["latex_logs"] = latex_logs
        pdf_rel = join(paper_dir, "paper.pdf")
        if not await workspace.exists(pdf_rel):
            raise RuntimeError(f"Expected compiled PDF missing: {workspace.host_path(pdf_rel)}")
        pdf = await workspace.read_bytes(pdf_rel)

        compiled_page_texts: list[str] | None = None
        try:
            compiled_page_texts = read_pdf_page_texts(pdf)
        except Exception as exc:
            if pdf[:4] == b"%PDF":
                raise RuntimeError(
                    "could not inspect compiled paper.pdf for rendered "
                    f"Supplementary Information: {exc}"
                ) from exc
        if compiled_page_texts is not None:
            rendered_supplement_report = validate_rendered_supplementary_information(
                compiled_page_texts
            )
            rendered_supplement_errors = rendered_supplement_report.pop("hard_errors")
            supplementary_report.update(rendered_supplement_report)
            supplementary_report["hard_errors"].extend(rendered_supplement_errors)
            if rendered_supplement_errors:
                raise RuntimeError(
                    "rendered Supplementary Information validation failed: "
                    + "; ".join(str(error) for error in rendered_supplement_errors)
                )

        page_report = validate_page_budget(
            tex_text=tex_text, pdf=pdf, page_texts=compiled_page_texts, catalog=self.venues
        )
        report["page_validation"] = page_report
        page_errors = _list_payload(page_report, "errors", "page validation", "errors")
        if page_errors:
            raise RuntimeError(
                "page-limit validation failed: " + "; ".join(str(error) for error in page_errors)
            )

        report["ok"] = True
        report["pdf_path"] = workspace.host_path(pdf_rel)

    async def _model(
        self,
        workspace: Workspace,
        paper_dir: str,
        report: dict[str, Any],
        *,
        failed: bool,
        infrastructure: bool,
    ) -> ValidationReport:
        errors = [str(report["error"])] if failed and report.get("error") else []
        fingerprint: str | None = None
        if failed and errors:
            try:
                fingerprint = validation_failure_fingerprint(
                    errors, await load_validation_tree(workspace, paper_dir)
                )
            except (RuntimeError, ValueError, OSError):
                fingerprint = None
        checks: dict[str, Any] = dict(report)
        if infrastructure:
            checks["infrastructure_error"] = True
        page_validation = report.get("page_validation") or {}
        feedback = report.get("feedback") or {}
        return ValidationReport(
            passed=bool(report.get("ok")),
            errors=errors,
            warnings=[str(message) for message in feedback.get("language_validation_warnings", [])],
            figure_warnings=[
                str(message) for message in feedback.get("figure_validation_warnings", [])
            ],
            checks=checks,
            page_count=page_validation.get("total_pages"),
            pdf_path=report.get("pdf_path"),
            fingerprint=fingerprint,
        )


__all__ = [
    "TEX_FILE_SUFFIXES",
    "WorkspacePaperToolchain",
    "load_validation_tree",
    "new_validation_report",
]
